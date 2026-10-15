import yaml
import subprocess
import platform
import os
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

def latex_escape(value):
    """Escape plain text once, including all LaTeX special characters."""
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


@dataclass
class BreakpointData:
    label: str
    dataset: xr.Dataset
    metadata: dict

class Report:
    def __init__(self, manifest: str):
        # 1. Get the absolute path to the manifest file
        self.manifest_path = Path(manifest).resolve()
        
        with open(self.manifest_path, "r") as f:
            self.manifest = yaml.safe_load(f)
            
        # 2. Set the root directory to wherever the manifest is located
        self.output_root = self.manifest_path.parent
        
        self._show_plots = False
        
        # LaTeX document state
        self._latex_content = []
        self._figures_dir = self.output_root / "report_figures"
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._fig_counter = 0
        self.author_name = ""

    def author(self, author: str):
        self.author_name = author

    def hardware_target(self, target: str):
        safe_target = latex_escape(target)
        self._latex_content.append(f"\\textbf{{Hardware Target:}} {safe_target}\\\\[0.5em]")

    def system_info(self):
        self.add_property("System", platform.platform())
        self.add_property("CPU", platform.processor() or platform.machine())
        self.add_property("Logical CPUs", os.cpu_count())

    def execution_time_sec(self, time_sec: float):
        self._latex_content.append(f"\\textbf{{Total Execution Time:}} {time_sec} s\\\\[1em]")

    def add_property(self, name: str, value):
        """Adds a generic bolded key-value pair to the LaTeX report."""
        # Convert to string and escape underscores for both name and value
        safe_name = latex_escape(name)
        safe_value = latex_escape(value)
        
        self._latex_content.append(f"\\textbf{{{safe_name}:}} {safe_value}\\\\[0.5em]")

    def show_plots(self, show: bool):
        self._show_plots = show

    def print(self, text: str):
        # Escape basic LaTeX characters
        safe_text = latex_escape(text)
        self._latex_content.append(f"{safe_text}\\\\[1em]")

    def section(self, title: str):
        safe_title = latex_escape(title)
        self._latex_content.append(f"\\section{{{safe_title}}}")

    def subsection(self, title: str):
        safe_title = latex_escape(title)
        self._latex_content.append(f"\\subsection{{{safe_title}}}")

    def load_input(self):
        """Open the original input NetCDF, including all variables and metadata."""
        metadata = self.manifest.get("input")
        if not metadata:
            raise ValueError("No input file was recorded in the manifest.")
        return BreakpointData(
            label="Input", dataset=xr.open_dataset(metadata["file"]), metadata=metadata
        )

    def load_bkp(self, label: str):
        """Loads a breakpoint's file and its manifest metadata, supporting shadow PA files."""
        # Find the breakpoint in the manifest
        bkp_meta = next((b for b in self.manifest.get("breakpoints", []) if b["label"] == label), None)

        # Check if it's actually the final output
        if not bkp_meta:
            fo_meta = self.manifest.get("final_output", {})
            if fo_meta.get("label") == label:
                bkp_meta = fo_meta

        # Fallback for Precision Analysis files not tracked in the manifest
        if not bkp_meta and label.endswith("_pa"):
            base_label = label[:-3] # Remove '_pa' to find the nominal breakpoint
            base_meta = next((b for b in self.manifest.get("breakpoints", []) if b["label"] == base_label), None)
            
            # If we didn't find the base label in breakpoints, check final_output
            if not base_meta:
                fo_meta = self.manifest.get("final_output", {})
                if fo_meta.get("label") == base_label:
                    base_meta = fo_meta
            # ---------------------------
            
            if base_meta:
                from pathlib import Path
                # Clone the nominal metadata so we have a valid dictionary to return
                bkp_meta = base_meta.copy()
                bkp_meta["label"] = label
                
                # Reconstruct the file path (e.g., breakpoints/name.nc -> breakpoints/name_pa.nc)
                nom_file = Path(base_meta["file"])
                bkp_meta["file"] = str(nom_file.with_name(f"{nom_file.stem}_pa{nom_file.suffix}"))

        if not bkp_meta:
            raise ValueError(f"Breakpoint '{label}' not found in manifest, and no nominal base breakpoint was found either.")
            
        file_path = self.output_root / bkp_meta["file"]
        
        # Gracefully handle both Zarr (for Dask) and NetCDF
        if file_path.suffix == '.zarr':
            ds = xr.open_zarr(file_path)
        else:
            ds = xr.open_dataset(file_path)
            
        return BreakpointData(label=label, dataset=ds, metadata=bkp_meta)

    def plot_1d_images(self, bkp: BreakpointData, var_name: str, indexes: tuple):
        """Generates a 1D plot for a precisely specified variable."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        data_array = bkp.dataset[var_name]
        
        # Slice the array down to 1D based on indexes
        slice_dict = {}
        for i, dim in enumerate(data_array.dims[:-1]):
            slice_dict[dim] = indexes[i] if i < len(indexes) else 0
            
        plot_data = data_array.isel(**{k: v for k, v in slice_dict.items() if k in data_array.dims})
        
        ax.plot(plot_data.values)
        ax.set_title(f"Breakpoint: {bkp.label}\nVariable: {var_name}")
        ax.set_xlabel(plot_data.dims[0] if plot_data.dims else "Index")
        ax.set_ylabel("Value")
        ax.grid(True)
        
        desc = f"1D visualization of variable {var_name} in breakpoint {bkp.label}. Execution time: {bkp.metadata.get('duration_sec')}s."
        self.plot(fig, desc)

    def plot_2d_images(self, bkp: BreakpointData, var_name: str, indexes: tuple):
        """Generates a 2D plot for a precisely specified variable."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data_array = bkp.dataset[var_name]
        
        # Slice the array down to 2D based on indexes
        slice_dict = {}
        for i, dim in enumerate(data_array.dims[:-2]):
            slice_dict[dim] = indexes[i] if i < len(indexes) else 0
            
        plot_data = data_array.isel(**{k: v for k, v in slice_dict.items() if k in data_array.dims})
        
        im = ax.imshow(plot_data.values, cmap='viridis')
        fig.colorbar(im, ax=ax)
        
        ax.set_title(f"Breakpoint: {bkp.label}\nVariable: {var_name}")
        
        desc = f"2D visualization of variable {var_name} in breakpoint {bkp.label}. Execution time: {bkp.metadata.get('duration_sec')}s."
        self.plot(fig, desc)

    def plot_precision_comparison(self, bkp_nom, bkp_pa, threshold: float = 1e-7, target_dtype: str = "float64"):
        """Compare floating-point values against a higher-precision replay."""
        return self._plot_comparison(bkp_nom, bkp_pa, threshold, floating_only=True)

    def plot_run_comparison(self, reference, current, threshold: float = 1e-7):
        """Compare numeric products from two runs, with reference-relative errors."""
        return self._plot_comparison(reference, current, threshold, floating_only=False)

    def _plot_comparison(self, bkp_nom, bkp_pa, threshold, floating_only):
        import numpy as np
        import matplotlib.pyplot as plt

        ds_nom = bkp_nom.dataset
        ds_pa = bkp_pa.dataset
        
        safe_label = latex_escape(bkp_nom.label)
        self._latex_content.append(f"\\subsubsection*{{Breakpoint: {safe_label}}}")

        # Track if we found any valid floating point arrays to compare
        compared_any = False

        if not floating_only:
            self.add_property("Reference-only variables", sorted(set(ds_nom.data_vars) - set(ds_pa.data_vars)))
            self.add_property("Current-only variables", sorted(set(ds_pa.data_vars) - set(ds_nom.data_vars)))
        for var_name in ds_nom.data_vars:
            # Skip if variable missing in PA run
            if var_name not in ds_pa.data_vars:
                continue

            nominal = ds_nom[var_name]
            candidate = ds_pa[var_name]
            if (nominal.dims != candidate.dims or nominal.shape != candidate.shape
                    or not nominal.coords.to_dataset().equals(candidate.coords.to_dataset())):
                self.print(f"Skipped {var_name}: dimensions or coordinates differ between runs.")
                continue
            arr_nom = nominal.values
            arr_pa = candidate.values

            # We only care about floating-point drift
            numeric_kinds = 'f' if floating_only else 'biuf'
            if arr_nom.dtype.kind not in numeric_kinds or arr_pa.dtype.kind not in numeric_kinds:
                if not floating_only:
                    self.add_property(f"{var_name} (reference)", np.array2string(arr_nom, threshold=10))
                    self.add_property(f"{var_name} (current)", np.array2string(arr_pa, threshold=10))
                continue
            # Promote integers before subtraction, including unsigned arrays.
            arr_nom = arr_nom.astype(np.float64)
            arr_pa = arr_pa.astype(np.float64)

            compared_any = True
            safe_var = latex_escape(var_name)

            # 1. Calculate Errors
            # Absolute Error
            abs_err = np.abs(arr_nom - arr_pa)
            
            # Relative Error (with epsilon to prevent division by zero)
            epsilon = 1e-12
            relative_to = arr_pa if floating_only else arr_nom
            rel_err = abs_err / (np.abs(relative_to) + epsilon)

            # Compute Summary Metrics (ignoring NaNs which are common in EO data)
            mean_abs_err = np.nanmean(abs_err)
            max_abs_err = np.nanmax(abs_err)
            mean_rel_err = np.nanmean(rel_err)
            
            flagged = mean_abs_err > threshold
            color = "red" if flagged else "black"

            # 2. Append Text Metrics to LaTeX
            self._latex_content.append(
                f"\\noindent\\textbf{{Variable ({safe_var}):}} "
                f"\\textcolor{{{color}}}{{Mean Abs Err = {mean_abs_err:.2e}}} "
                f"(Max Abs: {max_abs_err:.2e}, Mean Rel: {mean_rel_err:.2e}, Threshold: {threshold:.2e}) "
                f"\\textbf{{({'FLAGGED' if flagged else 'OK'})}}\\\\[0.5em]"
            )

            # 3. Generate Plots
            # Skip plotting for 0D scalars
            if arr_nom.ndim == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # -- 1D Data Plotting --
            if arr_nom.ndim == 1:
                ax1.plot(abs_err, color='blue', linewidth=1)
                ax1.set_title(f"Absolute Error: {var_name}")
                ax1.set_xlabel("Index")
                ax1.set_ylabel("Absolute Error")
                ax1.grid(True, linestyle='--', alpha=0.6)

                ax2.plot(rel_err, color='orange', linewidth=1)
                ax2.set_title(f"Relative Error: {var_name}")
                ax2.set_xlabel("Index")
                ax2.set_ylabel("Relative Error")
                ax2.grid(True, linestyle='--', alpha=0.6)

            # -- 2D/3D Data Plotting --
            else:
                # If 3D+ (e.g., Channels, Y, X), slice the first elements of extra dimensions
                if arr_nom.ndim > 2:
                    idx = tuple(0 for _ in range(arr_nom.ndim - 2))
                    plot_abs = abs_err[idx]
                    plot_rel = rel_err[idx]
                    slice_str = " (Slice)"
                else:
                    plot_abs = abs_err
                    plot_rel = rel_err
                    slice_str = ""

                im1 = ax1.imshow(plot_abs, cmap='viridis', aspect='auto')
                ax1.set_title(f"Absolute Error{slice_str}: {var_name}")
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                im2 = ax2.imshow(plot_rel, cmap='plasma', aspect='auto')
                ax2.set_title(f"Relative Error{slice_str}: {var_name}")
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            plt.tight_layout()
            
            # Delegate the saving and LaTeX embedding to the existing plotting method
            comparison = "Precision" if floating_only else "Run"
            self.plot(fig, f"{comparison} difference maps for {var_name}. Left: Absolute Error. Right: Relative Error.")

        if not compared_any:
            kind = "floating-point" if floating_only else "numeric"
            self._latex_content.append(f"\\noindent\\textit{{No matching {kind} variables found for comparison.}}\\\\[0.5em]")

    def plot(self, fig, description: str):
        """Saves a matplotlib figure and adds it to the LaTeX document."""
        self._fig_counter += 1
        fig_filename = f"fig_{self._fig_counter}.png"
        fig_path = self._figures_dir / fig_filename
        
        fig.savefig(fig_path, bbox_inches='tight')
        
        if self._show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        safe_desc = latex_escape(description)
            
        self._latex_content.append("\\begin{figure}[H]")
        self._latex_content.append("\\centering")
        self._latex_content.append(f"\\includegraphics[width=0.8\\textwidth]{{{fig_path.relative_to(self.output_root).as_posix()}}}")
        self._latex_content.append(f"\\caption{{{safe_desc}}}")
        self._latex_content.append("\\end{figure}")

    def report_save_pdf(self, filename: str):
        """Compiles the accumulated content into a PDF using pdflatex."""
        
        # Create a dedicated pdf/ directory inside output_root
        pdf_dir = self.output_root / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        
        tex_path = pdf_dir / filename.replace('.pdf', '.tex')
        pdf_path = pdf_dir / filename
        
        latex_template = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}
\\usepackage{{float}}
\\geometry{{a4paper, margin=1in}}

\\title{{EOPL Mission Simulator Report}}
\\author{{{latex_escape(self.author_name)}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

{chr(10).join(self._latex_content)}

\\end{{document}}
"""
        with open(tex_path, "w") as f:
            f.write(latex_template)
            
        print(f"Compiling LaTeX report: {tex_path}")
        print("-" * 40)
        
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "-output-directory", str(pdf_dir), str(tex_path)],
                           cwd=self.output_root, check=True, capture_output=True, text=True)
            print("-" * 40)
            print(f"Report generated successfully: {pdf_path}")
        except FileNotFoundError:
            print("-" * 40)
            print("WARNING: 'pdflatex' not found on system. Saved .tex file only.")
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"LaTeX compilation failed; see {tex_path.with_suffix('.log')}\n{error.stdout[-2000:]}") from error
        return pdf_path if pdf_path.exists() else tex_path
