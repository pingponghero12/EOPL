import yaml
import time
import numpy as np
from pathlib import Path
from typing import Callable, Any, Dict, List
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
import inspect
import shutil
import atexit
import json
import textwrap
import argparse
import xarray as xr

from .data import image_t
from .exceptions import ShapeError
from .ccdb import CCDB
from .log import Log


@dataclass
class RunOptions:
    """Centralized execution settings passed to all EOPL operations."""
    use_cuda: bool = False
    num_cores: int = 1
    precision: str = "float32"
    kernel_edge_access: str = "zero"  # "zero" or "edge"
    kernel_margin_to_add: int = 0
    dask_enabled: bool = False
    dask_chunk_direction: str = None # "ACT", "ALT", "Channels"

class Pipeline:
    def __init__(self):
        self.log = Log()
        self.local_config = {}
        self.global_config = {}
        self.ccdb = None
        self.rng = None
        self.manifest = {
            "eopl_version": "0.1.0",
            "operations": [],
            "breakpoints": [],
            "intermediates": [],
            "custom_report_blocks": []
        }
        
        # Run options defaults
        self.run_options = RunOptions()
        self.precision = self.run_options.precision
        
        # Internal state
        self._last_bkp_time = time.time()
        self._output_root = Path("outputs/")
        self._start_time = time.time()
        self._is_pa_run = False
        self._comparison_manifest_path = None
        self._generate_breakpoints = True

        # Fast-forward / Resume state
        self._target_start_bkp = None
        self._is_skipping = False

        # Centralize the temp directory path so the cleanup method knows where it is
        self._temp_dir = self._output_root / "temp"
        self._dask_transition_counter = 0
        self._variables = xr.Dataset()
        self._variable_names = set()
        self._pinned_vars = set()
        
        # Register the cleanup hook. This guarantees execution even on fatal Python exceptions.
        atexit.register(self.cleanup_temp_files)

    def _free_vram(self, data: image_t):
        if data is None:
            return None

        if data._device_array is not None:
            self.log.debug(f"Freeing VRAM at pointer {data._device_array}")
            data.release_device()

        return data

    @staticmethod
    def _images(data):
        """Normalize the public one-image-or-tuple form without wrapping it."""
        if data is None:
            return ()
        images = data if isinstance(data, tuple) else (data,)
        required = (
            "image", "array_name", "is_dask", "mark_host_modified",
            "mark_device_modified", "mark_synchronized", "release_device",
        )
        if not images or not all(
            all(hasattr(item, attribute) for attribute in required) for item in images
        ):
            raise TypeError("data must be an image_t or a non-empty tuple of image_t objects.")
        return images

    @staticmethod
    def _input_access(func, count):
        access = getattr(func, "input_access", None)
        if access is None:
            return ("readwrite",) * count
        if isinstance(access, str):
            access = (access,) * count
        access = tuple(access)
        if len(access) != count or any(mode not in {"read", "readwrite"} for mode in access):
            raise ValueError(
                f"Operation '{func.__name__}' input_access must contain one "
                "'read' or 'readwrite' value per input image."
            )
        return access

    def _sync_all_to_host(self, data):
        for item in self._images(data):
            self._sync_vram_to_host(item)
        return data

    def _dataset_for_save(self, data=None, clean=True):
        images = self._images(data)
        names = [item.array_name for item in images]
        if len(names) != len(set(names)):
            raise ValueError("All images saved together must have unique names.")

        # Every accumulated variable belongs to the product being written now.
        # ``pinned`` controls whether it remains available for later products.
        variable_names = self._variable_names
        variables = xr.Dataset(
            {name: self._variables[name] for name in sorted(variable_names)}
        )
        overlap = set(names).intersection(variable_names)
        if overlap:
            raise ValueError(f"Image and pipeline variable names overlap: {sorted(overlap)}")

        dataset = xr.Dataset({item.array_name: item.image() for item in images})
        dataset = xr.merge([dataset, variables], compat="no_conflicts")
        dataset.attrs["_eopl_image_names"] = json.dumps(names)
        dataset.attrs["_eopl_variable_names"] = json.dumps(sorted(variable_names))
        if len(names) == 1:
            dataset.attrs["_eopl_array_name"] = names[0]
        return dataset

    def _restore_dataset(self, dataset):
        encoded_names = dataset.attrs.get("_eopl_image_names")
        if encoded_names:
            names = json.loads(encoded_names)
        else:
            legacy_name = dataset.attrs.get("_eopl_array_name")
            names = [legacy_name or next(iter(dataset.data_vars))]

        images = tuple(image_t(dataset[name], name=name) for name in names)
        encoded_variables = dataset.attrs.get("_eopl_variable_names")
        if encoded_variables:
            auxiliary = json.loads(encoded_variables)
        else:
            auxiliary = [name for name in dataset.data_vars if name not in names]
        self._variables = xr.Dataset({name: dataset[name] for name in auxiliary})
        self._variable_names = set(auxiliary)
        self._pinned_vars = {
            name for name in auxiliary
            if dataset[name].attrs.get("_eopl_pinned") in (True, 1, "true", "True")
        }
        result = None if not images else images[0] if len(images) == 1 else images
        for item in images:
            self._enforce_precision(item)
        return result

    def _ensure_vram(self, data: image_t):
        """Make the device copy current, transferring only when it is stale or absent."""
        from eopal import eopal_cpp_core

        if not getattr(eopal_cpp_core, "HAS_CUDA_COMPILED", False):
            raise RuntimeError("EOPAL was built without CUDA support.")

        image = data.image().values
        if image.dtype != np.float32:
            raise TypeError(
                f"The current EOPAL CUDA backend requires float32 image data, got {image.dtype}."
            )
        if image.ndim != 3:
            raise ValueError(
                f"The current EOPAL CUDA backend requires a 3-D image, got {image.ndim}-D."
            )
        if not image.flags.c_contiguous:
            raise ValueError("The current EOPAL CUDA backend requires C-contiguous image data.")

        layout = (tuple(image.shape), image.dtype.str, image.nbytes)
        allocated_layout = (data._device_shape, data._device_dtype, data._device_nbytes)

        if data._device_array is not None and allocated_layout != layout:
            if data._host_dirty:
                raise RuntimeError(
                    "The image layout changed while the device held newer data; "
                    "synchronize before replacing the host image."
                )
            data = self._free_vram(data)

        if data._device_array is None:
            self.log.debug("Allocating VRAM and copying the current host image.")
            data._device_array = eopal_cpp_core.push_to_vram_numpy(image, 0)
            data._device_shape, data._device_dtype, data._device_nbytes = layout
            data.mark_synchronized()
        elif data._device_dirty:
            self.log.debug("Refreshing stale VRAM from the current host image.")
            data._device_array = eopal_cpp_core.push_to_vram_numpy(
                image, data._device_array
            )
            data.mark_synchronized()

        return data

    def _assert_no_vram_while_dask(self, data: image_t):
        if (
            data is not None
            and data.is_dask
            and data is not None
            and data._device_array is not None
        ):
            raise RuntimeError("Internal error: Dask-backed data must not keep a VRAM pointer.")

    def start_bkp(self, label: str):
        """Configures the pipeline to skip all operations until it finds this breakpoint."""

        # The fast forward mode needs to be rethought, tho current is ok enough
        # It should be possible not to force user to take care of it
        self._target_start_bkp = label
        self._is_skipping = True
        self.log.info(f"Pipeline configured to fast-forward. Skipping operations until breakpoint: '{label}'")

    def cleanup_temp_files(self):
        """Safely removes the temporary transition directory to prevent disk bloat."""
        # getattr is a safety net in case __init__ fails before _temp_dir is created
        temp_dir = getattr(self, '_temp_dir', None)
        
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                # Optional: you can log this or keep it silent
                # print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                self.log.warning(f"Warning: Failed to clean up temp directory {temp_dir}. Reason: {e}")

    def __del__(self):
        """Destructor to clean up when the Pipeline object is garbage collected naturally."""
        self.cleanup_temp_files()

    def image(self, data: image_t):
        if data == None:
            self.log.warning("Cant return image() for non initialized image. Check if you are not running fastforward mode to start from some BKP later.")
            return None
        return data.image()

    def parse_arguments(self, argv=None):
        """Parse ICD paths, precision replay, or a named comparison run."""
        parser = argparse.ArgumentParser(description="Run an EOPL pipeline")
        parser.add_argument("paths", nargs="*", help="global.xml local.xml input.nc output")
        parser.add_argument("--precision-analysis", action="store_true")
        parser.add_argument("--dir", help="Save under output/NAME and compare with output's baseline run")
        options = parser.parse_args(argv)
        if options.paths and len(options.paths) != 4:
            parser.error("Supply all four ICD paths: global local input output")
        paths = options.paths or ["global.xml", "local.xml", "input/input_image.nc", "outputs"]
        args = dict(zip(("global_config_path", "local_config_path", "input_path", "output_root"), paths))
        if options.precision_analysis and not (Path(args["output_root"]) / "run_manifest.yaml").exists():
            parser.error("Run the baseline without --precision-analysis first")
        self._is_pa_run = options.precision_analysis
        self._comparison_manifest_path = None
        if options.dir is not None:
            name = Path(options.dir)
            if not options.dir or name.is_absolute() or len(name.parts) != 1 or options.dir in (".", ".."):
                parser.error("--dir must be a single directory name, such as doubled")
            if options.precision_analysis:
                parser.error("Use --dir for run comparison, or --precision-analysis for precision replay")
            baseline = Path(args["output_root"]) / "run_manifest.yaml"
            if not baseline.exists():
                parser.error(f"Run the baseline without --dir first; missing {baseline}")
            self._comparison_manifest_path = baseline.resolve()
            args["output_root"] = str(Path(args["output_root"]) / name)
            self.manifest["comparison"] = {
                "name": options.dir, "reference_manifest": str(self._comparison_manifest_path)
            }
        self._output_root = Path(args["output_root"])
        self._temp_dir = self._output_root / "temp"

        return args

    def add_local_config(self, path: str):
        tree = ET.parse(path)
        for param in tree.getroot().findall("parameter"):
            self.local_config[param.attrib["name"]] = param.text

    def add_global_config(self, path: str):
        tree = ET.parse(path)
        for param in tree.getroot().findall("parameter"):
            self.global_config[param.attrib["name"]] = param.text
        
        seed = int(self.global_config.get("RandomSeed", 0))
        self.rng = np.random.default_rng(seed=seed)
        self.manifest["random_seed"] = seed

    def add_ccdb(self, path: str):
        self.ccdb = CCDB(path)
        self.manifest["ccdb"] = {"directory": str(self.ccdb.dir.resolve())}

    def set_run_options(self, use_cuda: bool = False, num_cores: int = 1, precision: str = "float32"):
        self.run_options.use_cuda = use_cuda
        self.run_options.num_cores = num_cores

        forced_precision = "float64" if self._is_pa_run else precision
        self.run_options.precision = forced_precision

        self.manifest["hardware_target"] = "GPU_CUDA" if use_cuda else "CPU"
        self.manifest["num_cores"] = num_cores
        self.manifest["precision"] = forced_precision

    def _enforce_precision(self, data: image_t):
        """
        Enforces floating-point precision across the pipeline to prevent accidental upcasting.
        If the target precision is not floating point, arrays are left unchanged.
        """
        if data is None or data.image() is None:
            return data

        precision = self.run_options.precision

        if precision not in ["float32", "float64"]:
            return data

        target_dtype = np.float32 if precision == "float32" else np.float64

        current_dtype = data.image().dtype
        if current_dtype.kind == "f" and current_dtype != target_dtype:
            data.data = data.image().astype(target_dtype)
            data.mark_host_modified()

        return data

    def set_image_type(self, precision: str = "float32"):
        # Only hijack precision for PA if the base requested precision is a float
        if self._is_pa_run and precision in ["float32", "float16"]:
            forced_precision = "float64"
        else:
            forced_precision = precision
            
        self.run_options.precision = forced_precision
        self.manifest["precision"] = forced_precision

    def generate_report(self, enabled: bool, print_all_bkp_data: bool = False, 
                        plot_1d_images: bool = True, plot_1d_indexes: tuple = (0,),
                        plot_2d_images: bool = True, plot_2d_indexes: tuple = (0, 0, 0),
                        report_format: str = "py"):
        """Generate an editable Python script or a Jupyter notebook (ipynb)."""
        if report_format not in ("py", "ipynb"):
            raise ValueError("report_format must be 'py' or 'ipynb'.")
        self.manifest["report_config"] = {
            "enabled": enabled,
            "report_format": report_format,
            "print_all_bkp_data": print_all_bkp_data,
            "plot_1d_images": plot_1d_images,
            "plot_1d_indexes": list(plot_1d_indexes),
            "plot_2d_images": plot_2d_images,
            "plot_2d_indexes": list(plot_2d_indexes)
        }

    def generate_breakpoints(self, enabled: bool):
        self._generate_breakpoints = enabled

    def setup_precision_analysis(self, precision: str = "float64", threshold_mean_abs_diff: float = 1e-7):
        if precision != "float64":
            raise ValueError("Precision replay currently supports float64 only.")
        if not np.isfinite(threshold_mean_abs_diff) or threshold_mean_abs_diff < 0:
            raise ValueError("The precision-analysis threshold must be finite and nonnegative.")
        self.manifest["precision_analysis"] = {
            "enabled": True,
            "comparison_precision": precision,
            "threshold_mean_abs_diff": threshold_mean_abs_diff
        }

    def read_input(self, path: str, dask: bool = False, dask_chunk_direction: str = "Channels"):
        import xarray as xr

        self.manifest["input"] = {"file": str(Path(path).resolve())}
        if self._is_skipping:
            self.log.debug("Skipping read_input due to fast-forward mode.")
            return

        if dask:
            # Chunking using a dictionary isolates the chunking to the specific dimension
            dataset = xr.open_dataset(path, chunks={dask_chunk_direction: 'auto'})
            self.run_options.dask_enabled = True
            self.run_options.dask_chunk_direction = dask_chunk_direction
        else:
            dataset = xr.open_dataset(path)
            self.run_options.dask_enabled = False
            self.run_options.dask_chunk_direction = None

        data = self._restore_dataset(dataset)
        if isinstance(data, tuple):
            raise ValueError(
                "Input contains multiple images; load it through a breakpoint or "
                "select a single image variable explicitly."
            )
        self._last_bkp_time = time.time()

        return data

    def run_operation(
        self,
        data,
        func: Callable,
        args: Dict[str, Any] = None,
        dask_acc_chunks: List[str] = None,
    ):
        """Run an operation on one image or a positional tuple of images.

        The operation's return value is returned unchanged.  This is what makes
        ordinary Python unpacking work for any number of image and scalar/array
        outputs, without a pipeline-specific output dictionary.
        """
        args = dict(args or {})
        if self._is_skipping:
            self.log.debug(f"Skipping operation '{func.__name__}'...")
            return data

        images = self._images(data)
        if not images:
            raise TypeError("run_operation requires image data; use run_function instead.")
        access = self._input_access(func, len(images))
        func_supports_cuda = getattr(func, "cuda_support", False)
        func_supports_dask = getattr(func, "dask_support", False)
        dask_images = [item for item in images if item.is_dask]
        dask_acc_chunks = dask_acc_chunks or ["Channels"]

        if dask_images:
            for item in dask_images:
                self._assert_no_vram_while_dask(item)
            if not func_supports_dask:
                raise RuntimeError(
                    f"Operation '{func.__name__}' does not support Dask, "
                    "but at least one input is Dask-backed."
                )
            for item in dask_images:
                chunked_dims = {
                    dim for dim, chunks in zip(item.image().dims, item.image().chunks)
                    if len(chunks) > 1
                }
                if chunked_dims and chunked_dims.isdisjoint(dask_acc_chunks):
                    self.set_dask(item, True, dask_acc_chunks[0])

        if hasattr(func, "shape_contract"):
            dims = dict(images[0].image().sizes)
            func.shape_contract(dims, args)
        elif hasattr(func, "eopl_shape_contract"):
            dims = dict(images[0].image().sizes)
            func.eopl_shape_contract(dims, args)

        sig = inspect.signature(func)
        if "rng" in sig.parameters:
            args["rng"] = self.rng
        actual_cuda_used = (
            self.run_options.use_cuda
            and not dask_images
            and func_supports_cuda
        )

        if actual_cuda_used:
            supported_dtypes = getattr(func, "cuda_dtypes", ("float32",))
            for item in images:
                image_dtype = str(item.image().dtype)
                if image_dtype not in supported_dtypes:
                    raise TypeError(
                        f"Operation '{func.__name__}' supports CUDA dtypes "
                        f"{supported_dtypes}, got {image_dtype}."
                    )
                self._ensure_vram(item)
            current_run_options = self.run_options
        else:
            current_run_options = replace(
                self.run_options,
                use_cuda=False,
                dask_enabled=bool(dask_images),
            )
            # A CPU operation only downloads device data that is newer.  Current
            # device allocations remain cached for a later GPU operation.
            for item in images:
                self._sync_vram_to_host(item)

        if "run_options" in sig.parameters:
            args["run_options"] = current_run_options
        start_time = time.time()
        try:
            result = func(*images, **args)
        except Exception:
            if actual_cuda_used:
                for item, mode in zip(images, access):
                    if mode == "readwrite":
                        item._host_dirty = False
                        self._free_vram(item)
            raise

        if actual_cuda_used:
            for item, mode in zip(images, access):
                if mode == "readwrite":
                    item.mark_device_modified()
        else:
            for item, mode in zip(images, access):
                if mode == "readwrite":
                    item.mark_host_modified()
                    self._enforce_precision(item)
            returned = result if isinstance(result, tuple) else (result,)
            for item in returned:
                if all(
                    hasattr(item, attribute)
                    for attribute in ("image", "mark_host_modified", "array_name")
                ) and not any(item is source for source in images):
                    self._enforce_precision(item)

        duration = time.time() - start_time
        backend_str = "CUDA" if actual_cuda_used else "CPU"
        if dask_images:
            backend_str += " (Dask)"
        self.manifest["operations"].append({
            "name": func.__name__,
            "duration_sec": round(duration, 4),
            "backend": backend_str,
        })
        self.log.info(f"Completed {func.__name__} via {backend_str} in {duration:.4f}s")
        return result

    def run_function(self, func: Callable, args: Dict[str, Any] = None):
        """Run a pipeline function that does not consume an image."""
        args = dict(args or {})
        if self._is_skipping:
            self.log.debug(f"Skipping function '{func.__name__}'...")
            return None
        signature = inspect.signature(func)
        dependencies = {
            "run_options": self.run_options,
            "rng": self.rng,
            "global_config": self.global_config,
            "local_config": self.local_config,
            "ccdb": self.ccdb,
        }
        for name, value in dependencies.items():
            if name in signature.parameters:
                args[name] = value
        start_time = time.time()
        result = func(**args)
        duration = time.time() - start_time
        self.manifest["operations"].append({
            "name": func.__name__,
            "type": "function",
            "duration_sec": round(duration, 4),
            "backend": "CPU",
        })
        return result

    def save_bkp(
        self,
        data=None,
        label: str = None,
        enabled: bool = True,
        file_type: str = "nc",
        dask: bool = False,
        dask_chunk_direction: str = "Channels",
        clean: bool = True,
    ):
        if isinstance(data, str) and label is None:
            label, data = data, None
        if not label:
            raise ValueError("A breakpoint label is required.")
        if file_type not in ("nc", "zarr"):
            raise ValueError(f"Unsupported file type '{file_type}'. Use 'nc' or 'zarr'.")
        save_label = f"{label}_pa" if self._is_pa_run else label
        filename = self._output_root / f"breakpoints/{save_label}.{file_type}"
        if self._is_skipping:
            if label == self._target_start_bkp:
                self.log.info(f"Reached target breakpoint '{label}'. Resuming pipeline execution.")
                self._is_skipping = False
                if not filename.exists():
                    raise FileNotFoundError(
                        f"Cannot resume from breakpoint '{label}'. File not found at {filename}"
                    )
                if file_type == "nc":
                    dataset = xr.open_dataset(
                        filename,
                        chunks={dask_chunk_direction: "auto"} if dask else None,
                    )
                else:
                    dataset = xr.open_zarr(
                        filename,
                        chunks={dask_chunk_direction: "auto"} if dask else None,
                    )
                if not dask:
                    dataset = dataset.compute()
                self._last_bkp_time = time.time()
                return self._restore_dataset(dataset)
            self.log.debug(f"Skipping breakpoint '{label}'...")
            return data

        enabled = enabled and self._generate_breakpoints
        if not dask:
            for item in self._images(data):
                if item.is_dask:
                    self.set_dask(item, False)
        needs_serialization = enabled
        if needs_serialization:
            self._sync_all_to_host(data)
            filename.parent.mkdir(parents=True, exist_ok=True)
            dataset = self._dataset_for_save(data, clean=clean)
            if file_type == "nc":
                dataset.to_netcdf(filename)
            else:
                dataset.to_zarr(filename, mode="w")

        duration = time.time() - self._last_bkp_time
        self._last_bkp_time = time.time()
        if enabled:
            self.manifest["breakpoints"].append({
                "label": label,
                "file": str(filename.relative_to(self._output_root)),
                "duration_sec": round(duration, 4),
            })
            if clean:
                self._clean_unpinned_variables()
        if dask:
            for item in self._images(data):
                self.set_dask(item, True, dask_chunk_direction)
        return data

    def set_dask(self, data: image_t, dask: bool = False, dask_chunk_direction: str = "Channels"):
        if data is None:
            raise RuntimeError("No data available. Cannot change Dask state before loading data.")
        self._images(data)
        if not dask and not data.is_dask:
            return data
        if dask and data.is_dask:
            chunks = dict(zip(data.image().dims, data.image().chunks))
            if (
                dask_chunk_direction in chunks
                and len(chunks[dask_chunk_direction]) > 1
                and all(
                    dim == dask_chunk_direction or len(dim_chunks) == 1
                    for dim, dim_chunks in chunks.items()
                )
            ):
                self.run_options.dask_enabled = True
                self.run_options.dask_chunk_direction = dask_chunk_direction
                return data
        if dask and dask_chunk_direction not in data.image().dims:
            raise ValueError(
                f"Cannot chunk along missing dimension '{dask_chunk_direction}'."
            )

        self._sync_vram_to_host(data)
        self._free_vram(data)
        if dask:
            data.data = data.image().chunk({
                dim: "auto" if dim == dask_chunk_direction else -1
                for dim in data.image().dims
            })
            self.run_options.dask_enabled = True
            self.run_options.dask_chunk_direction = dask_chunk_direction
            return data
        data.data = data.data.compute()
        self.run_options.dask_enabled = False
        self.run_options.dask_chunk_direction = None
        return data

    def start_dask_client(self):
        """Starts a local Dask cluster matching the pipeline's core count."""
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(
            n_workers=self.run_options.num_cores,
            threads_per_worker=1,
        )
        self.dask_client = Client(cluster)
        self.log.info(f"Dask client started: {self.dask_client.dashboard_link}")

    def add_variable(self, variable_name: str, variable, pinned: bool = False,
                     dims: tuple = None):
        """Add or replace a value in the pipeline-owned output accumulator."""
        if variable_name in self._variables:
            self._variables = self._variables.drop_vars(variable_name)
        if dims is not None:
            values = variable.data if isinstance(variable, xr.DataArray) else variable
            self._variables[variable_name] = (tuple(dims), values)
        else:
            self._variables[variable_name] = variable
        self._variable_names.add(variable_name)
        if pinned:
            self._pinned_vars.add(variable_name)
            self._variables[variable_name].attrs["_eopl_pinned"] = "true"
        else:
            self._pinned_vars.discard(variable_name)
            self._variables[variable_name].attrs.pop("_eopl_pinned", None)
        return variable

    def _clean_unpinned_variables(self):
        removable = self._variable_names.difference(self._pinned_vars)
        if removable:
            self._variables = self._variables.drop_vars(sorted(removable))
            self._variable_names.difference_update(removable)

    def change_variable_name(self, variable_name: str, new_name: str):
        if variable_name not in self._variables:
            raise KeyError(variable_name)
        self._variables = self._variables.rename({variable_name: new_name})
        self._variable_names.remove(variable_name)
        self._variable_names.add(new_name)
        if variable_name in self._pinned_vars:
            self._pinned_vars.remove(variable_name)
            self._pinned_vars.add(new_name)

    def add_to_bkp(self, data, variable_name: str, variable,
                   pinned: bool = False, dims: tuple = None):
        """Compatibility wrapper; new pipelines should call add_variable."""
        self.add_variable(variable_name, variable, pinned=pinned, dims=dims)
        return data

    def _sync_vram_to_host(self, data: image_t):
        if data is None:
            return None

        if data._device_array is not None and data._host_dirty:
            from eopal import eopal_cpp_core

            eopal_cpp_core.copy_to_host_numpy(
                data.image().values,
                data._device_array,
            )
            data.mark_synchronized()
        return data

    def save_output(self, data=None, label: str = None, file_type: str = "nc", clean: bool = True):
        """
        Saves the final dataset state to disk and registers it in the manifest.
        Unlike breakpoints, this represents the canonical final product of the pipeline.
        """
        if isinstance(data, str) and label is None:
            label, data = data, None
        if not label:
            raise ValueError("An output label is required.")
        if self._is_skipping:
            error_msg = f"Pipeline reached the end while trying to fast-forward to breakpoint '{self._target_start_bkp}'. Breakpoint not found."
            self.log.error(error_msg)
            raise RuntimeError(error_msg)

        save_label = f"{label}_pa" if self._is_pa_run else label
        filename = self._output_root / f"{save_label}.{file_type}"
        filename.parent.mkdir(parents=True, exist_ok=True)
        self._sync_all_to_host(data)
        dataset = self._dataset_for_save(data, clean=clean)
        if file_type == "nc":
            dataset.to_netcdf(filename)
        elif file_type == "zarr":
            dataset.to_zarr(filename, mode='w')
        else:
            raise ValueError(f"Unsupported file type '{file_type}'. Use 'nc' or 'zarr'.")

        # Capture the timing for the final operation phase
        duration = time.time() - self._last_bkp_time
        self._last_bkp_time = time.time()

        # Register the final output explicitly in the manifest
        self.manifest["final_output"] = {
            "label": label,
            "file": str(filename.relative_to(self._output_root)),
            "duration_sec": round(duration, 4)
        }
        
        self.finish(data)

    def finish(self, data=None):
        manifest_path = self._output_root / "run_manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self._is_pa_run:
            # Nominal run behavior: save manifest and create standard report
            self.manifest["total_execution_time_sec"] = (time.time() - self._start_time)
            with open(manifest_path, "w") as f:
                yaml.dump(self.manifest, f, sort_keys=False)
                
            self._generate_report_script(manifest_path)
            if self._comparison_manifest_path is not None:
                self._generate_report_comparison(manifest_path)
            self.log.info("Pipeline finished successfully. Manifest and report written.")
        else:
            # PA run behavior: DO NOT overwrite the manifest. Just generate the PA report.
            self._generate_report_script_precision(manifest_path)
            self.log.info("Precision analysis pipeline finished successfully. Precision report written.")

        for item in self._images(data):
            self._free_vram(item)

    def report_section(self, title: str):
        """Adds a main section to the generated report."""
        self.manifest["custom_report_blocks"].append({"type": "section", "content": title})

    def report_subsection(self, title: str):
        """Adds a subsection to the generated report."""
        self.manifest["custom_report_blocks"].append({"type": "subsection", "content": title})

    def report_print(self, text: str):
        """Adds a text paragraph to the generated report."""
        self.manifest["custom_report_blocks"].append({"type": "print", "content": str(text)})
        
    def report_add_property(self, name: str, value: Any):
        """Adds a key-value property line to the generated report."""
        self.manifest["custom_report_blocks"].append({"type": "property", "name": name, "value": str(value)})

    def report_inject_code(self, code_string: str):
        """
        Useful for custom matplotlib figures or complex logic that needs to run during report generation.
        """
        self.manifest["custom_report_blocks"].append({"type": "code", "content": code_string})

    def _generate_report_script(self, manifest_path: Path):
        """Dynamically inspects data on disk and generates the runnable report.py script."""
        import xarray as xr
        script_path = self._output_root / "report.py"
        
        pa_enabled = self.manifest.get("precision_analysis", {}).get("enabled", False)
        report_config = self.manifest.get("report_config", {})
        
        if not report_config.get("enabled", False):
            return 
            
        print_all = report_config.get("print_all_bkp_data", False)
        plot_1d = report_config.get("plot_1d_images", True)
        indexes_1d = tuple(report_config.get("plot_1d_indexes", (0,)))
        plot_2d = report_config.get("plot_2d_images", True)
        indexes_2d = tuple(report_config.get("plot_2d_indexes", (0,0,0)))
        

        custom_blocks_code = []
        for block in self.manifest.get("custom_report_blocks", []):
            b_type = block.get("type")
            if b_type == "section":
                safe_title = repr(block["content"])
                custom_blocks_code.append(f'    report.section({safe_title})')
            elif b_type == "subsection":
                safe_title = repr(block["content"])
                custom_blocks_code.append(f'    report.subsection({safe_title})')
            elif b_type == "print":
                # Use repr to safely handle quotes and newlines in the string
                safe_text = repr(block['content'])
                custom_blocks_code.append(f'    report.print({safe_text})')
            elif b_type == "property":
                safe_name = repr(block["name"])
                safe_val = repr(block['value'])
                custom_blocks_code.append(f'    report.add_property({safe_name}, {safe_val})')
            elif b_type == "code":
                # Indent the raw user code so it aligns inside the main() function
                indented = "\n".join([f"    {line}" for line in block['content'].split("\n")])
                custom_blocks_code.append(indented)
                
        custom_injection_string = "\n".join(custom_blocks_code)

        ccdb_lines = [
            '    # --- CCDB ---',
            '    # report.section("CCDB")',
        ]
        if self.ccdb is not None:
            ccdb_lines.append('    ccdb = CCDB(report.manifest["ccdb"]["directory"])')
            names = self.ccdb.keys()
            ccdb_lines.append('    # Available calibration parameters:')
            ccdb_lines.extend(f'    # ccdb.get({name!r})' for name in names)
        else:
            ccdb_lines.append('    # ccdb = CCDB("path/to/ccdb")')
            names = ()
        example_name = names[0] if names else "parameter_name"
        ccdb_lines.extend([
            '',
            f'    # value = ccdb.get({example_name!r})',
            '    # Scalar/string property or a short array preview:',
            f'    # report.add_property({example_name!r}, np.array2string(np.asarray(value), threshold=10))',
            '',
            '    # For a 1D calibration array:',
            '    # fig, ax = plt.subplots(figsize=(8, 4))',
            '    # ax.plot(np.asarray(value))',
            '    # report.plot(fig, "CCDB 1D calibration")',
            '',
            '    # For a 2D calibration array (slice extra dimensions first):',
            '    # fig, ax = plt.subplots(figsize=(8, 6))',
            '    # im = ax.imshow(np.asarray(value), cmap="viridis")',
            '    # fig.colorbar(im, ax=ax)',
            '    # report.plot(fig, "CCDB 2D calibration")',
        ])
        ccdb_injection_string = "\n".join(ccdb_lines)

        items_to_process = self.manifest.get("breakpoints", []).copy()
        if "final_output" in self.manifest:
            fo = self.manifest["final_output"].copy()
            fo["_is_final_output"] = True
            items_to_process.append(fo)
        if "input" in self.manifest:
            items_to_process.insert(0, {
                **self.manifest["input"], "label": "Input", "_is_input": True
            })

        input_block = ""
        bkp_blocks = []
        notebook_products = []
        first_index = 0 if "input" in self.manifest else 1
        for i, bkp in enumerate(items_to_process, first_index):
            label = bkp['label']
            is_final = bkp.get("_is_final_output", False)
            is_input = bkp.get("_is_input", False)
            
            var_name = "input_data" if is_input else f"product_{i}"
            
            block_lines = []
            if is_input:
                block_lines.extend([
                    '    # --- Input NetCDF ---',
                    '    report.section("Input NetCDF")',
                ])
            elif is_final:
                block_lines.extend([
                    f"    # Final Output: {label!r}",
                    f'    report.section({("Final Output: " + label)!r})'
                ])
            else:
                block_lines.extend([
                    f"    # Breakpoint {i}: {label!r}",
                    f'    report.section({("Breakpoint " + label)!r})'
                ])
                
            block_lines.extend([
                f'    {var_name} = report.load_input()' if is_input else
                f'    {var_name} = report.load_bkp({label!r})',
                ""
            ])

            ds_path = self._output_root / bkp['file']
            try:
                ds = xr.open_zarr(ds_path) if ds_path.suffix == ".zarr" else xr.open_dataset(ds_path)
                
                if is_input:
                    block_lines.extend([
                        '    report.add_property("Dimensions", dict(input_data.dataset.sizes))',
                        '    report.add_property("Global attributes", dict(input_data.dataset.attrs))',
                    ])
                    for var in ds.variables:
                        block_lines.extend([
                            f'    report.add_property({(var + " dimensions / dtype")!r}, (input_data.dataset[{var!r}].dims, str(input_data.dataset[{var!r}].dtype)))',
                            f'    report.add_property({(var + " attributes")!r}, dict(input_data.dataset[{var!r}].attrs))',
                        ])
                        if var in ds.data_vars and ds[var].ndim > 0:
                            block_lines.append(f'    report.add_property({var!r}, np.array2string(input_data.dataset[{var!r}].values, edgeitems=5, threshold=10))')

                if print_all or is_input:
                    # -- Coordinates --
                    block_lines.append("    # Default Coordinate Systems")
                    if list(ds.coords):
                        for coord in ds.coords:
                            block_lines.append(f'    report.add_property({coord!r}, np.array2string({var_name}.dataset[{coord!r}].values, edgeitems=5, threshold=10))')
                    else:
                        block_lines.append(f'    # report.add_property("Example_Coord", np.array2string({var_name}.dataset["example_coord"].values, edgeitems=5, threshold=10))')
                    block_lines.append("")
                    
                    # Inspect Variables
                    scalars, arr_1d, arr_2d = [], [], []
                    for var in ds.data_vars:
                        ndim = ds[var].ndim
                        if ndim == 0: scalars.append(var)
                        elif ds[var].dtype.kind not in "biuf": continue
                        elif ndim == 1: arr_1d.append(var)
                        else: arr_2d.append(var)
                            
                    # -- 0D Datasets --
                    block_lines.append("    # Default Dataset elements")
                    if scalars:
                        for var in scalars:
                            block_lines.append(f'    report.add_property({var!r}, {var_name}.dataset[{var!r}].values)')
                    else:
                        block_lines.append(f'    # report.add_property("Example_Scalar", {var_name}.dataset["example_scalar"].values)')
                    block_lines.append("")

                    # -- 1D Datasets --
                    block_lines.append("    # Default 1D Visualizations")
                    block_lines.append("    if plot_1d_images:")
                    block_lines.append('        report.subsection("Plots 1D")')
                    if arr_1d:
                        for var in arr_1d:
                            block_lines.append(f'        report.plot_1d_images({var_name}, {var!r}, plot_1d_indexes)')
                    else:
                        block_lines.append(f'        # report.plot_1d_images({var_name}, "example_1d_var", plot_1d_indexes)')
                    block_lines.append("")

                    # -- 2D/3D Datasets --
                    block_lines.append("    # Default 2D Visualizations")
                    block_lines.append("    if plot_2d_images:")
                    block_lines.append('        report.subsection("Plots 2D")')
                    if arr_2d:
                        for var in arr_2d:
                            block_lines.append(f'        report.plot_2d_images({var_name}, {var!r}, plot_2d_indexes)')
                    else:
                        block_lines.append(f'        # report.plot_2d_images({var_name}, "example_2d_var", plot_2d_indexes)')
                    block_lines.append("")
                else:
                    # Legacy fallback logic if print_all is false: just plot 2d objects
                    arr_2d = [var for var in ds.data_vars
                              if ds[var].ndim >= 2 and ds[var].dtype.kind in "biuf"]
                    if arr_2d:
                        block_lines.append("    # Default 2D Visualizations")
                        block_lines.append("    if plot_2d_images:")
                        block_lines.append('        report.subsection("Plots 2D")')
                        for var in arr_2d:
                            block_lines.append(f'        report.plot_2d_images({var_name}, {var!r}, plot_2d_indexes)')
                        block_lines.append("")

                inspection = [
                    f'display({var_name}.dataset)',
                    f'# display({var_name}.dataset.attrs)',
                    f'# display({var_name}.dataset.coords)',
                ]
                for var in ds.data_vars:
                    inspection.append(f'# display({var_name}.dataset[{var!r}])')
                    if ds[var].ndim > 0 and ds[var].dtype.kind in "biuf":
                        selection = {dim: 0 for dim in ds[var].dims[:-2]}
                        array = f'{var_name}.dataset[{var!r}]'
                        if selection:
                            array += f'.isel({selection!r})'
                        inspection.append(f'# {array}.plot()')
                notebook_products.append((
                    "Input NetCDF" if is_input else
                    ("Final Output: " if is_final else "Breakpoint: ") + label,
                    "\n".join(block_lines),
                    "\n".join(inspection),
                ))
                ds.close()
            except Exception as e:
                raise RuntimeError(f"Cannot inspect report product {ds_path}") from e

            block_lines.append("")
            
            if is_input:
                input_block = "\n".join(block_lines)
            else:
                bkp_blocks.append("\n".join(block_lines))

        script_content = f"""# AUTOMATICALLY GENERATED EOPL REPORT SCRIPT
import eopl.report
from eopl.ccdb import CCDB
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("{manifest_path.name}")

plot_1d_images = {plot_1d}
plot_1d_indexes = {indexes_1d}
plot_2d_images = {plot_2d}
plot_2d_indexes = {indexes_2d}
precision_analysis_enabled = {pa_enabled}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate EOPL Mission Report")
    parser.add_argument("--author", "-a", type=str, default="Unknown Author", 
                        help="Name of the report author")
    args = parser.parse_args()
    return args.author

def main():
    report = eopl.report.Report(manifest=MANIFEST_PATH)
    author = parse_args()
    report.author(author)

    report.section("Run Metadata")
    report.hardware_target(report.manifest.get("hardware_target", "Unknown"))
    report.execution_time_sec(report.manifest.get("total_execution_time_sec", 0.0))
    report.add_property("Precision", report.manifest.get("precision", "Unknown"))
    report.add_property("Number of Cores", report.manifest.get("num_cores", 1))
    report.show_plots(False)

{ccdb_injection_string}

{input_block}

    # --- Custom Pipeline Reporting ---
{custom_injection_string}
    # ---------------------------------

{chr(10).join(bkp_blocks)}
    # Custom Analysis Example
    report.section("Custom Analysis")
    # report.print("Custom Analysis")
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot([1, 2, 3], [1, 2, 3])
    # report.plot(fig, "Example custom figure description.")

    report.report_save_pdf("mission_report.pdf")

if __name__ == "__main__":
    main()
"""
        if report_config.get("report_format", "py") == "ipynb":
            sections = [("CCDB", ccdb_injection_string, "")]
            if self.ccdb is not None:
                sections[0] = ("CCDB", ccdb_injection_string,
                               'display(ccdb.keys())\n'
                               f'# calibration = ccdb.get({example_name!r})\n'
                               '# display(calibration)\n'
                               '# For an xarray.DataArray calibration:\n'
                               '# calibration.plot()')
            sections.extend(section for section in notebook_products if section[0] == "Input NetCDF")
            custom_display = []
            for block in self.manifest.get("custom_report_blocks", []):
                if block["type"] in ("section", "subsection", "print"):
                    prefix = {"section": "## ", "subsection": "### ", "print": ""}[block["type"]]
                    custom_display.append(f'display(Markdown({(prefix + block["content"])!r}))')
                elif block["type"] == "property":
                    custom_display.append(f'display({{ {block["name"]!r}: {block["value"]!r} }})')
            sections.append(("Custom Pipeline Reporting", custom_injection_string,
                             "\n".join(custom_display)))
            sections.extend(section for section in notebook_products if section[0] != "Input NetCDF")
            self._write_report_notebook(manifest_path, report_config, sections)
            return
        with open(script_path, "w") as f:
            f.write(script_content)

    def _write_report_notebook(self, manifest_path, config, sections,
                               filename="report.ipynb", precision=False, comparison=False):
        """Write standard notebook JSON without a runtime Jupyter dependency."""
        cells = []

        def add_cell(kind, source):
            cell = {
                "cell_type": kind, "id": f"cell-{len(cells)}", "metadata": {},
                "source": textwrap.dedent(source).strip().splitlines(keepends=True),
            }
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)

        title = "EOPL Run Comparison Report" if comparison else (
            "EOPL Precision Analysis Report" if precision else "EOPL Report")
        add_cell("markdown", f"# {title}\nRun cells from top to bottom. Uncomment inspection and plotting examples as needed.\n\n"
                 "Products are report wrappers: use `input_data.dataset` or `product_1.dataset` to inspect xarray data. "
                 "Select a variable with `.dataset['variable']` and slice extra dimensions with `.isel(...)` before `.plot()`.")
        add_cell("code", f'''\
            from pathlib import Path
            import numpy as np
            import matplotlib.pyplot as plt
            from IPython import get_ipython
            from IPython.display import display, Markdown
            from eopl.report import Report
            from eopl.ccdb import CCDB

            get_ipython().run_line_magic("matplotlib", "inline")
            # Open the notebook from the output directory, or edit this path.
            MANIFEST_PATH = Path({manifest_path.name!r})
            report = Report(MANIFEST_PATH)
            report.author("Your Name")
            report.show_plots(True)
            plot_1d_images = {config.get("plot_1d_images", True)!r}
            plot_1d_indexes = {tuple(config.get("plot_1d_indexes", (0,)))!r}
            plot_2d_images = {config.get("plot_2d_images", True)!r}
            plot_2d_indexes = {tuple(config.get("plot_2d_indexes", (0, 0, 0)))!r}
            precision_analysis_enabled = {self.manifest.get("precision_analysis", {}).get("enabled", False)!r}
        ''')
        add_cell("markdown", "## Run Metadata")
        add_cell("code", '''\
            report.section("Run Metadata")
            report.hardware_target(report.manifest.get("hardware_target", "Unknown"))
            report.execution_time_sec(report.manifest.get("total_execution_time_sec", 0.0))
            report.add_property("Precision", report.manifest.get("precision", "Unknown"))
            report.add_property("Number of Cores", report.manifest.get("num_cores", 1))
            display({key: report.manifest.get(key) for key in
                     ("hardware_target", "precision", "num_cores", "total_execution_time_sec")})
        ''')
        for title, code, inspection in sections:
            add_cell("markdown", "## " + title)
            if code.strip():
                add_cell("code", code)
            if inspection.strip():
                add_cell("code", inspection)
        add_cell("markdown", "## Optional PDF export\nUncomment to compile the report with pdflatex. Notebook inspection plots are separate from the PDF figures.")
        pdf_filename = "comparison_report.pdf" if comparison else (
            "precision_analysis_report.pdf" if precision else "mission_report.pdf")
        add_cell("code", f'# report.report_save_pdf({pdf_filename!r})')
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": sys.version.split()[0]},
            },
            "nbformat": 4, "nbformat_minor": 5,
        }
        (self._output_root / filename).write_text(json.dumps(notebook, indent=2) + "\n")

    def _generate_report_comparison(self, manifest_path):
        """Compare matching breakpoints and final products with the baseline run."""
        config = self.manifest.get("report_config", {})
        if not config.get("enabled", False):
            return
        with open(self._comparison_manifest_path) as stream:
            baseline = yaml.safe_load(stream)
        reference_breakpoints = {item["label"]: item for item in baseline.get("breakpoints", [])}
        current_breakpoints = {item["label"]: item for item in self.manifest.get("breakpoints", [])}
        pairs = [(item, current_breakpoints[label]) for label, item in reference_breakpoints.items()
                 if label in current_breakpoints]
        if "final_output" in baseline and "final_output" in self.manifest:
            pairs.append((baseline["final_output"], self.manifest["final_output"]))
        threshold = self.manifest.get("precision_analysis", {}).get("threshold_mean_abs_diff", 1e-7)
        metadata_code = f'''\
            reference = Report(report.output_root.parent / "run_manifest.yaml")
            report.section("Run Comparison Metadata")
            report.add_property("Reference directory", str(reference.output_root))
            report.add_property("Current run", {self.manifest["comparison"]["name"]!r})
            report.add_property("Reference precision", reference.manifest.get("precision"))
            report.add_property("Current precision", report.manifest.get("precision"))
            report.add_property("Flag threshold", {threshold!r})
            report.add_property("Reference-only breakpoints", {sorted(reference_breakpoints.keys() - current_breakpoints.keys())!r})
            report.add_property("Current-only breakpoints", {sorted(current_breakpoints.keys() - reference_breakpoints.keys())!r})
        '''
        sections = [("Run Comparison Metadata", metadata_code, 'display(reference.manifest)\ndisplay(report.manifest)')]
        for i, (original, current) in enumerate(pairs, 1):
            code = f'''\
                report.section({("Compare: " + current["label"])!r})
                product_{i}_reference = reference.load_bkp({original["label"]!r})
                product_{i}_current = report.load_bkp({current["label"]!r})
                report.plot_run_comparison(product_{i}_reference, product_{i}_current, threshold={threshold!r})
            '''
            inspection = (f'display(product_{i}_reference.dataset)\n'
                          f'display(product_{i}_current.dataset)')
            sections.append(("Compare: " + current["label"], code, inspection))
        if config.get("report_format", "py") == "ipynb":
            self._write_report_notebook(manifest_path, config, sections,
                                        filename="report_comparison.ipynb", comparison=True)
            return
        blocks = "\n\n".join(textwrap.indent(textwrap.dedent(code).strip(), "    ")
                              for _, code, _ in sections)
        script = f'''# AUTOMATICALLY GENERATED EOPL RUN COMPARISON
import argparse
from pathlib import Path
from eopl.report import Report

MANIFEST_PATH = Path(__file__).with_name({manifest_path.name!r})

def main():
    parser = argparse.ArgumentParser(description="Compare an EOPL run with its baseline")
    parser.add_argument("--author", "-a", default="Unknown Author")
    report = Report(MANIFEST_PATH)
    report.author(parser.parse_args().author)
    report.show_plots(False)

{blocks}

    report.report_save_pdf("comparison_report.pdf")

if __name__ == "__main__":
    main()
'''
        (self._output_root / "report_comparison.py").write_text(script)

    def _generate_report_script_precision(self, manifest_path: Path):
        """Generates the report_pa.py script for pixel-wise numerical analysis."""
        script_path = self._output_root / "report_pa.py"
        
        # We read the manifest from disk because self.manifest in the PA run 
        # doesn't contain the breakpoint history (we skipped appending to it).
        if not self.manifest.get("report_config", {}).get("enabled", False):
            return
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run the nominal pass first: missing {manifest_path}")

        with open(manifest_path, "r") as f:
            nominal_manifest = yaml.safe_load(f)

        pa_config = nominal_manifest.get("precision_analysis", {})
        threshold = pa_config.get("threshold_mean_abs_diff", 1e-7)
        target_precision = pa_config.get("comparison_precision", "float64")

        items_to_process = nominal_manifest.get("breakpoints", []).copy()
        if "final_output" in nominal_manifest:
            fo = nominal_manifest["final_output"].copy()
            fo["_is_final_output"] = True
            items_to_process.append(fo)

        bkp_blocks = []
        for i, bkp in enumerate(items_to_process, 1):
            label = bkp['label']
            is_final = bkp.get("_is_final_output", False)
            
            var_nom = f"product_{i}_nom"
            var_pa = f"product_{i}_pa"
            
            block_lines = []
            if is_final:
                block_lines.extend([
                    f"    # Final Output: {label!r}",
                    f'    report.section({("Precision Analysis Final Output: " + label)!r})'
                ])
            else:
                block_lines.extend([
                    f"    # Breakpoint {i}: {label!r}",
                    f'    report.section({("Precision Analysis: " + label)!r})'
                ])
                
            block_lines.extend([
                f'    {var_nom} = report.load_bkp({label!r})',
                f'    {var_pa} = report.load_bkp({(label + "_pa")!r})',
                f'    ',
                f'    # Execute pixel-wise comparison',
                f'    report.plot_precision_comparison({var_nom}, {var_pa}, threshold={threshold}, target_dtype="{target_precision}")',
                ""
            ])
            bkp_blocks.append("\n".join(block_lines))

        script_content = f"""# AUTOMATICALLY GENERATED EOPL PRECISION ANALYSIS REPORT
import eopl.report
import argparse
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("{manifest_path.name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate EOPL Precision Analysis Report")
    parser.add_argument("--author", "-a", type=str, default="Unknown Author", 
                        help="Name of the report author")
    args = parser.parse_args()
    return args.author

def main():
    report = eopl.report.Report(manifest=MANIFEST_PATH)
    author = parse_args()
    report.author(author)

    report.section("Precision Analysis Metadata")
    report.add_property("Nominal Precision", report.manifest.get("precision", "Unknown"))
    report.add_property("Comparison Target", "{target_precision}")
    report.add_property("Flag Threshold", {threshold})
    report.show_plots(False)

{chr(10).join(bkp_blocks)}

    report.report_save_pdf("precision_analysis_report.pdf")

if __name__ == "__main__":
    main()
"""
        report_config = nominal_manifest.get("report_config", {})
        if report_config.get("report_format", "py") == "ipynb":
            sections = [("Precision Analysis Metadata", f'''\
                report.section("Precision Analysis Metadata")
                report.add_property("Comparison Target", {target_precision!r})
                report.add_property("Flag Threshold", {threshold!r})
                display({{"comparison_precision": {target_precision!r}, "threshold": {threshold!r}}})
            ''', "")]
            for i, (item, code) in enumerate(zip(items_to_process, bkp_blocks), 1):
                title = ("Final Output: " if item.get("_is_final_output") else "Breakpoint: ") + item["label"]
                inspection = (f'display(product_{i}_nom.dataset)\n'
                              f'display(product_{i}_pa.dataset)\n'
                              f'# display(product_{i}_nom.dataset.attrs)\n'
                              f'# display(product_{i}_pa.dataset.attrs)')
                sections.append((title, code, inspection))
            self._write_report_notebook(manifest_path, report_config, sections,
                                        filename="report_pa.ipynb", precision=True)
            return
        with open(script_path, "w") as f:
            f.write(script_content)
