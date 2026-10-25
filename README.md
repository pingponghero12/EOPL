# EOPL

Earth Observation Processing Library: algorithms and orchestration for scientific pipelines,
NetCDF/Zarr products, checkpoints, and editable Python or Jupyter reports.

Version 0.2.0 integrates the algorithms previously provided by EOPAL. The CPU workflows are
covered by automated tests. CUDA code is retained, but GPU execution has not
been validated for this release.

## Installation

Python 3.10 or newer is required. From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,notebook]'
```

The source build requires a C++17 compiler and CMake 3.22 or newer. Build
isolation installs its Python build dependencies. This release uses Xarray
below 2026.4 with Zarr-Python 2.x; newer Xarray releases require Zarr-Python 3.
CMake detects CUDA when available; to explicitly build the CPU version:

```bash
python -m pip install -e '.[test,notebook]' \
  -Ccmake.define.CMAKE_CUDA_COMPILER=NOTFOUND
```

The algorithms and native gain core are included; no separate EOPAL installation
is required. A notebook frontend
must use the environment where EOPL is installed. PDF export additionally
requires `pdflatex`; without it, the LaTeX source and figures are still saved.

## Algorithms

Reusable operations live in `eopl.algorithms`:

```python
from eopl.algorithms.gain import apply_gain_1d, apply_gain_1d_dask, apply_gain_1d_cpp
from eopl.algorithms import block_mean_downsample, gaussian_psf, interpolate_spectra
```

The package also includes conservative spline upsampling, spectral-response
convolution, simplex projection, projected and nonnegative linear unmixing,
error metrics and spectral angle. Array operations accept NumPy or Xarray
where documented; gain operates on EOPL's `image_t`. Functions declare their
Dask and CUDA capabilities. The Python and Dask gain variants share Xarray's
broadcasting implementation; the C++ variant uses the packaged native core.
Existing `eopl.ops.gain` imports continue to work.

## Operation interface

An image is an `image_t` containing one `xarray.DataArray`. The usual image
layout is `(Channels, ALT, ACT)`, with ACT contiguous. Operations consume one
image or a tuple and return ordinary Python values:

```python
image, topology = pipeline.run_operation(
    data=image,
    func=derive_topology,
    args={},
)
image = pipeline.run_operation(
    data=(image, topology),
    func=apply_surface_model,
    args={},
)
pipeline.save_output((image, topology), 'final_output')
```

Declare `input_access`, `dask_support`, and `cuda_support` beside an operation:

```python
apply_surface_model.input_access = ('readwrite', 'read')
apply_surface_model.dask_support = True
apply_surface_model.cuda_support = False
```

Use `run_function` for image-free computations. Store their outputs with
`add_variable`; variables are combined with images into a Dataset when saving.
`pinned=True` keeps a variable available after checkpoints clean temporary
variables. `start_bkp(label)` resumes from a previously saved checkpoint.

## Examples

Each example is self-contained. Run from its own directory:

```bash
cd examples/gain_test_eopal
python run.py
```

The three examples are:

- `gain_test_eopal`: Python and packaged C++ gain operations, CCDB interpolation,
  checkpoints, and a commented multiplication-by-two comparison experiment.
- `prototype_geometry_module`: a Python circular-orbit demonstration using
  `run_function` and pipeline-owned variables, without an image input.
- `prototype_straylight_branch`: downsampling, a simple straylight model,
  resampling, and saving both the image and straylight products. Run
  `python setup_test_data.py` before its first execution.

These algorithms demonstrate interfaces rather than validated mission physics.
Reusable algorithms are in `eopl.algorithms`; example-specific simulation
models keep their Python, Dask, and C++ variants under the example's `algorithms/`.
Geometry deliberately has only Python. For Dask, select the `_dask` functions
and set `dask=True` on input and checkpoint calls. The local straylight C++
example compiles its kernels with `c++` on Linux; EOPL provides the gain core.

Scripts also accept the four ICD paths:

```bash
python run.py global.xml local.xml input/input_image.nc outputs
```

Paths are relative to the calling directory. The example config files and
CCDBs remain in their respective example directories.

## Reports and run comparison

Choose report sections and format together in `run.py`:

```python
from eopl import default_report_params

params = default_report_params()  # Final output only, with 2D plots.
params['report_format'] = 'ipynb'  # 'py' or 'ipynb'
params['generate_metadata'] = True
# params['generate_ccdb'] = True
# params['generate_input'] = True
# params['generate_breakpoints'] = True
# params['generate_custom_analysis'] = True
# params['generate_final'] = False
pl.generate_report(True, params)
```

`report_format='py'` generates an editable `report.py`; `'ipynb'` generates
`report.ipynb`. With no parameters, `generate_report(True)` includes only final
output. `ReportParams` defines the dictionary fields; omitted fields use the
defaults. Each factory call returns an independent dictionary. Plotting options
such as `plot_1d_images`, `plot_2d_images`, their `*_indexes`, and
`print_all_bkp_data` also belong in this dictionary.

The same section flags select content for named-run and precision comparison
reports; by default they compare only final products. Disabling report sections
does not disable saving inputs, breakpoints or products. Custom analysis controls
both pipeline-injected reporting and the editable custom-analysis example.
Calls using separate report keyword arguments should be migrated to the dictionary.
Open or execute the selected artifact:

```bash
jupyter lab outputs/report.ipynb
python outputs/report.py --author 'Your Name'
```

`report.load_input()` and `report.load_bkp(label)` return ordinary
`xarray.Dataset` objects. Inspect `product_1`, access variables with
`product_1['default_image']`, and read scientific metadata with `.attrs`.
To plot an image, slice extra dimensions first, for example
`product_1['default_image'].isel(Channels=0).plot()`.
Plotting and comparison helpers accept these Datasets directly. Run information
remains in `report.manifest`; use `report.load_bkp(label, with_metadata=True)`
only when you need the optional wrapper with `.label`, `.metadata` and `.dataset`.
Older generated reports using `.dataset` should be regenerated, or their loader
calls updated with `with_metadata=True`.
Optional PDF export writes under `outputs/pdf/`. Source input and CCDB paths
are recorded in the manifest; keep those files available when running reports.

For a same-precision study comparison, first run the baseline without `--dir`,
then change the operation or configuration and choose a named output directory:

```bash
python run.py
# Change an operation, such as the commented x2 block in the gain example.
python run.py --dir doubled
jupyter lab outputs/doubled/report_comparison.ipynb
```

The named run preserves the baseline and writes its own products, ordinary
report, and comparison report. Comparisons match checkpoint labels and compare
final products. The report format follows `generate_report`; with Python
output, run `python outputs/doubled/report_comparison.py`. Keep the baseline
beside its named runs. Reusing an existing name replaces that named run.

For precision analysis, enable `setup_precision_analysis` in the pipeline and
replay the baseline at float64:

```bash
python run.py
python run.py --precision-analysis
jupyter lab outputs/report_pa.ipynb
```

Use the same input, checkpoint labels, and output format for both passes.
The replay writes `*_pa` products and a precision report, preserving the nominal
manifest. `--dir` and `--precision-analysis` are separate modes. Dask operation
timings measure graph construction; checkpoint and total timings include
computation and I/O.

## Validation

```bash
python -m pytest
python -m pytest --cov=eopl --cov-report=term-missing --cov-report=html:coverage_html
```

CI builds and installs the CPU distribution before running the tests on
Python 3.10–3.12. It also checks source/wheel packaging and notebook structure.
GPU transfer tests use mocked backends; the separate CUDA build job is a
compilation check, not a hardware execution test.

## License and citation

EOPL is licensed under [Apache-2.0](LICENSE). Copyright and attribution
information is provided in [NOTICE](NOTICE).

If you use EOPL in research, please cite the software using
[CITATION.cff](CITATION.cff), also available through GitHub's **Cite this
repository** button. Academic citation is requested, not required by the
Apache-2.0 license. Third-party dependencies retain their own licenses.

## Roadmap

- 0.2.0: Integrated processing algorithms and simpler, configurable reports.
- 0.3.0: GPU execution using CuPy, with reliable memory transfers and
  synchronization, validated on GPU hardware.
- 0.4.0: Additional algorithm implementations with reference results and
  practical examples.
- Later: Further algorithms and potential Dask GPU execution using
  CuPy-backed arrays.
