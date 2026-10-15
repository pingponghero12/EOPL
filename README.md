# EOPL

Earth Observation Processing Library orchestration for configurable scientific
pipelines, NetCDF/Zarr products, breakpoints, precision analysis, Dask, and
CPU/CUDA execution through EOPAL operations.

The simulator uses Band Sequential (BSQ) layout because its most demanding
operations are expected to be the high-resolution convolutions and
deconvolutions performed by early SGM and IOM processing stages.

`image_t` contains one `xarray.DataArray` plus the synchronization state needed
to cache that image on a GPU. Pipeline variables are kept separately by the
`Pipeline` and are assembled into an `xarray.Dataset` only at a checkpoint or
final save.

Operations use ordinary Python arguments and return values:

```python
image, topology, samples, score = pipeline.run_operation(
    data=image,
    func=generate_topology,
    args={},
)

image = pipeline.run_operation(
    data=(image, topology),
    func=apply_surface_model,
    args={},
)
```

An operation declares how it uses each image once, beside its implementation:

```python
apply_surface_model.input_access = ("readwrite", "read")
```

EOPL uses this metadata and each image's freshness state to transfer only stale
host/device copies. A Dask-backed input makes that operation CPU/Dask-only; it
does not turn off the pipeline's CUDA preference for later operations.

Image-free modules such as geometry use `run_function` and pipeline variables:

```python
time, position = pipeline.run_function(func=simulate_geometry, args={})
pipeline.add_variable("time", time, dims=("time",), pinned=True)
pipeline.add_variable("position", position, dims=("time", "xyz"), pinned=True)
pipeline.save_output("geometry")
```

The four examples in `examples/` keep EOPL interface calls in `run.py` and
Python, Dask, and C++ implementations together in each `algorithms/` directory.

## Installation for development

From the directory containing both `EOPL` and `EOPAL`, create a clean virtual
environment and install both projects in editable mode:

```bash
cd /path/to/E2ES
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./EOPAL[test]"
python -m pip install -e "./EOPL[test]"
```

The native extensions are built automatically. If CUDA and `nvcc` are found,
CUDA support is compiled; otherwise a CPU-only extension is built.

```bash
python -m pip uninstall -y eopl eopal
python -m pip install -e "./EOPAL[test]"
python -m pip install -e "./EOPL[test]"
```

## Tests and coverage

Run the suites independently from the `E2ES` directory:

```bash
python -m pytest EOPAL/tests
python -m pytest EOPL/tests
```

Run both suites together:

```bash
python -m pytest EOPL/tests EOPAL/tests --import-mode=importlib
```

Generate terminal and HTML coverage reports for both packages:

```bash
python -m pytest EOPL/tests EOPAL/tests --import-mode=importlib \
  --cov=eopl --cov=eopal --cov-report=term-missing \
  --cov-report=html:coverage_html
```

Open `coverage_html/index.html` to browse the detailed report. GPU tests are not
run without hardware; CI compiles the CUDA target as a build-only check. Report
generation and precision-analysis code are intentionally not covered yet.
