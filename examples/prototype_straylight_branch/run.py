"""Runnable demonstration using the EOPL / EOPAL interface."""

from eopl.pipeline import Pipeline
from algorithms.downsample_for_straylight import (
    downsample_for_straylight,
    downsample_for_straylight_dask,
    downsample_for_straylight_cpp,
)
from algorithms.simulate_straylight import (
    simulate_straylight,
    simulate_straylight_dask,
    simulate_straylight_cpp,
)
from algorithms.resample_straylight import (
    resample_straylight,
    resample_straylight_dask,
    resample_straylight_cpp,
)
from algorithms.add_straylight import (
    add_straylight,
    add_straylight_dask,
    add_straylight_cpp,
)

# Select the Python, _dask, or _cpp functions in the operation calls below.
# For _dask image operations, set dask=True in read_input and save_bkp.

pl = Pipeline()
args = pl.parse_arguments()

# 1. INITIALIZATION AND CONFIGURATION PHASE
pl.add_local_config(args["local_config_path"])
pl.add_global_config(args["global_config_path"])
pl.set_run_options(use_cuda=False, num_cores=4, precision="float32")
# Change report_format to "ipynb" for a Jupyter notebook.
pl.generate_report(True, report_format="ipynb", print_all_bkp_data=True, plot_2d_indexes=(0,))
pl.generate_breakpoints(True)
pl.setup_precision_analysis(precision="float64", threshold_mean_abs_diff=1e-7)
image = pl.read_input(args["input_path"], dask=False)

# 2. OPERATION PHASE
low_resolution_image = pl.run_operation(
    data=image,
    func=downsample_for_straylight,
    args={"factor": 4},
)

# Demonstration: average each low-resolution pixel with its four neighbours
# using periodic edges, then take 2% of that signal as straylight.
straylight_low_resolution = pl.run_operation(
    data=low_resolution_image,
    func=simulate_straylight,
    args={"fraction": 0.02},
)

straylight_low_resolution = pl.save_bkp(
    straylight_low_resolution,
    "straylight_low_resolution",
    file_type="nc",
    dask=False,
)

del low_resolution_image

# Resample by repeating the nearest source cell; this is not linear interpolation.
straylight = pl.run_operation(
    data=(straylight_low_resolution, image),
    func=resample_straylight,
    args={},
)

straylight = pl.save_bkp(
    straylight,
    "straylight_full_resolution",
    file_type="nc",
    dask=False,
)

del straylight_low_resolution

# Consume both products; keep straylight for the final output.
image = pl.run_operation(
    data=(image, straylight),
    func=add_straylight,
    args={},
)

# 3. FINAL PHASE
pl.save_output((image, straylight), "straylight_output")
