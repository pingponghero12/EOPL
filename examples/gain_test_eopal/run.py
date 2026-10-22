import numpy as np
from eopl.pipeline import Pipeline
from eopl import utils

# Python, Dask, and C++ implementations
from eopl.algorithms.gain import apply_gain_1d, apply_gain_1d_dask, apply_gain_1d_cpp

# Compare two runs from this example directory:
# 1. python run.py
# 2. Uncomment the x2 operation below.
# 3. python run.py --dir doubled
# 4. jupyter lab outputs/doubled/report_comparison.ipynb
# With report_format="py": python outputs/doubled/report_comparison.py
# Matching breakpoints and the final output are compared with the baseline.

pl = Pipeline()
args = pl.parse_arguments()

### 1. INITIALIZATION AND CONFIGURATION PHASE
pl.add_local_config(args.get("local_config_path"))
pl.add_global_config(args.get("global_config_path"))
# In reality you would use pipeline.global_config["CCDBPath"], hardcoded here for clarity
pl.add_ccdb("ccdb/") 

pl.set_run_options(use_cuda=False, num_cores=4, precision="float32")
pl.set_image_type("float32")

# We use (0,) to slice only the first dimension ('channels'), leaving ALT and ACT for 2D plotting
# Change report_format to "ipynb" for a Jupyter notebook.
pl.generate_report(True, report_format="ipynb", print_all_bkp_data=True, plot_2d_images=True, plot_2d_indexes=(0,))
pl.generate_breakpoints(True)
pl.setup_precision_analysis(precision="float64", threshold_mean_abs_diff=1e-7)

data = pl.read_input(args.get("input_path"), dask=False)

### 2. OPERATION PHASE
# Get ACT size safely via xarray sizes property
act_size = data.image().sizes["ACT"]

pl.log.info(f"pipeline: data: load: {data.image()}")

# Load from CCDB and resize by linear interpolation
gain_array_1d = utils.lin_interp_resize(
    pl.ccdb.get("gain_array_1d"),
    size=act_size
)

pl.add_variable(
    "gain_array_1d",
    gain_array_1d,
    pinned=True,
    dims=("gain_array_1d_sampling",),
)

data = pl.run_operation(
    data=data,
    func=apply_gain_1d,
    args={"gain_array_1d": gain_array_1d, "apply_dimension": "ACT"}
)

pl.log.info(f"pipeline: data: after operation 1: {data.image()}")
data = pl.save_bkp(data, "after_gain_application", file_type="nc", dask=False)

pl.log.info(f"pipeline: data: after breakpoint 1: {data.image()}")

data = pl.run_operation(
    data=data,
    func=apply_gain_1d_cpp,
    args={"gain_array_1d": gain_array_1d, "apply_dimension": "ACT"}
)

pl.log.info(f"pipeline: data: after operation 2: {data.image()}")

# Uncomment to double the image in the second run.
# data = pl.run_operation(
#     data=data,
#     func=apply_gain_1d,
#     args={"gain_array_1d": np.full(act_size, 2), "apply_dimension": "ACT"},
# )

data = pl.save_bkp(data, "after_gain_application_2", file_type="nc", dask=False)
pl.log.info(f"pipeline: data: after breakpoint 2: {data.image()}")

### 3. FINAL PHASE
pl.save_output(data, "final_output", file_type="nc")
