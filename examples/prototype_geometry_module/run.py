"""Runnable demonstration using the EOPL / EOPAL interface."""

from eopl.pipeline import Pipeline
from algorithms.simulate_geometry import simulate_geometry

pl = Pipeline()
args = pl.parse_arguments()

# 1. INITIALIZATION AND CONFIGURATION PHASE
pl.add_local_config(args["local_config_path"])
pl.add_global_config(args["global_config_path"])
pl.set_run_options(use_cuda=False, num_cores=4, precision="float64")
# Change report_format to "ipynb" for a Jupyter notebook.
pl.generate_report(True, report_format="py", print_all_bkp_data=True, plot_2d_indexes=(0,))
pl.generate_breakpoints(True)
pl.setup_precision_analysis(precision="float64", threshold_mean_abs_diff=1e-7)

# 2. FUNCTION PHASE
time, positions, velocities, ground_contact = pl.run_function(
    func=simulate_geometry,
    args={"duration_s": 3600, "time_step_s": 10},
)

pl.add_variable("time", time, dims=("time",), pinned=True)
pl.add_variable("spacecraft_position", positions, dims=("time", "xyz"), pinned=True)
pl.add_variable("spacecraft_velocity", velocities, dims=("time", "xyz"), pinned=True)
pl.add_variable("ground_contact", ground_contact, dims=("time",), pinned=True)
pl.add_variable(
    "mission_identifier", pl.global_config["MissionID"], pinned=True
)

# The pipeline builds an internal xarray.Dataset only when saving/checkpointing.
pl.save_bkp("geometry_after_propagation", file_type="nc")

# 3. FINAL PHASE
pl.save_output("geometry_output", file_type="nc")
