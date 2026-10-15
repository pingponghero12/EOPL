"""Python demonstration of a circular equatorial orbit."""
import numpy as np

RADIUS_M = 7_000_000.0
MU = 3.986004418e14


# Python implementation
def simulate_geometry(duration_s, time_step_s, global_config, rng, run_options):
    time = np.arange(0, duration_s, time_step_s, dtype=run_options.precision)
    omega = np.sqrt(MU / RADIUS_M**3)
    phase = omega * time
    zero = np.zeros_like(time)
    positions = RADIUS_M * np.stack((np.cos(phase), np.sin(phase), zero), axis=1)
    velocities = RADIUS_M * omega * np.stack((-np.sin(phase), np.cos(phase), zero), axis=1)
    contact = np.cos(phase) > 0.5
    return time, positions.astype(run_options.precision), velocities.astype(run_options.precision), contact

simulate_geometry.input_access = ()
simulate_geometry.cuda_support = False
simulate_geometry.dask_support = False
