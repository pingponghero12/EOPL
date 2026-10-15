"""Generate this example's deterministic input image."""
from pathlib import Path
import numpy as np
import xarray as xr

input_directory = Path(__file__).resolve().parent / "input"
input_directory.mkdir(exist_ok=True)

channels, along_track, across_track = np.indices((2, 32, 48))
image = 1 + 0.2 * channels + 0.01 * along_track + 0.1 * np.sin(across_track / 5)
dataset = xr.Dataset({"default_image": (("Channels", "ALT", "ACT"), image)})
dataset.to_netcdf(input_directory / "input_image.nc")
