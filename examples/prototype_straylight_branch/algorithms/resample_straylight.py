"""Python, Dask, and C++ implementations of resample_straylight."""
import ctypes
import numpy as np
import xarray as xr
from eopal.data import image_t
from eopal.ops import block_mean_downsample
from .native import kernel, double_array, DOUBLE_PTR, SIZE


# Python implementation
def resample_straylight(straylight_low_resolution, target, run_options):
    low, full = straylight_low_resolution.image(), target.image()
    # Map target cell indices into source cells, including trimmed edge blocks.
    iy = np.arange(full.sizes['ALT']) * low.sizes['low_ALT'] // full.sizes['ALT']
    ix = np.arange(full.sizes['ACT']) * low.sizes['low_ACT'] // full.sizes['ACT']
    result = low.isel(low_ALT=xr.DataArray(iy, dims='ALT'), low_ACT=xr.DataArray(ix, dims='ACT'))
    result = result.drop_vars([name for name in ('low_ALT', 'low_ACT') if name in result.coords])
    result = result.assign_coords(full.coords)
    return image_t(result, name='straylight')

resample_straylight.input_access = ('read', 'read')
resample_straylight.cuda_support = False
resample_straylight.dask_support = False


# Dask implementation
def resample_straylight_dask(straylight_low_resolution, target, run_options):
    low, full = straylight_low_resolution.image(), target.image()
    # Map target cell indices into source cells, including trimmed edge blocks.
    iy = np.arange(full.sizes['ALT']) * low.sizes['low_ALT'] // full.sizes['ALT']
    ix = np.arange(full.sizes['ACT']) * low.sizes['low_ACT'] // full.sizes['ACT']
    result = low.isel(low_ALT=xr.DataArray(iy, dims='ALT'), low_ACT=xr.DataArray(ix, dims='ACT'))
    result = result.drop_vars([name for name in ('low_ALT', 'low_ACT') if name in result.coords])
    result = result.assign_coords(full.coords)
    return image_t(result, name='straylight')

resample_straylight_dask.input_access = ('read', 'read')
resample_straylight_dask.cuda_support = False
resample_straylight_dask.dask_support = True


# C++ implementation
def resample_straylight_cpp(straylight_low_resolution, target, run_options):
    full, low = target.image(), straylight_low_resolution.image()
    output = np.empty(full.shape, dtype=np.float64)
    kernel('resample', [DOUBLE_PTR, DOUBLE_PTR, SIZE, SIZE, SIZE, SIZE, SIZE],
           double_array(low.values), output, *full.shape, *low.shape[-2:])
    return image_t(full.copy(data=output.astype(run_options.precision)), name='straylight')

resample_straylight_cpp.input_access = ('read', 'read')
resample_straylight_cpp.cuda_support = False
resample_straylight_cpp.dask_support = False
