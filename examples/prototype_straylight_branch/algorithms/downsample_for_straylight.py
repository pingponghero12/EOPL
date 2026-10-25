"""Python, Dask, and C++ implementations of downsample_for_straylight."""
import ctypes
import numpy as np
import xarray as xr
from eopl import image_t
from eopl.algorithms import block_mean_downsample
from .native import kernel, double_array, DOUBLE_PTR, SIZE


# Python implementation
def downsample_for_straylight(image, factor, run_options):
    array = image.image()
    if factor < 1 or factor > min(array.sizes['ALT'], array.sizes['ACT']):
        raise ValueError('Downsample factor exceeds image size')
    low = block_mean_downsample(array, {'ALT': factor, 'ACT': factor})
    # Separate dimension names prevent accidental alignment with full-resolution data.
    return image_t(low.rename({'ALT': 'low_ALT', 'ACT': 'low_ACT'}), name='low_resolution_image')

downsample_for_straylight.input_access = ('read',)
downsample_for_straylight.cuda_support = False
downsample_for_straylight.dask_support = False


# Dask implementation
def downsample_for_straylight_dask(image, factor, run_options):
    array = image.image()
    if factor < 1 or factor > min(array.sizes['ALT'], array.sizes['ACT']):
        raise ValueError('Downsample factor exceeds image size')
    low = block_mean_downsample(array, {'ALT': factor, 'ACT': factor})
    # Separate dimension names prevent accidental alignment with full-resolution data.
    return image_t(low.rename({'ALT': 'low_ALT', 'ACT': 'low_ACT'}), name='low_resolution_image')

downsample_for_straylight_dask.input_access = ('read',)
downsample_for_straylight_dask.cuda_support = False
downsample_for_straylight_dask.dask_support = True


# C++ implementation
def downsample_for_straylight_cpp(image, factor, run_options):
    full = image.image()
    c, h, w = full.shape
    if factor < 1 or factor > min(h, w):
        raise ValueError('Downsample factor exceeds image size')
    output = np.empty((c, h//factor, w//factor), dtype=np.float64)
    kernel('downsample', [DOUBLE_PTR, DOUBLE_PTR, SIZE, SIZE, SIZE, SIZE],
           double_array(full.values), output, c, h, w, factor)
    # Reuse xarray's coarsened coordinate layout, without calculating image means.
    coords = {name: coord.coarsen({dim: factor for dim in coord.dims if dim in ('ALT', 'ACT')}, boundary='trim').mean()
              if any(dim in ('ALT', 'ACT') for dim in coord.dims) else coord
              for name, coord in full.coords.items()}
    low = xr.DataArray(output.astype(run_options.precision), dims=full.dims, coords=coords)
    return image_t(low.rename({'ALT': 'low_ALT', 'ACT': 'low_ACT'}), name='low_resolution_image')

downsample_for_straylight_cpp.input_access = ('read',)
downsample_for_straylight_cpp.cuda_support = False
downsample_for_straylight_cpp.dask_support = False
