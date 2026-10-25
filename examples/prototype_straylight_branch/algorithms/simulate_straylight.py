"""Python, Dask, and C++ implementations of simulate_straylight."""
import ctypes
import numpy as np
from eopl import image_t
from .native import kernel, double_array, DOUBLE_PTR, SIZE


# Python implementation
def simulate_straylight(low_resolution_image, fraction, run_options):
    array = low_resolution_image.image()
    blurred = (array + array.roll(low_ALT=1) + array.roll(low_ALT=-1)
               + array.roll(low_ACT=1) + array.roll(low_ACT=-1)) / 5
    return image_t(fraction * blurred, name='straylight_low_resolution')

simulate_straylight.input_access = ('read',)
simulate_straylight.cuda_support = False
simulate_straylight.dask_support = False


# Dask implementation
def simulate_straylight_dask(low_resolution_image, fraction, run_options):
    array = low_resolution_image.image()
    blurred = (array + array.roll(low_ALT=1) + array.roll(low_ALT=-1)
               + array.roll(low_ACT=1) + array.roll(low_ACT=-1)) / 5
    return image_t(fraction * blurred, name='straylight_low_resolution')

simulate_straylight_dask.input_access = ('read',)
simulate_straylight_dask.cuda_support = False
simulate_straylight_dask.dask_support = True


# C++ implementation
def simulate_straylight_cpp(low_resolution_image, fraction, run_options):
    low = low_resolution_image.image()
    output = np.empty(low.shape, dtype=np.float64)
    kernel('scatter', [DOUBLE_PTR, DOUBLE_PTR, SIZE, SIZE, SIZE, ctypes.c_double],
           double_array(low.values), output, *low.shape, fraction)
    return image_t(low.copy(data=output.astype(run_options.precision)), name='straylight_low_resolution')

simulate_straylight_cpp.input_access = ('read',)
simulate_straylight_cpp.cuda_support = False
simulate_straylight_cpp.dask_support = False
