"""Python, Dask, and C++ implementations of add_straylight."""
import ctypes
import numpy as np
from eopl import image_t
from .native import kernel, double_array, DOUBLE_PTR, SIZE


# Python implementation
def add_straylight(image, straylight, run_options):
    if image.image().dims != straylight.image().dims or image.image().shape != straylight.image().shape:
        raise ValueError("Image and straylight dimensions and shapes must match")
    return image_t(image.image() + straylight.image(), name=image.array_name)

add_straylight.input_access = ('read', 'read')
add_straylight.cuda_support = False
add_straylight.dask_support = False


# Dask implementation
def add_straylight_dask(image, straylight, run_options):
    if image.image().dims != straylight.image().dims or image.image().shape != straylight.image().shape:
        raise ValueError("Image and straylight dimensions and shapes must match")
    return image_t(image.image() + straylight.image(), name=image.array_name)

add_straylight_dask.input_access = ('read', 'read')
add_straylight_dask.cuda_support = False
add_straylight_dask.dask_support = True


# C++ implementation
def add_straylight_cpp(image, straylight, run_options):
    if image.image().dims != straylight.image().dims or image.image().shape != straylight.image().shape:
        raise ValueError("Image and straylight dimensions and shapes must match")
    full = image.image()
    output = np.empty(full.shape, dtype=np.float64)
    kernel('add', [DOUBLE_PTR, DOUBLE_PTR, DOUBLE_PTR, SIZE],
           double_array(full.values), double_array(straylight.image().values), output, full.size)
    return image_t(full.copy(data=output.astype(run_options.precision)), name=image.array_name)

add_straylight_cpp.input_access = ('read', 'read')
add_straylight_cpp.cuda_support = False
add_straylight_cpp.dask_support = False
