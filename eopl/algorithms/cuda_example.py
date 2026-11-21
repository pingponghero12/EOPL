"""Example EOPL operation that launches a kernel from a standalone CUDA file."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..data import image_t, require_cupy


@lru_cache(maxsize=1)
def _scale_kernel():
    cupy = require_cupy()
    source = Path(__file__).with_name("cuda") / "scale_image.cu"
    module = cupy.RawModule(code=source.read_text(encoding="utf-8"))
    return module.get_function("scale_image")


def scale_image(data, factor, run_options):
    """Multiply a float32 image by a scalar, preserving its coordinates and units."""
    if not run_options.use_cuda:
        image = data.image()
        return image_t(image.copy(deep=False, data=image.data * np.float32(factor)))

    cupy = require_cupy()
    data.to_device()
    image = data.image()
    if image.dtype != np.float32:
        raise TypeError("The scale_image CUDA kernel requires float32 pixels.")

    # The kernel indexes a flat C-contiguous array. CuPy only copies if needed.
    pixels = cupy.ascontiguousarray(image.data)
    result = cupy.empty_like(pixels)
    if pixels.size:
        threads = 256
        blocks = (pixels.size + threads - 1) // threads
        _scale_kernel()(
            (blocks,), (threads,), (pixels, result, np.uint64(pixels.size), np.float32(factor))
        )

    return image_t(image.copy(deep=False, data=result))


scale_image.input_access = ("read",)
scale_image.dask_support = True
scale_image.cuda_support = True
scale_image.cuda_dtypes = ("float32",)
