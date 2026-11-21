"""The lightweight image object shared by EOPL operations."""

from __future__ import annotations

from typing import Optional, Sequence
import sys

import numpy as np
import xarray as xr


def is_cupy_array(array) -> bool:
    """Identify a CuPy array without importing CuPy on the CPU path."""
    cupy = sys.modules.get("cupy")
    return cupy is not None and isinstance(array, cupy.ndarray)


def require_cupy():
    """Import the optional CUDA backend when it is first requested."""
    try:
        import cupy
    except ImportError as exc:
        raise RuntimeError(
            "CUDA execution requires CuPy. Install the CuPy wheel for your CUDA version."
        ) from exc
    return cupy


class image_t:
    """One Xarray image backed by a NumPy, Dask or CuPy array.

    Auxiliary scientific variables deliberately do not live here. They belong
    to the Pipeline, which only builds a Dataset when data is serialized.
    """

    def __init__(self, image, dims: Optional[Sequence[str]] = None,
                 name: Optional[str] = None, array_name: Optional[str] = None):
        selected_name = name or array_name
        if isinstance(image, xr.Dataset):
            selected_name = selected_name or image.attrs.get("_eopl_array_name")
            if selected_name is None:
                if not image.data_vars:
                    raise ValueError("Cannot create image_t from an empty Dataset.")
                selected_name = next(iter(image.data_vars))
            if selected_name not in image:
                raise KeyError(f"Image variable '{selected_name}' is not present in the Dataset.")
            array = image[selected_name]
        elif isinstance(image, xr.DataArray):
            selected_name = selected_name or image.name or "default_image"
            array = image
        else:
            values = image if is_cupy_array(image) else np.asarray(image)
            if dims is None:
                dims = tuple(f"dim_{index}" for index in range(values.ndim))
            if len(dims) != values.ndim:
                raise ValueError(f"Expected {values.ndim} dimension names, received {len(dims)}.")
            selected_name = selected_name or "default_image"
            array = xr.DataArray(values, dims=tuple(dims), name=selected_name)

        self.data = array.rename(selected_name)
        self.array_name = str(selected_name)

    def image(self) -> xr.DataArray:
        return self.data

    @property
    def is_dask(self) -> bool:
        return self.data.chunks is not None

    def replace_image(self, image: xr.DataArray) -> None:
        if not isinstance(image, xr.DataArray):
            raise TypeError("replace_image expects an xarray.DataArray.")
        self.data = image.rename(self.array_name)

    def change_array_name(self, new_name: str) -> None:
        self.array_name = str(new_name)
        self.data = self.data.rename(self.array_name)

    @property
    def is_cuda(self) -> bool:
        # Access the backing object without loading a lazily opened NetCDF array.
        return is_cupy_array(self.data.variable._data)

    def to_device(self) -> None:
        """Move pixels to CuPy, keeping coordinate metadata in Xarray."""
        if self.is_dask:
            raise ValueError("Compute the Dask image before moving it to CUDA.")
        if not self.is_cuda:
            cupy = require_cupy()
            self.data = self.data.copy(deep=False, data=cupy.asarray(self.data.data))

    def host_image(self) -> xr.DataArray:
        """Return a host image without replacing a live CuPy array."""
        if self.is_cuda:
            return self.data.copy(deep=False, data=require_cupy().asnumpy(self.data.data))
        return self.data

    def release_device(self) -> None:
        """Move pixels back to the host and release the image's GPU reference."""
        self.data = self.host_image()

    def close(self) -> None:
        self.release_device()


p_data = image_t

__all__ = ["image_t", "p_data"]
