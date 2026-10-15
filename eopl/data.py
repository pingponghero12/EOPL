"""The lightweight image object shared by EOPL operations."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import xarray as xr


class image_t:
    """One image and its optional device-side cache.

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
            values = np.asarray(image)
            if dims is None:
                dims = tuple(f"dim_{index}" for index in range(values.ndim))
            if len(dims) != values.ndim:
                raise ValueError(f"Expected {values.ndim} dimension names, received {len(dims)}.")
            selected_name = selected_name or "default_image"
            array = xr.DataArray(values, dims=tuple(dims), name=selected_name)

        self.data = array.rename(selected_name)
        self.array_name = str(selected_name)
        self._device_array = None
        self._host_dirty = False
        self._device_dirty = True
        self._device_shape = None
        self._device_dtype = None
        self._device_nbytes = 0

    def image(self) -> xr.DataArray:
        return self.data

    @property
    def is_dask(self) -> bool:
        return self.data.chunks is not None

    def replace_image(self, image: xr.DataArray) -> None:
        if not isinstance(image, xr.DataArray):
            raise TypeError("replace_image expects an xarray.DataArray.")
        self.data = image.rename(self.array_name)
        self.mark_host_modified()

    def change_array_name(self, new_name: str) -> None:
        self.array_name = str(new_name)
        self.data = self.data.rename(self.array_name)

    def mark_host_modified(self) -> None:
        self._host_dirty = False
        self._device_dirty = True

    def mark_device_modified(self) -> None:
        if self._device_array is None:
            raise RuntimeError("Cannot mark device data modified without a device allocation.")
        self._host_dirty = True
        self._device_dirty = False

    def mark_synchronized(self) -> None:
        self._host_dirty = False
        self._device_dirty = self._device_array is None

    def clear_device_state(self) -> None:
        self._device_array = None
        self._device_shape = None
        self._device_dtype = None
        self._device_nbytes = 0
        self._host_dirty = False
        self._device_dirty = True

    def release_device(self) -> None:
        pointer = self._device_array
        if pointer is None:
            return
        self.clear_device_state()
        from eopal import eopal_cpp_core
        eopal_cpp_core.free_vram(pointer)

    def close(self) -> None:
        self.release_device()

    def __del__(self):
        try:
            self.release_device()
        except Exception:
            pass


p_data = image_t

__all__ = ["image_t", "p_data"]
