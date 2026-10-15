"""Gain multiplication in Python, Dask, and the EOPAL C++ core."""
from typing import Any, Union
import numpy as np
import xarray as xr
from eopal.data import image_t
from eopal.ops.gain import apply_gain_1d as eopal_gain
from eopal.ops.gain import apply_gain_1d_cpp as eopal_gain_cpp

_apply_gain_1d_contract = eopal_gain.shape_contract


# Python implementation
def apply_gain_1d(
    data: image_t,
    gain_array_1d: Union[np.ndarray, xr.DataArray],
    run_options: Any,
    apply_dimension: str = "ACT",
) -> image_t:
    _apply_gain_1d_contract(
        dict(data.image().sizes),
        {"gain_array_1d": gain_array_1d, "apply_dimension": apply_dimension},
    )
    current_image = data.image()

    # Check if we need to wrap a raw numpy array or if it's already an xarray object
    if isinstance(gain_array_1d, xr.DataArray):
        gain_da = gain_array_1d.astype(current_image.dtype)
    else:
        # Wrap the 1D numpy array to explicitly define its broadcasting dimension
        gain_da = xr.DataArray(
            np.asarray(gain_array_1d, dtype=current_image.dtype),
            dims=[apply_dimension],
        )

    # Xarray automatically handles the broadcasting and alignment by dimension name
    result = current_image * gain_da
    if result.dtype != current_image.dtype:
        result = result.astype(current_image.dtype)
    data.replace_image(result.transpose(*current_image.dims))
    return data

apply_gain_1d.shape_contract = _apply_gain_1d_contract
apply_gain_1d.input_access = ("readwrite",)
apply_gain_1d.cuda_support = False
apply_gain_1d.dask_support = False


# Dask implementation
def apply_gain_1d_dask(
    data: image_t,
    gain_array_1d: Union[np.ndarray, xr.DataArray],
    run_options: Any,
    apply_dimension: str = "ACT",
) -> image_t:
    _apply_gain_1d_contract(
        dict(data.image().sizes),
        {"gain_array_1d": gain_array_1d, "apply_dimension": apply_dimension},
    )
    current_image = data.image()

    # Check if we need to wrap a raw numpy array or if it's already an xarray object
    if isinstance(gain_array_1d, xr.DataArray):
        gain_da = gain_array_1d.astype(current_image.dtype)
    else:
        # Wrap the 1D numpy array to explicitly define its broadcasting dimension
        gain_da = xr.DataArray(
            np.asarray(gain_array_1d, dtype=current_image.dtype),
            dims=[apply_dimension],
        )

    # Xarray automatically handles the broadcasting and alignment by dimension name
    result = current_image * gain_da
    if result.dtype != current_image.dtype:
        result = result.astype(current_image.dtype)
    data.replace_image(result.transpose(*current_image.dims))
    return data

apply_gain_1d_dask.shape_contract = _apply_gain_1d_contract
apply_gain_1d_dask.input_access = ("readwrite",)
apply_gain_1d_dask.cuda_support = False
apply_gain_1d_dask.dask_support = True


# C++ implementation
def apply_gain_1d_cpp(data, gain_array_1d, run_options, apply_dimension="ACT"):
    return eopal_gain_cpp(data, gain_array_1d, run_options, apply_dimension)


apply_gain_1d_cpp.shape_contract = _apply_gain_1d_contract
apply_gain_1d_cpp.input_access = ("readwrite",)
apply_gain_1d_cpp.cuda_support = True
apply_gain_1d_cpp.cuda_dtypes = ("float32",)
apply_gain_1d_cpp.dask_support = False
