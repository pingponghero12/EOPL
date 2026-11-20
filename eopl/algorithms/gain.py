import numpy as np
import xarray as xr
from typing import Any, Union
from ..data import image_t, require_cupy
from ..exceptions import ShapeError
from ..log import Log

def _apply_gain_1d_contract(dims: dict, args: dict):
    """Shape contract validator for apply_gain_1d."""
    apply_dim = args.get("apply_dimension", "ACT")
    if apply_dim not in dims:
        raise ShapeError(f"apply_gain_1d: Image does not have dimension '{apply_dim}'.")

    dim_size = dims[apply_dim]
    gain = args["gain_array_1d"]
    if np.ndim(gain) != 1:
        raise ShapeError(
            f"apply_gain_1d: gain_array_1d must be 1-D, got {np.ndim(gain)}-D."
        )
    if isinstance(gain, xr.DataArray) and gain.dims != (apply_dim,):
        raise ShapeError(
            f"apply_gain_1d: DataArray gain dimension must be '{apply_dim}', "
            f"got {gain.dims}."
        )
    gain_size = gain.shape[0]

    if gain_size != dim_size:
        raise ShapeError(
            f"apply_gain_1d: gain_array_1d has length {gain_size} but "
            f"dimension '{apply_dim}' has size {dim_size}."
        )


def apply_gain_1d(data: image_t, gain_array_1d: Union[np.ndarray, xr.DataArray], run_options: Any, apply_dimension: str = "ACT") -> image_t:
    """
    Apply a 1D per-pixel radiometric gain correction along a specified dimension.

    Multiplies the main image array element-wise by a 1D gain vector. If the input
    is a raw numpy array, it is broadcast along the specified apply_dimension.

    Parameters
    ----------
    data : image_t
        The current pipeline data object. Modified in place.
    gain_array_1d : Union[np.ndarray, xr.DataArray]
        1D array of gain coefficients. If an xarray.DataArray is provided, its
        existing dimensions are used for alignment.
    run_options : RunOptions
        Centralized execution settings passed from the pipeline.
    apply_dimension : str, optional
        Dimension along which to broadcast the gain vector if a raw numpy array
        is provided. Default: 'ACT'.

    Returns
    -------
    image_t
        The modified data object.
    """
    if run_options.use_cuda and not data.is_dask:
        return _apply_gain_1d_cupy(data, gain_array_1d, apply_dimension)

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

# Attach the contract and flags to the function
apply_gain_1d.eopl_shape_contract = _apply_gain_1d_contract
apply_gain_1d.cuda_support = True
apply_gain_1d.cuda_dtypes = ("float32", "float64", "uint32")
apply_gain_1d.dask_support = True
apply_gain_1d.input_access = ("readwrite",)
apply_gain_1d.shape_contract = _apply_gain_1d_contract


def apply_gain_1d_dask(data: image_t, gain_array_1d: Union[np.ndarray, xr.DataArray], run_options: Any, apply_dimension: str = "ACT") -> image_t:
    """Apply gain lazily using the same Xarray operations on Dask-backed data."""
    return apply_gain_1d(data, gain_array_1d, run_options, apply_dimension)


apply_gain_1d_dask.shape_contract = _apply_gain_1d_contract
apply_gain_1d_dask.eopl_shape_contract = _apply_gain_1d_contract
apply_gain_1d_dask.input_access = ("readwrite",)
apply_gain_1d_dask.cuda_support = False
apply_gain_1d_dask.dask_support = True


# Import the compiled Pybind module
try:
    from eopl import eopl_cpp_core
    HAS_CPP_CORE = True
except ImportError:
    HAS_CPP_CORE = False

def apply_gain_1d_cpp(pdata: image_t, gain_array_1d: Union[np.ndarray, xr.DataArray], run_options: Any, apply_dimension: str = "ACT") -> image_t:
    """Use the C++ CPU gain or delegate CUDA execution to CuPy."""
    if run_options.use_cuda:
        return _apply_gain_1d_cupy(pdata, gain_array_1d, apply_dimension)
    if not HAS_CPP_CORE:
        raise RuntimeError("C++/CUDA core module is not installed or failed to import.")

    log = Log()
    log.debug("Running C++ version")
    _apply_gain_1d_contract(
        dict(pdata.image().sizes),
        {"gain_array_1d": gain_array_1d, "apply_dimension": apply_dimension},
    )

    # Extract raw NumPy arrays from xarray
    image = pdata.image()
    if image.dims != ("Channels", "ALT", "ACT"):
        raise ValueError(
            "apply_gain_1d_cpp requires image dimensions ordered as "
            "('Channels', 'ALT', 'ACT')."
        )
    img_np = image.values
    cpu_dtypes = (np.dtype("float32"), np.dtype("float64"), np.dtype("uint32"))
    if img_np.dtype not in cpu_dtypes:
        raise TypeError(
            f"apply_gain_1d_cpp CPU supports float32, float64, and uint32; got {img_np.dtype}."
        )
    if not img_np.flags.c_contiguous:
        raise ValueError("apply_gain_1d_cpp requires C-contiguous image data.")

    # Safely extract the numpy array whether the input is xarray or raw numpy
    if isinstance(gain_array_1d, xr.DataArray):
        gain_np = gain_array_1d.values
    else:
        gain_np = np.asarray(gain_array_1d)

    gain_np = np.ascontiguousarray(gain_np, dtype=img_np.dtype)
    if gain_np.ndim != 1:
        raise ValueError(f"gain_array_1d must be 1-D, got {gain_np.ndim}-D.")

    eopl_cpp_core.apply_gain_1d(img_np, gain_np, False, False, 0, apply_dimension)

    return pdata

def _apply_gain_1d_cupy(data, gain_array_1d, apply_dimension):
    """Apply a dimension-aligned gain on the current CUDA device."""
    cupy = require_cupy()
    _apply_gain_1d_contract(
        dict(data.image().sizes),
        {"gain_array_1d": gain_array_1d, "apply_dimension": apply_dimension},
    )
    data.to_device()
    image = data.image()
    if isinstance(gain_array_1d, xr.DataArray):
        gain_array_1d = gain_array_1d.copy(
            deep=False, data=cupy.asarray(gain_array_1d.data, dtype=image.dtype)
        )
    else:
        gain_array_1d = xr.DataArray(
            cupy.asarray(gain_array_1d, dtype=image.dtype), dims=(apply_dimension,)
        )
    result = (image * gain_array_1d).transpose(*image.dims)
    data.replace_image(result.astype(image.dtype, copy=False))
    return data


# Attach flags for the Pipeline to read
apply_gain_1d_cpp.cuda_support = True
apply_gain_1d_cpp.cuda_dtypes = ("float32",)
apply_gain_1d_cpp.dask_support = False
apply_gain_1d_cpp.eopl_shape_contract = _apply_gain_1d_contract
apply_gain_1d_cpp.input_access = ("readwrite",)
apply_gain_1d_cpp.shape_contract = _apply_gain_1d_contract
