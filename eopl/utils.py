import numpy as np
import xarray as xr
from typing import Union, Tuple, Dict
import scipy.ndimage as ndimage

def create_xarray(
    data: np.ndarray, 
    coords: np.ndarray, 
    data_name: str = "data", 
    coord_name: str = "sampling"
) -> xr.DataArray:
    """Wraps a NumPy array into an xarray DataArray with specified coordinates."""
    return xr.DataArray(
        data,
        coords={coord_name: coords},
        dims=[coord_name],
        name=data_name
    )

def lin_interp_resize(array: np.ndarray, size: int) -> np.ndarray:
    """
    Resizes a 1D array to a new length using linear interpolation.
    Used for scaling CCDB parameters to match current simulation dimensions.
    """
    original_indices = np.linspace(0, 1, num=len(array))
    new_indices = np.linspace(0, 1, num=size)

    return np.interp(new_indices, original_indices, array) 


def interp_nd_edge(
    da: xr.DataArray, 
    target_coords: Dict[str, Union[xr.DataArray, np.ndarray]]
) -> xr.DataArray:
    """
    Interpolates an N-D xarray to match target samplings along specified dimensions.
    Uses 'extrapolate' to effectively continue the edge values for out-of-bounds coordinates.
    
    Parameters
    ----------
    da : xr.DataArray
        The input data array.
    target_coords : dict
        A dictionary mapping dimension names to target 1D coordinate arrays.
        Example: {"ACT": new_act_sampling, "ALT": new_alt_sampling}
    """
    # Clean the input dictionary to ensure they are raw numpy arrays for interpolation
    clean_coords = {
        dim: (val.coords[dim].values if isinstance(val, xr.DataArray) else val)
        for dim, val in target_coords.items()
    }
    
    # xarray's extrapolate handles edge cases when interpolating outside the original grid
    return da.interp(
        clean_coords, 
        kwargs={"bounds_error": False, "fill_value": "extrapolate"}
    )

def resample_kernel_crop_nd(
    kernel_da: xr.DataArray, 
    target_coords: Dict[str, Union[xr.DataArray, np.ndarray]]
) -> xr.DataArray:
    """
    Resamples an N-D kernel to a target sampling. 
    Drops values that fall outside the original kernel's domain, cropping it.
    """
    clean_coords = {
        dim: (val.coords[dim].values if isinstance(val, xr.DataArray) else val)
        for dim, val in target_coords.items()
    }

    # Interpolate, filling out-of-bounds with NaNs
    resampled = kernel_da.interp(
        clean_coords, 
        kwargs={"bounds_error": False, "fill_value": np.nan}
    )
    
    # Iteratively drop the NaNs along all modified dimensions to crop the array
    for dim in clean_coords.keys():
        resampled = resampled.dropna(dim=dim, how="all")
        
    return resampled

def shift_array(
    arr: np.ndarray, 
    shift: Union[float, Tuple[float, ...]], 
    mode: str = 'nearest'
) -> np.ndarray:
    """
    Shifts an N-dimensional array by integer or fractional pixels.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array (1D, 2D, or 3D).
    shift : float or tuple of floats
        The amount to shift along each axis. E.g., (1.5, -0.2).
    mode : str
        How to handle edges. 'nearest' (edge padding), 'wrap' (periodic/roll), or 'constant'.
    """
    if isinstance(shift, (int, float)):
        shift = (shift,) * arr.ndim
        
    if len(shift) != arr.ndim:
        raise ValueError(f"Shift tuple length ({len(shift)}) must match array dimensions ({arr.ndim}).")
        
    # ndimage.shift handles both integer rolls and subpixel interpolations perfectly
    return ndimage.shift(arr, shift, mode=mode, prefilter=True)

def upsample_cubic(
    image: np.ndarray, 
    scale: Union[float, Tuple[float, float]], 
    mode: str = 'reflect'
) -> np.ndarray:
    """
    Upsamples a 2D or 3D image using 3rd-order cubic spline interpolation.
    Clean implementation relying on SciPy's optimized ndimage module.
    """
    if isinstance(scale, (int, float)):
        zoom_factors = (float(scale), float(scale))
    else:
        zoom_factors = tuple(float(s) for s in scale)

    if image.ndim == 2:
        return ndimage.zoom(image, zoom_factors, order=3, mode=mode, prefilter=True)
    
    elif image.ndim == 3:
        # Scale spatial dimensions, keep channels intact (zoom factor 1.0)
        zoom_3d = (*zoom_factors, 1.0)
        return ndimage.zoom(image, zoom_3d, order=3, mode=mode, prefilter=True)
    
    else:
        raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D.")
