"""Generic spectral-grid and spectral-response operations."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


ArrayLike = Union[np.ndarray, xr.DataArray]


def _validate_wavelengths(wavelengths: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(wavelengths, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two values.")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must contain finite, strictly increasing values.")
    return values


def interpolate_spectra(
    data: ArrayLike,
    target_wavelengths: Sequence[float],
    *,
    source_wavelengths: Optional[Sequence[float]] = None,
    wavelength_dim: str = "wavelength",
    spectral_axis: int = -1,
    fill_value: float = np.nan,
) -> ArrayLike:
    """Linearly interpolate one or more spectra onto a new wavelength grid.

    For an :class:`xarray.DataArray`, the source grid is taken from the named
    coordinate unless ``source_wavelengths`` is supplied. NumPy inputs require
    ``source_wavelengths``. Dask-backed xarray inputs remain lazy.
    """

    target = _validate_wavelengths(target_wavelengths, "target_wavelengths")
    if isinstance(data, xr.DataArray):
        if wavelength_dim not in data.dims:
            raise ValueError(f"DataArray has no wavelength dimension '{wavelength_dim}'.")
        if source_wavelengths is not None:
            source = _validate_wavelengths(source_wavelengths, "source_wavelengths")
            if source.size != data.sizes[wavelength_dim]:
                raise ValueError("source_wavelengths length does not match the spectral dimension.")
            data = data.assign_coords({wavelength_dim: source})
        elif wavelength_dim not in data.coords:
            raise ValueError(f"DataArray has no coordinate for '{wavelength_dim}'.")
        else:
            _validate_wavelengths(data.coords[wavelength_dim].values, wavelength_dim)

        original_dtype = data.dtype
        result = data.interp(
            {wavelength_dim: target},
            method="linear",
            kwargs={"fill_value": fill_value},
        )
        if np.issubdtype(original_dtype, np.floating):
            result = result.astype(original_dtype)
        result.attrs = dict(data.attrs)
        return result

    values = np.asarray(data)
    if source_wavelengths is None:
        raise ValueError("source_wavelengths is required for NumPy input.")
    source = _validate_wavelengths(source_wavelengths, "source_wavelengths")
    axis = np.lib.array_utils.normalize_axis_index(spectral_axis, values.ndim)
    if values.shape[axis] != source.size:
        raise ValueError("source_wavelengths length does not match the spectral axis.")
    result = interp1d(
        source,
        values,
        axis=axis,
        kind="linear",
        bounds_error=False,
        fill_value=fill_value,
        assume_sorted=True,
    )(target)
    if np.issubdtype(values.dtype, np.floating):
        result = result.astype(values.dtype, copy=False)
    return result


def spectral_response_convolution(
    data: ArrayLike,
    wavelengths: Sequence[float],
    responses: ArrayLike,
    *,
    wavelength_dim: str = "wavelength",
    band_dim: str = "band",
    spectral_axis: int = -1,
) -> ArrayLike:
    """Integrate spectra with normalized response functions on a common grid.

    ``responses`` has shape ``(number_of_bands, number_of_wavelengths)``. The
    trapezoidal quadrature weights are included before each response is
    normalized, so a constant spectrum is preserved.
    """

    wavelength = _validate_wavelengths(wavelengths, "wavelengths")
    response_values = responses.values if isinstance(responses, xr.DataArray) else np.asarray(responses)
    if response_values.ndim != 2 or response_values.shape[1] != wavelength.size:
        raise ValueError("responses must have shape (bands, wavelengths).")
    if np.any(~np.isfinite(response_values)) or np.any(response_values < 0):
        raise ValueError("responses must be finite and non-negative.")

    intervals = np.diff(wavelength)
    quadrature = np.empty_like(wavelength)
    quadrature[0] = intervals[0] / 2
    quadrature[-1] = intervals[-1] / 2
    quadrature[1:-1] = (intervals[:-1] + intervals[1:]) / 2
    weights = response_values * quadrature[None, :]
    totals = weights.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Every spectral response must have positive integrated area.")
    weights = weights / totals

    if isinstance(data, xr.DataArray):
        if wavelength_dim not in data.dims or data.sizes[wavelength_dim] != wavelength.size:
            raise ValueError("The DataArray spectral dimension does not match wavelengths.")
        if isinstance(responses, xr.DataArray) and responses.ndim == 2:
            band_coord = responses.coords.get(responses.dims[0], np.arange(weights.shape[0]))
        else:
            band_coord = np.arange(weights.shape[0])
        response_da = xr.DataArray(
            weights.astype(data.dtype if np.issubdtype(data.dtype, np.floating) else np.float64),
            dims=(band_dim, wavelength_dim),
            coords={band_dim: band_coord, wavelength_dim: wavelength},
        )
        source = data.assign_coords({wavelength_dim: wavelength})
        result = xr.dot(source, response_da, dim=wavelength_dim)
        desired = list(data.dims)
        desired[desired.index(wavelength_dim)] = band_dim
        result = result.transpose(*desired)
        result.attrs = dict(data.attrs)
        return result

    values = np.asarray(data)
    axis = np.lib.array_utils.normalize_axis_index(spectral_axis, values.ndim)
    if values.shape[axis] != wavelength.size:
        raise ValueError("The NumPy spectral axis does not match wavelengths.")
    moved = np.moveaxis(values, axis, -1)
    result = np.tensordot(moved, weights.T, axes=([-1], [0]))
    result = np.moveaxis(result, -1, axis)
    if np.issubdtype(values.dtype, np.floating):
        result = result.astype(values.dtype, copy=False)
    return result


interpolate_spectra.cuda_support = False
interpolate_spectra.dask_support = True
spectral_response_convolution.cuda_support = False
spectral_response_convolution.dask_support = True


__all__ = ["interpolate_spectra", "spectral_response_convolution"]
