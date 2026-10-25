"""Generic spatial blur and integer-factor aggregation operations."""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import dask.array as da
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


ArrayLike = Union[np.ndarray, xr.DataArray]


def gaussian_psf(
    data: ArrayLike,
    sigma: Union[float, Sequence[float]],
    *,
    spatial_dims: Sequence[str] = ("along_track", "cross_track"),
    spatial_axes: Sequence[int] = (-2, -1),
    mode: str = "nearest",
    truncate: float = 4.0,
) -> ArrayLike:
    """Apply a Gaussian point-spread approximation over selected axes only."""

    if mode not in {"nearest", "reflect", "mirror", "wrap", "constant"}:
        raise ValueError(f"Unsupported boundary mode '{mode}'.")
    if np.isscalar(sigma):
        sigma_values = [float(sigma)] * len(
            spatial_dims if isinstance(data, xr.DataArray) else spatial_axes
        )
    else:
        sigma_values = [float(value) for value in sigma]
    if any(value < 0 for value in sigma_values):
        raise ValueError("sigma values must be non-negative.")

    if isinstance(data, xr.DataArray):
        if len(spatial_dims) != len(sigma_values) or any(
            dim not in data.dims for dim in spatial_dims
        ):
            raise ValueError(
                "spatial_dims and sigma must identify existing DataArray dimensions."
            )
        full_sigma = [0.0] * data.ndim
        for dim, value in zip(spatial_dims, sigma_values):
            full_sigma[data.get_axis_num(dim)] = value
        if data.chunks is not None:
            depth = {
                axis: int(np.ceil(truncate * value))
                for axis, value in enumerate(full_sigma)
                if value > 0
            }
            boundary = {
                "nearest": "nearest",
                "reflect": "reflect",
                "wrap": "periodic",
                "constant": 0,
            }.get(mode, "reflect")
            blurred = da.map_overlap(
                gaussian_filter,
                data.data,
                depth=depth,
                boundary=boundary,
                dtype=data.dtype,
                sigma=tuple(full_sigma),
                mode=mode,
                truncate=truncate,
            )
            return xr.DataArray(
                blurred,
                dims=data.dims,
                coords=data.coords,
                attrs=dict(data.attrs),
                name=data.name,
            )
        values = gaussian_filter(
            data.values, sigma=full_sigma, mode=mode, truncate=truncate
        )
        return xr.DataArray(
            values,
            dims=data.dims,
            coords=data.coords,
            attrs=dict(data.attrs),
            name=data.name,
        )

    values = np.asarray(data)
    if len(spatial_axes) != len(sigma_values):
        raise ValueError("spatial_axes and sigma must have the same length.")
    full_sigma = [0.0] * values.ndim
    for axis, value in zip(spatial_axes, sigma_values):
        full_sigma[np.lib.array_utils.normalize_axis_index(axis, values.ndim)] = value
    return gaussian_filter(values, sigma=full_sigma, mode=mode, truncate=truncate)


def block_mean_downsample(
    data: ArrayLike,
    factors: Union[Mapping[str, int], Sequence[int]],
    *,
    axes: Sequence[int] = (-2, -1),
) -> ArrayLike:
    """Downsample by non-overlapping integer-factor block means.

    Partial blocks at the high-index edge are discarded explicitly.
    """

    if isinstance(data, xr.DataArray):
        if not isinstance(factors, Mapping) or not factors:
            raise ValueError(
                "DataArray factors must be a non-empty dimension-to-factor mapping."
            )
        clean = {dim: int(factor) for dim, factor in factors.items()}
        if any(dim not in data.dims for dim in clean) or any(
            value <= 0 for value in clean.values()
        ):
            raise ValueError(
                "Downsampling dimensions must exist and factors must be positive."
            )
        return data.coarsen(clean, boundary="trim").mean()

    values = np.asarray(data)
    factor_values = tuple(int(value) for value in factors)
    if len(factor_values) != len(axes) or any(value <= 0 for value in factor_values):
        raise ValueError("factors and axes must have equal length and positive values.")
    result = values
    normalized_axes = [
        np.lib.array_utils.normalize_axis_index(axis, values.ndim) for axis in axes
    ]
    for axis, factor in sorted(zip(normalized_axes, factor_values)):
        size = result.shape[axis] - result.shape[axis] % factor
        slices = [slice(None)] * result.ndim
        slices[axis] = slice(0, size)
        result = result[tuple(slices)]
        shape = list(result.shape)
        shape[axis : axis + 1] = [size // factor, factor]
        result = result.reshape(shape).mean(axis=axis + 1)
    return result


gaussian_psf.cuda_support = False
gaussian_psf.dask_support = True
block_mean_downsample.cuda_support = False
block_mean_downsample.dask_support = True


def conservative_spline_upsample(data, factor):
    """Cubic-spline refinement of nonnegative cell averages on a uniform grid.

    The first two axes are spatial. Remaining axes are independent channels.
    Negative spline excursions are clipped, then each parent's child block is
    rescaled to recover its original mean. Thus the integral is preserved when
    each child has 1/factor**2 of its parent's area. This is not interpolation
    of point samples and does not preserve unweighted sums of pixel values.
    """
    from scipy.ndimage import zoom

    values = np.asarray(data)
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, np.integer))
        or factor < 1
    ):
        raise ValueError("factor must be a positive integer")
    if values.ndim < 2 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("data must be a floating array with two spatial axes")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("cell averages must be finite and nonnegative")
    ny, nx = values.shape[:2]
    channels = values.reshape(ny, nx, -1)
    output = np.empty(
        (ny * factor, nx * factor, channels.shape[-1]), dtype=values.dtype
    )
    for channel in range(channels.shape[-1]):
        parent = channels[..., channel]
        fine = zoom(parent, factor, order=3, mode="nearest", grid_mode=True)
        np.maximum(fine, 0, out=fine)
        blocks = fine.reshape(ny, factor, nx, factor)
        means = blocks.mean(axis=(1, 3), dtype=np.float64)
        ratio = np.divide(parent, means, out=np.zeros_like(means), where=means > 0)
        blocks *= ratio[:, None, :, None]
        missing = (means == 0) & (parent > 0)
        if np.any(missing):
            blocks[:] = np.where(
                missing[:, None, :, None], parent[:, None, :, None], blocks
            )
        output[..., channel] = fine
    return output.reshape((ny * factor, nx * factor) + values.shape[2:])


conservative_spline_upsample.cuda_support = False
conservative_spline_upsample.dask_support = False

__all__ = ["gaussian_psf", "block_mean_downsample", "conservative_spline_upsample"]
