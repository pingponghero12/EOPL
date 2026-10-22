"""Small, transparent linear-unmixing building blocks."""

from __future__ import annotations

import numpy as np

try:
    from numpy.lib.array_utils import normalize_axis_index
except ImportError:  # NumPy 1.x
    from numpy.core.multiarray import normalize_axis_index


def project_simplex(values, *, axis: int = -1, total: float = 1.0):
    """Project vectors onto ``x >= 0, sum(x) = total`` in Euclidean norm.

    The vectorized implementation follows the sorting-based simplex projection
    and supports arbitrary leading dimensions.
    """

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must contain at least one vector dimension.")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("total must be finite and positive.")
    axis = normalize_axis_index(axis, array.ndim)
    moved = np.moveaxis(array, axis, -1)
    if moved.shape[-1] == 0 or np.any(~np.isfinite(moved)):
        raise ValueError("values must contain finite, non-empty vectors.")
    ordered = np.sort(moved, axis=-1)[..., ::-1]
    cumulative = np.cumsum(ordered, axis=-1) - total
    indexes = np.arange(1, ordered.shape[-1] + 1, dtype=array.dtype)
    support = ordered - cumulative / indexes > 0
    rho = support.sum(axis=-1) - 1
    theta = np.take_along_axis(cumulative, rho[..., None], axis=-1)[..., 0] / (rho + 1)
    projected = np.maximum(moved - theta[..., None], 0)
    return np.moveaxis(projected, -1, axis).astype(array.dtype, copy=False)


def projected_linear_unmixing(
    observed,
    endmembers,
    *,
    spectral_axis: int = -1,
    rcond=None,
):
    """Least-squares abundance estimate followed by physical simplex projection.

    Parameters
    ----------
    observed:
        One or more spectra. The spectral axis must match the first dimension
        of ``endmembers``.
    endmembers:
        Matrix with shape ``(wavelength, component)``.

    Notes
    -----
    This is a fast deterministic baseline, not an exact fully constrained
    least-squares optimizer. The returned abundance vectors are non-negative
    and sum to one.
    """

    spectra = np.asarray(observed)
    matrix = np.asarray(endmembers)
    if spectra.ndim == 0 or matrix.ndim != 2:
        raise ValueError("observed must be an array and endmembers must be two-dimensional.")
    axis = normalize_axis_index(spectral_axis, spectra.ndim)
    if spectra.shape[axis] != matrix.shape[0]:
        raise ValueError("The observed spectral axis must match the endmember wavelength dimension.")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("endmembers must be finite.")
    dtype = np.result_type(spectra.dtype, matrix.dtype, np.float32)
    moved = np.moveaxis(spectra.astype(dtype, copy=False), axis, -1)
    inverse = np.linalg.pinv(matrix.astype(dtype, copy=False), rcond=rcond)
    abundance = moved @ inverse.T
    return project_simplex(abundance, axis=-1).astype(dtype, copy=False)


project_simplex.cuda_support = False
project_simplex.dask_support = False
projected_linear_unmixing.cuda_support = False
projected_linear_unmixing.dask_support = False


def nonnegative_linear_unmixing(observed, endmembers):
    """Solve nonnegative least squares independently for each spectrum.

    SciPy's active-set solver uses float64 internally. Results are cast to the
    joint input dtype (at least float32). Coefficients need not sum to one;
    their sum can be retained as a separate brightness scale.
    """
    from scipy.optimize import nnls

    y, e = np.asarray(observed), np.asarray(endmembers)
    if y.ndim < 1 or e.ndim != 2 or y.shape[-1] != e.shape[0]:
        raise ValueError("Expected spectra (..., wavelength) and matrix (wavelength, component).")
    if not np.isfinite(y).all() or not np.isfinite(e).all() or e.shape[1] == 0:
        raise ValueError("Inputs must be finite and the component basis nonempty.")
    dtype = np.result_type(y.dtype, e.dtype, np.float32)
    matrix = e.astype(np.float64)
    flat = y.reshape(-1, y.shape[-1]).astype(np.float64)
    result = np.empty((len(flat), e.shape[1]), dtype=dtype)
    for i, spectrum in enumerate(flat):
        result[i] = nnls(matrix, spectrum, maxiter=100 * e.shape[1])[0]
    return result.reshape(y.shape[:-1] + (e.shape[1],))


__all__ = ["project_simplex", "projected_linear_unmixing", "nonnegative_linear_unmixing"]
