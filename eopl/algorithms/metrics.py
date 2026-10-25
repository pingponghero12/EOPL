"""Reusable deterministic comparison metrics for scientific arrays."""

from __future__ import annotations

import numpy as np

try:
    from numpy.lib.array_utils import normalize_axis_index
except ImportError:  # NumPy 1.x
    from numpy.core.multiarray import normalize_axis_index


def error_metrics(reference, candidate, *, epsilon: float = 1e-12):
    """Return common scalar error summaries over finite paired samples."""

    ref, got = np.broadcast_arrays(np.asarray(reference), np.asarray(candidate))
    valid = np.isfinite(ref) & np.isfinite(got)
    if not np.any(valid):
        raise ValueError("No finite paired samples are available for comparison.")
    ref = ref[valid].astype(np.float64, copy=False)
    got = got[valid].astype(np.float64, copy=False)
    difference = got - ref
    absolute = np.abs(difference)
    relative = absolute / np.maximum(np.abs(ref), epsilon)
    return {
        "sample_count": int(ref.size),
        "mean_signed_bias": float(np.mean(difference)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "p99_absolute_error": float(np.percentile(absolute, 99)),
        "maximum_absolute_error": float(np.max(absolute)),
        "mean_relative_error": float(np.mean(relative)),
        "normalized_mae": float(np.sum(absolute) / max(np.sum(np.abs(ref)), epsilon)),
        "normalized_rmse": float(np.linalg.norm(difference) / max(np.linalg.norm(ref), epsilon)),
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.percentile(relative, 95)),
    }


def spectral_angle(reference, candidate, *, spectral_axis: int = -1, epsilon: float = 1e-12):
    """Calculate spectral angle in radians for paired spectral vectors."""

    ref, got = np.broadcast_arrays(np.asarray(reference), np.asarray(candidate))
    axis = normalize_axis_index(spectral_axis, ref.ndim)
    numerator = np.sum(ref * got, axis=axis)
    denominator = np.linalg.norm(ref, axis=axis) * np.linalg.norm(got, axis=axis)
    cosine = numerator / np.maximum(denominator, epsilon)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


error_metrics.cuda_support = False
error_metrics.dask_support = False
spectral_angle.cuda_support = False
spectral_angle.dask_support = False


__all__ = ["error_metrics", "spectral_angle"]
