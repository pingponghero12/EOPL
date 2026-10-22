"""Reusable Earth observation processing algorithms."""
from .gain import apply_gain_1d, apply_gain_1d_dask, apply_gain_1d_cpp
from .metrics import error_metrics, spectral_angle
from .spatial import block_mean_downsample, gaussian_psf, conservative_spline_upsample
from .spectral import interpolate_spectra, spectral_response_convolution
from .unmixing import project_simplex, projected_linear_unmixing, nonnegative_linear_unmixing

__all__ = [
    "apply_gain_1d",
    "apply_gain_1d_dask",
    "apply_gain_1d_cpp",
    "block_mean_downsample",
    "conservative_spline_upsample",
    "error_metrics",
    "gaussian_psf",
    "interpolate_spectra",
    "project_simplex",
    "projected_linear_unmixing",
    "nonnegative_linear_unmixing",
    "spectral_angle",
    "spectral_response_convolution",
]
