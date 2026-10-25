import numpy as np
import pytest
import xarray as xr

from eopl.algorithms.spectral import interpolate_spectra, spectral_response_convolution


def test_interpolation_matches_linear_reference_and_preserves_dtype():
    source = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    values = np.array([[2.0, 4.0, 8.0]], dtype=np.float32)
    result = interpolate_spectra(
        values,
        [1.0, 1.5, 2.0, 3.0, 4.0],
        source_wavelengths=source,
    )
    np.testing.assert_allclose(result, [[2.0, 3.0, 4.0, 6.0, 8.0]])
    assert result.dtype == np.float32


def test_xarray_interpolation_stays_lazy_with_dask():
    data = xr.DataArray(
        np.arange(12, dtype=np.float32).reshape(3, 4),
        dims=("pixel", "wavelength"),
        coords={"wavelength": [1.0, 2.0, 3.0, 4.0]},
    ).chunk({"pixel": 1})
    result = interpolate_spectra(data, [1.5, 2.5, 3.5])
    assert result.chunks is not None
    assert result.dtype == np.float32
    np.testing.assert_allclose(result.compute()[0], [0.5, 1.5, 2.5])


@pytest.mark.parametrize("bad", [[2.0, 1.0], [1.0, 1.0], [1.0, np.nan]])
def test_interpolation_rejects_invalid_wavelengths(bad):
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_spectra(np.ones(2), [1.0, 2.0], source_wavelengths=bad)


def test_response_integration_preserves_constant_and_axis():
    wavelengths = np.array([1.0, 2.0, 3.0, 4.0])
    responses = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    spectra = np.full((2, 4, 3), 7.0, dtype=np.float32)
    result = spectral_response_convolution(
        spectra, wavelengths, responses, spectral_axis=1
    )
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 7.0)


def test_response_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        spectral_response_convolution(np.ones(3), [1, 2, 3], [[1, -1, 1]])
