import numpy as np
import pytest
import xarray as xr

from eopl.algorithms.spatial import block_mean_downsample, gaussian_psf


def test_gaussian_psf_does_not_mix_spectral_axis():
    data = np.zeros((2, 9, 9), dtype=np.float32)
    data[1, 4, 4] = 1
    result = gaussian_psf(data, 1.0, spatial_axes=(1, 2))
    assert result.dtype == np.float32
    assert np.count_nonzero(result[0]) == 0
    assert result[1, 4, 4] < 1
    np.testing.assert_allclose(result[1].sum(), 1.0, rtol=1e-5)


def test_gaussian_psf_xarray_dask_matches_numpy():
    array = xr.DataArray(
        np.eye(12, dtype=np.float32), dims=("along_track", "cross_track")
    )
    expected = gaussian_psf(array, 0.8)
    actual = gaussian_psf(array.chunk({"along_track": 4, "cross_track": 4}), 0.8)
    assert actual.chunks is not None
    np.testing.assert_allclose(actual.compute(), expected, rtol=1e-6, atol=1e-7)


def test_block_mean_numpy_and_xarray():
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    expected = values.reshape(2, 2, 3, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(block_mean_downsample(values, (2, 2)), expected)
    array = xr.DataArray(values, dims=("y", "x"))
    np.testing.assert_allclose(
        block_mean_downsample(array, {"y": 2, "x": 2}), expected
    )


def test_block_mean_rejects_nonpositive_factor():
    with pytest.raises(ValueError, match="positive"):
        block_mean_downsample(np.ones((2, 2)), (1, 0))
