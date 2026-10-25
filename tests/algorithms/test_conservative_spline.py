import numpy as np
import pytest

from eopl.algorithms.spatial import conservative_spline_upsample


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("factor", [1, 3, 6])
def test_spline_preserves_each_cell_mean(dtype, factor):
    parent = np.random.default_rng(4).random((5, 7, 2)).astype(dtype)
    parent[1, 2] = 0
    fine = conservative_spline_upsample(parent, factor)
    assert fine.shape == (5 * factor, 7 * factor, 2)
    assert fine.dtype == dtype
    assert np.all(fine >= 0)
    recovered = fine.reshape(5, factor, 7, factor, 2).mean((1, 3))
    np.testing.assert_allclose(recovered, parent, rtol=5e-7, atol=1e-15)
    np.testing.assert_allclose(
        fine.sum(dtype=np.float64) / factor**2, parent.sum(dtype=np.float64), rtol=5e-7
    )


def test_spline_constant_and_invalid():
    np.testing.assert_allclose(conservative_spline_upsample(np.ones((4, 5)), 3), 1)
    for factor in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            conservative_spline_upsample(np.ones((4, 5)), factor)
    for values in (
        np.ones(5),
        np.ones((2, 2), dtype=int),
        np.array([[np.nan]]),
        -np.ones((2, 2)),
    ):
        with pytest.raises(ValueError):
            conservative_spline_upsample(values, 3)
