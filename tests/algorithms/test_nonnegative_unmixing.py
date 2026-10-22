import numpy as np
import pytest
from eopl.algorithms.unmixing import nonnegative_linear_unmixing

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_exact_nonnegative_mixture(dtype):
    e = np.array([[1, 0], [0, 1], [1, 1]], dtype=dtype)
    c = np.array([[0.2, 0.8], [0, 0.4]], dtype=dtype)
    actual = nonnegative_linear_unmixing(c @ e.T, e)
    np.testing.assert_allclose(actual, c, atol=1e-7)
    assert actual.dtype == dtype

def test_negative_solution_is_constrained():
    np.testing.assert_allclose(nonnegative_linear_unmixing([-1., 2.], np.eye(2)), [0, 2])
    with pytest.raises(ValueError):
        nonnegative_linear_unmixing([np.nan, 1], np.eye(2))
