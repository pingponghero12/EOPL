import numpy as np
import pytest

from eopl.algorithms.unmixing import project_simplex, projected_linear_unmixing


def test_simplex_projection_constraints_and_known_value():
    values = np.array([[0.2, -0.1, 1.3], [1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
    result = project_simplex(values)
    np.testing.assert_allclose(result[0], [0, 0, 1])
    np.testing.assert_allclose(result.sum(axis=-1), 1, rtol=1e-6)
    assert np.all(result >= 0)
    assert result.dtype == np.float32


def test_projected_unmixing_recovers_exact_mixtures():
    endmembers = np.array([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float64)
    abundance = np.array([[0.25, 0.75], [0.8, 0.2]], dtype=np.float64)
    observed = abundance @ endmembers.T
    result = projected_linear_unmixing(observed, endmembers)
    np.testing.assert_allclose(result, abundance, atol=1e-12)


def test_projected_unmixing_supports_nonfinal_spectral_axis():
    endmembers = np.eye(2, dtype=np.float32)
    observed = np.array([[0.2, 0.7], [0.8, 0.3]], dtype=np.float32)
    result = projected_linear_unmixing(observed, endmembers, spectral_axis=0)
    np.testing.assert_allclose(result, [[0.2, 0.8], [0.7, 0.3]], atol=1e-6)


def test_unmixing_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="spectral axis"):
        projected_linear_unmixing(np.ones(3), np.ones((2, 2)))
