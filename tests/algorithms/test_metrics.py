import numpy as np
import pytest

from eopl.algorithms.metrics import error_metrics, spectral_angle


def test_error_metrics_known_values_and_nan_mask():
    reference = np.array([1.0, 2.0, np.nan])
    candidate = np.array([2.0, 0.0, 5.0])
    metrics = error_metrics(reference, candidate)
    assert metrics["sample_count"] == 2
    assert metrics["mean_signed_bias"] == pytest.approx(-0.5)
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["maximum_absolute_error"] == pytest.approx(2.0)
    assert metrics["normalized_mae"] == pytest.approx(1.0)
    assert metrics["normalized_rmse"] == pytest.approx(1.0)


def test_spectral_angle_known_vectors():
    reference = np.array([[1.0, 0.0], [1.0, 0.0]])
    candidate = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(spectral_angle(reference, candidate), [0, np.pi / 2])


def test_error_metrics_rejects_all_invalid():
    with pytest.raises(ValueError, match="No finite"):
        error_metrics([np.nan], [np.nan])
