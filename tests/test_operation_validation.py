"""Check the backend agreement and stochastic precision assertions."""

import numpy as np
import pytest
import xarray as xr

from eopl import image_t, operation_compare_test, operation_numerical_test


def test_agreement_isolates_in_place_operations_and_accepts_backend_keywords():
    image = image_t(np.array([1.0, 2.0, np.nan]), dims=("SPE",))

    def scale(data, factor):
        data.image().data[:] *= factor
        return data, {"pixels": data.image().size}

    assert operation_compare_test(
        scale, scale, image, kwargs={"factor": 2}, reference_kwargs={"factor": 2 + 1e-8}
    )
    assert not operation_compare_test(
        scale, scale, image, kwargs={"factor": 2}, reference_kwargs={"factor": 3}
    )
    np.testing.assert_array_equal(image.image(), [1.0, 2.0, np.nan])


def test_agreement_checks_shapes_coordinates_and_integer_values():
    actual = xr.DataArray([1.0, np.nan], dims="SPE", coords={"SPE": [400, 500]})
    assert operation_compare_test(lambda: actual, lambda: actual.chunk(SPE=1))
    assert not operation_compare_test(
        lambda: actual, lambda: actual.assign_coords(SPE=[400, 501])
    )
    assert not operation_compare_test(lambda: actual, lambda: np.array([1.0]))
    assert not operation_compare_test(lambda: actual, lambda: actual, equal_nan=False)
    assert not operation_compare_test(
        lambda: np.array([2**60]), lambda: np.array([2**60 + 1]), rtol=0, atol=0
    )


def test_significant_bits_detect_rounding_noise_and_consistent_bias():
    pytest.importorskip("significantdigits")
    rng = np.random.default_rng(42)
    expected = np.array([1.0, 0.0, np.nan])

    def sample_reference(error):
        # Known artificial rounding errors exercise the actual statistical library.
        return expected + np.array([*rng.choice((-error, error), size=2), 0.0])

    assert operation_numerical_test(sample_reference, expected, 2.0**-30, min_bits=25)
    assert not operation_numerical_test(sample_reference, expected, 2.0**-12, min_bits=25)
    assert not operation_numerical_test(lambda: expected + 0.01, expected, min_bits=25)
    assert not operation_numerical_test(lambda: np.nan_to_num(expected), expected, min_bits=25)


def test_significant_bits_can_use_a_separate_reference_callable():
    pytest.importorskip("significantdigits")

    def multiply(values, factor):
        return values * factor

    assert operation_numerical_test(
        multiply, multiply, np.array([0.5, 2.0]), min_bits=45,
        kwargs={"factor": 2}, reference_kwargs={"factor": 2},
    )
    with pytest.raises(ValueError, match="at least 59"):
        operation_numerical_test(multiply, 1.0, min_bits=10, samples=58)
