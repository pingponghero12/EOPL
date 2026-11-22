"""Boolean checks for operation agreement and numerical precision."""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral

import numpy as np
import xarray as xr

from .data import image_t, is_cupy_array, require_cupy


def _run(operation, args, kwargs):
    # An in-place operation must not change another run's inputs or the caller's.
    arguments, options = deepcopy((args, kwargs))
    return operation(*arguments, **options)


def _comparison_inputs(actual, reference):
    if isinstance(actual, image_t):
        actual = actual.host_image()
    if isinstance(reference, image_t):
        reference = reference.host_image()
    if isinstance(actual, xr.DataArray) and isinstance(reference, xr.DataArray):
        if actual.dims != reference.dims or not actual.coords.equals(reference.coords):
            return None
    if isinstance(actual, xr.DataArray):
        actual = actual.to_numpy()
    if isinstance(reference, xr.DataArray):
        reference = reference.to_numpy()
    if is_cupy_array(actual):
        actual = require_cupy().asnumpy(actual)
    if is_cupy_array(reference):
        reference = require_cupy().asnumpy(reference)
    return np.asarray(actual), np.asarray(reference)


def _agree(actual, reference, rtol, atol, equal_nan):
    if isinstance(actual, xr.Dataset) or isinstance(reference, xr.Dataset):
        if not isinstance(actual, xr.Dataset) or not isinstance(reference, xr.Dataset):
            return False
        if not actual.coords.equals(reference.coords):
            return False
        return _agree(dict(actual.data_vars), dict(reference.data_vars), rtol, atol, equal_nan)
    if isinstance(actual, Mapping) or isinstance(reference, Mapping):
        return (
            isinstance(actual, Mapping)
            and isinstance(reference, Mapping)
            and actual.keys() == reference.keys()
            and all(_agree(actual[k], reference[k], rtol, atol, equal_nan) for k in actual)
        )
    if isinstance(actual, (tuple, list)) and isinstance(reference, (tuple, list)):
        return len(actual) == len(reference) and all(
            _agree(a, b, rtol, atol, equal_nan) for a, b in zip(actual, reference)
        )

    inputs = _comparison_inputs(actual, reference)
    if inputs is None:
        return False
    a, b = inputs
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "biufc" and b.dtype.kind in "biufc" and (
        a.dtype.kind in "fc" or b.dtype.kind in "fc"
    ):
        return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
    return bool(np.array_equal(a, b))


def operation_compare_test(
    operation, reference, *args, rtol=1e-7, atol=0.0, equal_nan=True,
    kwargs=None, reference_kwargs=None,
) -> bool:
    """Return whether two operations agree on copies of the same inputs.

    `kwargs` applies to both calls. `reference_kwargs` can override each keyword
    for the reference, for example to select a different backend. Floating-point
    values use NumPy tolerances relative to the reference. Shapes and Xarray
    coordinates must match, and integer and boolean values compare exactly.
    Supports images, arrays, scalars, datasets and nested sequences or mappings.
    Operation exceptions propagate.
    """
    if not all(np.isfinite(tolerance) and tolerance >= 0 for tolerance in (rtol, atol)):
        raise ValueError("rtol and atol must be finite and non-negative.")
    options = dict(kwargs or {})
    actual = deepcopy(_run(operation, args, options))
    expected = _run(reference, args, {**options, **(reference_kwargs or {})})
    return _agree(actual, expected, rtol, atol, equal_nan)


def operation_numerical_test(
    operation, reference, *args, min_bits, samples=60, kwargs=None, reference_kwargs=None,
) -> bool:
    """Check significant bits against a trusted result using significantdigits.

    Pass an instrumented operation that independently perturbs floating-point
    rounding each time it runs. The reference is a trusted scalar, array or
    image, or a callable that produces it. A callable receives the same inputs,
    with optional overrides from `reference_kwargs`.
    Returns True when every finite reference element meets `min_bits` using
    the distribution-free General estimator. At least 59 independent runs give
    a 95% confidence bound on 95% success for each element separately.
    Missing values and signed infinities must match the reference in every run.

    Absolute errors are scaled by the reference's binary exponent, using unit
    scale at zero. The score uses up to 52 binary64 fraction bits. Operations
    run on copies of the inputs. Use small test cases, since all outputs are
    collected for the estimate. This function does not instrument arithmetic.
    """
    if isinstance(min_bits, bool) or not isinstance(min_bits, Integral) or not 1 <= min_bits <= 52:
        raise ValueError("min_bits must be an integer from 1 to 52.")
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples < 59:
        raise ValueError("samples must be at least 59 for the 95% per-element bound.")
    try:
        import significantdigits as sd
    except ImportError as error:
        raise ImportError("Install eopl[numerics] to run the significant-bits test.") from error

    options = dict(kwargs or {})
    if callable(reference):
        reference = _run(reference, args, {**options, **(reference_kwargs or {})})
    reference = deepcopy(reference)

    # Boolean indexing copies each result, even when the kernel reuses a buffer.
    observations = []
    for _ in range(samples):
        result = _run(operation, args, options)
        pair = _comparison_inputs(result, reference)
        if pair is None:
            return False
        actual, expected = pair
        if actual.dtype.kind not in "biuf" or expected.dtype.kind not in "biuf":
            raise TypeError("The significant-bits test expects real numeric output and reference.")
        if actual.shape != expected.shape:
            return False
        finite = np.isfinite(expected)
        if not finite.any() or not np.array_equal(np.isfinite(actual), finite):
            return False
        if not np.array_equal(actual[~finite], expected[~finite], equal_nan=True):
            return False
        observations.append(actual[finite])

    bits = sd.significant_digits(
        np.stack(observations), expected[finite], error=sd.Error.Absolute,
        method=sd.Method.General, dtype=np.float64,
    )
    return bool(np.all(bits >= min_bits))
