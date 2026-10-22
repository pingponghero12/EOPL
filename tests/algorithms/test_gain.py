import numpy as np
import pytest
import xarray as xr

from eopl import image_t
from eopl.exceptions import ShapeError
from eopl.algorithms.gain import HAS_CPP_CORE, apply_gain_1d, apply_gain_1d_dask, apply_gain_1d_cpp


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("dimension", ["Channels", "ALT", "ACT"])
def test_python_gain_all_dimensions_and_dtypes(make_eopl_image, options, dtype, dimension):
    image = make_eopl_image(dtype=dtype)
    before = image.image().values.copy()
    size = image.image().sizes[dimension]
    gain = np.arange(1, size + 1, dtype=np.float64)
    result = apply_gain_1d(image, gain, options, apply_dimension=dimension)
    expected_gain = gain.astype(dtype)
    axis = image.image().get_axis_num(dimension)
    shape = [1] * before.ndim
    shape[axis] = size
    expected = (before * expected_gain.reshape(shape)).astype(dtype)
    assert result is image
    assert result.image().dtype == dtype
    assert result.image().dims == ("Channels", "ALT", "ACT")
    np.testing.assert_array_equal(result.image(), expected)


def test_python_gain_accepts_named_dataarray(make_eopl_image, options):
    image = make_eopl_image()
    gain = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=("ACT",))
    result = apply_gain_1d(image, gain, options, apply_dimension="ACT")
    np.testing.assert_array_equal(result.image()[0, 0], [1, 4, 9, 16])


def test_python_and_native_integer_policy_casts_gain_before_multiplication(make_eopl_image, options):
    image = make_eopl_image(dtype=np.uint32)
    before = image.image().values.copy()
    result = apply_gain_1d(image, np.full(4, 1.9), options)
    np.testing.assert_array_equal(result.image(), before)


@pytest.mark.parametrize('operation', [apply_gain_1d, apply_gain_1d_dask])
def test_python_gain_stays_lazy_with_dask(make_eopl_image, options, operation):
    image = make_eopl_image()
    image.data = image.data.chunk({"ACT": 2})
    result = operation(image, np.ones(4, dtype=np.float32) * 2, options)
    assert result.is_dask
    np.testing.assert_array_equal(
        result.image().compute(),
        np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4) * 2,
    )


@pytest.mark.parametrize(
    "gain,dimension,message",
    [
        (np.ones(4), "missing", "does not have dimension"),
        (np.ones((2, 2)), "ACT", "must be 1-D"),
        (np.ones(3), "ACT", "has length"),
    ],
)
def test_gain_shape_validation(make_eopl_image, options, gain, dimension, message):
    with pytest.raises(ShapeError, match=message):
        apply_gain_1d(make_eopl_image(), gain, options, apply_dimension=dimension)


def test_gain_rejects_dataarray_with_wrong_dimension(make_eopl_image, options):
    gain = xr.DataArray(np.ones(4), dims=("wrong",))
    with pytest.raises(ShapeError, match="DataArray gain dimension"):
        apply_gain_1d(make_eopl_image(), gain, options, apply_dimension="ACT")


@pytest.mark.skipif(not HAS_CPP_CORE, reason="native extension was not built")
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint32])
@pytest.mark.parametrize("dimension", ["Channels", "ALT", "ACT"])
def test_native_cpu_gain_all_supported_dtypes_and_dimensions(make_eopl_image, options, dtype, dimension):
    image = make_eopl_image(dtype=dtype)
    before = image.image().values.copy()
    size = image.image().sizes[dimension]
    gain = np.arange(1, size + 1, dtype=dtype)
    result = apply_gain_1d_cpp(image, gain, options, apply_dimension=dimension)
    axis = image.image().get_axis_num(dimension)
    shape = [1] * before.ndim
    shape[axis] = size
    np.testing.assert_array_equal(result.image(), before * gain.reshape(shape))


@pytest.mark.skipif(not HAS_CPP_CORE, reason="native extension was not built")
def test_native_cpu_rejects_unsupported_dtype(make_eopl_image, options):
    with pytest.raises(TypeError, match="supports float32, float64, and uint32"):
        apply_gain_1d_cpp(make_eopl_image(dtype=np.uint16), np.ones(4), options)


@pytest.mark.skipif(not HAS_CPP_CORE, reason="native extension was not built")
def test_native_cpu_requires_canonical_dimension_order(options):
    image = image_t(
        np.ones((4, 3, 2), dtype=np.float32),
        dims=("ACT", "ALT", "Channels"),
        name="image",
    )
    with pytest.raises(ValueError, match="dimensions ordered"):
        apply_gain_1d_cpp(image, np.ones(4), options)


@pytest.mark.skipif(not HAS_CPP_CORE, reason="native extension was not built")
def test_native_cpu_rejects_noncontiguous_image(options):
    values = np.ones((2, 3, 4), dtype=np.float32)[:, :, ::-1]
    image = image_t(values, dims=("Channels", "ALT", "ACT"), name="image")
    with pytest.raises(ValueError, match="C-contiguous"):
        apply_gain_1d_cpp(image, np.ones(4), options)


def test_operation_capabilities_are_declared():
    assert apply_gain_1d.dask_support
    assert not apply_gain_1d.cuda_support
    assert apply_gain_1d_cpp.cuda_support
    assert apply_gain_1d_cpp.cuda_dtypes == ("float32",)
    assert apply_gain_1d_cpp.input_access == ("readwrite",)


def test_legacy_gain_imports_use_the_packaged_algorithms():
    from eopl import apply_gain_1d as top_level_gain
    from eopl.ops.gain import apply_gain_1d as legacy_gain
    from eopl.ops.gain import apply_gain_1d_cpp as legacy_cpp_gain

    assert top_level_gain is legacy_gain is apply_gain_1d
    assert legacy_cpp_gain is apply_gain_1d_cpp
