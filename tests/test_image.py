import numpy as np
import pytest
import xarray as xr

from eopl import image_t, p_data


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.uint8, np.uint16, np.uint32, np.int16],
)
def test_image_t_keeps_numpy_dtype_and_dimensions(dtype):
    image = image_t(np.ones((2, 3), dtype=dtype), dims=("ALT", "ACT"), name="radiance")
    assert image.image().dims == ("ALT", "ACT")
    assert image.image().dtype == dtype
    assert image.array_name == "radiance"
    assert p_data is image_t


def test_dataarray_and_legacy_dataset_construction():
    array = xr.DataArray(np.ones((2,)), dims=("ACT",), name="signal")
    assert image_t(array).image().identical(array)

    dataset = xr.Dataset({"signal": array, "aux": ("ACT", [3, 4])})
    dataset.attrs["_eopl_array_name"] = "signal"
    image = image_t(dataset)
    assert image.array_name == "signal"
    assert list(image.image().coords) == list(array.coords)


def test_default_dimensions_and_renaming():
    image = image_t(np.zeros((2, 3)))
    assert image.image().dims == ("dim_0", "dim_1")
    image.change_array_name("renamed")
    assert image.array_name == "renamed"
    assert image.image().name == "renamed"


def test_invalid_dataset_and_dimension_count():
    with pytest.raises(ValueError, match="empty"):
        image_t(xr.Dataset())
    with pytest.raises(ValueError, match="dimension names"):
        image_t(np.zeros((2, 3)), dims=("x",))


def test_freshness_state_transitions(make_eopl_image):
    image = make_eopl_image()
    assert image._device_dirty and not image._host_dirty
    image._device_array = 12
    image.mark_synchronized()
    assert not image._device_dirty and not image._host_dirty
    image.mark_device_modified()
    assert image._host_dirty and not image._device_dirty
    image.mark_host_modified()
    assert image._device_dirty and not image._host_dirty
    image.clear_device_state()
    assert image._device_array is None and image._device_dirty


def test_replace_image_requires_dataarray(make_eopl_image):
    image = make_eopl_image()
    with pytest.raises(TypeError, match="DataArray"):
        image.replace_image(np.ones((2, 3, 4)))
