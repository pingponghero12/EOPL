import json

import numpy as np
import pytest
import xarray as xr

from eopl import Pipeline


def test_pipeline_accumulates_scalar_array_and_dataarray():
    pipeline = Pipeline()
    pipeline.add_variable("scalar", 3.5, pinned=True)
    pipeline.add_variable("vector", np.arange(3), dims=("sample",))
    source = xr.DataArray(np.ones((2, 3)), dims=("time", "xyz"))
    pipeline.add_variable("position", source, dims=("time", "xyz"), pinned=True)
    assert pipeline._variables["scalar"].item() == 3.5
    assert pipeline._variables["vector"].dims == ("sample",)
    assert pipeline._variables["position"].shape == (2, 3)


def test_replacing_and_renaming_variables_updates_pin_state():
    pipeline = Pipeline()
    pipeline.add_variable("value", 1, pinned=True)
    pipeline.change_variable_name("value", "renamed")
    assert "renamed" in pipeline._pinned_vars
    pipeline.add_variable("renamed", 2, pinned=False)
    assert "renamed" not in pipeline._pinned_vars
    assert pipeline._variables["renamed"].item() == 2


def test_geometry_checkpoint_and_output_need_no_image(tmp_path):
    pipeline = Pipeline()
    pipeline._output_root = tmp_path
    pipeline.add_variable("time", np.arange(4), dims=("time",), pinned=True)
    pipeline.add_variable("temporary", np.ones(2), dims=("scratch",), pinned=False)
    assert pipeline.save_bkp("geometry", clean=True) is None
    checkpoint = xr.open_dataset(tmp_path / "breakpoints" / "geometry.nc")
    assert "time" in checkpoint.variables
    assert "temporary" in checkpoint.variables
    assert json.loads(checkpoint.attrs["_eopl_image_names"]) == []
    checkpoint.close()

    assert "temporary" not in pipeline._variable_names
    pipeline.add_variable("temporary", np.ones(2), dims=("scratch",))
    pipeline.save_output("geometry_final", clean=False)
    output = xr.open_dataset(tmp_path / "geometry_final.nc")
    assert {"time", "temporary"}.issubset(output.variables)
    output.close()


def test_one_and_multiple_images_serialize_with_pipeline_variables(tmp_path, make_eopl_image):
    pipeline = Pipeline()
    pipeline._output_root = tmp_path
    pipeline.add_variable("quality", np.arange(4), dims=("ACT",), pinned=True)
    first = make_eopl_image(name="image")
    second = make_eopl_image(name="topology")
    returned = pipeline.save_bkp((first, second), "both")
    assert returned == (first, second)
    dataset = xr.open_dataset(tmp_path / "breakpoints" / "both.nc")
    assert set(dataset.data_vars) == {"image", "topology", "quality"}
    assert json.loads(dataset.attrs["_eopl_image_names"]) == ["image", "topology"]
    dataset.close()


def test_breakpoint_resume_restores_images_and_accumulator(tmp_path, make_eopl_image):
    writer = Pipeline()
    writer._output_root = tmp_path
    writer.add_variable("quality", [1, 2, 3, 4], dims=("ACT",), pinned=True)
    writer.save_bkp((make_eopl_image(name="image"), make_eopl_image(name="topology")), "state")

    reader = Pipeline()
    reader._output_root = tmp_path
    reader.start_bkp("state")
    restored = reader.save_bkp(None, "state")
    assert isinstance(restored, tuple) and len(restored) == 2
    assert [item.array_name for item in restored] == ["image", "topology"]
    np.testing.assert_array_equal(reader._variables["quality"], [1, 2, 3, 4])


def test_duplicate_image_names_and_name_overlap_are_rejected(make_eopl_image):
    pipeline = Pipeline()
    first, second = make_eopl_image(name="same"), make_eopl_image(name="same")
    with pytest.raises(ValueError, match="unique names"):
        pipeline._dataset_for_save((first, second))
    pipeline.add_variable("image", 1, pinned=True)
    with pytest.raises(ValueError, match="overlap"):
        pipeline._dataset_for_save(make_eopl_image(name="image"))
