from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from eopl import Pipeline
from eopl.algorithms import apply_gain_1d
from eopl.algorithms.cuda_example import scale_image


def test_scale_image_cpu_preserves_metadata(make_eopl_image):
    image = make_eopl_image()
    image.image().attrs["units"] = "1"
    expected = image.image() * 2

    result = scale_image(image, 2, SimpleNamespace(use_cuda=False))

    xr.testing.assert_allclose(result.image(), expected)
    assert result.image().attrs == image.image().attrs
    assert result.array_name == image.array_name


def test_cupy_gain_kernel_and_checkpoint(tmp_path, make_eopl_image):
    cupy = pytest.importorskip("cupy")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA is unavailable: {exc}")
    if device_count == 0:
        pytest.skip("No CUDA device")

    pipeline = Pipeline(output_root=tmp_path)
    pipeline.set_run_options(use_cuda=True)
    image = make_eopl_image()
    image.image().attrs["units"] = "1"
    gain = np.arange(1, 5, dtype=np.float32)
    expected = image.image().values * gain * 2

    image = pipeline.run_operation(image, apply_gain_1d, {"gain_array_1d": gain})
    pixels = image.image().data
    pipeline.save_bkp(image, "gain")
    assert image.image().data is pixels

    resumed = Pipeline(output_root=tmp_path)
    resumed.set_run_options(use_cuda=True)
    resumed.start_bkp("gain")
    restored = resumed.save_bkp(None, "gain")
    assert isinstance(restored.image().data, np.ndarray)

    for runner, source in ((pipeline, image), (resumed, restored)):
        result = runner.run_operation(source, scale_image, {"factor": 2})
        assert isinstance(result.image().data, cupy.ndarray)
        np.testing.assert_array_equal(result.host_image(), expected)

    pipeline.save_output(result, "scaled")
    with xr.open_dataset(Path(tmp_path) / "scaled.nc") as saved:
        np.testing.assert_array_equal(saved[result.array_name].values, expected)
