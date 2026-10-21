"""Check the packaged native CPU operation against the Python operation."""
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from eopl import image_t
from eopl.ops.gain import apply_gain_1d, apply_gain_1d_cpp


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'uint32'])
@pytest.mark.parametrize('dimension', ['Channels', 'ALT', 'ACT'])
def test_cpu_gain_matches_python(dtype, dimension):
    pytest.importorskip('eopl.eopl_cpp_core')
    values = np.arange(1, 25, dtype=dtype).reshape(2, 3, 4)
    source = xr.DataArray(values, dims=('Channels', 'ALT', 'ACT'))
    python_image = image_t(source.copy(deep=True))
    native_image = image_t(source.copy(deep=True))
    gain = np.arange(1, source.sizes[dimension] + 1, dtype=dtype)
    options = SimpleNamespace(use_cuda=False)
    apply_gain_1d(python_image, gain, options, dimension)
    apply_gain_1d_cpp(native_image, gain, options, dimension)
    xr.testing.assert_identical(python_image.image(), native_image.image())


@pytest.mark.parametrize('dtype', ['int32', 'uint16', 'uint64', '>u4'])
def test_native_cpu_gain_rejects_unsupported_dtypes(dtype):
    core = pytest.importorskip('eopl.eopl_cpp_core')
    image = np.ones((2, 3, 4), dtype=dtype)
    gain = np.ones(4, dtype=dtype)
    with pytest.raises(ValueError, match='CPU execution supports'):
        core.apply_gain_1d(image, gain, False, False, 0, 'ACT')
