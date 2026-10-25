import numpy as np
import pytest

from eopl import image_t


class Options:
    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda


@pytest.fixture
def options():
    return Options(False)


@pytest.fixture
def make_eopl_image():
    def factory(dtype=np.float32, dims=("Channels", "ALT", "ACT"), shape=(2, 3, 4)):
        values = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
        return image_t(values, dims=dims, name="image")

    return factory
