import numpy as np
import pytest

from eopl import image_t


@pytest.fixture
def make_eopl_image():
    def factory(dtype=np.float32, name="image", shape=(2, 3, 4)):
        values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        return image_t(values, dims=("Channels", "ALT", "ACT"), name=name)

    return factory
