import numpy as np
import pytest

from eopl import image_t, default_report_params


@pytest.fixture
def full_report_params():
    params = default_report_params()
    for name in params:
        if name.startswith('generate_'):
            params[name] = True
    params['plot_1d_images'] = True
    return params


@pytest.fixture
def make_eopl_image():
    def factory(dtype=np.float32, name="image", shape=(2, 3, 4)):
        values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        return image_t(values, dims=("Channels", "ALT", "ACT"), name=name)

    return factory
