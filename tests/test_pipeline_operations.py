import sys
import types

import numpy as np
import pytest

from eopl import Pipeline, image_t


def test_single_input_and_direct_return(make_eopl_image):
    pipeline = Pipeline()
    image = make_eopl_image()

    def increment(value, amount, run_options):
        value.replace_image(value.image() + amount)
        return value

    result = pipeline.run_operation(image, increment, {"amount": 2})
    assert result is image
    np.testing.assert_array_equal(result.image(), np.arange(24).reshape(2, 3, 4) + 2)


def test_tuple_inputs_are_positional_and_outputs_unpack_normally(make_eopl_image):
    pipeline = Pipeline()
    image = make_eopl_image(name="image")
    topology = make_eopl_image(name="topology")

    def derive(first, second, run_options):
        output = image_t(first.image() + second.image(), name="combined")
        return first, second, output.image().values, 4.5

    derive.input_access = ("read", "read")
    got_image, got_topology, array, scalar = pipeline.run_operation(
        data=(image, topology), func=derive, args={}
    )
    assert got_image is image and got_topology is topology
    assert isinstance(array, np.ndarray)
    assert scalar == 4.5


def test_dependency_injection_does_not_mutate_caller_args(make_eopl_image):
    pipeline = Pipeline()
    pipeline.rng = np.random.default_rng(9)
    args = {"amount": 3}

    def operation(value, amount, rng, run_options):
        assert rng is pipeline.rng
        assert run_options is not None
        return value

    pipeline.run_operation(make_eopl_image(), operation, args)
    assert args == {"amount": 3}


def test_run_function_has_no_image_and_returns_values_directly():
    pipeline = Pipeline()
    pipeline.global_config["mission"] = "demo"
    pipeline.local_config["step"] = "10"
    pipeline.rng = np.random.default_rng(2)

    def geometry(duration, global_config, local_config, rng, run_options, ccdb):
        assert global_config is pipeline.global_config
        assert local_config is pipeline.local_config
        assert rng is pipeline.rng
        assert ccdb is None
        return np.arange(duration), 7.0

    times, value = pipeline.run_function(geometry, {"duration": 3})
    np.testing.assert_array_equal(times, [0, 1, 2])
    assert value == 7.0
    assert pipeline.manifest["operations"][-1]["type"] == "function"


@pytest.mark.parametrize("precision,expected", [("float32", np.float32), ("float64", np.float64)])
def test_cpu_operation_enforces_requested_float_precision(make_eopl_image, precision, expected):
    pipeline = Pipeline()
    pipeline.set_run_options(precision=precision)
    image = make_eopl_image(dtype=np.float64 if expected is np.float32 else np.float32)

    def identity(value, run_options):
        return value

    pipeline.run_operation(image, identity)
    assert image.image().dtype == expected


def test_cpu_operation_enforces_precision_on_new_image_outputs(make_eopl_image):
    pipeline = Pipeline()
    pipeline.set_run_options(precision="float32")

    def create(value):
        return image_t(value.image().astype(np.float64), name="created"), np.ones(2)

    create.input_access = ("read",)
    created, ordinary_array = pipeline.run_operation(make_eopl_image(), create)
    assert created.image().dtype == np.float32
    assert ordinary_array.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.int16])
def test_cpu_precision_does_not_convert_integer_images(make_eopl_image, dtype):
    pipeline = Pipeline()
    pipeline.set_run_options(precision="float32")

    def identity(value, run_options):
        return value

    image = make_eopl_image(dtype=dtype)
    pipeline.run_operation(image, identity)
    assert image.image().dtype == dtype


def test_dask_disables_cuda_for_only_that_operation(make_eopl_image):
    pipeline = Pipeline()
    pipeline.set_run_options(use_cuda=True)
    numpy_image = make_eopl_image(name="numpy")
    dask_image = make_eopl_image(name="dask")
    pipeline.set_dask(dask_image, True, "ACT")

    def operation(first, second, run_options):
        assert not run_options.use_cuda
        assert run_options.dask_enabled
        return first, second

    operation.cuda_support = True
    operation.dask_support = True
    operation.input_access = ("read", "read")
    result = pipeline.run_operation((numpy_image, dask_image), operation)
    assert result == (numpy_image, dask_image)
    assert pipeline.run_options.use_cuda


def test_dask_unsupported_operation_is_rejected(make_eopl_image):
    pipeline = Pipeline()
    image = pipeline.set_dask(make_eopl_image(), True, "ACT")

    def operation(value):
        return value

    with pytest.raises(RuntimeError, match="does not support Dask"):
        pipeline.run_operation(image, operation)


def test_numpy_dask_numpy_and_rechunk_transitions(make_eopl_image):
    pipeline = Pipeline()
    image = make_eopl_image(shape=(2, 20, 30))
    original = image.image().values.copy()
    assert not image.is_dask
    assert pipeline.set_dask(image, True, "ACT") is image
    assert image.is_dask
    assert pipeline.set_dask(image, True, "ALT") is image
    assert image.is_dask
    assert pipeline.set_dask(image, False) is image
    assert not image.is_dask
    np.testing.assert_array_equal(image.image(), original)


def test_cuda_cache_moves_only_when_freshness_changes(monkeypatch, make_eopl_image):
    calls = {"push": [], "copy": [], "free": []}

    class Core:
        HAS_CUDA_COMPILED = True

        @staticmethod
        def push_to_vram_numpy(array, pointer):
            calls["push"].append(pointer)
            return pointer or 101

        @staticmethod
        def copy_to_host_numpy(array, pointer):
            calls["copy"].append(pointer)

        @staticmethod
        def free_vram(pointer):
            calls["free"].append(pointer)

    monkeypatch.setattr(sys.modules["eopl"], "eopl_cpp_core", Core, raising=False)

    pipeline = Pipeline()
    pipeline.set_run_options(use_cuda=True)
    image = make_eopl_image()

    def gpu_operation(value, run_options):
        assert run_options.use_cuda
        return value

    gpu_operation.cuda_support = True
    gpu_operation.cuda_dtypes = ("float32",)
    gpu_operation.input_access = ("readwrite",)

    pipeline.run_operation(image, gpu_operation)
    pipeline.run_operation(image, gpu_operation)
    assert calls["push"] == [0]
    assert calls["copy"] == []

    def cpu_operation(value, run_options):
        assert not run_options.use_cuda
        return value

    pipeline.run_operation(image, cpu_operation)
    assert calls["copy"] == [101]
    assert image._device_array == 101

    pipeline.run_operation(image, gpu_operation)
    assert calls["push"] == [0, 101]
    assert calls["copy"] == [101]
    image.release_device()
    assert calls["free"] == [101]


def test_cuda_tuple_inputs_are_cached_independently(monkeypatch, make_eopl_image):
    pushed = []

    class Core:
        HAS_CUDA_COMPILED = True

        @staticmethod
        def push_to_vram_numpy(array, pointer):
            result = pointer or 200 + len(pushed)
            pushed.append((array.shape, pointer))
            return result

        @staticmethod
        def free_vram(pointer):
            pass

    monkeypatch.setattr(sys.modules["eopl"], "eopl_cpp_core", Core, raising=False)
    pipeline = Pipeline()
    pipeline.set_run_options(use_cuda=True)
    first, second = make_eopl_image(name="first"), make_eopl_image(name="second")

    def combine(a, b, run_options):
        return a

    combine.cuda_support = True
    combine.input_access = ("read", "read")
    pipeline.run_operation((first, second), combine)
    pipeline.run_operation((first, second), combine)
    assert len(pushed) == 2
    first.clear_device_state()
    second.clear_device_state()


def test_read_only_cpu_operation_keeps_current_device_cache(monkeypatch, make_eopl_image):
    copied = []

    class Core:
        @staticmethod
        def copy_to_host_numpy(array, pointer):
            copied.append(pointer)

        @staticmethod
        def free_vram(pointer):
            pass

    monkeypatch.setattr(sys.modules["eopl"], "eopl_cpp_core", Core, raising=False)
    image = make_eopl_image()
    image._device_array = 55
    image._device_shape = image.image().shape
    image._device_dtype = image.image().dtype.str
    image._device_nbytes = image.image().nbytes
    image.mark_synchronized()

    def inspect_only(value):
        return float(value.image().mean())

    inspect_only.input_access = ("read",)
    assert Pipeline().run_operation(image, inspect_only) == 11.5
    assert copied == []
    assert not image._device_dirty and not image._host_dirty
    image.clear_device_state()


def test_cuda_build_and_layout_preconditions(monkeypatch, make_eopl_image):
    core = types.SimpleNamespace(HAS_CUDA_COMPILED=False)
    monkeypatch.setattr(sys.modules["eopl"], "eopl_cpp_core", core, raising=False)
    pipeline = Pipeline()
    pipeline.set_run_options(use_cuda=True)

    def operation(value):
        return value

    operation.cuda_support = True
    with pytest.raises(RuntimeError, match="without CUDA"):
        pipeline.run_operation(make_eopl_image(), operation)

    core.HAS_CUDA_COMPILED = True
    with pytest.raises(TypeError, match="supports CUDA dtypes"):
        pipeline.run_operation(make_eopl_image(dtype=np.uint32), operation)
    with pytest.raises(ValueError, match="3-D"):
        pipeline.run_operation(
            image_t(np.ones((2, 2), dtype=np.float32), dims=("ALT", "ACT")),
            operation,
        )


def test_failed_cuda_write_releases_uncertain_device_copy(monkeypatch, make_eopl_image):
    freed = []

    class Core:
        HAS_CUDA_COMPILED = True

        @staticmethod
        def push_to_vram_numpy(array, pointer):
            return pointer or 99

        @staticmethod
        def free_vram(pointer):
            freed.append(pointer)

    monkeypatch.setattr(sys.modules["eopl"], "eopl_cpp_core", Core, raising=False)
    pipeline = Pipeline()
    pipeline.set_run_options(use_cuda=True)
    image = make_eopl_image()

    def failing(value):
        raise RuntimeError("kernel failed")

    failing.cuda_support = True
    with pytest.raises(RuntimeError, match="kernel failed"):
        pipeline.run_operation(image, failing)
    assert freed == [99]
    assert image._device_array is None


def test_invalid_operation_input_and_access_metadata(make_eopl_image):
    pipeline = Pipeline()

    def operation(value):
        return value

    with pytest.raises(TypeError, match="image_t"):
        pipeline.run_operation(np.zeros((2, 2)), operation)
    with pytest.raises(TypeError, match="run_function"):
        pipeline.run_operation(None, operation)
    operation.input_access = ("write",)
    with pytest.raises(ValueError, match="input_access"):
        pipeline.run_operation(make_eopl_image(), operation)


def test_set_dask_rejects_missing_dimension(make_eopl_image):
    with pytest.raises(ValueError, match="missing dimension"):
        Pipeline().set_dask(make_eopl_image(), True, "missing")


def test_dask_to_numpy_does_not_require_chunk_dimension(make_eopl_image):
    pipeline = Pipeline()
    image = pipeline.set_dask(make_eopl_image(), True, "ACT")
    pipeline.set_dask(image, False, "not_used")
    assert not image.is_dask
