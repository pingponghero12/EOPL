"""XML and YAML configuration use the same Pipeline interface."""

import numpy as np
import pytest

from eopl import Pipeline, image_t


@pytest.mark.parametrize('extension', ['yml', 'yaml'])
def test_yaml_preserves_nested_settings_and_types(tmp_path, extension):
    """YAML settings are usable directly without converting strings to numbers."""
    path = tmp_path / f'local.{extension}'
    path.write_text('scene:\n  spacing: 6.0\nenabled: true\nbands: [1, 2]\n')

    pipeline = Pipeline()
    pipeline.add_local_config(path)

    assert pipeline.local_config == {
        'scene': {'spacing': 6.0}, 'enabled': True, 'bands': [1, 2]
    }


@pytest.mark.parametrize('extension, content', [
    ('xml', '<config><parameter name="RandomSeed">42</parameter></config>'),
    ('yml', 'RandomSeed: 42\n'),
])
def test_global_config_initialises_same_seed(tmp_path, extension, content):
    """XML and YAML global settings generate the same reproducible random draws."""
    path = tmp_path / f'global.{extension}'
    path.write_text(content)
    pipeline = Pipeline()

    pipeline.add_global_config(path)

    np.testing.assert_array_equal(pipeline.rng.normal(size=5), np.random.default_rng(42).normal(size=5))
    assert pipeline.manifest['random_seed'] == 42
    if extension == 'xml':
        assert pipeline.global_config['RandomSeed'] == '42'


@pytest.mark.parametrize('content', ['- item\n', '12\n', 'true\n', '1: value\n'])
def test_yaml_requires_named_settings(tmp_path, content):
    """Reject YAML documents that are not mappings with string parameter names."""
    path = tmp_path / 'config.yml'
    path.write_text(content)

    with pytest.raises(ValueError, match='mapping with string keys'):
        Pipeline().add_local_config(path)


def test_empty_yaml_and_multiple_config_files(tmp_path):
    """Empty documents are harmless and subsequent files update existing settings."""
    first = tmp_path / 'first.yml'
    first.write_text('mission: CHIME\n')
    empty = tmp_path / 'empty.yaml'
    empty.write_text('# No overrides\n')
    pipeline = Pipeline()

    pipeline.add_local_config(first)
    pipeline.add_local_config(empty)

    assert pipeline.local_config == {'mission': 'CHIME'}


def test_output_directory_without_argument_parsing(tmp_path):
    """A callable module can save its product without invoking a CLI parser."""
    pipeline = Pipeline(output_root=tmp_path / 'module')
    data = image_t(np.ones((2, 3)), dims=('ALT', 'ACT'), name='image')

    pipeline.save_output(data, 'result')

    assert (tmp_path / 'module' / 'result.nc').is_file()
    assert (tmp_path / 'module' / 'run_manifest.yaml').is_file()
