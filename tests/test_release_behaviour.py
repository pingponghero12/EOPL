"""Fail early for unsupported precision replays and honor disabled reporting."""
import numpy as np
import pytest

from eopl import Pipeline


def test_precision_cli_requires_baseline(tmp_path):
    with pytest.raises(SystemExit):
        Pipeline().parse_arguments(['global.xml', 'local.xml', 'input.nc', str(tmp_path), '--precision-analysis'])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('precision,threshold', [('float32', 1e-7), ('float64', -1), ('float64', np.inf), ('float64', np.nan)])
def test_invalid_precision_analysis_settings(precision, threshold):
    with pytest.raises(ValueError):
        Pipeline().setup_precision_analysis(precision, threshold)


def test_disabled_report_stays_disabled_for_replay(tmp_path):
    for replay in (False, True):
        pipeline = Pipeline()
        pipeline._output_root = tmp_path
        pipeline._is_pa_run = replay
        pipeline.generate_report(False)
        pipeline.add_variable('value', 1.0, pinned=True)
        pipeline.save_output('result')
    assert (tmp_path / 'result.nc').exists()
    assert (tmp_path / 'result_pa.nc').exists()
    assert not list(tmp_path.glob('report*'))
