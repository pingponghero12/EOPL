"""Named runs preserve the baseline and compare products with the same precision."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import xarray as xr

from eopl import Pipeline, image_t
from eopl.report import Report, BreakpointData


@pytest.mark.parametrize('report_format', ['py', 'ipynb'])
@pytest.mark.parametrize('file_type', ['nc', 'zarr'])
def test_named_run_and_comparison_report(tmp_path, monkeypatch, report_format, file_type):
    monkeypatch.chdir(tmp_path)
    baseline = Pipeline()
    baseline.parse_arguments([])
    baseline.set_run_options(precision='float32')
    baseline.generate_report(True, report_format=report_format)
    baseline.setup_precision_analysis()
    data = image_t(np.arange(1, 7, dtype=np.float32).reshape(2, 3), dims=('y', 'x'), name='image')
    baseline.save_bkp(data, 'after_gain', file_type=file_type)
    baseline.save_output(data, 'final', file_type=file_type)
    original_manifest = (tmp_path / 'outputs/run_manifest.yaml').read_bytes()
    variant = Pipeline()
    args = variant.parse_arguments(['--dir', 'doubled'])
    assert Path(args['output_root']) == Path('outputs') / 'doubled'
    variant.set_run_options(precision='float32')
    variant.generate_report(True, report_format=report_format)
    variant.setup_precision_analysis()
    changed = image_t(data.image() * 2, name='image')
    variant.save_bkp(changed, 'after_gain', file_type=file_type)
    variant.save_output(changed, 'final', file_type=file_type)
    assert (tmp_path / 'outputs/run_manifest.yaml').read_bytes() == original_manifest
    assert variant.manifest['precision'] == baseline.manifest['precision'] == 'float32'
    assert not variant._is_pa_run
    assert variant.manifest['comparison']['reference_manifest'] == str(tmp_path / 'outputs/run_manifest.yaml')
    report_path = tmp_path / 'outputs/doubled' / f'report_comparison.{report_format}'
    assert report_path.exists()
    if report_format == 'py':
        reports = []
        monkeypatch.setattr(sys, 'argv', ['report_comparison.py'])
        monkeypatch.setattr(Report, 'report_save_pdf', lambda report, name: reports.append(report))
        runpy.run_path(str(report_path), run_name='__main__')
        assert reports[0]._fig_counter == 2
        assert 'FLAGGED' in '\n'.join(reports[0]._latex_content)
    else:
        notebook = json.loads(report_path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(report_path), 'exec')
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        assert 'EOPL Run Comparison Report' in source
        assert 'product_1_reference = reference.load_bkp' in source
        assert 'report.plot_run_comparison' in source
        assert 'comparison_report.pdf' in source


@pytest.mark.parametrize('args', [ ['--dir', 'doubled'], ['--dir', '..'],
                                  ['--dir', '/tmp/run'], ['--dir', 'x', '--precision-analysis'] ])
def test_invalid_named_runs_fail_before_writing(tmp_path, monkeypatch, args):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        Pipeline().parse_arguments(args)
    assert not (tmp_path / 'outputs').exists()


def test_unsigned_difference_and_coordinate_mismatch(tmp_path):
    manifest = tmp_path / 'run_manifest.yaml'
    manifest.write_text('{}')
    report = Report(manifest)
    reference = BreakpointData('value', xr.Dataset({'count': ('x', np.array([1], dtype=np.uint32))}), {})
    current = BreakpointData('value', xr.Dataset({'count': ('x', np.array([0], dtype=np.uint32))}), {})
    report.plot_run_comparison(reference, current)
    assert 'Mean Abs Err = 1.00e+00' in '\n'.join(report._latex_content)
    current.dataset = current.dataset.assign_coords(x=[10])
    report.plot_run_comparison(reference, current)
    assert 'dimensions or coordinates differ' in '\n'.join(report._latex_content)
