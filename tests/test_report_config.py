"""Selected sections control report code, not the saved scientific products."""
import ast
import json
import runpy
import sys

import nbformat
import numpy as np
import pytest
import xarray as xr

from eopl import Pipeline, default_report_params
from eopl.report import Report


FLAGS = [name for name in default_report_params() if name.startswith('generate_')]


def test_report_params_are_independent_and_validated():
    first = default_report_params()
    first['generate_metadata'] = True
    assert not default_report_params()['generate_metadata']
    pipeline = Pipeline()
    pipeline.generate_report(True, {'generate_input': True})
    assert pipeline.manifest['report_config']['generate_input']
    assert pipeline.manifest['report_config']['generate_final']
    assert not pipeline.manifest['report_config']['generate_breakpoints']
    with pytest.raises(ValueError, match='Unknown report parameters'):
        pipeline.generate_report(True, {'generate_breakponts': True})
    with pytest.raises(TypeError, match='boolean'):
        pipeline.generate_report(True, {'generate_final': 'yes'})
    with pytest.raises(TypeError, match='dictionary'):
        pipeline.generate_report(True, False)


@pytest.mark.parametrize('report_format', ['py', 'ipynb'])
@pytest.mark.parametrize('mode', ['ordinary', 'comparison', 'precision'])
@pytest.mark.parametrize('selection', ['default', 'none', *FLAGS])
def test_section_selection(tmp_path, monkeypatch, report_format, mode, selection):
    monkeypatch.chdir(tmp_path)
    xr.Dataset({'image': (('y', 'x'), np.ones((2, 3), dtype=np.float32))}).to_netcdf('input.nc')
    calibration = tmp_path / 'ccdb'
    calibration.mkdir()
    (calibration / 'ccdb.xml').write_text(
        '<CalibrationDataBase><parameter name="gain" type="FLOAT">1.2</parameter></CalibrationDataBase>'
    )
    params = default_report_params()
    params['report_format'] = report_format
    if selection != 'default':
        params.update({name: name == selection for name in FLAGS})

    def save_run(argv):
        pipeline = Pipeline()
        pipeline.parse_arguments(argv)
        pipeline.set_run_options(precision='float32')
        pipeline.generate_report(True, params)
        pipeline.setup_precision_analysis()
        pipeline.add_ccdb(calibration)
        pipeline.report_section('User study')
        pipeline.report_inject_code('report.add_property("User result", 42)')
        image = pipeline.read_input('input.nc')
        pipeline.save_bkp(image, 'hidden_step')
        pipeline.save_output(image, 'final')
        return pipeline

    pipeline = save_run([])
    original_manifest = (tmp_path / 'outputs/run_manifest.yaml').read_bytes()
    filename = 'report'
    if mode == 'comparison':
        pipeline = save_run(['--dir', 'changed'])
        filename = 'report_comparison'
    elif mode == 'precision':
        pipeline = save_run(['--precision-analysis'])
        filename = 'report_pa'
    if mode != 'ordinary':
        assert (tmp_path / 'outputs/run_manifest.yaml').read_bytes() == original_manifest

    path = pipeline._output_root / f'{filename}.{report_format}'
    if report_format == 'py':
        source = path.read_text()
    else:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        source = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    tree = ast.parse(source)
    sections = [node.args[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'section' and node.args
                and isinstance(node.args[0], ast.Constant)]
    assert any('Metadata' in title for title in sections) == params['generate_metadata']
    assert ('CCDB' in sections) == params['generate_ccdb']
    assert ('Input NetCDF' in sections) == params['generate_input']
    assert any('hidden_step' in title for title in sections) == params['generate_breakpoints']
    assert any('final' in title.lower() for title in sections) == params['generate_final']
    assert ('Custom Analysis' in sections) == params['generate_custom_analysis']
    assert ('User study' in sections) == params['generate_custom_analysis']
    assert ('ccdb = CCDB(' in source) == params['generate_ccdb']
    assert ('input_data = report.load_input()' in source) == params['generate_input']
    assert (pipeline._output_root / 'final.nc').exists() or mode == 'precision'
    assert (tmp_path / 'outputs/breakpoints/hidden_step.nc').exists()

    if report_format == 'py':
        monkeypatch.setattr(sys, 'argv', [filename])
        monkeypatch.setattr(Report, 'report_save_pdf', lambda *args: None)
        runpy.run_path(str(path), run_name='__main__')


@pytest.mark.parametrize('report_format', ['py', 'ipynb'])
def test_minimal_report_does_not_need_original_input_or_ccdb(tmp_path, monkeypatch, report_format):
    monkeypatch.chdir(tmp_path)
    pipeline = Pipeline()
    pipeline.parse_arguments([])
    pipeline.generate_report(True, {'report_format': report_format})
    pipeline.manifest['input'] = {'file': str(tmp_path / 'missing.nc')}
    pipeline.manifest['ccdb'] = {'directory': str(tmp_path / 'missing-ccdb')}
    pipeline.manifest['breakpoints'] = [{'label': 'missing', 'file': 'missing-bkp.nc'}]
    pipeline.add_variable('result', 42, pinned=True)
    pipeline.save_output('final')
    path = pipeline._output_root / f'report.{report_format}'
    if report_format == 'py':
        monkeypatch.setattr(sys, 'argv', ['report.py'])
        reports = []
        monkeypatch.setattr(Report, 'report_save_pdf', lambda report, name: reports.append(report))
        runpy.run_path(str(path), run_name='__main__')
        assert '42' in '\n'.join(reports[0]._latex_content)
    else:
        notebook = json.loads(path.read_text())
        assert not any('## CCDB' in ''.join(cell['source']) for cell in notebook['cells'])
