"""Notebook reports retain product sections and executable inspection examples."""
import json

import numpy as np
import pytest
import xarray as xr

from eopl import Pipeline


@pytest.mark.parametrize('with_input', [True, False])
def test_notebook_sections_and_precision_replay(tmp_path, with_input):
    source = tmp_path / 'input.nc'
    xr.Dataset({'image': (('Channels', 'ALT', 'ACT'), np.ones((2, 3, 4)))}).to_netcdf(source)
    calibration = tmp_path / 'ccdb'
    calibration.mkdir()
    (calibration / 'ccdb.xml').write_text(
        '<CalibrationDataBase><parameter name="gain" type="FLOAT">1.2</parameter></CalibrationDataBase>'
    )
    for precision_run in (False, True):
        pipeline = Pipeline()
        pipeline._output_root = tmp_path / 'output'
        pipeline._is_pa_run = precision_run
        pipeline.set_run_options(precision='float32')
        pipeline.generate_report(True, report_format='ipynb', print_all_bkp_data=True)
        pipeline.setup_precision_analysis()
        pipeline.add_ccdb(calibration)
        pipeline.report_print('Mission analysis')
        pipeline.report_add_property('Instrument', 'Demo')
        if with_input:
            image = pipeline.read_input(source)
        else:
            image = None
            pipeline.add_variable('time', np.arange(4), dims=('time',), pinned=True)
        pipeline.save_bkp(image, 'after-operation')
        pipeline.save_output(image, 'final')
    assert not (pipeline._output_root / 'report.py').exists()
    assert not (pipeline._output_root / 'report_pa.py').exists()
    for filename in ('report.ipynb', 'report_pa.ipynb'):
        notebook = json.loads((pipeline._output_root / filename).read_text())
        assert notebook['nbformat'] == 4
        assert notebook['metadata']['kernelspec']['name'] == 'python3'
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                assert cell['execution_count'] is None and cell['outputs'] == []
                compile(''.join(cell['source']), filename, 'exec')
        headings = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'markdown']
        assert '## Run Metadata' in headings
        assert '## Breakpoint: after-operation' in headings
        assert '## Final Output: final' in headings
    nominal = json.loads((pipeline._output_root / 'report.ipynb').read_text())
    text = '\n'.join(''.join(cell['source']) for cell in nominal['cells'])
    assert text.index('## CCDB') < text.index('## Custom Pipeline Reporting')
    assert 'display(ccdb.keys())' in text
    assert 'display(Markdown(' in text
    assert "product_1 = report.load_bkp('after-operation')" in text
    if with_input:
        assert text.index('## Input NetCDF') < text.index('## Custom Pipeline Reporting')
        assert "# input_data.dataset['image'].isel({'Channels': 0}).plot()" in text
    else:
        assert '## Input NetCDF' not in text


def test_report_format_validation_and_default():
    pipeline = Pipeline()
    pipeline.generate_report(True)
    assert pipeline.manifest['report_config']['report_format'] == 'py'
    with pytest.raises(ValueError, match='report_format'):
        pipeline.generate_report(True, report_format='pdf')
