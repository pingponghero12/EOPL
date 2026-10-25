"""Report generation for original inputs and optional calibration databases."""
import runpy
import sys

import numpy as np
import pytest
import xarray as xr

from eopl import Pipeline
from eopl.report import Report


@pytest.mark.parametrize('with_input', [True, False])
@pytest.mark.parametrize('with_ccdb', [True, False])
def test_report_input_and_ccdb_sections(tmp_path, monkeypatch, with_input, with_ccdb, full_report_params):
    monkeypatch.chdir(tmp_path)
    pipeline = Pipeline()
    pipeline._output_root = tmp_path / 'outputs'
    pipeline.generate_report(True, full_report_params)
    pipeline.report_section('Custom section')
    if with_input:
        original = xr.Dataset({
            'image': (('y', 'x'), np.ones((3, 4), dtype=np.float64)),
            'scalar': 2.5,
            'profile': ('x', np.arange(4.0)),
            'labels': ('x', ['a', 'b', 'c', 'd']),
        }, coords={'x': [10, 20, 30, 40]}, attrs={'mission': 'Demo & ESA'})
        original.image.attrs['units'] = 'radiance'
        original.to_netcdf('input.nc')
        image = pipeline.read_input('input.nc')
        image.replace_image(image.image() * 2)
    else:
        image = None
        pipeline.add_variable('value', 1, pinned=True)
    if with_ccdb:
        calibration = tmp_path / 'ccdb'
        calibration.mkdir()
        (calibration / 'ccdb.xml').write_text(
            '<CalibrationDataBase><parameter name="gain" type="FLOAT">1.2</parameter>'
            '<parameter name="mission" type="STRING">Demo</parameter></CalibrationDataBase>'
        )
        pipeline.add_ccdb('ccdb')
        assert pipeline.ccdb.keys() == ('gain', 'mission')
    pipeline.save_bkp(image, 'after_operation')
    pipeline.save_output(image, 'final')
    script = pipeline._output_root / 'report.py'
    source = script.read_text()
    assert source.index('# --- CCDB ---') < source.index('# --- Custom Pipeline Reporting ---')
    assert "product_1 = report.load_bkp('after_operation')" in source
    assert '# ax.plot(np.asarray(value))' in source
    assert '# im = ax.imshow(np.asarray(value)' in source
    assert ('ccdb = CCDB(report.manifest' in source) == with_ccdb
    if with_input:
        assert source.index('report.section("Input NetCDF")') < source.index('# --- Custom Pipeline Reporting ---')
        report = Report(pipeline._output_root / 'run_manifest.yaml')
        loaded = report.load_input()
        xr.testing.assert_identical(loaded, original)
        loaded.close()
    else:
        assert 'input_data = report.load_input()' not in source
    # Execute the editable script from another directory, without requiring LaTeX.
    monkeypatch.chdir(pipeline._output_root)
    monkeypatch.setattr(sys, 'argv', ['report.py'])
    reports = []
    monkeypatch.setattr(Report, 'report_save_pdf', lambda report, name: reports.append(report))
    runpy.run_path(str(script), run_name='__main__')
    assert len(reports) == 1
    if with_input:
        content = '\n'.join(reports[0]._latex_content)
        for expected in ('Global attributes', 'radiance', 'labels', 'profile', 'scalar'):
            assert expected in content
