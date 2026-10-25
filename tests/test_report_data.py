"""Reports expose Xarray products while retaining internal reporting context."""
import numpy as np
import pytest
import xarray as xr
import yaml

from eopl.report import BreakpointData, Report


@pytest.mark.parametrize('file_type', ['nc', 'zarr'])
def test_report_loaders_return_datasets_and_preserve_metadata(tmp_path, file_type):
    original = xr.Dataset(
        {'image': (('y', 'x'), np.arange(6, dtype=np.float32).reshape(2, 3)),
         'profile': ('x', np.arange(3, dtype=np.float32))},
        attrs={'mission': 'Demo'},
    )
    original.image.attrs['units'] = 'radiance'
    original.to_netcdf(tmp_path / 'input.nc')
    path = tmp_path / f'checkpoint.{file_type}'
    if file_type == 'nc':
        original.to_netcdf(path)
    else:
        original.to_zarr(path)
    metadata = {'label': 'after_gain', 'file': path.name, 'duration_sec': 0.25}
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(yaml.safe_dump({
        'input': {'file': str(tmp_path / 'input.nc')},
        'breakpoints': [metadata],
    }))
    report = Report(manifest)
    product = report.load_bkp('after_gain')
    input_data = report.load_input()
    assert isinstance(product, xr.Dataset)
    assert isinstance(input_data, xr.Dataset)
    xr.testing.assert_identical(product, original)
    xr.testing.assert_identical(input_data, original)
    assert product['image'].attrs == {'units': 'radiance'}

    context = report._as_breakpoint(product)
    assert context.label == 'after_gain'
    assert context.metadata == metadata
    wrapped = report.load_bkp('after_gain', with_metadata=True)
    assert isinstance(wrapped, BreakpointData)
    assert wrapped.label == 'after_gain'
    assert wrapped.metadata == metadata
    xr.testing.assert_identical(wrapped.dataset, product)
    assert report.load_input(with_metadata=True).label == 'Input'

    report.plot_1d_images(product, 'profile', ())
    report.plot_2d_images(product, 'image', ())
    content = '\n'.join(report._latex_content)
    assert r'after\_gain' in content
    assert '0.25s' in content
    product.to_netcdf(tmp_path / 'export.nc')
    with xr.open_dataset(tmp_path / 'export.nc') as exported:
        xr.testing.assert_identical(exported, original)
    product.close()
    input_data.close()
    wrapped.dataset.close()


def test_comparison_accepts_plain_datasets_without_a_manifest_entry(tmp_path):
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text('{}')
    report = Report(manifest)
    reference = xr.Dataset({'count': ('x', np.array([1], dtype=np.uint32))})
    current = xr.Dataset({'count': ('x', np.array([0], dtype=np.uint32))})
    report.plot_run_comparison(reference, current)
    assert 'Mean Abs Err = 1.00e+00' in '\n'.join(report._latex_content)
    xr.testing.assert_identical(reference, xr.Dataset({'count': ('x', np.array([1], dtype=np.uint32))}))
    with pytest.raises(TypeError, match='xarray.Dataset'):
        report.plot_1d_images(np.ones(3), 'count', ())
