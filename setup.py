from setuptools import setup, find_packages

setup(
    name='eopl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'xarray',
        'netcdf4',  # Required for xarray to read/write .nc files
        'pyyaml',
        'matplotlib',
        'dask',
        'distributed',
        'zarr<3',
        'scipy'
    ]
)
