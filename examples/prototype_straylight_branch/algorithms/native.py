"""Compile and load this example's local C++ CPU kernels."""
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import numpy as np

DOUBLE_PTR = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
SIZE = ctypes.c_size_t


@lru_cache(maxsize=1)
def native_library(source):
    source = Path(source)
    library = source.with_suffix(".so")
    if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
        subprocess.run([
            "c++", "-std=c++17", "-O2", "-shared", "-fPIC",
            str(source), "-o", str(library),
        ], check=True)
    return ctypes.CDLL(str(library.resolve()))


def double_array(values):
    return np.ascontiguousarray(values, dtype=np.float64)


def kernel(name, types, *args):
    function = getattr(native_library(Path(__file__).with_name("kernels.cpp")), name)
    function.argtypes = types
    function.restype = None
    function(*args)
