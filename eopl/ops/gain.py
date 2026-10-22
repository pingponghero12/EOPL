"""Compatibility imports; new code should use eopl.algorithms.gain."""
from ..algorithms.gain import (
    HAS_CPP_CORE,
    apply_gain_1d,
    apply_gain_1d_dask,
    apply_gain_1d_cpp,
)
