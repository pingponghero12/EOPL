#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

py::tuple apply_gain_1d_wrapper(
                                py::array img, py::array gain,
                                bool use_cuda, bool device_dirty, size_t device_ptr_in, 
                                std::string apply_dimension);
