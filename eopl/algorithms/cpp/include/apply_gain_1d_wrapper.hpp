#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

void apply_gain_1d_wrapper(py::array img, py::array gain, std::string apply_dimension);
