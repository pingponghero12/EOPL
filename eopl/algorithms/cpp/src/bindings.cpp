#include <pybind11/pybind11.h>
#include "apply_gain_1d_wrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(eopl_cpp_core, m) {
    m.doc() = "EOPL C++ CPU operations";
    m.def("apply_gain_1d", &apply_gain_1d_wrapper,
          py::arg("img").noconvert(), py::arg("gain").noconvert(),
          py::arg("apply_dimension") = "ACT", "Apply 1D gain in place on CPU");
}
