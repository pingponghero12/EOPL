#include <pybind11/pybind11.h>
#include "apply_gain_1d_wrapper.hpp"
#include "apply_gain_1d.hpp" // Included purely so it knows about free_vram

namespace py = pybind11;

PYBIND11_MODULE(eopl_cpp_core, m) {
    m.doc() = "EOPL C++/CUDA Core Backend";

    // Bind the 1D gain operation
    m.def("apply_gain_1d", &apply_gain_1d_wrapper, 
          py::arg("img").noconvert(), py::arg("gain").noconvert(), py::arg("use_cuda"),
          py::arg("device_dirty"), py::arg("device_ptr_in"), py::arg("apply_dimension") = "ACT",
          "Apply 1D Gain (CPU/CUDA)");
          
    // Bind memory management
    m.def("free_vram", &free_vram, "Free GPU memory given an integer pointer");
    
    // Expose CUDA availability to Python
#ifdef EOPL_USE_CUDA
    m.attr("HAS_CUDA_COMPILED") = true;
#else
    m.attr("HAS_CUDA_COMPILED") = false;
#endif
}
