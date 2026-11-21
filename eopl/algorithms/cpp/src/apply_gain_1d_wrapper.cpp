#include "apply_gain_1d_wrapper.hpp"
#include "apply_gain_1d.hpp"
#include "tensor.hpp"
#include <stdexcept>

void apply_gain_1d_wrapper(py::array img, py::array gain, std::string apply_dimension) {
    py::buffer_info img_info = img.request();
    py::buffer_info gain_info = gain.request();

    if (img_info.ndim != 3) {
        throw std::invalid_argument("img must be a 3-D [Channels, ALT, ACT] array");
    }
    if (gain_info.ndim != 1) {
        throw std::invalid_argument("gain must be a 1-D array");
    }
    if (!img.writeable()) {
        throw std::invalid_argument("img must be writable");
    }
    if (!(img.flags() & py::array::c_style) || !(gain.flags() & py::array::c_style)) {
        throw std::invalid_argument("img and gain must be C-contiguous");
    }
    if (img_info.format != gain_info.format || img_info.itemsize != gain_info.itemsize) {
        throw std::invalid_argument("img and gain must have the same dtype");
    }

    size_t Channels = img_info.shape[0];
    size_t ALT = img_info.shape[1];
    size_t ACT = img_info.shape[2];

    // Calculate memory stride based on the target dimension
    size_t stride = 1;
    size_t gain_size = gain_info.shape[0];

    // Strides calculated for the [Channels, ALT, ACT] layout
    if (apply_dimension == "ACT") {
        stride = 1;                 // ACT is contiguous in memory
    } else if (apply_dimension == "ALT") {
        stride = ACT;               // Skip a row to move down
    } else if (apply_dimension == "Channels" || apply_dimension == "band") {
        stride = ALT * ACT;         // Skip a whole image plane to change bands
    } else {
        throw std::invalid_argument("Unsupported apply_dimension: " + apply_dimension);
    }

    size_t expected_gain_size = apply_dimension == "ACT" ? ACT :
                                apply_dimension == "ALT" ? ALT : Channels;
    if (gain_size != expected_gain_size) {
        throw std::invalid_argument("gain length does not match apply_dimension");
    }

    if (img_info.item_type_is_equivalent_to<float>()) {
        EOPLTensor<float> tensor = {
            static_cast<float*>(img_info.ptr), ACT, ALT, Channels
        };
        apply_gain_1d_cpu(
            tensor, static_cast<float*>(gain_info.ptr), stride, gain_size
        );
    } else if (img_info.item_type_is_equivalent_to<double>()) {
        EOPLTensor<double> tensor = {
            static_cast<double*>(img_info.ptr), ACT, ALT, Channels
        };
        apply_gain_1d_cpu(
            tensor, static_cast<double*>(gain_info.ptr), stride, gain_size
        );
    } else if (img_info.item_type_is_equivalent_to<std::uint32_t>()) {
        // uint32 may use either unsigned int or unsigned long on Windows.
        EOPLTensor<std::uint32_t> tensor = {
            static_cast<std::uint32_t*>(img_info.ptr), ACT, ALT, Channels
        };
        apply_gain_1d_cpu(
            tensor, static_cast<std::uint32_t*>(gain_info.ptr), stride, gain_size
        );
    } else {
        throw std::invalid_argument(
            "CPU execution supports float32, float64, and uint32 arrays"
        );
    }
}
