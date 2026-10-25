#include "apply_gain_1d.hpp"
#include "tensor.hpp"

template <typename T>
void apply_gain_1d_cpu_impl(EOPLTensor<T>& img, const T* gain_data, size_t stride, size_t gain_size) {
    size_t total_elements = img.ACT * img.ALT * img.Channels;
    for (size_t i = 0; i < total_elements; ++i) {
        // Dynamic dimension alignment math
        size_t gain_idx = (i / stride) % gain_size;
        img.data[i] *= gain_data[gain_idx];
    }
}

void apply_gain_1d_cpu(EOPLTensor<float>& img, const float* gain_data, size_t stride, size_t gain_size) {
    apply_gain_1d_cpu_impl(img, gain_data, stride, gain_size);
}

void apply_gain_1d_cpu(EOPLTensor<double>& img, const double* gain_data, size_t stride, size_t gain_size) {
    apply_gain_1d_cpu_impl(img, gain_data, stride, gain_size);
}

void apply_gain_1d_cpu(EOPLTensor<std::uint32_t>& img, const std::uint32_t* gain_data, size_t stride, size_t gain_size) {
    apply_gain_1d_cpu_impl(img, gain_data, stride, gain_size);
}
