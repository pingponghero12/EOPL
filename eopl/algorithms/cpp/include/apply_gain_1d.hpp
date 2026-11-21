#pragma once
#include "tensor.hpp"
#include <cstdint>

// CPU declaration
void apply_gain_1d_cpu(EOPLTensor<float>& img, const float* gain_data, size_t stride, size_t gain_size);
void apply_gain_1d_cpu(EOPLTensor<double>& img, const double* gain_data, size_t stride, size_t gain_size);
void apply_gain_1d_cpu(EOPLTensor<std::uint32_t>& img, const std::uint32_t* gain_data, size_t stride, size_t gain_size);
