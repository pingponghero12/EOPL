#pragma once
#include "tensor.hpp"
#include <cstdint>

// CPU declaration
void apply_gain_1d_cpu(EOPLTensor<float>& img, const float* gain_data, size_t stride, size_t gain_size);
void apply_gain_1d_cpu(EOPLTensor<double>& img, const double* gain_data, size_t stride, size_t gain_size);
void apply_gain_1d_cpu(EOPLTensor<std::uint32_t>& img, const std::uint32_t* gain_data, size_t stride, size_t gain_size);

// CUDA declarations hidden behind macro
#ifdef EOPL_USE_CUDA
    void run_gain_kernel(float* dev_img, const float* dev_gain, size_t total_elements, size_t stride, size_t gain_size);
    void free_vram(size_t device_ptr_int);
#else
    inline void free_vram(size_t) {} 
#endif
