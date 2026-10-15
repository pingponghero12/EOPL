#include "apply_gain_1d.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>

__global__ void apply_gain_1d_kernel(float *img_data, const float *gain_data,
                                     size_t total_elements, size_t stride,
                                     size_t gain_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    // Dynamic dimension alignment math
    size_t gain_idx = (idx / stride) % gain_size;
    img_data[idx] *= gain_data[gain_idx];
  }
}

void run_gain_kernel(float *dev_img, const float *dev_gain,
                     size_t total_elements, size_t stride, size_t gain_size) {
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;
  apply_gain_1d_kernel<<<blocks, threads>>>(dev_img, dev_gain, total_elements,
                                            stride, gain_size);

  // 1. Catch synchronous errors (e.g., invalid grid/block dimensions)
  cuda_utils::check(cudaGetLastError());

  // 2. Catch asynchronous errors (e.g., architecture mismatch, segfaults)
  cuda_utils::check(cudaDeviceSynchronize());
}

void free_vram(size_t device_ptr_int) {
  float *dev_ptr = reinterpret_cast<float *>(device_ptr_int);
  cuda_utils::free(dev_ptr);
}
