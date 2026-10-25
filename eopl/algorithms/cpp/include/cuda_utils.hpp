#pragma once

#ifdef EOPL_USE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cuda_utils {

    inline void check(cudaError_t result) {
        if (result != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(result));
        }
    }

    template<typename T>
    T* alloc(size_t num_elements) {
        T* dev_ptr = nullptr;
        check(cudaMalloc(&dev_ptr, num_elements * sizeof(T)));
        return dev_ptr;
    }

    template<typename T>
    void copy_to_device(T* dev_ptr, const T* host_ptr, size_t num_elements) {
        check(cudaMemcpy(dev_ptr, host_ptr, num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void copy_to_host(T* host_ptr, const T* dev_ptr, size_t num_elements) {
        check(cudaMemcpy(host_ptr, dev_ptr, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void free(T* dev_ptr) {
        if (dev_ptr) check(cudaFree(dev_ptr));
    }
}
#endif
