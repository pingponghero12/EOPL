#pragma once
#include <cstddef>

template <typename T>
struct EOPLTensor {
    T* data;        // Level 6: raw pointer — host RAM or device vRAM
    size_t ACT;     // Across Track dimension
    size_t ALT;     // Along Track dimension
    size_t Channels; // Spectral channels
    bool is_device; // True when data resides in GPU vRAM
};
