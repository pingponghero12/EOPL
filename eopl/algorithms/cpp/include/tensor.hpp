#pragma once
#include <cstddef>

template <typename T>
struct EOPLTensor {
    T* data;        // Host array data
    size_t ACT;     // Across Track dimension
    size_t ALT;     // Along Track dimension
    size_t Channels; // Spectral channels
};
