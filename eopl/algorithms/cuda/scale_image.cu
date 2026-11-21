extern "C" __global__ void scale_image(
    const float* input, float* output, unsigned long long size, float factor)
{
    // Each thread handles one pixel in the flattened image.
    auto pixel = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel < size) {
        output[pixel] = input[pixel] * factor;
    }
}
