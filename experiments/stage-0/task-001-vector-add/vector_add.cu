#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(error));             \
            return EXIT_FAILURE;                                           \
        }                                                                    \
    } while (false)

__global__ void vector_add(const float* left, const float* right, float* output,
                           int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}

int main() {
    constexpr int count = 1 << 20;
    constexpr int block_size = 256;
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);

    std::vector<float> host_left(count, 1.0f);
    std::vector<float> host_right(count, 2.0f);
    std::vector<float> host_output(count, 0.0f);

    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_left, bytes));
    CUDA_CHECK(cudaMalloc(&device_right, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));

    CUDA_CHECK(cudaMemcpy(device_left, host_left.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_right, host_right.data(), bytes, cudaMemcpyHostToDevice));

    int grid_size = (count + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size>>>(device_left, device_right, device_output, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (float value : host_output) {
        if (value != 3.0f) {
            correct = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(device_left));
    CUDA_CHECK(cudaFree(device_right));
    CUDA_CHECK(cudaFree(device_output));

    std::printf("vector_add: %s\n", correct ? "PASS" : "FAIL");
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
