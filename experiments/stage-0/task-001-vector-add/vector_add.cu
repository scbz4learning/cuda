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
                           int count, int* thread_indices,
                           int* thread_is_valid) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    thread_indices[thread_ordinal] = index;
    thread_is_valid[thread_ordinal] = index < count;

    if (index < count) {
        output[index] = left[index] + right[index];
    }
}

int main() {
    constexpr int count = 14;
    constexpr int block_size = 4;
    // 计算数组需要的空间
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    
    // 主机分配空间
    std::vector<float> host_left(count, 1.0f);
    std::vector<float> host_right(count, 2.0f);
    std::vector<float> host_output(count, 0.0f);

    // 设备上分配空间
    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_output = nullptr;
    int* device_thread_indices = nullptr;
    int* device_thread_is_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&device_left, bytes));
    CUDA_CHECK(cudaMalloc(&device_right, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));

    // 主机到设备的数据传输
    // 只传输了input, output 不需要传输
    // output 默认是脏数据, 但是不影响, 因为 a + b 会直接覆盖
    CUDA_CHECK(cudaMemcpy(device_left, host_left.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_right, host_right.data(), bytes, cudaMemcpyHostToDevice));

    // 分配线程划分
    int grid_size = (count + block_size - 1) / block_size;
    int total_threads = grid_size * block_size; // 这里分配的线程 = 16 个, 最后两个会越界, 所以核函数需要判断来保证正确性
    size_t thread_info_bytes = static_cast<size_t>(total_threads) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&device_thread_indices, thread_info_bytes));
    CUDA_CHECK(cudaMalloc(&device_thread_is_valid, thread_info_bytes));

    vector_add<<<grid_size, block_size>>>(device_left, device_right, device_output,
                                          count, device_thread_indices,
                                          device_thread_is_valid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    std::vector<int> host_thread_indices(total_threads);
    std::vector<int> host_thread_is_valid(total_threads);
    CUDA_CHECK(cudaMemcpy(host_thread_indices.data(), device_thread_indices,
                          thread_info_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_thread_is_valid.data(), device_thread_is_valid,
                          thread_info_bytes, cudaMemcpyDeviceToHost));

    std::printf("count=%d, block_size=%d, grid_size=%d, launched_threads=%d\n",
                count, block_size, grid_size, total_threads);
    std::printf("Thread mapping (printed in stable order):\n");
    for (int thread_ordinal = 0; thread_ordinal < total_threads; ++thread_ordinal) {
        int block_index = thread_ordinal / block_size;
        int thread_index = thread_ordinal % block_size;
        if (host_thread_is_valid[thread_ordinal]) {
            std::printf("  blockIdx.x=%d threadIdx.x=%d -> index=%d : valid, output=%.0f\n",
                        block_index, thread_index,
                        host_thread_indices[thread_ordinal],
                        host_output[host_thread_indices[thread_ordinal]]);
        } else {
            std::printf("  blockIdx.x=%d threadIdx.x=%d -> index=%d : out of range, skipped\n",
                        block_index, thread_index,
                        host_thread_indices[thread_ordinal]);
        }
    }

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
    CUDA_CHECK(cudaFree(device_thread_indices));
    CUDA_CHECK(cudaFree(device_thread_is_valid));

    std::printf("vector_add: %s\n", correct ? "PASS" : "FAIL");
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
