#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// 设备查询是所有后续实验的前提：不知道 SM 数、共享内存上限和峰值带宽，
// 就无法判断一个 kernel 能不能跑、跑得慢是因为访存还是因为算力。
//
// 本 task 只引入两件事：
//   1. cudaGetDeviceProperties 的查询与解释；
//   2. CUDA_CHECK 错误检查宏，以及“什么时候该检查”。
//
// 计时是 task-004 的内容，这里不要加计时逻辑。

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                       \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(error));            \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (false)

// TODO(学习者): 实现 print_device_info。
//
// 需要查询并打印下面这些字段（字段名见 README.md 第 1 步，或直接在
// /usr/local/cuda/include/driver_types.h 里搜 struct cudaDeviceProp）：
//
//   name, major, minor, multiProcessorCount, warpSize,
//   l2CacheSize, sharedMemPerBlock, sharedMemPerMultiprocessor,
//   regsPerBlock, regsPerMultiprocessor,
//   maxThreadsPerBlock, maxThreadsPerMultiProcessor,
//   memoryClockRate, memoryBusWidth, totalGlobalMem, clockRate,
//   asyncEngineCount, deviceOverlap
//
// 打印要求：
//   - 逐行打印，一个字段一行，格式自定但要能让人一眼看出字段名和值；
//   - 派生量（compute capability 字符串、峰值带宽、各类容量的 KiB/MiB/GiB
//     换算）一并打印，峰值带宽的公式由你自己推导，不要照抄；
//   - 不要打印原始结构体，用有意义的字段名。
//
// 峰值带宽的推导要求见 README.md 第 2 步：你要写清用了哪两个字段、
// 为什么需要它们，并用一个外部可查的规格数字校验你的公式。
static void print_device_info(int device) {
}

// 用一个最小 kernel 证明这台设备真的能跑，并演示 CUDA_CHECK 的位置。
// kernel 不做任何计算，只写入一个 0，作为“这台设备可用”的证据。
__global__ void probe_kernel(float* out) {
    out[0] = 0.0f;
}

int main(int argc, char** argv) {
    const int device = (argc > 1) ? std::atoi(argv[1]) : 0;

    int count = 0;
    // 故意不加 CUDA_CHECK：这是本 task 要讨论的情况之一。
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    std::printf("visible devices: %d\n", count);
    if (device < 0 || device >= count) {
        std::fprintf(stderr, "device index %d out of range [0, %d)\n", device, count);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(device));
    print_device_info(device);

    // 最小可用性验证：分配、启动、取值、释放。
    // 启动语句之后的第一个 CUDA_CHECK 检查的是“启动配置是否被接受”，
    // 不是 kernel 内部是否出错。这两者的区别见 README.md 第 3 步。
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, sizeof(float)));
    CUDA_CHECK(cudaMemset(out, -1.0f, sizeof(float)));
    probe_kernel<<<1, 1>>>(out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float value = -1.0f;
    CUDA_CHECK(cudaMemcpy(&value, out, sizeof(float), cudaMemcpyDeviceToHost));
    if (value != 0.0f) {
        std::fprintf(stderr, "probe_kernel did not write 0.0f, got %f\n", value);
        return EXIT_FAILURE;
    }
    std::printf("\nprobe: device %d is usable\n", device);
    CUDA_CHECK(cudaFree(out));
    return EXIT_SUCCESS;
}
