#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#define TILE 32
#define BLOCK_ROWS 16
#define BLOCK_THREADS (TILE * BLOCK_ROWS)
#define TILE_ELEMENTS (TILE * TILE)

#define WARMUP 5
#define REPS 20

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t error = (call);                                   \
        if (error != cudaSuccess) {                                   \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__,  \
                         __LINE__, cudaGetErrorString(error));        \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (false)

// 只碰 shared memory 的 bank conflict 探针，作用是把 transpose 里被 DRAM 带宽
// 掩盖掉的 shared 冲突代价单独放大出来。
//
// 两个 kernel 的唯一区别必须是 shared memory 的布局和索引方式；循环次数、线程数、
// grid、global 访存都必须相同。
//
// 验收条件（详见 README.md 第 5 步）：
//   1. probe_no_pad 中，同一 warp 的 32 个 lane 读同一个 bank 的 32 个不同地址；
//      probe_padded 中，同一 warp 的 32 个 lane 读 32 个不同 bank。
//   2. 索引必须随循环变量变化，否则编译器会把 load 提到循环外，测到的不是
//      shared memory 访问。acc 最终必须等于 iterations，否则说明循环被优化掉了。
//
// 签名约定：in 全部为 1.0f，tile 由 in 载入；out[global_tid] = acc。
// TODO(学习者): 实现下面两个 kernel。
__global__ void probe_no_pad(const float* __restrict__ in, float* __restrict__ out,
                             int iterations) {
}

__global__ void probe_padded(const float* __restrict__ in, float* __restrict__ out,
                             int iterations) {
}

struct Timing {
    double min_us;
    double median_us;
    double mean_us;
};

struct Variant {
    const char* name;
    std::function<void()> launch;
};

template <typename Launch>
Timing time_kernel(Launch launch) {
    for (int i = 0; i < WARMUP; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> samples;
    samples.reserve(REPS);
    for (int i = 0; i < REPS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        samples.push_back(static_cast<double>(elapsed_ms) * 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double total = 0.0;
    for (double sample : samples) {
        total += sample;
    }
    std::sort(samples.begin(), samples.end());

    Timing timing;
    timing.min_us = samples.front();
    timing.median_us = (REPS % 2 == 1)
                           ? samples[REPS / 2]
                           : 0.5 * (samples[REPS / 2 - 1] + samples[REPS / 2]);
    timing.mean_us = total / static_cast<double>(REPS);
    return timing;
}

int main(int argc, char** argv) {
    int blocks = (argc > 1) ? std::atoi(argv[1]) : 160;
    int iterations = (argc > 2) ? std::atoi(argv[2]) : 10000;
    int device = (argc > 3) ? std::atoi(argv[3]) : 0;

    if (blocks <= 0 || iterations <= 0) {
        std::fprintf(stderr, "blocks and iterations must be positive\n");
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("device=%d %s (sm_%d%d) SMs=%d\n", device, prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("blocks=%d block=dim3(%d,%d)=%d threads tile=%dx%d iterations=%d warmup=%d reps=%d\n",
                blocks, TILE, BLOCK_ROWS, BLOCK_THREADS, TILE, TILE, iterations, WARMUP, REPS);
    std::printf("shared loads per launch = %d * %d = %.3e\n\n", blocks * BLOCK_THREADS, iterations,
                static_cast<double>(blocks) * BLOCK_THREADS * iterations);

    const size_t in_count = static_cast<size_t>(blocks) * TILE_ELEMENTS;
    const size_t out_count = static_cast<size_t>(blocks) * BLOCK_THREADS;
    const double expected = static_cast<double>(out_count) * iterations;

    std::vector<float> host_in(in_count, 1.0f);
    std::vector<float> host_out(out_count, -1.0f);

    float* device_in = nullptr;
    float* device_out = nullptr;
    CUDA_CHECK(cudaMalloc(&device_in, in_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_out, out_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_in, host_in.data(), in_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block(TILE, BLOCK_ROWS);
    const dim3 grid(blocks, 1, 1);

    std::vector<Variant> variants;
    variants.push_back({"probe_no_pad",
                        [&] { probe_no_pad<<<grid, block>>>(device_in, device_out, iterations); }});
    variants.push_back({"probe_padded",
                        [&] { probe_padded<<<grid, block>>>(device_in, device_out, iterations); }});

    std::printf("%-15s %-6s %10s %12s %10s %14s\n", "variant", "result", "min_us", "median_us",
                "mean_us", "loads/thread");
    double median_no_pad = 0.0;
    double median_padded = 0.0;
    for (const Variant& variant : variants) {
        CUDA_CHECK(cudaMemset(device_out, 0, out_count * sizeof(float)));
        variant.launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(host_out.data(), device_out, out_count * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double total = 0.0;
        for (float value : host_out) {
            total += static_cast<double>(value);
        }
        const bool pass = std::fabs(total - expected) <= 1e-3 * expected;

        const Timing timing = time_kernel(variant.launch);
        std::printf("%-15s %-6s %10.2f %12.2f %10.2f %14d\n", variant.name, pass ? "PASS" : "FAIL",
                    timing.min_us, timing.median_us, timing.mean_us, iterations);
        if (!pass) {
            std::printf("%-15s acc_total=%.6e expected=%.6e (循环可能被优化掉了，或索引没按 1.0f 累加)\n",
                        "", total, expected);
        }
        if (std::strcmp(variant.name, "probe_no_pad") == 0) {
            median_no_pad = timing.median_us;
        } else {
            median_padded = timing.median_us;
        }
    }

    if (median_no_pad > 0.0 && median_padded > 0.0) {
        std::printf("\nmedian(probe_no_pad) / median(probe_padded) = %.2fx\n",
                    median_no_pad / median_padded);
    }

    CUDA_CHECK(cudaFree(device_in));
    CUDA_CHECK(cudaFree(device_out));
    return EXIT_SUCCESS;
}
