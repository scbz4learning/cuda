#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

// 每个 block 处理一个 TILE x TILE 的 tile，使用 TILE x BLOCK_ROWS 个线程，
// 每个线程处理 TILE / BLOCK_ROWS 个元素（与上游
// resources/foundations/cuda-samples/cpp/6_Performance/transpose/transpose.cu
// 的线程组织一致）。
#define TILE 32
#define BLOCK_ROWS 16
#define BLOCK_THREADS (TILE * BLOCK_ROWS)

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

// 约定：in[y * width + x]，out[y * width + x]，转置后 out[y * width + x] == in[x * width + y]。
// 这个版本读合并、写不合并，是本 task 的 baseline。
__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out, int width, int height) {
    const int x = blockIdx.x * TILE + threadIdx.x;
    const int y = blockIdx.y * TILE + threadIdx.y;
    for (int i = 0; i < TILE; i += BLOCK_ROWS) {
        if (x < width && (y + i) < height) {
            out[(y + i) * width + x] = in[x * width + (y + i)];
        }
    }
}

// TODO(学习者): 实现共享内存分块转置。
// 要求见 README.md 第 2 步，摘要：
//   1. 动态共享内存，extern __shared__，行距由参数 pitch 决定；
//      行 r 列 c 的地址必须是 r * pitch + c，不能写死 TILE。
//   2. 阶段 1 读入输入 tile（行 [by*TILE, ...)，列 [bx*TILE, ...)），全局读要合并。
//   3. __syncthreads()。
//   4. 阶段 2 写回输出 tile（行 [bx*TILE, ...)，列 [by*TILE, ...)），全局写要合并；
//      从 shared 读元素时索引是转置过的，这一步的 bank 行为就是本 task 的分析对象。
//   5. 循环复用 tile 时，阶段 2 结束与下一轮阶段 1 开始之间也要同步。
// 保持函数体为空时，harness 的 tiled_pitch32 和 tiled_pitch33 都会 FAIL。
__global__ void transpose_tiled(const float* __restrict__ in,
                                float* __restrict__ out, int width, int height,
                                int pitch) {
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

int verify(const std::vector<float>& expected, const std::vector<float>& actual,
           int* mismatches, long long* first_bad) {
    *mismatches = 0;
    *first_bad = -1;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            if (*first_bad < 0) {
                *first_bad = static_cast<long long>(i);
            }
            ++(*mismatches);
        }
    }
    return *mismatches == 0;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int device = (argc > 2) ? std::atoi(argv[2]) : 0;

    if (n < TILE || n % TILE != 0) {
        std::fprintf(stderr, "N must be a positive multiple of %d, got %d\n", TILE, n);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const double peak_gb_per_s =
        2.0 * static_cast<double>(prop.memoryClockRate) * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;

    std::printf("N=%d device=%d %s (sm_%d%d) SMs=%d L2=%zu B peak(2*memClock*busWidth/8)=%.1f GB/s\n",
                n, device, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                static_cast<size_t>(prop.l2CacheSize), peak_gb_per_s);
    std::printf("warmup=%d reps=%d block=dim3(%d,%d) grid=dim3(%d,%d)\n\n", WARMUP, REPS, TILE,
                BLOCK_ROWS, n / TILE, n / TILE);

    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = count * sizeof(float);

    // 0.5 的整数倍，保证精确比较有效；不要引入 NaN。
    std::vector<float> host_in(count);
    for (size_t i = 0; i < count; ++i) {
        host_in[i] = static_cast<float>((i * 2654435761ull) % 1999) * 0.5f;
    }
    std::vector<float> host_expected(count);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            host_expected[static_cast<size_t>(y) * n + x] =
                host_in[static_cast<size_t>(x) * n + y];
        }
    }
    std::vector<float> host_actual(count, -1.0f);

    float* device_in = nullptr;
    float* device_out = nullptr;
    CUDA_CHECK(cudaMalloc(&device_in, bytes));
    CUDA_CHECK(cudaMalloc(&device_out, bytes));
    CUDA_CHECK(cudaMemcpy(device_in, host_in.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE, BLOCK_ROWS);
    const dim3 grid(n / TILE, n / TILE);

    std::vector<Variant> variants;
    variants.push_back({"naive", [&] { transpose_naive<<<grid, block>>>(device_in, device_out, n, n); }});
    variants.push_back({"tiled_pitch32", [&] {
                          transpose_tiled<<<grid, block, TILE * TILE * sizeof(float)>>>(
                              device_in, device_out, n, n, TILE);
                      }});
    variants.push_back({"tiled_pitch33", [&] {
                          transpose_tiled<<<grid, block, TILE * (TILE + 1) * sizeof(float)>>>(
                              device_in, device_out, n, n, TILE + 1);
                      }});

    std::printf("%-15s %-6s %10s %12s %10s %12s\n", "variant", "result", "min_us",
                "median_us", "mean_us", "GB/s");
    for (const Variant& variant : variants) {
        // 先清零，让 FAIL 时的输出可复现。
        CUDA_CHECK(cudaMemset(device_out, 0, bytes));
        variant.launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(host_actual.data(), device_out, bytes, cudaMemcpyDeviceToHost));
        int mismatches = 0;
        long long first_bad = -1;
        const bool pass = verify(host_expected, host_actual, &mismatches, &first_bad);

        const Timing timing = time_kernel(variant.launch);
        const double moved_bytes = 2.0 * static_cast<double>(bytes);
        const double gb_per_s = moved_bytes / (timing.median_us * 1e-6) / 1e9;

        std::printf("%-15s %-6s %10.2f %12.2f %10.2f %12.1f\n", variant.name,
                    pass ? "PASS" : "FAIL", timing.min_us, timing.median_us, timing.mean_us,
                    gb_per_s);
        if (!pass) {
            std::printf("%-15s mismatches=%d/%zu first_bad_index=%lld (y=%lld x=%lld)\n", "",
                        mismatches, count, first_bad, first_bad / n, first_bad % n);
        }
    }
    std::printf("\nmoved bytes per launch = 2 * %zu * %zu * sizeof(float) = %.1f MiB\n", count,
                count, 2.0 * static_cast<double>(bytes) / (1024.0 * 1024.0));

    CUDA_CHECK(cudaFree(device_in));
    CUDA_CHECK(cudaFree(device_out));
    return EXIT_SUCCESS;
}
