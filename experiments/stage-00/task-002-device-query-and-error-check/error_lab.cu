#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// 错误检查 lab。
//
// 本 task 只引入一件事：CUDA 错误在什么位置被报出、报出的是什么。
// 每个 case 必须单独跑一个进程，原因是 illegal 和 misalign 两个 case
// 会让整个进程的 CUDA 上下文进入不可用状态。run.sh 已按进程隔离调用。
//
// 用法：./error_lab <case>
//   case: launchcfg | dynsmem | baddevice | hugealloc | illegal | misalign
//
// 这里的 kernel 全部是故意写错的。不要修它们——它们是实验材料。

// 只用于实验准备阶段（建上下文、小额分配）。被观察的调用一律不用它。
#define CUDA_CHECK_LOCAL(call)                                             \
    do {                                                                   \
        cudaError_t error_ = (call);                                       \
        if (error_ != cudaSuccess) {                                       \
            std::fprintf(stderr, "setup failed at %s:%d: %s\n", __FILE__,  \
                         __LINE__, cudaGetErrorString(error_));            \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    } while (false)

static void show(const char* tag, cudaError_t err) {
    std::printf("  %-46s : %s\n", tag, cudaGetErrorString(err));
}

// 合法 kernel，用作对照。
__global__ void k_ok(float* p) {
    p[threadIdx.x] = static_cast<float>(threadIdx.x);
}

// 越界写。offset 单位是 float 元素，由 main 传入。
__global__ void k_illegal_write(float* p, long long offset) {
    p[offset + threadIdx.x] = 1.0f;
}

// 把 4 字节对齐的地址当作 16 字节对齐的 float4 读取。
__global__ void k_misaligned_read(float* p) {
    const float4 v = *reinterpret_cast<const float4*>(p + 1);
    p[0] = v.x + v.y + v.z + v.w;
}

// 需要动态共享内存的 kernel。
__global__ void k_dynamic_smem(float* p) {
    extern __shared__ float tile[];
    tile[threadIdx.x] = 1.0f;
    __syncthreads();
    p[0] = tile[0];
}

static int run_launchcfg() {
    CUDA_CHECK_LOCAL(cudaSetDevice(0));
    float* p = nullptr;
    CUDA_CHECK_LOCAL(cudaMalloc(&p, sizeof(float) * 64));

    k_ok<<<1, 64>>>(p);
    show("baseline valid launch: cudaGetLastError()", cudaGetLastError());
    k_ok<<<1, 0>>>(p);
    show("block dim = 0: cudaGetLastError()", cudaGetLastError());
    k_ok<<<0, 32>>>(p);
    show("grid dim = 0: cudaGetLastError()", cudaGetLastError());
    k_ok<<<1, 2048>>>(p);
    show("block dim = 2048 (> 1024): cudaGetLastError()", cudaGetLastError());
    k_ok<<<1, 64>>>(p);
    show("valid launch again: cudaGetLastError()", cudaGetLastError());
    show("cudaDeviceSynchronize()", cudaDeviceSynchronize());
    CUDA_CHECK_LOCAL(cudaFree(p));
    return EXIT_SUCCESS;
}

static int run_dynsmem() {
    CUDA_CHECK_LOCAL(cudaSetDevice(0));
    float* p = nullptr;
    CUDA_CHECK_LOCAL(cudaMalloc(&p, sizeof(float) * 64));
    const size_t smem = 64 * 1024;  // 超过静态共享内存的 48 KB 上限

    show("before launch (clean state)", cudaPeekAtLastError());
    k_dynamic_smem<<<1, 32, smem>>>(p);
    show("right after launch: cudaGetLastError()", cudaGetLastError());
    show("after that: cudaPeekAtLastError()", cudaPeekAtLastError());
    show("cudaDeviceSynchronize()", cudaDeviceSynchronize());

    // 放开上限后再试一次。阶段 03 讲 shared memory 时会再次用到这一步。
    show("cudaFuncSetAttribute(max dyn smem, 64KB)",
         cudaFuncSetAttribute(reinterpret_cast<const void*>(k_dynamic_smem),
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(smem)));
    k_dynamic_smem<<<1, 32, smem>>>(p);
    show("relaunch: cudaGetLastError()", cudaGetLastError());
    show("relaunch: cudaDeviceSynchronize()", cudaDeviceSynchronize());
    CUDA_CHECK_LOCAL(cudaFree(p));
    return EXIT_SUCCESS;
}

static int run_baddevice() {
    int count = 0;
    show("cudaGetDeviceCount", cudaGetDeviceCount(&count));
    std::printf("  (count = %d)\n", count);
    show("cudaSetDevice(99)", cudaSetDevice(99));
    show("cudaSetDevice(0) after the bad one", cudaSetDevice(0));
    show("cudaGetLastError() (does get clear it?)", cudaGetLastError());
    show("cudaGetLastError() again", cudaGetLastError());
    return EXIT_SUCCESS;
}

static int run_hugealloc() {
    void* p = nullptr;
    show("cudaMalloc(1 << 20)", cudaMalloc(&p, size_t(1) << 20));
    std::printf("  (small allocation succeeded, ptr=%p)\n", p);
    show("cudaMalloc(1 << 62)", cudaMalloc(&p, size_t(1) << 62));
    show("cudaGetLastError()", cudaGetLastError());
    float* small = nullptr;
    show("cudaMalloc(4096) after the failure", cudaMalloc(&small, 4096));
    if (p != nullptr) {
        CUDA_CHECK_LOCAL(cudaFree(p));
    }
    return EXIT_SUCCESS;
}

static int run_illegal() {
    // 只分配 1 KB，然后写到 512 MB 之外。越界距离必须远大于单个 2 MB
    // 映射页，否则写入可能落在同一页内而不触发故障——这一点也是本
    // case 要记录的观察。
    CUDA_CHECK_LOCAL(cudaSetDevice(0));
    float* p = nullptr;
    CUDA_CHECK_LOCAL(cudaMalloc(&p, 1024));
    const long long offset = 512LL * 1024 * 1024 / static_cast<long long>(sizeof(float));

    k_illegal_write<<<1, 32>>>(p, offset);
    show("right after launch: cudaGetLastError()", cudaGetLastError());
    show("cudaPeekAtLastError()", cudaPeekAtLastError());
    show("cudaDeviceSynchronize()", cudaDeviceSynchronize());

    std::printf("  --- context state after the fault ---\n");
    float* q = nullptr;
    show("cudaMalloc(4096)", cudaMalloc(&q, 4096));
    int count = 0;
    show("cudaGetDeviceCount", cudaGetDeviceCount(&count));
    show("cudaFree(p)", cudaFree(p));
    show("cudaDeviceReset()", cudaDeviceReset());
    show("cudaMalloc(4096) after reset", cudaMalloc(&q, 4096));
    return EXIT_SUCCESS;
}

static int run_misalign() {
    // cudaMalloc 保证至少 256 字节对齐，所以从 +1 个 float 处读
    // float4 一定不对齐。
    CUDA_CHECK_LOCAL(cudaSetDevice(0));
    float* p = nullptr;
    CUDA_CHECK_LOCAL(cudaMalloc(&p, 1024));
    std::printf("  pointer = %p (address mod 16 = %llu)\n", static_cast<void*>(p),
                reinterpret_cast<unsigned long long>(p) % 16ull);

    k_misaligned_read<<<1, 32>>>(p);
    show("right after launch: cudaGetLastError()", cudaGetLastError());
    show("cudaPeekAtLastError()", cudaPeekAtLastError());
    show("cudaDeviceSynchronize()", cudaDeviceSynchronize());

    float* q = nullptr;
    show("cudaMalloc(4096) after the fault", cudaMalloc(&q, 4096));
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf(
            "usage: ./error_lab <launchcfg|dynsmem|baddevice|hugealloc|illegal|misalign>\n");
        return EXIT_SUCCESS;
    }
    const char* which = argv[1];
    if (std::strcmp(which, "launchcfg") == 0) {
        return run_launchcfg();
    }
    if (std::strcmp(which, "dynsmem") == 0) {
        return run_dynsmem();
    }
    if (std::strcmp(which, "baddevice") == 0) {
        return run_baddevice();
    }
    if (std::strcmp(which, "hugealloc") == 0) {
        return run_hugealloc();
    }
    if (std::strcmp(which, "illegal") == 0) {
        return run_illegal();
    }
    if (std::strcmp(which, "misalign") == 0) {
        return run_misalign();
    }
    std::fprintf(stderr, "unknown case: %s\n", which);
    return EXIT_FAILURE;
}
