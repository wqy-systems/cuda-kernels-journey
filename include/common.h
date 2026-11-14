#pragma once
// 通用工具：错误检查 / GPU 计时 / 结果校验 / 设备信息
// 后面每个 kernel 都会用到，所以第一天就把它们写顺手。

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---- 错误检查 ----
// 每个 CUDA API 调用都用它包一层，出错时打印文件行号并退出。
// 不写这个的话，出错了你只会看到一堆莫名其妙的结果。
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "\n[CUDA ERROR] %s:%d\n  call: %s\n  reason: %s\n", \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---- GPU 计时 ----
// 用 cudaEvent 而不是 clock()/std::chrono：
// kernel 是异步启动的，CPU 侧的计时器测不到 kernel 真正跑完的时刻。
struct GpuTimer {
    cudaEvent_t start_ev, stop_ev;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_ev)); }
    // 阻塞等待 kernel 结束，返回毫秒
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        return ms;
    }
};

// ---- 结果校验 ----
// 返回 CPU 参考值与 GPU 结果的最大绝对误差。float 累加类 kernel 用它判断正确性。
inline float maxAbsError(const float* ref, const float* got, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(ref[i] - got[i]);
        if (d > m) m = d;
    }
    return m;
}

// ---- 带宽 / 算力报告 ----
// kernel 优化的第一步是知道"理论上限是多少"，否则你不知道还有多少空间。
inline void reportBandwidth(const char* name, double bytes, float ms) {
    double gbs = bytes / (ms * 1e6);  // bytes / (ms * 1e-3) / 1e9
    printf("  %-22s %8.3f ms   %8.1f GB/s\n", name, ms, gbs);
}

inline void reportGflops(const char* name, double flops, float ms) {
    double gflops = flops / (ms * 1e6);
    printf("  %-22s %8.3f ms   %8.1f GFLOP/s\n", name, ms, gflops);
}

// ---- 设备信息 ----
inline void printDeviceInfo() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    printf("========================================\n");
    printf("GPU          : %s\n", p.name);
    printf("Compute Cap  : %d.%d\n", p.major, p.minor);
    printf("SM 数量      : %d\n", p.multiProcessorCount);
    printf("显存         : %.1f GB\n", p.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("显存带宽     : %.1f GB/s (理论峰值)\n", 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6);
    printf("L2 Cache     : %d KB\n", p.l2CacheSize / 1024);
    printf("每 SM 共享内存: %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("每 SM 寄存器  : %d K\n", p.regsPerMultiprocessor / 1024);
    printf("最大线程/块   : %d\n", p.maxThreadsPerBlock);
    printf("========================================\n\n");
}
