// 01 · vector add —— 你的第一个 kernel
//
// 目的不是"算出结果"，而是建立三件事的直觉：
//   1. host 内存和 device 内存是两块地方，要显式搬运
//   2. kernel 是异步的，测时间必须用 cudaEvent 同步
//   3. 这个 kernel 是"纯 memory-bound"，性能上限由显存带宽决定
//
// 编译：make 01_vector_add/vector_add
// 运行：./01_vector_add/vector_add
// 分析：ncu --set full -o va ./01_vector_add/vector_add

#include <cstdio>
#include <cstdlib>
#include "common.h"

// N = 2^24 = 16,777,216 个 float
// 内存占用：a(64MB) + b(64MB) + c(64MB) = 192MB，8G 显存毫无压力
constexpr int N = 1 << 24;
constexpr int BLOCK_SIZE = 256;
constexpr int GRID_SIZE = 1024;  // 故意限制 grid 大小，逼你用 grid-stride loop

// ---------------------------------------------------------------------------
// kernel：为什么用 grid-stride loop 而不是 "if (idx < n) c[idx] = a[idx]+b[idx]"？
//
// 1. grid-stride 让 kernel 能处理任意长度输入，不依赖 grid 大小恰好覆盖 n
// 2. 显式指定 grid 大小可以稳定控制占用率（occupancy），便于做性能实验
// 3. 每次循环访问的地址间隔 = blockDim*gridDim，仍然是合并访问（coalesced），
//    不会破坏访存效率
// ---------------------------------------------------------------------------
__global__ void vectorAdd(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ c,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程编号
    int stride = blockDim.x * gridDim.x;               // 一轮跨越多少个线程
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// CPU 参考实现，用来校验 GPU 结果
void vectorAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

int main() {
    printDeviceInfo();

    const size_t bytes = sizeof(float) * N;

    // ---- 1. 分配 host 内存 ----
    // 用 cudaMallocHost 而不是 malloc：pinned 内存允许 DMA 直接搬运，
    // H2D/D2H 带宽通常能翻倍。后面做通信（GPUDirect）时这个概念会再次出现。
    float *h_a, *h_b, *h_c, *h_ref;
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));
    CUDA_CHECK(cudaMallocHost(&h_ref, bytes));

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i % 1000) * 0.001f;
        h_b[i] = static_cast<float>(i % 777) * 0.002f;
    }

    // ---- 2. 分配 device 内存并拷贝 ----
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    GpuTimer copy_timer;
    copy_timer.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    float copy_ms = copy_timer.stop();

    cudaDeviceSynchronize();

    // ---- 3. 启动 kernel ----
    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    GpuTimer kernel_timer;
    kernel_timer.start();
    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, N);
    float kernel_ms = kernel_timer.stop();

    CUDA_CHECK(cudaGetLastError());  // 启动失败（比如 grid 配置越界）会在这里暴露

    // ---- 4. 拷回并校验 ----
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    vectorAddCPU(h_a, h_b, h_ref, N);

    float err = maxAbsError(h_ref, h_c, N);
    printf("\n[结果校验] 最大绝对误差 = %.3e  %s\n\n", err, err < 1e-5f ? "PASS" : "FAIL");

    // ---- 5. 性能与理论上限对比 ----
    // vector add 每个元素读 2 个 float、写 1 个 float = 12 bytes
    double total_bytes = 3.0 * N * sizeof(float);
    printf("[性能]\n");
    reportBandwidth("vectorAdd", total_bytes, kernel_ms);
    reportBandwidth("H2D copy x2", 2.0 * bytes, copy_ms);

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    double peak = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6;
    double achieved = total_bytes / (kernel_ms * 1e6);
    printf("\n  理论峰值带宽 %.1f GB/s，实测 %.1f GB/s，达到 %.1f%%\n",
           peak, achieved, achieved / peak * 100);
    printf("  → 如果低于 70%%，用 ncu 看 DRAM throughput 与 memory 相关 stall\n\n");

    // ---- 6. 清理 ----
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaFreeHost(h_ref));

    return 0;
}
