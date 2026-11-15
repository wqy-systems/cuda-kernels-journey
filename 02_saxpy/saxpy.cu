// 02 · SAXPY: y = a * x + y
//
// 和 01 vector_add 的三处关键不同，看懂这三条你就懂了：
//   1. kernel 多一个标量参数 a（常数，不进显存，直接当参数传）
//   2. y 是「读进来又写回去」，所以它是普通指针，不能加 const
//   3. 校验之前必须先把 y 的原始值留一份副本（h_ref），
//      否则 kernel 已经把 y 改掉了，参考值算出来必然是错的

#include <cstdio>
#include <cstdlib>
#include "common.h"

constexpr int N = 1 << 24;          // 16,777,216 个元素，三个数组共 192MB
constexpr int BLOCK_SIZE = 256;
constexpr int GRID_SIZE = 216;     // 故意不铺满，逼 kernel 用 grid-stride

// ---------------------------------------------------------------------------
// kernel：和 vector_add 只差循环体那一行
// ---------------------------------------------------------------------------
__global__ void saxpyKernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            float a,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 我是全队第几号
    int stride = blockDim.x * gridDim.x;               // 一趟能派出去多少人
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];                        // 唯一的区别就在这一行
    }
}

// CPU 参考实现
void saxpyCPU(const float* x, float* y, float a, int n) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

int main() {
    printDeviceInfo();

    const size_t bytes = sizeof(float) * N;
    const float a = 2.5f;

    // ---- 1. host 内存 ----
    // pinned 内存（cudaMallocHost）让 DMA 能直接搬运，H2D/D2H 带宽翻倍
    float *h_x, *h_y, *h_ref;
    CUDA_CHECK(cudaMallocHost(&h_x, bytes));
    CUDA_CHECK(cudaMallocHost(&h_y, bytes));
    CUDA_CHECK(cudaMallocHost(&h_ref, bytes));

    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i % 1000) * 0.001f;
        h_y[i] = static_cast<float>(i % 777) * 0.002f;
        h_ref[i] = h_y[i];      // ← 关键：抢在 kernel 之前把 y 的原始值存一份
    }

    // ---- 2. device 内存并拷贝 ----
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // ---- 3. 启动 kernel ----
    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);
    GpuTimer timer;

    // 热身一次：把 GPU 频率拉满，结果丢弃
    saxpyKernel<<<grid, block>>>(d_x, d_y, a, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 1e9f;
    for (int rep = 0; rep < 10; ++rep) { ... }   // 保持原样


    CUDA_CHECK(cudaGetLastError());

    // ---- 4. 拷回 ----
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // ---- 5. 校验 ----
    // 参考值用 h_ref（y 的原始副本），不能用 h_y（已被 kernel 改写）
    saxpyCPU(h_x, h_ref, a, N);
    float err = maxAbsError(h_ref, h_y, N);
    printf("\n[结果校验] 最大绝对误差 = %.3e  %s\n\n",
           err, err < 1e-4f ? "PASS" : "FAIL");

    // ---- 6. 性能 ----
    // 每个元素：读 x(4B) + 读 y(4B) + 写 y(4B) = 12 字节
    double total_bytes = 3.0 * N * sizeof(float);
    printf("[性能]\n");
    reportBandwidth("saxpy", total_bytes, ms);

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    double peak = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6;
    double achieved = total_bytes / (ms * 1e6);
    printf("\n  理论峰值 %.1f GB/s，实测 %.1f GB/s，达到 %.1f%%\n\n",
           peak, achieved, achieved / peak * 100);

    // ---- 7. 释放 ----
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));
    CUDA_CHECK(cudaFreeHost(h_ref));

    return 0;
}
