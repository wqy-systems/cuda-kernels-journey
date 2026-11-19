// 10 · 我写的第二个 CUDA kernel：saxpy
//
// kernel 部分由王庆尧于 2026-11-16 独立写出。
// 那五行是一个字母一个字母敲出来的，中间跟 gridDim 的那个 e 搏斗了五次，最后赢了。
// 其余的 main 是脚手架，用来让它能真的跑起来、能验证结果。
//
// 编译：cmake --build build -j
// 运行：./build/bin/my_saxpy

#include <cstdio>
#include <cstdlib>
#include "common.h"

constexpr int N = 1 << 22;      // 4M 个元素，16MB
constexpr int BLOCK = 256;
constexpr int GRID = 1024;
constexpr float A = 2.5f;

// ===========================================================================
// ↓↓↓ 这是你自己写的部分 ↓↓↓
// ===========================================================================
__global__ void saxpy(const float* __restrict__ x,
                      float* __restrict__ y,
                      float a,
                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}
// ===========================================================================
// ↑↑↑ 上面五行是你独立写出来的 ↑↑↑
// ===========================================================================

int main() {
    printDeviceInfo();

    const size_t bytes = sizeof(float) * N;

    float *h_x, *h_y, *h_ref;
    CUDA_CHECK(cudaMallocHost(&h_x, bytes));
    CUDA_CHECK(cudaMallocHost(&h_y, bytes));
    CUDA_CHECK(cudaMallocHost(&h_ref, bytes));

    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i % 1000) * 0.001f;
        h_y[i] = static_cast<float>(i % 777) * 0.002f;
        h_ref[i] = A * h_x[i] + h_y[i];     // CPU 参考：先算好，别等 kernel 改了 y 再算
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK);
    dim3 grid(GRID);

    // 热身 + 十次取最快
    saxpy<<<grid, block>>>(d_x, d_y, A, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float best = 1e9f;
    for (int r = 0; r < 10; ++r) {
        timer.start();
        saxpy<<<grid, block>>>(d_x, d_y, A, N);
        float ms = timer.stop();
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());

    // saxpy 是累加操作（y = a*x + y），跑 11 遍就等于加了 11 遍。
    // 计时已经跑完了，现在把 y 恢复成初始值，再单独跑一遍用来校验。
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    saxpy<<<grid, block>>>(d_x, d_y, A, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    float err = maxAbsError(h_ref, h_y, N);

    printf("\n[结果校验] 最大绝对误差 = %.3e  %s\n", err, err < 1e-4f ? "PASS" : "FAIL");
    printf("[性能] %zu MB in %.3f ms = %.1f GB/s\n\n",
           bytes / 1048576, best, (2.0 * bytes) / (best * 1e6));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));
    CUDA_CHECK(cudaFreeHost(h_ref));

    return 0;
}
