// 09 · 我写的第一个 CUDA kernel
//
// kernel 部分由王庆尧于 2026-11-16 独立写出（原稿只有下面那五行）。
// 其余的 main 是脚手架，用来让 kernel 能真的跑起来、能验证结果。
//
// 编译：cmake --build build -j
// 运行：./build/bin/my_first_kernel

#include <cstdio>
#include <cstdlib>
#include "common.h"

constexpr int N = 1 << 22;      // 4M 个元素，16MB
constexpr int BLOCK = 256;
constexpr int GRID = 1024;

// ===========================================================================
// ↓↓↓ 这是你自己写的部分（只改了一个拼写错误 threaIdx → threadIdx）↓↓↓
// ===========================================================================
__global__ void addOne(const float* __restrict__ in,
                       float* __restrict__ out,
                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = in[i] + 1.0f;
    }
}
// ===========================================================================
// ↑↑↑ 上面五行是你独立写出来的 ↑↑↑
// ===========================================================================

int main() {
    printDeviceInfo();

    const size_t bytes = sizeof(float) * N;

    float *h_in, *h_out, *h_ref;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));
    CUDA_CHECK(cudaMallocHost(&h_ref, bytes));

    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i % 1000) * 0.001f;
        h_ref[i] = h_in[i] + 1.0f;      // CPU 参考：每个元素加 1
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK);
    dim3 grid(GRID);

    // 热身 + 十次取最快（用你已经学会的测量方法）
    addOne<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float best = 1e9f;
    for (int r = 0; r < 10; ++r) {
        timer.start();
        addOne<<<grid, block>>>(d_in, d_out, N);
        float ms = timer.stop();
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    float err = maxAbsError(h_ref, h_out, N);

    printf("\n[结果校验] 最大绝对误差 = %.3e  %s\n", err, err == 0.0f ? "PASS" : "FAIL");
    printf("[性能] %zu MB in %.3f ms = %.1f GB/s\n\n",
           bytes / 1048576, best, (2.0 * bytes) / (best * 1e6));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFreeHost(h_ref));

    return 0;
}
