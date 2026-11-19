// 11 · 我写的第三个 CUDA kernel：reduce（归约求和）
//
// kernel 由王庆尧于 2026-11-17 分两段写出：
//   前半段：idx / stride / for 循环累加（他自己写的）
//   后半段：共享内存 + __syncthreads + 树形合并（他自己写的）
// 原稿里两个关键字少写了一条下划线，此处补全。
//
// 编译：cmake --build build -j
// 运行：./build/bin/my_reduce

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.h"

constexpr int N = 1 << 24;      // 16,777,216 个元素，64MB
constexpr int BLOCK = 256;      // 必须是 256，因为下面 sdata[256] 且从 128 开始合并
constexpr int GRID = 1024;

// ===========================================================================
// ↓↓↓ 这是你自己写的部分 ↓↓↓
// ===========================================================================
__global__ void reduce(const float* __restrict__ in,
                       float* __restrict__ out,
                       int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    // ---- 前半段：每个线程先把自己那段加起来 ----
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += in[i];
    }

    // ---- 后半段：把本 block 的 256 个部分和合成 1 个 ----
    sdata[tid] = sum;
    __syncthreads();

    for (int s = 128; s > 0; s /= 2) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}
// ===========================================================================
// ↑↑↑ 上面是你独立写出来的 ↑↑↑
// ===========================================================================

int main() {
    printDeviceInfo();

    const size_t bytes = sizeof(float) * N;
    printf("数据量 : %d 个 float = %.0f MB\n", N, bytes / 1048576.0);
    printf("线程数 : %d × %d = %d，每个线程加 %d 个数\n\n",
           GRID, BLOCK, GRID * BLOCK, N / (GRID * BLOCK));

    // ---- host ----
    float* h_in;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes));
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 97) * 0.01f;

    // CPU 参考值（用 double，避免 float 累加掉精度）
    double cpuSum = 0.0;
    for (int i = 0; i < N; ++i) cpuSum += h_in[i];
    printf("CPU 总和 : %.4f\n\n", cpuSum);

    // ---- device ----
    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(float) * GRID));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK);
    dim3 grid(GRID);

    // 热身 + 十次取最快
    reduce<<<grid, block>>>(d_in, d_partial, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float best = 1e9f;
    for (int r = 0; r < 10; ++r) {
        timer.start();
        reduce<<<grid, block>>>(d_in, d_partial, N);
        float ms = timer.stop();
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());

    // 把 1024 个 block 的部分和拷回来，在 CPU 上做最后一步合并
    float* h_partial = (float*)malloc(sizeof(float) * GRID);
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float) * GRID,
                          cudaMemcpyDeviceToHost));

    double gpuSum = 0.0;
    for (int i = 0; i < GRID; ++i) gpuSum += h_partial[i];

    double relErr = std::fabs(gpuSum - cpuSum) / cpuSum;

    printf("[结果] GPU 总和 = %.4f\n", gpuSum);
    printf("[校验] 相对误差 = %.3e  %s\n", relErr, relErr < 1e-5 ? "PASS" : "FAIL");
    printf("[性能] 读 %zu MB，%.3f ms，带宽 %.1f GB/s\n\n",
           bytes / 1048576, best, bytes / (best * 1e6));

    // 参考：同一份数据的三种归约写法（都在 04_reduce 里）
    printf("对比 04_reduce 的三个版本（同尺寸数据）：\n");
    printf("  原子加     219 GB/s\n");
    printf("  共享内存树 261 GB/s\n");
    printf("  你自己写的  %.1f GB/s ← 看这里\n\n", bytes / (best * 1e6));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFreeHost(h_in));
    free(h_partial);

    return 0;
}
