// 07 · 四路累加：治「排队做加法」
//
// 上一版分块 GEMM 只快 1.3 倍，因为真正的瓶颈不是搬数据，
// 而是这条链：
//     sum += As[ty][0]*Bs[0][tx];   ← 必须等它算完
//     sum += As[ty][1]*Bs[1][tx];   ← 才能算这个
//     sum += As[ty][2]*Bs[2][tx];   ← 再等
//   …… 每一步都依赖上一步的结果，CPU/GPU 只能一个一个来。
//
// 解法：开 4 个独立的累加器，同时推进 4 条链：
//     acc0 += 第 0 项；acc1 += 第 1 项；acc2 += 第 2 项；acc3 += 第 3 项
//     四条链互不依赖，硬件可以一起算。
//   （就像食堂开 4 个窗口，不用所有人挤一个）
//
// 最后把 4 个结果加起来就行。
//
// 编译：cmake --build build -j
// 运行：./build/bin/gemm_unroll

#include <cstdio>
#include <cstdlib>
#include "common.h"

constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// 版本 A：单累加器（对照组，和 06 一样）
// ---------------------------------------------------------------------------
__global__ void gemm1acc(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C, int n) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < n; t += TILE) {
        As[threadIdx.y][threadIdx.x] = A[row * n + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * n + col];
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }
    C[row * n + col] = acc;
}

// ---------------------------------------------------------------------------
// 版本 B：四路累加
//   注意 for 循环步长变成 4，一次写 4 行代码。
//   #pragma unroll 让编译器把循环完全展开，生成 4 条独立的指令流。
// ---------------------------------------------------------------------------
__global__ void gemm4acc(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C, int n) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int t = 0; t < n; t += TILE) {
        As[threadIdx.y][threadIdx.x] = A[row * n + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * n + col];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; k += 4) {
            int ty = threadIdx.y, tx = threadIdx.x;
            acc0 += As[ty][k    ] * Bs[k    ][tx];
            acc1 += As[ty][k + 1] * Bs[k + 1][tx];
            acc2 += As[ty][k + 2] * Bs[k + 2][tx];
            acc3 += As[ty][k + 3] * Bs[k + 3][tx];
        }

        __syncthreads();
    }

    // 四条链的结果合起来（这一步的顺序无所谓，误差很小）
    C[row * n + col] = (acc0 + acc1) + (acc2 + acc3);
}

float cpuDot(const float* A, const float* B, int row, int col, int n) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) s += (double)A[row * n + k] * B[k * n + col];
    return (float)s;
}

int main() {
    printDeviceInfo();
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    double peak = p.multiProcessorCount * 128.0 * 2.0 * (p.clockRate / 1.0e6) * 1e9;
    printf("算力峰值 : %.1f TFLOPS\n\n", peak / 1e12);

    const int n = 1024;                 // 只测一个尺寸，省时间
    size_t one = sizeof(float) * (size_t)n * n;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, one));
    CUDA_CHECK(cudaMalloc(&d_B, one));
    CUDA_CHECK(cudaMalloc(&d_C, one));

    float *h_A = (float*)malloc(one), *h_B = (float*)malloc(one), *h_C = (float*)malloc(one);
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        h_A[i] = (float)((int)(i % 100) - 50) * 0.01f;   // int 转换，别再下溢
        h_B[i] = (float)((int)(i % 73) - 36) * 0.02f;
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_A, one, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, one, cudaMemcpyHostToDevice));

    using Kern = void (*)(const float*, const float*, float*, int);
    Kern kernels[2] = {gemm1acc, gemm4acc};
    const char* names[2] = {"一个累加器", "四个累加器"};
    float best[2] = {1e9f, 1e9f}, err[2] = {0.f, 0.f};

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    GpuTimer timer;
    for (int r = 0; r < 10; ++r) {          // 交错测量：两个版本轮流跑
        for (int v = 0; v < 2; ++v) {
            if (r == 0) CUDA_CHECK(cudaMemset(d_C, 0, one));   // 第一次前清零
            timer.start();
            kernels[v]<<<grid, block>>>(d_A, d_B, d_C, n);
            float ms = timer.stop();
            if (ms < best[v]) best[v] = ms;
        }
    }
    CUDA_CHECK(cudaGetLastError());

    // 校验：每个版本单独跑一次再读，不能共用最后一轮的结果
    for (int v = 0; v < 2; ++v) {
        kernels[v]<<<grid, block>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, one, cudaMemcpyDeviceToHost));
        for (int t = 0; t < 32; ++t) {
            int row = (t * 37 + 11) % n, col = (t * 53 + 7) % n;
            float d = fabsf(cpuDot(h_A, h_B, row, col, n) - h_C[row * n + col]);
            if (d > err[v]) err[v] = d;
        }
    }

    printf("================================================\n");
    printf("  %-12s %10s %12s %10s %12s\n", "版本", "耗时(ms)", "GFLOP/s", "占峰值", "校验");
    printf("  ------------------------------------------------\n");
    for (int v = 0; v < 2; ++v) {
        double g = 2.0 * n * n * n / (best[v] * 1e6);
        printf("  %-12s %10.3f %12.1f %9.2f%% %12.0e\n",
               names[v], best[v], g, g * 1e9 / peak * 100, err[v]);
    }
    printf("  ------------------------------------------------\n");
    printf("  四路提速  %.2f 倍\n", best[0] / best[1]);
    printf("================================================\n\n");

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    return 0;
}
