// 06 · 分块 GEMM：用共享内存提高数据复用
//
// 朴素版的问题：每个线程算 C 的一个元素，都要把 A 的一行、B 的一列
// 从全局内存读一遍。A 的每个元素被读 N 次，B 的每个元素被读 M 次。
// 算术强度固定 0.25，比硬件平衡点 75 差 300 倍。
//
// 分块版的解法：
//   把 K 维切成一段一段（每段 TILE=16 个数），
//   每一轮：
//     1. 一个 block 的 256 个线程合作，把 A 的 16×16 一块和 B 的 16×16 一块
//        从全局内存搬进共享内存（只搬一次）
//     2. 这 256 个线程各自算自己那个 C 元素在这一轮的 16 次乘加，
//        全部从共享内存读
//     3. 同步，进入下一轮
//
//   全局内存的读取量降到原来的 1/TILE，算术强度从 0.25 提到 4。
//   再配合寄存器累加（acc 留在寄存器里，不每轮写回显存），实际复用更高。
//
// 编译：cmake --build build -j
// 运行：./build/bin/gemm_tiled

#include <cstdio>
#include <cstdlib>
#include "common.h"

constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// 分块 GEMM
//   一个 block = 16×16 = 256 个线程，负责算 C 里的一块 16×16
//   每个线程算其中一个元素，用 acc 在寄存器里跨轮累加
// ---------------------------------------------------------------------------
__global__ void gemmTiled(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    // 片上的两块“工作台”，16×16 个 float = 1KB
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // 这个线程负责的 C 元素坐标
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;   // 留在寄存器里，整趟 K 循环都不写回显存

    // 沿 K 维一段一段往前推
    for (int t = 0; t < K; t += TILE) {
        // 1. 合作搬砖：256 个线程，每人搬 A 的一个元素、B 的一个元素
        //    A 的这一块：行 = row，列 = t + threadIdx.x
        //    B 的这一块：行 = t + threadIdx.y，列 = col
        As[threadIdx.y][threadIdx.x] = A[row * K + (t + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];

        __syncthreads();   // 等这一块的 256 个元素都搬完

        // 2. 16 次乘加，全部从共享内存读
        //    这一轮 A 块被 16 个线程用，B 块也被 16 个线程用 —— 这就是复用
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();   // 等所有人算完，才能覆盖下一轮的共享内存
    }

    C[row * N + col] = acc;   // 循环结束后只写一次
}

// ---------------------------------------------------------------------------
// 朴素版（对照组，和 05_gemm 里一样）
// ---------------------------------------------------------------------------
__global__ void gemmNaive(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

float cpuDot(const float* A, const float* B, int row, int col, int K, int N) {
    double s = 0.0;
    for (int k = 0; k < K; ++k) s += (double)A[row * K + k] * B[k * N + col];
    return (float)s;
}

int main() {
    printDeviceInfo();

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    const double peakFLOPS = p.multiProcessorCount * 128.0 * 2.0 * (p.clockRate / 1.0e6) * 1e9;
    printf("算力峰值 : %.1f TFLOPS\n\n", peakFLOPS / 1e12);

    // 只测 1024 和 2048：4096 的朴素版太慢，浪费整机时间
    const int sizes[] = {1024, 2048};
    using Kernel = void (*)(const float*, const float*, float*, int, int, int);
    Kernel kernels[2] = {gemmNaive, gemmTiled};
    const char* names[2] = {"朴素", "分块16"};

    printf("==========================================================================\n");
    printf("  %6s %8s %12s %12s %10s %10s\n",
           "尺寸", "版本", "耗时(ms)", "GFLOP/s", "占峰值", "相对提速");
    printf("  ------------------------------------------------------------------------\n");

    for (int s = 0; s < 2; ++s) {
        int n = sizes[s];
        size_t one = sizeof(float) * (size_t)n * n;

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, one));
        CUDA_CHECK(cudaMalloc(&d_B, one));
        CUDA_CHECK(cudaMalloc(&d_C, one));

        float *h_A = (float*)malloc(one), *h_B = (float*)malloc(one), *h_C = (float*)malloc(one);
        for (size_t i = 0; i < (size_t)n * n; ++i) {
            h_A[i] = (float)((int)(i % 100) - 50) * 0.01f;
            h_B[i] = (float)((int)(i % 73) - 36) * 0.02f;
        }
        CUDA_CHECK(cudaMemcpy(d_A, h_A, one, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, one, cudaMemcpyHostToDevice));

        float best[2] = {1e9f, 1e9f};
        float err[2] = {0.f, 0.f};

        for (int v = 0; v < 2; ++v) {
            // 分块版必须用 TILE×TILE 的 block；朴素版用 16×16 也刚好
            dim3 block(TILE, TILE);
            dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

            // 交错：两个版本轮流跑，避免温度差污染结果
            CUDA_CHECK(cudaMemset(d_C, 0, one));
            kernels[v]<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            GpuTimer timer;
            int reps = 5;
            for (int r = 0; r < reps; ++r) {
                timer.start();
                kernels[v]<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
                float ms = timer.stop();
                if (ms < best[v]) best[v] = ms;
            }
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(h_C, d_C, one, cudaMemcpyDeviceToHost));
            for (int t = 0; t < 32; ++t) {
                int row = (t * 37 + 11) % n, col = (t * 53 + 7) % n;
                float d = fabsf(cpuDot(h_A, h_B, row, col, n, n) - h_C[row * n + col]);
                if (d > err[v]) err[v] = d;
            }

            double gflops = 2.0 * n * n * n / (best[v] * 1e6);
            printf("  %6d %8s %12.3f %12.1f %9.2f%%  %s  校验 %.0e\n",
                   n, names[v], best[v], gflops, gflops * 1e9 / peakFLOPS * 100,
                   v == 0 ? "  —  " : "", err[v]);
        }

        printf("  %6d %8s %26s  %.2f 倍\n", n, "→", "", best[0] / best[1]);
        printf("  ------------------------------------------------------------------------\n");

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
        free(h_A); free(h_B); free(h_C);
    }

    printf("==========================================================================\n\n");
    printf("重点不是绝对数字，是那个「相对提速」。\n");
    printf("朴素版算术强度 0.25，分块 16 后变成 4（每元素少搬 16 倍）。\n");
    printf("但离平衡点 75 还差很远 —— 继续分块、寄存器 tiling、Tensor Core，才到 cuBLAS。\n");

    return 0;
}
