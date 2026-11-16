// 08 · Tensor Core GEMM：换机器
//
// 前面用 CUDA 核心（通用工人）写 GEMM，最好只到 9.3% 的 FP32 算力峰值。
// 这一版换成 Tensor Core（矩阵乘专用机器），一条指令算完一整块 16×16 的乘加。
//
// 核心就三句：
//   wmma::load_matrix_sync(a_frag, ...);   // 把 A 的一小块装进 fragment
//   wmma::load_matrix_sync(b_frag, ...);   // 把 B 的一小块装进 fragment
//   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // 一块乘加，一次完成
//
// 代价：输入是 FP16（半精度），累加用 FP32。
//
// 编译：cmake --build build -j
// 运行：./build/bin/gemm_wmma

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>
#include "common.h"

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// 把 float 转成 half（Tensor Core 只吃 half）
__global__ void toHalf(const float* __restrict__ in, __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// ---------------------------------------------------------------------------
// Tensor Core GEMM
//   一个 warp（32 线程）负责算 C 里的一个 16×16 小块
//   整个 block 就 32 个线程 —— 不需要 256 个线程搬砖了
// ---------------------------------------------------------------------------
__global__ void gemmWmma(const __half* __restrict__ A,
                         const __half* __restrict__ B,
                         float* __restrict__ C,
                         int n) {
    int row = blockIdx.y * WMMA_M;   // 这一块 C 的起始行
    int col = blockIdx.x * WMMA_N;   // 这一块 C 的起始列

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);   // 累加器清零（在寄存器里）

    // 沿 K 维一块一块往前推
    for (int k = 0; k < n; k += WMMA_K) {
        // ldm = n：因为矩阵是按行连续存的，行与行之间隔 n 个元素
        wmma::load_matrix_sync(a_frag, A + row * n + k, n);
        wmma::load_matrix_sync(b_frag, B + k * n + col, n);

        // 这一句 = 16×16×16 = 4096 次乘加，Tensor Core 一条指令做完
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + row * n + col, c_frag, n, wmma::mem_row_major);
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
    double fp32peak = p.multiProcessorCount * 128.0 * 2.0 * (p.clockRate / 1.0e6) * 1e9;
    printf("CUDA 核心 FP32 峰值 : %.1f TFLOPS\n", fp32peak / 1e12);
    printf("对比基准：分块 FP32 版本 1864 GFLOP/s（占 FP32 峰值 9.3%%）\n\n");

    const int n = 1024;
    size_t bytesF = sizeof(float) * (size_t)n * n;
    size_t bytesH = sizeof(__half) * (size_t)n * n;

    float *h_A = (float*)malloc(bytesF), *h_B = (float*)malloc(bytesF), *h_C = (float*)malloc(bytesF);
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        h_A[i] = (float)((int)(i % 100) - 50) * 0.01f;
        h_B[i] = (float)((int)(i % 73) - 36) * 0.02f;
    }

    float *d_Af, *d_Bf;
    __half *d_Ah, *d_Bh;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_Af, bytesF));
    CUDA_CHECK(cudaMalloc(&d_Bf, bytesF));
    CUDA_CHECK(cudaMalloc(&d_Ah, bytesH));
    CUDA_CHECK(cudaMalloc(&d_Bh, bytesH));
    CUDA_CHECK(cudaMalloc(&d_C, bytesF));
    CUDA_CHECK(cudaMemcpy(d_Af, h_A, bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bf, h_B, bytesF, cudaMemcpyHostToDevice));

    // float -> half 转换
    int threads = 256;
    int blocks = (n * n + threads - 1) / threads;
    toHalf<<<blocks, threads>>>(d_Af, d_Ah, n * n);
    toHalf<<<blocks, threads>>>(d_Bf, d_Bh, n * n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 一个 warp 算一个 16×16 块
    dim3 block(32);
    dim3 grid(n / WMMA_N, n / WMMA_M);

    // 热身
    gemmWmma<<<grid, block>>>(d_Ah, d_Bh, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float best = 1e9f;
    for (int r = 0; r < 10; ++r) {
        timer.start();
        gemmWmma<<<grid, block>>>(d_Ah, d_Bh, d_C, n);
        float ms = timer.stop();
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesF, cudaMemcpyDeviceToHost));
    float maxerr = 0.0f;
    for (int t = 0; t < 64; ++t) {
        int row = (t * 37 + 11) % n, col = (t * 53 + 7) % n;
        float d = fabsf(cpuDot(h_A, h_B, row, col, n) - h_C[row * n + col]);
        if (d > maxerr) maxerr = d;
    }

    double gflops = 2.0 * n * n * n / (best * 1e6);
    printf("================================================\n");
    printf("  Tensor Core (wmma, FP16 in / FP32 acc)\n");
    printf("  耗时       : %.4f ms\n", best);
    printf("  算力       : %.1f GFLOP/s\n", gflops);
    printf("  对比 FP32 分块版 : %.2f 倍\n", gflops / 1864.1);
    printf("  校验误差   : %.3e  %s（FP16 精度低，1e-2 内算对）\n",
           maxerr, maxerr < 2e-2f ? "PASS" : "FAIL");
    printf("================================================\n\n");

    CUDA_CHECK(cudaFree(d_Af)); CUDA_CHECK(cudaFree(d_Bf));
    CUDA_CHECK(cudaFree(d_Ah)); CUDA_CHECK(cudaFree(d_Bh));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    return 0;
}
