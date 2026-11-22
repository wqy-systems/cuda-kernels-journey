#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.h"

constexpr int BK = 16, NTHREADS = 256;

// ---------- 4×4 + float4（上一轮的赢家，当基准） ----------
constexpr int BM4 = 64, BN4 = 64, TM4 = 4, TN4 = 4;

__global__ void tt4v(const float* __restrict__ A, const float* __restrict__ B,
                     float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM4][BK];
    __shared__ float Bs[BK][BN4];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int rowC = blockIdx.y * BM4, colC = blockIdx.x * BN4;
    float acc[TM4][TN4] = {};

    constexpr int CA = BM4 * BK / 4, CB = BK * BN4 / 4;
    constexpr int AC = BK / 4, BC = BN4 / 4;

    for (int t = 0; t < K; t += BK) {
        for (int i = tid; i < CA; i += NTHREADS) {
            int r = i / AC, c = (i % AC) * 4;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&A[(rowC + r) * K + t + c]);
        }
        for (int i = tid; i < CB; i += NTHREADS) {
            int r = i / BC, c = (i % BC) * 4;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&B[(t + r) * N + colC + c]);
        }
        __syncthreads();

        int tr = threadIdx.y * TM4, tc = threadIdx.x * TN4;
        for (int k = 0; k < BK; ++k) {
            float a[TM4];
            for (int i = 0; i < TM4; ++i) a[i] = As[tr + i][k];
            float4 b = *reinterpret_cast<const float4*>(&Bs[k][tc]);
            float bv[TN4] = {b.x, b.y, b.z, b.w};
            for (int i = 0; i < TM4; ++i)
                for (int j = 0; j < TN4; ++j) acc[i][j] += a[i] * bv[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM4; ++i)
        for (int j = 0; j < TN4; ++j)
            C[(rowC + threadIdx.y * TM4 + i) * N + (colC + threadIdx.x * TN4 + j)] = acc[i][j];
}

// ---------- 8×8 + float4 ----------
constexpr int BM8 = 128, BN8 = 128, TM8 = 8, TN8 = 8;

__global__ void tt8v(const float* __restrict__ A, const float* __restrict__ B,
                     float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM8][BK];
    __shared__ float Bs[BK][BN8];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int rowC = blockIdx.y * BM8, colC = blockIdx.x * BN8;
    float acc[TM8][TN8] = {};

    constexpr int CA = BM8 * BK / 4, CB = BK * BN8 / 4;
    constexpr int AC = BK / 4, BC = BN8 / 4;

    for (int t = 0; t < K; t += BK) {
        for (int i = tid; i < CA; i += NTHREADS) {
            int r = i / AC, c = (i % AC) * 4;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&A[(rowC + r) * K + t + c]);
        }
        for (int i = tid; i < CB; i += NTHREADS) {
            int r = i / BC, c = (i % BC) * 4;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&B[(t + r) * N + colC + c]);
        }
        __syncthreads();

        int tr = threadIdx.y * TM8, tc = threadIdx.x * TN8;
        for (int k = 0; k < BK; ++k) {
            float a[TM8];
            for (int i = 0; i < TM8; ++i) a[i] = As[tr + i][k];      // 竖着，8 条
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tc]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][tc + 4]);
            float bv[TN8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            for (int i = 0; i < TM8; ++i)
                for (int j = 0; j < TN8; ++j) acc[i][j] += a[i] * bv[j];   // 64 次
        }
        __syncthreads();
    }
    for (int i = 0; i < TM8; ++i)
        for (int j = 0; j < TN8; ++j)
            C[(rowC + threadIdx.y * TM8 + i) * N + (colC + threadIdx.x * TN8 + j)] = acc[i][j];
}

float cpuDot(const float* A, const float* B, int row, int col, int K, int N) {
    double s = 0.0;
    for (int k = 0; k < K; ++k) s += (double)A[row * K + k] * B[k * N + col];
    return (float)s;
}

int main() {
    printDeviceInfo();
    int sizes[2] = {1024, 2048};

    for (int s = 0; s < 2; ++s) {
        int n = sizes[s];
        size_t one = sizeof(float) * (size_t)n * n;
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, one));
        CUDA_CHECK(cudaMalloc(&dB, one));
        CUDA_CHECK(cudaMalloc(&dC, one));
        float *hA = (float*)malloc(one), *hB = (float*)malloc(one), *hC = (float*)malloc(one);
        for (size_t i = 0; i < (size_t)n * n; ++i) {
            hA[i] = (float)((int)(i % 100) - 50) * 0.01f;
            hB[i] = (float)((int)(i % 73) - 36) * 0.02f;
        }
        CUDA_CHECK(cudaMemcpy(dA, hA, one, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, one, cudaMemcpyHostToDevice));

        dim3 blk(16, 16);
        dim3 g4((n + BM4 - 1) / BM4, (n + BM4 - 1) / BM4);
        dim3 g8((n + BM8 - 1) / BM8, (n + BM8 - 1) / BM8);

        float best[2] = {1e9f, 1e9f};
        for (int r = 0; r < 3; ++r) {
            tt4v<<<g4, blk>>>(dA, dB, dC, n, n, n);
            tt8v<<<g8, blk>>>(dA, dB, dC, n, n, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int rep = 0; rep < 6; ++rep) {
            { GpuTimer t; t.start(); tt4v<<<g4, blk>>>(dA, dB, dC, n, n, n);
              float m = t.stop(); if (m < best[0]) best[0] = m; }
            { GpuTimer t; t.start(); tt8v<<<g8, blk>>>(dA, dB, dC, n, n, n);
              float m = t.stop(); if (m < best[1]) best[1] = m; }
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hC, dC, one, cudaMemcpyDeviceToHost));
        float err = 0.0f;
        for (int t = 0; t < 32; ++t) {
            int row = (t * 37 + 11) % n, col = (t * 53 + 7) % n;
            float d = fabsf(cpuDot(hA, hB, row, col, n, n) - hC[row * n + col]);
            if (d > err) err = d;
        }

        double f1 = 2.0 * n * n * n / (best[0] * 1e6);
        double f2 = 2.0 * n * n * n / (best[1] * 1e6);
        printf("\n  === n = %d   (4x4: %d blocks, 8x8: %d blocks) ===\n",
               n, g4.x * g4.y, g8.x * g8.y);
        printf("  4x4+vec4 : %8.3f ms  %9.1f GFLOP/s\n", best[0], f1);
        printf("  8x8+vec4 : %8.3f ms  %9.1f GFLOP/s   %.2f 倍\n", best[1], f2, f2 / f1);
        printf("  校验误差 : %.0e\n", err);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }
    return 0;
}
