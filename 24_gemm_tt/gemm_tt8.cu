#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.h"

constexpr int BM = 128, BN = 128, BK = 16;
constexpr int TM = 8,  TN = 8;

__global__ void gemmTT8(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    int rowC = blockIdx.y * BM;
    int colC = blockIdx.x * BN;

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK) {
        for (int idx = tid; idx < BM * BK; idx += 256) {
            int r = idx / BK, c = idx % BK;
            As[r][c] = A[(rowC + r) * K + (t + c)];
        }
        for (int idx = tid; idx < BK * BN; idx += 256) {
            int r = idx / BN, c = idx % BN;
            Bs[r][c] = B[(t + r) * N + (colC + c)];
        }
        __syncthreads();

        int tr = threadIdx.y * TM;
        int tc = threadIdx.x * TN;

        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            for (int i = 0; i < TM; ++i) a[i] = As[tr + i][k];
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tc + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];     // 64 次
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(rowC + threadIdx.y * TM + i) * N + (colC + threadIdx.x * TN + j)] = acc[i][j];
}

// 对照：4×4 版（原样搬过来）
constexpr int BM4 = 64, BN4 = 64;
__global__ void gemmTT4(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K) {
    __shared__ float As[BM4][BK];
    __shared__ float Bs[BK][BN4];
    int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    int rowC = blockIdx.y * BM4;
    int colC = blockIdx.x * BN4;
    float acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK) {
        for (int idx = tid; idx < BM4 * BK; idx += 256) {
            int r = idx / BK, c = idx % BK;
            As[r][c] = A[(rowC + r) * K + (t + c)];
        }
        for (int idx = tid; idx < BK * BN4; idx += 256) {
            int r = idx / BN4, c = idx % BN4;
            Bs[r][c] = B[(t + r) * N + (colC + c)];
        }
        __syncthreads();
        int tr = threadIdx.y * 4, tc = threadIdx.x * 4;
        for (int k = 0; k < BK; ++k) {
            float a[4], b[4];
            for (int i = 0; i < 4; ++i) a[i] = As[tr + i][k];
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tc + j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            C[(rowC + threadIdx.y * 4 + i) * N + (colC + threadIdx.x * 4 + j)] = acc[i][j];
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
        dim3 g4((n + 63) / 64, (n + 63) / 64);
        dim3 g8((n + 127) / 128, (n + 127) / 128);

        float best[2] = {1e9f, 1e9f};
        for (int r = 0; r < 3; ++r) {
            gemmTT4<<<g4, blk>>>(dA, dB, dC, n, n, n);
            gemmTT8<<<g8, blk>>>(dA, dB, dC, n, n, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int rep = 0; rep < 6; ++rep) {
            GpuTimer t1; t1.start();
            gemmTT4<<<g4, blk>>>(dA, dB, dC, n, n, n);
            float m1 = t1.stop(); if (m1 < best[0]) best[0] = m1;

            GpuTimer t2; t2.start();
            gemmTT8<<<g8, blk>>>(dA, dB, dC, n, n, n);
            float m2 = t2.stop(); if (m2 < best[1]) best[1] = m2;
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hC, dC, one, cudaMemcpyDeviceToHost));
        float err = 0.0f;
        for (int t = 0; t < 32; ++t) {
            int row = (t * 37 + 11) % n, col = (t * 53 + 7) % n;
            float d = fabsf(cpuDot(hA, hB, row, col, n, n) - hC[row * n + col]);
            if (d > err) err = d;
        }

        double f4 = 2.0 * n * n * n / (best[0] * 1e6);
        double f8 = 2.0 * n * n * n / (best[1] * 1e6);
        printf("\n  === n = %d   (4x4 用 %d 个 block，8x8 用 %d 个) ===\n",
               n, g4.x * g4.y, g8.x * g8.y);
        printf("  4x4 : %8.3f ms  %9.1f GFLOP/s\n", best[0], f4);
        printf("  8x8 : %8.3f ms  %9.1f GFLOP/s   %.2f 倍\n", best[1], f8, f8 / f4);
        printf("  校验误差 : %.0e\n", err);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }
    return 0;
}
