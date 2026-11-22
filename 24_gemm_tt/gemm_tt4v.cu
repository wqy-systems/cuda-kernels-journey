#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.h"

constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4, NTHREADS = 256;

// ---------------- 4×4，原始版（对照） ----------------
__global__ void gemmTT4(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int rowC = blockIdx.y * BM, colC = blockIdx.x * BN;
    float acc[TM][TN] = {};

    for (int t = 0; t < K; t += BK) {
        for (int i = tid; i < BM * BK; i += NTHREADS) {
            int r = i / BK, c = i % BK;
            As[r][c] = A[(rowC + r) * K + t + c];
        }
        for (int i = tid; i < BK * BN; i += NTHREADS) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = B[(t + r) * N + colC + c];
        }
        __syncthreads();

        int tr = threadIdx.y * TM, tc = threadIdx.x * TN;
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            for (int i = 0; i < TM; ++i) a[i] = As[tr + i][k];
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tc + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(rowC + threadIdx.y * TM + i) * N + (colC + threadIdx.x * TN + j)] = acc[i][j];
}

// ---------------- 4×4 + float4 ----------------
__global__ void gemmTT4Vec(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int rowC = blockIdx.y * BM, colC = blockIdx.x * BN;
    float acc[TM][TN] = {};

    constexpr int CA = BM * BK / 4;    // A 这一块切成多少份 float4
    constexpr int CB = BK * BN / 4;    // B 同理
    constexpr int AC = BK / 4;         // A 每行几份
    constexpr int BC = BN / 4;         // B 每行几份

    for (int t = 0; t < K; t += BK) {
        // 全局 -> 共享，一次 16 字节
        for (int i = tid; i < CA; i += NTHREADS) {
            int r = i / AC, c = (i % AC) * 4;
            float4 v = *reinterpret_cast<const float4*>(&A[(rowC + r) * K + t + c]);
            *reinterpret_cast<float4*>(&As[r][c]) = v;
        }
        for (int i = tid; i < CB; i += NTHREADS) {
            int r = i / BC, c = (i % BC) * 4;
            float4 v = *reinterpret_cast<const float4*>(&B[(t + r) * N + colC + c]);
            *reinterpret_cast<float4*>(&Bs[r][c]) = v;
        }
        __syncthreads();

        int tr = threadIdx.y * TM, tc = threadIdx.x * TN;
        for (int k = 0; k < BK; ++k) {
            // 竖着取，还是 4 条指令
            float a0 = As[tr + 0][k];
            float a1 = As[tr + 1][k];
            float a2 = As[tr + 2][k];
            float a3 = As[tr + 3][k];
            // 横着取，一条指令
            float4 b = *reinterpret_cast<const float4*>(&Bs[k][tc]);

            acc[0][0] += a0 * b.x;  acc[0][1] += a0 * b.y;
            acc[0][2] += a0 * b.z;  acc[0][3] += a0 * b.w;
            acc[1][0] += a1 * b.x;  acc[1][1] += a1 * b.y;
            acc[1][2] += a1 * b.z;  acc[1][3] += a1 * b.w;
            acc[2][0] += a2 * b.x;  acc[2][1] += a2 * b.y;
            acc[2][2] += a2 * b.z;  acc[2][3] += a2 * b.w;
            acc[3][0] += a3 * b.x;  acc[3][1] += a3 * b.y;
            acc[3][2] += a3 * b.z;  acc[3][3] += a3 * b.w;
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(rowC + threadIdx.y * TM + i) * N + (colC + threadIdx.x * TN + j)] = acc[i][j];
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
        dim3 grd((n + BN - 1) / BN, (n + BM - 1) / BM);

        float best[2] = {1e9f, 1e9f};
        for (int r = 0; r < 3; ++r) {
            gemmTT4<<<grd, blk>>>(dA, dB, dC, n, n, n);
            gemmTT4Vec<<<grd, blk>>>(dA, dB, dC, n, n, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int rep = 0; rep < 6; ++rep) {
            { GpuTimer t; t.start();
              gemmTT4<<<grd, blk>>>(dA, dB, dC, n, n, n);
              float m = t.stop(); if (m < best[0]) best[0] = m; }
            { GpuTimer t; t.start();
              gemmTT4Vec<<<grd, blk>>>(dA, dB, dC, n, n, n);
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
        printf("\n  === n = %d ===\n", n);
        printf("  4x4      : %8.3f ms  %9.1f GFLOP/s\n", best[0], f1);
        printf("  4x4+vec4 : %8.3f ms  %9.1f GFLOP/s   %.2f 倍\n", best[1], f2, f2 / f1);
        printf("  校验误差 : %.0e\n", err);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }
    return 0;
}
