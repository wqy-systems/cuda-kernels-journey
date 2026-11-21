#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.h"

constexpr int BM = 64;   // 一个 block 负责 C 的 64 行
constexpr int BN = 64;   // 负责 C 的 64 列
constexpr int BK = 16;   // 每轮沿 K 推进 16
constexpr int TM = 4;    // 每个线程算 4 行
constexpr int TN = 4;    // 每个线程算 4 列

// block = 16×16 = 256 个线程，合起来刚好 64×64 个输出
__global__ void gemmTT(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    int rowC = blockIdx.y * BM;
    int colC = blockIdx.x * BN;

    // 16 个累加器，全程住在寄存器里
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int t = 0; t < K; t += BK) {
        // 合作把 A 的 64×16 搬上片（每人 4 个）
        for (int idx = tid; idx < BM * BK; idx += 256) {
            int r = idx / BK, c = idx % BK;
            As[r][c] = A[(rowC + r) * K + (t + c)];
        }
        // 合作把 B 的 16×64 搬上片
        for (int idx = tid; idx < BK * BN; idx += 256) {
            int r = idx / BN, c = idx % BN;
            Bs[r][c] = B[(t + r) * N + (colC + c)];
        }
        __syncthreads();

        int tr = threadIdx.y * TM;
        int tc = threadIdx.x * TN;

        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            for (int i = 0; i < TM; ++i) a[i] = As[tr + i][k];   // 竖条 4 个
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tc + j];   // 横条 4 个
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];                    // 4×4 = 16 次
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(rowC + threadIdx.y * TM + i) * N + (colC + threadIdx.x * TN + j)] = acc[i][j];
}

__global__ void gemmTiled16(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    constexpr int T = 16;
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < K; t += T) {
        As[threadIdx.y][threadIdx.x] = A[row * K + (t + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

float cpuDot(const float* A, const float* B, int row, int col, int K, int N) {
    double s = 0.0;
    for (int k = 0; k < K; ++k) s += (double)A[row * K + k] * B[k * N + col];
    return (float)s;
}

int main() {
    printDeviceInfo();
    int n = 1024;
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

    dim3 blkTT(16, 16), grdTT((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 blk16(16, 16), grd16((n + 15) / 16, (n + 15) / 16);

    float best[2] = {1e9f, 1e9f};
    for (int r = 0; r < 3; ++r) {
        gemmTiled16<<<grd16, blk16>>>(dA, dB, dC, n, n, n);
        gemmTT<<<grdTT, blkTT>>>(dA, dB, dC, n, n, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int rep = 0; rep < 6; ++rep) {
        GpuTimer t1; t1.start();
        gemmTiled16<<<grd16, blk16>>>(dA, dB, dC, n, n, n);
        float m1 = t1.stop();
        if (m1 < best[0]) best[0] = m1;

        GpuTimer t2; t2.start();
        gemmTT<<<grdTT, blkTT>>>(dA, dB, dC, n, n, n);
        float m2 = t2.stop();
        if (m2 < best[1]) best[1] = m2;
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
    printf("\n  TILE16 1×1 : %.3f ms   %.1f GFLOP/s\n", best[0], f1);
    printf("  TT    4×4 : %.3f ms   %.1f GFLOP/s   %.2f 倍\n", best[1], f2, f2 / f1);
    printf("  校验误差 : %.0e\n", err);
    return 0;
}
