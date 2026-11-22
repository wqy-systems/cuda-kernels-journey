#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.h"

constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4, NTHREADS = 256;

template <int MINB>
__global__ void __launch_bounds__(NTHREADS, MINB)
tt4v(const float* __restrict__ A, const float* __restrict__ B,
     float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int rowC = blockIdx.y * BM, colC = blockIdx.x * BN;
    float acc[TM][TN] = {};

    constexpr int CA = BM * BK / 4, CB = BK * BN / 4;
    constexpr int AC = BK / 4, BC = BN / 4;

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

        int tr = threadIdx.y * TM, tc = threadIdx.x * TN;
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            for (int i = 0; i < TM; ++i) a[i] = As[tr + i][k];
            float4 b = *reinterpret_cast<const float4*>(&Bs[k][tc]);
            float bv[TN] = {b.x, b.y, b.z, b.w};
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * bv[j];
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

using KFn = void (*)(const float*, const float*, float*, int, int, int);

int main() {
    printDeviceInfo();
    int sizes[2] = {1024, 2048};
    KFn fns[3]  = {tt4v<2>, tt4v<4>, tt4v<6>};
    const char* nm[3] = {"默认", "强制4块", "强制6块"};

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
        dim3 grd((n + BM - 1) / BM, (n + BM - 1) / BM);
        float best[3] = {1e9f, 1e9f, 1e9f};

        for (int r = 0; r < 3; ++r)
            for (int v = 0; v < 3; ++v) fns[v]<<<grd, blk>>>(dA, dB, dC, n, n, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int rep = 0; rep < 6; ++rep)
            for (int v = 0; v < 3; ++v) {
                GpuTimer t; t.start();
                fns[v]<<<grd, blk>>>(dA, dB, dC, n, n, n);
                float m = t.stop();
                if (m < best[v]) best[v] = m;
            }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hC, dC, one, cudaMemcpyDeviceToHost));
        float err = 0.0f;
        for (int t = 0; t < 32; ++t) {
            int row = (t * 37 + 11) % n, col = (t * 53 + 7) % n;
            float d = fabsf(cpuDot(hA, hB, row, col, n, n) - hC[row * n + col]);
            if (d > err) err = d;
        }

        printf("\n  === n = %d ===\n", n);
        for (int v = 0; v < 3; ++v) {
            double f = 2.0 * n * n * n / (best[v] * 1e6);
            printf("  %-10s %8.3f ms  %9.1f GFLOP/s\n", nm[v], best[v], f);
        }
        printf("  校验误差 : %.0e\n", err);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }
    return 0;
}
