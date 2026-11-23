#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_pipeline.h>
#include "common.h"

constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4, NTHREADS = 256;

// ============================================================ v7 双缓冲（寄存器预取）
template <int MINB>
__global__ void __launch_bounds__(NTHREADS, MINB)
k_db(const float* __restrict__ A, const float* __restrict__ B,
     float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];
    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const int rowC = blockIdx.y * BM;
    const int colC = blockIdx.x * BN;

    constexpr int AC = BK / 4, BC = BN / 4;
    const int rA = tid / AC, cA = (tid % AC) * 4;
    const int rB = tid / BC, cB = (tid % BC) * 4;

    float acc[TM][TN] = {};

    *reinterpret_cast<float4*>(&As[0][rA][cA]) =
        *reinterpret_cast<const float4*>(&A[(rowC + rA) * K + cA]);
    *reinterpret_cast<float4*>(&Bs[0][rB][cB]) =
        *reinterpret_cast<const float4*>(&B[(rB) * N + colC + cB]);
    __syncthreads();

    const int tr = threadIdx.y * TM, tc = threadIdx.x * TN;
    int buf = 0;

    for (int t = 0; t < K; t += BK) {
        const bool has_next = (t + BK < K);
        float4 na, nb;
        if (has_next) {
            na = *reinterpret_cast<const float4*>(&A[(rowC + rA) * K + t + BK + cA]);
            nb = *reinterpret_cast<const float4*>(&B[(t + BK + rB) * N + colC + cB]);
        }
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[buf][tr + i][k];
            float4 b = *reinterpret_cast<const float4*>(&Bs[buf][k][tc]);
            float bv[TN] = {b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * bv[j];
        }
        if (has_next) {
            const int nxt = 1 - buf;
            *reinterpret_cast<float4*>(&As[nxt][rA][cA]) = na;
            *reinterpret_cast<float4*>(&Bs[nxt][rB][cB]) = nb;
        }
        __syncthreads();
        buf = 1 - buf;
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(rowC + threadIdx.y * TM + i) * N + (colC + threadIdx.x * TN + j)] = acc[i][j];
}

// ============================================================ v9 cp.async 三阶段
constexpr int STAGES = 3;

__global__ void k_async(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[STAGES][BM][BK];
    __shared__ float Bs[STAGES][BK][BN];
    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const int rowC = blockIdx.y * BM;
    const int colC = blockIdx.x * BN;

    constexpr int AC = BK / 4, BC = BN / 4;
    const int rA = tid / AC, cA = (tid % AC) * 4;
    const int rB = tid / BC, cB = (tid % BC) * 4;

    const int chunks = K / BK;
    float acc[TM][TN] = {};

    // 预取前 STAGES-1 块，异步直接进共享内存
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s >= chunks) break;
        __pipeline_memcpy_async(&As[s][rA][cA],
                                &A[(rowC + rA) * K + s * BK + cA], 16);
        __pipeline_memcpy_async(&Bs[s][rB][cB],
                                &B[(s * BK + rB) * N + colC + cB], 16);
        __pipeline_commit();
    }

    const int tr = threadIdx.y * TM, tc = threadIdx.x * TN;

    for (int i = 0; i < chunks; ++i) {
        __pipeline_wait_prior(STAGES - 2);   // 只留 STAGES-2 组没完成
        __syncthreads();                     // 数据可见 + 上一轮算完（buffer 可覆盖）

        const int nx = i + STAGES - 1;
        if (nx < chunks) {
            const int s = nx % STAGES;
            __pipeline_memcpy_async(&As[s][rA][cA],
                                    &A[(rowC + rA) * K + nx * BK + cA], 16);
            __pipeline_memcpy_async(&Bs[s][rB][cB],
                                    &B[(nx * BK + rB) * N + colC + cB], 16);
        }
        __pipeline_commit();                 // 保持 commit 计数对齐

        const int s = i % STAGES;
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            #pragma unroll
            for (int r = 0; r < TM; ++r) a[r] = As[s][tr + r][k];
            float4 b = *reinterpret_cast<const float4*>(&Bs[s][k][tc]);
            float bv[TN] = {b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int r = 0; r < TM; ++r)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[r][j] += a[r] * bv[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(rowC + threadIdx.y * TM + i) * N + (colC + threadIdx.x * TN + j)] = acc[i][j];
}

// ============================================================ 主函数
float cpuDot(const float* A, const float* B, int row, int col, int K, int N) {
    double s = 0.0;
    for (int k = 0; k < K; ++k) s += (double)A[row * K + k] * B[k * N + col];
    return (float)s;
}

using KFn = void (*)(const float*, const float*, float*, int, int, int);

int main() {
    printDeviceInfo();
    int sizes[2] = {1024, 2048};
    KFn fns[2] = {k_db<2>, k_async};
    const char* nm[2] = {"v7_db", "v9_cpasync"};

    for (int si = 0; si < 2; ++si) {
        int n = sizes[si];
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

        for (int r = 0; r < 3; ++r)
            for (int v = 0; v < 2; ++v) fns[v]<<<grd, blk>>>(dA, dB, dC, n, n, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int rep = 0; rep < 6; ++rep)
            for (int v = 0; v < 2; ++v) {
                GpuTimer t; t.start();
                fns[v]<<<grd, blk>>>(dA, dB, dC, n, n, n);
                float ms = t.stop();
                if (ms < best[v]) best[v] = ms;
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
        for (int v = 0; v < 2; ++v) {
            double g = 2.0 * n * n * n / (best[v] * 1e6);
            printf("  %-12s %8.3f ms  %9.1f GFLOP/s\n", nm[v], best[v], g);
        }
        printf("  cp.async 相对 v7 : %.2f 倍\n", best[0] / best[1]);
        printf("  校验误差 : %.0e\n", err);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }
    return 0;
}
