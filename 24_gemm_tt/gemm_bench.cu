// ===========================================================================
// GEMM 擂台：所有版本放进同一个文件、同一份二进制，一次跑出完整对比表
//
//   v1 naive      朴素版，一个线程一个输出
//   v2 tiled16    共享内存分块 TILE=16，1×1 线程分块
//   v3 tt4        4×4 线程分块，标量访存
//   v4 tt4v       4×4 + float4 访存
//   v5 tt4v_lb4   4×4 + float4 + launch_bounds(256,4)
//   v6 tt8v       8×8 + float4
//   v7 db         4×4 + float4 + 双缓冲（预取到寄存器）
//   v8 db_lb4     双缓冲 + launch_bounds(256,4)
//
// 编译：nvcc -O3 -Iinclude -Xptxas -v -o 24_gemm_tt/gemm_bench 24_gemm_tt/gemm_bench.cu
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.h"

constexpr int BK = 16, NTHREADS = 256;

// ---------------------------------------------------------------- v1 naive
__global__ void k1_naive(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float s = 0.0f;
    for (int k = 0; k < K; ++k) s += A[row * K + k] * B[k * N + col];
    C[row * N + col] = s;
}

// -------------------------------------------------------------- v2 tiled16
constexpr int T16 = 16;
__global__ void k2_tiled16(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[T16][T16];
    __shared__ float Bs[T16][T16];
    int row = blockIdx.y * T16 + threadIdx.y;
    int col = blockIdx.x * T16 + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < K; t += T16) {
        As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < T16; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ---------------------------------------- v3~v6：线程分块（模板控制一切）
template <int BM, int BN, int TM, int TN, bool VEC, int MINB>
__global__ void __launch_bounds__(NTHREADS, MINB)
k_tt(const float* __restrict__ A, const float* __restrict__ B,
     float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int rowC = blockIdx.y * BM, colC = blockIdx.x * BN;
    float acc[TM][TN] = {};

    constexpr int CA = BM * BK / 4, CB = BK * BN / 4;
    constexpr int AC = BK / 4, BC = BN / 4;

    for (int t = 0; t < K; t += BK) {
        if (VEC) {
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
        } else {
            for (int i = tid; i < BM * BK; i += NTHREADS) {
                int r = i / BK, c = i % BK;
                As[r][c] = A[(rowC + r) * K + t + c];
            }
            for (int i = tid; i < BK * BN; i += NTHREADS) {
                int r = i / BN, c = i % BN;
                Bs[r][c] = B[(t + r) * N + colC + c];
            }
        }
        __syncthreads();

        int tr = threadIdx.y * TM, tc = threadIdx.x * TN;
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            for (int i = 0; i < TM; ++i) a[i] = As[tr + i][k];
            float bv[TN];
            if (VEC) {
                for (int j = 0; j < TN; j += 4) {
                    float4 b = *reinterpret_cast<const float4*>(&Bs[k][tc + j]);
                    bv[j] = b.x; bv[j + 1] = b.y; bv[j + 2] = b.z; bv[j + 3] = b.w;
                }
            } else {
                for (int j = 0; j < TN; ++j) bv[j] = Bs[k][tc + j];
            }
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * bv[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(rowC + threadIdx.y * TM + i) * N + (colC + threadIdx.x * TN + j)] = acc[i][j];
}

// ------------------------------------- v7/v8：双缓冲（4×4 + float4 + 预取）
constexpr int BM4 = 64, BN4 = 64, TM4 = 4, TN4 = 4;

template <int MINB>
__global__ void __launch_bounds__(NTHREADS, MINB)
k_db(const float* __restrict__ A, const float* __restrict__ B,
     float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][BM4][BK];
    __shared__ float Bs[2][BK][BN4];

    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const int rowC = blockIdx.y * BM4;
    const int colC = blockIdx.x * BN4;

    // 256 个线程、256 个 float4，每个线程固定负责 A 和 B 各一份
    constexpr int AC = BK / 4;    // 4  -> A 每行 4 份
    constexpr int BC = BN4 / 4;   // 16 -> B 每行 16 份
    const int rA = tid / AC, cA = (tid % AC) * 4;
    const int rB = tid / BC, cB = (tid % BC) * 4;

    float acc[TM4][TN4] = {};

    // 预取第 0 块，直接进共享内存
    *reinterpret_cast<float4*>(&As[0][rA][cA]) =
        *reinterpret_cast<const float4*>(&A[(rowC + rA) * K + 0 + cA]);
    *reinterpret_cast<float4*>(&Bs[0][rB][cB]) =
        *reinterpret_cast<const float4*>(&B[(0 + rB) * N + colC + cB]);
    __syncthreads();

    const int tr = threadIdx.y * TM4;
    const int tc = threadIdx.x * TN4;
    int buf = 0;

    for (int t = 0; t < K; t += BK) {
        const bool has_next = (t + BK < K);

        // 先把下一块读进寄存器：长延迟，跟下面的计算重叠
        float4 na, nb;
        if (has_next) {
            na = *reinterpret_cast<const float4*>(&A[(rowC + rA) * K + t + BK + cA]);
            nb = *reinterpret_cast<const float4*>(&B[(t + BK + rB) * N + colC + cB]);
        }

        // 用当前这块算
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM4];
            #pragma unroll
            for (int i = 0; i < TM4; ++i) a[i] = As[buf][tr + i][k];
            float4 b = *reinterpret_cast<const float4*>(&Bs[buf][k][tc]);
            float bv[TN4] = {b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int i = 0; i < TM4; ++i)
                #pragma unroll
                for (int j = 0; j < TN4; ++j) acc[i][j] += a[i] * bv[j];
        }

        // 存进另一块，跟上面的计算不冲突（读的是 buf，写的是 1-buf）
        if (has_next) {
            const int nxt = 1 - buf;
            *reinterpret_cast<float4*>(&As[nxt][rA][cA]) = na;
            *reinterpret_cast<float4*>(&Bs[nxt][rB][cB]) = nb;
        }

        __syncthreads();   // 每轮只要一次
        buf = 1 - buf;
    }

    for (int i = 0; i < TM4; ++i)
        for (int j = 0; j < TN4; ++j)
            C[(rowC + threadIdx.y * TM4 + i) * N + (colC + threadIdx.x * TN4 + j)] = acc[i][j];
}

// ---------------------------------------------------------------- 版本表
using KFn = void (*)(const float*, const float*, float*, int, int, int);

struct Variant {
    const char* name;
    KFn         fn;
    int         bM, bN;   // block 线程排布
    int         tM, tN;   // 一个 block 负责 C 的多大一块
};

float cpuDot(const float* A, const float* B, int row, int col, int K, int N) {
    double s = 0.0;
    for (int k = 0; k < K; ++k) s += (double)A[row * K + k] * B[k * N + col];
    return (float)s;
}

int main() {
    printDeviceInfo();

    Variant vs[] = {
        {"v1_naive",    k1_naive,                       16, 16,  16,  16},
        {"v2_tiled16",  k2_tiled16,                     16, 16,  16,  16},
        {"v3_tt4",      k_tt<64, 64, 4, 4, false, 2>,   16, 16,  64,  64},
        {"v4_tt4v",     k_tt<64, 64, 4, 4, true,  2>,   16, 16,  64,  64},
        {"v5_tt4v_lb4", k_tt<64, 64, 4, 4, true,  4>,   16, 16,  64,  64},
        {"v6_tt8v",     k_tt<128,128, 8, 8, true,  2>,  16, 16, 128, 128},
        {"v7_db",       k_db<2>,                        16, 16,  64,  64},
        {"v8_db_lb4",   k_db<4>,                        16, 16,  64,  64},
    };
    const int NV = sizeof(vs) / sizeof(vs[0]);

    int sizes[2] = {1024, 2048};

    printf("\n%-6s %-14s %10s %12s %14s\n", "n", "版本", "耗时(ms)", "GFLOP/s", "相对v2_tiled");
    printf("--------------------------------------------------------------\n");

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

        float best[8];
        for (int v = 0; v < NV; ++v) best[v] = 1e9f;

        const int skip = (n >= 2048) ? 0 : -1;   // 2048 时跳过朴素版

        for (int r = 0; r < 2; ++r)
            for (int v = 0; v < NV; ++v) {
                if (v == skip) continue;
                dim3 blk(vs[v].bM, vs[v].bN);
                dim3 grd((n + vs[v].tN - 1) / vs[v].tN,
                         (n + vs[v].tM - 1) / vs[v].tM);
                vs[v].fn<<<grd, blk>>>(dA, dB, dC, n, n, n);
            }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int rep = 0; rep < 5; ++rep)
            for (int v = 0; v < NV; ++v) {
                if (v == skip) continue;
                dim3 blk(vs[v].bM, vs[v].bN);
                dim3 grd((n + vs[v].tN - 1) / vs[v].tN,
                         (n + vs[v].tM - 1) / vs[v].tM);
                GpuTimer t; t.start();
                vs[v].fn<<<grd, blk>>>(dA, dB, dC, n, n, n);
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

        for (int v = 0; v < NV; ++v) {
            if (best[v] > 1e8f) {
                printf("%-6d %-14s %10s %12s %14s\n", n, vs[v].name, "skip", "skip", "-");
                continue;
            }
            double g = 2.0 * n * n * n / (best[v] * 1e6);
            double r = (v == 1) ? 1.0 : best[1] / best[v];
            printf("%-6d %-14s %10.3f %12.1f %13.2fx\n", n, vs[v].name, best[v], g, r);
        }
        printf("  (校验误差 %.0e)\n", err);
        printf("--------------------------------------------------------------\n");

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }
    return 0;
}
