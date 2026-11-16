// 05 · GEMM 第一版：朴素实现，用来撞墙
//
// C[M×N] = A[M×K] × B[K×N]
//
// 前四个 kernel 都是「带宽受限」：每个数据读一两次就完事。
// GEMM 是「算力受限」：每个数据要被重复使用几百次，
// 只要复用得好，就能突破带宽的限制。
//
// 但这一版完全没有复用：每个线程算 C 的一个元素，要把
// A 的一整行（K 个数）和 B 的一整列（K 个数）从头读到尾。
// 结果是 —— 同一个 A[i][k] 被读了 N 次，同一个 B[k][j] 被读了 M 次。
//
// 这一版故意写得笨，目的是让你看到「撞墙」的现场。
//
// 编译：cmake --build build -j
// 运行：./build/bin/gemm

#include <cstdio>
#include <cstdlib>
#include "common.h"

// ---------------------------------------------------------------------------
// 朴素 GEMM：一个线程算 C 的一个元素
//
//   row = 第几行，col = 第几列
//   sum = Σ A[row][k] × B[k][col]   （k 从 0 到 K-1）
//
// 注意这里每一次内层循环都要读两个全局内存：
//   A[row * K + k]  —— 同一个 warp 里 32 个线程读的是同一个地址（广播，不算太糟）
//   B[k * N + col]  —— 32 个线程读连续地址（合并访问，也不糟）
//
// 单个访问本身不难看，问题是「次数」：
//   A 的每个元素被读 N 次，B 的每个元素被读 M 次。
//   1024³ 的矩阵，A 只有 4MB，却要被读 1024 遍 = 4GB 流量。
// ---------------------------------------------------------------------------
__global__ void gemmNaive(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// CPU 参考实现（只算指定的几个点，全算太慢）
float cpuDot(const float* A, const float* B, int row, int col, int K, int N) {
    double sum = 0.0;
    for (int k = 0; k < K; ++k) sum += static_cast<double>(A[row * K + k]) * B[k * N + col];
    return static_cast<float>(sum);
}

int main() {
    printDeviceInfo();

    // 算力峰值：SM 数 × 每 SM 的 CUDA 核心数 × 每核心每周期 2 次浮点 × 频率
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    const double clockGHz = p.clockRate / 1.0e6;
    const double peakFLOPS = p.multiProcessorCount * 128.0 * 2.0 * clockGHz * 1e9;  // FLOP/s
    const double peakBW = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6; // GB/s
    const double balance = peakFLOPS / (peakBW * 1e9);   // 算术强度平衡点 FLOP/byte

    printf("算力峰值 : %.1f TFLOPS  (%d SM x 128 核 x 2 x %.2f GHz)\n",
           peakFLOPS / 1e12, p.multiProcessorCount, clockGHz);
    printf("带宽峰值 : %.1f GB/s\n", peakBW);
    printf("平衡点   : %.1f FLOP/byte  ← 算术强度超过这个数，才可能算力受限\n",
           balance);
    printf("           意思是：每从显存搬 1 字节，至少要干 %.0f 次浮点运算才不亏\n\n",
           balance);

    const int sizes[] = {512, 1024, 2048, 4096};
    const int nsizes = 4;

    printf("=====================================================================================\n");
    printf("  %6s %10s %12s %10s %10s %14s\n",
           "尺寸", "耗时(ms)", "GFLOP/s", "占算力峰值", "算术强度", "是平衡点的");
    printf("  -----------------------------------------------------------------------------------\n");

    for (int s = 0; s < nsizes; ++s) {
        const int n = sizes[s];
        const size_t one = sizeof(float) * (size_t)n * n;

        // 8G 显存足够：4096³ 时每个矩阵 64MB，三个共 192MB
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, one));
        CUDA_CHECK(cudaMalloc(&d_B, one));
        CUDA_CHECK(cudaMalloc(&d_C, one));

        float *h_A = (float*)malloc(one);
        float *h_B = (float*)malloc(one);
        float *h_C = (float*)malloc(one);
        for (size_t i = 0; i < (size_t)n * n; ++i) {
            h_A[i] = static_cast<float>(static_cast<int>(i % 100) - 50) * 0.01f;
            h_B[i] = static_cast<float>((int)(i % 73) - 36) * 0.02f;
        }
        CUDA_CHECK(cudaMemcpy(d_A, h_A, one, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, one, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (n + 15) / 16);

        // 热身一次
        gemmNaive<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 大尺寸跑得慢，少跑几次
        const int reps = (n >= 2048) ? 3 : 5;
        GpuTimer timer;
        float best = 1e9f;
        for (int r = 0; r < reps; ++r) {
            timer.start();
            gemmNaive<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
            float ms = timer.stop();
            if (ms < best) best = ms;
        }
        CUDA_CHECK(cudaGetLastError());

        // 抽查 64 个点校验（全算太慢）
        CUDA_CHECK(cudaMemcpy(h_C, d_C, one, cudaMemcpyDeviceToHost));
        float maxerr = 0.0f;
        for (int t = 0; t < 64; ++t) {
            int row = (t * 37 + 11) % n;
            int col = (t * 53 + 7) % n;
            float ref = cpuDot(h_A, h_B, row, col, n, n);
            float d = fabsf(ref - h_C[row * n + col]);
            if (d > maxerr) maxerr = d;
        }

        double flops = 2.0 * n * n * n;                  // 乘加各算一次
        double gflops = flops / (best * 1e6);
        // 朴素版的算术强度：搬了 M*N*K*2 个 float，干了 2*M*N*K 次运算
        double intensity = flops / (2.0 * n * n * n * sizeof(float));

        printf("  %6d %10.3f %12.1f %9.2f%% %10.2f %11.1f 倍  校验 %.0e %s\n",
               n, best, gflops, gflops * 1e9 / peakFLOPS * 100,
               intensity, balance / intensity,
               maxerr, maxerr < 1e-2f ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A); free(h_B); free(h_C);
    }

    printf("=====================================================================================\n\n");
    printf("看最后两列：\n");
    printf("  算术强度 = 每搬 1 字节干了几次运算。朴素版固定是 %.2f，永远不变。\n", 0.25);
    printf("  平衡点是 %.0f。差了两个数量级 —— 所以它被内存卡死，算力几乎全闲。\n", balance);
    printf("  尺寸越大越惨：小矩阵能塞进 32MB 的 L2 缓存，大矩阵只能老老实实走显存。\n");
    printf("\n下一步：让一块数据被多个线程复用（共享内存分块），把算术强度提上去。\n");

    return 0;
}
