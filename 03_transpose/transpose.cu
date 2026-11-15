// 03 · 矩阵转置：三个版本，交错测量，公平对比
//
// 转置本身没有任何计算，纯粹搬数据。理论上应该跑满带宽（256 GB/s）。
// 三个版本的差别只有一件事：读和写的地址是不是连续的。
//
//   transposeNaive  —— 读连续、写跳跃（写请求散成 8 倍）
//   transposeTiled  —— 共享内存中转，但 block 用了 1024 线程，占用率低
//   transposeTiled8 —— 共享内存中转 + block 只 256 线程，每人搬 4 行
//
// 【本版新增】交错测量：
//   上一版的测法是「A 跑 10 次、B 跑 10 次、C 跑 10 次」，
//   结果先跑的 A 遇上冷 GPU、后跑的 C 遇上热 GPU，条件根本不公平。
//   现在改成「第 1 轮跑 A、B、C，第 2 轮再跑 A、B、C……」，
//   三者在时间上均匀分布，温度条件基本一致，数字才有可比性。
//
// 编译：make 03_transpose/transpose
// 运行：./03_transpose/transpose

#include <cstdio>
#include <cstdlib>
#include "common.h"

constexpr int N = 8192;        // N × N 矩阵，单个 64MB
constexpr int TILE = 32;       // 分块大小 32×32
constexpr int ROWS = 8;        // 第三版 block 的 y 方向线程数
constexpr int REPS = 20;       // 交错测量的轮数
constexpr int WARMUP = 3;      // 每个版本各预热几次

// ---------------------------------------------------------------------------
// 版本一：朴素转置
//   读：threadIdx.x 连续 → 地址连续 → 合并访问 ✓
//   写：threadIdx.x 连续，但 out[col * n + row]，地址隔 16KB → 散开 ✗
// ---------------------------------------------------------------------------
__global__ void transposeNaive(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < n) {
        out[col * n + row] = in[row * n + col];
    }
}

// ---------------------------------------------------------------------------
// 版本二：共享内存分块，block = 32×32 = 1024 线程
//   tile 第二维写 TILE + 1，让相邻行的地址间隔变成 33（与 32 互质），
//   避免 32 个线程挤到同一个 bank 上排队。
//
//   缺点：1024 个线程的 block，一个 SM 最多只放 1 个 → 占用率低。
// ---------------------------------------------------------------------------
__global__ void transposeTiled(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n) {
    __shared__ float tile[TILE][TILE + 1];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[row * n + col];

    __syncthreads();

    int outCol = blockIdx.y * TILE + threadIdx.x;
    int outRow = blockIdx.x * TILE + threadIdx.y;
    out[outRow * n + outCol] = tile[threadIdx.x][threadIdx.y];
}

// ---------------------------------------------------------------------------
// 版本三：共享内存分块，block 只 32×8 = 256 线程，每个线程搬 4 行
//   同一个 SM 能放 6 个 block，等待同步的时候有别人接管显存。
// ---------------------------------------------------------------------------
__global__ void transposeTiled8(const float* __restrict__ in,
                                float* __restrict__ out,
                                int n) {
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;   // 列号
    int y = blockIdx.y * TILE + threadIdx.y;   // 行号

    for (int j = 0; j < TILE; j += ROWS) {     // 每人负责 4 行
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// CPU 参考实现
void transposeCPU(const float* in, float* out, int n) {
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            out[c * n + r] = in[r * n + c];
}

// 一个版本的全部信息
using KernelFn = void (*)(const float*, float*, int);

struct Variant {
    const char* name;
    const char* note;
    KernelFn    kern;
    dim3        grid;
    dim3        block;
    float       best_ms;
    float       err;
};

int main() {
    printDeviceInfo();

    const int n = N;
    const size_t bytes = sizeof(float) * n * n;
    printf("矩阵大小 : %d x %d，单个 %.0f MB\n", n, n, bytes / 1048576.0);
    printf("每次转置搬运 : 读 %.0f MB + 写 %.0f MB = %.0f MB\n\n",
           bytes / 1048576.0, bytes / 1048576.0, 2 * bytes / 1048576.0);

    // ---- host ----
    float *h_in, *h_out, *h_ref;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));
    CUDA_CHECK(cudaMallocHost(&h_ref, bytes));

    for (int i = 0; i < n * n; ++i) h_in[i] = static_cast<float>(i % 1000) * 0.001f;
    transposeCPU(h_in, h_ref, n);

    // ---- device ----
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const double total_bytes = 2.0 * bytes;

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    const double peak = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6;

    // ---- 三个版本 ----
    Variant vs[3] = {
        {"版本一 朴素",
         "读连续、写跳跃",
         transposeNaive,
         dim3((n + 31) / 32, (n + ROWS - 1) / ROWS),
         dim3(32, ROWS),
         1e9f, 0.f},

        {"版本二 分块1024",
         "读连续、写连续，占用率仅 1 个 block/SM",
         transposeTiled,
         dim3((n + TILE - 1) / TILE, (n + TILE - 1) / TILE),
         dim3(TILE, TILE),
         1e9f, 0.f},

        {"版本三 分块256x4",
         "读连续、写连续，占用率 6 个 block/SM",
         transposeTiled8,
         dim3((n + TILE - 1) / TILE, (n + TILE - 1) / TILE),
         dim3(TILE, ROWS),
         1e9f, 0.f},
    };

    // ---- 预热：每个版本各跑 WARMUP 次 ----
    printf("预热中（每个版本 %d 次）...\n", WARMUP);
    for (int w = 0; w < WARMUP; ++w)
        for (int i = 0; i < 3; ++i)
            vs[i].kern<<<vs[i].grid, vs[i].block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 交错测量：第 r 轮依次跑三个版本 ----
    printf("交错测量 %d 轮...\n\n", REPS);
    GpuTimer timer;
    for (int r = 0; r < REPS; ++r) {
        for (int i = 0; i < 3; ++i) {
            timer.start();
            vs[i].kern<<<vs[i].grid, vs[i].block>>>(d_in, d_out, n);
            float ms = timer.stop();
            if (ms < vs[i].best_ms) vs[i].best_ms = ms;
        }
    }
    CUDA_CHECK(cudaGetLastError());

    // ---- 逐个校验正确性 ----
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        vs[i].kern<<<vs[i].grid, vs[i].block>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        vs[i].err = maxAbsError(h_ref, h_out, static_cast<size_t>(n) * n);
    }

    // ---- 结果 ----
    printf("=================== 对比（交错测量，条件一致）===================\n");
    printf("  %-20s %10s %10s %8s %8s\n", "版本", "最快(ms)", "GB/s", "占峰值", "校验");
    printf("  --------------------------------------------------------------------\n");
    for (int i = 0; i < 3; ++i) {
        double bw = total_bytes / (vs[i].best_ms * 1e6);
        printf("  %-20s %10.3f %10.1f %7.1f%% %8s\n",
               vs[i].name, vs[i].best_ms, bw, bw / peak * 100,
               vs[i].err == 0.0f ? "PASS" : "FAIL");
    }
    printf("  --------------------------------------------------------------------\n");
    printf("  版本三比版本一快 %.2f 倍\n", vs[0].best_ms / vs[2].best_ms);
    printf("  版本三比版本二快 %.2f 倍\n", vs[1].best_ms / vs[2].best_ms);
    printf("===================================================================\n\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFreeHost(h_ref));

    return 0;
}
