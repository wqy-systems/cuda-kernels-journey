// 04 · 归约求和：把 6700 万个数字加成一个
//
// 这个操作看起来简单，但它是 Softmax、LayerNorm、注意力分数求和的底层。
// 难点不在"加"，而在"怎么合并"：
//   每个线程先把自己那段加起来，得到几十万个「部分和」，
//   然后这些部分和要怎么变成一个总数？
//
// 三个版本的差别就是合并方式：
//   v1 原子加     —— 每个线程直接 atomicAdd 到同一个地址
//   v2 共享内存树 —— 每个 block 内用树形归约合成一个，再原子加
//   v3 warp shuffle —— 用寄存器交换代替共享内存，省掉同步
//
// 另外注意一个容易踩的坑：
//   6700 万个 float 相加，用 float 累加会掉精度（尾数只有 24 位），
//   所以这里统一用 double 累加。
//
// 编译：make 04_reduce/reduce
// 运行：./04_reduce/reduce

#include <cstdio>
#include <cstdlib>
#include "common.h"

constexpr int N = 1 << 26;     // 67,108,864 个 float = 256MB
constexpr int BLOCK = 256;
constexpr int GRID = 2048;     // 2048 × 256 = 524288 个线程
constexpr int REPS = 10;
constexpr int WARMUP = 3;

// ---------------------------------------------------------------------------
// v1：每个线程把自己那段加起来，然后原子加到结果上
//
// 问题：524288 个线程抢着改同一个地址，只能排队。
//      原子操作是串行的，这就是它的瓶颈。
// ---------------------------------------------------------------------------
__global__ void reduceAtomic(const float* __restrict__ in,
                             double* __restrict__ out,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double sum = 0.0;
    for (int i = idx; i < n; i += stride) sum += in[i];
    atomicAdd(out, sum);
}

// ---------------------------------------------------------------------------
// v2：每个线程算完自己那段后，先在 block 内部用共享内存做「树形归约」，
//     最后每个 block 只出一个数，再原子加。
//
// 为什么是树而不是链：
//   链式：0+1, 结果 +2, 结果 +3 …… 要 256 步
//   树形：两两相加，256 → 128 → 64 → …… → 1，只要 8 步
//   2^8 = 256。这就是为什么每步 sync 一次还能快很多。
//
//   原子操作的数量也从 524288 降到 2048（每个 block 一次）。
// ---------------------------------------------------------------------------
__global__ void reduceShared(const float* __restrict__ in,
                             double* __restrict__ out,
                             int n) {
    __shared__ double sdata[BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = idx; i < n; i += stride) sum += in[i];
    sdata[tid] = sum;

    __syncthreads();

    // 树形归约：每轮活下来的线程减半
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}

// ---------------------------------------------------------------------------
// v3：前半段用 warp shuffle 在寄存器之间交换数据，完全不碰共享内存
//
// __shfl_down_sync 让同一个 warp 里的 32 个线程互相读对方的寄存器值：
//   第 1 步：每个线程加上「往下数 16 个」的邻居
//   第 2 步：加上往下 8 个
//   …… 32 → 16 → 8 → 4 → 2 → 1，只要 5 步，而且不需要 __syncthreads()
//
// 一个 warp 归约成 1 个数之后，才写进共享内存（只有 8 个 warp 要处理）。
// ---------------------------------------------------------------------------
__global__ void reduceWarp(const float* __restrict__ in,
                           double* __restrict__ out,
                           int n) {
    __shared__ double warpSums[BLOCK / 32];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = idx; i < n; i += stride) sum += in[i];

    // warp 内归约：32 个线程 → 1 个
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 每个 warp 的第 0 个线程把结果写进共享内存
    if ((tid & 31) == 0) warpSums[tid >> 5] = sum;
    __syncthreads();

    // 最后 8 个值再归约一次（这里只有 8 个，用最简单的串行就行）
    if (tid == 0) {
        double total = 0.0;
        for (int i = 0; i < BLOCK / 32; ++i) total += warpSums[i];
        atomicAdd(out, total);
    }
}

using ReduceFn = void (*)(const float*, double*, int);

const char* NAMES[3] = {"v1 原子加", "v2 共享内存树", "v3 warp shuffle"};
ReduceFn    FNS[3]   = {reduceAtomic, reduceShared, reduceWarp};

int main() {
    printDeviceInfo();

    const size_t bytes = sizeof(float) * N;
    printf("数据量 : %d 个 float = %.0f MB\n", N, bytes / 1048576.0);
    printf("线程数 : %d × %d = %d\n", GRID, BLOCK, GRID * BLOCK);
    printf("每个线程要加 : %d 个数\n\n", N / (GRID * BLOCK));

    // ---- host ----
    float* h_in;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes));
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 97) * 0.01f;

    // CPU 参考值（用 double 累加，避免精度问题）
    double ref = 0.0;
    for (int i = 0; i < N; ++i) ref += h_in[i];
    printf("CPU 参考和 : %.6f\n\n", ref);

    // ---- device ----
    float* d_in;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    double* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));

    // ---- 预热 ----
    printf("预热中...\n");
    for (int w = 0; w < WARMUP; ++w) {
        for (int i = 0; i < 3; ++i) {
            CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
            FNS[i]<<<GRID, BLOCK>>>(d_in, d_out, N);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 交错测量 ----
    printf("交错测量 %d 轮...\n\n", REPS);
    GpuTimer timer;
    float best[3] = {1e9f, 1e9f, 1e9f};

    for (int r = 0; r < REPS; ++r) {
        for (int i = 0; i < 3; ++i) {
            CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
            timer.start();
            FNS[i]<<<GRID, BLOCK>>>(d_in, d_out, N);
            float ms = timer.stop();
            if (ms < best[i]) best[i] = ms;
        }
    }
    CUDA_CHECK(cudaGetLastError());

    // ---- 校验 ----
    double got[3];
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
        FNS[i]<<<GRID, BLOCK>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&got[i], d_out, sizeof(double), cudaMemcpyDeviceToHost));
    }

    // ---- 结果 ----
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    const double peak = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6;

    printf("=================== 归约（只读一遍 %0.f MB）===================\n", bytes / 1048576.0);
    printf("  %-18s %10s %10s %8s %12s\n", "版本", "最快(ms)", "GB/s", "占峰值", "结果误差");
    printf("  ----------------------------------------------------------------------\n");
    for (int i = 0; i < 3; ++i) {
        double bw = bytes / (best[i] * 1e6);
        double rel = (ref == 0.0) ? 0.0 : (got[i] - ref) / ref;
        printf("  %-18s %10.3f %10.1f %7.1f%% %11.2e\n",
               NAMES[i], best[i], bw, bw / peak * 100, rel);
    }
    printf("  ----------------------------------------------------------------------\n");
    printf("  参考值 %.6f   三个版本的结果：%.6f / %.6f / %.6f\n",
           ref, got[0], got[1], got[2]);
    printf("  v1 比 v3 慢 %.2f 倍\n", best[0] / best[2]);
    printf("======================================================================\n\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));

    return 0;
}
