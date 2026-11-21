#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error: %s  (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int BLOCK = 256;

void* dev_alloc(size_t bytes) { void* p = nullptr; CK(cudaMalloc(&p, bytes)); return p; }

__device__ __forceinline__ float block_max(float v, float* sdata, int tid) {
    sdata[tid] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    return sdata[0];
}

__device__ __forceinline__ float block_sum(float v, float* sdata, int tid) {
    sdata[tid] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

__global__ void softmax_v1(const float* __restrict__ in,
                           float* __restrict__ out,
                           int cols) {
    __shared__ float sdata[BLOCK];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = in  + (size_t)row * cols;
    float*       y = out + (size_t)row * cols;

    float m = -1e30f;
    for (int i = tid; i < cols; i += BLOCK) m = fmaxf(m, x[i]);
    m = block_max(m, sdata, tid);
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK) sum += expf(x[i] - m);
    sum = block_sum(sum, sdata, tid);

    float inv = 1.0f / sum;
    for (int i = tid; i < cols; i += BLOCK) y[i] = expf(x[i] - m) * inv;
}

int main() {
    const int    rows = 32768, cols = 1024;
    const size_t n = (size_t)rows * cols;
    const size_t bytes = n * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    for (size_t i = 0; i < n; ++i) h_in[i] = (float)((int)(i % 1000) - 500) * 0.01f;

    float* d_in  = (float*)dev_alloc(bytes);
    float* d_out = (float*)dev_alloc(bytes);
    CK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < 3; ++i) softmax_v1<<<rows, BLOCK>>>(d_in, d_out, cols);
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    float best = 1e30f;
    for (int r = 0; r < 10; ++r) {
        cudaEventRecord(s);
        softmax_v1<<<rows, BLOCK>>>(d_in, d_out, cols);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float t = 0; cudaEventElapsedTime(&t, s, e);
        if (t < best) best = t;
    }

    float* h_out = (float*)malloc(bytes);
    CK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double max_abs = 0.0, max_rs = 0.0;
    for (int r = 0; r < rows; ++r) {
        const float* x = h_in  + (size_t)r * cols;
        const float* y = h_out + (size_t)r * cols;
        float m = -1e30f;
        for (int i = 0; i < cols; ++i) m = fmaxf(m, x[i]);
        double ssum = 0.0, ysum = 0.0;
        for (int i = 0; i < cols; ++i) ssum += expf(x[i] - m);
        for (int i = 0; i < cols; ++i) {
            double d = fabs(expf(x[i] - m) / ssum - y[i]);
            if (d > max_abs) max_abs = d;
            ysum += y[i];
        }
        double t = fabs(ysum - 1.0);
        if (t > max_rs) max_rs = t;
    }

    printf("rows=%d  cols=%d   数据 %.0f MB\n\n", rows, cols, 2 * bytes / 1048576.0);
    printf("最快耗时   : %.3f ms\n", best);
    printf("有效带宽   : %.1f GB/s\n", 2.0 * bytes / (best * 1e6));
    printf("最大误差   : %.3e\n", max_abs);
    printf("每行和偏离1: %.3e\n", max_rs);
    return 0;
}
