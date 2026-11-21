#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error: %s  (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int K = 9;

void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    CK(cudaMalloc(&p, bytes));
    return p;
}

__global__ void conv1d_naive(const float* __restrict__ in,
                             const float* __restrict__ w,
                             float* __restrict__ out,
                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; ++k)
        acc += in[i + k] * w[k];

    out[i] = acc;
}

int main() {
    const int    n         = 32 * 1024 * 1024;              // 32M 个输出
    const size_t in_bytes  = (size_t)(n + K - 1) * sizeof(float);
    const size_t out_bytes = (size_t)n * sizeof(float);

    float* h_in  = (float*)malloc(in_bytes);
    float* h_ref = (float*)malloc(out_bytes);
    float  h_w[K];
    for (int i = 0; i < n + K - 1; ++i) h_in[i] = (float)((i % 100) - 50) * 0.01f;
    for (int k = 0; k < K; ++k)         h_w[k]   = 1.0f / K;

    float* d_in  = (float*)dev_alloc(in_bytes);
    float* d_out = (float*)dev_alloc(out_bytes);
    float* d_w   = (float*)dev_alloc(K * sizeof(float));
    CK(cudaMemcpy(d_in,  h_in, in_bytes,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w,   h_w,  K * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid  = (n + block - 1) / block;

    for (int i = 0; i < 3; ++i) conv1d_naive<<<grid, block>>>(d_in, d_w, d_out, n);
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    float best = 1e30f;
    for (int r = 0; r < 10; ++r) {
        cudaEventRecord(s);
        conv1d_naive<<<grid, block>>>(d_in, d_w, d_out, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s, e);
        if (ms < best) best = ms;
    }

    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += h_in[i + k] * h_w[k];
        h_ref[i] = acc;
    }
    float* h_got = (float*)malloc(out_bytes);
    CK(cudaMemcpy(h_got, d_out, out_bytes, cudaMemcpyDeviceToHost));
    float err = 0.0f;
    for (int i = 0; i < n; ++i) { float d = fabsf(h_ref[i] - h_got[i]); if (d > err) err = d; }

    double min_bytes   = (double)in_bytes + (double)out_bytes;
    double naive_bytes = (double)n * K * 4.0 + (double)out_bytes;

    printf("K = %d, n = %d (输出 %d MB)\n\n", K, n, (int)(out_bytes / 1048576));
    printf("最快耗时   : %.3f ms\n", best);
    printf("最大误差   : %.3e\n\n", err);
    printf("理想流量   : %.0f MB   (in 读一次 + out 写一次)\n", min_bytes / 1048576.0);
    printf("理想带宽   : %.1f GB/s\n", min_bytes / (best * 1e6));
    printf("朴素读量   : %.0f MB   (每个输出读 %d 次)\n", naive_bytes / 1048576.0, K);
    printf("朴素带宽   : %.1f GB/s\n", naive_bytes / (best * 1e6));

    free(h_in); free(h_ref); free(h_got);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_w);
    return 0;
}
