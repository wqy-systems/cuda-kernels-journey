#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error: %s  (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int K     = 9;
constexpr int BLOCK = 256;

void* dev_alloc(size_t bytes) { void* p = nullptr; CK(cudaMalloc(&p, bytes)); return p; }

__global__ void conv1d_naive(const float* __restrict__ in,
                             const float* __restrict__ w,
                             float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; ++k) acc += in[i + k] * w[k];
    out[i] = acc;
}

__global__ void conv1d_smem(const float* __restrict__ in,
                            const float* __restrict__ w,
                            float* __restrict__ out, int n) {
    __shared__ float tile[BLOCK + K - 1];

    int base = blockIdx.x * BLOCK;
    int tid  = threadIdx.x;

    for (int j = tid; j < BLOCK + K - 1; j += BLOCK)
        tile[j] = in[base + j];
    __syncthreads();

    int i = base + tid;
    if (i >= n) return;

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; ++k) acc += tile[tid + k] * w[k];
    out[i] = acc;
}

int main() {
    const int    n         = 32 * 1024 * 1024;
    const size_t in_bytes  = (size_t)(n + K - 1) * sizeof(float);
    const size_t out_bytes = (size_t)n * sizeof(float);

    float* h_in = (float*)malloc(in_bytes);
    float  h_w[K];
    for (int i = 0; i < n + K - 1; ++i) h_in[i] = (float)((i % 100) - 50) * 0.01f;
    for (int k = 0; k < K; ++k)         h_w[k]   = 1.0f / K;

    float* d_in  = (float*)dev_alloc(in_bytes);
    float* d_out = (float*)dev_alloc(out_bytes);
    float* d_w   = (float*)dev_alloc(K * sizeof(float));
    CK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, h_w, K * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (n + BLOCK - 1) / BLOCK;

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    float ms_naive = 1e30f, ms_smem = 1e30f;
    for (int r = 0; r < 12; ++r) {
        if (r >= 2) {
            cudaEventRecord(s);
            conv1d_naive<<<grid, BLOCK>>>(d_in, d_w, d_out, n);
            cudaEventRecord(e); cudaEventSynchronize(e);
            float t = 0; cudaEventElapsedTime(&t, s, e); if (t < ms_naive) ms_naive = t;
        } else {
            conv1d_naive<<<grid, BLOCK>>>(d_in, d_w, d_out, n);
        }

        if (r >= 2) {
            cudaEventRecord(s);
            conv1d_smem<<<grid, BLOCK>>>(d_in, d_w, d_out, n);
            cudaEventRecord(e); cudaEventSynchronize(e);
            float t = 0; cudaEventElapsedTime(&t, s, e); if (t < ms_smem) ms_smem = t;
        } else {
            conv1d_smem<<<grid, BLOCK>>>(d_in, d_w, d_out, n);
        }
    }

    double min_bytes = (double)in_bytes + (double)out_bytes;
    printf("朴素    : %.3f ms   %.1f GB/s\n", ms_naive, min_bytes / (ms_naive * 1e6));
    printf("共享内存: %.3f ms   %.1f GB/s\n", ms_smem,  min_bytes / (ms_smem  * 1e6));
    printf("差异    : %+.1f%%  (正数=smem 更快)\n", (ms_naive / ms_smem - 1.0) * 100.0);

    free(h_in); cudaFree(d_in); cudaFree(d_out); cudaFree(d_w);
    return 0;
}
