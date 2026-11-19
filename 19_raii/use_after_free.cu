#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(expr) printf("%-30s : %s\n", #expr, cudaGetErrorString(expr))

int main() {
    float* p = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&p), 1024 * sizeof(float));
    printf("p = %p\n", (void*)p);

    CHECK(cudaFree(p));
    CHECK(cudaFree(p));                              // 还了又还

    CHECK(cudaMemset(p, 0, 1024 * sizeof(float)));   // 往不归你的显存写
    CHECK(cudaMemcpy(p, p, 4, cudaMemcpyDeviceToDevice));

    float h = -1.0f;
    cudaMemcpy(&h, p, sizeof(float), cudaMemcpyDeviceToHost);
    printf("读回来的 h = %f\n", h);

    printf("退出码 0 —— 全程没有一次崩溃\n");
    return 0;
}
