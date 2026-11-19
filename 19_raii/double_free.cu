#include <cstdio>
#include <cuda_runtime.h>

int main() {
#if 1
    printf("===== 版本 A：cudaFree 两次 =====\n");
    float* p = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&p), 1024 * sizeof(float));
    printf("p = %p\n", (void*)p);

    cudaError_t e1 = cudaFree(p);
    printf("第一次 cudaFree: %s\n", cudaGetErrorString(e1));
    cudaError_t e2 = cudaFree(p);
    printf("第二次 cudaFree: %s\n", cudaGetErrorString(e2));
    printf("程序还活着，退出码 0\n\n");
#endif

#if 0
    printf("===== 版本 B：delete[] 两次 =====\n");
    float* q = new float[1024];
    printf("q = %p\n", (void*)q);
    delete[] q;
    printf("第一次 delete 完成\n");
    delete[] q;
    printf("第二次 delete 完成（这行永远不会打印）\n");
#endif
    return 0;
}
