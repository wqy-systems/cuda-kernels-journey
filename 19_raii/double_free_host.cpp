#include <cstdio>
int main() {
    printf("===== 版本 B：delete[] 两次 =====\n");
    float* q = new float[1024];
    printf("q = %p\n", (void*)q);
    delete[] q;
    printf("第一次 delete 完成\n");
    delete[] q;
    printf("第二次 delete 完成（这行永远不会打印）\n");
    return 0;
}
