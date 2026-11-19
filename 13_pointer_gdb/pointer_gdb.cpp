// C++ 指针补习：用 gdb 亲手"看见"指针
//
// 你写 CUDA 时碰过这些：&d_x、const float* x、__restrict__
// 这个小程序把它们全放在你眼前，用 gdb 的 print 一个一个看。

#include <cstdio>
#include <cstdlib>

void change(int* p) {
    *p = 999;      // 顺着门牌号，去改那家的东西
}

int main() {
    int a = 42;           // 一个普通变量：一间房子，里面放着 42
    int* p = &a;          // p 是一张纸条，上面写着 a 的门牌号

    printf("a 的值   = %d\n", a);
    printf("a 的地址 = %p\n", (void*)&a);
    printf("p 存的   = %p\n", (void*)p);
    printf("*p       = %d\n", *p);

    // 堆上申请三个连续的 int（跟 cudaMalloc 一个道理，只不过在内存里）
    int* heap = (int*)malloc(sizeof(int) * 3);
    heap[0] = 10;
    heap[1] = 20;
    heap[2] = 30;

    printf("heap[0]  = %d\n", heap[0]);
    printf("heap[1]  = %d\n", heap[1]);

    // 为什么 cudaMalloc 要写 &d_x：
    // 因为函数要"把新地址写进你的变量里"，得知道你变量在哪儿
    change(&a);
    printf("change 之后 a = %d\n", a);

    free(heap);
    return 0;
}
