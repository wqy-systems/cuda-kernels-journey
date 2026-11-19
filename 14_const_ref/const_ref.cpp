// C++ 补习第二弹：const 和引用
//
// 你写 kernel 时的 const float* __restrict__ x 和 float* y
// 到底在说什么，今天亲眼看见。

#include <cstdio>

void byValue(int x) {           // 传值：拿到的是复印件
    x = 100;
}

void byPointer(int* p) {        // 传指针：拿到门牌号
    *p = 200;
}

void byReference(int& r) {      // 传引用：拿到的是"别名"
    r = 300;
}

void readOnly(const int* p) {
    printf("readOnly 读到 %d\n", *p);
    // *p = 999;   ← 把这行取消注释试试，编译会报错！
    // 因为 const 说"我指向的东西只读，不许改"
}

int main() {
    int a = 42;

    int* p = &a;                // 普通指针：能改指向，也能通过它改 a
    const int* cp = &a;         // 指向 const 的指针：能通过它读 a，不能改 a
    int* const pc = &a;         // const 指针：指针本身不能改指向，但能改 a

    int& r = a;                 // 引用 = a 的别名，r 就是 a 本人

    printf("&a = %p\n", (void*)&a);
    printf("&r = %p\n", (void*)&r);   // 和 &a 一样！

    readOnly(cp);

    byValue(a);
    printf("byValue 后   a = %d\n", a);   // 42，复印件改了不影响原件

    byPointer(&a);
    printf("byPointer 后 a = %d\n", a);   // 200

    byReference(a);
    printf("byRef 后     a = %d\n", a);   // 300

    return 0;
}
