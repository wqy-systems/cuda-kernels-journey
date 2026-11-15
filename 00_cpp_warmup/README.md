# C++ 最小集（只为读懂 CUDA 代码）

不是 C++ 教程。只讲读懂 `saxpy.cu` 需要的五件事，一小时内搞定。

## 怎么用

```bash
cd ~/dev/cuda-kernels-journey
g++ -std=c++17 -o 00_cpp_warmup/warmup 00_cpp_warmup/warmup.cpp
./00_cpp_warmup/warmup
```

先跑一遍，**对照屏幕输出读代码里的注释**。五段跑完，回来做下面的任务。

## 五个概念，一句话各一条

| 概念 | 一句话 | 在 saxpy.cu 里对应 |
|---|---|---|
| 地址与指针 | `&x` 取门牌号，`*p` 按门牌号上门 | `cudaMalloc(&d_x, bytes)` |
| 数组即指针 | `a[i]` 本质是 `*(a + i)`，往后走 i 户 | kernel 里的 `x[i]`、`y[i]` |
| 指针传参 | 想让函数改你的变量，就得把地址交给它 | 同上，所有 `&` |
| const 位置 | 在 `*` 左边管内容，在 `*` 右边管指针 | `const float* x` / `float* y` |
| 循环 | 从 0 开始，条件成立就继续 | `for (int i = idx; i < n; i += stride)` |

## 动手任务（做完才算过）

改 `warmup.cpp`，每改一个就重新编译跑一次。

**任务 1｜看完地址差**
把 `float arr[5]` 改成 `double arr[5]`，重新编译。`arr[1]` 的门牌号比 `arr[0]` 大多少？
→ 答案应该变成 8。为什么？（提示：`sizeof(double)`）

**任务 2｜验证 `a[i]` 就是 `*(a+i)`**
加一行 `printf("%.1f %.1f\n", arr[2], *(arr + 2));`，看两个数是否相同。

**任务 3｜亲手踩一次传值的坑**
把 `changeByValue(b)` 改成 `changeByPointer(b)`（去掉 `&`），编译。
→ 会报错。**把报错原文抄到下面**，然后想想编译器在提醒你什么。

```
（这里贴报错）
```

**任务 4｜数组求和**
写一个函数 `float arraySum(float* data, int n)`，用 for 循环把数组加起来，返回结果。
在 main 里调用它算 `data` 的和，看结果是不是 17.5。
→ 这个函数的结构，跟 kernel 里那个 for 循环一模一样，只是没有 stride。

**任务 5｜用 stride 遍历**
写一个循环，只遍历下标为偶数的元素累加（0、2、4），也就是 `i += 2`。
→ 这就是 stride 的用法。把 `i += 2` 换成 `i += stride`，就是 kernel 干的事。

## 三个自查问题

1. `cudaMalloc(&d_x, bytes)` 如果写成 `cudaMalloc(d_x, bytes)`，会发生什么？
2. `const float* __restrict__ x` 里的 `const` 是在管什么？为什么 `y` 没有？
3. kernel 里 `i += stride` 换成 `i++` 会怎样？（先猜，后面我们跑实验验证）

答不上来就回去重跑一遍 warmup，或者直接问我。

## 完成后

```bash
git add 00_cpp_warmup
git commit -m "W02: C++ 最小集练习"
git push
```

然后回到 `02_saxpy/saxpy.cu`，你会发现能读懂了。
