# CMakeLists.txt —— 整个工程的构建配方
#
# 用法（在仓库根目录）：
#   cmake -B build          配置：读这个文件，生成构建文件到 build/
#   cmake --build build -j  编译：所有目标，-j 表示并行
#   ./build/bin/transpose   运行
#
# 改了 .cu 源码后只需要再跑 cmake --build build（增量编译，只重编改过的）
# 只有改了这个 CMakeLists.txt 才需要重新 cmake -B build

cmake_minimum_required(VERSION 3.21)

# project() 会同时启用 C++ 和 CUDA 两种语言
# LANGUAGES CUDA 这一句让 CMake 知道 .cu 文件要用 nvcc 编
project(cuda_kernels_journey LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 关键一行：目标架构
#
# 不写这一行，CMake 会为 5-6 种架构各编一遍，编译时间翻好几倍。
# RTX 4070 Laptop 是 Ada 架构，compute capability 8.9 → 写 89。
#
# 换机器时改这里：3090/4090 是 86/89，A100 是 80，H100 是 90。
# 也可以写 "native"，让 CMake 自动检测本机显卡。
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 89)

# 所有可执行文件统一放到 build/bin/，不要在源码目录里到处散落
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# 目标 1：纯 C++ 的练习程序（不需要 nvcc）
# ---------------------------------------------------------------------------
add_executable(warmup 00_cpp_warmup/warmup.cpp)

# ---------------------------------------------------------------------------
# 目标 2~N：所有 CUDA kernel，自动发现
#
# file(GLOB) 会去找所有 0X_xxx/ 目录下的 .cu 文件，
# 每个 .cu 编成一个同名可执行文件。以后新加 05_gemm/gemm.cu 不用改这里。
#
# 注意：GLOB 是"配置时"快照，新加了文件要重新跑一次 cmake -B build 才会被发现。
# ---------------------------------------------------------------------------
file(GLOB KERNEL_SOURCES "[0-9][0-9]_*/*.cu")

foreach(src IN LISTS KERNEL_SOURCES)
    get_filename_component(kernel_name ${src} NAME_WE)   # 去掉路径和扩展名

    add_executable(${kernel_name} ${src})

    # 让 #include "common.h" 能找到，等价于 Makefile 里的 -Iinclude
    target_include_directories(${kernel_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # -lineinfo：把源码行号和机器码对应起来，ncu 才能定位到具体哪一行
    target_compile_options(${kernel_name} PRIVATE -lineinfo)
endforeach()

# 配置时打印一下发现了哪些 kernel，方便确认 GLOB 有没有漏
message(STATUS "发现的 kernel 源文件：")
foreach(src IN LISTS KERNEL_SOURCES)
    message(STATUS "  ${src}")
endforeach()
