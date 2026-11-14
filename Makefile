# 编译规则 —— 4070 Laptop 是 sm_89
NVCC      := nvcc
ARCH      := -arch=sm_89
NVCCFLAGS := -O3 -std=c++17 $(ARCH) -lineinfo -Xcompiler -Wall -Iinclude

# 自动收集所有子目录下的 .cu（排除以 _ 开头的临时文件）
SRCS := $(shell find . -name '*.cu' -not -name '_*' | sort)
BINS := $(patsubst %.cu,%,$(SRCS))

.PHONY: all clean list

all: $(BINS)

# 一条通用规则：a/b.cu -> a/b
%: %.cu include/common.h
	@echo "==> 编译 $<"
	$(NVCC) $(NVCCFLAGS) -o $@ $<

list:
	@echo "可编译目标："
	@for b in $(BINS); do echo "  $$b"; done

clean:
	rm -f $(BINS)
	rm -f *.ncu-rep *.nsys-rep
