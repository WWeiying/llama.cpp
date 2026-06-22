## llama.cpp

### 整体流程

```bash
用户运行：
  build/bin/llama-cli -m model.gguf -p "Hello" -n 1

入口：
  tools/cli/cli.cpp
    main()
    common_params_parse()
    llama_backend_init()
    ctx_cli.ctx_server.load_model(params)
    inference_thread -> ctx_cli.ctx_server.start_loop()
    ctx_cli.generate_completion()

server/任务系统：
  tools/server/server-context.cpp
    server_context_impl
    load_model()
      common_init_from_params()
      model_tgt = llama_init->model()
      ctx_tgt   = llama_init->context()
    slot / queue / task / batch
    eventually calls llama_decode()

llama 核心：
  src/llama-context.cpp
    llama_decode()
      ctx->decode(batch)
    llama_context::decode()
    llama_context::process_ubatch()
      model.build_graph(...)
      graph_compute(...)

graph 构建：
  src/llama-model.cpp
    llama_model::build_graph
```



## 3. `src/llama-context.cpp`

目的：知道 `llama_decode()` 后面发生什么。

只看这些关键词：

```
grep -n "llama_decode" src/llama-context.cpp
grep -n "llama_context::decode" src/llama-context.cpp
grep -n "process_ubatch" src/llama-context.cpp
grep -n "build_graph" src/llama-context.cpp
grep -n "graph_compute" src/llama-context.cpp
```

看完你要知道：

```
llama_decode → decode → process_ubatch → build_graph → graph_compute
```

------

## 4. `src/llama-model.cpp`

目的：知道根据模型架构如何选择 graph builder。

只看这些关键词：

```
grep -n "build_graph" src/llama-model.cpp
grep -n "llm_build" src/llama-model.cpp
grep -n "LLM_ARCH" src/llama-model.cpp
```

看完你要知道：

```
llama_model::build_graph 会根据模型架构分发到具体模型实现。
```

------

## 5. `src/models/llama.cpp`

目的：看 LLaMA-like 模型结构如何搭成 ggml graph。

只看这些关键词：

```
grep -n "ggml_mul_mat" src/models/llama.cpp
grep -n "ggml_rms_norm\|ggml_norm" src/models/llama.cpp
grep -n "ggml_rope" src/models/llama.cpp
grep -n "ggml_soft_max" src/models/llama.cpp
grep -n "ggml_silu\|ggml_add\|ggml_mul" src/models/llama.cpp
```

看完你要知道：

```
Transformer block 被拆成 RMSNorm、mul_mat、RoPE、Softmax、FFN 等 ggml op。
```





### 围绕 CVA6/Ara/RVV 目标，看懂 llama.cpp + ggml 推理主路径、量化格式、矩阵乘热点，并能改/测一个 RVV 或自定义 kernel。

```
llama.cpp 推理主流程
ggml 计算图
ggml CPU/RVV backend
量化 matmul / vec_dot kernel

理解 Transformer 到 ggml 算子的映射
重点看：
RMSNorm
Q/K/V projection
RoPE
Attention
FFN gate/up/down
lm_head
KV cache
```



| 优先级 | 模块                           | 为什么看                     |
| ------ | ------------------------------ | ---------------------------- |
| 最高   | `llama_decode` / 推理主路径    | 理解一次 token 如何生成      |
| 最高   | `ggml_compute_forward_mul_mat` | Transformer 最大热点         |
| 最高   | `ggml_vec_dot_q8_0_q8_0`       | 最适合 Ara/RVV 第一个 kernel |
| 最高   | `ggml_vec_dot_q4/q5/q6_*`      | 量化模型关键热点             |
| 高     | GGUF 加载                      | 理解模型权重怎么进内存       |
| 高     | KV cache                       | 理解 decode 阶段             |
| 中     | tokenizer / sampler            | 知道作用即可                 |
| 低     | server                         | 暂时不看                     |
| 低     | multimodal                     | 暂时不看                     |
| 低     | grammar                        | 暂时不看                     |



  你接下来最有效的阅读路径是：先读 llama.cpp/examples/simple/simple.cpp:80，再对照 llama.cpp/include/llama.h:1，然后进入 src/llama-model.cpp、sr
  c/llama-context.cpp、src/llama-graph.cpp。这条线能最快理解“模型怎么加载、prompt 怎么变 token、token 怎么进计算图、下一个 token 怎么采样出来”。





下面是重新整理后的 **1–5 完整新版**，尽量少用环境变量，去掉重复内容，按你当前记录收敛成一套可执行流程。整理依据是你贴的现有命令和调试记录。

------

## 1. Rocky Linux 跑 llama.cpp

```bash
###############################################################################
# 1. Rocky Linux 跑 llama.cpp
###############################################################################

sudo dnf update -y
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git cmake python3 python3-pip wget curl which openssl-devel

mkdir -p /home/wangwy/llama
cd /home/wangwy/llama

# 如果还没有源码：
# git clone https://github.com/ggml-org/llama.cpp

cd /home/wangwy/llama/llama.cpp

rm -rf build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

ls -lh build/bin/llama-cli
ls -lh build/bin/llama-bench
###############################################################################
# 下载 Qwen2.5-0.5B Q8 GGUF
###############################################################################

python3 -m pip install --user -U huggingface_hub

export PATH="$HOME/.local/bin:$PATH"

mkdir -p /home/wangwy/llama/models

hf download bartowski/Qwen2.5-0.5B-Instruct-GGUF \
  --include "Qwen2.5-0.5B-Instruct-Q8_0.gguf" \
  --local-dir /home/wangwy/llama/models

ls -lh /home/wangwy/llama/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf
###############################################################################
# x86 运行 Qwen Q8
###############################################################################

cd /home/wangwy/llama/llama.cpp

./build/bin/llama-cli \
  -m /home/wangwy/llama/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -p "Hello, introduce yourself briefly." \
  -n 64 \
  -c 512 \
  -t $(nproc)

./build/bin/llama-cli \
  -m /home/wangwy/llama/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -p "用三句话解释什么是 RISC-V 向量扩展。" \
  -n 128 \
  -c 512 \
  -t $(nproc)

./build/bin/llama-bench \
  -m /home/wangwy/llama/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -t 1,2,4,8
```

------

## 2. perf 调试

```bash
###############################################################################
# 2. perf 调试
###############################################################################

sudo dnf install -y perf kernel-tools valgrind

cd /home/wangwy/llama/llama.cpp

rm -rf build-prof

cmake -S . -B build-prof \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O2 -g -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -fno-omit-frame-pointer"

cmake --build build-prof --config RelWithDebInfo -j$(nproc)

file build-prof/bin/llama-cli
###############################################################################
# perf record
###############################################################################

cd /home/wangwy/llama/llama.cpp

# 如果权限不足：
# cat /proc/sys/kernel/perf_event_paranoid
# sudo sysctl kernel.perf_event_paranoid=1
# sudo sysctl kernel.perf_event_paranoid=0

perf record -F 999 --call-graph fp \
  -- ./build-prof/bin/llama-cli \
  -m /home/wangwy/llama/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -p "Hello" \
  -n 64 \
  -c 128 \
  -t 1 \
  --no-warmup

ls -lh perf.data
###############################################################################
# perf report / annotate
###############################################################################

cd /home/wangwy/llama/llama.cpp

perf report

perf report --stdio --no-children > perf-self.txt
perf report --stdio --children > perf-callgraph.txt

grep -E "ggml|llama|mul_mat|vec_dot|q4|q5|q6|q8|rope|rms|soft" \
  perf-self.txt | head -100

perf annotate --stdio --symbol=ggml_compute_forward_mul_mat | less
perf annotate --stdio --symbol=ggml_vec_dot_q8_0_q8_0 | less
```

常见热点一般会在：

```text
ggml_compute_forward_mul_mat
ggml_vec_dot_q8_0_q8_0
ggml_vec_dot_q5_0_q8_0
ggml_vec_dot_q6_K_q8_K
ggml_gemm_q8_0_16x1_q8_0
```

------

## 3. QEMU 标量版本

```bash
###############################################################################
# 3. QEMU 标量版本：Buildroot / C++ / QEMU Linux
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

# 如果 buildroot 子模块缺失：
# git submodule update --init --recursive

unset LD_LIBRARY_PATH

make -C buildroot \
  BR2_EXTERNAL="../br2-ext-tree" \
  BR2_DEFCONFIG=../configs/qemu/buildroot64_defconfig \
  defconfig

# 确认 C++ 支持
grep -E "BR2_TOOLCHAIN_BUILDROOT_CXX|BR2_INSTALL_LIBSTDCPP" buildroot/.config

# 如果没有 C++，进入 menuconfig：
# make -C buildroot BR2_EXTERNAL="../br2-ext-tree" menuconfig
# Toolchain 里打开 Enable C++ support

make XLEN=64 BOARD=qemu

ls install64_qemu/fw_dynamic.bin
ls install64_qemu/Image
ls install64_qemu/rootfs.cpio
ls buildroot/output/host/bin/qemu-system-riscv64
ls buildroot/output/host/bin/riscv64-linux-gcc
ls buildroot/output/host/bin/riscv64-linux-g++
###############################################################################
# 生成标量 toolchain 文件
###############################################################################

mkdir -p /home/wangwy/llama/toolchains

cat > /home/wangwy/llama/toolchains/riscv64-cva6-scalar.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER "/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-gcc")
set(CMAKE_CXX_COMPILER "/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-g++")

set(CMAKE_C_FLAGS "-march=rv64gc -mabi=lp64d")
set(CMAKE_CXX_FLAGS "-march=rv64gc -mabi=lp64d")
EOF

cat /home/wangwy/llama/toolchains/riscv64-cva6-scalar.cmake
###############################################################################
# 编译标量 llama.cpp
###############################################################################

cd /home/wangwy/llama/llama.cpp

rm -rf build-rv64-cva6-scalar

cmake -S . -B build-rv64-cva6-scalar \
  -DCMAKE_TOOLCHAIN_FILE=/home/wangwy/llama/toolchains/riscv64-cva6-scalar.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DGGML_RVV=OFF \
  -DGGML_RV_ZFH=OFF \
  -DGGML_RV_ZVFH=OFF \
  -DGGML_RV_ZICBOP=OFF \
  -DGGML_RV_ZIHINTPAUSE=OFF \
  -DGGML_OPENMP=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build-rv64-cva6-scalar --target llama-completion -j$(nproc)
cmake --build build-rv64-cva6-scalar --target llama-bench -j$(nproc)

file build-rv64-cva6-scalar/bin/llama-completion
file build-rv64-cva6-scalar/bin/llama-bench
###############################################################################
# 打包标量程序和模型
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

mkdir -p rootfs/opt/llama

cp /home/wangwy/llama/llama.cpp/build-rv64-cva6-scalar/bin/llama-completion \
  rootfs/opt/llama/llama-completion

cp /home/wangwy/llama/llama.cpp/build-rv64-cva6-scalar/bin/llama-bench \
  rootfs/opt/llama/llama-bench

cp /home/wangwy/llama/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  rootfs/opt/llama/

chmod +x rootfs/opt/llama/llama-completion
chmod +x rootfs/opt/llama/llama-bench

unset LD_LIBRARY_PATH
make XLEN=64 BOARD=qemu
###############################################################################
# 启动 QEMU 标量版本
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

buildroot/output/host/bin/qemu-system-riscv64 \
  -M virt \
  -cpu rv64 \
  -m 2G \
  -nographic \
  -bios install64_qemu/fw_dynamic.bin \
  -initrd install64_qemu/rootfs.cpio \
  -kernel install64_qemu/Image \
  -append "rootwait root=/dev/ram ro console=ttyS0"
```

QEMU guest 里运行：

```sh
cd /opt/llama

./llama-completion \
  -m Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -p "Hello" \
  -n 4 \
  -c 64 \
  -t 1 \
  --no-mmap \
  --no-warmup

./llama-bench \
  -m Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -t 1
```

------

## 4. QEMU RVV 版本

```bash
###############################################################################
# 4. QEMU RVV 版本：打开 Buildroot RVV
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

unset LD_LIBRARY_PATH

make -C buildroot \
  BR2_EXTERNAL="../br2-ext-tree" \
  BR2_DEFCONFIG=../configs/qemu/buildroot64_defconfig \
  defconfig

make -C buildroot BR2_EXTERNAL="../br2-ext-tree" menuconfig
```

菜单里确认：

```bash
Target options
  RISC-V instruction set extensions
    [*] Vector extension / RVV / V

Toolchain
  [*] Enable C++ support
```

保存后：

```bash
grep -E "BR2_RISCV_ISA.*V|BR2_TOOLCHAIN_BUILDROOT_CXX|BR2_INSTALL_LIBSTDCPP" \
  buildroot/.config

make -C buildroot \
  BR2_EXTERNAL="../br2-ext-tree" \
  BR2_DEFCONFIG=../configs/qemu/buildroot64_defconfig \
  savedefconfig
###############################################################################
# Linux kernel 打开 RVV
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

make -C buildroot BR2_EXTERNAL="../br2-ext-tree" linux-menuconfig

# 在 kernel menuconfig 中搜索：
# /RISCV_ISA_V
# 打开 RISC-V Vector extension support

unset LD_LIBRARY_PATH

make -C buildroot clean
make XLEN=64 BOARD=qemu

grep CONFIG_RISCV_ISA_V buildroot/output/build/linux-*/.config
###############################################################################
# 生成 RVV toolchain 文件
###############################################################################

mkdir -p /home/wangwy/llama/toolchains

cat > /home/wangwy/llama/toolchains/riscv64-cva6-rvv.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER "/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-gcc")
set(CMAKE_CXX_COMPILER "/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-g++")

set(CMAKE_C_FLAGS "-march=rv64gcv -mabi=lp64d")
set(CMAKE_CXX_FLAGS "-march=rv64gcv -mabi=lp64d")
EOF

/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-gcc \
  -march=rv64gcv \
  -mabi=lp64d \
  -x c /dev/null \
  -c \
  -o /tmp/rvv_check.o

file /tmp/rvv_check.o
###############################################################################
# 编译 RVV llama.cpp
###############################################################################

cd /home/wangwy/llama/llama.cpp

rm -rf build-rv64-cva6-rvv

cmake -S . -B build-rv64-cva6-rvv \
  -DCMAKE_TOOLCHAIN_FILE=/home/wangwy/llama/toolchains/riscv64-cva6-rvv.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DGGML_RVV=ON \
  -DGGML_OPENMP=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build-rv64-cva6-rvv --target llama-completion -j$(nproc)
cmake --build build-rv64-cva6-rvv --target llama-bench -j$(nproc)

file build-rv64-cva6-rvv/bin/llama-completion

/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-objdump \
  -d build-rv64-cva6-rvv/bin/llama-completion \
  | grep -E "vsetvli|vle|vse|vfmul|vfwmul|vfmacc|vwmul" \
  | head -50
###############################################################################
# 编译 RVV 测试程序
###############################################################################

cat > /tmp/rvv_test.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

int main(void) {
    int8_t a[16];
    int8_t b[16];
    int16_t c[16];

    for (int i = 0; i < 16; i++) {
        a[i] = i;
        b[i] = i + 1;
        c[i] = 0;
    }

    size_t vl = __riscv_vsetvl_e8m1(16);

    vint8m1_t va = __riscv_vle8_v_i8m1(a, vl);
    vint8m1_t vb = __riscv_vle8_v_i8m1(b, vl);
    vint16m2_t vc = __riscv_vwmul_vv_i16m2(va, vb, vl);

    __riscv_vse16_v_i16m2(c, vc, vl);

    printf("RVV test OK, vl=%zu\n", vl);

    for (int i = 0; i < 16; i++) {
        printf("%d ", c[i]);
    }

    printf("\n");
    return 0;
}
EOF

/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-gcc \
  -O2 \
  -march=rv64gcv \
  -mabi=lp64d \
  /tmp/rvv_test.c \
  -o /tmp/rvv_test

file /tmp/rvv_test
###############################################################################
# 打包 RVV 程序、测试程序和模型
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

mkdir -p rootfs/opt/llama
mkdir -p rootfs/opt/rvv

cp /home/wangwy/llama/llama.cpp/build-rv64-cva6-rvv/bin/llama-completion \
  rootfs/opt/llama/llama-completion-rvv

cp /home/wangwy/llama/llama.cpp/build-rv64-cva6-rvv/bin/llama-bench \
  rootfs/opt/llama/llama-bench-rvv

cp /home/wangwy/llama/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  rootfs/opt/llama/

cp /tmp/rvv_test rootfs/opt/rvv/

chmod +x rootfs/opt/llama/llama-completion-rvv
chmod +x rootfs/opt/llama/llama-bench-rvv
chmod +x rootfs/opt/rvv/rvv_test

unset LD_LIBRARY_PATH
make XLEN=64 BOARD=qemu
###############################################################################
# 启动 QEMU RVV
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

buildroot/output/host/bin/qemu-system-riscv64 \
  -M virt \
  -cpu rv64,v=true,zfh=true,zfhmin=true,zvfh=true,zvfhmin=true,vlen=1024,elen=64 \
  -m 2G \
  -nographic \
  -bios install64_qemu/fw_dynamic.bin \
  -initrd install64_qemu/rootfs.cpio \
  -kernel install64_qemu/Image \
  -append "rootwait root=/dev/ram ro console=ttyS0"
```

QEMU guest 里运行：

```sh
cat /proc/cpuinfo

/opt/rvv/rvv_test

cd /opt/llama

./llama-completion-rvv \
  -m Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -p "Hello" \
  -n 4 \
  -c 64 \
  -t 1 \
  --no-mmap \
  --no-warmup

./llama-bench-rvv \
  -m Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -t 1
```

------

## 5. GDB 调试

这里给的是你最后选择的方式：**把源码打进 rootfs，在 QEMU guest 内直接用完整 gdb 调试**。

```bash
###############################################################################
# 5. GDB 调试：Buildroot 打开完整 gdb
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

unset LD_LIBRARY_PATH

make -C buildroot \
  BR2_EXTERNAL="../br2-ext-tree" \
  BR2_DEFCONFIG=../configs/qemu/buildroot64_defconfig \
  defconfig

make -C buildroot BR2_EXTERNAL="../br2-ext-tree" menuconfig
```

菜单里确认：

```bash
Target packages
  Debugging, profiling and benchmark
    gdb
      [*] full debugger
      [*] gdbserver
```

保存后检查：

```bash
grep -E "BR2_PACKAGE_GDB|BR2_PACKAGE_GDB_DEBUGGER|BR2_PACKAGE_GDB_SERVER" \
  buildroot/.config

make -C buildroot \
  BR2_EXTERNAL="../br2-ext-tree" \
  BR2_DEFCONFIG=../configs/qemu/buildroot64_defconfig \
  savedefconfig

make -C buildroot BR2_EXTERNAL="../br2-ext-tree" gdb-dirclean
make -C buildroot BR2_EXTERNAL="../br2-ext-tree" gdb

find buildroot/output/target -name "gdb" -o -name "gdbserver"
###############################################################################
# 编译 RVV 调试版 llama.cpp
###############################################################################

cd /home/wangwy/llama/llama.cpp

rm -rf build-rv64-cva6-rvv-gdb

cmake -S . -B build-rv64-cva6-rvv-gdb \
  -DCMAKE_TOOLCHAIN_FILE=/home/wangwy/llama/toolchains/riscv64-cva6-rvv.cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O1 -g -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -fno-omit-frame-pointer" \
  -DGGML_NATIVE=OFF \
  -DGGML_RVV=ON \
  -DGGML_OPENMP=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build-rv64-cva6-rvv-gdb --target llama-completion -j$(nproc)
cmake --build build-rv64-cva6-rvv-gdb --target llama-bench -j$(nproc)

/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-readelf \
  -S build-rv64-cva6-rvv-gdb/bin/llama-completion \
  | grep debug
###############################################################################
# 打包源码、调试程序和模型
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

rm -rf rootfs/home/wangwy/llama/llama.cpp

mkdir -p rootfs/home/wangwy/llama/llama.cpp
mkdir -p rootfs/opt/llama

# 源码只用于 gdb 显示源码行，不复制 build、models、perf.data。
rsync -a /home/wangwy/llama/llama.cpp/src      rootfs/home/wangwy/llama/llama.cpp/
rsync -a /home/wangwy/llama/llama.cpp/ggml     rootfs/home/wangwy/llama/llama.cpp/
rsync -a /home/wangwy/llama/llama.cpp/include  rootfs/home/wangwy/llama/llama.cpp/
rsync -a /home/wangwy/llama/llama.cpp/common   rootfs/home/wangwy/llama/llama.cpp/
rsync -a /home/wangwy/llama/llama.cpp/examples rootfs/home/wangwy/llama/llama.cpp/

cp /home/wangwy/llama/llama.cpp/build-rv64-cva6-rvv-gdb/bin/llama-completion \
  rootfs/opt/llama/llama-completion-rvv-gdb

cp /home/wangwy/llama/llama.cpp/build-rv64-cva6-rvv-gdb/bin/llama-bench \
  rootfs/opt/llama/llama-bench-rvv-gdb

cp /home/wangwy/llama/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  rootfs/opt/llama/

chmod +x rootfs/opt/llama/llama-completion-rvv-gdb
chmod +x rootfs/opt/llama/llama-bench-rvv-gdb

du -sh rootfs/home/wangwy/llama/llama.cpp
ls -lh rootfs/opt/llama

unset LD_LIBRARY_PATH
make XLEN=64 BOARD=qemu
###############################################################################
# 启动 QEMU 调试环境
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

buildroot/output/host/bin/qemu-system-riscv64 \
  -M virt \
  -cpu rv64,v=true,zfh=true,zfhmin=true,zvfh=true,zvfhmin=true,vlen=1024,elen=64 \
<<<<<<< HEAD
  -m 2G \
  -nographic \
  -bios install64_qemu/fw_dynamic.bin \
  -initrd install64_qemu/rootfs.cpio \
  -kernel install64_qemu/Image \
  -append "rootwait root=/dev/ram ro console=ttyS0"

# 如果你的 QEMU 不认识 vlen / elen 参数，改用：
#
# buildroot/output/host/bin/qemu-system-riscv64 \
#   -M virt \
#   -cpu rv64,v=true \
#   -m 1G \
#   -nographic \
#   -bios install64_qemu/fw_dynamic.bin \
#   -initrd install64_qemu/rootfs.cpio \
#   -kernel install64_qemu/Image \
#   -append "rootwait root=/dev/ram ro console=ttyS0"


###############################################################################
# 19. 以下命令在 QEMU guest 里面执行
###############################################################################

# 登录：
#
#   buildroot login: root
#   Password: 直接回车


# 19.1 检查 Linux 是否看到 V 扩展
cat /proc/cpuinfo

# 看 isa 字段里有没有 v。
# 希望看到类似：
#
#   rv64imafdcv
#
# 或者 isa 字段中包含 v。


# 19.2 跑最小 RVV 测试程序
/opt/rvv/rvv_test

# 成功应看到：
#
#   RVV test OK
#   vl = ...
#
# 如果报 Illegal instruction，说明 RVV 没真正跑起来。


# 19.3 跑 RVV 版 llama.cpp
cd /opt/llama

ls -lh

./llama-completion-rvv \
  -m Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -p "Hello" \
  -n 4 \
  -c 64 \
  -t 1 \
  --no-mmap \
  --no-warmup


# 19.4 跑 RVV 版 benchmark
./llama-bench-rvv \
  -m SmolLM2-135M-Instruct-Q4_K_M.gguf \
  -t 1


###############################################################################
# 20. 常见问题排查
###############################################################################

# 问题 1：Buildroot .config 里没有 RVV
#
# host 侧：
#
#   cd /home/wangwy/openproject/cva6/cva6-sdk
#   make -C buildroot BR2_EXTERNAL="../br2-ext-tree" menuconfig
#
# 菜单里搜索：
#
#   /RVV
#   /RISCV_ISA_RVV
#   /Vector
#
# 打开后保存，再执行：
#
#   make -C buildroot BR2_EXTERNAL="../br2-ext-tree" \
#     BR2_DEFCONFIG=../configs/qemu/buildroot64_defconfig \
#     savedefconfig


# 问题 2：gcc 不支持 -march=rv64gcv
#
# 检查：
#
#   grep -E "BR2_RISCV_ISA.*V" buildroot/.config
#
# 如果没有 RVV，回到 menuconfig 打开。
#
# 如果有 RVV 但还是不支持，建议：
#
#   make -C buildroot clean
#   make XLEN=64 BOARD=qemu


# 问题 3：QEMU guest 里 rvv_test 报 Illegal instruction
#
# 检查三件事：
#
#   1. QEMU 启动参数有：
#        -cpu rv64,v=true
#
#   2. Linux kernel config 有：
#        CONFIG_RISCV_ISA_V=y
#
#   3. guest 里：
#        cat /proc/cpuinfo
#      isa 字段有 v


# 问题 4：llama.cpp 二进制没有 RVV 指令
#
# 检查：
#
#   CMake 里必须有：
#     -DGGML_RVV=ON
#
#   toolchain 文件里必须有：
#     -march=rv64gcv
#
#   objdump 里应该能 grep 到：
#     vsetvli
#     vle
#     vse
#     vmul


###############################################################################
# 21. 成功标准
###############################################################################

# 成功标准 1：
#   buildroot/.config 有 BR2_RISCV_ISA_RVV 或 CUSTOM_RVV
#
# 成功标准 2：
#   kernel .config 有 CONFIG_RISCV_ISA_V=y
#
# 成功标准 3：
#   QEMU guest 里 cat /proc/cpuinfo 能看到 isa 包含 v
#
# 成功标准 4：
#   /opt/rvv/rvv_test 能运行
#
# 成功标准 5：
#   /opt/llama/llama-completion-rvv 能生成 token
#
# 成功标准 6：
#   /opt/llama/llama-bench-rvv 能跑完
#
# 注意：
#   QEMU RVV 只验证 RVV 软件流程。
#   它不代表 Ara RTL 性能，也不代表真实 CVA6 + Ara 的时序和带宽。
```

## QEMU + GDB + llama.cpp

```bash
###############################################################################
# 0. 方案说明
###############################################################################

# 这个方案使用 QEMU system-mode 自带 gdb stub。
#
# 优点：
#   不需要 guest 里安装 gdbserver
#   可以调 kernel / user 程序 / illegal instruction
#   适合你现在的 QEMU RVV llama.cpp 调试
#
# 缺点：
#   它调的是“整台虚拟机”，不是某个进程
#   用户态符号需要手动处理
#   比 gdbserver 稍微麻烦
#
# 推荐做法：
#   1. 把 llama.cpp 编译成 non-PIE，地址固定
#   2. QEMU 启动时加 -s -S
#   3. host gdb 连接 :1234
#   4. gdb 里设置 hbreak llama_decode 等断点
#   5. continue 让 Linux 启动
#   6. 在 QEMU guest 里运行 llama-completion-rvv-gdb
#   7. host gdb 会停在断点处


###############################################################################
# 1. 编译一个适合 system gdb stub 调试的 RVV llama.cpp
###############################################################################

cd /home/wangwy/llama/llama.cpp

rm -rf build-rv64-cva6-rvv-gdb

cmake -S . -B build-rv64-cva6-rvv-gdb \
  -DCMAKE_TOOLCHAIN_FILE=/home/wangwy/llama/toolchains/riscv64-cva6-rvv.cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O2 -g -fno-omit-frame-pointer -fno-pie" \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -fno-omit-frame-pointer -fno-pie" \
  -DCMAKE_EXE_LINKER_FLAGS="-no-pie" \
  -DGGML_NATIVE=OFF \
  -DGGML_RVV=ON \
  -DGGML_OPENMP=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build-rv64-cva6-rvv-gdb --target llama-completion -j$(nproc)
cmake --build build-rv64-cva6-rvv-gdb --target llama-bench -j$(nproc)

file build-rv64-cva6-rvv-gdb/bin/llama-completion
file build-rv64-cva6-rvv-gdb/bin/llama-bench

# 检查是否是 non-PIE。
# 期望 Type 是 EXEC。
# 如果是 DYN，说明还是 PIE，后面符号地址会麻烦。
/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-readelf \
  -h build-rv64-cva6-rvv-gdb/bin/llama-completion \
  | grep Type

# 检查是否有调试信息。
/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-readelf \
  -S build-rv64-cva6-rvv-gdb/bin/llama-completion \
  | grep debug

# 检查是否有 RVV 指令。
/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-objdump \
  -d build-rv64-cva6-rvv-gdb/bin/llama-completion \
  | grep -E "vsetvli|vle|vse|vfmul|vfwmul|vfmacc|vwmul" \
  | head -50


###############################################################################
# 2. 拷贝调试版程序和模型到 rootfs overlay
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

mkdir -p rootfs/opt/llama

cp /home/wangwy/llama/llama.cpp/build-rv64-cva6-rvv-gdb/bin/llama-completion \
  rootfs/opt/llama/llama-completion-rvv-gdb

cp /home/wangwy/llama/llama.cpp/build-rv64-cva6-rvv-gdb/bin/llama-bench \
  rootfs/opt/llama/llama-bench-rvv-gdb

# 调试时先用小模型，别一上来调 Qwen Q8。
cp /home/wangwy/llama/models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  rootfs/opt/llama/

chmod +x rootfs/opt/llama/llama-completion-rvv-gdb
chmod +x rootfs/opt/llama/llama-bench-rvv-gdb

ls -lh rootfs/opt/llama


###############################################################################
# 3. 重新打包 QEMU rootfs
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

unset LD_LIBRARY_PATH

make XLEN=64 BOARD=qemu


###############################################################################
# 4. 启动 QEMU system-mode，并打开 gdb stub
###############################################################################

cd /home/wangwy/openproject/cva6/cva6-sdk

buildroot/output/host/bin/qemu-system-riscv64 \
  -M virt \
  -cpu rv64,v=true,zfh=true,zfhmin=true,zvfh=true,zvfhmin=true,vlen=1024,elen=64 \
  -m 2G \
  -nographic \
  -bios install64_qemu/fw_dynamic.bin \
  -initrd install64_qemu/rootfs.cpio \
  -kernel install64_qemu/Image \
  -append "rootwait root=/dev/ram ro console=ttyS0" \
  -s -S

# 说明：
#
#   -s
#     等价于：
#       -gdb tcp::1234
#
#   -S
#     QEMU 启动后先暂停 CPU，等待 gdb 连接。
#
# 此时 QEMU 窗口不会继续启动 Linux，这是正常的。
# 需要在 host 另一个终端连接 gdb 后执行 continue。


###############################################################################
# 5. host 新开一个终端，启动 riscv64-linux-gdb
###############################################################################

cd /home/wangwy/llama/llama.cpp

/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-gdb \
  build-rv64-cva6-rvv-gdb/bin/llama-completion

# 进入 gdb 后输入下面命令。


###############################################################################
# 6. 以下命令在 host gdb 里面执行
###############################################################################

set pagination off
set print pretty on
set disassemble-next-line on

# 连接 QEMU system gdb stub。
target remote :1234

# 因为这是 system-mode，先连上的是整机 CPU。
# 现在 Linux 还没启动，因为 QEMU 用了 -S。
#
# 先设置硬件断点。用 hbreak 比 break 更适合 system-mode，
# 因为用户程序此时还没加载到内存，软件断点可能插不进去。
#
# 先不要断 main，容易和其他程序混淆。
# 断 llama.cpp 独有函数更稳。
hbreak llama_decode
hbreak ggml_compute_forward_mul_mat
hbreak ggml_gemm_q8_0_16x1_q8_0

# 查看断点
info breakpoints

# 继续，让 QEMU Linux 开始启动。
continue

# 此时回到 QEMU 终端，等待 Linux 启动到 login。


###############################################################################
# 7. 以下命令在 QEMU guest 里面执行
###############################################################################

# 登录：
#   buildroot login: root
#   Password: 直接回车

# 先确认 RVV / ZVFH 是否暴露。
cat /proc/cpuinfo

# 建议关闭 ASLR，避免用户程序地址随机化。
# 如果前面已经编成 non-PIE，这一步仍然建议做。
echo 0 > /proc/sys/kernel/randomize_va_space

# 进入 llama 目录。
cd /opt/llama

ls -lh

# 开始运行调试版程序。
# 注意：一旦命中 gdb 断点，QEMU 终端会卡住，
# 这时要回到 host gdb 终端操作。
./llama-completion-rvv-gdb \
  -m SmolLM2-135M-Instruct-Q4_K_M.gguf \
  -p "Hello" \
  -n 1 \
  -c 32 \
  -t 1 \
  --no-mmap \
  --no-warmup


###############################################################################
# 8. 断点命中后，在 host gdb 里面操作
###############################################################################

# 如果命中 llama_decode，host gdb 会显示停在该函数。
# 常用命令如下：

bt

info args

info locals

list

x/20i $pc

info registers

continue

# 如果命中 ggml_compute_forward_mul_mat：
bt
x/20i $pc
continue

# 如果命中 ggml_gemm_q8_0_16x1_q8_0：
bt
x/30i $pc
disassemble
continue


###############################################################################
# 9. 如果想单步进入源码
###############################################################################

# 下一行，不进入函数：
next

# 进入函数：
step

# 执行完当前函数返回：
finish

# 查看当前源码附近：
list

# 查看当前汇编附近：
x/20i $pc

# 查看当前函数反汇编：
disassemble


###############################################################################
# 10. 如果程序触发 Illegal instruction
###############################################################################

# 如果非法指令发生时 gdb 停住，执行：

bt

x/20i $pc

info registers

disassemble

# 重点看 $pc 当前指令。
#
# 例如：
#
#   fcvt.h.s
#   fsh
#   flh
#
# 这类是 Zfh / 半精度标量浮点。
#
# 例如：
#
#   vfwmul.vf
#   vfmacc.vv
#   vfcvt.f.x.v
#
# 这类是 RVV vector float / vector half。
#
# 如果真实 CVA6/Ara 不支持这些扩展，后续要么：
#   1. 硬件支持对应扩展；
#   2. 修改 ggml RVV backend，避开这些指令。


###############################################################################
# 11. 如果 hbreak 没有命中怎么办？
###############################################################################

# 原因可能是：
#   1. 程序仍然是 PIE，地址随机化了；
#   2. 断点设置得太早，QEMU system gdb stub 没能匹配；
#   3. 函数被内联或没被调用；
#   4. 当前运行路径没走到这个函数。
#
# 处理方法 A：确认程序是 non-PIE。
#
# host 上执行：
/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-readelf \
  -h /home/wangwy/llama/llama.cpp/build-rv64-cva6-rvv-gdb/bin/llama-completion \
  | grep Type

# 期望：
#   Type: EXEC
#
# 如果是：
#   Type: DYN
#
# 说明它还是 PIE，需要重新检查 CMake 里的：
#   -fno-pie
#   -no-pie
#
# 处理方法 B：先在 QEMU guest 里运行程序，
# 然后在 host gdb 里按 Ctrl+C 暂停整机 CPU。
#
# host gdb 里：
#   Ctrl+C
#   bt
#   x/20i $pc
#
# 如果刚好停在 llama 程序里，再设置普通 break 或 hbreak。
#
# 处理方法 C：用固定地址断点。
#
# host 上查函数地址：
/home/wangwy/openproject/cva6/cva6-sdk/buildroot/output/host/bin/riscv64-linux-nm \
  -C /home/wangwy/llama/llama.cpp/build-rv64-cva6-rvv-gdb/bin/llama-completion \
  | grep "llama_decode"

# 假设输出里地址是：
#   0000000000123456 T llama_decode
#
# gdb 里可以：
#   hbreak *0x123456


###############################################################################
# 12. 如果程序仍然是 PIE，如何调？
###############################################################################

# 如果 readelf 显示 Type: DYN，说明是 PIE。
# 那需要知道运行时加载基地址。
#
# 在 QEMU guest 里先运行程序，并让它跑久一点。
# 可以在另一个阶段用：
#
#   cat /proc/$(pidof llama-completion-rvv-gdb)/maps
#
# 找到类似：
#
#   555555554000-555555900000 r-xp ... /opt/llama/llama-completion-rvv-gdb
#
# 这个起始地址就是程序 text 映射基址。
#
# 然后 host gdb 里：
#
#   add-symbol-file /home/wangwy/llama/llama.cpp/build-rv64-cva6-rvv-gdb/bin/llama-completion 0x555555554000
#
# 再设置断点：
#
#   hbreak llama_decode
#
# 但这个流程比较麻烦，所以更推荐第 1 步直接编成 non-PIE。


###############################################################################
# 13. 调试完退出
###############################################################################

# host gdb 里：
quit

# 如果提示：
#   A debugging session is active.
#   Quit anyway? (y or n)
#
# 输入：
y

# QEMU 终端里可以 Ctrl+C 或关掉 QEMU。


###############################################################################
# 14. 推荐第一次只做这个最小调试流程
###############################################################################

# host 终端 1：
#
#   启动 QEMU：
#     qemu-system-riscv64 ... -s -S
#
# host 终端 2：
#
#   riscv64-linux-gdb build-rv64-cva6-rvv-gdb/bin/llama-completion
#
# gdb 里：
#
#   target remote :1234
#   hbreak llama_decode
#   continue
#
# QEMU guest 里：
#
#   echo 0 > /proc/sys/kernel/randomize_va_space
#   cd /opt/llama
#   ./llama-completion-rvv-gdb -m SmolLM2-135M-Instruct-Q4_K_M.gguf -p "Hello" -n 1 -c 32 -t 1 --no-mmap --no-warmup
#
# host gdb 命中后：
#
#   bt
#   x/20i $pc
#   continue
```



## Qwen2.5-0.5B Q8 下载、x86 验证、QEMU 验证

```bash
###############################################################################
# 0. 路径约定
###############################################################################

# llama.cpp 源码
export LLAMA_SRC=/home/wangwy/llama/llama.cpp

# 模型目录
export MODEL_DIR=/home/wangwy/llama/models

# Qwen2.5-0.5B Q8 模型路径
export QWEN_Q8_MODEL=/home/wangwy/llama/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf

# cva6-sdk 路径
export CVA6_SDK=/home/wangwy/openproject/cva6/cva6-sdk


###############################################################################
# 1. x86 主机：下载 Qwen2.5-0.5B-Instruct Q8_0 GGUF
###############################################################################

mkdir -p "$MODEL_DIR"

# 确保 hf 命令存在
python3 -m pip install --user -U huggingface_hub

export PATH="$HOME/.local/bin:$PATH"

which hf
hf --help

# 推荐下载 bartowski 版本，文件名清楚，大小约 0.53GB
hf download bartowski/Qwen2.5-0.5B-Instruct-GGUF \
  --include "Qwen2.5-0.5B-Instruct-Q8_0.gguf" \
  --local-dir "$MODEL_DIR"

# 检查模型
ls -lh "$QWEN_Q8_MODEL"


###############################################################################
# 2. x86 主机：用 llama-cli 测试 Qwen Q8
###############################################################################

cd "$LLAMA_SRC"

# 如果还没编译 x86 版本，则先编译
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build --config Release -j$(nproc)

# 最小英文测试
./build/bin/llama-cli \
  -m "$QWEN_Q8_MODEL" \
  -p "Hello, introduce yourself briefly." \
  -n 64 \
  -c 512 \
  -t $(nproc)

# 中文测试
./build/bin/llama-cli \
  -m "$QWEN_Q8_MODEL" \
  -p "用三句话解释什么是 RISC-V 向量扩展。" \
  -n 128 \
  -c 512 \
  -t $(nproc)

# 交互模式
./build/bin/llama-cli \
  -m "$QWEN_Q8_MODEL" \
  -cnv \
  -c 512 \
  -n 128 \
  -t $(nproc)


###############################################################################
# 3. x86 主机：Qwen Q8 benchmark
###############################################################################

cd "$LLAMA_SRC"

./build/bin/llama-bench \
  -m "$QWEN_Q8_MODEL" \
  -t 1,2,4,8


###############################################################################
# 4. QEMU/CVA6：确认你已经交叉编译了 riscv64 llama.cpp
###############################################################################

# 前面你已经走过这个流程：
#   build-rv64-cva6-scalar/bin/llama-completion
#   build-rv64-cva6-scalar/bin/llama-bench
#
# 确认它们存在并且是 RISC-V ELF：

cd "$LLAMA_SRC"

file build-rv64-cva6-scalar/bin/llama-completion
file build-rv64-cva6-scalar/bin/llama-bench

# 应该看到类似：
# ELF 64-bit LSB executable, UCB RISC-V


###############################################################################
# 5. QEMU/CVA6：把 Qwen Q8 模型和 riscv64 程序放进 rootfs overlay
###############################################################################

cd "$CVA6_SDK"

mkdir -p rootfs/opt/llama

# 拷贝 riscv64 程序
cp "$LLAMA_SRC/build-rv64-cva6-scalar/bin/llama-completion" \
  rootfs/opt/llama/

cp "$LLAMA_SRC/build-rv64-cva6-scalar/bin/llama-bench" \
  rootfs/opt/llama/

# 拷贝 Qwen2.5-0.5B Q8 模型
cp "$QWEN_Q8_MODEL" \
  rootfs/opt/llama/

chmod +x rootfs/opt/llama/llama-completion
chmod +x rootfs/opt/llama/llama-bench

ls -lh rootfs/opt/llama


###############################################################################
# 6. QEMU/CVA6：重新打包 rootfs
###############################################################################

cd "$CVA6_SDK"

unset LD_LIBRARY_PATH

make XLEN=64 BOARD=qemu


###############################################################################
# 7. 启动 QEMU
###############################################################################

cd "$CVA6_SDK"

buildroot/output/host/bin/qemu-system-riscv64 \
  -M virt \
  -cpu rv64 \
=======
>>>>>>> 1b5d990d833d99142a9fd724e9e2ec20bc09ccf3
  -m 2G \
  -nographic \
  -bios install64_qemu/fw_dynamic.bin \
  -initrd install64_qemu/rootfs.cpio \
  -kernel install64_qemu/Image \
  -append "rootwait root=/dev/ram ro console=ttyS0"
```

QEMU guest 里：

```bash
which gdb
gdb --version

ls -lh /home/wangwy/llama/llama.cpp/src/llama-context.cpp
ls -lh /opt/llama/llama-completion-rvv-gdb

cd /opt/llama

<<<<<<< HEAD
ls -lh

# 最小推理，先用很小参数验证能跑
./llama-completion \
  -m Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -p "Hello" \
  -n 4 \
  -c 64 \
  -t 1 \
  --no-mmap \
  --no-warmup

# 稍微长一点
./llama-completion \
  -m Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -p "用一句话解释什么是 RISC-V。" \
  -n 16 \
  -c 128 \
  -t 1 \
  --no-mmap \
  --no-warmup

# benchmark
./llama-bench \
  -m Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -t 1
  
=======
gdb ./llama-completion-rvv-gdb
>>>>>>> 1b5d990d833d99142a9fd724e9e2ec20bc09ccf3
```

gdb 里：

```bash
set pagination off
set print pretty on
set print thread-events off
set disassemble-next-line on

directory /home/wangwy/llama/llama.cpp
directory /home/wangwy/llama/llama.cpp/src
directory /home/wangwy/llama/llama.cpp/ggml/src
directory /home/wangwy/llama/llama.cpp/ggml/src/ggml-cpu

break llama_decode
break ggml_compute_forward_mul_mat
break ggml_vec_dot_q8_0_q8_0

run \
  -m Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -p "Hello" \
  -n 1 \
  -c 32 \
  -t 1 \
  --no-mmap \
  --no-warmup
```

断点命中后：

```bash
bt
list
info args
info locals
n
s
continue
```

看 RVV kernel 时：

```bash
bt
x/40i $pc
disassemble
info registers
ni
```

## 简短提醒

```bash
1. x86 用 llama-cli；QEMU riscv64 当前用 llama-completion。
2. QEMU 标量版用于验证 riscv64 Linux + llama.cpp 基础流程。
3. QEMU RVV 版用于验证 RVV 软件流程，不代表真实 CVA6/Ara 性能。
4. RVV 版 llama.cpp 可能用到 zfh / zvfh，所以 QEMU 启动参数里保留 zfh、zvfh。
5. gdb 调试时先用 -n 1 -c 32，避免 QEMU 太慢。
```
