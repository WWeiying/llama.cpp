# ggml 核心机制教学（从 llama_decode 到一条 RVV 指令）

> 这是 `rvv-kernel-tutorial.md` 的**上层篇**。内核文档讲"一条权重×激活怎么算"，这份讲"它是怎么被调起来的、数据从哪来"。
>
> 配套源码全部**内嵌在文档里**，你只看这份文档即可，不必翻源文件。源码片段为节选（用 `...` 表示省略），行号标注便于你想深挖时定位。
>
> 覆盖层次：数据模型 -> 计算图 -> 后端调度 -> CPU 线程池 -> op 分发 -> mul_mat 内部 -> 接回内核文档。
> 涉及的 C 语法在第 0 章先讲一次，后面遇到只点名不复述。

---

## 第 0 章 · 读 ggml 需要的 C 语法（先过一遍）

ggml 是 C 写的（核心），用到这些你可能不熟的写法。先认一次，后面不再解释。

### 0.1 `typedef struct { ... } Name;`
给匿名结构体起别名，之后 `Name x;` 即可，不用写 `struct Name`。
```c
typedef struct { ggml_half d; int8_t qs[32]; } block_q8_0;
// 等价于 struct {...}; 再 typedef ... block_q8_0;
```

### 0.2 `enum`
一组命名整数常量。ggml 用它给"类型/算子"编号：
```c
enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_F16, GGML_TYPE_Q4_0, ... };
enum ggml_op   { GGML_OP_MUL_MAT, GGML_OP_RMS_NORM, GGML_OP_SOFT_MAX, ... };
// enum 值能当数组下标用 -> 这就是 type_traits[type] 的原理
```

### 0.3 `void *` + 强制类型转换
`void *` 是"任意指针"。ggml 的内核签名用它接收"任意量化类型"的数据，进函数后再 cast 成具体块类型：
```c
void ggml_vec_dot_q8_0_q8_0(..., const void * vx, ..., const void * vy, ...) {
    const block_q8_0 * x = vx;   // 把无类型指针解释成 q8_0 块数组
    const block_q8_0 * y = vy;
}
```
这就是为什么所有 vec_dot 签名长得一样（统一 `void*`），却能处理不同量化 —— 多态靠的是函数指针表（见 0.4）。

### 0.4 函数指针 ★最关键
变量可以存"一个函数的地址"，之后像调用函数一样调用它。ggml 的多态核心：
```c
typedef void (*ggml_vec_dot_t)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
//      返回void  指针名         参数列表
ggml_vec_dot_t f = ggml_vec_dot_q8_0_q8_0;  // 把函数地址存进变量
f(n, s, bs, vx, bx, vy, by, nrc);           // 通过变量调用 = 调用 ggml_vec_dot_q8_0_q8_0
```
**理解这个，就理解了 ggml 怎么"根据张量类型自动选对内核"**：它有一张表，每种类型存着对应的函数指针。

### 0.5 `union` + 匿名结构
联合体：多个成员共用同一块内存。ggml 用它让 `{d, m}` 两个 fp16 既能分开访问、又能当一个 32 位整体：
```c
typedef struct {
    union {
        struct { ggml_half d; ggml_half m; };  // 分开看：两个 fp16
        ggml_half2 dm;                          // 合起来看：一个 32 位
    };
    uint8_t qs[16];
} block_q4_1;
```

### 0.6 `static_assert`
编译期断言，条件不满足直接编译失败。ggml 用它锁死块大小，防止结构体被意外改坏：
```c
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");
```

### 0.7 `restrict` / `GGML_RESTRICT`
告诉编译器"这个指针指向的内存不会和别的指针重叠"，允许更激进的优化（向量化的前提）。`GGML_RESTRICT` 是跨平台宏，等于 `__restrict`。

### 0.8 `size_t` / `int64_t` / `uint8_t`
定宽/平台相关整数类型（来自 `<stdint.h>`）。`size_t`=内存大小/下标（无符号），`int64_t`=64位有符号，`uint8_t`=字节。ggml 张量维度用 `int64_t`，步长用 `size_t`。

### 0.9 原子操作 / 多线程
```c
atomic_store_explicit(&x, v, memory_order_relaxed); // 原子写
ggml_barrier(threadpool);                            // 所有线程在此汇合
```
ggml CPU 后端多线程跑一张图，用原子变量发"任务块"、用 barrier 同步阶段。第 5 章会用到。

---

## 第 1 章 · ggml 是什么

一句话：**ggml = 张量 + 计算图 + 多后端执行器**，是个为推理优化的轻量 C 张量库。

三件事：
1. **描述计算**：你用 `ggml_mul_mat(ctx, a, b)` 等函数搭一张"计算图"（DAG），此时**不算**，只记录"要算什么"。
2. **分配内存**：图建好后，分配器给中间张量安排显存/内存。
3. **执行**：把图交给某个后端（CPU/CUDA/...）真正算出数值。

llama.cpp 就是用 ggml 把 Transformer 拼成图，再让 ggml 算。你的 RVV 内核是第 3 步里 CPU 后端算 `mul_mat` 时调用的叶子函数。

---

## 第 2 章 · 数据模型：`ggml_tensor`

一切的中心。**张量 = 多维数组 + 它是怎么算出来的**。完整结构（`ggml/include/ggml.h:667`）：

```c
struct ggml_tensor {
    enum ggml_type type;              // 数据类型: F32/F16/Q8_0/Q4_K...

    struct ggml_backend_buffer * buffer;  // 数据落在哪个后端的哪块缓冲

    int64_t ne[GGML_MAX_DIMS];        // number of elements，每维元素数 (GGML_MAX_DIMS=4)
    size_t  nb[GGML_MAX_DIMS];        // stride in bytes，每维步长(字节)
                                      // nb[0] = 一个元素(或块)的字节数
                                      // nb[1] = nb[0] * (ne[0]/块大小)
                                      // nb[i] = nb[i-1] * ne[i-1]
    // 计算信息
    enum ggml_op op;                  // 这个张量是哪个算子的输出 (MUL_MAT/ADD/...)
    int32_t op_params[...];           // 算子参数(如 RoPE 的频率、softmax 的 scale)
    int32_t flags;

    struct ggml_tensor * src[GGML_MAX_SRC];  // 输入张量(最多约10个) -> 图的"边"

    struct ggml_tensor * view_src;    // 若是视图(view)，指向原张量
    size_t               view_offs;   // 视图在原张量里的偏移

    void * data;                      // 真正的数据指针
    char name[GGML_MAX_NAME];         // 名字(调试用，如 "blk.0.attn_q.weight")
    void * extra;
    char padding[8];
};
```

### 2.1 `ne[]` 和 `nb[]`：维度与步长

ggml 张量**列优先**，最多 4 维。`ne[0]` 是最内层（连续）维度。

例：一个 `[896, 151936]` 的 Q8_0 权重（lm_head）：
- `ne[0]=896`（每行 896 个权重），`ne[1]=151936`（15 万行）。
- `nb[0]` = 一个 Q8_0 块的字节数 / 块内元素数... 实际 `nb[0]=ggml_type_size(Q8_0)=34`，`nb[1]=nb[0]*(ne[0]/32)=34*28=952`（一行 952 字节）。

**为什么内核签名里有 `bx/by/bs/nb`**：因为内核要按步长跳到下一行。`mul_mat` 把这些步长传给 vec_dot，让它知道每行多少字节。这就是 0.8 里 `size_t nb[]` 的用途。

### 2.2 `op` 和 `src[]`：图就藏在张量里

每个张量记着"我是谁算出来的(`op`)、我的输入是谁(`src[]`)"。顺着 `src` 往回走就是整张计算图。没有单独的"图对象存边"，**边就是 `src` 指针**。

### 2.3 `data` 和 `buffer`：数据在哪

- `data` 指向真正的数字。权重的 `data` 来自 mmap 的 GGUF 文件；中间结果的 `data` 来自分配器给的临时缓冲。
- `buffer` 标明这块数据归哪个后端管（CPU 内存 / CUDA 显存）。

### 2.4 视图(view)：零拷贝切片
`view_src != NULL` 表示这个张量不拥有数据，只是另一个张量的一个"窗口"（共享 `data`，自己的 `ne/nb/offs`）。KV cache 的读写、reshape 全靠它，不复制数据。

---

## 第 3 章 · 计算图与"懒执行"

### 3.1 建图时不算

```c
struct ggml_tensor * cur;
cur = ggml_mul_mat(ctx, model.wq, inpL);   // 只创建一个 op=MUL_MAT 的张量, src[0]=wq, src[1]=inpL
cur = ggml_add(ctx, cur, model.bq);         // 再叠一个 ADD
cur = ggml_rms_norm(ctx, cur, eps);         // ...
```
每个 `ggml_xxx()` 只是 **new 一个张量、填好 `op` 和 `src`**，返回它。数值还没算。llama.cpp 的 `llama_model::build_graph` 就是这样把整个 Transformer 拼出来。

### 3.2 `ggml_cgraph`：把图收集起来

```c
struct ggml_cgraph {
    int n_nodes;
    struct ggml_tensor ** nodes;   // 按拓扑序排好的所有"要计算的"张量
    struct ggml_tensor ** leafs;   // 叶子(权重/输入，不需要计算)
    ...
};
```
`ggml_build_forward_expand(gf, cur)` 从最终输出 `cur` 出发，沿 `src` 回溯，把所有节点按**拓扑序**收进 `gf->nodes`。执行时顺着 `nodes` 一个个算，保证算某节点时它的输入已算好。

### 3.3 op_params：算子的"配置"
有些 op 需要参数（softmax 的缩放、RoPE 的 base、norm 的 eps），存在 `tensor->op_params`（一小块 int32 数组，用时 memcpy 进/出）。

---

## 第 4 章 · 后端与调度（你 backtrace 的上半段）

你在 gdb 里抓到的调用栈，上半段就是这一层。从下往上看：

```
llama_decode                                   // 对外 API: 算一个 batch
 -> llama_context::decode                       // src/llama-context.cpp
 -> llama_context::process_ubatch               // 拆成 micro-batch
 -> llama_context::graph_compute                // 准备好图, 交给后端
 -> ggml_backend_sched_graph_compute_async      // 调度器入口
 -> ggml_backend_sched_compute_splits           // 把图按后端切成若干段
 -> ggml_backend_graph_compute_async
 -> ggml_backend_cpu_graph_compute              // ★ 轮到 CPU 后端
 -> ggml_graph_compute                          // 进入 CPU 线程池(第5章)
```

### 4.1 ggml-backend 抽象
`ggml_backend` 是接口（一组函数指针，又是 0.4 的函数指针！）：每个后端实现 `graph_compute`、`buffer` 分配等。CPU 后端的 `graph_compute` 就是 `ggml_backend_cpu_graph_compute`。

### 4.2 ggml_backend_sched 调度器
当有多个后端（如部分层在 GPU、部分在 CPU），`sched` 负责：
1. 给每个节点**分配后端**；
2. 把图切成**连续同后端的段(split)**；
3. 在后端间**搬数据**（GPU↔CPU），按段依次执行。
纯 CPU 推理时它退化成"整张图一段、全给 CPU"，但调用链还是走它。

---

## 第 5 章 · CPU 后端：线程池 + op 分发

### 5.1 ggml_graph_compute：多线程跑图
`ggml_backend_cpu_graph_compute` -> `ggml_graph_compute`：起 N 个线程，每个线程跑 `ggml_graph_compute_thread`：

```c
// 简化骨架 (ggml-cpu.c:3064 附近)
static thread_ret_t ggml_graph_compute_thread(void * data) {
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {  // 按拓扑序遍历节点
        struct ggml_tensor * node = cgraph->nodes[node_n];
        ggml_compute_forward(&params, node);                     // 算这个节点
        ggml_barrier(params.threadpool);                         // 所有线程算完这个节点再下一个
    }
}
```
- 多个线程**协作算同一个节点**（不是一线程一节点），靠节点内部的分块（第 6 章）切活。
- `ggml_barrier`（0.9）保证算下一个节点前，当前节点全部完工（因为下个节点可能依赖它）。
- 你 `llama-bench -t 1` 用单线程，所以 backtrace 里只有一个 `ggml_graph_compute_thread`。

### 5.2 ggml_compute_forward：按 op 分发
```c
// ggml-cpu.c:1800 附近
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    switch (tensor->op) {
        case GGML_OP_RMS_NORM: ggml_compute_forward_rms_norm(params, tensor); break;
        case GGML_OP_MUL_MAT:  ggml_compute_forward_mul_mat (params, tensor); break;  // ★
        case GGML_OP_SOFT_MAX: ggml_compute_forward_soft_max(params, tensor); break;
        ... // 每个算子一个 case
    }
}
```
一个大 `switch`（0.2 的 enum 当 case），把张量派给对应算子的实现。`mul_mat` 这一支就通向你的内核。

### 5.3 type_traits_cpu：量化类型 -> 内核函数指针表 ★

这是"为什么 Q8_0 权重会调到 `ggml_vec_dot_q8_0_q8_0`"的答案。结构（`ggml-cpu.h:116`）：

```c
struct ggml_type_traits_cpu {
    ggml_from_float_t from_float;     // 把 F32 量化成本类型 (函数指针)
    ggml_vec_dot_t    vec_dot;        // 本类型的点积内核 (函数指针)
    enum ggml_type    vec_dot_type;   // 点积时，激活方要先转成哪种类型
    int64_t           nrows;          // 一次能算几行
};
```

一张按类型索引的表（`ggml-cpu.c:211`，0.2 enum 当下标 + 0.4 函数指针）：

```c
static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q8_0] = {
        .from_float   = quantize_row_q8_0,        // <- 你内核文档 4.1 那个函数!
        .vec_dot      = ggml_vec_dot_q8_0_q8_0,   // <- 你内核文档 4.2 那个函数!
        .vec_dot_type = GGML_TYPE_Q8_0,           // 激活也要先量化成 Q8_0
        .nrows        = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float   = quantize_row_q4_0,
        .vec_dot      = ggml_vec_dot_q4_0_q8_0,   // q4_0 权重, 但激活用 q8_0!
        .vec_dot_type = GGML_TYPE_Q8_0,           // <- 所以 q4_0 的点积是 q4_0 × q8_0
    },
    [GGML_TYPE_Q4_K] = {
        .from_float   = quantize_row_q4_K,
        .vec_dot      = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type = GGML_TYPE_Q8_K,           // K 系列激活用 q8_K
    },
    ... // 每种量化一行
};
```

**三个关键认知**：
1. `vec_dot` 字段就是你内核文档里那些函数 —— **这张表就是"上层"和"内核"的接缝**。
2. `vec_dot_type` 解释了内核命名 `q4_0_q8_0`：权重是 q4_0，但激活先被量化成 q8_0 再点积。所以激活方**总是 q8 系列**（int8 好算）。
3. 取内核：`type_traits_cpu[src0->type].vec_dot` —— 用权重类型查表，拿到函数指针就调。

---

## 第 6 章 · `mul_mat` 内部（你 backtrace 的核心）

`ggml_compute_forward_mul_mat`（`ggml-cpu.c:1245`）算 `dst = src0 × src1`：src0=权重(量化)，src1=激活(F32)。它分**两阶段**，中间一个 barrier。

### 阶段 1：把激活 src1 量化成 vec_dot_type

权重是 Q8_0，但激活 src1 是 F32。要点积，先把 src1 也量化成 `vec_dot_type`(=Q8_0)：

```c
// ggml-cpu.c:1300 附近
if (src1->type != vec_dot_type) {                    // 激活类型 != 点积要求的类型
    char * wdata = params->wdata;                     // 临时工作缓冲(work data)
    ggml_from_float_t const from_float = type_traits_cpu[vec_dot_type].from_float; // = quantize_row_q8_0
    for (int64_t i11 = ith; i11 < ne11; i11 += nth) { // 多线程分行: 线程 ith 处理 ith, ith+nth, ...
        from_float((float *)(src1->data + ...),       // 输入: 一行 F32 激活
                   (void  *)(wdata     + ...),         // 输出: 量化进 wdata
                   ne10);                              // 长度
    }
}
ggml_barrier(params->threadpool);                     // ★ 等所有线程量化完
```
- `from_float` 又是函数指针(0.4)，对 Q8_0 就是 `quantize_row_q8_0`（内核文档 4.1）。
- **这解释了为什么 `quantize_row_q8_0` 会在 perf 里排很高**：每次 mul_mat 都要量化整个激活矩阵。
- `wdata` 是每次算 mul_mat 临时借的缓冲，存量化后的激活。

### 阶段 2：分块，每块循环调 vec_dot

```c
// ggml-cpu.c:1400 附近
// 把输出矩阵 dst 切成 nchunk0 × nchunk1 块, 线程用原子计数抢块
int current_chunk = ith;
while (current_chunk < nchunk0 * nchunk1) {
    // 算出本块负责的行列范围 ir0_start..ir0_end, ir1_start..ir1_end
    ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type,
            num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);
    current_chunk = atomic_fetch_add(&params->threadpool->current_chunk, 1); // 抢下一块
}
```

`one_chunk` 内部就是双重循环调 vec_dot（`ggml-cpu.c:1234`，你 backtrace 的 #1 帧）：

```c
// 简化
ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;  // = ggml_vec_dot_q8_0_q8_0
const void * wdata = params->wdata;                            // 量化后的激活
for (int64_t ir1 = ir1_start; ir1 < ir1_end; ir1++) {         // 遍历激活列
    for (int64_t ir0 = ir0_start; ir0 < ir0_end; ir0++) {     // 遍历权重行
        vec_dot(ne00, &dst[...], 0,
                (const char*)src0->data + ir0*nb01,            // 一行权重
                (const char*)wdata      + ir1*row_size,        // 一列量化激活
                0, 1);                                          // -> 你的 RVV 内核!
    }
}
```

**到这里就接上内核文档了**：`vec_dot(...)` = `ggml_vec_dot_q8_0_q8_0(n=896, s, ..., vx=权重行, vy=激活列, ..., nrc=1)`，正是你 gdb 断点停的地方。`num_rows_per_vec_dot`(=backtrace 里的 1) 就是 `nrc`。

### 阶段 1+2 全景

```
mul_mat(权重Q8_0, 激活F32):
  [阶段1] 每行激活 F32 --quantize_row_q8_0--> wdata(Q8_0)   ← from_float 函数指针
          ggml_barrier  (等全部量化完)
  [阶段2] 分块, 每(行权重×列激活):
            ggml_vec_dot_q8_0_q8_0(权重行, wdata里的激活列)  ← vec_dot 函数指针
          原子抢下一块
```

---

## 第 7 章 · 完整链路（一张图记住）

```
llama_decode(batch)                              对外 API
  llama_context::decode / process_ubatch         拆 micro-batch
    llama_model::build_graph                      [建图] 拼出 Transformer 的 ggml DAG (第3章)
    graph_compute
      ggml_backend_sched_*                        [调度] 切分/选后端 (第4章)
        ggml_backend_cpu_graph_compute
          ggml_graph_compute                      [线程池] N 线程 (第5.1)
            ggml_graph_compute_thread
              ggml_compute_forward (switch op)    [分发] 按 op 派发 (第5.2)
                ggml_compute_forward_mul_mat      [算子] (第6章)
                  阶段1 from_float=quantize_row_q8_0   ← 内核文档 4.1
                  阶段2 vec_dot=ggml_vec_dot_q8_0_q8_0 ← 内核文档 4.2
                          └─ vsetvli/vle8/vwmul/vwredsum  ← 你 gdb 看的指令
```

从一个 token 的 `llama_decode`，到一条 `vwmul.vv`，全程打通。

---

## 第 8 章 · 这份文档"没"覆盖什么（诚实边界）

掌握上面 1-6 层 + 内核文档，你能完全看懂 **CPU 推理的计算路径**，足以改/测 RVV 内核。以下属于 ggml 更外围或其他方向，**对你 RVV 目标非必需**，需要时再单独学：

| 主题 | 在哪 | 何时需要 |
|---|---|---|
| 建图细节（每个 op 怎么拼 Transformer）| `src/llama-model.cpp build_graph` | 想加新算子/改网络结构时 |
| GGUF 加载、权重 mmap | `src/llama-model-loader.cpp` | 想懂权重怎么进内存时 |
| 内存分配器 `ggml_gallocr` | `ggml/src/ggml-alloc.c` | 想懂中间张量复用时 |
| 其他算子内部（RoPE/softmax/norm）| `ggml-cpu/ops.cpp` | 想优化非 mul_mat 算子时（也有 RVV 空间）|
| 量化框架（ref 实现、round）| `ggml/src/ggml-quants.c` | 想懂量化数值细节时 |
| 其他后端 CUDA/Metal/Vulkan | `ggml/src/ggml-*` | 跨后端对比时 |
| 训练 / autograd | `ggml/src/ggml-opt.cpp` | 你不需要（纯推理）|

需要哪块，我可以照这份文档的风格（内嵌源码 + C 语法）继续补。

---

## 附录 · 本文 C 语法速查

| 语法 | 用途 | 章节 |
|---|---|---|
| `typedef struct{}` | 结构体别名 | 0.1 |
| `enum` | 命名整数常量，可当数组下标 | 0.2 |
| `void *` + cast | 统一接口接任意类型，进函数再解释 | 0.3 |
| **函数指针** `(*f)()` | ggml 多态核心：表里存函数地址 | 0.4 |
| `union` + 匿名结构 | 同内存多视角看 | 0.5 |
| `static_assert` | 编译期锁结构大小 | 0.6 |
| `restrict` | 指针不重叠，助向量化 | 0.7 |
| `size_t/int64_t/uint8_t` | 定宽整数（步长/维度/字节）| 0.8 |
| 原子 + barrier | CPU 多线程协作 | 0.9 |

**一句话主线**：ggml 用 **enum 编号类型/算子**，用**函数指针表**(type_traits)把"类型"映射到"内核函数"，用 **switch** 把"算子"映射到"算子实现"，用 **void* + 块结构体** 统一处理各种量化。看懂这三个映射，就看懂了 ggml 的骨架。
