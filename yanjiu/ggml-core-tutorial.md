# ggml 核心机制教学（从 llama_decode 到一条 RVV 指令）· 专业版

> 这是 `rvv-kernel-tutorial.md` 的**上层篇**。内核文档讲"一条权重×激活怎么算"，这份讲"它怎么被调起来、数据从哪来、内存怎么管"。
>
> 配套源码全部**内嵌文档**，你只看这份即可，不必翻源文件。源码为节选（`...` 表省略），行号便于深挖。
>
> 覆盖：C 语法 -> 数据模型 -> 计算图 -> 后端调度 -> CPU 线程池 -> op 分发 -> mul_mat 内部 -> 非 matmul 算子 -> 内存/buffer -> 接回内核文档。
> 涉及的 C 语法第 0 章讲一次，后面遇到只点名。

---

## 第 0 章 · 读 ggml 需要的 C 语法

ggml 核心是 C，用到这些写法。先认一次，后面不复述。

### 0.1 `typedef struct { ... } Name;`
给匿名结构体起别名，之后 `Name x;` 即可。
```c
typedef struct { ggml_half d; int8_t qs[32]; } block_q8_0;
```

### 0.2 `enum`
一组命名整数常量，**能当数组下标**。ggml 用它给类型/算子编号：
```c
enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_F16, GGML_TYPE_Q4_0, ... };
enum ggml_op   { GGML_OP_MUL_MAT, GGML_OP_RMS_NORM, GGML_OP_SOFT_MAX, ... };
// 这就是 type_traits[type] 和 switch(op) 的基础
```

### 0.3 `void *` + 强制类型转换
`void *` 是"任意指针"。内核签名用它接收任意量化类型，进函数再 cast 成具体块：
```c
void ggml_vec_dot_q8_0_q8_0(..., const void * vx, ..., const void * vy, ...) {
    const block_q8_0 * x = vx;   // 把无类型指针解释成 q8_0 块数组
}
```
所有 vec_dot 签名统一（`void*`），多态靠函数指针表（0.4）。

### 0.4 函数指针 ★最关键
变量可存"函数地址"，像函数一样调用。**ggml 多态的核心**：
```c
typedef void (*ggml_vec_dot_t)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
//      返回void  指针类型名      参数列表
ggml_vec_dot_t f = ggml_vec_dot_q8_0_q8_0;  // 存函数地址
f(n, s, bs, vx, bx, vy, by, nrc);           // 通过变量调用 = 调用那个函数
```
理解它 = 理解 ggml 怎么"按张量类型自动选对内核"：有张表，每种类型存对应函数指针。

### 0.5 `union` + 匿名结构
多成员共用内存。ggml 让 `{d,m}` 既能分开访问、又能当 32 位整体：
```c
typedef struct {
    union { struct { ggml_half d; ggml_half m; }; ggml_half2 dm; };
    uint8_t qs[16];
} block_q4_1;
```

### 0.6 `static_assert`
编译期断言，不满足直接编译失败。锁死块大小防改坏：
```c
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");
```

### 0.7 `restrict` / `GGML_RESTRICT`
告诉编译器"此指针不与别的指针重叠"，允许激进优化（向量化前提）。`GGML_RESTRICT` = `__restrict`。

### 0.8 定宽整数
`size_t`=内存大小/下标(无符号)，`int64_t`=64位有符号(张量维度)，`size_t nb[]`=步长，`uint8_t/int8_t`=字节。来自 `<stdint.h>`。

### 0.9 原子 / 多线程
```c
atomic_fetch_add(&x, 1);          // 原子自增(抢任务块)
ggml_barrier(threadpool);         // 所有线程在此汇合(阶段同步)
```

### 0.10 宏：`GGML_ASSERT` 与 `..._LOCALS`
```c
GGML_ASSERT(cond);   // 断言，失败打印并中止(release 也保留, 不像 assert)
GGML_TENSOR_BINARY_OP_LOCALS  // 展开成一堆局部变量声明: ne00,ne01..., nb00,nb01...
                              // 把 src0/src1/dst 的 ne[]/nb[] 拆成具名变量, 省得到处写下标
```
看到 `ne00` 就是 `src0->ne[0]`，`nb12` 是 `src1->nb[2]`，`ne0` 是 `dst->ne[0]`。这套命名贯穿 ggml-cpu。

### 0.11 `const` 正确性 / 三元 / `constexpr`
`const T *` = 不可改的数据；`cond ? a : b` 三元表达式；C++ 部分用 `if constexpr`(编译期分支，模板里裁掉不用的代码)。ops.cpp 是 C++。

---

## 第 1 章 · ggml 是什么

一句话：**ggml = 张量 + 计算图 + 多后端执行器**，为推理优化的轻量 C/C++ 张量库。

三步走：
1. **描述计算**：用 `ggml_mul_mat(ctx,a,b)` 等搭一张"计算图"(DAG)，此时**不算**，只记录"要算什么"。
2. **分配内存**：图建好后，分配器给中间张量安排内存（可复用）。
3. **执行**：把图交给后端(CPU/CUDA/...)真正算出数值。

llama.cpp 用 ggml 把 Transformer 拼成图再算。你的 RVV 内核是第 3 步里 CPU 后端算 `mul_mat` 时调用的叶子函数。

---

## 第 2 章 · 数据模型：`ggml_tensor`

一切的中心。**张量 = 多维数组 + 它是怎么算出来的**。完整结构（`ggml/include/ggml.h:667`）：

```c
struct ggml_tensor {
    enum ggml_type type;              // 数据类型: F32/F16/Q8_0/Q4_K...
    struct ggml_backend_buffer * buffer;  // 数据落在哪个后端的缓冲

    int64_t ne[GGML_MAX_DIMS];        // 每维元素数 (GGML_MAX_DIMS=4)
    size_t  nb[GGML_MAX_DIMS];        // 每维步长(字节)

    enum ggml_op op;                  // 这个张量是哪个算子的输出
    int32_t op_params[...];           // 算子参数(RoPE 频率, softmax scale, norm eps)
    int32_t flags;
    struct ggml_tensor * src[GGML_MAX_SRC];  // 输入张量 -> 图的"边"

    struct ggml_tensor * view_src;    // 若是视图, 指向原张量
    size_t               view_offs;   // 视图偏移
    void * data;                      // 真正的数据指针
    char name[GGML_MAX_NAME];         // 名字(如 "blk.0.attn_q.weight")
    void * extra; char padding[8];
};
```

### 2.1 `ne[]`/`nb[]`：维度与步长（含内存布局算例）

ggml 张量**列优先**、最多 4 维，`ne[0]` 是最内层连续维度。步长规则：

```
nb[0] = ggml_type_size(type)               // 一个元素(或一个量化块)的字节数
nb[1] = nb[0] * (ne[0] / 块大小)            // 跳一整行
nb[i] = nb[i-1] * ne[i-1]                   // 高维以此类推
```

**算例**：lm_head 权重 `[896, 151936]` 的 Q8_0 张量（ne[0]=896, ne[1]=151936）：
```
块大小=32, ggml_type_size(Q8_0)=34 字节(=2 fp16 缩放 + 32 int8)
nb[0] = 34
nb[1] = 34 * (896/32) = 34 * 28 = 952 字节  (一行 896 个权重占 952 字节)

内存里(列优先, 一行连续)：
  行0:  [块0:34B][块1:34B]...[块27:34B]   共 952B
  行1:  紧接其后...
  访问第 r 行: (char*)data + r*nb[1]
```

**这就是内核签名里 `bx/by/bs` 的来历**：`mul_mat` 把 `nb01`(权重行步长) 等传给 vec_dot，让它知道跳到下一行要走多少字节。0.8 的 `size_t nb[]` 在此落地。

### 2.2 `op`/`src[]`：图藏在张量里
每个张量记着"我是谁算出来的(`op`)、输入是谁(`src[]`)"。顺 `src` 回溯就是整张图。**没有单独的边对象，边就是 `src` 指针**。

### 2.3 `data`/`buffer`：数据在哪
- `data` 指向真正的数字。权重的 `data` 来自 mmap 的 GGUF；中间结果来自分配器(第 9 章)。
- `buffer` 标明数据归哪个后端管（CPU 内存 / CUDA 显存）。

### 2.4 视图(view)：零拷贝切片
`view_src != NULL` 表示不拥有数据，只是另一个张量的"窗口"（共享 `data`，自己的 `ne/nb/offs`）。例：从 KV cache 取一段：
```c
struct ggml_tensor * k = ggml_view_2d(ctx, kv_cache, n_embd, n_kv, nb1, offset);
// k->data = kv_cache->data; k->view_offs=offset; 不复制
```
reshape/permute/KV 读写全靠它，不搬数据。

---

## 第 3 章 · 计算图与"懒执行"

### 3.1 建图时不算
```c
struct ggml_tensor * cur;
cur = ggml_mul_mat(ctx, model.wq, inpL);  // 只 new 一个 op=MUL_MAT 的张量, src[0]=wq, src[1]=inpL
cur = ggml_add(ctx, cur, model.bq);        // 再叠 ADD
```
每个 `ggml_xxx()` 只是 **new 一个张量、填 `op` 和 `src`**，返回它。数值没算。

### 3.2 建图实例：一个注意力块的骨架
`llama_model::build_graph`(src/llama-model.cpp) 大致这样拼（简化）：
```c
// inpL = 本层输入 [n_embd, n_tokens]
cur = build_rms_norm(inpL, model.attn_norm);      // RMSNorm
Qcur = ggml_mul_mat(ctx, model.wq, cur);          // Q 投影  -> MUL_MAT(你的内核!)
Kcur = ggml_mul_mat(ctx, model.wk, cur);          // K 投影
Vcur = ggml_mul_mat(ctx, model.wv, cur);          // V 投影
Qcur = ggml_rope(ctx, Qcur, inp_pos, ...);        // 位置编码
// ... 写入 KV cache(view), 算 attention scores(mul_mat), softmax, 加权 V(mul_mat) ...
cur  = ggml_mul_mat(ctx, model.wo, cur);          // 输出投影
cur  = ggml_add(ctx, cur, inpL);                  // 残差
// FFN: rms_norm -> gate/up(mul_mat) -> silu -> down(mul_mat) -> 残差
```
**关键观察**：一层 Transformer = 7 个 `ggml_mul_mat`(q/k/v/o + gate/up/down) + norm/rope/softmax/add。`mul_mat` 是绝对热点 -> 你的 RVV vec_dot 被调最多。整个模型 = 这样的层堆叠 + 最后 lm_head(又一个大 mul_mat)。

### 3.3 `ggml_cgraph`：收集成图
```c
struct ggml_cgraph {
    int n_nodes;
    struct ggml_tensor ** nodes;   // 拓扑序排好的待计算张量
    struct ggml_tensor ** leafs;   // 叶子(权重/输入)
};
```
`ggml_build_forward_expand(gf, cur)` 从输出 `cur` 沿 `src` 回溯，把节点按**拓扑序**收进 `nodes`。执行时顺序算，保证算某节点时输入已就绪。

### 3.4 op_params
需参数的 op(softmax scale、RoPE base、norm eps)把参数存 `tensor->op_params`(小 int32 数组)，算时 `memcpy` 出来(你会在第 7 章 RMSNorm 看到 `memcpy(&eps, ...op_params...)`)。

---

## 第 4 章 · 后端与调度（你 backtrace 的上半段）

gdb 抓到的调用栈上半段就是这层。从下往上：
```
llama_decode                                   对外 API: 算一个 batch
 -> llama_context::decode                       src/llama-context.cpp
 -> llama_context::process_ubatch               拆 micro-batch
 -> llama_context::graph_compute                准备图, 交后端
 -> ggml_backend_sched_graph_compute_async      调度器入口
 -> ggml_backend_sched_compute_splits           把图按后端切段
 -> ggml_backend_graph_compute_async
 -> ggml_backend_cpu_graph_compute              ★ 轮到 CPU 后端
 -> ggml_graph_compute                          进 CPU 线程池(第5章)
```

### 4.1 ggml-backend 抽象
`ggml_backend` 是接口（一组函数指针，又是 0.4！）：每个后端实现 `graph_compute`、`buffer` 分配等。CPU 后端的 `graph_compute` = `ggml_backend_cpu_graph_compute`。

### 4.2 ggml_backend_sched 调度器
多后端时（部分层 GPU、部分 CPU），`sched` 负责：① 给每个节点分配后端；② 把图切成连续同后端的**段(split)**；③ 段间搬数据(GPU↔CPU)。纯 CPU 推理时退化成"整图一段全给 CPU"，但调用链仍走它。

---

## 第 5 章 · CPU 后端：线程池 + op 分发

### 5.1 ggml_graph_compute：多线程跑图
```c
// 简化骨架 (ggml-cpu.c:3064 附近)
static thread_ret_t ggml_graph_compute_thread(void * data) {
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {  // 按拓扑序遍历节点
        struct ggml_tensor * node = cgraph->nodes[node_n];
        ggml_compute_forward(&params, node);                     // 算这个节点
        ggml_barrier(params.threadpool);                         // 全线程算完再下一个
    }
}
```
- **多线程协作算同一个节点**（不是一线程一节点），靠节点内部分块(第 6 章)切活。
- `ggml_barrier`(0.9) 保证算下个节点前当前节点全完工（下个可能依赖它）。
- 你 `llama-bench -t 1` 单线程, 所以 backtrace 只有一个 `ggml_graph_compute_thread`。

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
大 `switch`(0.2 enum 当 case)把张量派给对应算子。`mul_mat` 一支通向你的内核。

### 5.3 type_traits_cpu：量化类型 -> 内核函数指针表 ★
"为什么 Q8_0 权重会调到 `ggml_vec_dot_q8_0_q8_0`"的答案。结构(`ggml-cpu.h:116`)：
```c
struct ggml_type_traits_cpu {
    ggml_from_float_t from_float;     // 把 F32 量化成本类型 (函数指针)
    ggml_vec_dot_t    vec_dot;        // 本类型的点积内核 (函数指针)
    enum ggml_type    vec_dot_type;   // 点积时, 激活方要先转成哪种类型
    int64_t           nrows;          // 一次能算几行
};
```
按类型索引的表(`ggml-cpu.c:211`, enum 当下标 + 函数指针)：
```c
static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q8_0] = {
        .from_float   = quantize_row_q8_0,        // <- 内核文档 4.1!
        .vec_dot      = ggml_vec_dot_q8_0_q8_0,   // <- 内核文档 4.2!
        .vec_dot_type = GGML_TYPE_Q8_0,           // 激活也要先量化成 Q8_0
        .nrows        = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float   = quantize_row_q4_0,
        .vec_dot      = ggml_vec_dot_q4_0_q8_0,   // q4_0 权重, 但激活用 q8_0!
        .vec_dot_type = GGML_TYPE_Q8_0,           // 所以是 q4_0 × q8_0
    },
    [GGML_TYPE_Q4_K] = {
        .vec_dot      = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type = GGML_TYPE_Q8_K,           // K 系列激活用 q8_K
    },
    ... // 每种量化一行
};
```
**三个关键认知**：
1. `vec_dot` 字段就是内核文档里那些函数 —— **这张表是"上层"和"内核"的接缝**。
2. `vec_dot_type` 解释内核命名 `q4_0_q8_0`：权重 q4_0，激活先量化成 q8_0 再点积。激活方**总是 q8 系列**(int8 好算)。
3. 取内核：`type_traits_cpu[src0->type].vec_dot`，拿到函数指针就调。

---

## 第 6 章 · `mul_mat` 内部（你 backtrace 的核心）

`ggml_compute_forward_mul_mat`(`ggml-cpu.c:1245`)算 `dst = src0 × src1`：src0=权重(量化)，src1=激活(F32)。分**两阶段**，中间一个 barrier。

### 阶段 1：把激活 src1 量化成 vec_dot_type
```c
// ggml-cpu.c:1300 附近
if (src1->type != vec_dot_type) {                    // 激活类型 != 点积要求的类型
    char * wdata = params->wdata;                     // 临时工作缓冲
    ggml_from_float_t const from_float = type_traits_cpu[vec_dot_type].from_float; // = quantize_row_q8_0
    for (int64_t i11 = ith; i11 < ne11; i11 += nth) { // 多线程分行
        from_float((float *)(src1->data + ...),       // 输入: 一行 F32 激活
                   (void  *)(wdata     + ...), ne10);  // 输出: 量化进 wdata
    }
}
ggml_barrier(params->threadpool);                     // ★ 等所有线程量化完
```
- `from_float` 又是函数指针，对 Q8_0 = `quantize_row_q8_0`(内核文档 4.1)。
- **这解释了为什么 `quantize_row_q8_0` 在 perf 里排很高**：每次 mul_mat 都量化整个激活矩阵。

### 阶段 2：分块，每块循环调 vec_dot
```c
// ggml-cpu.c:1400 附近
int current_chunk = ith;
while (current_chunk < nchunk0 * nchunk1) {           // 输出矩阵切块, 线程原子抢块
    // 算本块行列范围...
    ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type,
            num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);
    current_chunk = atomic_fetch_add(&params->threadpool->current_chunk, 1); // 抢下一块
}
```
`one_chunk` 内部双重循环调 vec_dot(`ggml-cpu.c:1234`, 你 backtrace 的 #1 帧)：
```c
ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;  // = ggml_vec_dot_q8_0_q8_0
const void * wdata = params->wdata;                            // 量化后的激活
for (int64_t ir1 = ir1_start; ir1 < ir1_end; ir1++)           // 遍历激活列
    for (int64_t ir0 = ir0_start; ir0 < ir0_end; ir0++)       // 遍历权重行
        vec_dot(ne00, &dst[...], 0,
                (const char*)src0->data + ir0*nb01,            // 一行权重
                (const char*)wdata      + ir1*row_size, 0, 1); // 一列量化激活 -> 你的内核!
```

**接上内核文档**：`vec_dot(...)` = `ggml_vec_dot_q8_0_q8_0(n=896, s, ..., vx=权重行, vy=激活列, ..., nrc=1)`，正是你 gdb 停的地方。`num_rows_per_vec_dot`(=backtrace 里的 1) 就是 `nrc`。

### 全景
```
mul_mat(权重Q8_0, 激活F32):
  [阶段1] 每行激活 F32 --quantize_row_q8_0--> wdata(Q8_0)   ← from_float 函数指针
          ggml_barrier  (等全部量化完)
  [阶段2] 分块, 每(行权重×列激活):
            ggml_vec_dot_q8_0_q8_0(权重行, wdata里的激活列)  ← vec_dot 函数指针
          原子抢下一块
```

> 注：若权重走 repack 路径(VLEN=256 的 q8_0 等)，这里换成调 `gemv/gemm`(内核文档 4.5)，权重已在加载时重排(第 9 章)。

---

## 第 7 章 · 非 matmul 算子怎么算（以 RMSNorm 为例）

不是所有算子都像 mul_mat 那样精雕。看一个逐元素算子 `ggml_compute_forward_rms_norm_f32`(`ops.cpp:3758`)：

```c
static void ggml_compute_forward_rms_norm_f32(const ggml_compute_params * params, ggml_tensor * dst, ...) {
    const ggml_tensor * src0 = dst->src[0];
    const int ith = params->ith, nth = params->nth;   // 线程号/总数
    GGML_TENSOR_BINARY_OP_LOCALS                        // 展开 ne00.. nb00.. 等局部变量(0.10)

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));        // 取算子参数(3.4)

    for (int64_t i03 = 0; i03 < ne03; i03++)
    for (int64_t i02 = 0; i02 < ne02; i02++)
    for (int64_t i01 = ith; i01 < ne01; i01 += nth) {   // 多线程按行分(同 mul_mat 的 i11+=nth)
        const float * x = (float *)((char *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);

        ggml_float sum = 0.0;                            // 用 double 累加防误差
        for (int64_t i00 = 0; i00 < ne00; i00++)         // 纯标量循环! "worth switching to SIMD?"
            sum += (ggml_float)(x[i00] * x[i00]);        // 平方和

        const float mean  = sum/ne00;
        const float scale = 1.0f/sqrtf(mean + eps);      // RMSNorm: 1/sqrt(mean(x^2)+eps)

        float * y = (float *)((char *)dst->data + i01*nb1 + ...);
        // y[i] = x[i] * scale  (后面还可能 fuse 一个 *weight)
    }
}
```

**三个教学点**：
1. **算子层模式**：`GGML_TENSOR_*_LOCALS` 展开维度 -> 多重循环遍历高维 -> `ith/nth` 多线程分行 -> 内层算。几乎所有 ops.cpp 算子都是这骨架。
2. **它是纯标量的**！注释 `// worth switching to explicit SIMD?` 明示未向量化。对比 mul_mat 的精雕，这是**RVV 低垂果实** —— 你可以用 `simd-mappings.h` 的 `GGML_F32_VEC_*`(内核文档第 5 章) 把这个平方和向量化。
3. **op_params**：`eps` 从这里取(3.4)。

> 同类逐元素算子(SiLU、GELU、scale、add)多数也是标量或半向量。想优化 mul_mat 之外的 RVV，从这些入手。

---

## 第 8 章 · 完整链路（一张图记住）

```
llama_decode(batch)                              对外 API
  llama_context::decode / process_ubatch         拆 micro-batch
    llama_model::build_graph                      [建图] Transformer -> ggml DAG (第3章)
    graph_compute
      ggml_backend_sched_*                        [调度] 切分/选后端 (第4章)
        ggml_backend_cpu_graph_compute
          ggml_graph_compute                      [线程池] N 线程 (5.1)
            ggml_graph_compute_thread
              ggml_compute_forward (switch op)    [分发] 按 op 派发 (5.2)
                ├─ ggml_compute_forward_rms_norm  逐元素算子(标量, 第7章)
                └─ ggml_compute_forward_mul_mat   ★ 热点算子 (第6章)
                     阶段1 from_float=quantize_row_q8_0   ← 内核文档 4.1
                     阶段2 vec_dot=ggml_vec_dot_q8_0_q8_0 ← 内核文档 4.2
                             └─ vsetvli/vle8/vwmul/vwredsum  ← 你 gdb 看的指令
```
从一个 token 的 `llama_decode`，到一条 `vwmul.vv`，全程打通。

---

## 第 9 章 · 内存：buffer、repack 与分配器

### 9.1 backend buffer
张量的 `data` 不是 `malloc` 来的，而是从**后端 buffer** 里分配的一块。`ggml_backend_buffer` 抽象了"这块内存在哪、怎么读写"(CPU 内存 / CUDA 显存)。CPU 普通 buffer 就是对齐的主存。

### 9.2 repack extra buffer（连接你的 repack 内核）
有一种**特殊 CPU buffer**：权重 set 进去时自动重排成交织布局。机制(`repack.cpp:4733`)：
```c
static void ggml_backend_cpu_repack_buffer_set_tensor(..., struct ggml_tensor * tensor,
                                                       const void * data, ...) {
    auto * tensor_traits = ...;
    tensor_traits->repack(tensor, data, size);   // 普通 q8_0 -> block_q8_0x16 交织(一次性)
}
```
- 模型加载时，若某权重张量符合条件(`get_tensor_traits`(:4710) 按 VLEN 判定，如 q8_0 仅 VLEN=256)，它就被放进这种 buffer，加载即重排。
- 之后 mul_mat 对这种张量改调 `gemv/gemm`(内核文档 4.5)，因为权重已是交织格式。
- **这就把"上层 buffer"和"底层 repack 内核"接上了**：交织布局在加载时一次性准备好，运行时直接用。

### 9.3 图分配器 `ggml_gallocr`（ggml-alloc.c）
中间张量(每层的 Q/K/V/attention scores 等)不能每个都长期占内存 —— 它们用完即可释放。`gallocr` 在执行前**规划复用**：分析图的生命周期，让不重叠的中间张量**共享同一块内存**，把峰值内存压到最小。
- 权重(leafs)常驻(来自 mmap)；中间结果(nodes)在一块复用池里轮转。
- 这就是为什么跑 0.5B 模型不需要为每个中间张量单独留内存。

---

## 第 10 章 · 这份文档"没"覆盖什么（诚实边界）

掌握 1-9 章 + 内核文档，你能完全看懂 **CPU 推理的计算路径与内存模型**，足以改/测 RVV 内核。以下属更外围或其他方向，**对 RVV 目标非必需**：

| 主题 | 在哪 | 何时需要 |
|---|---|---|
| 每个 op 的建图细节 | `src/llama-model.cpp build_graph` | 加新算子/改结构时 |
| GGUF 解析、权重 mmap 细节 | `src/llama-model-loader.cpp` | 想懂文件格式时 |
| 采样/分词 | `src/llama-sampler.cpp` `llama-vocab.cpp` | 改生成策略时 |
| KV cache 实现 | `src/llama-kv-cache*.cpp` | 改注意力/长上下文时 |
| 量化 ref 实现与舍入 | `ggml/src/ggml-quants.c` | 抠量化数值时 |
| 其他后端 CUDA/Metal/Vulkan | `ggml/src/ggml-*` | 跨后端对比 |
| 训练 / autograd | `ggml/src/ggml-opt.cpp` | 纯推理用不到 |

需要哪块，照本文风格(内嵌源码 + C 语法)继续补即可。

---

## 附录 · 本文 C 语法 / 机制速查

| 项 | 用途 | 章节 |
|---|---|---|
| `typedef struct{}` `enum` `union` | 类型定义 / 枚举(当下标) / 共用内存 | 0.1/0.2/0.5 |
| `void *` + cast | 统一接口接任意类型 | 0.3 |
| **函数指针** | ggml 多态核心(type_traits 表) | 0.4 |
| `static_assert` `restrict` | 编译期锁大小 / 助向量化 | 0.6/0.7 |
| 定宽整数 / 原子+barrier | 步长维度 / 多线程协作 | 0.8/0.9 |
| `GGML_ASSERT` `..._LOCALS` 宏 | 断言 / 展开 ne/nb 局部变量 | 0.10 |
| ggml_tensor (ne/nb/op/src/data) | 数据模型 | 2 |
| cgraph / build_forward_expand | 计算图收集 | 3 |
| backend / sched | 后端抽象 / 调度切分 | 4 |
| type_traits_cpu | 类型->内核函数指针表 | 5.3 |
| mul_mat 两阶段 | 量化激活 + 分块 vec_dot | 6 |
| ops 算子骨架 | LOCALS+多重循环+ith/nth | 7 |
| repack buffer / gallocr | 加载时重排 / 中间张量复用 | 9 |

**一句话主线**：ggml 用 **enum 编号类型/算子**，用**函数指针表(type_traits)** 把"类型"映射到"内核函数"，用 **switch** 把"算子"映射到"算子实现"，用 **void\*+块结构体** 统一处理各种量化，用 **buffer + gallocr** 管理内存与权重重排。看懂这几个映射，就看懂了 ggml 的骨架。
