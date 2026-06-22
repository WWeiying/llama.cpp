# llama.cpp 推理全流程教学（原理 + 数学 + 代码精讲）

> 这是三件套的第三份。前两份讲 **ggml**（计算引擎）和 **RVV 内核**；这一份讲 **llama 层** —— 把一个具体大模型（以你的 Qwen2.5-0.5B 为例）从"一段文字"变成"下一个 token"的完整软件流程。
>
> 三位一体：**LLM 原理 + 详细数学** -> **代码精讲（内嵌源码）** -> **C++ 语法**。每个数学步骤都标出对应的 ggml 算子和 llama 函数，让"公式"和"代码"一一对上。
>
> 模型常量（Qwen2.5-0.5B，贯穿全文做数值例）：
> ```
> n_vocab = 151936   词表大小
> n_embd  = 896      隐藏维度
> n_layer = 24       Transformer 层数
> n_head  = 14       注意力头数         head_dim = 896/14 = 64
> n_head_kv = 2      KV 头数(GQA!)      n_embd_kv = 2*64 = 128
> n_ff    = 4864     FFN 中间维度
> rope_type = NEOX   freq_base = 1e6
> ```
> 源码内嵌，行号以当前仓库为准。配合 `ggml-core-tutorial.md`（下层）食用。

---

# 第 0 章 · 读 llama 层需要的 C++ 语法

`src/llama-*.cpp` 是 **C++**（ggml 核心是 C）。比前两份文档的 C 语法多了这些，先认一次。

### 0.1 class / 成员函数 / public-private
```cpp
class llm_graph_context {
public:
    ggml_tensor * build_norm(ggml_tensor * cur, ...) const;  // 成员函数
    ggml_tensor * build_ffn(...) const;
protected:
    const llama_hparams & hparams;   // 成员变量(引用)
};
```
- `const` 在函数末尾 = 这个方法不修改对象（只读）。
- `class` 默认成员私有，`struct` 默认公开，其余相同。

### 0.2 继承 + 虚函数 + vtable（多态）
两种多态你都会遇到：
```cpp
// (A) C++ 虚函数: 派生类 override 基类方法
struct llama_model_qwen2 : llama_model {
    std::unique_ptr<llm_graph_context> build_arch_graph(...) const override;
};
// (B) 手写 C 风格 vtable(函数指针结构体), 采样器用这种:
struct llama_sampler_i {
    void (*apply)(llama_sampler * smpl, llama_token_data_array * cur_p);  // 函数指针(见上份 0.4)
    void (*free) (llama_sampler * smpl);
};
```
`override` 关键字：声明"我在覆盖基类虚函数"，写错签名会编译报错。

### 0.3 智能指针 std::unique_ptr / make_unique
```cpp
std::unique_ptr<llm_graph_context> llm = std::make_unique<graph>(*this, params);
//  独占所有权的指针, 离开作用域自动 delete(RAII), 不用手动释放。
llm->build_pooling(...);   // 像普通指针一样用 ->
```

### 0.4 引用 `&` 与 const 引用
```cpp
void load_hparams(llama_model_loader & ml);          // 引用: 传的是本体, 不拷贝
const llama_layer & layer = model.layers[il];        // const 引用: 只读且不拷贝
```
引用 = 变量别名，比指针安全（不能为空、不能改指向）。`const &` 是 C++ 传大对象的标准方式（省拷贝）。

### 0.5 结构化绑定 `auto [a,b,c]`（C++17）
```cpp
auto [Qcur, Kcur, Vcur] = build_qkv(...);   // 函数返回一个含3个张量的结构, 一次拆成3个变量
```
你会在 qwen2.cpp 看到，等价于"一次返回多个值并解包"。

### 0.6 lambda 匿名函数
```cpp
auto has_lora = [this](ggml_tensor * w) { ... return false; };  // [捕获](参数){体}
//  [this]=捕获当前对象; 之后可像函数一样 has_lora(up) 调用
```

### 0.7 std 容器
- `std::vector<T>` 动态数组（`.data()` 取裸指针，`.size()` 长度，`.push_back()` 追加）。
- `std::string` 字符串。
- `std::unordered_map<K,V>` 哈希表（分词器用 token->id）。

### 0.8 模板（上份见过）
```cpp
template <int K, int N> struct block { ... };   // 编译期参数, repack 交织块用
```

> 语法到此。下面先讲**原理与数学**（让代码有意义），再**精讲代码**。

---

# 第一部分 · LLM 原理与数学

## 第 1 章 · 宏观：自回归下一词预测

大语言模型做的事**只有一件**：给定前面的 token 序列，预测**下一个 token 的概率分布**。

```
输入:  [The, cat, sat, on, the]
模型:  P(下一个 | 前面)  ->  {mat: 0.6, floor: 0.2, sofa: 0.1, ...}  (151936 个词的概率)
采样:  选一个(如 mat) -> 追加 -> 再预测下一个 -> 循环
```

这就是"自回归生成"。一次 `llama_decode` = 把当前序列过一遍模型，得到最后一个位置的 **logits**（151936 维未归一化分数）；采样器把 logits 变成一个 token；循环。

数学上，模型是一个函数 `f`：
```
logits = f(token序列)            logits ∈ ℝ^151936
P = softmax(logits)              概率分布
next_token ~ P                   采样
```
`f` 内部 = 嵌入 + 24 层 Transformer block + 最终投影。下面逐层拆。

---

## 第 2 章 · token -> 向量（嵌入）

token 是整数 id。第一步用 `token_embd` 权重表（形状 `[n_embd=896, n_vocab=151936]`）**查表**：每个 token id 取出一个 896 维向量。

```
token id = 5  ->  token_embd 的第 5 列  ->  x ∈ ℝ^896
一个序列 T 个 token  ->  矩阵 X ∈ ℝ^{896 × T}
```
- 对应 ggml 算子：`ggml_get_rows(tok_embd, tokens)`。
- llama 函数：`build_inp_embd`。
- 无数学计算，纯查表。输出记作 `inpL`（input to Layer）。

---

## 第 3 章 · 一个 Transformer 层（核心，24 层重复）

每层把 `[896 × T]` 的隐藏状态变换成同形状的新状态，结构（Qwen2 = Pre-Norm + GQA + SwiGLU）：

```
        inpL ───────────────────────┐(残差)
          │                          │
      RMSNorm(attn_norm)             │
          │                          │
      Self-Attention(GQA + RoPE)     │
          │                          │
          + ◄────────────────────────┘
          │ = ffn_inp ───────────────┐(残差)
      RMSNorm(ffn_norm)              │
          │                          │
      FFN(SwiGLU)                    │
          │                          │
          + ◄────────────────────────┘
          │
        下一层
```

### 3.1 RMSNorm（归一化）

**作用**：把向量缩放到稳定的数值范围，让训练/推理稳定。Qwen2 用 RMSNorm（比 LayerNorm 简单，不减均值）。

**公式**（对一个 896 维向量 x）：
```
              x_i
y_i = ───────────────────── · w_i
       sqrt( (1/n)Σx_j² + ε )

n=896, ε=1e-6, w=可学习权重(attn_norm)
```
即：算出 x 的均方根 RMS = sqrt(mean(x²))，每个分量除以它，再逐元素乘权重 w。

**数值例**（取 4 维示意，实际 896 维）：
```
x = [3, -4, 0, 0]
mean(x²) = (9+16+0+0)/4 = 6.25
RMS = sqrt(6.25+ε) ≈ 2.5
y = [3/2.5, -4/2.5, 0, 0] · w = [1.2, -1.6, 0, 0] · w
```
- ggml 算子：`ggml_rms_norm` + `ggml_mul`（乘 w）。
- llama 函数：`build_norm(..., LLM_NORM_RMS)`。
- 计算实现：你在 `ggml-core` 文档第 7 章见过那个**纯标量**的 `ggml_compute_forward_rms_norm_f32`（RVV 优化点）。

### 3.2 Q/K/V 投影 + GQA

注意力要把每个 token 的向量投影成三个角色：**Query（我要找什么）**、**Key（我是什么，供别人找）**、**Value（我携带的信息）**。各是一个矩阵乘：

```
Q = Wq · x      Wq: [896, 896]   ->  Q ∈ ℝ^896  (= 14 头 × 64)
K = Wk · x      Wk: [896, 128]   ->  K ∈ ℝ^128  (= 2 头 × 64)   ← 注意只有 128!
V = Wv · x      Wv: [896, 128]   ->  V ∈ ℝ^128  (= 2 头 × 64)
```

**GQA（Grouped-Query Attention，分组查询注意力）**：Qwen2-0.5B 有 14 个 Q 头但只有 **2 个 KV 头**。14 个 Q 头分成 2 组，每组 7 个 Q 头**共享**同一对 K/V 头。好处：KV cache 小 7 倍（省显存、省带宽），decode 更快。

```
Q 头:  0 1 2 3 4 5 6 | 7 8 9 10 11 12 13
        └── 共享 KV0 ┘   └── 共享 KV1 ──┘
```
- ggml 算子：每个投影是 `ggml_mul_mat`（**你 RVV 内核优化的就是这个**！一层 7 个 mul_mat：q/k/v/o + gate/up/down）。
- llama 函数：`build_qkv`（内部 `build_lora_mm` = mul_mat 包装）。

### 3.3 RoPE（旋转位置编码）

注意力本身不分先后顺序，必须注入"位置"信息。RoPE 的做法：**按 token 位置 pos，把 Q/K 向量的每一对维度旋转一个角度**。

**公式**：把 head_dim=64 维分成 32 对 `(x_{2i}, x_{2i+1})`，第 i 对旋转角度 `pos · θ_i`，其中
```
θ_i = base^(-2i/d) = 1000000^(-2i/64)     i = 0..31

[x'_{2i} ]   [cos(pos·θ_i)  -sin(pos·θ_i)] [x_{2i} ]
[x'_{2i+1}] = [sin(pos·θ_i)   cos(pos·θ_i)] [x_{2i+1}]
```
- 低维（i 小）θ 大，转得快（管局部）；高维 θ 小，转得慢（管全局）。
- **关键性质**：两个向量旋转后做内积，结果只依赖它们的**相对位置** `pos_q - pos_k` —— 这正是注意力需要的"相对距离"。
- Qwen2 用 **NEOX 变体**：配对的是 `(i, i+d/2)` 而非相邻 `(2i,2i+1)`，数学等价。
- ggml 算子：`ggml_rope_ext`；llama：qwen2.cpp 里对 Qcur/Kcur 各调一次。
- 只转 Q 和 K，**不转 V**（V 只携带信息，不参与位置匹配）。

### 3.4 注意力（Attention）

核心思想：每个 token 的新表示 = **所有它能看到的 token 的 Value 的加权平均**，权重 = Query 和 Key 的匹配度。

**三步公式**（单个头，T 个 token）：
```
① 打分:   S = QᵀK / √d        S ∈ ℝ^{T×T}, d=64    S[i,j]=Q_i·K_j/√64
② 归一:   A = softmax(S + mask)   每行(对所有 j)归一成概率, mask 屏蔽未来
③ 加权:   out_i = Σ_j A[i,j] · V_j
```

**因果掩码（causal mask）**：第 i 个 token 只能看 `j ≤ i`（不能看未来）。实现上把 `j > i` 的 S 设成 `-∞`，softmax 后权重为 0。

**√d 缩放**：d=64，`√d=8`。点积随维度增大而变大，除以 √d 防止 softmax 进入饱和区（梯度消失/输出过尖）。

**数值例**（2 个 token，d 简化）：
```
Q_2·K_0=2, Q_2·K_1=6, Q_2·K_2=4   (token2 对 0/1/2 的匹配)
缩放(/√d, 设√d=2): [1, 3, 2]
softmax([1,3,2]) = [e¹,e³,e²]/Σ ≈ [0.09, 0.67, 0.24]
out_2 = 0.09·V_0 + 0.67·V_1 + 0.24·V_2   (主要看 token1)
```

**ggml 里的实现**（注意 `ggml_mul_mat(a,b)` 算的是"沿 a、b 的第 0 维做内积"，结果 `[a.ne1, b.ne1]`）：
```
kq  = ggml_mul_mat(K, Q)        // [n_kv, T, n_head]  = 每个(query i, key j)的分数 S
kq  = ggml_soft_max_ext(kq, mask, scale=1/√d)   // ②缩放+掩码+softmax 一步
kqv = ggml_mul_mat(V, kq)       // [head_dim, T, n_head]  = ③加权 V
```
- llama 函数：`build_attn` -> `build_attn_mha`。
- **GQA 广播**：2 个 KV 头被 14 个 Q 头共享，ggml 的 mul_mat 自动按头维广播。

### 3.5 KV cache（为什么 decode 快）

生成第 t 个 token 时，注意力要算它的 Q 和**前面所有 token 的 K/V** 的匹配。前面 token 的 K/V 上一步已经算过 —— **缓存起来**，这一步只算新 token 的 K/V 并追加，不重算历史。

```
prefill(处理 prompt T 个 token): 算 T 个 K/V, 全部存入 cache
decode(每生成 1 个):  只算 1 个新 K/V, 追加; Q 与 cache 里 [0..t] 的 K/V 做注意力
```
- 这就是为什么 **prefill 是计算密集（大矩阵乘 gemm）、decode 是访存密集（矩阵×向量 gemv）** —— 直接对应你 RVV 内核里的 gemm vs gemv！
- GQA 让 cache 小 7 倍。

### 3.6 残差连接

每个子层（注意力、FFN）的输出**加回输入**：`out = sublayer(norm(x)) + x`。让梯度直通、深层网络可训练。
- ggml 算子：`ggml_add`。

### 3.7 FFN / SwiGLU（前馈网络）

注意力后接一个逐位置的前馈网络，做非线性变换、扩展表达力。Qwen2 用 **SwiGLU**（门控）：

**公式**：
```
FFN(x) = Wdown · ( SiLU(Wgate·x) ⊙ (Wup·x) )

Wgate: [896, 4864]   ->  gate ∈ ℝ^4864
Wup:   [896, 4864]   ->  up   ∈ ℝ^4864
SiLU(z) = z·σ(z) = z/(1+e^{-z})        逐元素激活
⊙ = 逐元素相乘(门控)
Wdown: [4864, 896]   ->  回到 896 维
```
- **门控直觉**：`SiLU(gate)` 当"开关"，逐元素调制 `up` 的每个分量 —— 哪些特征通过、通过多少。
- 3 个矩阵乘（gate/up/down）—— 又是你 RVV 内核的活。FFN 通常是模型里**最大的计算量**（4864 比 896 大）。
- ggml 算子：`ggml_mul_mat` ×3 + `ggml_silu` + `ggml_mul`。
- llama 函数：`build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR)`。

---

## 第 4 章 · 输出：从最后一层到 token

24 层之后：
```
① 最终 RMSNorm(output_norm)             [896 × T]
② lm_head 投影:  logits = Woutput · x   Woutput:[896,151936] -> logits ∈ ℝ^151936
   (Qwen2 里 Woutput 与 token_embd 共享权重 = tied embedding)
③ 只取最后一个位置的 logits(要预测的下一个)
④ softmax -> 概率; 采样器选一个 token
```
- ②是**最大的单个 mul_mat**（896×151936，词表巨大）。你 gdb 里 `n=896` 的那次 vec_dot 调用很可能就是它。
- ggml：`ggml_mul_mat(output, cur)`；llama：qwen2.cpp 结尾 `build_lora_mm(model.output, cur)`。

**至此原理讲完**。下面看代码怎么把这套数学搭出来、跑起来。

---

# 第二部分 · 代码精讲

## 第 5 章 · 入口：`simple.cpp` 的主循环

`examples/simple/simple.cpp` 是最小可运行示例，整个项目的骨架。主循环（删节）：

```cpp
// 1) 分词: 文本 -> token id 数组
std::vector<llama_token> prompt_tokens(n_prompt);
llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(),
               prompt_tokens.size(), true, true);

// 2) 建采样器链(这里只加一个 greedy = 取最大)
llama_sampler * smpl = llama_sampler_chain_init(sparams);
llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

// 3) 把 prompt 打包成一个 batch
llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

// 4) 主循环: decode -> sample -> 追加 -> 再 decode
for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
    llama_decode(ctx, batch);                          // ★ 跑一遍模型(第一部分全部数学)
    n_pos += batch.n_tokens;

    new_token_id = llama_sampler_sample(smpl, ctx, -1); // ★ 从 logits 采一个 token
    if (llama_vocab_is_eog(vocab, new_token_id)) break; // 遇到结束符停

    char buf[128];
    int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true); // token->文本
    printf("%.*s", n, buf);

    batch = llama_batch_get_one(&new_token_id, 1);      // 下一轮只喂这 1 个新 token
}
```

**这就是全部**。`llama_decode`(第 6 章) 内部 = 第一部分的所有数学；`llama_sampler_sample`(第 9 章) = 采样；`llama_tokenize`/`token_to_piece`(第 8 章) = 文本↔token。
- 第一轮 batch = 整个 prompt（**prefill**，n_tokens>1）；之后每轮 batch = 1 个新 token（**decode**）。这就是 prefill/decode 的来源。

---

## 第 6 章 · 公共 API 与关键类型（`include/llama.h`）

应用只通过 `llama.h` 的 C API 用 llama.cpp。关键类型：
```c
llama_model    * model;   // 加载好的模型(权重 + 超参 + 词表)
llama_context  * ctx;     // 一次推理会话(KV cache, 计算缓冲, 状态)
llama_token      id;      // 一个 token = int32
llama_batch      batch;   // 一批待处理 token(+位置/序列id)
llama_sampler  * smpl;    // 采样器(链)
```
核心动词：`llama_model_load_from_file` -> `llama_init_from_model`(建 ctx) -> `llama_decode`(算) -> `llama_sampler_sample`(采) -> `llama_token_to_piece`(解码文本)。

---

## 第 7 章 · 模型加载：GGUF 文件 -> 可运行模型

### 7.1 架构识别与张量命名（`llama-arch.cpp`）
GGUF 元数据有 `general.architecture="qwen2"`，映射到 `LLM_ARCH_QWEN2`(:33)。每个架构定义**张量命名模板**：
```cpp
{ LLM_TENSOR_ATTN_Q, "blk.%d.attn_q" },   // 第 il 层的 Q 权重叫 "blk.{il}.attn_q.weight"
{ LLM_TENSOR_ATTN_Q, {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT} }, // 它参与 MUL_MAT
```
加载器据此从 GGUF 找到每个权重张量。

### 7.2 读超参 + 绑定张量（`src/models/qwen2.cpp`）
每个架构一个文件。Qwen2 的加载：
```cpp
void llama_model_qwen2::load_arch_tensors(llama_model_loader &) {
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {                  // 没有独立 lm_head -> 与词嵌入共享(tied)
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }
    for (int i = 0; i < n_layer; ++i) {    // 每层的权重
        auto & layer = layers[i];
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);  // wq/wk/wv
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
    }
}
```
- `create_tensor` 在权重 buffer(可能是 mmap 的 GGUF，或 repack buffer) 里登记一个张量，`data` 指向文件中的权重字节。
- 注意 `{n_embd, n_ff}` 等形状，正好对应第一部分的矩阵维度。
- **这就是第一部分数学里 Wq/Wk/Wgate... 的来源** —— 它们现在是 `layer.wq`、`layer.ffn_gate` 等张量，等着建图时被 `mul_mat` 引用。

---

## 第 8 章 · ★建图：把数学拼成 ggml 图（`src/models/qwen2.cpp`）

这是 llama 层最核心的代码 —— 把第一部分每个公式翻译成 ggml 算子。**完整精讲 qwen2 的 `graph` 构造函数**（这就是第 3 章那张结构图的代码版）：

```cpp
llama_model_qwen2::graph::graph(const llama_model & model, const llm_graph_params & params)
        : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();   // = 64

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);     // 第2章: token id -> 向量 [896 × T]
    ggml_tensor * inp_pos = build_inp_pos();   // 每个 token 的位置(给 RoPE)
    auto * inp_attn = build_attn_inp_kv();     // 注意力输入(KV cache 句柄 + 掩码)
    ggml_tensor * inp_out_ids = build_inp_out_ids();  // 只需输出哪些位置(decode 只要最后一个)

    for (int il = 0; il < n_layer; ++il) {     // ★ 24 层循环
        ggml_tensor * inpSA = inpL;            // 存一份做残差(3.6)

        // ── 3.1 注意力前 RMSNorm ──
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

        // ── 3.2-3.4 自注意力 ──
        {
            // 3.2 Q/K/V 投影(返回3个张量, C++17 结构化绑定)
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // 3.3 对 Q、K 各做 RoPE(注入位置), V 不转
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, ...);

            // 3.4 注意力: 内部做 KQ/√d -> softmax(+掩码) -> ·V -> 输出投影 Wo
            cur = build_attn(inp_attn, model.layers[il].wo, ..., 
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), il);    // scale = 1/√64 = 1/8
        }

        // decode 优化: 最后一层只保留要输出的位置(省算 lm_head)
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // ── 3.6 残差 ──
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);

        // ── 3.1 FFN 前 RMSNorm ──
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);

        // ── 3.7 FFN(SwiGLU) ──
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);

        // ── 3.6 残差 ──
        cur = ggml_add(ctx0, cur, ffn_inp);
        inpL = cur;        // 这层输出 = 下一层输入
    }

    // ── 第4章 输出 ──
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);  // ①最终 norm
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur, model.output_s);             // ②lm_head -> logits
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);   // 把图收集成拓扑序(见 ggml-core 文档 3.3)
}
```

**关键认知**：这个函数**不做任何计算**，只是 new 一堆张量、连成图（懒执行，见 ggml-core 文档第 3 章）。真正算数在后面 `graph_compute`。一层产生约 `7 个 mul_mat + 2 norm + 2 rope + 1 softmax + 几个 add` 的节点；24 层 + 输出 = 整张图几百个节点。

下面精讲它调用的几个 builder。

### 8.1 `build_norm`（RMSNorm 代码）
```cpp
ggml_tensor * llm_graph_context::build_norm(ggml_tensor * cur, ggml_tensor * mw, ggml_tensor * mb,
                                            llm_norm_type type, int il) const {
    switch (type) {
        case LLM_NORM_RMS: cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;  // 3.1 公式主体
        ...
    }
    if (mw) cur = ggml_mul(ctx0, cur, mw);   // 乘可学习权重 w
    if (mb) cur = ggml_add(ctx0, cur, mb);   // 加 bias(Qwen2 无, mb=NULL)
    return cur;
}
```
`ggml_rms_norm` = `x/sqrt(mean(x²)+ε)`；`ggml_mul` = `·w`。完全对应 3.1 公式。

### 8.2 `build_qkv`（Q/K/V 投影代码，分离路径）
```cpp
// 分离 Q/K/V(Qwen2 走这条; 也有融合 wqkv 路径)
Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);   // Q = Wq·x   [896×T]
if (layer.wq_b) Qcur = ggml_add(ctx0, Qcur, layer.wq_b);   // Qwen2 的 Q/K/V 有 bias
Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);   // K = Wk·x   [128×T]
Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);   // V = Wv·x   [128×T]
...
// reshape 成 [head_dim, n_head, T] 把"头"这一维拆出来
Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);  // [64,14,T]
Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);  // [64, 2,T] ← GQA
Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);  // [64, 2,T]
return { Qcur, Kcur, Vcur };   // 结构化绑定的来源
```
- `build_lora_mm`(:1085) 内部就是 `ggml_mul_mat(w, cur)`（外加可选 LoRA 适配器）—— **你的 RVV vec_dot 最终为它服务**。
- reshape 把 896 拆成 `14 头 × 64`，K/V 是 `2 头 × 64`（GQA，对应 3.2）。

### 8.3 `build_attn_mha`（注意力数学代码，非 flash 路径）
```cpp
// q,k,v 先 permute 成 [head_dim, T, n_head] 便于按头算
q = ggml_permute(ctx0, q, 0, 2, 1, 3);
k = ggml_permute(ctx0, k, 0, 2, 1, 3);
v = ggml_permute(ctx0, v, 0, 2, 1, 3);

ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);        // ① S = QᵀK, 形状[n_kv, T, n_head]
ggml_mul_mat_set_prec(kq, GGML_PREC_F32);            // 分数要 F32(数值范围大)

kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, ...); // ② (S·scale + 掩码)再 softmax
                                                          //   kq_scale = 1/√d = 1/8
ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);       // ③ out = Σ A·V, [head_dim, T, n_head]
cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);           // 头维换回
cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], ...); // 合并头 -> [896, T]
```
精确对应 3.4 的三步。`ggml_soft_max_ext` 把"缩放 + 加掩码 + softmax"融成一个算子。`kq_mask` 来自 `inp_attn`（含因果掩码 + KV cache 的有效范围）。返回后 `build_attn` 再乘输出投影 `Wo`。

### 8.4 `build_ffn`（SwiGLU 代码）
```cpp
ggml_tensor * tmp = build_lora_mm(up, cur);      // up = Wup·x        [4864×T]
// gate(并行 PAR 模式: gate 也作用在原 cur 上)
cur = build_lora_mm(gate, cur);                  // gate = Wgate·x    [4864×T]
// (type_op = LLM_FFN_SILU, 下面对 gate 做 SiLU 再乘 up)
case LLM_FFN_SILU:  cur = ggml_silu(ctx0, cur);  // SiLU(gate)
...
cur = ggml_mul(ctx0, cur, tmp);                  // SiLU(gate) ⊙ up   逐元素门控
cur = build_lora_mm(down, cur);                  // down = Wdown·(...)  [896×T]
```
对应 3.7：`Wdown·(SiLU(Wgate·x) ⊙ (Wup·x))`。3 个 mul_mat + silu + mul。

---

## 第 9 章 · 解码驱动：`llama_context::decode`（图怎么被算）

`llama_decode`(llama-context.cpp:4054) -> `llama_context::decode`(:1680)。它把一个 batch 跑完，流程：
```
1. 把 batch 拆成 ubatch(micro-batch, 受 n_ubatch 限制)
2. KV cache: find_slot 给这批 token 分配缓存槽位(第10章)
3. 调 model.build_graph(...) 建图(第8章)
4. ggml_backend_sched_graph_compute 执行图(ggml-core 文档 4-6章)
     -> 线程池 -> ggml_compute_forward -> mul_mat -> 你的 RVV vec_dot
5. 从 res->t_logits 取出 logits, 存进 ctx 供采样
```
- 你 gdb 的 backtrace 顶部就是这里(`decode` -> `process_ubatch` -> `graph_compute`)，底部是 vec_dot。这一章把顶和底接上了。
- batch 有多个 token = prefill（图里 T>1，mul_mat 是矩阵×矩阵=gemm）；batch 1 个 token = decode（T=1，矩阵×向量=gemv）。**直接决定你内核走 gemm 还是 gemv 路径。**

---

## 第 10 章 · KV cache 实现（`llama-kv-cache.cpp`）

### 10.1 存储布局
每层一对 K/V 张量，形状 `[n_embd_gqa, kv_size, n_stream]`：
```cpp
// llama-kv-cache.cpp:245
ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
```
- `n_embd_k_gqa = 128`(GQA, 2头×64)，`kv_size` = 最大缓存位置数，每层一份。
- **cell（槽位）**：缓存里第 i 个位置，记录 `pos[i]`(token位置)、`seq[i]`(属于哪个序列)等元数据(`llama-kv-cells.h`)。

### 10.2 分配槽位 find_slot
新一批 token 来时，`find_slot`(:907) 在 cells 里找空槽（或滑窗可复用的槽），返回每个 token 落在哪个 cell：
```cpp
bool can_use = cells.is_empty(idx);     // 空槽可用
... // 或 SWA 滑窗外的旧槽可复用
if (can_use) res.idxs[s].push_back(idx);
```

### 10.3 写入新 K/V、读出历史 K/V
```cpp
// 写: 把当前 token 的 K 存进 cache 的指定槽位(ggml_set_rows)
ggml_tensor * llama_kv_cache::cpy_k(...) { return ggml_set_rows(ctx, k, k_cur, k_idxs); }
// 读: 取出 [0..n_kv) 范围的历史 K 作为一个 view, 喂给注意力
ggml_tensor * llama_kv_cache::get_k(...) {
    return ggml_view_4d(ctx, k, n_embd_head_k, n_head_kv, n_kv, ns, ...);  // 零拷贝视图
}
```
- `n_kv` = 本步注意力要扫描的缓存位置数（pad 成定值，让图能复用）。
- 历史 K/V 用 **view**（ggml-core 文档 2.4）取出，不拷贝。注意力的 `k`/`v` 就来自这里 —— 所以 3.4 的"前面所有 token 的 K/V"在代码里就是 cache 的这个 view。

### 10.4 context-shift（上下文滑动）
缓存满时，对 cell 的 `pos` 做 RoPE 位置平移来复用旧槽（`pos_add` 累积 `shift[i]`），而非清空重算。长文本生成靠它。

---

## 第 11 章 · 分词器：文本 <-> token（`llama-vocab.cpp`）

Qwen2 用 **字节级 BPE（Byte-Pair Encoding）**。

### 11.1 数据结构
```cpp
std::unordered_map<std::string, llama_token> token_to_id;   // 文本 -> id
std::vector<token_data>                       id_to_token;   // id -> {text, score, attr}
std::unordered_map<pair, int> bpe_ranks;       // (左token,右token) -> 合并优先级(rank)
```
`bpe_ranks` 来自 GGUF 的 merges 表：训练时统计的"哪两个子词该合并、按什么顺序"。

### 11.2 BPE 算法
思路：先把文本按正则切成小片，每个片再拆成单字节符号（双向链表），然后**不断合并 rank 最小（最优先）的相邻对**，直到没有可合并的：
```cpp
// llama-vocab.cpp:640  优先队列合并循环
while (!work_queue.empty()) {
    auto bigram = work_queue.pop_move();        // 取出当前 rank 最小的相邻对
    auto & left = symbols[bigram.left];
    auto & right = symbols[bigram.right];
    if (left.n == 0 || right.n == 0) continue;  // 已被合并过, 跳过
    if (left_token + right_token != bigram.text) continue; // 过期项, 跳过

    left.n += right.n;          // 合并: 右符号并入左符号
    right.n = 0;
    left.next = right.next;     // 从链表摘除右符号
    add_new_bigram(left.prev, bigram.left);   // 新产生的两个相邻对入队
    add_new_bigram(bigram.left, left.next);
}
```
`add_new_bigram` 查 `bpe_ranks` 拿优先级；查不到（不可合并）就丢弃。合并完，每个剩下的符号段查 `token_to_id` 得到 token。
- `add_bos`：句首可选加 BOS 特殊 token。
- 反向（token->文本）：`llama_token_to_piece` 查 `id_to_token` 取文本，处理字节级转义（如 `▁`->空格）。

---

## 第 12 章 · 采样器：logits -> token（`llama-sampler.cpp`）

### 12.1 抽象：vtable + 链
每个采样器是一个带 vtable 的对象（0.2 的手写 vtable）：
```c
struct llama_sampler_i {
    void (*apply)(llama_sampler * smpl, llama_token_data_array * cur_p);  // 核心: 改候选数组
    ...
};
```
候选数组：
```c
typedef struct llama_token_data { llama_token id; float logit; float p; } llama_token_data;
typedef struct llama_token_data_array {
    llama_token_data * data; size_t size; int64_t selected; bool sorted;
} llama_token_data_array;
```
**链**：多个采样器顺序作用在同一个候选数组上，每个就地修改（过滤/缩放/选择）：
```cpp
// llama-sampler.cpp:642  链的 apply
for (auto & smpl : chain->samplers) {
    llama_sampler_apply(smpl.ptr, cur_p);   // 依次跑 top-k -> top-p -> temp -> dist ...
}
```

### 12.2 主入口 llama_sampler_sample
```cpp
// llama-sampler.cpp:806 (删节)
const auto * logits = llama_get_logits_ith(ctx, idx);    // 取最后位置的 logits(第4章③)
cur.resize(n_vocab);
for (token_id = 0; token_id < n_vocab; token_id++)
    cur[token_id] = {token_id, logits[token_id], 0.0f};  // 151936 个候选
llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
llama_sampler_apply(smpl, &cur_p);                       // 跑采样链
return cur_p.data[cur_p.selected].id;                    // 返回被选中的 token
```

### 12.3 代表采样器
```cpp
// greedy: 取 logit 最大(简单/确定)  llama-sampler.cpp:963
cur_p->selected = 0;
for (i=1; i<size; ++i) if (data[i].logit > data[selected].logit) selected = i;

// temperature: logits 除以 T(T<1 更尖锐/确定, T>1 更平/随机)  :265
for (i) cur_p->data[i].logit /= temp;

// top-k: 只留 logit 最大的 k 个  :317
partial_sort(cur_p, k);  cur_p->size = k;

// top-p(nucleus): softmax 后, 按概率累加到 ≥p 截断  :1351
softmax(cur_p); 累积 cum_sum 直到 ≥ p, 截断;

// dist: 最终按概率分布随机抽一个(categorical)  :1036
softmax; rnd = U(0,1)*sum; 累加概率直到 ≥ rnd, 选中该 token;
```
典型链：`top-k -> top-p -> temp -> dist`（先粗筛、再核采样、再调温度、最后随机抽）。`simple.cpp` 只用 greedy（确定性输出）。

---

## 第 13 章 · 全链路追踪（一个 token 的一生）

```
"introduce yourself" (文本)
  └─11章 llama_tokenize ──> [token id 数组]
        └─5章 llama_batch_get_one ──> batch
              └─9章 llama_decode / decode
                    ├─10章 KV cache find_slot 分配槽位
                    ├─8章 build_graph 建图(qwen2.cpp):
                    │      embd -> [24× (RMSNorm->QKV->RoPE->Attn->+ ->RMSNorm->FFN->+)] -> norm -> lm_head
                    │      = 第一部分全部数学
                    └─ggml-core文档4-6章 graph_compute 执行:
                          线程池 -> ggml_compute_forward(switch op)
                            -> mul_mat -> quantize_row_q8_0 + ggml_vec_dot_q8_0_q8_0
                               -> rvv-kernel文档: vsetvli/vle8/vwmul/vwredsum  ← 最底层
                    └─> logits ∈ ℝ^151936
              └─12章 llama_sampler_sample(logits) ──> 下一个 token id
        └─11章 llama_token_to_piece ──> 文本片段 "I"
  └─ 追加 token, 回到 decode, 循环......
```

**三份文档在此合一**：本文(llama 层) -> `ggml-core`(执行引擎) -> `rvv-kernel`(叶子指令)。从一句话到一条 `vwmul.vv`，全程贯通。

---

## 第 14 章 · 边界：这份文档没细讲的

| 主题 | 在哪 | 说明 |
|---|---|---|
| flash attention | `build_attn_mha` 的 `use_flash_attn` 分支 | 把 ②③融成一个 `ggml_flash_attn_ext` 算子(省内存)，本文讲的是非 flash 经典路径 |
| 融合 QKV / MLA / MoE | build_qkv 的 wqkv 分支；其他架构 | 其他模型变体，Qwen2-0.5B 不用 |
| batch/ubatch 细节 | `llama-batch.cpp` | 多序列、位置分配 |
| chat 模板 | `common/chat*`、`llama-chat.cpp` | 把对话格式化成 prompt(在 tokenize 之前) |
| 投机解码、并行采样 | sampler + 多 seq | 加速技巧 |
| 量化怎么产生 | `llama-quant.cpp`、`ggml-quants.c` | 你 RVV 内核处理的格式的"出厂"过程 |

---

## 附录 · 原理 <-> 代码 <-> 算子 对照速查

| LLM 步骤(数学) | llama 函数 | ggml 算子 | 你的 RVV 内核 |
|---|---|---|---|
| token->向量(查表) | build_inp_embd | ggml_get_rows | - |
| RMSNorm | build_norm | ggml_rms_norm + ggml_mul | (标量, 可优化) |
| Q/K/V 投影 | build_qkv/build_lora_mm | ggml_mul_mat | ✓ vec_dot/gemv |
| RoPE | (qwen2.cpp) | ggml_rope_ext | - |
| 注意力打分 QᵀK | build_attn_mha | ggml_mul_mat | ✓ |
| softmax(+掩码+scale) | build_attn_mha | ggml_soft_max_ext | (标量, 可优化) |
| 加权 V | build_attn_mha | ggml_mul_mat | ✓ |
| 残差 | (qwen2.cpp) | ggml_add | - |
| FFN SwiGLU | build_ffn | ggml_mul_mat×3 + ggml_silu + ggml_mul | ✓ |
| lm_head | build_lora_mm | ggml_mul_mat | ✓(最大那个) |
| 采样 | llama_sampler_sample | (CPU 标量) | - |

**一句话主线**：llama 层 = **加载**(GGUF->张量) + **建图**(把 Transformer 数学翻译成 ggml 算子) + **驱动**(decode 循环跑图取 logits) + **两端**(分词/采样)。建图(第 8 章 qwen2.cpp)是灵魂 —— 它把第一部分的每个公式一一对应成一个 ggml 算子，而那些 `mul_mat` 最终落到你优化的 RVV 内核上。
