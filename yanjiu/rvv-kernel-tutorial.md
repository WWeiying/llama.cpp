# ggml RISC-V / RVV 内核逐行教学

> 目标：读完这份文档 + 跟着在 gdb 里实跑，你能看懂 `ggml/src/ggml-cpu/arch/riscv/` 下**所有** RVV 内核，并能自己改/写一个。
>
> 方法：先把"RVV 编程模型 + intrinsic 命名规则"讲透一次（第 1-2 章），后面逐行讲代表函数时，**已讲过的语法不再重复**，只点出新东西。每类量化只挑一个代表，吃透代表 = 吃透一类。
>
> 配套源码（行号以当前仓库为准）：
> - `arch/riscv/quants.c` —— 量化 + vec_dot
> - `arch/riscv/repack.cpp` —— gemv/gemm 重排
> - `simd-mappings.h` —— 通用 F32 宏
> - `arch/riscv/cpu-feats.cpp` —— 运行时检测

---

## 第 1 章 · RVV 编程模型（地基）

### 1.1 五个核心量：VLEN / SEW / LMUL / vl / vtype

RVV 是**向量长度无关（VLA）**的：同一份机器码能跑在任意向量宽度的硬件上。理解它先记住 5 个量：

| 量 | 含义 | 谁决定 |
|---|---|---|
| **VLEN** | 一个向量寄存器的物理位宽 | 硬件固定（你 QEMU 配的 `vlen=1024`）|
| **SEW** | 每个元素多少位（8/16/32/64）| 软件用 `vsetvli` 设 |
| **LMUL** | 几个寄存器合成一组（1/2/4/8 或分数 1/2,1/4）| 软件用 `vsetvli` 设 |
| **vl** | 这一条向量指令实际处理多少个元素 | `vsetvli` 返回 |
| **VLMAX** | 当前 SEW/LMUL 下一次最多能处理多少元素 | `= LMUL * VLEN / SEW` |

关系式（务必记牢）：

```
VLMAX = LMUL * VLEN / SEW
```

**实例（你在 gdb 里亲眼见过的）**：`ggml_vec_dot_q8_0_q8_0` 里 `vsetvli zero,a4,e8,m2`，寄存器读出 `vtype=0xc1, vl=32`：

```
SEW=8(e8), LMUL=2(m2), VLEN=1024
VLMAX = 2 * 1024 / 8 = 256
但代码请求 vl = 32   ->  利用率 32/256 = 12.5%
```

> 关键认知：很多 ggml RVV 内核把 `vl` 写死成"一个量化块的大小"（如 32），这在 VLEN=128 上刚好填满，在 VLEN=1024 上只用 1/8。**这是全项目最大的优化主题。**

### 1.2 vtype 寄存器的位编码

`vsetvli` 把 SEW/LMUL/agnostic 标志打包进 `vtype`。会手工解码它，调试时非常有用：

```
vtype 位布局（低 8 位）：
  bit 7   : vma  (mask agnostic)   掩码外元素是否"随意"
  bit 6   : vta  (tail agnostic)   尾部元素是否"随意"
  bit 5:3 : vsew  000=e8 001=e16 010=e32 011=e64
  bit 2:0 : vlmul 000=m1 001=m2 010=m4 011=m8  101=mf8 110=mf4 111=mf2

例：vtype=0xc1=0b1100_0001
   vma=1, vta=1, vsew=000(e8), vlmul=001(m2)  ->  e8,m2,ta,ma
```

"agnostic（随意）"= 允许硬件对尾部/掩码外的元素写垃圾值，给硬件优化空间；ggml 几乎全用 `ta,ma`。

### 1.3 寄存器与 LMUL 分组

- 32 个向量寄存器 `v0`-`v31`。`v0` 还兼任 **mask 寄存器**。
- **LMUL=m2** 表示用 2 个寄存器拼成一组：写 `v8`（i8m2）实际占用 `v8,v9`。m4 占 4 个，m8 占 8 个。
  - 所以 m8 类型只能用 `v0/v8/v16/v24` 这种对齐的起点。
- **加宽运算**（widening，名字带 `w`）输出位宽翻倍 -> LMUL 也翻倍：`vwmul` 把两个 `i8m1` 乘成 `i16m2`。这是量化点积的命脉（int8×int8 会溢出 int8，必须升到 int16/int32）。

### 1.4 strip-mining：VLA 的标准循环

因为 vl 由硬件能力决定，处理任意长度 n 的标准写法是：

```c
for (size_t avl = n; avl > 0; ) {
    size_t vl = __riscv_vsetvl_e8m1(avl); // 硬件给本次能处理的 vl
    // ... 用 vl 处理 ...
    ptr += vl; avl -= vl;
}
```

> 注意：ggml 的量化内核**大多没用 strip-mining**，而是固定 `vl=block_size`，因为量化块大小固定（32 或 256）。这正是它们不随 VLEN 扩展的原因。

---

## 第 2 章 · intrinsic 命名规则（吃透这个 = 能读所有内核）

RVV C intrinsic 名字是**拼出来的**，看懂规则就不用背。通式：

```
__riscv_  v<op>  _<operands>  _<type>[ _<src_type> ][ _<mask/policy> ]
```

### 2.1 拆解一个例子

```
__riscv_vfredmax_vs_f32m8_f32m1
        └─op──┘ └┘ └──┬──┘ └──┬──┘
        浮点归约最大  vs  输入f32m8  输出f32m1
```

- `vf` = 浮点操作；`redmax` = 归约求最大。
- `_vs` = 操作数形态：vector + scalar。
- `_f32m8` 输入类型，`_f32m1` 输出类型（归约把一整组规约成 1 个标量，放在 m1 的第 0 元素）。

### 2.2 op 前缀对照

| 前缀 | 含义 | 例 |
|---|---|---|
| `vle{N}` / `vse{N}` | 载入/存回 N 位元素 | `vle8`, `vse32` |
| `vadd vsub vmul` | 整数加减乘 | `vsub_vx_i8m1` |
| `vand vor vsrl vsll` | 位运算（与/或/逻辑右移/左移）| `vand_vx_u8m1` |
| `vw{op}` | **加宽**（widening，输出位宽翻倍）| `vwmul`, `vwadd`, `vwredsum` |
| `vn{op}` / `v{op}.w` | **窄化**（narrowing，输出位宽减半）| `vncvt`, `vfncvt` |
| `vf{op}` | 浮点版 | `vfmul`, `vfmacc`, `vfadd` |
| `vredsum vredmax` | 整数归约；`vfred*` 浮点归约 | `vwredsum`（加宽归约）|
| `vmacc` / `vfmacc` | 乘累加 `acc += a*b` | `vmacc_vx`, `vfmacc_vv` |
| `vmv` / `vfmv` | 移动（向量内/标量进出）| `vmv_x_s`（取标量）|
| `vmseq vmslt` | 比较出**掩码** | `vmseq_vx_..._b8` |
| `vid` | 写入 0,1,2,3,... 索引 | `vid_v_i32m1` |
| `vrgather` | 按索引 gather（查表）| 码本量化用 |
| `vget` / `vset` | 从大 LMUL 组里取/放子寄存器 | `vget_v_i16m2_i16m1` |
| `vreinterpret` | 位不变重解释类型（u8<->i8）| 无指令开销 |
| `vsetvl(i)` | 配置 SEW/LMUL，返回 vl | `vsetvl_e8m1` |

### 2.3 operands 形态后缀

| 后缀 | 含义 |
|---|---|
| `_vv` | 两个向量 |
| `_vx` | 向量 + 标量（整数 GPR）|
| `_vf` | 向量 + 标量（浮点）|
| `_vi` | 向量 + 立即数 |
| `_vs` | 向量 -> 标量（归约专用）|
| `_wv` / `_wx` | 加宽，且第一个操作数已是宽类型 |

### 2.4 type 后缀

```
{i|u|f}{8|16|32|64}{m1|m2|m4|m8|mf2|mf4|mf8}
 │      │            └ LMUL（mf2=1/2 分数 LMUL）
 │      └ 元素位宽
 └ i=有符号整数 u=无符号 f=浮点
```

### 2.5 掩码 / policy 后缀

- `_m`：受掩码控制；`_mu`：masked + mask-undisturbed（掩码外元素保持原值）；`_tu`：tail-undisturbed。
- 例 `vsub_vx_i8m1_mu(mask, dst, src, x, vl)`：仅在 mask 为真的 lane 上做 `src - x`，其余保持 `dst`。

### 2.6 掩码类型 vboolN

比较类指令产出**掩码寄存器**，类型是 `vboolN_t`，其中 `N = SEW / LMUL`：

```
e8,m1  -> N=8/1=8   -> vbool8_t   （所以 vmseq_..._b8）
e16,m2 -> N=16/2=8  -> vbool8_t
e32,m1 -> N=32      -> vbool32_t
```

记住 N 是"每个掩码位覆盖多少 SEW"，不是 lane 数。

> 到这里规则就讲完了。后面任何一个没见过的 intrinsic，按 2.1 的方法拆，都能秒懂。下面进入代码。

---

## 第 3 章 · 量化块格式速览（读内核前必看）

量化 = 把一组 float 压成 `{缩放因子, 低位整数}`。点积时反过来：整数相乘求和，再乘缩放。块格式（`ggml-common.h`）：

```c
#define QK8_0 32
typedef struct { ggml_half d; int8_t qs[32]; } block_q8_0;
//  d=fp16 缩放, qs=32 个 int8。一个块 = 2 + 32 = 34 字节。

#define QK4_0 32
typedef struct { ggml_half d; uint8_t qs[16]; } block_q4_0;
//  32 个权重打包进 16 字节：低 nibble=权重0..15, 高 nibble=权重16..31, 零点=8。

typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;
//  K 系列点积时激活方用的格式：256 个 int8 + 每 16 个一组的预求和 bsums。

#define QK_K 256
typedef struct {              // q3_K: 256 权重/块, 3.4 bit/权重
    uint8_t hmask[32];        // 每个权重的"第3位"(高位)
    uint8_t qs[64];           // 每个权重的低 2 位
    uint8_t scales[12];       // 16 个 6-bit 子块缩放(压缩存)
    ggml_half d;              // super-block 总缩放
} block_q3_K;
```

**两层缩放**是 K 系列的关键：一个 256 元素的 super-block 里再分 16 个子块，每子块一个 6-bit 小缩放，外面再乘一个 fp16 总缩放。传统量化（q4_0/q8_0）只有一层缩放、块也只有 32。

---

## 第 4 章 · 代表函数逐行

### 4.1 【量化入门】`quantize_row_q8_0` (quants.c:32)

把一行 float 量化成 q8_0 块。最短，用来认 intrinsic。

```c
size_t vl = QK8_0;                  // =32, 固定一个块的元素数（注意：写死，不随 VLEN 扩展）
for (int i = 0; i < nb; i++) {
    vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x+i*QK8_0, vl);   // 载入 32 个 float 到 f32m8
```
- `vle32_v_f32m8`：按 2.2/2.4 规则 = 载入 32 位元素，目标 f32m8（m8 占 8 个寄存器）。

```c
    vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);              // 取绝对值
    vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);           // 标量 0 灌进一个 m1 向量当归约初值
    vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vfabs,tmp,vl); // 归约求最大 -> 标量在[0]
    float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);                    // 把向量[0]取成普通 float
```
- 这四行做的就是 `amax = max(|x[0..31]|)`。
- `vfmv_v_f` = 浮点标量广播进向量；`vfmv_f_s` = 反向，取出向量第 0 个元素。归约结果总在第 0 元素。

```c
    const float d  = amax / 127.0f;         // 缩放：让最大值映射到 int8 的 127
    const float id = d ? 1.0f/d : 0.0f;     // 倒数（乘比除快）
    y[i].d = GGML_CPU_FP32_TO_FP16(d);      // 存 fp16 缩放（这步会生成 zfh 半精度指令）
    vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);   // 每个元素 *id，缩放到 [-127,127]
```

```c
    vint16m4_t vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);  // 浮点->整数 + 窄化: f32m8 -> i16m4
    vint8m2_t  vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);    // 整数窄化: i16m4 -> i8m2
    __riscv_vse8_v_i8m2(y[i].qs, vs, vl);                // 存 32 个 int8
}
```
- `vfncvt_x_f_w`：`f`=源是浮点，`x`=目标整数，`w`=源是宽类型（窄化）。f32(32位)->i16(16位)，LMUL 也 m8->m4。
- `vncvt_x_x_w`：纯整数窄化 i16->i8，m4->m2。
- 两步窄化是因为 RVV 一条窄化指令只能减半位宽，32->8 要走两步。

**这一段学到的 intrinsic（后面不再解释）**：`vle32/vse8, vfabs, vfmv_v_f, vfredmax_vs, vfmv_f_s, vfmul_vf, vfncvt, vncvt`。

> 推广：`quantize_row_q8_1`(:73) 只是多算一个 `s=sum(qs)*d`（用 `vwredsum` 加宽归约，见下）。`quantize_row_q8_K`(:122) 块是 256、还填 `bsums`。骨架完全一样。

---

### 4.2 【点积骨架】`ggml_vec_dot_q8_0_q8_0` (quants.c:435)

计算两个 q8_0 量化向量的点积。你已在 gdb 里逐指令看过，这里对照源码定型。

```c
const int qk = QK8_0;               // 32
const int nb = n / qk;              // 块数, n=896 -> 28
size_t vl = qk;                     // 32（又是写死一个块）
for (; ib < nb; ++ib) {
    vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[ib].qs, vl);   // 载 32 个权重 int8
    vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[ib].qs, vl);   // 载 32 个激活 int8

    vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);  // 加宽乘: i8×i8 -> i16
    vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);            // 归约初值 0
    vint32m1_t v_sum  = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl); // 加宽归约求和 -> i32 标量
    int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);                 // 取出整数和

    sumf += sumi * (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d)); // 乘两层 fp16 缩放
}
*s = sumf;
```

**核心四步（量化点积的万能套路）**：
1. `vle8` 载入两边 int8；
2. `vwmul` 加宽相乘（防溢出）；
3. `vwredsum` 加宽归约求和 -> 一个整数 `sumi`；
4. 标量收尾：`sumi * d_x * d_y` 累加进 `sumf`。

新 intrinsic：`vwmul_vv`（加宽乘）、`vwredsum_vs`（加宽归约）、`vmv_x_s`（取标量整数）。

> 低效点（你 gdb 实测）：vl=32 vs VLMAX=256，且每块一次 `vwredsum`（归约延迟高）。优化方向见第 7 章。

---

### 4.3 【位解包】`ggml_vec_dot_q4_0_q8_0` (quants.c:222)

q4_0 把 32 个权重压进 16 字节。比 q8_0 多一步"解包 nibble"，并用 `vwmacc` 省一次归约。

```c
size_t vl = qk / 2;                 // =16！一次处理半个块（16 字节 = 32 个 nibble）
for (; ib < nb; ++ib) {
    vuint8m1_t tx = __riscv_vle8_v_u8m1(x[ib].qs, vl);     // 载 16 字节打包权重
    vint8m1_t  y0 = __riscv_vle8_v_i8m1(y[ib].qs,    vl);  // 激活前 16
    vint8m1_t  y1 = __riscv_vle8_v_i8m1(y[ib].qs+16, vl);  // 激活后 16

    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);   // 低 nibble -> 权重 0..15
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);   // 高 nibble -> 权重 16..31

    vint8m1_t x_ai = __riscv_vreinterpret_v_u8m1_i8m1(x_a); // u8 当 i8 看（位不变，0 开销）
    vint8m1_t x_li = __riscv_vreinterpret_v_u8m1_i8m1(x_l);

    vint8m1_t v0 = __riscv_vsub_vx_i8m1(x_ai, 8, vl);      // 减零点 8 -> 有符号权重
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(x_li, 8, vl);

    vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);          // 低半: v0*y0
    vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl); // 累加高半: += v1*y1

    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl); // 16 个元素归约一次
    int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);
    sumf += sumi * GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
}
```

**比 q8_0 新增的点**：
- 解包：`vand`(取低 4 位) + `vsrl`(取高 4 位) + `vsub`(减零点 8)。这是所有 4-bit 类型的通用手法。
- `vreinterpret`：u8 和 i8 位模式一样，只是告诉编译器换个类型看，**不产生指令**。
- `vwmacc_vv`：加宽乘累加 `acc += v1*y1`。用它把"低半积"和"高半积"合到一个 i16 向量里，最后**只归约 16 个元素一次**（q8_0 是归约 32 个）。

> 推广：`q5_0`(:328) 再多一步——第 5 位存在单独的 `qh[4]` bitmask 里，用 `vand`+移位拼回去；`q4_1/q5_1` 多一个 `min` 偏移项。解包手法同源。

---

### 4.4 【K 量化 + VLEN 特化】`ggml_vec_dot_q3_K_q8_K` (分发器 :1607)

这是**学优化的核心**：K 量化块大（256）、两层缩放，且本类型按 VLEN 写了多个版本。

#### 4.4.1 运行时分发（:1607）

```c
void ggml_vec_dot_q3_K_q8_K(...) {
#if defined __riscv_xtheadvector
    ggml_vec_dot_q3_K_q8_K_xtheadvector(...);   // T-Head 0.7.1 老变体
#elif defined __riscv_v
    switch (__riscv_vlenb() * 8) {              // vlenb=VLEN字节数, *8=位数
        case 128:  ..._vl128(...);  break;
        case 256:  ..._vl256(...);  break;
        case 512:  ..._vl512(...);  break;
        case 1024: ..._vl1024(...); break;
        default:   ..._generic(...); break;     // 标量兜底
    }
#else
    ..._generic(...);
#endif
}
```
- `__riscv_vlenb()` 返回 VLEN 的**字节**数（运行时读 `vlenb` CSR）。×8 得位宽，据此选最优内核版本。
- **这就是"VLEN 特化"的调度骨架**。你要给 q8_0/q4_K 做优化，就是仿照这个，给它们补上 `_vlNNN` 版本和这套 switch。

#### 4.4.2 看 `_vl256` 版（:1257，intrinsics 写法，最易读）

先解压 6-bit 子块缩放（标量位运算，不用细抠，知道在干嘛即可）：

```c
memcpy(aux, x[i].scales, 12);
utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
... // 把 12 字节里压缩的 16 个 6-bit 缩放，拆成 16 个 int8 放进 utmp
int8_t * scale = (int8_t *)utmp;
for (int j = 0; j < 16; ++j) scale[j] -= 32;   // 6-bit 缩放是有偏移的，减 32 还原符号
```

主循环：256 个权重分成 2 段、每段 128（`j += 128`）：

```c
size_t vl = 32;
vuint8m1_t vqh = __riscv_vle8_v_u8m1(qh, vl);   // 载入高位 mask（每权重第3位）

for (int j = 0; j < QK_K; j += 128) {
    vuint8m1_t q3_x = __riscv_vle8_v_u8m1(q3, vl);   // 载 32 字节低位

    // 一个字节里塞了 4 个权重的低 2 位，移位+掩码拆出 4 组：
    vint8m1_t q3_0 = ..vand(q3_x, 0x03)..;              // bit 0-1
    vint8m1_t q3_1 = ..vand(vsrl(q3_x,2), 0x03)..;      // bit 2-3
    vint8m1_t q3_2 = ..vand(vsrl(q3_x,4), 0x03)..;      // bit 4-5
    vint8m1_t q3_3 = ..vand(vsrl(q3_x,6), 0x03)..;      // bit 6-7
```
- 这就是 4.3 的解包手法推广到 2-bit：一个 u8 拆成 4 个 2-bit 值。

补"第 3 位"（用掩码做条件减法）：

```c
    vuint8m1_t qh_m0 = __riscv_vand_vx_u8m1(vqh, m, vl);      // 取出 hmask 当前位
    vbool8_t vmask_0 = __riscv_vmseq_vx_u8m1_b8(qh_m0, 0, vl);// ==0 -> 掩码真
    vint8m1_t q3_m0 = __riscv_vsub_vx_i8m1_mu(vmask_0, q3_0, q3_0, 0x4, vl); // 掩码真处 q3_0-=4
    m <<= 1;
    // q3_1/q3_2/q3_3 同理，m 每次左移取下一位
```
- **新概念：掩码运算。** `vmseq_vx_..._b8` 比较出 `vbool8_t` 掩码；`vsub_vx_..._mu` 是掩码版减法（mask-undisturbed：掩码假的 lane 保持原值）。
- 含义：3-bit 量化值 = 低 2 位 - (高位为 0 ? 4 : 0)，把无符号 0..7 映射到有符号 -4..3。

乘激活、按子块缩放、归约：

```c
    vint16m2_t a0 = __riscv_vwmul_vv_i16m2(q3_m0, __riscv_vle8_v_i8m1(q8, vl), vl); // 权重×激活 -> i16
    ... a1,a2,a3 同理（q8 偏移 32/64/96）

    vl = 16;
    // 一个 i16m2 含 32 个元素=两个 16 的子块，用 vget 取出各半，分别乘对应 6-bit 缩放：
    vint32m2_t aux0_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0,0), scale[0], vl);
    vint32m2_t aux0_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0,1), scale[1], vl);
    ... 8 个子块 ...

    // 分段累积归约：每个子块加进运行总和（vredsum 的初值串成链）
    vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(vadd(aux0_0,aux0_1), vzero, vl);
    vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(vadd(aux1_0,aux1_1), isum0, vl);
    ... isum2, isum3（前一个的结果当下一个的初值，省得分别取标量再相加）
    sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);
}
const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;  // super-block 总缩放
sumf += d * sum_t;
```

**这一段新增的 intrinsic / 思想**：
- `vmseq`（出掩码）、`_mu` 掩码运算、`vbool8_t`；
- `vget_v_i16m2_i16m1`：从 m2 组里取出第 0/1 个 m1 子寄存器（拆子块）；
- `vredsum`（非加宽归约）+ **初值串链**：把多次归约的中间和接力传递，避免反复 `vmv_x_s`；
- **两层缩放**：子块 6-bit `scale[]` 在向量里乘，super-block fp16 `d` 在标量收尾乘。

> `_vl128` 版（:1107）逻辑相同，但因为 VLEN 只有 128、装不下，改用**内联汇编**精细排布寄存器、一次算 128。读它能看到"小 VLEN 怎么抠"；日常学习读 `_vl256` 足够。
>
> 推广：`q4_K/q5_K/q6_K` 都是这套"super-block + 子块缩放 + 解包 + 加宽乘 + 分段归约"，只是位宽和打包方式不同。看懂 q3_K 的 `_vl256`，它们就都能读。**注意 q4_K/q5_K/q6_K 没做 VLEN 特化**（只有单一版本）——是你优化的好目标。

---

### 4.5 【重排乘法】`ggml_gemv_q8_0_16x1_q8_0` (repack.cpp:447)

前面的 `vec_dot` 算"一行权重 × 一列激活"。repack 把**16 列权重交织存放**，一次算 16 个输出 —— 这是"填满向量"的正解。

```c
const int ncols_interleaved = 16;
const block_q8_0 * a_ptr = (const block_q8_0 *) vy;        // 激活（单列）
for (int x = 0; x < nc / ncols_interleaved; x++) {        // 每次出 16 列结果
    const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb); // 16 列交织的权重

    vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);  // 16 个 float 累加器（每列一个）

    for (int l = 0; l < nb; l++) {                         // 遍历块
        vint32m2_t sumi = __riscv_vmv_v_x_i32m2(0, 16);    // 16 个 int 累加器

        for (int i = 0; i < QK8_0; i++) {                  // 块内 32 个元素，逐元素
            vint8mf2_t b_0 = __riscv_vle8_v_i8mf2(&b_ptr[l].qs[i*16], 16); // 16 列在同一元素位的权重
            sumi = __riscv_vwadd_wv_i32m2(sumi,
                       __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i], 16), 16); // 16 列 ×同一个激活标量
        }
        // 16 列各自的 fp16 缩放，一次性向量化乘
        vfloat16m1_t b_d = __riscv_vle16_v_f16m1((const _Float16 *)b_ptr[l].d, 16);
        vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(const _Float16 *)&a_ptr[l].d, 16);
        sumf = __riscv_vfmacc_vv_f32m2(sumf, __riscv_vfcvt_f_x_v_f32m2(sumi, 16), d_0, 16);
    }
    __riscv_vse32_v_f32m2(s + x*16, sumf, 16);             // 存 16 个结果
}
```

**关键差异（对比 4.2 的 vec_dot）**：
- 向量的 lane 维度从"块内元素"变成了"**16 个输出列**"。`vwmul_vx(b_0, a_ptr[l].qs[i], 16)` = 16 列权重 × **同一个**激活标量。
- **没有块内归约**！因为不是在"块内元素"方向求和，而是 16 列并行累加，求和沿着外层 `i`/`l` 循环用 `vwadd_wv` 累加完成。归约延迟问题消失。
- `vint8mf2_t`：**分数 LMUL**（半个寄存器），因为只要 16 个 lane。
- `vfwmul_vf`：加宽浮点乘（f16×f16 -> f32）；`vfcvt_f_x_v`：int32 -> float32；`vfmacc_vv`：float 乘累加。

新 intrinsic：`vwadd_wv`（加宽加，宽+窄）、`vwmul_vx`、`vfwmul_vf`、`vfcvt_f_x_v`、`vfmacc_vv`、分数 LMUL `mf2`。

> 这就是为什么 repack 路线效率高：用"输出列"当向量维度，天然填满、无归约。代价是权重要预先重排成 `q8_0x16` 交织布局。`ggml_gemm_*`(:1336) 是它的"多激活列"版（prefill 用）。

---

## 第 5 章 · 通用 F32 路径（非量化算子）

RMSNorm、softmax、`vec_dot_f32` 等不涉及量化，走 `simd-mappings.h` 的宏（`vec.cpp`/`ops.cpp` 调用）：

```c
// simd-mappings.h:1265, #elif defined(__riscv_v_intrinsic)
#define GGML_F32_EPR  4                      // 每向量 4 个 float（写死 128bit！）
#define GGML_F32x4    vfloat32m1_t
#define GGML_F32x4_FMA(a,b,c) __riscv_vfmacc_vv_f32m1(a,b,c, GGML_F32_EPR)
#define GGML_F32x4_LOAD(x)    __riscv_vle32_v_f32m1(x, GGML_F32_EPR)
#define GGML_F32x4_ADD(a,b)   __riscv_vfadd_vv_f32m1(a,b, GGML_F32_EPR)
```

- 上层算子用 `GGML_F32_VEC_*` 这套抽象宏，换架构只换映射。
- **同样写死 4-wide**：VLEN=1024 下 f32m1 能放 32 个 float，却只用 4 个 -> 又是 12.5%。这是量化内核之外的另一类全局优化点。

---

## 第 6 章 · 运行时检测 `cpu-feats.cpp`

```c
struct riscv64_features {
    bool has_rvv = false;
    riscv64_features() {
        struct riscv_hwprobe probe;
        probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
        syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0); // 问内核支持哪些扩展
        has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);    // 检测 V 扩展
    }
};
```
- Linux `riscv_hwprobe` 系统调用查 CPU 能力，决定 CPU 后端"得分"，多后端时择优。简单了解即可。

---

## 第 7 章 · 贯穿主题 & 优化机会

### 7.1 一句话主题

> **ggml 的 RVV 量化内核大多把 `vl` 写死成一个量化块（32 或 256），在 VLEN=128 上刚好填满；VLEN>128 时按比例浪费向量宽度。** 只有 q1_0/q2_K/q3_K/iq4_nl/mxfp4 补了 `_vlNNN` 多版本。

### 7.2 三个反复出现的低效点

1. **向量没填满**：`vl=block` vs `VLMAX`。VLEN=1024 时常只用 12.5%。
2. **每块一次归约**：`vwredsum`/`vredsum` 延迟高、串行化。块越小、归约越频繁。
3. **频繁切 vtype**：一圈里 `e8->e16->e32` 多次 `vsetvli`，真实硬件可能停顿。

### 7.3 两种"填满向量"的范式（项目里都有现成参考）

| 范式 | 思路 | 参考 |
|---|---|---|
| **多块/大 vl** | 一次吃多个块，分段归约 | q3_K `_vl256`/`_vl512` |
| **列交织 repack** | 用"输出列"当向量维度，无块内归约 | `gemv/gemm_*_16x1` |

### 7.4 未特化、可下手的目标

`ggml_vec_dot_q8_0_q8_0`、`q4_K`、`q5_K`、`q6_K`、通用 F32 宏 —— 都只有单一固定-vl 版本。仿 4.4.1 的 switch + 4.4.2 的多块写法补一个 `_vl1024`，用 `llama-bench` 对比 tg/pp，就是一个完整的优化实验。

---

## 附录 A · RVV intrinsic 速查（本文出现过的）

| intrinsic | 作用 | 首次出现 |
|---|---|---|
| `vsetvl(i)_e8m1` | 配置 SEW/LMUL，返回 vl | 1.4 |
| `vle{N}/vse{N}` | 载入/存回 | 4.1 |
| `vfabs` `vfmul_vf` | 浮点绝对值/乘标量 | 4.1 |
| `vfredmax_vs` `vfmv_f_s` `vfmv_v_f` | 浮点归约最大/取标量/广播 | 4.1 |
| `vfncvt` `vncvt` | 浮点->整数窄化 / 整数窄化 | 4.1 |
| `vwmul_vv` | 加宽乘 i8×i8->i16 | 4.2 |
| `vwredsum_vs` | 加宽归约求和 | 4.2 |
| `vmv_x_s` `vmv_v_x` | 取标量整数/广播整数 | 4.2 |
| `vand_vx` `vsrl_vx` `vsll_vi` | 位与/逻辑右移/左移 | 4.3 |
| `vreinterpret` | 位不变换类型（0 开销）| 4.3 |
| `vsub_vx` | 减标量 | 4.3 |
| `vwmacc_vv` | 加宽乘累加 | 4.3 |
| `vmseq_vx_..._b8` | 比较出掩码 vbool8 | 4.4 |
| `vsub_vx_..._mu` | 掩码版减法 | 4.4 |
| `vget_v_i16m2_i16m1` | 取子寄存器 | 4.4 |
| `vredsum_vs` | 归约求和（非加宽）+ 初值串链 | 4.4 |
| `vwadd_wv` | 加宽加（宽+窄）| 4.5 |
| `vwmul_vx` `vfwmul_vf` | 加宽乘标量 / 加宽浮点乘标量 | 4.5 |
| `vfcvt_f_x_v` | int->float 转换 | 4.5 |
| `vfmacc_vv` `vfadd_vv` | 浮点乘累加/加 | 4.5 / 5 |
| `vid` `vslidedown` `vslideup` | 索引序列/向量滑移（q3_K 内联汇编里）| 4.4 |
| `vrgather` `vluxei` | gather/查表（码本量化 iq*）| 7.4 待学 |

## 附录 B · 学习顺序与过关标准

| 阶段 | 读 | 过关标准 |
|---|---|---|
| 1 | 第 1-2 章 | 能手工解码 `vtype=0xc1`，会拆任意 intrinsic 名 |
| 2 | 4.1 quantize_q8_0 | 能说清"一行 float -> {d, 32×int8}" |
| 3 | 4.2 vec_dot_q8_0 | 能默写"载入->vwmul->vwredsum->缩放"四步 |
| 4 | 4.3 vec_dot_q4_0 | 能解释 nibble 解包 + 零点 8 + vwmacc 省归约 |
| 5 | 4.4 q3_K `_vl256` + 分发器 | 能讲清两层缩放、掩码补高位、分段归约、VLEN 分发 |
| 6 | 4.5 gemv_16x1 | 能说出 repack 为何无块内归约、代价是什么 |
| 7 | 第 7 章 | 能指出 q8_0/q4_K 的优化点并说出改法 |
| 8（毕业）| 动手 | 给一个未特化内核补 `_vl1024`，llama-bench 验证 |

---

*配套调试：每个函数都可在 QEMU + gdb 里下断点实跑（见 `readme.md` §5）。建议边读边 `disassemble`/`info registers vtype vl` 对照，源码与汇编互相印证。*
