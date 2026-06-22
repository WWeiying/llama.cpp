# ggml RISC-V / RVV 内核逐行教学（专业版）

> 目标：读完这份文档 + 在 gdb 里实跑，你能看懂 `ggml/src/ggml-cpu/arch/riscv/` 下**所有** RVV 内核，理解它们的性能特性，并能自己写/改一个。
>
> 方法：先把"RVV 编程模型 + intrinsic 命名规则"讲透一次（第 1-2 章），后面已讲过的语法不再重复，只点新东西。每类量化只挑一个代表，吃透代表 = 吃透一类。
>
> 配套源码全部**内嵌文档**，行号以当前仓库为准：
> - `arch/riscv/quants.c` —— 量化 + vec_dot
> - `arch/riscv/repack.cpp` / `repack.h` —— gemv/gemm 重排
> - `simd-mappings.h` —— 通用 F32 宏
> - `arch/riscv/cpu-feats.cpp` —— 运行时检测
>
> 阅读建议：配合 `ggml-core-tutorial.md`（上层篇）食用，那份讲"内核怎么被调起来、数据从哪来"。

---

# 第一部分 · RVV 基础

## 第 1 章 · RVV 编程模型（地基）

### 1.1 五个核心量：VLEN / SEW / LMUL / vl / vtype

RVV 是**向量长度无关（VLA, Vector-Length Agnostic）**的：同一份机器码能跑在任意向量宽度的硬件上（128 位到 65536 位）。这与 x86 SSE/AVX（宽度写死在指令里）根本不同。先记住 5 个量：

| 量 | 含义 | 谁决定 |
|---|---|---|
| **VLEN** | 一个向量寄存器的物理位宽 | 硬件固定（你 QEMU 配的 `vlen=1024`）|
| **SEW** | Selected Element Width，每个元素多少位（8/16/32/64）| 软件用 `vsetvli` 设 |
| **LMUL** | Length MULtiplier，几个寄存器合成一组（1/2/4/8 或分数 1/2,1/4,1/8）| 软件用 `vsetvli` 设 |
| **vl** | Vector Length，这一条向量指令实际处理多少个元素 | `vsetvli` 返回 |
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

> **贯穿全书的主题**：很多 ggml RVV 内核把 `vl` 写死成"一个量化块的大小"（32 或 256），这在 VLEN=128 上刚好填满，在 VLEN=1024 上只用 1/8。**这是全项目最大的优化空间。**

### 1.2 向量寄存器堆与 LMUL 分组

```
32 个向量寄存器 v0..v31，每个 VLEN 位宽（你这里 1024 位 = 128 字节）：

         <------------------ VLEN=1024 bit ------------------>
   v0   [ e | e | e | ... ]      e8 时一个 v 装 128 个 int8
   v1   [ ........ ]
   ...
   v31  [ ........ ]

LMUL 把多个寄存器拼成一组：
   m1 -> 1 个寄存器  (v8)              e8m1: 128 个元素
   m2 -> 2 个寄存器  (v8,v9)          e8m2: 256 个元素   ← 你 gdb 看到的
   m4 -> 4 个寄存器  (v8,v9,v10,v11)  e8m4: 512 个元素
   m8 -> 8 个寄存器  (v8..v15)        e8m8: 1024 个元素
   mf2-> 半个寄存器(分数LMUL)          e8mf2: 64 个元素   ← repack 用
```

**对齐规则**：m2 类型只能用偶数号起点（v0/v2/v4...），m4 用 v0/v4/v8...，m8 用 v0/v8/v16/v24。写内核选寄存器号时必须遵守，否则编译/执行错。

- `v0` 还兼任**掩码寄存器**（见 1.6）。用掩码时别拿 v0 存数据。

### 1.3 加宽 / 窄化：量化点积的命脉

int8 × int8 最大 127×127=16129，**溢出 int8（±127）**，必须升位。RVV 提供成对的加宽/窄化指令：

```
加宽 widening (名字带 w)：输出位宽翻倍 -> LMUL 也翻倍
   vwmul:  i8m1 × i8m1 -> i16m2     (8位*8位 = 16位)
   vwredsum: i16m4 归约 -> i32       (求和时再升一级防溢出)

窄化 narrowing (名字带 n 或 .w)：输出位宽减半
   vfncvt: f32m8 -> i16m4           (量化时 float 转 int)
   vncvt:  i16m4 -> i8m2            (再砍半到 int8)
```

一条窄化只能减半，所以 f32(32)->i8(8) 要走两步（先 ->i16 再 ->i8），你会在 `quantize_row_q8_0` 看到。

### 1.4 strip-mining：VLA 的标准循环

因为 vl 由硬件能力决定，处理任意长度 n 的"教科书"写法是：

```c
for (size_t avl = n; avl > 0; ) {
    size_t vl = __riscv_vsetvl_e8m1(avl); // 硬件给本次能处理的 vl(<=VLMAX, <=avl)
    // ... 用 vl 处理一段 ...
    ptr += vl; avl -= vl;
}
```
这样一份码在 VLEN=128 跑 8 次、VLEN=1024 跑 1 次，自动适配。

> **ggml 的反模式**：量化内核**大多不用 strip-mining**，而是固定 `vl=block_size`（32 或 256），因为量化块大小固定。这正是它们不随 VLEN 扩展、在大 VLEN 上浪费的根因。读懂这点，你就知道优化的本质 = 把"按块"改成"按 VLMAX strip-mine"。

### 1.5 vtype 寄存器位编码（调试必备）

`vsetvli` 把 SEW/LMUL/policy 打包进 `vtype` CSR。手工解码它在 gdb 里极有用：

```
vtype 低 8 位：
  bit 7   : vma  (mask agnostic)
  bit 6   : vta  (tail agnostic)
  bit 5:3 : vsew  000=e8 001=e16 010=e32 011=e64
  bit 2:0 : vlmul 000=m1 001=m2 010=m4 011=m8  101=mf8 110=mf4 111=mf2

例：vtype=0xc1=0b1100_0001
   vma=1, vta=1, vsew=000(e8), vlmul=001(m2)  ->  e8,m2,ta,ma
```

- **tail agnostic (ta)**：vl 之外（尾部）的元素，硬件可写垃圾值或保持原值（实现自选），给硬件优化余地。
- **mask agnostic (ma)**：掩码屏蔽掉的元素同理。
- 对应的"undisturbed"策略（tu/mu）则**保证保持原值**，代价是可能更慢。ggml 几乎全用 `ta,ma`（最快），只有掩码版减法等少数用 `mu`（需保留未选中 lane 的值）。

### 1.6 掩码与 vstart / 取整（补充）

- **掩码**：比较类指令（`vmseq` 等）产出 `vboolN_t`，存进掩码寄存器（通常 v0）。后续指令带 `_m/_mu` 后缀可受掩码控制，只在为真的 lane 上动作。`N = SEW/LMUL`（见 2.6）。
- **vstart**：指令从第几个元素开始（异常恢复用），正常代码是 0，不用管。
- **取整**：浮点转整数（`vfncvt`）用 `frm`（浮点舍入模式，默认 round-to-nearest-even）；定点指令用 `vxrm`。ggml 量化依赖默认 round-to-nearest，与标量 ref 实现的 `roundf` 可能有 ±1 的极小差异（不影响推理质量）。

---

## 第 2 章 · intrinsic 命名规则（吃透这个 = 能读所有内核）

RVV 的 C intrinsic 名字是**拼出来的**，看懂规则就不用背任何一个。通式：

```
__riscv_  v<op>  _<operands>  _<type>[ _<src_type> ][ _<mask/policy> ]
```

### 2.1 拆解四个例子（解码训练）

```
① __riscv_vle32_v_f32m8(ptr, vl)
   vle32 = 载入32位元素; _v = 单位步长向量载入; _f32m8 = 目标类型
   读：把 ptr 处连续的 vl 个 float 载入 f32m8 寄存器组

② __riscv_vwmul_vv_i16m4(a, b, vl)
   vw=加宽 mul=乘; _vv=两个向量; _i16m4=结果类型
   读：a、b 逐元素相乘，结果加宽到 i16m4

③ __riscv_vfredmax_vs_f32m8_f32m1(v, init, vl)
   vf=浮点 redmax=归约求最大; _vs=向量+标量; _f32m8 输入 _f32m1 输出
   读：求 v 里 vl 个元素的最大值，和 init[0] 一起，结果放 f32m1 的[0]

④ __riscv_vsub_vx_i8m1_mu(mask, dst, src, x, vl)
   vsub=减; _vx=向量减标量; _i8m1; _mu=掩码且未选中lane保持dst原值
   读：mask 为真的 lane 算 src-x，其余保持 dst
```

掌握这四个，任何没见过的 intrinsic 都能照此拆解。

### 2.2 op 前缀对照

| 前缀 | 含义 | 例 |
|---|---|---|
| `vle{N}` / `vse{N}` | 单位步长 载入/存回 N 位元素 | `vle8`, `vse32` |
| `vlse{N}` / `vsse{N}` | **跨步**载入/存（带 stride）| 较慢，少用 |
| `vluxei` / `vsuxei` | **索引(gather/scatter)** | 码本量化用 |
| `vadd vsub vmul` | 整数加减乘 | `vsub_vx_i8m1` |
| `vand vor vxor vsrl vsll vsra` | 位运算（与/或/异或/逻辑右移/左移/算术右移）| `vand_vx_u8m1` |
| `vw{op}` | **加宽**（输出位宽翻倍）| `vwmul`, `vwadd`, `vwredsum`, `vwmacc` |
| `vn{op}` / `vf n cvt` | **窄化**（输出位宽减半）| `vncvt`, `vfncvt` |
| `vf{op}` | 浮点版 | `vfmul`, `vfmacc`, `vfadd`, `vfabs` |
| `vred{sum,max,min}` / `vfred*` | 整数/浮点归约 | `vredsum`, `vwredsum`(加宽), `vfredmax` |
| `vmacc` / `vfmacc` / `vnmsac` | 乘累加 `acc += a*b` / 乘减 | `vmacc_vx`, `vfmacc_vv` |
| `vmv` / `vfmv` | 移动（向量内/标量进出/整组复制）| `vmv_x_s`, `vmv1r` |
| `vmseq vmslt vmsne` | 比较出**掩码** | `vmseq_vx_..._b8` |
| `vid` / `viota` | 写索引 0,1,2,... / 掩码前缀和 | `vid_v_i32m1` |
| `vrgather` | 按索引在向量内 gather（查表）| 码本量化核心 |
| `vslideup/down` `vslide1up/down` | 向量元素整体滑移 | q3_K 汇编里 |
| `vget` / `vset` | 从大 LMUL 组里取/放子寄存器 | `vget_v_i16m2_i16m1` |
| `vreinterpret` | 位不变重解释类型（u8<->i8）| **无指令开销** |
| `vsetvl(i)` | 配置 SEW/LMUL，返回 vl | `vsetvl_e8m1` |

### 2.3 operands 形态后缀

| 后缀 | 含义 |
|---|---|
| `_vv` | 两个向量 |
| `_vx` | 向量 + 标量（整数 GPR）|
| `_vf` | 向量 + 标量（浮点 FPR）|
| `_vi` | 向量 + 立即数 |
| `_vs` | 向量 -> 标量（归约专用，结果在[0]）|
| `_wv` / `_wx` | 加宽运算，且第一个操作数已是宽类型 |

### 2.4 type 后缀

```
{i|u|f}{8|16|32|64}{m1|m2|m4|m8|mf2|mf4|mf8}
 │      │            └ LMUL（mf2=1/2 分数 LMUL，省寄存器）
 │      └ 元素位宽
 └ i=有符号整数 u=无符号 f=浮点
```

### 2.5 掩码 / policy 后缀

`_m`：受掩码控制；`_mu`：masked + mask-undisturbed（未选中 lane 保持原值）；`_tu`：tail-undisturbed；`_tumu`：两者都 undisturbed。不带后缀 = agnostic（最快）。

### 2.6 掩码类型 vboolN

比较类产出 `vboolN_t`，其中 **`N = SEW / LMUL`**（每个掩码位覆盖多少 SEW，不是 lane 数）：

```
e8,m1  -> N=8/1=8   -> vbool8_t   （所以 vmseq_..._b8）
e16,m2 -> N=16/2=8  -> vbool8_t
e32,m1 -> N=32      -> vbool32_t
```

> 规则到此讲完。后面任何没见过的 intrinsic，按 2.1 方法拆即可。

---

## 第 3 章 · 量化块格式速览（读内核前必看）

量化 = 把一组 float 压成 `{缩放因子, 低位整数}`。点积时反过来：整数相乘求和，再乘缩放。格式（`ggml-common.h`）：

```c
#define QK8_0 32
typedef struct { ggml_half d; int8_t qs[32]; } block_q8_0;
//  d=fp16 缩放, qs=32 个 int8。块 = 2+32 = 34 字节。重建: x[i] = d * qs[i]

#define QK4_0 32
typedef struct { ggml_half d; uint8_t qs[16]; } block_q4_0;
//  32 个权重打包进 16 字节：低 nibble=权重0..15, 高 nibble=权重16..31, 零点=8。
//  重建: x[i] = d * (nibble - 8)

#define QK5_0 32
typedef struct { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0;
//  比 q4_0 多 qh[4]=32 个"第5位"，拼成 5-bit 量化值

typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;
//  K 系列点积时激活方格式：256 个 int8 + 每 16 个一组的预求和 bsums

#define QK_K 256
typedef struct {              // q3_K: 256 权重/块, 3.4 bit/权重
    uint8_t hmask[32];        // 每个权重的"第3位"(高位)
    uint8_t qs[64];           // 每个权重的低 2 位
    uint8_t scales[12];       // 16 个 6-bit 子块缩放(压缩存)
    ggml_half d;              // super-block 总缩放
} block_q3_K;

typedef struct {              // q6_K: 256 权重/块, 6.5 bit/权重
    uint8_t ql[128];          // 低 4 位
    uint8_t qh[64];           // 高 2 位
    int8_t  scales[16];       // 16 个 8-bit 子块缩放
    ggml_half d;              // super-block 总缩放
} block_q6_K;
```

**两层缩放**是 K 系列的关键：一个 256 元素的 super-block 里再分 16 个子块，每子块一个小缩放（6-bit 或 8-bit），外面再乘一个 fp16 总缩放。传统量化（q4_0/q8_0）只有一层缩放、块只有 32。

---

# 第二部分 · 代表函数逐行

## 第 4 章 · 五个代表内核

### 4.1 【量化入门】`quantize_row_q8_0` (quants.c:32)

把一行 float 量化成 q8_0 块。最短，用来认 intrinsic。

```c
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    const int nb = k / QK8_0;        // 块数
    block_q8_0 * GGML_RESTRICT y = vy;
    size_t vl = QK8_0;               // =32, 固定一个块（注意：写死，不随 VLEN 扩展）
    for (int i = 0; i < nb; i++) {
        vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x+i*QK8_0, vl);        // 载入 32 个 float

        vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);            // 取绝对值
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);          // 归约初值 0
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vfabs,tmp,vl); // 求最大 -> [0]
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);                  // 取出标量 = max(|x|)

        const float d  = amax / 127.0f;          // 缩放: 让最大绝对值映射到 int8 的 127
        const float id = d ? 1.0f/d : 0.0f;      // 倒数(乘比除快)
        y[i].d = GGML_CPU_FP32_TO_FP16(d);       // 存 fp16 缩放(生成 zfh 半精度指令)

        vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);          // 每元素 *id -> [-127,127]
        vint16m4_t   vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);           // f32->i16 窄化(含取整)
        vint8m2_t    vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);             // i16->i8 窄化
        __riscv_vse8_v_i8m2(y[i].qs, vs, vl);                          // 存 32 个 int8
    }
}
```

**逻辑**：求绝对值最大 `amax` -> 定缩放 `d=amax/127` -> 每个值乘 `1/d` 再四舍五入成 int8。
- `vfmv_v_f` 广播标量进向量；`vfmv_f_s` 反向取出向量[0]。归约结果总在[0]。
- 两步窄化：`vfncvt`(f32->i16) + `vncvt`(i16->i8)，因为一条窄化只能减半（见 1.3）。

**本段学到（后面不再解释）**：`vle32/vse8, vfabs, vfmv_v_f, vfredmax_vs, vfmv_f_s, vfmul_vf, vfncvt, vncvt`。

> 推广：`quantize_row_q8_1`(:73) 多算 `s=sum(qs)*d`（用 `vwredsum`）；`quantize_row_q8_K`(:122) 块为 256、还填 `bsums`。骨架一致。

#### 实数算例（建立数值直觉）

设一个块前 4 个 float 是 `[2.0, -1.0, 0.5, 0.1]`，且 `amax=2.0`：
```
d  = 2.0 / 127 = 0.015748
id = 1/d = 63.5
qs[0] = round(2.0  * 63.5) = round(127.0) = 127
qs[1] = round(-1.0 * 63.5) = round(-63.5) = -64   (round-to-nearest-even)
qs[2] = round(0.5  * 63.5) = round(31.75) = 32
qs[3] = round(0.1  * 63.5) = round(6.35)  = 6
y.d = fp16(0.015748)
重建检验: qs[0]*d = 127*0.015748 = 2.0 ✓   qs[2]*d = 32*0.015748 = 0.504 ≈ 0.5 ✓
```
量化误差就来自这个 round（如 0.5 -> 0.504）。

---

### 4.2 【点积骨架】`ggml_vec_dot_q8_0_q8_0` (quants.c:435)

两个 q8_0 量化向量的点积。你已在 gdb 逐指令看过。

```c
const int qk = QK8_0;               // 32
const int nb = n / qk;              // 块数, n=896 -> 28
size_t vl = qk;                     // 32（又写死一个块）
for (; ib < nb; ++ib) {
    vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[ib].qs, vl);   // 32 个权重 int8
    vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[ib].qs, vl);   // 32 个激活 int8

    vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);   // 加宽乘 i8×i8->i16
    vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);            // 归约初值
    vint32m1_t v_sum  = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl); // 加宽归约 -> i32
    int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);                // 取整数和

    sumf += sumi * (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
}
*s = sumf;
```

**量化点积的万能四步**：
1. `vle8` 载入两边 int8；
2. `vwmul` 加宽相乘（防溢出）；
3. `vwredsum` 加宽归约求和 -> 整数 `sumi`；
4. 标量收尾：`sumi * d_x * d_y` 累加。

数学：`Σ (d_x·qx_i)(d_y·qy_i) = d_x·d_y · Σ qx_i·qy_i`。缩放提到求和外面，整数部分纯 int8 算，最后乘一次浮点 —— 这就是量化能加速的根本原因。

> 低效点（你 gdb 实测）：vl=32 vs VLMAX=256（12.5%），且每块一次 `vwredsum`（高延迟）。见第 7 章性能模型。

---

### 4.3 【位解包】`ggml_vec_dot_q4_0_q8_0` (quants.c:222)

q4_0 把 32 个权重压进 16 字节，比 q8_0 多一步"解包 nibble"，并用 `vwmacc` 省一次归约。

```c
size_t vl = qk / 2;                 // =16！一次处理半块(16 字节 = 32 个 nibble)
for (; ib < nb; ++ib) {
    vuint8m1_t tx = __riscv_vle8_v_u8m1(x[ib].qs, vl);     // 16 字节打包权重
    vint8m1_t  y0 = __riscv_vle8_v_i8m1(y[ib].qs,    vl);  // 激活前 16
    vint8m1_t  y1 = __riscv_vle8_v_i8m1(y[ib].qs+16, vl);  // 激活后 16

    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);   // 低 nibble -> 权重 0..15
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);   // 高 nibble -> 权重 16..31

    vint8m1_t x_ai = __riscv_vreinterpret_v_u8m1_i8m1(x_a);// u8 当 i8 看(位不变, 0 开销)
    vint8m1_t x_li = __riscv_vreinterpret_v_u8m1_i8m1(x_l);

    vint8m1_t v0 = __riscv_vsub_vx_i8m1(x_ai, 8, vl);      // 减零点 8 -> 有符号权重
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(x_li, 8, vl);

    vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);           // 低半: v0*y0
    vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);// 累加高半: += v1*y1

    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl); // 归约 16 个
    int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);
    sumf += sumi * GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
}
```

**比 q8_0 新增**：
- 解包：`vand`(低 4 位) + `vsrl`(高 4 位) + `vsub`(减零点 8)。所有 4-bit 类型通用。
- `vreinterpret`：u8/i8 位模式相同，只换类型看，**不产生指令**。
- `vwmacc_vv`：加宽乘累加，把"低半积"和"高半积"合到一个 i16 向量，**只归约 16 个元素一次**（q8_0 归约 32 个）—— 一个小优化。

> 推广：`q5_0`(:328) 多一步——第 5 位在 `qh[4]` bitmask 里，用 `vand`+移位拼回；`q4_1/q5_1` 多 `min` 偏移项。解包手法同源。

---

### 4.4 【K 量化 + VLEN 特化】`ggml_vec_dot_q3_K_q8_K` (分发器 :1607)

**学优化的核心**：K 量化块大（256）、两层缩放，且本类型按 VLEN 写了多版本。

#### 4.4.1 运行时分发（:1607）

```c
void ggml_vec_dot_q3_K_q8_K(...) {
#if defined __riscv_xtheadvector
    ggml_vec_dot_q3_K_q8_K_xtheadvector(...);   // T-Head 0.7.1 老变体
#elif defined __riscv_v
    switch (__riscv_vlenb() * 8) {              // vlenb=VLEN字节数, *8=位宽
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
- `__riscv_vlenb()` 运行时读 `vlenb` CSR（VLEN 字节数），×8 得位宽，据此选最优内核。
- **这就是 VLEN 特化的调度骨架**。给 q8_0/q4_K 做优化，就是仿这套 switch + 补 `_vlNNN` 版本。

#### 4.4.2 `_vl256` 版逐行（:1257，intrinsics，最易读）

先解压 6-bit 子块缩放（标量位运算）：

```c
memcpy(aux, x[i].scales, 12);
utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
utmp[1] =  (aux[1] & kmask2)       | (((aux[2] >> 2) & kmask1) << 4);
utmp[0] =  (aux[0] & kmask2)       | (((aux[2] >> 0) & kmask1) << 4);
int8_t * scale = (int8_t *)utmp;
for (int j = 0; j < 16; ++j) scale[j] -= 32;   // 6-bit 缩放有偏移, 减 32 还原符号
```
（这是把 12 字节里压缩的 16 个 6-bit 缩放拆成 16 个 int8。细节不必抠，知道"在解压子块缩放"即可。）

主循环：256 权重分 2 段、每段 128：

```c
size_t vl = 32;
vuint8m1_t vqh = __riscv_vle8_v_u8m1(qh, vl);     // 高位 mask(每权重第3位)
for (int j = 0; j < QK_K; j += 128) {
    vuint8m1_t q3_x = __riscv_vle8_v_u8m1(q3, vl); // 32 字节低位

    // 一个字节塞 4 个权重的低 2 位, 移位+掩码拆 4 组：
    vint8m1_t q3_0 = ..vand(q3_x, 0x03)..;             // bit 0-1
    vint8m1_t q3_1 = ..vand(vsrl(q3_x,2), 0x03)..;     // bit 2-3
    vint8m1_t q3_2 = ..vand(vsrl(q3_x,4), 0x03)..;     // bit 4-5
    vint8m1_t q3_3 = ..vand(vsrl(q3_x,6), 0x03)..;     // bit 6-7

    // 补第 3 位（掩码条件减法）：3-bit 值 = 低2位 - (高位为0 ? 4 : 0)
    vuint8m1_t qh_m0 = __riscv_vand_vx_u8m1(vqh, m, vl);
    vbool8_t vmask_0 = __riscv_vmseq_vx_u8m1_b8(qh_m0, 0, vl); // ==0 -> 掩码真
    vint8m1_t q3_m0  = __riscv_vsub_vx_i8m1_mu(vmask_0, q3_0, q3_0, 0x4, vl); // 真处 -=4
    m <<= 1;
    // q3_1/q3_2/q3_3 同理, m 每次左移取下一位

    // 乘激活 -> i16
    vint16m2_t a0 = __riscv_vwmul_vv_i16m2(q3_m0, __riscv_vle8_v_i8m1(q8,    vl), vl);
    vint16m2_t a1 = __riscv_vwmul_vv_i16m2(q3_m1, __riscv_vle8_v_i8m1(q8+32, vl), vl);
    // a2, a3 同理(q8+64, q8+96)

    vl = 16;
    // a0 含 32 个元素 = 两个 16 的子块, vget 取各半, 分别乘对应子块缩放
    vint32m2_t aux0_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0,0), scale[0], vl);
    vint32m2_t aux0_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0,1), scale[1], vl);
    // ... 8 个子块 aux1_0 .. aux3_1

    // 分段累积归约：前一个的结果当下一个的初值, 串成链, 省去反复取标量
    vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(vadd(aux0_0,aux0_1), vzero, vl);
    vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(vadd(aux1_0,aux1_1), isum0, vl);
    vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(vadd(aux2_0,aux2_1), isum1, vl);
    vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(vadd(aux3_0,aux3_1), isum2, vl);
    sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);
    q3 += 32; q8 += 128; scale += 8;
}
const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;  // super-block 总缩放
sumf += d * sum_t;
```

**新增 intrinsic / 思想**：
- `vmseq`(出掩码)、`_mu` 掩码运算、`vbool8_t`；
- `vget_v_i16m2_i16m1`：从 m2 组取出第 0/1 个 m1 子寄存器（拆子块）；
- `vredsum` + **初值串链**：多次归约接力传中间和，避免反复 `vmv_x_s`；
- **两层缩放**：子块 `scale[]` 在向量里乘，super-block `d` 标量收尾乘。

#### 4.4.3 `_vl128` 版为何用内联汇编（:1107）

VLEN=128 时 e8m1 只能装 16 个 int8，装不下整块，编译器自动向量化效果差，于是**手写内联汇编**精细排布寄存器、一次算 128 个权重。它分两段汇编：

```c
// 第 1 段：解压 16 个 6-bit 子块缩放（intrinsics 版是那段 utmp 位运算）
__asm__ __volatile__(
    "vsetivli zero, 12, e8, m1\n\t"     // 一次处理 12 字节 scales
    "vle8.v v0, (%[s6b])\n\t"           // 载入压缩缩放
    "vslidedown.vi v1, v0, 1\n\t"       // 向量元素整体下移(取高位部分)
    "vslideup.vi v0, v2, 1\n\t"         // 上移拼接
    "vid.v v9\n\t"                       // 生成索引 0,1,2,3
    "vsll.vi v9, v9, 1\n\t"             // {0,2,4,6} 移位量
    "vsrl.vv v4, v1, v9\n\t"            // 按不同位移右移(并行解 4 个 6-bit)
    "vand.vx ...\n vor.vv ...\n"        // 拼回 8-bit 缩放
    "vsub.vx v0, v7, %[c]\n\t"          // 减 32
    "vse8.v v0, (%[scale])" : ...);     // 存进 scale[]
// 第 2 段：主计算，每 128 权重一轮(逻辑同 vl256, 但手工调度)
//   vsrl/vand 解 2-bit; vmseq+vadd(-4,v0.t) 补高位; vwmul 乘激活;
//   8 次 vwredsum 出 8 个子块和(v8..v15); vmul/vmacc 乘缩放; vmv.x.s 取出累加
```

**新指令**（仅此处用，认识即可）：
- `vslidedown/up`、`vslide1up`：向量元素整体平移（像移位寄存器），用来重排压缩的缩放字节。
- `vid.v`：生成 `0,1,2,3,...` 索引向量。
- `vmv1r.v`：整寄存器复制。
- `v0.t`：汇编里的掩码后缀（`.t`=用 v0 当掩码），等价 intrinsic 的 `_mu`。

> 日常学习读 `_vl256`（intrinsics）足够；`_vl128` 知道"小 VLEN 用汇编硬抠、逻辑相同"即可。
>
> 推广：`q4_K/q5_K/q6_K` 都是"super-block + 子块缩放 + 解包 + 加宽乘 + 分段归约"，只是位宽/打包不同。看懂 q3_K `_vl256` 它们就都能读。**注意 q4_K/q5_K/q6_K 没做 VLEN 特化**（仅单一版本）——优化好目标。

---

### 4.5 【重排乘法】`ggml_gemv_q8_0_16x1_q8_0` (repack.cpp:447)

`vec_dot` 算"一行权重 × 一列激活"。repack 把**16 列权重交织存放**，一次算 16 个输出 —— "填满向量"的正解。

#### 4.5.1 交织数据布局 `block_q8_0x16`（repack.h:23）

```c
template <int K, int N> struct block { ggml_half d[N]; int8_t qs[QK8_0*N*K/8]; };
using block_q8_0x16 = block<8, 16>;   // K=8(位宽), N=16(交织列数)
//  => { ggml_half d[16]; int8_t qs[512]; }   16 个缩放 + 16 列×32 = 512 个 int8
```

**交织 = 把 16 列同一位置的元素摆在一起**：

```
普通布局(每列一个 block_q8_0)：
  列0: [d0][q0_0 q0_1 ... q0_31]
  列1: [d1][q1_0 q1_1 ... q1_31]
  ...

交织布局(block_q8_0x16.qs)：把 16 列的"第 i 个元素"连排
  qs: [q0_0 q1_0 q2_0 ... q15_0 | q0_1 q1_1 ... q15_1 | ...]
       └─── 16 列的第0个 ───┘ └─── 16 列的第1个 ───┘
  d:  [d0 d1 d2 ... d15]
```

这样一条 `vle8` 就能取到"16 列的第 i 个权重"，喂给向量。

#### 4.5.2 gemv 逐行（decode 用，1 列激活）

```c
const int ncols_interleaved = 16;
const block_q8_0 * a_ptr = (const block_q8_0 *) vy;       // 激活(单列)
for (int x = 0; x < nc / ncols_interleaved; x++) {       // 每轮出 16 列结果
    const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb); // 16 列交织权重

    vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16); // 16 个 float 累加器(每列一个)
    for (int l = 0; l < nb; l++) {                        // 遍历块
        vint32m2_t sumi = __riscv_vmv_v_x_i32m2(0, 16);   // 16 个 int 累加器
        for (int i = 0; i < QK8_0; i++) {                 // 块内 32 个位置
            vint8mf2_t b_0 = __riscv_vle8_v_i8mf2(&b_ptr[l].qs[i*16], 16); // 16 列的第 i 个权重
            sumi = __riscv_vwadd_wv_i32m2(sumi,
                       __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i], 16), 16); // ×同一激活标量, 累加
        }
        vfloat16m1_t b_d = __riscv_vle16_v_f16m1((const _Float16 *)b_ptr[l].d, 16); // 16 列缩放
        vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(const _Float16 *)&a_ptr[l].d, 16);
        sumf = __riscv_vfmacc_vv_f32m2(sumf, __riscv_vfcvt_f_x_v_f32m2(sumi, 16), d_0, 16);
    }
    __riscv_vse32_v_f32m2(s + x*16, sumf, 16);            // 存 16 个结果
}
```

**与 vec_dot 的本质差异**：
- 向量的 lane 维度从"块内元素"变成"**16 个输出列**"。`vwmul_vx(b_0, a_ptr[l].qs[i], 16)` = 16 列权重 × **同一个**激活标量。
- **没有块内归约**！求和沿外层 `i`/`l` 循环用 `vwadd_wv` 累加完成。归约延迟问题消失。
- `vint8mf2_t`：分数 LMUL（半寄存器），因为只要 16 lane。

**新增**：`vwadd_wv`(加宽加,宽+窄)、`vwmul_vx`、`vfwmul_vf`(加宽浮点乘)、`vfcvt_f_x_v`(int->float)、`vfmacc_vv`、分数 LMUL `mf2`。

#### 4.5.3 gemm 版（prefill 用，多列激活，repack.cpp:1336）

`gemm` 是 gemv 的"多激活列"推广：外面多套一层"激活列"循环，一次算 `16 列权重 × M 列激活` 的小块，进一步摊薄权重加载。结构上把 gemv 内层复制成处理 4 个激活列（你 perf 见过的 `..._mul_mat_one_chunk` 在 `nrows>3` 时走它）。

#### 4.5.4 权重何时被重排？repack buffer 机制（repack.cpp:4726）

交织布局不是凭空来的 —— 加载权重时，若该张量走 repack 路径，ggml 用一种**特殊 buffer 类型**，在 `set_tensor` 时调 `repack()` 把普通 q8_0 权重转成 `block_q8_0x16`：

```c
// ggml_backend_cpu_repack_buffer_set_tensor (repack.cpp:4733)
static void ..._set_tensor(..., struct ggml_tensor * tensor, const void * data, ...) {
    auto * tensor_traits = ...;
    tensor_traits->repack(tensor, data, size);  // 普通布局 -> 交织布局, 一次性预处理
}
```
是否走 repack、用哪种交织，由 `get_tensor_traits`(:4710) 按 VLEN 决定（你上轮见过：q8_0 仅 VLEN=256 启用）。

> 这就是 repack 高效的本质：用"输出列"当向量维度，天然填满、无归约。代价是权重要**预先重排**（一次性，加载时做）。

---

# 第三部分 · 其余路径与运行时

## 第 5 章 · 通用 F32 路径（非量化算子）

RMSNorm、softmax、`vec_dot_f32` 等不涉及量化，走 `simd-mappings.h` 的宏（`vec.cpp`/`ops.cpp` 调用）：

```c
// simd-mappings.h:1265, #elif defined(__riscv_v_intrinsic)
#define GGML_F32_EPR  4                      // 每向量 4 个 float（写死 128bit！）
#define GGML_F32x4    vfloat32m1_t
#define GGML_F32x4_FMA(a,b,c) __riscv_vfmacc_vv_f32m1(a,b,c, GGML_F32_EPR)
#define GGML_F32x4_LOAD(x)    __riscv_vle32_v_f32m1(x, GGML_F32_EPR)
#define GGML_F32x4_ADD(a,b)   __riscv_vfadd_vv_f32m1(a,b, GGML_F32_EPR)
#define GGML_F32x4_REDUCE(...) ...
```

- 上层算子用 `GGML_F32_VEC_*` 抽象宏，换架构只换映射。
- **同样写死 4-wide**：VLEN=1024 下 f32m1 能放 32 个 float，却只用 4 -> 又是 12.5%。量化内核之外的另一类全局优化点。
- 注意：**不是所有算子都用了这套宏**。许多逐元素算子是纯标量循环（见上层篇 RMSNorm），完全没向量化 —— 那是更低垂的果子。

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

# 第四部分 · 性能与实战

## 第 7 章 · 性能模型（为什么慢，慢在哪）

写优化前先有"成本直觉"。RVV 内核三大成本来源：

### 7.1 向量利用率（吞吐）
```
利用率 = vl / VLMAX = vl / (LMUL*VLEN/SEW)
```
固定 `vl=32` 的 q8_0 在 VLEN=1024、e8m2(VLMAX=256) 下只有 12.5%。**等价于把 1024 位的机器当 128 位用**。

### 7.2 归约延迟（最贵）
`vwredsum`/`vredsum` 是**跨 lane 求和**，硬件内部要做树形规约或串行累加，延迟通常是普通向量指令的数倍到数十倍，且**难以和其他指令重叠**（强依赖）。q8_0 每 32 元素就归约一次 -> 28 块归约 28 次，是主要瓶颈。
> 优化原则：**尽量少归约**。要么一次处理多块再归约（摊薄），要么像 repack 那样用"列维度"消掉块内归约。

### 7.3 vtype 切换开销
改变 `vsetvli` 的 SEW/LMUL 可能让硬件重配置流水线、产生气泡。只改 vl（同 vtype）很便宜，改 SEW/LMUL 较贵。q8_0 内层一圈切 `e8->e16->e32` 三次（你 gdb 见过）。
> 优化原则：把同 vtype 的操作聚在一起，减少切换。

### 7.4 访存
单位步长 `vle`/`vse` 最快；跨步 `vlse`、索引 `vluxei`(gather) 慢得多（码本量化 iq* 用 gather，所以慢）。

### 7.5 各代表内核成本速记（每块/每段）

| 内核 | 加载 | 乘 | 归约 | vtype 切换 | 利用率@1024 |
|---|---|---|---|---|---|
| q8_0 vec_dot | 2×vle8 | 1×vwmul | 1×vwredsum | 3 | 12.5% |
| q4_0 vec_dot | 3×vle8+解包 | vwmul+vwmacc | 1×vwredsum | 中 | 12.5% |
| q3_K vl256 | 多 | 多 vwmul | 4×vredsum/段 | 多 | ~25% |
| gemv 16x1 | 交织 vle | vwmul_vx 链 | **0(块内)** | 少 | 高(填满16列) |

---

## 第 8 章 · 写一个 RVV 内核（坑 + 清单 + 练习）

### 8.1 常见坑

1. **LMUL 寄存器对齐**：m2 用偶数号、m4 用 4 的倍数、m8 用 v0/v8/v16/v24。intrinsic 里编译器替你分配，写汇编时要自己守。
2. **v0 是掩码寄存器**：用掩码时别拿 v0 存数据，会被覆盖。
3. **加宽 = LMUL 翻倍 = 寄存器压力**：`vwmul` 把 m4 输入变 m8 输出，占满 8 个寄存器，别再同时占用冲突的号。
4. **vl 是 vsetvl 返回值**：strip-mining 必须用返回的 vl（可能 < 请求值）；固定 `vl=32` 且 VLMAX≥32 时才恒等于 32。
5. **尾部处理**：n 不是块整数倍时要补尾循环。ggml 内核普遍 `assert(n % qk == 0)` 来回避，因为张量维度是块对齐的。
6. **窄化分两步**：32->8 不能一条指令，要 ->16->8。
7. **reinterpret 免费，cvt 收费**：同位宽换符号用 `vreinterpret`(0 开销)，跨位宽/浮点整数才用 `vcvt`。

### 8.2 改内核后的验证清单

```
[ ] 数值正确：和 scalar 版(_generic)对拍, 或 llama-cli 输出文本合理
[ ] objdump 确认生成了预期指令(vsetvli 的 vl/SEW/LMUL 对不对)
[ ] gdb 看 vtype/vl 寄存器, 确认填满了向量
[ ] llama-bench 对比 tg/pp, 确实变快
[ ] 不破坏其他 VLEN(用 switch 分发, 别改公共路径)
```

### 8.3 练习：给 q8_0 写一个 vlen 友好版（草图）

思路：一次吃 `G = VLMAX/32` 个块（VLEN=1024 时 G=8），摊薄加载/乘法/vtype 切换，再向量化缩放。

```c
// 伪代码草图(教学用, 非可编译)
size_t vlmax = __riscv_vsetvlmax_e8m2();   // 1024 时 = 256
int G = vlmax / 32;                         // 一次 8 块
for (ib = 0; ib + G <= nb; ib += G) {
    // 一次性加载 G 块的权重和激活(连续内存, 单位步长, 高效)
    vint8m2_t wx = __riscv_vle8_v_i8m2(&x[ib].qs..., G*32);   // 但 qs 不连续(被 d 隔开)!
    // -> 现实中需按块加载或预处理布局; 这是难点, 也是 repack 存在的理由
    // 对每块做一次 vwredsum(仍 G 次归约), 但加载/乘法已摊薄
    // 把 G 个 sumi 凑成向量, 一次乘 G 个 scale 的向量, 减少标量 fmadd
}
// 处理剩余不足 G 块的尾部
```

**关键洞察**：q8_0 块间被 `d` 字段隔开（`{d, qs[32]}` 交替），`qs` 在内存里**不连续**，所以"一次 vle 吃多块"并不直接成立 —— 这正是 ggml 设计 **repack 交织布局**（把 d 和 qs 分开存）的根本原因。所以：
- **轻量优化**：在现有布局上摊薄 vtype 切换、用更大 LMUL 的归约。
- **彻底优化**：给 q8_0 也接上 repack 路径（参考 4.5），用"列维度"消归约 —— 收益最大，是 `repack.cpp:4713` 那几个 `// TODO` 的意义所在。

---

## 附录 A · RVV intrinsic 速查（本文出现过的）

| intrinsic | 作用 | 首次出现 |
|---|---|---|
| `vsetvl(i)` `vsetvlmax` | 配置 SEW/LMUL，返回 vl / 取 VLMAX | 1.4 / 8.3 |
| `vle{N}/vse{N}` | 单位步长 载入/存回 | 4.1 |
| `vfabs` `vfmul_vf` | 浮点绝对值/乘标量 | 4.1 |
| `vfredmax_vs` `vfmv_f_s` `vfmv_v_f` | 浮点归约最大/取标量/广播 | 4.1 |
| `vfncvt` `vncvt` | 浮点->整数窄化 / 整数窄化 | 4.1 |
| `vwmul_vv` | 加宽乘 i8×i8->i16 | 4.2 |
| `vwredsum_vs` | 加宽归约求和 | 4.2 |
| `vmv_x_s` `vmv_v_x` | 取标量整数/广播整数 | 4.2 |
| `vand_vx` `vsrl_vx` `vsll_vi` `vsra` | 位与/逻辑右移/左移/算术右移 | 4.3 |
| `vreinterpret` | 位不变换类型（0 开销）| 4.3 |
| `vsub_vx` | 减标量 | 4.3 |
| `vwmacc_vv` | 加宽乘累加 | 4.3 |
| `vmseq_vx_..._b8` | 比较出掩码 vbool8 | 4.4 |
| `vsub_vx_..._mu` | 掩码版减法 | 4.4 |
| `vget_v_i16m2_i16m1` | 取子寄存器 | 4.4 |
| `vredsum_vs` | 归约求和（非加宽）+ 初值串链 | 4.4 |
| `vslidedown/up` `vslide1up` `vid` `vmv1r` | 元素滑移/索引/整组复制（vl128 汇编）| 4.4.3 |
| `vwadd_wv` | 加宽加（宽+窄）| 4.5 |
| `vwmul_vx` `vfwmul_vf` | 加宽乘标量 / 加宽浮点乘标量 | 4.5 |
| `vfcvt_f_x_v` | int->float 转换 | 4.5 |
| `vfmacc_vv` `vfadd_vv` | 浮点乘累加/加 | 4.5 / 5 |
| `vrgather` `vluxei` | gather/查表（码本量化 iq*）| 7.4 |

## 附录 B · 学习顺序与过关标准

| 阶段 | 读 | 过关标准 |
|---|---|---|
| 1 | 第 1-2 章 | 能手工解码 `vtype=0xc1`，会拆任意 intrinsic 名 |
| 2 | 4.1 quantize_q8_0 + 算例 | 能说清"一行 float -> {d, 32×int8}"并手算一个值 |
| 3 | 4.2 vec_dot_q8_0 | 能默写"载入->vwmul->vwredsum->缩放"四步及其数学 |
| 4 | 4.3 vec_dot_q4_0 | 能解释 nibble 解包 + 零点 8 + vwmacc 省归约 |
| 5 | 4.4 q3_K `_vl256` + 分发器 | 能讲清两层缩放、掩码补高位、分段归约、VLEN 分发 |
| 6 | 4.5 gemv_16x1 + 交织布局 | 能说出 repack 为何无块内归约、代价是预重排 |
| 7 | 第 7 章 | 能定位一个内核的三大成本(利用率/归约/vtype) |
| 8（毕业）| 第 8 章 + 动手 | 给一个未特化内核补 `_vl1024`，llama-bench 验证 |

## 附录 C · 外部参考

- RVV intrinsics 文档：`riscv-non-isa/rvv-intrinsic-doc`（GitHub，查任意 intrinsic 签名）
- RVV 1.0 规范：`riscv/riscv-v-spec`（查指令精确语义）
- 活样本：你的 gdb + `gdb.log` 就是最好教材，每个函数都可下断点 `disassemble` 对照

---

*配套调试：每个函数都可在 QEMU + gdb 下断点实跑（见 `readme.md` §5）。建议边读边 `disassemble`/`info registers vtype vl` 对照，源码与汇编互相印证。上层调用关系见 `ggml-core-tutorial.md`。*
