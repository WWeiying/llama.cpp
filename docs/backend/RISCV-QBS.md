# RISC-V QBS CPU path

The RISC-V QBS path is an experimental `GGML_OP_MUL_MAT` implementation for
processors that expose the Quantized Block Stream (QBS) instruction contract.
It keeps the normal GGML graph and logical tensor types, but replaces eligible
quantized matrix multiplication with a persistent weight repack and QBS
GEMV/GEMM commands. The ordinary RVV implementation remains the fallback.

## Supported pairs

| Weight type | Runtime activation | Block elements |
| --- | --- | ---: |
| `Q3_K` | `Q8_K` | 256 |
| `Q4_K` | `Q8_K` | 256 |
| `Q5_K` | `Q8_K` | 256 |
| `Q6_K` | `Q8_K` | 256 |
| `Q8_0` | `Q8_0` | 32 |
| `Q4_0` | `Q8_0` | 32 |

The first implementation handles two-dimensional `GGML_OP_MUL_MAT` weight
tensors. Other weight formats, non-block-aligned K dimensions, unsupported
capabilities, and `GGML_OP_MUL_MAT_ID` use the existing CPU/RVV path.

## Build and runtime selection

QBS is disabled by default and requires both RVV and CPU repacking:

```bash
cmake -S . -B build-riscv-qbs \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/riscv64-toolchain.cmake \
  -DGGML_RVV=ON \
  -DGGML_CPU_REPACK=ON \
  -DGGML_RISCV_QBS=ON \
  -DGGML_RISCV_QBS_INCLUDE_DIR=/path/containing/qbs_abi.h
```

On QBS hardware, set `GGML_RISCV_QBS=1` **before model loading**. Capability
detection is cached and CPU_REPACK chooses the persistent weight layout while
model tensors are allocated; enabling QBS after that point cannot retrofit an
already loaded model. The explicit opt-in prevents an ordinary RVV processor
from executing the non-standard capability-query instruction. After opt-in,
selection is automatic and still requires:

- a supported logical weight type;
- block-aligned K within the reported maximum block count;
- both R4 weight and row-major/M4-interleaved activation layouts;
- compatible QBS architecture, descriptor, and numerical-contract versions.

If any check fails, QBS creates no transformed tensor and GGML selects its
normal CPU/RVV implementation.

## Data and execution path

At model load, the CPU_REPACK buffer converts each selected weight tensor from
GGUF row-major blocks to R4 block-major storage. Full groups contain four
output rows. A final group of one to three rows is zero-padded to the four-row
R4 ABI shape; the CPU_REPACK buffer reports the padded allocation size while
the logical GGML tensor shape remains unchanged. The transformed buffer is
retained with the model and reused by every inference step; repacking is not
part of per-token execution.

For each `MUL_MAT`, GGML dynamically quantizes F32 activations to `Q8_K` or
`Q8_0`. Four complete input rows use the M4-interleaved activation layout and
one QBS GEMM call. Remaining input rows use row-major activation blocks and M1
GEMV calls. The QBS wrapper further tiles N to the hardware-reported command
width (currently at most 32 output rows); K is represented as a descriptor
block count. Thus the same path covers decode (`M=1`), prefill (`M>=4`), N
tails, and M tails without changing the GGML graph.

## Validation modes

`GGML_RISCV_QBS_EMULATE=1` is a correctness-only mode for an ordinary RVV
QEMU. It exercises the same tensor selection, R4 repack, activation packing,
M/N/K scheduling, and fallback logic, but evaluates each QBS command with the
GGML reference dot products instead of executing the custom opcode. It must
not be used for performance claims.

The directed cross-platform regression covers Q3_K, Q4_K, Q5_K, Q6_K, Q8_0,
and Q4_0 with both M1 GEMV and M8 GEMM. Its `N=35` shape forces a 32-row command
plus a three-row padded R4 tail, while M8 forces two independent M4 activation
groups. A separate CPU_REPACK test allocates and initializes a logical
`Q4_K[512 x 35]` tensor through the real transformed-buffer interface and
checks its padded allocation and bytes.

Optional diagnostics are:

- `GGML_RISCV_REPACK_TRACE=1`: print each selected transformed tensor;
- `GGML_RISCV_QBS_TRACE=1`: print per-format GEMV/GEMM call and work totals at
  process exit.

The CVA6 QEMU integration in
`/home/wangwy/llama/platforms/cva6-qemu/run-qbs-token-check.sh` runs the same
Qwen2.5 model through normal RVV and emulated QBS, compares greedy output byte
for byte, and requires both M4 prefill and M1 decode activity. Cycle and memory
performance must instead be measured with equivalent real-model operators on
QBS RTL.

The completed Qwen2.5-1.5B-Instruct Q4_K_M check used a ten-token prompt and
generated two tokens. Both runs exited successfully and produced byte-identical
text. The QBS trace recorded 8,032 GEMV and 2,656 GEMM calls for Q4_K, plus
1,360 GEMV and 432 GEMM calls for Q6_K. This confirms that real model loading,
persistent repacking, prefill, decode, and per-format dispatch all exercised
the QBS integration; it remains a functional result rather than a QEMU
performance measurement.
