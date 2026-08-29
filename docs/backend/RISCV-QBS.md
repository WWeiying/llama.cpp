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
| `Q2_K` | `Q8_K` | 256 |
| `Q8_0` | `Q8_0` | 32 |
| `Q4_0` | `Q8_0` | 32 |
| `Q5_0` | `Q8_0` | 32 |
| `IQ4_NL` | `Q8_0` | 32 |

The implementation handles two-dimensional `GGML_OP_MUL_MAT` weights and
three-dimensional `GGML_OP_MUL_MAT_ID` weights whose number of rows per expert
is divisible by four. The latter restriction keeps every R4 group within one
expert and preserves the original expert stride. Other weight formats,
non-block-aligned K dimensions, unsupported capabilities, and incompatible
tensor shapes use the existing CPU/RVV path.

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
- a positive K dimension aligned to the selected profile's block size;
- both R4 weight and row-major/M4-interleaved activation layouts;
- compatible QBS architecture, descriptor, and numerical-contract versions.

The current profile contract associates one activation scale with each weight
block. Consequently, `Q1_0` and `Q2_0` are not aliases for the 256-element
`Q2_K` profile, and `Q4_1`/`Q5_1` cannot use the `Q8_0` activation profile
without adding their asymmetric `Q8_1` correction semantics.

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
one QBS GEMM call. A two- or three-row tail remains row-major and is issued as
one M2 or M3 command; only a one-row tail uses M1 GEMV. The QBS wrapper further
tiles N to the hardware-reported command width (currently at most 32 output
rows).

For `MUL_MAT_ID`, GGML groups selected token rows by expert and invokes the
same QBS M1 kernel on the expert-local R4 region. This is a correctness-first
MoE path: it preserves arbitrary expert IDs and output-slot placement but does
not yet batch rows selecting the same expert into M2-M4 commands.

The descriptor represents K with at most the hardware-reported block count
(currently 256). A larger block-aligned K is a functional software fallback:
the wrapper issues consecutive K segments and accumulates their FP32 partial
results in deterministic segment order. Because the existing R4 layout is
contiguous for a K segment only within one four-row weight group, this path
uses N tiles of at most four rows. It preserves functional coverage without a
second persistent weight layout, but repeats command and activation overhead
and is not the optimized long-K path. It can differ in low FP32 bits from a
hypothetical single command because partial sums are rounded at segment
boundaries. Thus the same GGML integration covers decode (`M=1`), prefill
(`M>=4`), N/M tails, and long K without changing the graph, while ordinary
block-count-limited commands retain their original scheduling.

## Validation modes

`GGML_RISCV_QBS_EMULATE=1` is a correctness-only mode for an ordinary RVV
QEMU. It exercises the same tensor selection, R4 repack, activation packing,
M/N/K scheduling, and fallback logic, but evaluates each QBS command by
dequantizing its blocks and using an explicitly scalar dot product instead of
executing the custom opcode. This path is intentionally independent of the
RVV dot implementation under test and must not be used for performance claims.

The directed cross-platform regression covers Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
Q8_0, Q4_0, Q5_0, and IQ4_NL with M1, M2, M3, and M4 commands. Its `N=35`
shape forces a 32-row command plus a three-row padded R4 tail. A separate
`Q4_0` case uses `K=8960`
(280 blocks) to exercise software K segmentation with M1, M2, M3, and M4. A
CPU_REPACK test allocates and initializes a logical
`Q4_K[512 x 35]` tensor through the real transformed-buffer interface and
checks its padded allocation and bytes. A three-expert `Q2_K` graph test also
executes `GGML_OP_MUL_MAT_ID` and compares every routed result with an
independent dequantized scalar oracle. A companion negative test uses six
rows per expert and verifies that CPU_REPACK retains the ordinary GGML layout,
because an R4 group cannot be padded without crossing an expert boundary.

Optional diagnostics are:

- `GGML_RISCV_QBS_FORMATS=Q2_K,Q5_0`: restrict QBS tensor selection to the
  listed weight formats. Unlisted supported formats retain the standard RVV
  path. If the variable is unset or contains `ALL`, all supported formats are
  eligible;
- `GGML_RISCV_REPACK_TRACE=1`: print each selected transformed tensor;
- `GGML_RISCV_QBS_TRACE=1`: print exact per-format static coverage and dynamic
  execution totals at process exit. Static coverage deduplicates tensors by
  tensor identity and reports candidate/selected tensors, logical weight
  elements, segmentation, and fallback reasons, including an explicit
  `fallback_format_filter` count. Here a candidate is a weight
  tensor whose logical type is one of the nine supported pairs above and which
  reaches CPU_REPACK selection; it is not a count of every model tensor or
  every GGML operator. Dynamic execution reports
  logical GEMV/GEMM work, actual M1/M2/M3/M4 command counts, native versus
  emulated commands, command-level dot elements, and K-segmented commands;
- `GGML_RISCV_QBS_TRACE_TENSORS=1`: additionally print the selection decision
  and fallback reason for every unique candidate tensor.

The static and dynamic counters answer different questions. Tensor coverage
describes which persistent weights were transformed; command counters describe
what inference actually executed. `dot_elements` counts logical GGML work,
whereas `command_dot_elements` is reconstructed from every issued command.
They must match for a complete run, including K-segmented execution.

The CVA6 QEMU integration in
`/home/wangwy/llama/platforms/cva6-qemu/run-qbs-token-check.sh` runs the same
Qwen2.5 model through normal RVV and emulated QBS, compares greedy output byte
for byte, and requires both M4 prefill and M1 decode activity. Cycle and memory
performance must instead be measured with equivalent real-model operators on
QBS RTL.

The `Xaraqbs` QEMU model additionally executes the native `qbinfo` and
`qbexec` opcodes. The directed native regression compares all nine profile
pairs against an independent dequantized scalar oracle. The full-model native
check runs standard RVV and QBS in one Linux boot, requires byte-identical
generated output, and confirms Q4_K/Q6_K GEMV and GEMM activity. This remains
an architectural functional test; QEMU timing is not a hardware performance
measurement.

The completed Qwen2.5-1.5B-Instruct Q4_K_M check used a ten-token prompt and
generated two tokens. Both runs exited successfully and produced byte-identical
text. The QBS trace recorded 8,032 GEMV and 2,656 GEMM calls for Q4_K, plus
1,360 GEMV and 432 GEMM calls for Q6_K. This confirms that real model loading,
persistent repacking, prefill, decode, and per-format dispatch all exercised
the QBS integration; it remains a functional result rather than a QEMU
performance measurement.
