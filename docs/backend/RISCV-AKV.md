# RISC-V AKV attention integration

The experimental RISC-V AKV path accelerates strict Decode and causal Prefill
profiles of `GGML_OP_FLASH_ATTN_EXT`. Unsupported operations return to the
unchanged CPU/RVV implementation. The GGML adapter, standalone real-model
benchmarks, and functional tests use the same AKV ABI and native kernels.

## Build

Configure a RISC-V build with RVV, scalar half precision, vector half
precision, and the AKV runtime source directory:

```bash
cmake -S . -B build-rv64-akv \
  -DGGML_RVV=ON \
  -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZVFH=ON \
  -DGGML_RISCV_AKV=ON \
  -DGGML_RISCV_AKV_RUNTIME_DIR=/path/to/software/akv
```

`GGML_RISCV_AKV` is independent of `GGML_RISCV_QBS`; both can be compiled into
one CPU backend. QBS selects supported quantized matrix operators, AKV selects
supported attention operators, and ordinary RVV remains the common fallback.

## Runtime controls

- `GGML_RISCV_AKV=1` enables the native capability query and AKV execution on a
  platform that implements the AKV query and command instructions.
- `GGML_RISCV_AKV_EMULATE=1` enables the functional AKV model on an ordinary
  RVV system or QEMU. It verifies framework integration and numerical behavior,
  but does not model native instruction timing.
- `GGML_RISCV_AKV_KERNEL=auto|v1|v2` selects a Decode kernel policy. Prefill
  requires `auto` or `v2` because it uses the token-axis v2 contract.
- `GGML_RISCV_AKV_TRACE=1` reports candidate and executed operators, Decode and
  Prefill counts, kernel versions, GQA groups, K/V group-tokens, attention MACs,
  Prefill query tokens and attention pairs, and exact fallback classes.

With neither enable variable set, every Flash-Attention operation uses the
standard CPU/RVV implementation.

## Functional model

Decode emulation follows the grouped online-Softmax dataflow through the
existing callback model. Prefill emulation calls the shared contiguous-Prefill
reference executor. The latter performs the same deterministic F32-to-F16
query narrowing, causal-prefix validation, stable online Softmax, and F32 value
accumulation contract as the native Prefill entry point. Both paths are
functional models rather than instruction-order or cycle models. Native
numerical error and cycle counts therefore require the RTL operator benchmark
that executes the shared assembly kernel.

The repository-side model validator compares complete full-vocabulary logits
and requires equal generated output, equal top-1 logits at every recorded
decode step, and a maximum absolute logits difference no greater than 0.001 by
default.

## Common selection rules

Both profiles require batch one; F32 query and output; F16 K, V, and mask; a
finite positive attention scale; a valid GGML worker team; and no reference-only
execution, attention sinks, ALiBi/max bias, or logit softcap. Tensor addresses,
strides, dimensions, and the advertised AKV capability must satisfy the native
ABI before execution is selected.

Unsupported operations return `false` before issuing an AKV command or writing
output, so GGML can execute the original RVV implementation without observable
AKV state. A failure after this complete preflight is fatal rather than being
silently re-executed after potentially visible architectural effects.

## Decode profile

Decode requires one query token. The token-axis v2 path supports head dimensions
64, 96, and 128 and GQA groups from one through the device-advertised maximum
(currently eight). The legacy v1 path remains restricted to its D128/GQA6
profile. K/V may use descriptor-compatible row and head strides; v2 K/V bases
and strides obey the 32-byte payload alignment contract. The mask must contain
one nonempty finite prefix followed only by exact F16 negative infinity.

For every K/V head, the adapter converts the corresponding F32 query group to
F16, creates an immutable AKV plan, and invokes the shared runtime. The native
kernel keeps the query group and online-Softmax state resident while streaming
each active K/V token once. It writes F32 results directly in GGML destination
layout.

## Prefill profile

Prefill is selected only for at least 15 query tokens, which is the current
measured crossover gate against the tiled RVV implementation. It supports
head dimensions 64, 96, and 128 and GQA groups from one through eight, subject
to the runtime capability. Query length and K/V capacity must each fit the
16-bit token contract. Query/K/V head counts must fit the 32-bit ABI.

The accepted contiguous layouts are:

```text
Query   F32 [query_heads][query_tokens][head_dim]
Key/V   F16 [kv_heads][kv_capacity][head_dim]
Mask    F16 [query_tokens][kv_capacity]
Output  F32 [query_tokens][query_heads][head_dim]
```

K/V bases must meet the 32-byte payload alignment. For past-token count `P`,
mask row `i` must have exactly `P + i + 1` finite entries followed by exact F16
negative infinity through `kv_capacity`; this simultaneously validates dense
causal self-attention, ordinary Prefill, and chunked Prefill. Finite prefix
entries may carry the additive mask values consumed by the attention equation.

Execution is bounded by one fixed workspace for one GQA group and one 64-query
block. The adapter narrows that block's Query rows to F16. The native kernel
then traverses 64-token K/V tiles outside the Query work, reusing each resident
tile across all causally visible queries in the block while retaining F32
online-Softmax maximum, sum, and output state. The optimized D128/GQA6 path can
process two queries together and consume the optional four-column token panel;
other supported shapes use the same contract through generic RVV kernels.
Neither model identity nor context length is encoded in hardware.

## GGML threading and accounting

GGML invokes a compute node on every worker and places a barrier after the node.
AKV currently owns one hidden context, so every worker performs the same
side-effect-free selection, worker zero executes the complete operator, and the
remaining workers join GGML's normal post-node barrier. This prevents concurrent
commands from aliasing the context. Coverage and work counters are consequently
updated once per operator by worker zero, not once per callback.

For Prefill, `prefill_attention_pairs` is the exact sum of visible K/V tokens
over all query heads. `attention_macs` counts both QK and probability-times-V
multiply-accumulate work, and `kv_group_tokens` counts the K/V prefix loaded by
each K/V-head/query-block group. These are semantic work counters; they are not
cycle estimates.
