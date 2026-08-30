# RISC-V AKV attention integration

The experimental RISC-V AKV path accelerates one strict Decode
`GGML_OP_FLASH_ATTN_EXT` profile while preserving the existing CPU/RVV
implementation as the fallback. It shares the same AKV runtime and native
kernel with the standalone real-model benchmark.

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
one CPU backend. Runtime capability and shape checks decide which operator, if
any, uses each mechanism.

## Runtime controls

- `GGML_RISCV_AKV=1` enables the native capability query and native AKV
  execution path on a platform already known to implement the AKV query
  instruction.
- `GGML_RISCV_AKV_EMULATE=1` selects software emulation for framework and model
  correctness checks on an ordinary RVV QEMU. It does not model hardware
  cycles.
- `GGML_RISCV_AKV_TRACE=1` reports candidate operations, executed operations,
  GQA groups, and exact fallback classes at process exit.

With neither enable variable set, every Flash-Attention operation uses the
unchanged standard CPU/RVV implementation.

The emulation callback follows the same grouped online-Softmax dataflow and
F16 accumulator contract. Its query-key dot, F16 accumulator scaling, and F16
value accumulation deliberately reuse the same GGML RVV helpers as the
standard path. It is still not an instruction-order model of the native RVV
kernel because the native kernel uses its own vector exponential
approximation. Model checks therefore validate graph dispatch, runtime
planning, output quality, and fallback. Native numerical error and cycle
counts require the RTL operator benchmark with the shared assembly kernel.

The repository-side model validator compares complete full-vocabulary logits
and requires equal generated output, equal top-1 logits at every recorded
decode step, and a maximum absolute logits difference no greater than 0.001 by
default.

## Strict profile

The current hardware profile is selected only when all of these conditions
hold:

- Decode query length and batch are both one;
- query is F32, K/V/mask are F16, and output is F32;
- head dimension is 128 and six query heads share each K/V head;
- execution is not reference-only, and GGML provides a valid worker team;
- attention sinks, ALiBi/max bias, and logit softcap are absent;
- Q, K, V, mask, and output strides satisfy the native descriptor contract;
- the active mask is one finite prefix followed only by exact F16 `-inf`;
- native mode reports a valid enabled device with the required profile.

Unsupported operations return to the original Flash-Attention loop before any
AKV state is created. Once a strict plan has been accepted, a plan or execution
failure is fatal rather than silently re-executing the operation and risking
duplicate architectural effects.

For each K/V head, the adapter converts its six F32 query rows to F16, creates
an immutable AKV plan, and invokes the shared runtime. The native kernel keeps
the query group and online-softmax state resident while streaming each K/V
token once. Output is written directly in the GGML destination layout. QBS
continues to handle eligible quantized matrix operators; AKV handles only the
strict Decode attention operation above.

GGML invokes a compute node on every worker and places a barrier after that
node. AKV v1 has one hidden context, so every worker makes the same dispatch
decision, worker zero executes all AKV groups, and the remaining workers join
GGML's normal node barrier without entering the standard implementation. This
preserves multi-worker graph execution without allowing concurrent commands to
alias the single context. Coverage counters are consequently counted once per
operator by worker zero, not once per worker callback.
