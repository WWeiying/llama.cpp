#pragma once

struct ggml_compute_params;
struct ggml_tensor;

enum class ggml_riscv_quant_kv_status { complete, fallback, failed };

ggml_riscv_quant_kv_status ggml_riscv_quant_kv_flash_attn(const ggml_compute_params * params, ggml_tensor * dst);
