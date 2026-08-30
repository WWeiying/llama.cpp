#pragma once

struct ggml_compute_params;
struct ggml_tensor;

bool ggml_riscv_akv_flash_attn(const ggml_compute_params * params, ggml_tensor * dst);
