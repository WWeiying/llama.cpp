#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// Marker types keep the QBS layouts distinct from the standard RVV repacks.
struct ggml_riscv_qbs_q4_k_tag {};
struct ggml_riscv_qbs_q5_k_tag {};
struct ggml_riscv_qbs_q6_k_tag {};
struct ggml_riscv_qbs_q3_k_tag {};
struct ggml_riscv_qbs_q8_0_tag {};
struct ggml_riscv_qbs_q4_0_tag {};

bool ggml_riscv_qbs_supports_tensor(ggml_type type, int64_t k_elements,
                                    int64_t output_rows);

int ggml_riscv_qbs_repack_r4(void * dst, size_t dst_size, const void * src,
                             size_t src_size, int64_t output_rows,
                             int64_t k_blocks, size_t block_bytes);

void ggml_riscv_qbs_gemv(ggml_type weight_type, int n, float * dst,
                          size_t dst_stride, const void * weights,
                          const void * activations, int input_rows,
                          int output_rows);

void ggml_riscv_qbs_gemm(ggml_type weight_type, int n, float * dst,
                          size_t dst_stride, const void * weights,
                          const void * activations, int input_rows,
                          int output_rows);
