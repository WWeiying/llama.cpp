#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

struct ggml_compute_params;

// Marker types keep the QBS layouts distinct from the standard RVV repacks.
struct ggml_riscv_qbs_q4_k_tag {};

struct ggml_riscv_qbs_q5_k_tag {};

struct ggml_riscv_qbs_q6_k_tag {};

struct ggml_riscv_qbs_q3_k_tag {};

struct ggml_riscv_qbs_q8_0_tag {};

struct ggml_riscv_qbs_q4_0_tag {};

struct ggml_riscv_qbs_q2_k_tag {};

struct ggml_riscv_qbs_q5_0_tag {};

struct ggml_riscv_qbs_iq4_nl_tag {};

bool ggml_riscv_qbs_supports_tensor(ggml_type type, int64_t k_elements, int64_t output_rows);

void ggml_riscv_qbs_trace_tensor_decision(const void * tensor_identity,
                                          const char * tensor_name,
                                          ggml_type    type,
                                          int          n_dims,
                                          int64_t      k_elements,
                                          int64_t      output_rows,
                                          int64_t      matrix_count,
                                          bool         selected);

bool ggml_riscv_qbs_trace_activation_lifetime_enabled();

// Returns true only when this invocation can consume an already resident,
// semantically identical activation and may therefore skip quantization.
bool ggml_riscv_qbs_prepare_activation_context(const ggml_compute_params * params,
                                               const ggml_tensor *         op,
                                               ggml_type                   activation_type,
                                               int64_t                     activation_rows,
                                               int64_t                     k_elements);

void ggml_riscv_qbs_trace_activation_lifetime(const void * op_identity,
                                              const char * op_name,
                                              const void * weight_identity,
                                              const char * weight_name,
                                              ggml_type    weight_type,
                                              const void * source_tensor_identity,
                                              const char * source_name,
                                              const void * source_data,
                                              const void * quantized_data,
                                              size_t       quantized_bytes,
                                              ggml_type    activation_type,
                                              int64_t      activation_rows,
                                              int64_t      k_elements,
                                              int64_t      output_rows,
                                              uint64_t     graph_epoch,
                                              bool         quantize_time_valid,
                                              uint64_t     quantize_time_us);

int ggml_riscv_qbs_repack_r4(ggml_type    type,
                             void *       dst,
                             size_t       dst_size,
                             const void * src,
                             size_t       src_size,
                             int64_t      output_rows,
                             int64_t      k_blocks);

void ggml_riscv_qbs_gemv(ggml_type    weight_type,
                         int          n,
                         float *      dst,
                         size_t       dst_stride,
                         const void * weights,
                         const void * activations,
                         int          input_rows,
                         int          output_rows);

void ggml_riscv_qbs_gemm(ggml_type    weight_type,
                         int          n,
                         float *      dst,
                         size_t       dst_stride,
                         const void * weights,
                         const void * activations,
                         int          input_rows,
                         int          output_rows);

extern "C" void ggml_riscv_qbs_end_graph(uint64_t graph_epoch);
