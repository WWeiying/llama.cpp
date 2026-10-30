#pragma once

#include "ggml.h"

#if defined(GGML_USE_RISCV_QUANT_KV)
#include "qbs/qbs.h"
#endif

#include <cstddef>
#include <cstdint>

struct ggml_compute_params;

struct ggml_riscv_qbs_activation_plan {
    bool   wide_m;
    size_t packed_bytes;
    size_t row_major_offset;
    size_t row_major_bytes;
    size_t workspace_bytes;
};

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

#if defined(GGML_USE_RISCV_QUANT_KV)
// Native-only cached capabilities; null when disabled, emulated or unavailable.
const qbs_device_t * ggml_riscv_qbs_direct_device();

// Synchronous task under the QBS context mutex, without consuming its reservation.
// Returns the task result unchanged, or -1 for null/nested/unavailable entry.
// The executor is valid only on the task thread during this call. The task must
// not reenter other QBS context APIs or acquire the caller's AKV context mutex.
int ggml_riscv_qbs_direct_run(int (*task)(void *, qbs_command_executor_t), void * opaque);
#endif

// Read once when constructing a weight-layout trait; zero keeps global R4.
unsigned ggml_riscv_qbs_weight_segment_blocks();

void ggml_riscv_qbs_trace_tensor_decision(const void * tensor_identity,
                                          const char * tensor_name,
                                          ggml_type    type,
                                          int          n_dims,
                                          int64_t      k_elements,
                                          int64_t      output_rows,
                                          int64_t      matrix_count,
                                          bool         selected);

bool ggml_riscv_qbs_trace_activation_lifetime_enabled();

// Selects the activation layout once for a complete MUL_MAT invocation.  A
// wide-M plan contains both the final packed region and a non-overlapping
// row-major staging region used by the generic GGML quantizer.
bool ggml_riscv_qbs_plan_activations(ggml_type                          weight_type,
                                     int64_t                            input_rows,
                                     int64_t                            output_rows,
                                     int64_t                            k_elements,
                                     ggml_riscv_qbs_activation_plan * plan);

int ggml_riscv_qbs_pack_wide_activations(ggml_type    weight_type,
                                         const void * row_major,
                                         size_t       row_major_bytes,
                                         int64_t      input_rows,
                                         int64_t      k_elements,
                                         void *       packed,
                                         size_t       packed_bytes);

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
                                              uint64_t     quantize_time_us,
                                              bool         prepared_cache_active = false);

int ggml_riscv_qbs_repack_r4(ggml_type    type,
                             void *       dst,
                             size_t       dst_size,
                             const void * src,
                             size_t       src_size,
                             int64_t      output_rows,
                             int64_t      k_blocks,
                             unsigned     segment_k_blocks = 0);

void ggml_riscv_qbs_mul_mat_view(ggml_type weight_type, int k, float * dst, size_t dst_stride,
                               const void * weights, const void * activations,
                               int input_rows, int output_rows, int64_t storage_rows,
                               int64_t row_start, unsigned segment_k_blocks, bool wide_m);

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

void ggml_riscv_qbs_gemm_wide(ggml_type    weight_type,
                              int          n,
                              float *      dst,
                              size_t       dst_stride,
                              const void * weights,
                              const void * activations,
                              int          input_rows,
                              int          output_rows);

extern "C" void ggml_riscv_qbs_end_graph(uint64_t graph_epoch);
