#ifndef GGML_RVV_FUSED_RMS_OUTPUT_H
#define GGML_RVV_FUSED_RMS_OUTPUT_H

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

static inline bool ggml_rvv_rms_row_alias_safe(const void *output,
                                             const void *input, size_t bytes) {
    const uintptr_t a = (uintptr_t)output, b = (uintptr_t)input;
    return a == b || (a < b ? b-a : a-b) >= bytes;
}

/* Ordinary valid tensor memory, entry vstart=0. Exact in-place rows are safe;
   shifted overlaps retain the original sequential scalar loop at the caller. */
static inline bool ggml_rvv_fused_rms_output(float *output, const float *input,
                                            const float *weight, float scale, size_t n) {
    if (n > SIZE_MAX / sizeof(float)) return false;
    const size_t bytes = n * sizeof(float);
    if (!ggml_rvv_rms_row_alias_safe(output, input, bytes) ||
        !ggml_rvv_rms_row_alias_safe(output, weight, bytes)) return false;
    for (size_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m4(n-i);
        const vfloat32m4_t x = __riscv_vle32_v_f32m4(input+i, vl);
        const vfloat32m4_t w = __riscv_vle32_v_f32m4(weight+i, vl);
        // Keep two separate FP32 roundings, as in (x * scale) * weight.
        const vfloat32m4_t normalized = __riscv_vfmul_vf_f32m4(x, scale, vl);
        const vfloat32m4_t result = __riscv_vfmul_vv_f32m4(normalized, w, vl);
        __riscv_vse32_v_f32m4(output+i, result, vl);
        i += vl;
    }
    return true;
}

#endif
