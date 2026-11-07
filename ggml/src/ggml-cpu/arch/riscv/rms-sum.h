#ifndef GGML_RVV_ORDERED_RMS_SUM_H
#define GGML_RVV_ORDERED_RMS_SUM_H

#include <stddef.h>
#include <riscv_vector.h>

/* Match sum += (double)(x[i] * x[i]): round the square in FP32, then
   widen and accumulate in original element order with an FP64 seed. */
static inline double ggml_rvv_ordered_rms_sum(const float *x, size_t n) {
    vfloat64m1_t sum = __riscv_vfmv_v_f_f64m1(0.0, 1);
    for (size_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m4(n - i);
        const vfloat32m4_t value = __riscv_vle32_v_f32m4(x + i, vl);
        const vfloat32m4_t square = __riscv_vfmul_vv_f32m4(value, value, vl);
        sum = __riscv_vfwredosum_vs_f32m4_f64m1(square, sum, vl);
        i += vl;
    }
    return __riscv_vfmv_f_s_f64m1_f64(sum);
}

#endif
