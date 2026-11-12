#ifndef GGML_RISCV_Q8_K_SIGNED_MAX_H
#define GGML_RISCV_Q8_K_SIGNED_MAX_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

/* Finite input, disjoint output. Equal magnitudes select the first element;
 * all zeros produce +0. The caller performs the quantizer domain preflight. */
static inline float ggml_riscv_q8_K_signed_max(const float *x, size_t count,
                                              float *absolute) {
  *absolute = 0.0f;
  if (!count) return 0.0f;
  const size_t span = __riscv_vsetvl_e32m1(count < 32 ? count : 32);
  vuint32m1_t maxima = __riscv_vmv_v_x_u32m1(0, span);
  for (size_t i = 0; i < count;) {
    size_t remaining = count - i;
    size_t vl = __riscv_vsetvl_e32m1(remaining < span ? remaining : span);
    vuint32m1_t bits = __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vle32_v_f32m1(x + i, vl));
    bits = __riscv_vand_vx_u32m1(bits, UINT32_C(0x7fffffff), vl);
    maxima = __riscv_vmaxu_vv_u32m1_tu(maxima, maxima, bits, vl);
    i += vl;
  }
  vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, 1);
  uint32_t maximum = __riscv_vmv_x_s_u32m1_u32(
      __riscv_vredmaxu_vs_u32m1_u32m1(maxima, zero, span));
  if (!maximum) return 0.0f;
  memcpy(absolute, &maximum, sizeof(maximum));
  for (size_t i = 0; i < count;) {
    size_t remaining = count - i;
    size_t vl = __riscv_vsetvl_e32m1(remaining < span ? remaining : span);
    vuint32m1_t bits = __riscv_vreinterpret_v_f32m1_u32m1(__riscv_vle32_v_f32m1(x + i, vl));
    bits = __riscv_vand_vx_u32m1(bits, UINT32_C(0x7fffffff), vl);
    long first = __riscv_vfirst_m_b32(__riscv_vmseq_vx_u32m1_b32(bits, maximum, vl), vl);
    if (first >= 0) return x[i + (size_t)first];
    i += vl;
  }
  __builtin_trap();
}

#endif
