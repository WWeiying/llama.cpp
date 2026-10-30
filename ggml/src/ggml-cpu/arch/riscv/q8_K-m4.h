#pragma once

#include "ggml-impl.h"
#include "simd-mappings.h"
#include "arch-fallback.h"
#include "repack.h"
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__riscv_vector) && !defined(__FAST_MATH__) && \
    (!defined(__FINITE_MATH_ONLY__) || !__FINITE_MATH_ONLY__)
#define GGML_RISCV_Q8_K_M4_VECTOR 1
#include <riscv_vector.h>
#include "q8_K-signed-max.h"
#else
#define GGML_RISCV_Q8_K_M4_VECTOR 0
#endif

namespace ggml_riscv_q8_K_m4 {

#if GGML_RISCV_Q8_K_M4_VECTOR
// Integer-only domain checks preserve the caller's floating-point flags.
template <size_t MaxAVL = 32>
inline bool vector_domain(const float * x, int64_t k) {
    static_assert(MaxAVL > 0, "invalid AVL cap");
    constexpr size_t block = 256;
    const size_t span = __riscv_vsetvl_e32m1(MaxAVL < block ? MaxAVL : block);
    for (int64_t b = 0; b < k / int64_t(block); ++b) {
        for (int row = 0; row < 4; ++row) {
            auto maximum = __riscv_vmv_v_x_u32m1(0, span);
            auto minimum_minus_one = __riscv_vmv_v_x_u32m1(UINT32_MAX, span);
            for (size_t offset = 0; offset < block;) {
                const size_t remaining = block - offset;
                const size_t vl = __riscv_vsetvl_e32m1(remaining < span ? remaining : span);
                const auto input = __riscv_vle32_v_f32m1(x + row*k + b*block + offset, vl);
                const auto bits = __riscv_vreinterpret_v_f32m1_u32m1(input);
                const auto absolute = __riscv_vand_vx_u32m1(bits, UINT32_C(0x7fffffff), vl);
                const auto minus_one = __riscv_vsub_vx_u32m1(absolute, 1, vl);
                maximum = __riscv_vmaxu_vv_u32m1_tu(maximum, maximum, absolute, vl);
                minimum_minus_one = __riscv_vminu_vv_u32m1_tu(
                    minimum_minus_one, minimum_minus_one, minus_one, vl);
                offset += vl;
            }
            const auto zero = __riscv_vmv_v_x_u32m1(0, 1);
            const auto ones = __riscv_vmv_v_x_u32m1(UINT32_MAX, 1);
            const uint32_t max_bits = __riscv_vmv_x_s_u32m1_u32(
                __riscv_vredmaxu_vs_u32m1_u32m1(maximum, zero, span));
            const uint32_t min_minus_one = __riscv_vmv_x_s_u32m1_u32(
                __riscv_vredminu_vs_u32m1_u32m1(minimum_minus_one, ones, span));
            if (min_minus_one < UINT32_C(0x007fffff) ||
                (max_bits && (max_bits < UINT32_C(0x08800000) ||
                              max_bits > UINT32_C(0x77000000)))) return false;
        }
    }
    return true;
}
#endif

// Four contiguous rows, K a multiple of 256, and nonoverlapping input/output.
// The signed-max tie rule, fused rounding and block sums match the tested
// GNU production generic. Revalidate this contract when changing its compiler.
template <size_t MaxAVL = 16>
inline bool quantize(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    static_assert(MaxAVL > 0 && MaxAVL <= 16, "invalid AVL cap");
    static_assert(QK_K == 256, "native block layout changed");
    assert(k >= 0 && k % QK_K == 0 && k / QK_K <= INT_MAX);
#if GGML_RISCV_Q8_K_M4_VECTOR
    unsigned frm;
    asm volatile("frrm %0" : "=r"(frm) : : "memory");
    if (frm != 0 || k == 0 || !vector_domain(x, k)) {
        ggml_quantize_mat_q8_K_4x1_ara_generic(x, vy, k);
        return false;
    }
    auto * y = static_cast<block_q8_Kx4 *>(vy);
    for (int64_t b = 0; b < k / QK_K; ++b) {
        float scales[4];
        for (int row = 0; row < 4; ++row) {
            float amax;
            const float maximum = ggml_riscv_q8_K_signed_max(
                x + row * k + b * QK_K, QK_K, &amax);
            scales[row] = amax ? -127.0f / maximum : 0.0f;
            y[b].d[row] = amax ? 1.0f / scales[row] : 0.0f;
        }
        for (int row = 0; row < 4; ++row) {
            for (int group = 0; group < QK_K / 16; ++group) {
                int sum = 0;
                for (size_t offset = 0; offset < 16;) {
                    const size_t remaining = 16 - offset;
                    const size_t vl = __riscv_vsetvl_e32m1(remaining < MaxAVL ? remaining : MaxAVL);
                    const size_t element = group * 16 + offset;
                    const auto input = __riscv_vle32_v_f32m1(x + row*k + b*QK_K + element, vl);
                    const auto bias = __riscv_vfmv_v_f_f32m1(12582912.0f, vl);
                    const auto biased = __riscv_vfmacc_vf_f32m1(bias, scales[row], input, vl);
                    auto bits = __riscv_vreinterpret_v_f32m1_i32m1(biased);
                    bits = __riscv_vand_vx_i32m1(bits, 0x007fffff, vl);
                    bits = __riscv_vsub_vx_i32m1(bits, 0x00400000, vl);
                    const auto q16 = __riscv_vncvt_x_x_w_i16mf2(bits, vl);
                    const auto q8 = __riscv_vncvt_x_x_w_i8mf4(q16, vl);
                    __riscv_vsse8_v_i8mf4(y[b].qs + 4*element + row, 4, q8, vl);
                    const auto zero = __riscv_vmv_v_x_i16m1(0, 1);
                    const auto subtotal = __riscv_vwredsum_vs_i8mf4_i16m1(q8, zero, vl);
                    sum += __riscv_vmv_x_s_i16m1_i16(subtotal);
                    offset += vl;
                }
                y[b].bsums[group*4 + row] = static_cast<int16_t>(sum);
            }
        }
    }
    return true;
#else
    ggml_quantize_mat_q8_K_4x1_ara_generic(x, vy, k);
    return false;
#endif
}
} // namespace ggml_riscv_q8_K_m4
