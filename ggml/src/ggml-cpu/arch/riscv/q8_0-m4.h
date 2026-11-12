#pragma once

// The tested RISC-V roundf implementation restores conversion flags.
// Revalidate bytes and fflags when changing compiler or libm.
#include "ggml-impl.h"
#include "simd-mappings.h"
#include "arch-fallback.h"
#include "repack.h"

#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__riscv_vector) && !defined(__FAST_MATH__) && \
    (!defined(__FINITE_MATH_ONLY__) || !__FINITE_MATH_ONLY__)
#define GGML_RISCV_Q8_0_M4_VECTOR 1
#include <riscv_vector.h>
#else
#define GGML_RISCV_Q8_0_M4_VECTOR 0
#endif

namespace ggml_riscv_q8_0_m4 {

// Integer-only preflight: no FP exceptions, output writes, or global state.
// Reject any subnormal/nonfinite input, and very small nonzero block maxima.
// With amax >= 2^-110, d >= 2^-117 and 1/d cannot overflow binary32.
// This conservative guard deliberately leaves some safe inputs to generic.
inline bool vector_domain(const float * x, int64_t k) {
    for (int64_t b = 0; b < k / QK8_0; ++b) {
        for (int row = 0; row < 4; ++row) {
            uint32_t maximum = 0;
            for (int j = 0; j < QK8_0; ++j) {
                uint32_t bits;
                std::memcpy(&bits, x + row * k + b * QK8_0 + j, sizeof(bits));
                bits &= 0x7fffffffU;
                if (bits >= 0x7f800000U || (bits != 0 && bits < 0x00800000U)) {
                    return false;
                }
                if (bits > maximum) maximum = bits;
            }
            if (maximum != 0 && maximum < 0x08800000U) return false;
        }
    }
    return true;
}

// Same preconditions as production: four contiguous rows, k a multiple of 32,
// suitably aligned output, no input/output overlap. True means RVV was used.
// Non-RNE callers retain their environment and use the real production generic.
// MaxAVL is only a test hook for forcing multiple strip-mining iterations.
template <size_t MaxAVL = QK8_0>
inline bool quantize(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    static_assert(MaxAVL > 0 && MaxAVL <= QK8_0, "invalid AVL cap");
    static_assert(QK8_0 == 32 && sizeof(block_q8_0x4) == 136, "layout changed");
    assert(k >= 0 && k % QK8_0 == 0 && k / QK8_0 <= INT_MAX);
#if GGML_RISCV_Q8_0_M4_VECTOR
    unsigned frm;
    asm volatile("frrm %0" : "=r"(frm) : : "memory");
    if (frm != 0 || k == 0 || !vector_domain(x, k)) {
        ggml_quantize_mat_q8_0_4x1_generic(x, vy, k);
        return false;
    }

    auto * y = static_cast<block_q8_0x4 *>(vy);
    float srcv[4][QK8_0];
    float id[4];
    const int nb = static_cast<int>(k / QK8_0);
    for (int i = 0; i < nb; ++i) {
        // Keep the production scalar evaluation order and half conversion.
        for (int row = 0; row < 4; ++row) {
            float amax = 0.0f;
            for (int j = 0; j < QK8_0; ++j) {
                srcv[row][j] = x[row * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row][j]));
            }
            const float d = amax / ((1 << 7) - 1);
            id[row] = d ? 1.0f / d : 0.0f;
            y[i].d[row] = GGML_CPU_FP32_TO_FP16(d);
        }
        for (int row = 0; row < 4; ++row) {
            for (size_t j = 0; j < QK8_0;) {
                const size_t remaining = QK8_0 - j;
                const size_t vl = __riscv_vsetvl_e32m1(remaining < MaxAVL ? remaining : MaxAVL);
                vfloat32m1_t product = __riscv_vle32_v_f32m1(srcv[row] + j, vl);
                product = __riscv_vfmul_vf_f32m1(product, id[row], vl);
                unsigned flags;
                // The read/write vector operand orders the flag read AFTER mul
                // and BEFORE conversion. A memory clobber alone cannot do this.
                asm volatile("frflags %0" : "=r"(flags), "+vr"(product) : : "memory");
                const vint32m1_t q32 = __riscv_vfcvt_x_f_v_i32m1_rm(product, __RISCV_FRM_RMM, vl);
                const vint16mf2_t q16 = __riscv_vncvt_x_x_w_i16mf2(q32, vl);
                const vint8mf4_t q8 = __riscv_vncvt_x_x_w_i8mf4(q16, vl);
                __riscv_vsse8_v_i8mf4(y[i].qs + 4 * j + row, 4, q8, vl);
                // All guarded products round into [-127,127]. Integer narrowing
                // is exact and nonsaturating; no vxrm/vxsat changes are needed.
                // Drop conversion NX only, never flags raised by mul or scales.
                asm volatile("fsflags %0" : : "r"(flags), "vr"(q8) : "memory");
                j += vl;
            }
        }
    }
    return true;
#else
    ggml_quantize_mat_q8_0_4x1_generic(x, vy, k);
    return false;
#endif
}

} // namespace ggml_riscv_q8_0_m4
