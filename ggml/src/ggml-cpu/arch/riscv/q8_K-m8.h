#pragma once

#include "q8_K-m4-packed.h"
#include <limits>

namespace ggml_riscv_q8_K_m8 {

struct block {
    float d[8];
    int8_t qs[8*QK_K];
    int16_t bsums[8*(QK_K/16)];
};
static_assert(sizeof(block) == 8*sizeof(block_q8_K), "M8 block layout changed");

inline bool valid_buffers(const float *x, void *output, int64_t rows, int64_t k) {
    const size_t limit = std::numeric_limits<size_t>::max();
    if (!x || !output || rows <= 0 || rows % 8 || k <= 0 || k % QK_K ||
        k/QK_K > INT_MAX || uint64_t(k) > limit/sizeof(float)) {
        return false;
    }
    const size_t row_bytes = size_t(k)*sizeof(float);
    if (uint64_t(rows) > limit/row_bytes || uint64_t(rows) > limit/(size_t(k/QK_K)*sizeof(block_q8_K))) {
        return false;
    }
    const size_t input_bytes = size_t(rows)*row_bytes;
    const size_t output_bytes = size_t(rows)*size_t(k/QK_K)*sizeof(block_q8_K);
    const uintptr_t in = reinterpret_cast<uintptr_t>(x), out = reinterpret_cast<uintptr_t>(output);
    const uintptr_t address_limit = std::numeric_limits<uintptr_t>::max();
    if (in % alignof(float) || out % alignof(block) ||
        input_bytes > address_limit - in || output_bytes > address_limit - out) {
        return false;
    }
    return in >= out ? in - out >= output_bytes : out - in >= input_bytes;
}

inline bool enabled() {
    return GGML_RISCV_Q8_K_M4_VECTOR && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ &&
        ggml_riscv_accel_true(ggml_riscv_accel_option("GGML_RISCV_QBS_M4_PREPARE")) &&
        ggml_riscv_q8_K_m4_packed::enabled() &&
        ggml_riscv_q8_K_m4_packed::parallel_sums_enabled();
}

inline void require_zero_vstart() {
#if GGML_RISCV_Q8_K_M4_VECTOR
    unsigned vstart;
    asm volatile("csrr %0,vstart" : "=r"(vstart) :: "memory");
    GGML_ASSERT(vstart == 0);
#endif
}

#if GGML_RISCV_Q8_K_M4_VECTOR
inline void sums(vint8m1_t q, int16_t *out, size_t vl) {
    auto sum = __riscv_vsext_vf2_i16m2(q, vl);
    sum = __riscv_vadd_vv_i16m2(sum, __riscv_vslidedown_vx_i16m2(sum, 8, vl), vl);
    sum = __riscv_vadd_vv_i16m2(sum, __riscv_vslidedown_vx_i16m2(sum, 4, vl), vl);
    sum = __riscv_vadd_vv_i16m2(sum, __riscv_vslidedown_vx_i16m2(sum, 2, vl), vl);
    sum = __riscv_vadd_vv_i16m2(sum, __riscv_vslidedown_vx_i16m2(sum, 1, vl), vl);
    const size_t groups = vl/16;
    auto index = __riscv_vsll_vx_u16m2(__riscv_vid_v_u16m2(groups), 4, groups);
    sum = __riscv_vrgather_vv_i16m2(sum, index, groups);
    __riscv_vsse16_v_i16m2(out, 8*sizeof(int16_t), sum, groups);
}

inline void store(int8_t *out, vint8m1_t q0, vint8m1_t q1,
                  vint8m1_t q2, vint8m1_t q3, size_t vl) {
    using namespace ggml_riscv_q8_K_m4_packed;
    auto words = widen(q0, vl);
    words = __riscv_vor_vv_u32m4(words, __riscv_vsll_vx_u32m4(widen(q1, vl), 8, vl), vl);
    words = __riscv_vor_vv_u32m4(words, __riscv_vsll_vx_u32m4(widen(q2, vl), 16, vl), vl);
    words = __riscv_vor_vv_u32m4(words, __riscv_vsll_vx_u32m4(widen(q3, vl), 24, vl), vl);
    __riscv_vsse32_v_u32m4(reinterpret_cast<alias_word *>(out), 8, words, vl);
}
#endif

// Rejection writes nothing. The caller must then use the original M4 path.
inline bool quantize(const float *x, void *output, int64_t rows, int64_t k) {
    if (!valid_buffers(x, output, rows, k)) {
        return false;
    }
    require_zero_vstart();
#if GGML_RISCV_Q8_K_M4_VECTOR && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    unsigned frm;
    asm volatile("frrm %0" : "=r"(frm) :: "memory");
    if (frm) {
        return false;
    }
    const size_t span = __riscv_vsetvl_e32m4(128);
    if (span < 16 || span % 16 || QK_K % span) {
        return false;
    }
    // Check every M4 producer before the first output store, including late groups.
    for (int64_t row = 0; row < rows; row += 4) {
        if (!ggml_riscv_q8_K_m4::vector_domain(x + row*k, k)) {
            return false;
        }
    }
    auto *y = static_cast<block *>(output);
    for (int64_t first = 0; first < rows; first += 8) {
        for (int64_t b = 0; b < k/QK_K; ++b) {
            auto &destination = y[(first/8)*(k/QK_K) + b];
            float scale[8];
            for (int row = 0; row < 8; ++row) {
                float absolute;
                const float maximum = ggml_riscv_q8_K_signed_max(x + (first + row)*k + b*QK_K, QK_K, &absolute);
                scale[row] = absolute ? -127.0f/maximum : 0.0f;
                destination.d[row] = absolute ? 1.0f/scale[row] : 0.0f;
            }
            for (size_t offset = 0; offset < QK_K; offset += span) {
                const size_t vl = __riscv_vsetvl_e32m4(span);
                for (int half = 0; half < 8; half += 4) {
                    using ggml_riscv_q8_K_m4_packed::encode;
                    const auto q0 = encode(x + (first + half)*k + b*QK_K + offset, scale[half], vl);
                    sums(q0, destination.bsums + 8*(offset/16) + half, vl);
                    const auto q1 = encode(x + (first + half + 1)*k + b*QK_K + offset, scale[half + 1], vl);
                    sums(q1, destination.bsums + 8*(offset/16) + half + 1, vl);
                    const auto q2 = encode(x + (first + half + 2)*k + b*QK_K + offset, scale[half + 2], vl);
                    sums(q2, destination.bsums + 8*(offset/16) + half + 2, vl);
                    const auto q3 = encode(x + (first + half + 3)*k + b*QK_K + offset, scale[half + 3], vl);
                    sums(q3, destination.bsums + 8*(offset/16) + half + 3, vl);
                    store(destination.qs + 8*offset + half, q0, q1, q2, q3, vl);
                }
            }
        }
    }
    return true;
#else
    return false;
#endif
}

} // namespace ggml_riscv_q8_K_m8
