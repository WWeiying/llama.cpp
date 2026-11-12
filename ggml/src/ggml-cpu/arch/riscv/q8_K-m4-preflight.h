#pragma once

#include "q8_K-m4-packed.h"
#ifdef GGML_USE_RISCV_QBS_M4_PREFLIGHT
#include "m4_preflight.h"
#endif

namespace ggml_riscv_q8_K_preflight {

inline bool enabled(int64_t k, bool single_worker, bool lookup) {
#ifdef GGML_USE_RISCV_QBS_M4_PREFLIGHT
    static const bool requested = [] {
        const char *value = ggml_riscv_accel_option("GGML_RISCV_QBS_M4_PREFLIGHT");
        return value && std::strcmp(value, "1") == 0;
    }();
    static const bool prepare = [] {
        const char *value = ggml_riscv_accel_option("GGML_RISCV_QBS_M4_PREPARE");
        return value && std::strcmp(value, "1") == 0;
    }();
    static const bool cache = [] {
        const char *value = ggml_riscv_accel_option("GGML_RISCV_QBS_M4_CACHE");
        return value && std::strcmp(value, "1") == 0;
    }();
    const bool measured_k = k == 1536 || k == 2048 || k == 4096;
    return requested && measured_k && single_worker && lookup &&
        prepare && cache &&
        ggml_riscv_q8_K_m4_packed::enabled() &&
        ggml_riscv_q8_K_m4_packed::parallel_sums_enabled();
#else
    (void) k; (void) single_worker; (void) lookup;
    return false;
#endif
}

inline uint32_t producer_tag(bool selected) {
    return selected ? 4u : ggml_riscv_q8_K_m4_packed::producer_tag();
}

inline bool quantize(bool selected, const float *input, void *output, int64_t k) {
#ifdef GGML_USE_RISCV_QBS_M4_PREFLIGHT
    if (selected) return qbs_experimental_m4_preflight::quantize(input, output, k);
#else
    (void) selected;
#endif
    return ggml_riscv_q8_K_m4_packed::selected_quantize(input, output, k);
}

} // namespace ggml_riscv_q8_K_preflight
