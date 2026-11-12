#ifndef GGML_RISCV_ACCEL_OPTIONS_H
#define GGML_RISCV_ACCEL_OPTIONS_H

#include <cstdlib>
#include <cstring>

#ifndef GGML_RISCV_ACCEL_AUTO
#define GGML_RISCV_ACCEL_AUTO 0
#endif

inline bool ggml_riscv_accel_true(const char * value) {
    return value && (std::strcmp(value, "1") == 0 ||
                     std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
                     std::strcmp(value, "yes") == 0 || std::strcmp(value, "YES") == 0 ||
                     std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0);
}

inline bool ggml_riscv_accel_auto_enabled() {
    const char * value = std::getenv("GGML_RISCV_ACCEL_AUTO");
    return GGML_RISCV_ACCEL_AUTO != 0 && (!value || ggml_riscv_accel_true(value));
}

// Defaults express software intent only, never platform or ownership authority.
inline const char * ggml_riscv_accel_option(const char * name) {
    if (!name) {
        return nullptr;
    }
    if (const char * value = std::getenv(name)) {
        return value;
    }
    const char * const enabled[] = {
        "GGML_RISCV_QBS", "GGML_RISCV_AKV",
        "GGML_RISCV_QBS_ACTIVATION_CONTEXT", "GGML_RISCV_QBS_CROSS_OP_CONTEXT",
        "GGML_RISCV_QBS_M4_PREPARE", "GGML_RISCV_QBS_M4_PACKED",
        "GGML_RISCV_QBS_M4_PARALLEL_SUMS", "GGML_RISCV_QBS_M4_CACHE",
        "GGML_RISCV_QBS_M4_PREFLIGHT",
        "GGML_RISCV_AKV_QUANT_KV", "GGML_RISCV_AKV_QUANT_PREFILL",
        "GGML_RISCV_AKV_REGISTER_PV",
    };
    for (const char * option : enabled) {
        if (std::strcmp(name, option) == 0) {
            return ggml_riscv_accel_auto_enabled() ? "1" : nullptr;
        }
    }
    if (std::strcmp(name, "GGML_RISCV_AKV_KERNEL") == 0) {
        return ggml_riscv_accel_auto_enabled() ? "auto" : nullptr;
    }
    return nullptr;
}

inline bool ggml_riscv_accel_resident_requested(unsigned dim, unsigned rows,
                                               unsigned tokens) {
    const char * value = std::getenv("GGML_RISCV_AKV_RESIDENT");
    if (value && std::strcmp(value, "auto") != 0) {
        return ggml_riscv_accel_true(value);
    }
    return ggml_riscv_accel_auto_enabled() && dim == 128 && rows == 2 &&
           tokens >= 128 && tokens <= 256;
}

#endif
