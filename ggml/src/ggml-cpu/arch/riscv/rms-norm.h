#ifndef GGML_RISCV_RMS_NORM_H
#define GGML_RISCV_RMS_NORM_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(__riscv_v_intrinsic) && __riscv_xlen == 64 && \
    defined(__riscv_v_elen_fp) && __riscv_v_elen_fp >= 64 && __riscv_flen >= 64
#define GGML_RISCV_ORDERED_RMS_AVAILABLE 1
#include "rms-sum.h"
#include "rms-output.h"
#else
#define GGML_RISCV_ORDERED_RMS_AVAILABLE 0
#endif

struct ggml_rvv_rms_context {
    bool enabled, trace, fused;
    int ith, nth;
    int64_t width;
    uint64_t sum_fast, sum_scalar, output_fast, output_scalar;
};

static inline bool ggml_rvv_rms_flag(const char *name) {
    const char *value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
}

/* Snapshot policy once per blocking operator call, never once per row. */
static inline ggml_rvv_rms_context ggml_rvv_rms_begin(bool fused, int64_t width, int ith, int nth) {
#if GGML_RISCV_ORDERED_RMS_AVAILABLE
    uintptr_t start, rounding;
    __asm__ volatile("csrr %0,vstart;frrm %1" : "=r"(start),"=r"(rounding) :: "memory");
    const bool requested = ggml_rvv_rms_flag("GGML_RISCV_ORDERED_RMS");
    return {requested && start == 0 && rounding <= 4 && width > 0 &&
                uint64_t(width) <= SIZE_MAX/sizeof(float),
            ggml_rvv_rms_flag("GGML_RISCV_ORDERED_RMS_TRACE"),fused,ith,nth,width,0,0,0,0};
#else
    return {false,false,fused,ith,nth,width,0,0,0,0};
#endif
}

static inline bool ggml_rvv_rms_sum_row(ggml_rvv_rms_context *c, const float *x, size_t n, double *sum) {
#if GGML_RISCV_ORDERED_RMS_AVAILABLE
    if (c->enabled) {
        *sum = ggml_rvv_ordered_rms_sum(x,n);
        if (c->trace) ++c->sum_fast;
        return true;
    }
#else
    (void)x; (void)n; (void)sum;
#endif
    if (c->trace) ++c->sum_scalar;
    return false;
}

static inline bool ggml_rvv_rms_output_row(ggml_rvv_rms_context *c, float *y,
        const float *x, const float *w, float scale, size_t n) {
#if GGML_RISCV_ORDERED_RMS_AVAILABLE
    if (c->enabled && ggml_rvv_fused_rms_output(y,x,w,scale,n)) {
        if (c->trace) ++c->output_fast;
        return true;
    }
#else
    (void)y; (void)x; (void)w; (void)scale; (void)n;
#endif
    if (c->trace) ++c->output_scalar;
    return false;
}

static inline void ggml_rvv_rms_finish(const ggml_rvv_rms_context *c) {
    if (c->trace) {
        std::fprintf(stderr,"GGML_RISCV_RMS_EXEC enabled=%u fused=%u ith=%d nth=%d width=%lld "
            "sum_fast=%llu sum_scalar=%llu output_fast=%llu output_scalar=%llu\n",
            unsigned(c->enabled),unsigned(c->fused),c->ith,c->nth,(long long)c->width,
            (unsigned long long)c->sum_fast,(unsigned long long)c->sum_scalar,
            (unsigned long long)c->output_fast,(unsigned long long)c->output_scalar);
    }
}

#endif
