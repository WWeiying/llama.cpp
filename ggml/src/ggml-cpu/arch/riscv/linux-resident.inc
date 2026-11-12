// Included only by the opt-in Linux portal and its host tests.
#include "accel-options.h"
#include "client.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace ggml_akv_linux {

enum class result { declined, complete, failed };

struct counters {
    std::atomic<uint64_t> portal_calls{}, portal_groups{}, portal_declines{}, portal_failed{};
};
static counters counts;

static void report() {
    if (!ggml_riscv_accel_true(std::getenv("GGML_RISCV_AKV_TRACE"))) {
        return;
    }
    std::fprintf(stderr, "GGML_RISCV_AKV_LINUX_PORTAL portal_calls=%llu portal_groups=%llu portal_declines=%llu portal_failed=%llu\n",
                 (unsigned long long) counts.portal_calls.load(), (unsigned long long) counts.portal_groups.load(),
                 (unsigned long long) counts.portal_declines.load(), (unsigned long long) counts.portal_failed.load());
}

static result decline() {
    counts.portal_declines.fetch_add(1, std::memory_order_relaxed);
    return result::declined;
}

static result fail() {
    counts.portal_failed.fetch_add(1, std::memory_order_relaxed);
    return result::failed;
}

struct span { uintptr_t begin, end; };

static bool tensor_span(const ggml_tensor * t, size_t element, span & out) {
    if (!t->data || t->nb[0] != element || uintptr_t(t->data) % element) {
        return false;
    }
    size_t bytes = element;
    for (unsigned i = 0; i < 4; ++i) {
        if (t->ne[i] <= 0 || t->nb[i] % element ||
            (t->ne[i] > 1 && t->nb[i] < bytes)) {
            return false;
        }
        size_t extra;
        if (__builtin_mul_overflow(size_t(t->ne[i] - 1), t->nb[i], &extra) ||
            __builtin_add_overflow(bytes, extra, &bytes)) {
            return false;
        }
    }
    out.begin = uintptr_t(t->data);
    if (bytes > size_t(PTRDIFF_MAX) || bytes > UINTPTR_MAX - out.begin) {
        return false;
    }
    out.end = out.begin + bytes;
    return true;
}

static bool overlaps(const span & a, const span & b) {
    return a.begin < b.end && b.begin < a.end;
}

struct shape {
    uint32_t dimension, rows, tokens, scale_bits;
    size_t output_elements;
};

static bool inspect(const ggml_compute_params * params, const ggml_tensor * dst, shape & s) {
    if (!params || !dst || params->nth != 1 || params->ith != 0 || params->use_ref || dst->src[4]) {
        return false;
    }
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    if (!q || !k || !v || !mask || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // Small memcpy builtins can use RVV even with loop/SLP vectorization disabled.
    s.scale_bits = static_cast<uint32_t>(dst->op_params[0]);
    const uint32_t bias = static_cast<uint32_t>(dst->op_params[1]);
    const uint32_t softcap = static_cast<uint32_t>(dst->op_params[2]);
    // No FP instructions before the post-open state guard. Accept either signed zero.
    if ((s.scale_bits >> 31) || !(s.scale_bits & 0x7fffffffU) ||
        s.scale_bits > 0x3f800000U ||
        (bias & 0x7fffffffU) || (softcap & 0x7fffffffU)) {
        return false;
    }
    if ((q->ne[0] != 64 && q->ne[0] != 96 && q->ne[0] != 128) || q->ne[1] != 1 ||
        q->ne[2] <= 0 || q->ne[2] > UINT32_MAX || q->ne[3] != 1 ||
        k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] || k->ne[1] <= 0 || v->ne[1] != k->ne[1] ||
        k->ne[2] <= 0 || v->ne[2] != k->ne[2] || k->ne[3] != 1 || v->ne[3] != 1 ||
        q->ne[2] % k->ne[2] || dst->ne[0] != q->ne[0] || dst->ne[1] != q->ne[2] ||
        dst->ne[2] != 1 || dst->ne[3] != 1 || mask->ne[0] < k->ne[1] ||
        mask->ne[1] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    span inputs[4], output;
    if (!tensor_span(q, 4, inputs[0]) || !tensor_span(k, 2, inputs[1]) ||
        !tensor_span(v, 2, inputs[2]) || !tensor_span(mask, 2, inputs[3]) || !tensor_span(dst, 4, output)) {
        return false;
    }
    for (const auto & input : inputs) {
        if (overlaps(input, output)) {
            return false;
        }
    }
    s.dimension = uint32_t(q->ne[0]);
    s.rows = uint32_t(q->ne[2] / k->ne[2]);
    if (__builtin_mul_overflow(size_t(q->ne[2]), size_t(s.dimension), &s.output_elements) ||
        s.output_elements > size_t(PTRDIFF_MAX) / sizeof(float)) {
        return false;
    }
    const auto * m = static_cast<const ggml_fp16_t *>(mask->data);
    size_t active = 0;
    while (active < size_t(k->ne[1]) && m[active] != 0xfc00U) {
        if ((m[active] & 0x7c00U) == 0x7c00U || active == ARA_RESIDENT_MAX_TOKENS) {
            return false;
        }
        ++active;
    }
    for (size_t t = active; t < size_t(k->ne[1]); ++t) {
        if (m[t] != 0xfc00U) {
            return false;
        }
    }
    // All-masked rows stay with the original path, without opening/probing a device.
    s.tokens = uint32_t(active);
    return active != 0;
}

static int read_state() {
#ifdef GGML_AKV_LINUX_PORTAL_TEST
    extern int test_read_state();
    return test_read_state();
#else
    unsigned long vstart, fcsr;
    asm volatile("csrr %0, vstart" : "=r"(vstart) : : "memory");
    if (vstart) {
        return -ERANGE;
    }
    asm volatile("csrr %0, fcsr" : "=r"(fcsr) : : "memory");
    return (fcsr & 0xe0U) ? -EOPNOTSUPP : 0;
#endif
}

static result try_flash_attn(const ggml_compute_params * params, ggml_tensor * dst) {
    const char * option = std::getenv("GGML_RISCV_AKV_LINUX_PORTAL");
    if ((option && std::strcmp(option, "auto") && !ggml_riscv_accel_true(option)) ||
        !ggml_riscv_accel_true(ggml_riscv_accel_option("GGML_RISCV_AKV")) ||
        ggml_riscv_accel_true(ggml_riscv_accel_option("GGML_RISCV_AKV_EMULATE"))) {
        return result::declined;
    }
    static const bool registered = [] { std::atexit(report); return true; }();
    (void) registered;
    shape s;
    if (!inspect(params, dst, s)) {
        return decline();
    }
    ara_resident_info info;
    const int fd = ara_resident_linux_open(&info);
    if (fd < 0) {
        switch (-fd) {
            case ENOENT: case ENXIO: case ENODEV: case EOPNOTSUPP: case EBUSY:
            case EACCES: case EPERM: case EPROTO:
                return decline();
            default:
                return fail();
        }
    }
    struct fd_guard {
        int fd;
        ~fd_guard() { ara_resident_linux_close(fd); }
    } guard{fd};
    const int state = read_state();
    if (state) {
        return state == -EOPNOTSUPP ? decline() : fail();
    }
    // Allocate every buffer before the first ioctl. Nothing writes dst until all groups complete.
    const size_t kv_elements = size_t(s.tokens) * s.dimension;
    std::unique_ptr<ggml_fp16_t[]> packed(new (std::nothrow) ggml_fp16_t[2 * kv_elements + ARA_RESIDENT_MAX_HEADS * s.dimension + s.tokens]);
    std::unique_ptr<float[]> output(new (std::nothrow) float[s.output_elements]);
    if (!packed || !output) {
        return decline();
    }
    auto * key = packed.get();
    auto * value = key + kv_elements;
    auto * query = value + kv_elements;
    auto * mask = query + ARA_RESIDENT_MAX_HEADS * s.dimension;
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    std::memcpy(mask, dst->src[3]->data, s.tokens * sizeof(*mask));
    const ggml_from_float_t convert = ggml_get_type_traits_cpu(GGML_TYPE_F16)->from_float;
    if (!convert) {
        return decline();
    }
    counts.portal_calls.fetch_add(1, std::memory_order_relaxed);
    size_t completed = 0;
    for (size_t kv = 0; kv < size_t(k->ne[2]); ++kv) {
        for (size_t t = 0; t < s.tokens; ++t) {
            std::memcpy(key + t * s.dimension, static_cast<const char *>(k->data) + kv * k->nb[2] + t * k->nb[1], s.dimension * 2);
            std::memcpy(value + t * s.dimension, static_cast<const char *>(v->data) + kv * v->nb[2] + t * v->nb[1], s.dimension * 2);
        }
        for (uint32_t start = 0; start < s.rows;) {
            const uint32_t heads = std::min(s.rows - start, ARA_RESIDENT_MAX_HEADS);
            const size_t first = kv * s.rows + start;
            for (uint32_t h = 0; h < heads; ++h) {
                convert(reinterpret_cast<const float *>(static_cast<const char *>(q->data) + (first + h) * q->nb[2]), query + h * s.dimension, s.dimension);
            }
            ara_resident_job job{};
            job.abi = ARA_RESIDENT_ABI;
            job.bytes = sizeof(job);
            job.heads = heads;
            job.dimension = s.dimension;
            job.tokens = s.tokens;
            job.scale_bits = s.scale_bits;
            job.query = uintptr_t(query);
            job.key = uintptr_t(key);
            job.value = uintptr_t(value);
            job.mask = uintptr_t(mask);
            job.output = uintptr_t(output.get() + first * s.dimension);
            const long rc = ara_resident_linux_execute(fd, &job);
            if (rc) {
                if (!completed && (rc == -EOPNOTSUPP || rc == -ENODEV || rc == -EBUSY)) {
                    return decline();
                }
                return fail();
            }
            ++completed;
            counts.portal_groups.fetch_add(1, std::memory_order_relaxed);
            start += heads;
        }
    }
    for (size_t h = 0; h < size_t(q->ne[2]); ++h) {
        std::memcpy(static_cast<char *>(dst->data) + h * dst->nb[1], output.get() + h * s.dimension, s.dimension * sizeof(float));
    }
    return result::complete;
}

static bool flash_attn(const ggml_compute_params * params, ggml_tensor * dst) {
    const result status = try_flash_attn(params, dst);
    if (status == result::failed) {
        GGML_ABORT("Linux Resident portal failed; the attention operation must not be retried");
    }
    return status == result::complete;
}

} // namespace ggml_akv_linux
