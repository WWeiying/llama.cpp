// Included only by the opt-in Linux portal and its host tests.
#include "accel-options.h"
#include "client.h"
#include "view_layout.h"
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
    std::atomic<uint64_t> view_registrations{}, view_batches{}, view_input_bytes{};
};
static counters counts;

static void report() {
    if (!ggml_riscv_accel_true(std::getenv("GGML_RISCV_AKV_TRACE"))) {
        return;
    }
    std::fprintf(stderr, "GGML_RISCV_AKV_LINUX_PORTAL portal_calls=%llu portal_groups=%llu portal_declines=%llu portal_failed=%llu\n",
                 (unsigned long long) counts.portal_calls.load(), (unsigned long long) counts.portal_groups.load(),
                 (unsigned long long) counts.portal_declines.load(), (unsigned long long) counts.portal_failed.load());
    std::fprintf(stderr, "GGML_RISCV_AKV_LINUX_VIEWS registrations=%llu batches=%llu copied_input_bytes=%llu\n",
                 (unsigned long long) counts.view_registrations.load(), (unsigned long long) counts.view_batches.load(),
                 (unsigned long long) counts.view_input_bytes.load());
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
    unsigned order[4] = {0, 1, 2, 3};
    for (unsigned i = 0; i < 4; ++i) {
        if (t->ne[i] <= 0 || t->nb[i] % element) return false;
        // GGML permutes head-interleaved KV to [D, tokens, heads, 1].
        for (unsigned j = i; j && t->nb[order[j]] < t->nb[order[j - 1]]; --j)
            std::swap(order[j], order[j - 1]);
    }
    size_t bytes = element;
    for (unsigned axis : order) {
        if (t->ne[axis] == 1) continue;
        if (t->nb[axis] < bytes) return false;
        size_t extra;
        if (__builtin_mul_overflow(size_t(t->ne[axis] - 1), t->nb[axis], &extra) ||
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

struct workspace {
    std::unique_ptr<ggml_fp16_t[]> packed;
    std::unique_ptr<float[]> output;
    size_t packed_capacity = 0, output_capacity = 0;

    bool reserve(size_t p, size_t o) {
        if (p > packed_capacity) {
            std::unique_ptr<ggml_fp16_t[]> next(new (std::nothrow) ggml_fp16_t[p]);
            if (!next) return false;
            packed = std::move(next);
            packed_capacity = p;
        }
        if (o > output_capacity) {
            std::unique_ptr<float[]> next(new (std::nothrow) float[o]);
            if (!next) return false;
            output = std::move(next);
            output_capacity = o;
        }
        return true;
    }
};

struct session_fd {
    int fd = -1;
    int view_support = 0;
    unsigned next_view = 0;
    ara_resident_kv_view views[ARA_RESIDENT_MAX_VIEWS]{};
    void reset() {
        if (fd >= 0) ara_resident_linux_close(fd);
        fd = -1;
        view_support = 0;
        next_view = 0;
        for (auto & view : views) view = {};
    }
    ~session_fd() { reset(); }
};

static session_fd & thread_session() {
    static thread_local session_fd session;
    return session;
}

static result unavailable(int error) {
    switch (-error) {
        case ENOENT: case ENXIO: case ENODEV: case EOPNOTSUPP: case EBUSY:
        case EACCES: case EPERM: case EPROTO:
            return decline();
        default:
            return fail();
    }
}

static result try_views(session_fd & session, const shape & s, ggml_tensor * dst,
                        workspace & work) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    if (k->ne[2] > ARA_RESIDENT_MAX_KV_HEADS || k->nb[1] > UINT32_MAX ||
        k->nb[2] > UINT32_MAX || v->nb[1] > UINT32_MAX || v->nb[2] > UINT32_MAX)
        return decline();
    ara_resident_kv_view layout{};
    layout.abi = ARA_RESIDENT_VIEW_ABI;
    layout.bytes = sizeof(layout);
    layout.dimension = s.dimension;
    layout.kv_heads = uint32_t(k->ne[2]);
    layout.capacity = uint32_t(std::min<int64_t>(k->ne[1], ARA_RESIDENT_MAX_TOKENS));
    layout.key = uintptr_t(k->data);
    layout.value = uintptr_t(v->data);
    layout.key_token_stride = uint32_t(k->nb[1]);
    layout.key_head_stride = uint32_t(k->nb[2]);
    layout.value_token_stride = uint32_t(v->nb[1]);
    layout.value_head_stride = uint32_t(v->nb[2]);
    ara_resident_view_sizes sizes;
    if (!ara_resident_check_view(&layout, &sizes)) return decline();
    if (!session.view_support) {
        ara_resident_view_caps caps;
        const long rc = ara_resident_linux_view_caps(session.fd, &caps);
        if (rc) {
            if (rc == -ENOTTY || rc == -EOPNOTSUPP) session.view_support = -1;
            else session.reset();
            return rc == -ENOTTY ? decline() : unavailable(int(rc));
        }
        session.view_support = 1;
    }
    if (session.view_support < 0) return decline();
    uint64_t handle = 0;
    for (auto & view : session.views) {
        ara_resident_kv_view identity = view;
        identity.handle = 0;
        if (view.handle && !std::memcmp(&identity, &layout, sizeof(layout))) {
            handle = view.handle;
            break;
        }
    }
    if (!handle) {
        auto & slot = session.views[session.next_view];
        if (slot.handle) {
            const long rc = ara_resident_linux_release_kv(session.fd, slot.handle);
            if (rc) {
                session.reset();
                return unavailable(int(rc));
            }
        }
        slot = {};
        const long rc = ara_resident_linux_register_kv(session.fd, &layout);
        if (rc) {
            session.reset();
            if (rc == -ENOMEM || rc == -ENOSPC || rc == -ESTALE) return decline();
            return unavailable(int(rc));
        }
        slot = layout;
        session.next_view = (session.next_view + 1) % ARA_RESIDENT_MAX_VIEWS;
        handle = layout.handle;
        counts.view_registrations.fetch_add(1, std::memory_order_relaxed);
    }
    const ggml_from_float_t convert = ggml_get_type_traits_cpu(GGML_TYPE_F16)->from_float;
    if (!convert || !work.reserve(s.output_elements, s.output_elements)) return decline();
    for (size_t h = 0; h < size_t(q->ne[2]); ++h)
        convert(reinterpret_cast<const float *>(static_cast<const char *>(q->data) + h * q->nb[2]),
                work.packed.get() + h * s.dimension, s.dimension);
    counts.portal_calls.fetch_add(1, std::memory_order_relaxed);
    size_t first = 0, completed = 0;
    while (first < size_t(q->ne[2])) {
        ara_resident_batch_group groups[ARA_RESIDENT_MAX_BATCH]{};
        unsigned count = 0, rows = 0;
        while (count < ARA_RESIDENT_MAX_BATCH && first + rows < size_t(q->ne[2])) {
            const size_t head = first + rows;
            const uint32_t n = std::min(s.rows - uint32_t(head % s.rows), ARA_RESIDENT_MAX_HEADS);
            groups[count++] = {uint32_t(head / s.rows), n, 0};
            rows += n;
        }
        ara_resident_batch batch{};
        batch.abi = ARA_RESIDENT_VIEW_ABI;
        batch.bytes = sizeof(batch);
        batch.count = count;
        batch.dimension = s.dimension;
        batch.tokens = s.tokens;
        batch.scale_bits = s.scale_bits;
        batch.handle = handle;
        batch.groups = uintptr_t(groups);
        batch.query = uintptr_t(work.packed.get() + first * s.dimension);
        batch.mask = uintptr_t(dst->src[3]->data);
        batch.output = uintptr_t(work.output.get() + first * s.dimension);
        const long rc = ara_resident_linux_execute_batch(session.fd, &batch);
        if (rc) {
            session.reset();
            if (!completed && (rc == -ESTALE || rc == -EOPNOTSUPP || rc == -ENODEV || rc == -EBUSY))
                return decline();
            return fail();
        }
        completed += count;
        first += rows;
        counts.portal_groups.fetch_add(count, std::memory_order_relaxed);
        counts.view_batches.fetch_add(1, std::memory_order_relaxed);
        counts.view_input_bytes.fetch_add(size_t(rows) * s.dimension * 2 + s.tokens * 2,
                                          std::memory_order_relaxed);
    }
    for (size_t h = 0; h < size_t(q->ne[2]); ++h)
        std::memcpy(static_cast<char *>(dst->data) + h * dst->nb[1],
                    work.output.get() + h * s.dimension, s.dimension * sizeof(float));
    return result::complete;
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
    const bool views = ggml_riscv_accel_true(std::getenv("GGML_RISCV_AKV_LINUX_VIEWS"));
    const bool retain = views || ggml_riscv_accel_true(std::getenv("GGML_RISCV_AKV_LINUX_SESSION"));
    int fd = retain ? ara_resident_linux_admit() : ara_resident_linux_open(&info);
    if (fd < 0) {
        return unavailable(fd);
    }
    struct fd_guard {
        int fd;
        ~fd_guard() { ara_resident_linux_close(fd); }
    } guard{retain ? -1 : fd};
    const int state = read_state();
    if (state) {
        return state == -EOPNOTSUPP ? decline() : fail();
    }
    // Construct TLS only after the scalar policy/ISA and CSR guards.
    session_fd * session = nullptr;
    if (retain) {
        session = &thread_session();
        if (session->fd < 0) {
            fd = ara_resident_linux_open(&info);
            if (fd < 0) return unavailable(fd);
            session->fd = fd;
        }
        fd = session->fd;
    }
    // Cache storage, never tensor contents. Initialize only after F/V admission.
    static thread_local workspace cached;
    workspace temporary;
    const char * reuse = std::getenv("GGML_RISCV_AKV_LINUX_REUSE");
    // Bound retained output to 64 KiB per worker; larger legal shapes remain usable.
    workspace & work = ((!reuse || ggml_riscv_accel_true(reuse)) && s.output_elements <= 16384)
                        ? cached : temporary;
    if (views) return try_views(*session, s, dst, work);
    // Allocate every buffer before the first ioctl. Nothing writes dst until all groups complete.
    const size_t kv_elements = size_t(s.tokens) * s.dimension;
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    const bool pack_key = k->nb[1] != s.dimension * 2;
    const bool pack_value = v->nb[1] != s.dimension * 2;
    if (!work.reserve((size_t(pack_key) + size_t(pack_value)) * kv_elements +
                      ARA_RESIDENT_MAX_HEADS * s.dimension, s.output_elements)) {
        return decline();
    }
    auto * key = work.packed.get();
    auto * value = key + size_t(pack_key) * kv_elements;
    auto * query = value + size_t(pack_value) * kv_elements;
    const auto * mask = static_cast<const ggml_fp16_t *>(dst->src[3]->data);
    const ggml_from_float_t convert = ggml_get_type_traits_cpu(GGML_TYPE_F16)->from_float;
    if (!convert) {
        return decline();
    }
    counts.portal_calls.fetch_add(1, std::memory_order_relaxed);
    size_t completed = 0;
    for (size_t kv = 0; kv < size_t(k->ne[2]); ++kv) {
        const auto * source_key = static_cast<const char *>(k->data) + kv * k->nb[2];
        const auto * source_value = static_cast<const char *>(v->data) + kv * v->nb[2];
        if (pack_key || pack_value) {
            for (size_t t = 0; t < s.tokens; ++t) {
                if (pack_key) std::memcpy(key + t * s.dimension, source_key + t * k->nb[1], s.dimension * 2);
                if (pack_value) std::memcpy(value + t * s.dimension, source_value + t * v->nb[1], s.dimension * 2);
            }
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
            job.key = pack_key ? uintptr_t(key) : uintptr_t(source_key);
            job.value = pack_value ? uintptr_t(value) : uintptr_t(source_value);
            job.mask = uintptr_t(mask);
            job.output = uintptr_t(work.output.get() + first * s.dimension);
            const long rc = ara_resident_linux_execute(fd, &job);
            if (rc) {
                // Drop stale/error sessions, but never replay this operation.
                if (session) session->reset();
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
        std::memcpy(static_cast<char *>(dst->data) + h * dst->nb[1], work.output.get() + h * s.dimension, s.dimension * sizeof(float));
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
