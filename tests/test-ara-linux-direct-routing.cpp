// Host routing only: real GGML, SDK plans and admission; no native instructions.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "arch/riscv/akv.h"
#include "arch/riscv/linux-direct.h"
#include "arch/riscv/quant-kv.h"
#include "resident_attention/executor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#ifdef NDEBUG
#error "Routing checks require assertions"
#endif

static bool measuring, leased;
static size_t allocations, lease_allocations, lease_frees;
static unsigned attention_calls;

static void allocation() {
    if (measuring) ++allocations;
    if (leased) ++lease_allocations;
}

extern "C" {
void * __real_malloc(size_t);
void * __real_calloc(size_t, size_t);
void * __real_realloc(void *, size_t);
void * __real_aligned_alloc(size_t, size_t);
int __real_posix_memalign(void **, size_t, size_t);
void __real_free(void *);
void * __wrap_malloc(size_t n) { allocation(); return __real_malloc(n); }
void * __wrap_calloc(size_t n, size_t s) { allocation(); return __real_calloc(n, s); }
void * __wrap_realloc(void * p, size_t n) { allocation(); return __real_realloc(p, n); }
void * __wrap_aligned_alloc(size_t a, size_t n) { allocation(); return __real_aligned_alloc(a, n); }
int __wrap_posix_memalign(void ** p, size_t a, size_t n) { allocation(); return __real_posix_memalign(p, a, n); }
void __wrap_free(void * p) { if (p && leased) ++lease_frees; __real_free(p); }
int __wrap_fprintf(FILE * stream, const char * format, ...) {
    assert(!leased);
    va_list args;
    va_start(args, format);
    const int result = std::vfprintf(stream, format, args);
    va_end(args);
    return result;
}
}

// libstdc++'s own malloc calls bypass --wrap. Intercept C++ allocation too.
void * operator new(size_t n) {
    allocation();
    if (void * p = __real_malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void * operator new(size_t n, std::align_val_t a) {
    allocation();
    void * p = nullptr;
    if (!__real_posix_memalign(&p, size_t(a), n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void * p) noexcept { __wrap_free(p); }
void operator delete[](void * p) noexcept { __wrap_free(p); }
void operator delete(void * p, size_t) noexcept { __wrap_free(p); }
void operator delete[](void * p, size_t) noexcept { __wrap_free(p); }
void operator delete(void * p, std::align_val_t) noexcept { __wrap_free(p); }
void operator delete[](void * p, std::align_val_t) noexcept { __wrap_free(p); }
void operator delete(void * p, size_t, std::align_val_t) noexcept { __wrap_free(p); }
void operator delete[](void * p, size_t, std::align_val_t) noexcept { __wrap_free(p); }

struct terminal_error { char message[256]; };
extern "C" [[noreturn]] void __wrap_ggml_abort(const char *, int, const char * format, ...) {
    terminal_error error{};
    va_list args;
    va_start(args, format);
    std::vsnprintf(error.message, sizeof(error.message), format, args);
    va_end(args);
    throw error;
}

template <typename T> struct buffer {
    T * data = nullptr;
    size_t size;
    explicit buffer(size_t n) : size(n) {
        assert(!posix_memalign(reinterpret_cast<void **>(&data), 64, n * sizeof(T)));
        std::fill(data, data + size, T{});
    }
    ~buffer() { std::free(data); }
    buffer(const buffer &) = delete;
    buffer & operator=(const buffer &) = delete;
};

struct fixture {
    unsigned dim, rows, kv_heads, heads, tokens, capacity, query_tokens;
    size_t qrow, krow, vrow, outrow, outtoken;
    buffer<float> queries, outputs;
    buffer<ggml_fp16_t> keys, values, masks;
    ggml_compute_params params{};
    ggml_tensor q{}, k{}, v{}, mask{}, dst{};

    static void tensor(ggml_tensor & t, ggml_type type, void * data,
                       int64_t d, int64_t n, int64_t h, size_t scalar, size_t row, size_t head) {
        t.type = type; t.data = data;
        t.ne[0] = d; t.ne[1] = n; t.ne[2] = h; t.ne[3] = 1;
        t.nb[0] = scalar; t.nb[1] = row; t.nb[2] = head; t.nb[3] = h * head;
    }

    fixture(unsigned d = 128, unsigned r = 2, unsigned kv = 2, unsigned t = 129, unsigned m = 1) :
        dim(d), rows(r), kv_heads(kv), heads(r * kv), tokens(t), capacity(t + m + 7), query_tokens(m),
        qrow(d + 8), krow(d + 16), vrow(d + 32), outrow(m == 1 ? d + 12 : d),
        outtoken(heads * outrow + 16), queries(heads * m * qrow), outputs(m * outtoken),
        keys(kv * capacity * krow), values(kv * capacity * vrow), masks(m * capacity) {
        params.nth = 1;
        tensor(q, GGML_TYPE_F32, queries.data, d, m, heads, 4, qrow * 4, m * qrow * 4);
        tensor(k, GGML_TYPE_F16, keys.data, d, capacity, kv, 2, krow * 2, capacity * krow * 2);
        tensor(v, GGML_TYPE_F16, values.data, d, capacity, kv, 2, vrow * 2, capacity * vrow * 2);
        tensor(mask, GGML_TYPE_F16, masks.data, capacity, m, 1, 2, capacity * 2, m * capacity * 2);
        tensor(dst, GGML_TYPE_F32, outputs.data, d, heads, m, 4, outrow * 4, outtoken * 4);
        dst.op = GGML_OP_FLASH_ATTN_EXT;
        dst.src[0] = &q; dst.src[1] = &k; dst.src[2] = &v; dst.src[3] = &mask;
        const float scale = 0.125f;
        std::memcpy(dst.op_params, &scale, sizeof(scale));
        std::fill(outputs.data, outputs.data + outputs.size, -99.0f);
        std::fill(masks.data, masks.data + masks.size, ggml_fp16_t(0xfc00));
        for (unsigned h = 0; h < heads; ++h)
            for (unsigned i = 0; i < m; ++i)
                for (unsigned x = 0; x < d; ++x)
                    queries.data[(h * m + i) * qrow + x] = float(1 + (h + i + x) % 17) / 64;
        for (unsigned h = 0; h < kv; ++h)
            for (unsigned i = 0; i < capacity; ++i)
                for (unsigned x = 0; x < d; ++x) {
                    keys.data[(h * capacity + i) * krow + x] = ggml_fp32_to_fp16(float((h + i + x) % 9) / 64);
                    values.data[(h * capacity + i) * vrow + x] = ggml_fp32_to_fp16(value(h, x));
                }
        for (unsigned i = 0; i < m; ++i)
            std::fill(masks.data + i * capacity, masks.data + i * capacity + t + i, ggml_fp16_t(0));
    }

    static float value(unsigned kv, unsigned x) { return float(1 + kv + x % 7) / 32; }
    static float marker(unsigned h, unsigned x) { return float(1000 + 256 * h + x); }
    unsigned groups() const { return kv_heads * ((rows + AKV_MAX_Q_ROWS - 1) / AKV_MAX_Q_ROWS); }
    void unchanged() const { for (size_t i = 0; i < outputs.size; ++i) assert(outputs.data[i] == -99.0f); }
    void check_output(bool reference = false) const {
        for (unsigned token = 0; token < query_tokens; ++token) {
            for (size_t i = 0; i < outtoken; ++i) {
                const unsigned head = i / outrow, x = i % outrow;
                const float actual = outputs.data[token * outtoken + i];
                if (head >= heads || x >= dim) assert(actual == -99.0f);
                else if (reference) assert(std::fabs(actual - value(head / rows, x)) < 0.003f);
                else assert(actual == marker(head, x));
            }
        }
    }
};

static uint64_t resident_info_word() {
    return akv_resident_qk_capability_word(AKV_RESIDENT_CAPABILITY_INDEX, 1) |
           akv_resident_pv_capability_word(AKV_RESIDENT_CAPABILITY_INDEX, 1);
}

struct mock_state {
    fixture * f = nullptr;
    bool available = true, prepare_ok = true, begin_ok = true, release_ok = true;
    bool legacy_ok = true, acquire_ok = true, reference = false, poisoned = false;
    unsigned available_calls = 0, prepares = 0, begins = 0, ends = 0;
    unsigned infos = 0, reads = 0, acquires = 0, releases = 0, current_calls = 0;
    unsigned resident = 0, native = 0, features = 0, prefill = 0, quant = 0, executed = 0;
    unsigned group_tiles = 0, group_tails = 0, expected_groups = 0;
    uintptr_t vstart = 0, frm = 0;
    uint64_t info4 = resident_info_word();
    bool provider_owned = false;
    const void * scratch = nullptr;
    size_t scratch_bytes = 0;
    ggml_akv_direct::span spans[ggml_akv_direct::max_spans]{};
    size_t span_count = 0;
} mock;

static bool covered(const void * base, size_t bytes, bool writable) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(base);
    for (size_t i = 0; i < mock.span_count; ++i) {
        const auto & s = mock.spans[i];
        const uintptr_t first = reinterpret_cast<uintptr_t>(s.base);
        if ((!writable || s.writable) && address >= first && address - first <= s.bytes &&
            bytes <= s.bytes - (address - first)) return true;
    }
    return false;
}

static akv_platform_result_t read_state(void *, akv_platform_thread_state_v1 * state) {
    assert(leased);
    ++mock.reads;
    *state = {128, mock.vstart, (mock.frm << 5) | 8, 5};
    return AKV_PLATFORM_READY;
}
static uintptr_t acquire(void *, const akv_platform_request_v1 * request) {
    assert(leased && !mock.provider_owned && mock.executed == 0);
    assert(request->workers == 1 && request->groups == mock.expected_groups);
    assert(request->query_tokens == mock.f->query_tokens);
    for (size_t i = 0; i < request->span_count; ++i) {
        const auto & span = request->spans[i];
        assert(covered(span.base, span.bytes, span.writable));
    }
    ++mock.acquires;
    if (!mock.acquire_ok) return 0;
    mock.provider_owned = true;
    return 17;
}
static void release(void *, uintptr_t cookie) {
    assert(leased && mock.provider_owned && cookie == 17);
    ++mock.releases;
    mock.provider_owned = false;
}

static const akv_resident_platform_provider_v1 runtime_provider = {
    1, sizeof(runtime_provider), AKV_PLATFORM_VALIDATED_TEST_MODEL, AKV_PLATFORM_REQUIRED_ASSURANCES,
    {1}, AKV_PLATFORM_QK_PV_V1, 1, 1, 4, 1024, nullptr, read_state, acquire, release,
};
static const ggml_akv_legacy_contract_v1 legacy = {
    1, sizeof(legacy), &runtime_provider, {1}, GGML_AKV_LEGACY_BASE | GGML_AKV_LEGACY_PREFILL | GGML_AKV_LEGACY_QUANT_KV, 1,
};
extern "C" const ggml_akv_legacy_contract_v1 * ggml_akv_legacy_current_v1() {
    return leased && mock.legacy_ok ? &legacy : nullptr;
}
extern "C" const akv_resident_platform_provider_v1 * akv_resident_platform_current_v1() {
    ++mock.current_calls;
    return leased ? &runtime_provider : nullptr;
}

namespace ggml_akv_direct {
bool active() { return leased; }
bool available() { assert(!leased); ++mock.available_calls; return mock.available && !mock.poisoned; }
bool prepare(const span * spans, size_t count) {
    assert(!leased && mock.available_calls && count <= max_spans);
    ++mock.prepares;
    mock.span_count = count;
    std::copy(spans, spans + count, mock.spans);
    const auto & f = *mock.f;
    for (const auto * t : {&f.q, &f.k, &f.v, &f.mask, &f.dst})
        assert(covered(t->data, ggml_nbytes(t), t == &f.dst));
    if (f.query_tokens == 1) {
        assert(count == 6 && spans[count - 1].writable);
        mock.scratch = spans[count - 1].base;
        mock.scratch_bytes = spans[count - 1].bytes;
        assert(mock.scratch && mock.scratch_bytes);
    }
    return mock.prepare_ok;
}
bool begin() {
    assert(!leased && mock.prepares == 1);
    ++mock.begins;
    leased = mock.begin_ok;
    return leased;
}
bool end() {
    assert(leased && !mock.provider_owned);
    ++mock.ends;
    leased = false;
    if (!mock.release_ok) mock.poisoned = true;
    return mock.release_ok;
}
const akv_resident_platform_provider_v1 * provider() { return leased ? &runtime_provider : nullptr; }
const ggml_akv_legacy_contract_v1 * legacy_contract() { return ggml_akv_legacy_current_v1(); }
statistics stats() { return {}; }
const char * last_error() { return nullptr; }
}

extern "C" uint64_t akv_native_info(void *, unsigned index) {
    assert(leased && mock.begins == 1 && !mock.vstart && !mock.frm);
    ++mock.infos;
    if (index == AKV_RESIDENT_CAPABILITY_INDEX) return mock.info4;
    return index < 2 ? akv_capability_word(index, 1) : akv_v2_capability_word(index, 1);
}

static akv_status_t execute_plan(const akv_attention_plan_t * plan, akv_attention_v2_workspace_t * workspace,
                                 const akv_attention_features_t * features) {
    assert(leased && mock.provider_owned && mock.acquires == 1);
    assert(!lease_allocations && !lease_frees && akv_attention_plan_v2_is_valid(plan));
    const auto & f = *mock.f;
    const auto & d = plan->descriptor;
    const unsigned per_kv = (f.rows + AKV_MAX_Q_ROWS - 1) / AKV_MAX_Q_ROWS;
    const unsigned kv = mock.executed / per_kv;
    const unsigned start = (mock.executed % per_kv) * AKV_MAX_Q_ROWS;
    const unsigned first = kv * f.rows + start;
    assert(kv < f.kv_heads && d.q_rows == std::min(f.rows - start, unsigned(AKV_MAX_Q_ROWS)));
    assert(d.head_dim == f.dim && d.kv_length == f.tokens && plan->scale == 0.125f);
    assert(d.k_base == reinterpret_cast<uintptr_t>(f.keys.data) + kv * f.k.nb[2]);
    assert(d.v_base == reinterpret_cast<uintptr_t>(f.values.data) + kv * f.v.nb[2]);
    assert(d.k_token_stride_bytes == f.k.nb[1] && d.v_token_stride_bytes == f.v.nb[1]);
    assert(plan->mask == f.masks.data && plan->output == f.outputs.data + first * f.outrow);
    assert(plan->output_row_stride_bytes == f.dst.nb[1]);
    assert(covered(plan, sizeof(*plan), true) && covered(workspace, sizeof(*workspace), true));
    assert(covered(reinterpret_cast<const void *>(uintptr_t(d.q_base)), d.q_rows * f.dim * 2, true));
    for (unsigned h = 0; h < d.q_rows; ++h) {
        const auto * query = reinterpret_cast<const ggml_fp16_t *>(uintptr_t(d.q_base) + h * d.q_row_stride_bytes);
        for (unsigned x = 0; x < f.dim; ++x)
            assert(query[x] == ggml_fp32_to_fp16(f.queries.data[(first + h) * f.qrow + x]));
    }
    mock.group_tiles += (d.kv_length + AKV_V2_TILE_TOKENS - 1) / AKV_V2_TILE_TOKENS;
    mock.group_tails += d.kv_length % AKV_V2_TILE_TOKENS != 0;
    ++mock.executed;
    if (mock.reference) return akv_attention_execute_v2_with_features_reference(plan, features);
    for (unsigned h = 0; h < d.q_rows; ++h)
        for (unsigned x = 0; x < f.dim; ++x) f.outputs.data[(first + h) * f.outrow + x] = fixture::marker(first + h, x);
    return AKV_STATUS_OK;
}

extern "C" akv_status_t akv_experimental_attention_execute_v2_native(const akv_device_t * device,
    const akv_attention_plan_t * plan, akv_attention_v2_workspace_t * workspace,
    const akv_attention_features_t * features, const akv_resident_attention_options_t * options, unsigned automatic) {
    const auto & caps = device->capabilities;
    assert(caps.resident_valid && caps.resident_capability_version == 1 &&
           caps.resident_pv_supported && caps.resident_qk_supported &&
           caps.resident_nr_lanes == 4 && caps.resident_vlen_bits == 1024);
    assert(akv_resident_attention_select(device, plan, options, 0) == AKV_RESIDENT_USE_SHARED);
    assert(automatic == unsigned(std::getenv("AKV_PAIR_HEADS") == nullptr));
    ++mock.resident;
    return execute_plan(plan, workspace, features);
}
extern "C" akv_status_t akv_attention_execute_v2_native(const akv_attention_plan_t * p, akv_attention_v2_workspace_t * w) {
    assert(std::getenv("AKV_PAIR_HEADS")); ++mock.native; return execute_plan(p, w, nullptr);
}
extern "C" akv_status_t akv_attention_execute_v2_native_auto(const akv_device_t *, const akv_attention_plan_t * p, akv_attention_v2_workspace_t * w) {
    assert(!std::getenv("AKV_PAIR_HEADS")); ++mock.native; return execute_plan(p, w, nullptr);
}
extern "C" akv_status_t akv_attention_execute_v2_with_features_native(const akv_attention_plan_t * p,
    akv_attention_v2_workspace_t * w, const akv_attention_features_t * f) {
    assert(std::getenv("AKV_PAIR_HEADS")); ++mock.features; return execute_plan(p, w, f);
}
extern "C" akv_status_t akv_attention_execute_v2_with_features_native_auto(const akv_device_t *, const akv_attention_plan_t * p,
    akv_attention_v2_workspace_t * w, const akv_attention_features_t * f) {
    assert(!std::getenv("AKV_PAIR_HEADS")); ++mock.features; return execute_plan(p, w, f);
}
extern "C" akv_status_t akv_attention_execute_native(const akv_attention_plan_t *) {
    assert(false && "v1 is not selected by these tests"); return AKV_STATUS_BAD_ARGUMENT;
}
extern "C" akv_status_t akv_attention_execute_v2_prefill_native(const akv_device_t * device,
    const akv_attention_v2_prefill_problem_t * p, akv_attention_v2_prefill_workspace_t * w) {
    assert(leased && mock.provider_owned && mock.acquires == 1 && !mock.resident && !mock.native);
    const auto & f = *mock.f;
    assert(p->query == f.queries.data && p->key == f.keys.data && p->value == f.values.data);
    assert(p->mask == f.masks.data && p->output == f.outputs.data);
    assert(p->query_tokens == f.query_tokens && p->query_heads == f.heads && p->kv_heads == f.kv_heads);
    assert(p->head_dim == f.dim && p->kv_capacity == f.capacity);
    ++mock.prefill;
    return akv_attention_execute_v2_prefill_reference(device, p, w);
}
ggml_riscv_quant_kv_status ggml_riscv_quant_kv_flash_attn(const ggml_compute_params *, ggml_tensor * dst) {
    // The legacy F16 prefill path also asks this helper; its type gate declines.
    if (dst->src[1]->type != GGML_TYPE_Q4_0 && dst->src[1]->type != GGML_TYPE_Q8_0)
        return ggml_riscv_quant_kv_status::fallback;
    // Quantized KV has its own admission; the Direct-U F16 lease must be absent.
    assert(!leased && !mock.prepares && !mock.begins && !mock.infos);
    assert(dst->src[1]->type == GGML_TYPE_Q4_0 || dst->src[1]->type == GGML_TYPE_Q8_0);
    ++mock.quant;
    return ggml_riscv_quant_kv_status::fallback;
}

static void reset(fixture & f) {
    assert(!leased && !measuring);
    mock = {};
    mock.f = &f;
    mock.expected_groups = f.query_tokens == 1 ? f.groups() : f.kv_heads;
    allocations = lease_allocations = lease_frees = 0;
    setenv("GGML_RISCV_AKV", "1", 1);
    setenv("GGML_RISCV_AKV_EMULATE", "0", 1);
    setenv("GGML_RISCV_ACCEL_AUTO", "1", 1);
    setenv("GGML_RISCV_AKV_KERNEL", "v2", 1);
    setenv("GGML_RISCV_AKV_RESIDENT", "1", 1);
    setenv("GGML_RISCV_AKV_PORTABLE", "1", 1);
    setenv("GGML_RISCV_AKV_QUANT_KV", "1", 1);
    setenv("GGML_RISCV_AKV_TRACE", "0", 1);
    unsetenv("AKV_PAIR_HEADS");
}
static bool run(fixture & f) {
    ++attention_calls;
    measuring = true;
    try {
        const bool result = ggml_riscv_akv_flash_attn(&f.params, &f.dst);
        measuring = false;
        assert(!leased && !mock.provider_owned && !lease_allocations && !lease_frees);
        return result;
    } catch (...) { measuring = false; throw; }
}
static void no_execution(const fixture & f) {
    assert(!mock.infos && !mock.executed && !mock.prefill && !mock.resident && !mock.native && !mock.features);
    f.unchanged();
}
static void successful(fixture & f, bool resident, bool features = false) {
    ggml_akv_resident_counters_v1 before{}, after{};
    ggml_akv_resident_snapshot_v1(&before);
    assert(run(f));
    ggml_akv_resident_snapshot_v1(&after);
    const auto groups = f.groups();
    assert(mock.prepares == 1 && mock.begins == 1 && mock.ends == 1);
    assert(mock.acquires == 1 && mock.releases == 1 && mock.executed == groups);
    assert(mock.resident == (resident ? groups : 0));
    assert(mock.features == (features ? groups : 0));
    assert(mock.native == (!resident && !features ? groups : 0));
    assert(mock.group_tiles == groups * ((f.tokens + AKV_V2_TILE_TOKENS - 1) / AKV_V2_TILE_TOKENS));
    assert(mock.group_tails == groups * unsigned(f.tokens % AKV_V2_TILE_TOKENS != 0));
    assert(after.completed - before.completed == 1 && after.failed == before.failed);
    assert(after.resident_groups - before.resident_groups == (resident ? groups : 0));
    assert(after.native_groups - before.native_groups == (resident ? 0 : groups));
    f.check_output(mock.reference);
}

static void unsupported() {
    for (unsigned mutation = 0; mutation < 13; ++mutation) {
        fixture f; reset(f);
        switch (mutation) {
            case 0: f.params.nth = 2; break;
            case 1: f.params.ith = 1; break;
            case 2: f.params.use_ref = true; break;
            case 3: f.dst.src[0] = nullptr; break;
            case 4: f.dst.src[3] = nullptr; break;
            case 5: f.q.type = GGML_TYPE_F16; break;
            case 6: f.k.type = GGML_TYPE_F32; break;
            case 7: f.v.type = GGML_TYPE_F32; break;
            case 8: f.q.ne[1] = 0; break;
            case 9: f.q.ne[2] = 0; break;
            case 10: f.q.ne[2] = 4097; break;
            case 11: setenv("GGML_RISCV_AKV", "0", 1); break;
            case 12: f.params.nth = 0; break;
        }
        assert(!run(f) && !mock.prepares && !mock.begins && !mock.acquires);
        no_execution(f);
    }
}
static void no_device() {
    fixture f; reset(f); mock.available = false;
    assert(!run(f) && mock.available_calls == 1 && !mock.prepares && !mock.begins);
    assert(!allocations); no_execution(f);
}
static void preparation_failures(bool prepare) {
    fixture f; reset(f);
    if (prepare) mock.prepare_ok = false; else mock.begin_ok = false;
    assert(!run(f) && mock.prepares == 1 && mock.begins == unsigned(!prepare) && !mock.ends);
    assert(!mock.acquires); no_execution(f);
}
static void matrix() {
    for (unsigned dim : {64, 96, 128})
        for (unsigned rows : {1, 2, 3, 6, 8, 9, 17})
            for (unsigned tokens : {127, 128, 129, 257}) {
                fixture f(dim, rows, rows % 2 + 2, tokens); reset(f); successful(f, true);
            }
}
static void policy() {
    struct test_case { unsigned d, r, t; const char * setting; bool shared; };
    for (const auto & c : {test_case{128, 2, 128, nullptr, true}, {128, 2, 256, "auto", true},
            {128, 2, 127, nullptr, false}, {128, 2, 257, nullptr, false},
            {96, 2, 129, nullptr, false}, {128, 3, 129, nullptr, false},
            {128, 2, 129, "0", false}, {64, 3, 129, "1", true}}) {
        for (bool explicit_pair : {false, true}) {
            fixture f(c.d, c.r, 2, c.t); reset(f);
            if (c.setting) setenv("GGML_RISCV_AKV_RESIDENT", c.setting, 1);
            else unsetenv("GGML_RISCV_AKV_RESIDENT");
            if (explicit_pair) setenv("AKV_PAIR_HEADS", "0", 1);
            successful(f, c.shared);
        }
    }
}
static bool resident_capability_fallback(const char * name) {
    const uint64_t valid = resident_info_word();
    struct test_case { const char * name; uint64_t word; akv_status_t status; };
    const test_case cases[] = {
        {"info4-absent", 0, AKV_STATUS_RUNTIME_UNAVAILABLE},
        {"info4-pv-only", akv_resident_pv_capability_word(AKV_RESIDENT_CAPABILITY_INDEX, 1), AKV_STATUS_OK},
        {"info4-qk-only", akv_resident_qk_capability_word(AKV_RESIDENT_CAPABILITY_INDEX, 1), AKV_STATUS_ABI_MISMATCH},
        {"info4-version", (valid & ~(UINT64_C(0xff) << AKV_RESIDENT_CAPABILITY_VERSION_SHIFT)) |
                          (UINT64_C(2) << AKV_RESIDENT_CAPABILITY_VERSION_SHIFT), AKV_STATUS_ABI_MISMATCH},
        {"info4-lanes", (valid & ~(UINT64_C(0xff) << AKV_RESIDENT_CAPABILITY_NR_LANES_SHIFT)) |
                        (UINT64_C(8) << AKV_RESIDENT_CAPABILITY_NR_LANES_SHIFT), AKV_STATUS_ABI_MISMATCH},
        {"info4-vlen", (valid & ~(UINT64_C(0xffff) << AKV_RESIDENT_CAPABILITY_VLEN_SHIFT)) |
                       (UINT64_C(512) << AKV_RESIDENT_CAPABILITY_VLEN_SHIFT), AKV_STATUS_ABI_MISMATCH},
        {"info4-reserved", valid | (UINT64_C(1) << 63), AKV_STATUS_ABI_MISMATCH},
    };
    for (const auto & c : cases) {
        if (std::strcmp(name, c.name)) continue;
        // SDK parsing must preserve native capabilities when INFO4 is unusable.
        akv_device_t device{};
        assert(akv_device_init_reference(&device) == AKV_STATUS_OK);
        auto & caps = device.capabilities;
        assert(akv_capabilities_decode_resident(c.word, &caps) == c.status);
        assert(caps.valid && caps.enabled && caps.token_axis_valid && caps.token_axis_enabled);
        assert(c.status == AKV_STATUS_OK ? caps.resident_valid && !caps.resident_qk_supported : !caps.resident_valid);
        unsigned call = 0;
        for (bool explicit_request : {false, true}) {
            for (bool explicit_pair : {false, true}) {
                fixture f(128, 2, 3, 129); reset(f); mock.info4 = c.word;
                if (!explicit_request) unsetenv("GGML_RISCV_AKV_RESIDENT");
                if (explicit_pair) setenv("AKV_PAIR_HEADS", "0", 1);
                successful(f, false);
                assert(mock.infos == (call++ == 0 ? AKV_RESIDENT_CAPABILITY_INDEX + 1u : 0u));
            }
        }
        return true;
    }
    return false;
}
static void reuse() {
    fixture f(96, 9, 3, 129); reset(f); successful(f, true);
    assert(allocations > 0);
    const auto * scratch = mock.scratch;
    const auto bytes = mock.scratch_bytes;
    std::fill(f.outputs.data, f.outputs.data + f.outputs.size, -99.0f);
    reset(f); setenv("GGML_RISCV_AKV_TRACE", "1", 1); successful(f, true);
    assert(!allocations && mock.scratch == scratch && mock.scratch_bytes == bytes && !mock.infos);
}
static void features() {
    for (bool explicit_pair : {false, true}) {
        fixture f(96, 9, 2, 129); reset(f);
        f.masks.data[3] = ggml_fp32_to_fp16(0.5f);
        if (explicit_pair) setenv("AKV_PAIR_HEADS", "0", 1);
        successful(f, false, true);
    }
}
static void reference_routes() {
    for (bool resident : {false, true}) {
        fixture f(64, 3, 2, 9); reset(f); mock.reference = true;
        setenv("GGML_RISCV_AKV_RESIDENT", resident ? "1" : "0", 1);
        successful(f, resident);
    }
}
static void prefill() {
    for (unsigned dim : {64, 96, 128}) {
        fixture f(dim, dim == 96 ? 3 : 2, 2, 5, AKV_PREFILL_QUERY_BLOCK_TOKENS + 1); reset(f);
        setenv("GGML_RISCV_AKV_TRACE", "1", 1);
        assert(run(f) && mock.prefill == 1 && !mock.executed && !mock.resident);
        assert(mock.begins == 1 && mock.ends == 1 && mock.acquires == 1 && mock.releases == 1);
        f.check_output(true);
    }
}
static void legacy_required() {
    fixture f; reset(f); mock.legacy_ok = false;
    assert(!run(f) && mock.begins == 1 && mock.ends == 1 && !mock.acquires);
    no_execution(f);
}
static void admission_denied() {
    fixture f; reset(f); mock.acquire_ok = false;
    assert(!run(f) && mock.acquires == 1 && !mock.releases && !mock.executed);
    assert(mock.begins == 1 && mock.ends == 1); f.unchanged();
}
static void state_denied() {
    fixture f; reset(f); mock.frm = 1;
    assert(!run(f) && mock.begins == 1 && mock.ends == 1); no_execution(f);
    reset(f); mock.vstart = 1;
    bool terminal = false;
    try { (void)run(f); } catch (const terminal_error &) { terminal = true; }
    assert(terminal && !leased && mock.ends == 1 && !mock.acquires); no_execution(f);
}
static void failed_release() {
    fixture f; reset(f); mock.release_ok = false;
    bool terminal = false;
    try { (void)run(f); } catch (const terminal_error & e) {
        terminal = true; assert(std::strstr(e.message, "release failed"));
    }
    assert(terminal && !leased && mock.poisoned && mock.ends == 1 && mock.releases == 1);
    assert(mock.executed == f.groups() && !mock.native && !mock.prefill && !mock.quant);
    f.check_output();
    const auto before = mock.executed;
    assert(!run(f) && mock.executed == before && mock.begins == 1 && mock.ends == 1);
}
static void quant_fallback() {
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        fixture f; reset(f); f.k.type = f.v.type = type;
        assert(!run(f) && mock.quant == 1 && !mock.begins && !mock.prepares && !mock.acquires);
        no_execution(f);
        reset(f); setenv("GGML_RISCV_AKV_QUANT_KV", "0", 1);
        assert(!run(f) && !mock.quant && !mock.begins); no_execution(f);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s CASE\n", argv[0]); return 2; }
    // Initialize real GGML's conversion tables before measuring the route.
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    assert(ctx); ggml_free(ctx);
    const char * name = argv[1];
    if (!std::strcmp(name, "unsupported")) unsupported();
    else if (!std::strcmp(name, "no-device")) no_device();
    else if (!std::strcmp(name, "prepare-failure")) preparation_failures(true);
    else if (!std::strcmp(name, "begin-failure")) preparation_failures(false);
    else if (!std::strcmp(name, "matrix")) matrix();
    else if (!std::strcmp(name, "policy")) policy();
    else if (!std::strcmp(name, "reuse")) reuse();
    else if (!std::strcmp(name, "features")) features();
    else if (!std::strcmp(name, "reference")) reference_routes();
    else if (!std::strcmp(name, "prefill")) prefill();
    else if (!std::strcmp(name, "legacy-required")) legacy_required();
    else if (!std::strcmp(name, "admission-denied")) admission_denied();
    else if (!std::strcmp(name, "state-denied")) state_denied();
    else if (!std::strcmp(name, "failed-release")) failed_release();
    else if (!std::strcmp(name, "quant-fallback")) quant_fallback();
    else if (!resident_capability_fallback(name)) { std::fprintf(stderr, "unknown case: %s\n", name); return 2; }
    assert(!leased && !mock.provider_owned && !lease_allocations && !lease_frees);
    ggml_akv_resident_counters_v1 counts{};
    ggml_akv_resident_snapshot_v1(&counts);
    std::printf("{\"case\":\"%s\",\"attention_calls\":%u,\"resident_groups\":%llu,\"native_groups\":%llu}\n",
                name, attention_calls, static_cast<unsigned long long>(counts.resident_groups),
                static_cast<unsigned long long>(counts.native_groups));
    std::printf("DIRECT_ROUTING_PASS case=%s native_execution=0 board_run=0\n", name);
}
