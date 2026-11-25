#include "linux-direct.h"
#include "direct_client.h"

#include <asm/unistd.h>
#include <fcntl.h>
#include <linux/mman.h>
#include <stdint.h>

#ifndef GGML_AKV_DIRECT_CONTRACT_SHA256
#define GGML_AKV_DIRECT_CONTRACT_SHA256 ""
#endif

// Callbacks must remain scalar even when the rest of GGML uses RVV.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize ("no-tree-vectorize", "no-tree-slp-vectorize", "no-tree-loop-distribute-patterns")
#endif

namespace ggml_akv_direct {
namespace {

#ifdef GGML_AKV_LINUX_DIRECT_TEST
extern long test_syscall(long, long, long, long);
extern void test_read_state(akv_platform_thread_state_v1 *);
extern void test_restore_state(const akv_platform_thread_state_v1 *);
extern void test_init_state();
extern void test_touch(const span &);
extern void test_stack(uintptr_t *, uintptr_t *);
extern void test_fatal();
#endif

static long call(long nr, long x = 0, long y = 0, long z = 0) {
#ifdef GGML_AKV_LINUX_DIRECT_TEST
    return test_syscall(nr, x, y, z);
#elif defined(__riscv) && __riscv_xlen == 64
    register long a0 asm("a0") = x;
    register long a1 asm("a1") = y;
    register long a2 asm("a2") = z;
    register long a3 asm("a3") = 0;
    register long a4 asm("a4") = 0;
    register long a5 asm("a5") = 0;
    register long a7 asm("a7") = nr;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a7) : "memory");
    return a0;
#else
#error "Linux Direct-U requires RV64 or GGML_AKV_LINUX_DIRECT_TEST"
#endif
}

[[noreturn]] static void fatal() {
#ifdef GGML_AKV_LINUX_DIRECT_TEST
    test_fatal();
#else
    call(__NR_exit_group, 127);
#endif
    __builtin_trap();
}

static void read_state(akv_platform_thread_state_v1 & out) {
#ifdef GGML_AKV_LINUX_DIRECT_TEST
    test_read_state(&out);
#else
    asm volatile("csrr %0,vstart; csrr %1,vlenb; csrr %2,fcsr; csrr %3,vcsr"
                 : "=r"(out.vstart), "=r"(out.vlenb), "=r"(out.fcsr), "=r"(out.vcsr) : : "memory");
#endif
}

static void restore_state(const akv_platform_thread_state_v1 & in) {
#ifdef GGML_AKV_LINUX_DIRECT_TEST
    test_restore_state(&in);
#else
    asm volatile("csrw fcsr,%0; csrw vcsr,%1; csrw vstart,%2"
                 : : "r"(in.fcsr), "r"(in.vcsr), "r"(in.vstart) : "memory");
#endif
}

static bool valid_state(const akv_platform_thread_state_v1 & in) {
    if (in.vstart) fatal();
    return in.vlenb == 128 && !(in.fcsr >> 5) && in.vcsr <= 7;
}

struct runtime {
    int fd = -1;
    long pid = 0, tid = 0;
    bool prepared = false, leased = false, locked = false, csr_safe = false, checked = false;
    uintptr_t cookie = 0, next_cookie = 0;
    uint64_t generation = 0;
    span spans[max_spans];
    size_t count = 0;
    uintptr_t stack_low = 0, stack_high = 0;
};
static runtime state{};
static statistics counts;
static const char * error = "not prepared";
static bool decline(const char * reason) { error = reason; return false; }

static bool equal(const char * a, const char * b) {
    while (*a && *a == *b) { ++a; ++b; }
    return *a == *b;
}

static bool prefix(const char * s, const char * p) {
    while (*p) if (*s++ != *p++) return false;
    return true;
}

static int digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool number(const char * & p, uintptr_t & value, unsigned radix) {
    value = 0;
    bool any = false;
    for (int d; (d = digit(*p)) >= 0 && unsigned(d) < radix; ++p) {
        if (value > (UINTPTR_MAX - unsigned(d)) / radix) return false;
        value = value * radix + unsigned(d);
        any = true;
    }
    return any;
}

static void spaces(const char * & p) { while (*p == ' ' || *p == '\t') ++p; }

struct lines {
    int fd;
    char bytes[4096];
    size_t pos = 0, size = 0;
    bool failed = false;

    explicit lines(const char * path) : fd(int(call(__NR_openat, AT_FDCWD, long(path), O_RDONLY | O_CLOEXEC))) {}
    ~lines() { if (fd >= 0) call(__NR_close, fd); }
    int get() {
        if (pos == size) {
            const long n = call(__NR_read, fd, long(bytes), sizeof(bytes));
            if (n <= 0) { failed = n < 0; return -1; }
            pos = 0; size = size_t(n);
        }
        return static_cast<unsigned char>(bytes[pos++]);
    }
    bool next(char * out, size_t capacity) {
        size_t n = 0;
        for (int c; (c = get()) >= 0;) {
            if (c == '\n') { out[n] = 0; return true; }
            if (n + 1 >= capacity) { failed = true; return false; }
            out[n++] = char(c);
        }
        if (n) failed = true;
        return false;
    }
};

static bool single_thread() {
    lines input("/proc/self/status");
    if (input.fd < 0) return false;
    bool threads = false, tracer = false;
    char line[1024];
    while (input.next(line, sizeof(line))) {
        const char * p = nullptr;
        if (prefix(line, "Threads:")) p = line + 8;
        else if (prefix(line, "TracerPid:")) p = line + 10;
        else continue;
        spaces(p);
        uintptr_t n;
        if (!number(p, n, 10)) return false;
        spaces(p);
        if (*p) return false;
        if (line[0] == 'T' && line[1] == 'h') threads = n == 1;
        else tracer = n == 0;
    }
    return !input.failed && threads && tracer;
}

static akv_platform_result_t logical_read(void *, akv_platform_thread_state_v1 * out);
static uintptr_t logical_acquire(void *, const akv_platform_request_v1 *);
static void logical_release(void *, uintptr_t);

static akv_resident_platform_provider_v1 binding = {
    1, sizeof(binding), AKV_PLATFORM_EXTERNAL_DEPLOYMENT, AKV_PLATFORM_REQUIRED_ASSURANCES,
    {}, AKV_PLATFORM_QK_PV_V1, 1, 1, 4, 1024, &state, logical_read, logical_acquire, logical_release
};
static ggml_akv_legacy_contract_v1 legacy = {
    1, sizeof(legacy), &binding, {}, GGML_AKV_LEGACY_BASE | GGML_AKV_LEGACY_PREFILL, 1
};

static bool identity() {
    const char * p = GGML_AKV_DIRECT_CONTRACT_SHA256;
    size_t length = 0;
    while (p[length]) ++length;
    if (length != 64) return false;
    unsigned nonzero = 0;
    for (size_t i = 0; i < 32; ++i) {
        const int a = digit(p[2 * i]), b = digit(p[2 * i + 1]);
        if (a < 0 || b < 0) return false;
        binding.contract_sha256[i] = legacy.contract_sha256[i] = uint8_t(a * 16 + b);
        nonzero |= unsigned(a | b);
    }
    return nonzero != 0;
}

struct mapping {
    uintptr_t low = 0, high = 0;
    bool readable = false, writable = false, special = false, flags = false, locked = false;
};

struct inventory {
    static constexpr size_t capacity = 64 * 1024;
    static constexpr size_t max_mappings = 512;
    char saved[capacity], pending[capacity];
    size_t saved_size = 0, pending_size = 0, count = 0;
    mapping regions[max_mappings];
    bool valid = false, all_locked = false;
};
static inventory cache{};

static bool header(const char * p, mapping & m) {
    uintptr_t ignored;
    if (!number(p, m.low, 16) || *p++ != '-' || !number(p, m.high, 16) ||
        m.low >= m.high || *p++ != ' ') return false;
    spaces(p);
    m.readable = *p == 'r';
    if (*p != 'r' && *p != '-') return false;
    ++p; m.writable = *p == 'w';
    if (*p != 'w' && *p != '-') return false;
    ++p;
    if (*p != 'x' && *p != '-') return false;
    ++p;
    if (*p++ != 'p' || *p++ != ' ') return false; // No shared mappings.
    spaces(p);
    if (!number(p, ignored, 16) || *p++ != ' ') return false;
    spaces(p);
    if (!number(p, ignored, 16) || *p++ != ':' || !number(p, ignored, 16) || *p++ != ' ') return false;
    spaces(p);
    if (!number(p, ignored, 10)) return false;
    if (*p && *p != ' ') return false;
    spaces(p);
    m.special = equal(p, "[vvar]") || equal(p, "[vvar_vclock]") || equal(p, "[vdso]") || equal(p, "[vsyscall]");
    if (!m.special && *p && *p != '/' && !equal(p, "[heap]") && !equal(p, "[stack]") && !prefix(p, "[anon:")) return false;
    if (prefix(p, "/dev/") || prefix(p, "/sys/") || prefix(p, "/proc/")) return false;
    m.flags = false;
    return true;
}

static bool vm_flags(const char * p, mapping & m) {
    bool rd = false, wr = false, locked = false;
    while (*p) {
        spaces(p);
        if (!*p) break;
        if (!p[1] || (p[2] && p[2] != ' ')) return false;
        const unsigned token = (unsigned(p[0]) << 8) | unsigned(p[1]);
        rd |= token == ('r' << 8 | 'd');
        wr |= token == ('w' << 8 | 'r');
        locked |= token == ('l' << 8 | 'o');
        switch (token) {
            case 'i' << 8 | 'o': case 'p' << 8 | 'f': case 'm' << 8 | 'm':
            case 'h' << 8 | 't': case 's' << 8 | 'h': case 'u' << 8 | 'm': case 'u' << 8 | 'w':
                if (!m.special) return false;
                break;
            case 'r' << 8 | 'd': case 'w' << 8 | 'r': case 'e' << 8 | 'x':
            case 'm' << 8 | 'r': case 'm' << 8 | 'w': case 'm' << 8 | 'e': case 'm' << 8 | 's':
            case 'g' << 8 | 'd': case 'l' << 8 | 'o': case 'd' << 8 | 'c': case 'd' << 8 | 'e':
            case 'a' << 8 | 'c': case 'n' << 8 | 'r': case 's' << 8 | 'r': case 'r' << 8 | 'r':
            case 'd' << 8 | 'd': case 's' << 8 | 'd': case 'h' << 8 | 'g': case 'n' << 8 | 'h':
            case 'm' << 8 | 'g': case 'w' << 8 | 'f':
                break;
            default:
                if (!m.special) return false;
        }
        p += 2;
    }
    m.locked = locked;
    return rd == m.readable && wr == m.writable;
}

static bool read_maps() {
    ++counts.maps_reads;
    lines input("/proc/self/maps");
    if (input.fd < 0) return false;
    size_t size = 0;
    for (int c; (c = input.get()) >= 0;) {
        if (size == inventory::capacity) return false;
        cache.pending[size++] = char(c);
    }
    cache.pending_size = size;
    return !input.failed && size && cache.pending[size - 1] == '\n';
}

static bool same_maps() {
    if (cache.pending_size != cache.saved_size) return false;
    for (size_t i = 0; i < cache.saved_size; ++i) if (cache.pending[i] != cache.saved[i]) return false;
    return true;
}

static bool qualify() {
    ++counts.smaps_reads;
    lines input("/proc/self/smaps");
    if (input.fd < 0) return false;
    cache.count = 0;
    cache.all_locked = true;
    mapping current;
    uintptr_t previous_end = 0;
    bool have = false;
    char line[8192];
    auto finish = [&]() {
        if (!current.flags || cache.count == inventory::max_mappings) return false;
        cache.regions[cache.count++] = current;
        if (!current.special && current.readable && !current.locked) cache.all_locked = false;
        return true;
    };
    while (input.next(line, sizeof(line))) {
        const char * p = line;
        uintptr_t ignored;
        if (number(p, ignored, 16) && *p == '-') {
            if (have && !finish()) return false;
            if (!header(line, current) || current.low < previous_end) return false;
            previous_end = current.high;
            have = true;
        } else if (prefix(line, "VmFlags:")) {
            if (!have || current.flags || !vm_flags(line + 8, current)) return false;
            current.flags = true;
        } else {
            if (!have) return false;
            p = line;
            while ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || *p == '_') ++p;
            if (*p != ':') return false;
        }
    }
    return !input.failed && have && finish();
}

static bool covered(const span * spans, size_t count) {
    for (size_t i = 0; i <= count; ++i) {
        uintptr_t cursor = i == count ? state.stack_low : uintptr_t(spans[i].base);
        const uintptr_t end = i == count ? state.stack_high : uintptr_t(spans[i].base) + spans[i].bytes;
        for (size_t j = 0; j < cache.count && cursor < end; ++j) {
            const mapping & m = cache.regions[j];
            if (cursor < m.low || cursor >= m.high) continue;
            if (!m.readable || m.special || ((i == count || spans[i].writable) && !m.writable)) return false;
            cursor = m.high < end ? m.high : end;
        }
        if (cursor != end) return false;
    }
    return true;
}

static bool memory_ready(const span * spans, size_t count) {
    if (!read_maps()) return decline("maps read failed or inventory too large");
    if (cache.valid && same_maps()) {
        ++counts.cache_hits;
        return covered(spans, count) || decline("span outside approved normal RAM");
    }
    cache.valid = false;
    if (!qualify()) return decline("smaps normal RAM qualification failed");
    if (!covered(spans, count)) return decline("span outside approved normal RAM");
    if (!state.locked || !cache.all_locked) {
        ++counts.lock_calls;
        if (call(__NR_mlockall, MCL_CURRENT | MCL_FUTURE)) return decline("mlockall failed");
        state.locked = true;
    }
    cache.saved_size = cache.pending_size;
    for (size_t i = 0; i < cache.saved_size; ++i) cache.saved[i] = cache.pending[i];
    if (!read_maps() || !same_maps()) return decline("maps changed during qualification");
    cache.valid = true;
    return true;
}

__attribute__((noinline)) static void touch_stack() {
#ifdef GGML_AKV_LINUX_DIRECT_TEST
    test_stack(&state.stack_low, &state.stack_high);
#else
    volatile unsigned char bytes[stack_reserve_bytes];
    for (size_t i = 0; i < sizeof(bytes); i += 4096) bytes[i] = 0;
    bytes[sizeof(bytes) - 1] = 0;
    state.stack_low = uintptr_t(bytes);
    state.stack_high = uintptr_t(bytes) + sizeof(bytes);
    asm volatile("" : : "r"(bytes) : "memory");
#endif
}

static void touch(const span & s) {
#ifdef GGML_AKV_LINUX_DIRECT_TEST
    test_touch(s);
#else
    const uintptr_t begin = uintptr_t(s.base), end = begin + s.bytes;
    uintptr_t p = begin;
    for (;;) {
        const unsigned char value = *reinterpret_cast<const volatile unsigned char *>(p);
        if (s.writable) *reinterpret_cast<volatile unsigned char *>(p) = value;
        const uintptr_t step = 4096 - p % 4096;
        if (end - p <= step) break;
        p += step;
    }
#endif
}

static bool contains(const akv_platform_span_v1 & s) {
    const uintptr_t base = uintptr_t(s.base);
    if (!base || !s.bytes || s.bytes > UINTPTR_MAX - base || s.writable > 1) return false;
    for (size_t i = 0; i < state.count; ++i) {
        const span & prepared = state.spans[i];
        const uintptr_t start = uintptr_t(prepared.base);
        if (base >= start && base - start <= prepared.bytes && s.bytes <= prepared.bytes - (base - start) &&
            (!s.writable || prepared.writable)) return true;
    }
    return false;
}

static akv_platform_result_t logical_read(void *, akv_platform_thread_state_v1 * out) {
    if (!state.leased || !out) return AKV_PLATFORM_KEEP_NATIVE;
    read_state(*out);
    if (out->vstart) return AKV_PLATFORM_BAD_VSTART;
    return valid_state(*out) ? AKV_PLATFORM_READY : AKV_PLATFORM_KEEP_NATIVE;
}

static uintptr_t logical_acquire(void *, const akv_platform_request_v1 * r) {
    if (!state.leased || state.cookie || !r || r->abi_version != 1 || r->struct_bytes != sizeof(*r) ||
        r->requested != 1 || r->workers != 1 || r->decode_head_rows > 1 || !r->query_tokens || !r->groups ||
        r->refill_overlap || !r->spans || !r->span_count || r->span_count > max_spans) return 0;
    akv_platform_thread_state_v1 csr;
    read_state(csr);
    if (!valid_state(csr)) return 0;
    for (size_t i = 0; i < r->span_count; ++i) if (!contains(r->spans[i])) return 0;
    if (state.next_cookie == UINTPTR_MAX) fatal();
    state.cookie = ++state.next_cookie;
    return state.cookie;
}

static void logical_release(void *, uintptr_t cookie) {
    if (!state.leased || !cookie || cookie != state.cookie) fatal();
    state.cookie = 0;
}

} // namespace

bool available() {
    if (state.leased) fatal();
    state.checked = false;
    state.prepared = false;
    struct csr_guard {
        bool saved = false;
        akv_platform_thread_state_v1 csr;
        ~csr_guard() { if (saved) restore_state(csr); }
    } guard;
    if (state.csr_safe) {
        read_state(guard.csr);
        if (!valid_state(guard.csr)) return decline("unsupported vector length or non-RNE state");
        guard.saved = true;
    }
    if (!single_thread() || call(__NR_geteuid) != 0) return decline("single untraced root thread required");
    const long pid = call(__NR_getpid), tid = call(__NR_gettid);
    if (pid <= 1 || tid <= 0 || !identity()) return decline("missing deployment contract or invalid owner");
    if (state.fd >= 0 && (state.pid != pid || state.tid != tid)) {
        ara_akv_direct_close(state.fd);
        state.fd = -1;
        state.prepared = false;
        state.locked = false;
        cache.valid = false;
    }
    if (state.fd < 0) {
        ara_akv_direct_info info;
        info.generation = 0;
        state.fd = ara_akv_direct_open(&info);
        if (state.fd < 0) return decline("Direct-U device or OS/ISA admission unavailable");
        state.generation = info.generation;
        state.pid = pid; state.tid = tid;
    }
    // Recheck V policy on every call, even when retaining the descriptor.
    const long policy = call(__NR_prctl, 70);
    if (policy < 0 || (policy & 3) != 2) {
        // A changed F/V policy violates the fixed-launcher contract.
        if (guard.saved) fatal();
        return decline("vector policy disabled");
    }
    if (!guard.saved) {
        read_state(guard.csr);
        if (!valid_state(guard.csr)) return decline("unsupported vector length or non-RNE state");
        guard.saved = true;
    }
    state.csr_safe = true;
#ifdef GGML_AKV_LINUX_DIRECT_TEST
    test_init_state();
#else
    asm volatile("fmv.d.x ft0,zero; vsetivli zero,1,e8,m1,ta,ma; vmv.v.i v0,0"
                 : : : "ft0", "v0", "memory");
#endif
    state.checked = true;
    error = nullptr;
    return true;
}

bool prepare(const span * spans, size_t count) {
    if (state.leased) fatal();
    ++counts.prepares;
    state.prepared = false;
    state.count = 0;
    if (!state.checked && !available()) return false;
    state.checked = false;
    if (!spans || !count || count > max_spans) return decline("invalid span inventory");
    for (size_t i = 0; i < count; ++i) {
        const uintptr_t base = uintptr_t(spans[i].base);
        if (!base || !spans[i].bytes || spans[i].bytes > UINTPTR_MAX - base) return decline("invalid span extent");
    }
    akv_platform_thread_state_v1 csr;
    read_state(csr);
    if (!valid_state(csr)) return decline("unsupported vector length or non-RNE state");
    touch_stack();
    const bool ok = memory_ready(spans, count);
    if (ok) {
        for (size_t i = 0; i < count; ++i) {
            touch(spans[i]);
            state.spans[i].base = spans[i].base;
            state.spans[i].bytes = spans[i].bytes;
            state.spans[i].writable = spans[i].writable;
        }
        state.count = count;
    }
    restore_state(csr);
    state.prepared = ok;
    if (ok) error = nullptr;
    return ok;
}

bool begin() {
    if (state.leased) fatal();
    if (!state.prepared) return decline("prepare required");
    state.prepared = false;
    // No intervening work is permitted after prepare; only scalar syscalls here.
    akv_platform_thread_state_v1 csr;
    read_state(csr);
    if (!valid_state(csr)) return decline("unsupported vector length or non-RNE state");
    if (call(__NR_getpid) != state.pid || call(__NR_gettid) != state.tid) {
        restore_state(csr);
        return decline("owner changed after prepare");
    }
    ara_akv_direct_lease lease;
    volatile unsigned char * bytes = reinterpret_cast<volatile unsigned char *>(&lease);
    for (size_t i = 0; i < sizeof(lease); ++i) bytes[i] = 0;
    lease.abi = ARA_AKV_DIRECT_ABI; lease.bytes = sizeof(lease);
    ++counts.acquires;
    const long rc = call(__NR_ioctl, state.fd, ARA_AKV_DIRECT_ACQUIRE, long(&lease));
    restore_state(csr);
    if (rc) return decline("kernel lease acquisition failed");
    state.leased = true;
    bool valid = lease.abi == ARA_AKV_DIRECT_ABI && lease.bytes == sizeof(lease) && !lease.flags &&
        !lease.reserved0 && lease.generation > state.generation && lease.owner_tid == uint64_t(state.tid) && !lease.hart;
    for (unsigned i = 0; i < 4; ++i) valid = valid && !lease.reserved[i];
    if (!valid) { end(); return decline("malformed kernel lease response"); }
    state.generation = lease.generation;
    error = nullptr;
    return true;
}

bool end() {
    if (!state.leased) return false;
    const bool unclosed_scope = state.cookie != 0;
    akv_platform_thread_state_v1 csr;
    read_state(csr);
    ++counts.releases;
    const long rc = call(__NR_ioctl, state.fd, ARA_AKV_DIRECT_RELEASE);
    if (rc) fatal();
    restore_state(csr);
    state.leased = false; state.cookie = 0; state.count = 0;
    if (unclosed_scope) fatal();
    return true;
}

bool active() { return state.leased; }
const akv_resident_platform_provider_v1 * provider() { return state.leased ? &binding : nullptr; }
const ggml_akv_legacy_contract_v1 * legacy_contract() { return state.leased ? &legacy : nullptr; }
void invalidate() {
    if (state.leased) fatal();
    state.prepared = false; state.checked = false; state.locked = false;
    cache.valid = false;
}
const char * last_error() { return error; }
statistics stats() { return counts; }

} // namespace ggml_akv_direct

extern "C" const akv_resident_platform_provider_v1 * akv_resident_platform_current_v1() {
    return ggml_akv_direct::provider();
}
extern "C" const ggml_akv_legacy_contract_v1 * ggml_akv_legacy_current_v1() {
    return ggml_akv_direct::legacy_contract();
}
