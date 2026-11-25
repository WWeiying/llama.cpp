// Standalone host test: compile this TU with the pinned SDK and direct-client include paths.
#define GGML_AKV_LINUX_DIRECT_TEST 1
#ifndef GGML_AKV_DIRECT_CONTRACT_SHA256
#define GGML_AKV_DIRECT_CONTRACT_SHA256 "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
#endif
#include "../ggml/src/ggml-cpu/arch/riscv/linux-direct.cpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct mock_system {
    std::string status = "Name:\tunit\nThreads:\t1\nTracerPid:\t0\n";
    std::string smaps;
    std::string maps, maps_override;
    size_t cursor[3] = {};
    int opens = 0, closes = 0, reads = 0, locks = 0, acquires = 0, releases = 0;
    int init = 0, touches = 0, stack = 0, syscalls = 0, csr_reads = 0;
    long open_error = 0, lock_error = 0, acquire_error = 0, release_error = 0;
    long pid = 100, tid = 100, uid = 0, policy = 2;
    uint64_t generation = 10;
    int bad_lease = 0;
    bool kernel_lease = false, os_allowed = false, fail_read = false, clobber_csr = false;
    akv_platform_thread_state_v1 csr = {128, 0, 0x11, 5};
    std::vector<ggml_akv_direct::span> touched;
} mock;

std::string maps(const std::string & permissions = "rw-p", const std::string & flags = "rd wr mr mw me ac sd lo",
                 const std::string & path = "[heap]") {
    return "1000-9000 " + permissions + " 00000000 00:00 0 " + path + "\nSize: 32 kB\nVmFlags: " + flags +
        "\n100000-104000 r-xp 00000000 08:01 123 /usr/bin/model\nVmFlags: rd ex mr me lo\n"
        "700000-780000 rw-p 00000000 00:00 0 [stack]\nVmFlags: rd wr mr mw me gd ac lo\n";
}

void reset() {
    mock = mock_system{};
    mock.smaps = maps();
    ggml_akv_direct::state = ggml_akv_direct::runtime{};
    ggml_akv_direct::cache = ggml_akv_direct::inventory{};
    ggml_akv_direct::counts = {};
    ggml_akv_direct::error = "not prepared";
}

const ggml_akv_direct::span inputs[] = {{reinterpret_cast<void *>(0x1801), 8192, false},
                                      {reinterpret_cast<void *>(0x5000), 4096, true}};

template<class F> void dies(F f) {
    bool caught = false;
    try { f(); } catch (const std::runtime_error &) { caught = true; }
    assert(caught);
}

void same_csr(const akv_platform_thread_state_v1 & a, const akv_platform_thread_state_v1 & b) {
    assert(a.vlenb == b.vlenb && a.vstart == b.vstart && a.fcsr == b.fcsr && a.vcsr == b.vcsr);
}
}

extern "C" int ara_akv_direct_open(ara_akv_direct_info * info) {
    assert(!mock.kernel_lease);
    ++mock.opens;
    if (mock.open_error) return int(mock.open_error);
    info->generation = mock.generation;
    mock.os_allowed = true;
    return 42;
}
extern "C" void ara_akv_direct_close(int fd) {
    assert(fd == 42 && !mock.kernel_lease);
    ++mock.closes;
}

namespace ggml_akv_direct {
namespace {
long test_syscall(long nr, long x, long y, long z) {
    ++mock.syscalls;
    if (mock.kernel_lease) assert(nr == __NR_ioctl && static_cast<unsigned long>(y) == ARA_AKV_DIRECT_RELEASE);
    if (mock.clobber_csr) { mock.csr.vstart = 0; mock.csr.vcsr = 0; }
    if (nr == __NR_getpid) return mock.pid;
    if (nr == __NR_gettid) return mock.tid;
    if (nr == __NR_geteuid) return mock.uid;
    if (nr == __NR_prctl) { assert(x == 70); return mock.policy; }
    if (nr == __NR_openat) {
        const char * path = reinterpret_cast<const char *>(y);
        const int index = !std::strcmp(path, "/proc/self/status") ? 0 : !std::strcmp(path, "/proc/self/smaps") ? 1 : 2;
        assert(index != 2 || !std::strcmp(path, "/proc/self/maps"));
        if (index == 2) {
            mock.maps = mock.maps_override;
            if (mock.maps.empty()) {
                for (size_t pos = 0; pos < mock.smaps.size();) {
                    const size_t end = mock.smaps.find('\n', pos);
                    if (end == std::string::npos) break;
                    const std::string line = mock.smaps.substr(pos, end - pos);
                    if (line.find('-') < line.find(' ')) mock.maps += line + "\n";
                    pos = end + 1;
                }
            }
        }
        mock.cursor[index] = 0;
        return 100 + index;
    }
    if (nr == __NR_close) { assert(x >= 100 && x <= 102); return 0; }
    if (nr == __NR_read) {
        assert(x >= 100 && x <= 102);
        if (mock.fail_read) return -5;
        const size_t index = size_t(x - 100);
        const auto & text = index == 0 ? mock.status : index == 1 ? mock.smaps : mock.maps;
        size_t n = text.size() - mock.cursor[index];
        if (n > size_t(z)) n = size_t(z);
        // Exercise parser boundaries, not just a single complete read.
        if (n > 73) n = 73;
        std::memcpy(reinterpret_cast<void *>(y), text.data() + mock.cursor[index], n);
        mock.cursor[index] += n;
        ++mock.reads;
        return long(n);
    }
    if (nr == __NR_mlockall) {
        assert(x == (MCL_CURRENT | MCL_FUTURE) && mock.stack && !mock.kernel_lease);
        ++mock.locks;
        return mock.lock_error;
    }
    assert(nr == __NR_ioctl && x == 42);
    if (static_cast<unsigned long>(y) == ARA_AKV_DIRECT_ACQUIRE) {
        assert(!mock.kernel_lease && mock.touches == 2 && mock.stack && mock.init && mock.locks);
        ++mock.acquires;
        if (mock.acquire_error) return mock.acquire_error;
        auto & lease = *reinterpret_cast<ara_akv_direct_lease *>(z);
        assert(lease.abi == 1 && lease.bytes == sizeof(lease) && !lease.flags && !lease.reserved0 &&
               !lease.generation && !lease.owner_tid && !lease.hart);
        for (auto r : lease.reserved) assert(!r);
        lease.generation = ++mock.generation;
        lease.owner_tid = uint32_t(mock.tid);
        switch (mock.bad_lease) {
            case 1: ++lease.abi; break;
            case 2: --lease.bytes; break;
            case 3: lease.flags = 1; break;
            case 4: lease.reserved0 = 1; break;
            case 5: lease.generation = 0; break;
            case 6: ++lease.owner_tid; break;
            case 7: lease.hart = 1; break;
            case 8: lease.reserved[2] = 1; break;
        }
        mock.kernel_lease = true;
        return 0;
    }
    assert(static_cast<unsigned long>(y) == ARA_AKV_DIRECT_RELEASE && z == 0 && mock.kernel_lease);
    ++mock.releases;
    if (mock.release_error) return mock.release_error;
    mock.kernel_lease = false;
    return 0;
}

void test_read_state(akv_platform_thread_state_v1 * out) {
    assert(mock.os_allowed);
    ++mock.csr_reads;
    *out = mock.csr;
}
void test_restore_state(const akv_platform_thread_state_v1 * in) { mock.csr = *in; }
void test_init_state() { assert(mock.os_allowed && !mock.kernel_lease && !mock.csr.vstart); ++mock.init; }
void test_touch(const span & s) {
    assert(!mock.kernel_lease && mock.locks);
    ++mock.touches;
    mock.touched.push_back(s);
}
void test_stack(uintptr_t * low, uintptr_t * high) {
    assert(!mock.kernel_lease && mock.os_allowed);
    ++mock.stack;
    *low = 0x700000; *high = *low + stack_reserve_bytes;
}
void test_fatal() { throw std::runtime_error("fail closed"); }
}
}

static void lifecycle() {
    using namespace ggml_akv_direct;
    reset();
    assert(!provider() && !legacy_contract() && !begin() && !end());
    assert(available());
    auto original = mock.csr;
    mock.clobber_csr = true;
    assert(prepare(inputs, 2));
    same_csr(original, mock.csr);
    assert(!provider() && begin() && active());
    same_csr(original, mock.csr);
    const auto * p = provider();
    const auto * c = ggml_akv_legacy_current_v1();
    assert(p && p == akv_resident_platform_current_v1() && c == legacy_contract() && c->provider == p);
    assert(c->operation_mask == (GGML_AKV_LEGACY_BASE | GGML_AKV_LEGACY_PREFILL) && c->whole_process_ram == 1);
    assert(!std::memcmp(c->contract_sha256, p->contract_sha256, 32));
    akv_platform_span_v1 regions[] = {{inputs[0].base, inputs[0].bytes, 0}, {inputs[1].base, inputs[1].bytes, 1}};
    akv_platform_request_v1 request = {1, sizeof(request), 1, 1, 1, 1, 12, 0, regions, 2};
    const int calls = mock.syscalls;
    const uintptr_t cookie = p->acquire(p->context, &request);
    assert(cookie && !p->acquire(p->context, &request));
    for (unsigned group = 0; group < request.groups; ++group) {
        akv_platform_thread_state_v1 out;
        assert(p->read_state(p->context, &out) == AKV_PLATFORM_READY);
        same_csr(original, out);
    }
    p->release(p->context, cookie);
    assert(mock.syscalls == calls && mock.acquires == 1 && !mock.releases && active());
    mock.csr.fcsr = 0x1f;
    mock.csr.vcsr = 7;
    original = mock.csr;
    assert(end() && !active() && !provider());
    same_csr(original, mock.csr);
    assert(mock.releases == 1 && !begin());
    mock.touches = 0;
    assert(prepare(inputs, 2) && begin() && end());
    assert(mock.opens == 1 && mock.acquires == 2 && mock.releases == 2 && mock.locks == 1);
    const auto s = stats();
    assert(s.prepares == 2 && s.maps_reads == 3 && s.smaps_reads == 1 && s.cache_hits == 1 && s.lock_calls == 1);
    assert(s.acquires == 2 && s.releases == 2 && !last_error());
}

static void admission_failures() {
    using namespace ggml_akv_direct;
    reset(); mock.open_error = -2;
    assert(!available() && !mock.init && !mock.csr_reads && !mock.acquires);
    reset(); mock.status = "Threads: 2\nTracerPid: 0\n";
    assert(!available() && !mock.opens && !mock.csr_reads);
    reset(); mock.status = "Threads: 1\nTracerPid: 10\n";
    assert(!available() && !mock.opens);
    reset(); mock.uid = 1;
    assert(!available() && !mock.opens);
    reset(); mock.policy = 1;
    assert(!available() && !mock.csr_reads);
    reset(); mock.policy = 2 | 16;
    assert(available());
    reset(); mock.csr.fcsr = 0x20;
    assert(!available() && !mock.init);
    reset(); mock.csr.vlenb = 64;
    assert(!available() && !mock.init);
    reset(); mock.csr.vstart = 3;
    dies([] { available(); });
    assert(!mock.init);
    reset(); assert(available()); mock.csr.vstart = 3; mock.clobber_csr = true;
    const int calls = mock.syscalls;
    dies([] { available(); });
    assert(mock.syscalls == calls);
    reset(); mock.lock_error = -12;
    assert(!prepare(inputs, 2) && !mock.touches && !begin());
    reset(); mock.fail_read = true;
    assert(!prepare(inputs, 2) && !mock.acquires);
    reset();
    const span bad[] = {{nullptr, 1, false}, {reinterpret_cast<void *>(1), 0, false},
                        {reinterpret_cast<void *>(UINTPTR_MAX - 1), 4, true}};
    for (const auto & s : bad) assert(!prepare(&s, 1));
    assert(!prepare(nullptr, 1) && !prepare(inputs, 0) && !prepare(inputs, max_spans + 1));
    assert(!mock.locks);
}

static void mapping_failures() {
    using namespace ggml_akv_direct;
    std::vector<std::string> bad;
    for (const char * flag : {"io", "pf", "mm", "ht", "sh", "um", "uw", "zz"})
        bad.push_back(maps("rw-p", std::string("rd wr ") + flag));
    bad.push_back(maps("rw-s"));
    bad.push_back(maps("r--p", "rd mr lo"));
    bad.push_back(maps("rw-p", "rd wr", "/dev/dax0.0"));
    bad.push_back(maps("rw-p", "rd wr", "[unknown]"));
    bad.push_back(maps("rw-p", "rd wr", "/proc/1/mem"));
    bad.push_back("1000-9000 rw-p 00000000 00:00 0 [heap]\n");
    bad.push_back("garbage\n" + maps());
    bad.push_back(maps() + "\ntruncated");
    bad.push_back(maps() + "800000-900000 rw-p 00000000 00:00 0 /tmp/device\nVmFlags: rd wr pf\n");
    auto hole = maps(); hole.replace(0, 9, "2000-9000"); bad.push_back(hole);
    auto stack_hole = maps(); stack_hole.replace(stack_hole.find("780000"), 6, "770000"); bad.push_back(stack_hole);
    for (const auto & text : bad) {
        reset(); mock.smaps = text;
        assert(!prepare(inputs, 2) && !mock.touches && !mock.locks && !begin());
    }
    reset();
    mock.smaps += "800000-802000 r--p 00000000 00:00 0 [vvar]\nVmFlags: rd pf io\n";
    assert(prepare(inputs, 2));
    const span special = {reinterpret_cast<void *>(0x800000), 1, false};
    assert(!prepare(&special, 1));
    reset(); assert(prepare(inputs, 2));
    mock.smaps = maps("rw-p", "rd wr");
    invalidate(); // maps cannot observe whole-VMA munlock; launcher must notify.
    mock.lock_error = -12;
    assert(!prepare(inputs, 2) && mock.locks == 2 && last_error());
}

static void inventory_cache() {
    using namespace ggml_akv_direct;
    reset(); assert(available());
    const int calls = mock.syscalls;
    assert(prepare(inputs, 2));
    assert(mock.opens == 1 && mock.init == 1); // prepare consumes available(), without a second probe.
    assert(mock.syscalls > calls);
    assert(prepare(inputs, 2));
    assert(stats().smaps_reads == 1 && stats().cache_hits == 1 && mock.locks == 1);
    mock.smaps += "800000-801000 rw-p 00000000 00:00 0 [anon:new]\nVmFlags: rd wr lo\n";
    assert(prepare(inputs, 2));
    assert(stats().smaps_reads == 2 && mock.locks == 1);
    mock.smaps += "900000-901000 rw-p 00000000 00:00 0 /tmp/device\nVmFlags: rd wr pf\n";
    assert(!prepare(inputs, 2) && stats().smaps_reads == 3 && last_error());
    reset(); assert(prepare(inputs, 2));
    mock.smaps += "800000-801000 rw-p 00000000 00:00 0 [anon:new]\nVmFlags: rd wr\n";
    assert(prepare(inputs, 2) && mock.locks == 2);
    reset(); assert(prepare(inputs, 2));
    invalidate(); assert(!begin() && prepare(inputs, 2));
    assert(stats().smaps_reads == 2 && mock.locks == 2);
    reset(); mock.maps_override = std::string(inventory::capacity + 1, 'a');
    assert(!prepare(inputs, 2) && !mock.locks);
    reset(); mock.maps_override = "truncated";
    assert(!prepare(inputs, 2));
}

static void lease_failures() {
    using namespace ggml_akv_direct;
    for (int bad = 1; bad <= 8; ++bad) {
        reset(); mock.bad_lease = bad;
        assert(prepare(inputs, 2) && !begin() && !active());
        assert(mock.acquires == 1 && mock.releases == 1 && !mock.kernel_lease);
    }
    reset(); mock.acquire_error = -16;
    assert(prepare(inputs, 2) && !begin() && !begin() && !mock.releases);
    reset(); assert(prepare(inputs, 2)); ++mock.pid;
    assert(!begin() && !mock.acquires);
    reset(); assert(prepare(inputs, 2)); mock.csr.vstart = 1;
    dies([] { begin(); }); assert(!mock.acquires);
    reset(); assert(prepare(inputs, 2) && begin()); mock.release_error = -5;
    dies([] { end(); }); assert(mock.kernel_lease && mock.releases == 1);
    reset(); assert(prepare(inputs, 2) && begin());
    dies([] { begin(); }); dies([] { prepare(inputs, 2); });
    assert(end());
    reset(); assert(prepare(inputs, 2) && begin());
    const auto * p = provider();
    akv_platform_span_v1 region = {inputs[0].base, inputs[0].bytes, 1};
    akv_platform_request_v1 r = {1, sizeof(r), 1, 1, 1, 1, 4, 0, &region, 1};
    assert(!p->acquire(p->context, &r)); // Read-only preparation cannot authorize writes.
    region.writable = 0; region.bytes += 1;
    assert(!p->acquire(p->context, &r));
    region.bytes -= 1; r.workers = 2;
    assert(!p->acquire(p->context, &r));
    r.workers = 1; r.query_tokens = 16; r.decode_head_rows = 0;
    const uintptr_t cookie = p->acquire(p->context, &r); // Whole F16 prefill scope.
    assert(cookie);
    mock.csr.vstart = 7;
    akv_platform_thread_state_v1 out;
    assert(p->read_state(p->context, &out) == AKV_PLATFORM_BAD_VSTART);
    const auto csr = mock.csr;
    p->release(p->context, cookie);
    same_csr(csr, mock.csr);
    assert(end()); same_csr(csr, mock.csr);
    reset(); assert(prepare(inputs, 2) && begin());
    dies([] { provider()->release(nullptr, 1); });
    assert(end());
    reset(); assert(prepare(inputs, 2) && begin());
    r.query_tokens = 1;
    assert(provider()->acquire(nullptr, &r));
    dies([] { end(); }); assert(!mock.kernel_lease && mock.releases == 1);
}

int main() {
    if (!ggml_akv_direct::identity()) {
        reset();
        assert(!ggml_akv_direct::available() && !mock.opens && !mock.csr_reads && ggml_akv_direct::last_error());
        std::puts("PASS: missing/malformed deployment pin disables Direct-U before CSR or device access");
        return 0;
    }
    lifecycle();
    admission_failures();
    mapping_failures();
    inventory_cache();
    lease_failures();
    std::puts("PASS: Direct-U preparation, smaps, locks, CSR guards, whole-call lease, logical scopes, release fail-closed (mock OS only)");
}
