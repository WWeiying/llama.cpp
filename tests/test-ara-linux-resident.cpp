#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "client.h"
#include "job_layout.h"
#include <asm/unistd.h>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#define GGML_AKV_LINUX_PORTAL_TEST 1
#include "arch/riscv/linux-resident.inc"

extern "C" long ara_resident_linux_execute_scalar(int, ara_resident_job *);

struct fixture {
    ggml_compute_params params{};
    ggml_tensor q{}, k{}, v{}, mask{}, dst{};
    std::vector<float> queries, outputs;
    std::vector<ggml_fp16_t> keys, values, masks;
    unsigned dim, rows, kv_heads, capacity, active;

    static void tensor(ggml_tensor & t, ggml_type type, void * data, int64_t d0, int64_t d1, int64_t d2,
                       size_t n0, size_t n1, size_t n2) {
        t.type = type;
        t.data = data;
        t.ne[0] = d0; t.ne[1] = d1; t.ne[2] = d2; t.ne[3] = 1;
        t.nb[0] = n0; t.nb[1] = n1; t.nb[2] = n2; t.nb[3] = n2 * d2;
    }

    fixture(unsigned d = 96, unsigned r = 3, unsigned kv = 2, unsigned cap = 320, unsigned t = 129) :
        queries(kv * r * (d + 8)), outputs(kv * r * (d + 12), -99.0f),
        keys(kv * (cap * (d + 4) + 16)), values(kv * (cap * (d + 8) + 32)), masks(cap + 16, 0xfc00),
        dim(d), rows(r), kv_heads(kv), capacity(cap), active(t) {
        params.nth = 1;
        tensor(q, GGML_TYPE_F32, queries.data(), d, 1, kv * r, 4, d * 4, (d + 8) * 4);
        tensor(k, GGML_TYPE_F16, keys.data(), d, cap, kv, 2, (d + 4) * 2, (cap * (d + 4) + 16) * 2);
        tensor(v, GGML_TYPE_F16, values.data(), d, cap, kv, 2, (d + 8) * 2, (cap * (d + 8) + 32) * 2);
        tensor(mask, GGML_TYPE_F16, masks.data(), cap + 16, 1, 1, 2, (cap + 16) * 2, (cap + 16) * 2);
        tensor(dst, GGML_TYPE_F32, outputs.data(), d, kv * r, 1, 4, (d + 12) * 4, kv * r * (d + 12) * 4);
        dst.op = GGML_OP_FLASH_ATTN_EXT;
        dst.src[0] = &q; dst.src[1] = &k; dst.src[2] = &v; dst.src[3] = &mask;
        const uint32_t scale = 0x3e000000U;
        std::memcpy(dst.op_params, &scale, 4);
        for (unsigned h = 0; h < kv * r; ++h) {
            for (unsigned x = 0; x < d; ++x) queries[h * (d + 8) + x] = float(h + x % 9) / 16;
        }
        for (unsigned h = 0; h < kv; ++h) {
            for (unsigned token = 0; token < cap; ++token) {
                for (unsigned x = 0; x < d; ++x) {
                    keys[h * k.nb[2] / 2 + token * k.nb[1] / 2 + x] = ggml_compute_fp32_to_fp16(float((h + token + x) % 17) / 32);
                    values[h * v.nb[2] / 2 + token * v.nb[1] / 2 + x] = ggml_compute_fp32_to_fp16(float((h + token + x) % 23) / 32);
                }
            }
        }
        for (unsigned token = 0; token < t && token < cap; ++token) masks[token] = token % 2 ? 0x8000 : 0xb800;
    }

    void unchanged() const {
        for (float x : outputs) assert(x == -99.0f);
    }
};

struct mock_state {
    int open_error = 0, info_error = 0, state_error = 0;
    int bad_info = 0, opens = 0, infos = 0, closes = 0, reads = 0, conversions = 0, executes = 0;
    int fail_group = -1;
    long execute_error = 0;
    unsigned first = 0;
    bool live = false;
    fixture * f = nullptr;
} mock;

static void reset(fixture & f) {
    mock = {};
    mock.fail_group = -1;
    mock.f = &f;
    setenv("GGML_RISCV_AKV", "1", 1);
    unsetenv("GGML_RISCV_AKV_EMULATE");
    unsetenv("GGML_RISCV_AKV_LINUX_PORTAL");
    unsetenv("GGML_RISCV_AKV_TRACE");
}

namespace ggml_akv_linux {
int test_read_state() {
    assert(mock.live && mock.infos == 1 && !mock.executes && !mock.conversions);
    ++mock.reads;
    return mock.state_error;
}
}

static void convert(const float * src, void * dst, int64_t n) {
    assert(mock.live && mock.reads == 1 && !mock.state_error);
    ++mock.conversions;
    for (int64_t i = 0; i < n; ++i) static_cast<ggml_fp16_t *>(dst)[i] = ggml_compute_fp32_to_fp16(src[i]);
}

extern "C" const ggml_type_traits_cpu * ggml_get_type_traits_cpu(ggml_type type) {
    assert(type == GGML_TYPE_F16);
    static const ggml_type_traits_cpu traits{convert, nullptr, GGML_TYPE_F16, 1};
    return &traits;
}

extern "C" void ggml_abort(const char *, int, const char *, ...) {
    throw std::runtime_error("terminal portal failure");
}

extern "C" long ara_resident_test_syscall(long nr, long a, long b, long c) {
    if (nr == __NR_openat) {
        assert(!std::strcmp(reinterpret_cast<const char *>(b), "/dev/ara-resident"));
        assert(!mock.live);
        ++mock.opens;
        if (mock.open_error) return mock.open_error;
        mock.live = true;
        return 42;
    }
    if (nr == __NR_close) {
        assert(a == 42 && mock.live);
        mock.live = false;
        ++mock.closes;
        return 0;
    }
    assert(nr == __NR_ioctl && a == 42 && mock.live);
    if (static_cast<unsigned long>(b) == ARA_RESIDENT_GET_INFO) {
        ++mock.infos;
        if (mock.info_error) return mock.info_error;
        auto & info = *reinterpret_cast<ara_resident_info *>(c);
        info.abi = ARA_RESIDENT_ABI;
        info.bytes = sizeof(info);
        info.features = ARA_RESIDENT_FEATURE_ATOMIC_GROUP | ARA_RESIDENT_FEATURE_COPY_IO;
        info.dimensions = ARA_RESIDENT_DIM64 | ARA_RESIDENT_DIM96 | ARA_RESIDENT_DIM128;
        info.max_heads = ARA_RESIDENT_MAX_HEADS;
        info.max_tokens = ARA_RESIDENT_MAX_TOKENS;
        info.nr_lanes = 4; info.vlen_bits = 1024;
        switch (mock.bad_info) {
            case 1: info.abi++; break;
            case 2: info.features = ARA_RESIDENT_FEATURE_ATOMIC_GROUP; break;
            case 3: info.dimensions = ARA_RESIDENT_DIM128; break;
            case 4: info.max_heads--; break;
            case 5: info.max_tokens--; break;
            case 6: info.reserved[0] = 1; break;
            case 7: info.nr_lanes = 8; break;
            case 8: info.vlen_bits = 512; break;
            case 9: info.bytes--; break;
        }
        return 0;
    }
    assert(static_cast<unsigned long>(b) == ARA_RESIDENT_EXECUTE);
    auto & job = *reinterpret_cast<ara_resident_job *>(c);
    auto & f = *mock.f;
    f.unchanged();
    ara_resident_sizes sizes;
    assert(ara_resident_job_sizes(&job, &sizes));
    assert(mock.reads == 1 && job.dimension == f.dim && job.tokens == f.active);
    assert(job.scale_bits == 0x3e000000U);
    const unsigned kv = mock.first / f.rows;
    const unsigned start = mock.first % f.rows;
    assert(job.heads == std::min(f.rows - start, ARA_RESIDENT_MAX_HEADS));
    const auto * query = reinterpret_cast<const ggml_fp16_t *>(uintptr_t(job.query));
    const auto * key = reinterpret_cast<const ggml_fp16_t *>(uintptr_t(job.key));
    const auto * value = reinterpret_cast<const ggml_fp16_t *>(uintptr_t(job.value));
    const auto * mask = reinterpret_cast<const ggml_fp16_t *>(uintptr_t(job.mask));
    for (unsigned h = 0; h < job.heads; ++h) {
        for (unsigned x = 0; x < f.dim; ++x) {
            assert(query[h * f.dim + x] == ggml_compute_fp32_to_fp16(f.queries[(mock.first + h) * (f.dim + 8) + x]));
        }
    }
    for (unsigned t = 0; t < f.active; ++t) {
        assert(mask[t] == f.masks[t]);
        for (unsigned x = 0; x < f.dim; ++x) {
            assert(key[t * f.dim + x] == f.keys[kv * f.k.nb[2] / 2 + t * f.k.nb[1] / 2 + x]);
            assert(value[t * f.dim + x] == f.values[kv * f.v.nb[2] / 2 + t * f.v.nb[1] / 2 + x]);
        }
    }
    auto * out = reinterpret_cast<float *>(uintptr_t(job.output));
    assert(out != f.dst.data);
    for (unsigned h = 0; h < job.heads; ++h) {
        for (unsigned x = 0; x < f.dim; ++x) out[h * f.dim + x] = float((mock.first + h) * f.dim + x + 1);
    }
    const int index = mock.executes++;
    mock.first += job.heads;
    if (index == mock.fail_group) return mock.execute_error;
    job.fflags = 0x11;
    return 0;
}

// Host substitute for the RV64 assembly entry; the real scalar client is linked.
extern "C" long ara_resident_linux_execute(int fd, ara_resident_job * job) {
    assert(mock.reads == 1 && !mock.state_error);
    return ara_resident_linux_execute_scalar(fd, job);
}

using ggml_akv_linux::result;

static result run(fixture & f) {
    const auto r = ggml_akv_linux::try_flash_attn(&f.params, &f.dst);
    assert(!mock.live);
    return r;
}

static void shape_declines() {
    const std::vector<std::function<void(fixture &)>> mutations = {
        [](fixture & f) { f.params.nth = 2; },
        [](fixture & f) { f.params.nth = 0; },
        [](fixture & f) { f.params.ith = 1; },
        [](fixture & f) { f.params.use_ref = true; },
        [](fixture & f) { f.dst.src[0] = nullptr; },
        [](fixture & f) { f.dst.src[1] = nullptr; },
        [](fixture & f) { f.dst.src[2] = nullptr; },
        [](fixture & f) { f.dst.src[3] = nullptr; },
        [](fixture & f) { f.dst.src[4] = &f.q; },
        [](fixture & f) { f.q.type = GGML_TYPE_F16; },
        [](fixture & f) { f.k.type = GGML_TYPE_Q8_0; },
        [](fixture & f) { f.v.type = GGML_TYPE_Q4_0; },
        [](fixture & f) { f.mask.type = GGML_TYPE_F32; },
        [](fixture & f) { f.dst.type = GGML_TYPE_F16; },
        [](fixture & f) { f.q.ne[0] = 256; },
        [](fixture & f) { f.q.ne[1] = 2; },
        [](fixture & f) { f.q.ne[2] = 5; },
        [](fixture & f) { f.q.ne[2] = 0; },
        [](fixture & f) { f.q.ne[3] = 2; },
        [](fixture & f) { f.k.ne[0]--; },
        [](fixture & f) { f.v.ne[1]--; },
        [](fixture & f) { f.k.ne[2] = 0; },
        [](fixture & f) { f.v.ne[2]++; },
        [](fixture & f) { f.v.ne[3]++; },
        [](fixture & f) { f.dst.ne[1]--; },
        [](fixture & f) { f.dst.ne[2]++; },
        [](fixture & f) { f.mask.ne[0] = 1; },
        [](fixture & f) { f.mask.ne[1] = 2; },
        [](fixture & f) { f.q.data = nullptr; },
        [](fixture & f) { f.k.data = nullptr; },
        [](fixture & f) { f.dst.data = nullptr; },
        [](fixture & f) { f.q.nb[0] = 8; },
        [](fixture & f) { f.mask.nb[0] = 4; },
        [](fixture & f) { f.q.nb[2] = 4; },
        [](fixture & f) { f.k.nb[1] = 2; },
        [](fixture & f) { f.v.nb[2] = f.dim * 2; },
        [](fixture & f) { f.dst.nb[1] = 4; },
        [](fixture & f) { f.k.nb[1]++; },
        [](fixture & f) { f.q.data = reinterpret_cast<char *>(f.q.data) + 1; },
        [](fixture & f) { f.q.data = reinterpret_cast<void *>(UINTPTR_MAX - 3); },
        [](fixture & f) { f.k.nb[2] = SIZE_MAX - 1; },
        [](fixture & f) { f.q.nb[2] = SIZE_MAX - 3; },
        [](fixture & f) { f.dst.data = f.q.data; },
        [](fixture & f) { f.dst.data = f.k.data; },
        [](fixture & f) { f.dst.data = f.v.data; },
        [](fixture & f) { f.dst.data = f.mask.data; },
        [](fixture & f) { f.dst.data = reinterpret_cast<char *>(f.q.data) + f.dim * 4; },
        [](fixture & f) { f.masks[0] = 0x7c00; },
        [](fixture & f) { f.masks[0] = 0x7e00; },
        [](fixture & f) { f.masks[1] = 0xfc00; },
        [](fixture & f) { f.masks[f.active + 1] = 0; },
        [](fixture & f) { std::fill(f.masks.begin(), f.masks.end(), 0xfc00); },
        [](fixture & f) { std::fill(f.masks.begin(), f.masks.end(), 0); },
    };
    for (const auto & mutation : mutations) {
        fixture f;
        reset(f);
        mutation(f);
        assert(run(f) == result::declined);
        assert(!mock.opens && !mock.reads && !mock.conversions && !mock.executes);
        f.unchanged();
    }
    for (unsigned param = 0; param < 3; ++param) {
        for (uint32_t bits : {0U, 0x80000000U, 0xbf800000U, 0x3f800000U, 0x7f800000U, 0x7fc00000U}) {
            if ((param == 0 && bits == 0x3f800000U) || (param != 0 && !(bits & 0x7fffffffU))) continue;
            fixture f;
            reset(f);
            std::memcpy(reinterpret_cast<char *>(f.dst.op_params) + param * 4, &bits, 4);
            assert(run(f) == result::declined && !mock.opens);
        }
    }
}

int main() {
    shape_declines();
    for (unsigned d : {64U, 96U, 128U}) {
        for (unsigned rows : {1U, 2U, 8U, 9U, 17U}) {
            for (unsigned tokens : {1U, 128U, 129U, 256U}) {
                fixture f(d, rows, 2, 320, tokens);
                reset(f);
                assert(run(f) == result::complete);
                assert(mock.opens == 1 && mock.infos == 1 && mock.reads == 1 && mock.closes == 1);
                assert(mock.executes == int(2 * ((rows + 7) / 8)) && mock.conversions == int(2 * rows));
                for (unsigned h = 0; h < 2 * rows; ++h) {
                    for (unsigned x = 0; x < d + 12; ++x) {
                        const float expected = x < d ? float(h * d + x + 1) : -99.0f;
                        assert(f.outputs[h * (d + 12) + x] == expected);
                    }
                }
            }
        }
    }
    for (const char * option : {"0", "false", "off", "invalid"}) {
        fixture f; reset(f);
        setenv("GGML_RISCV_AKV_LINUX_PORTAL", option, 1);
        assert(run(f) == result::declined && !mock.opens);
    }
    for (const char * option : {"1", "auto"}) {
        fixture f; reset(f);
        setenv("GGML_RISCV_AKV_LINUX_PORTAL", option, 1);
        assert(run(f) == result::complete);
    }
    for (const char * name : {"GGML_RISCV_AKV", "GGML_RISCV_AKV_EMULATE"}) {
        fixture f; reset(f);
        setenv(name, !std::strcmp(name, "GGML_RISCV_AKV") ? "0" : "1", 1);
        assert(run(f) == result::declined && !mock.opens);
    }
    for (int error : {-ENOENT, -ENODEV, -ENXIO, -EACCES, -EPERM, -EBUSY, -EOPNOTSUPP}) {
        fixture f; reset(f); mock.open_error = error;
        assert(run(f) == result::declined && !mock.reads && !mock.conversions && !mock.closes);
    }
    for (int bad = 1; bad <= 9; ++bad) {
        fixture f; reset(f); mock.bad_info = bad;
        assert(run(f) == result::declined && mock.closes == 1 && !mock.reads && !mock.conversions);
    }
    for (int error : {-ENODEV, -EOPNOTSUPP, -EFAULT, -EIO}) {
        fixture f; reset(f); mock.info_error = error;
        assert(run(f) == (error == -ENODEV || error == -EOPNOTSUPP ? result::declined : result::failed));
        assert(mock.closes == 1 && !mock.reads && !mock.conversions && !mock.executes);
    }
    {
        fixture f; reset(f);
        unsetenv("GGML_RISCV_AKV");
        setenv("GGML_RISCV_ACCEL_AUTO", "1", 1);
        assert(run(f) == result::complete);
        unsetenv("GGML_RISCV_ACCEL_AUTO");
    }
    for (int error : {-EFAULT, -EIO, -EINTR}) {
        fixture f; reset(f); mock.open_error = error;
        assert(run(f) == result::failed && !mock.reads && !mock.conversions);
    }
    for (int error : {-EOPNOTSUPP, -ERANGE}) {
        fixture f; reset(f); mock.state_error = error;
        assert(run(f) == (error == -ERANGE ? result::failed : result::declined));
        assert(mock.reads == 1 && mock.closes == 1 && !mock.conversions && !mock.executes);
    }
    for (long error : {-EOPNOTSUPP, -ENODEV, -EBUSY, -EFAULT, -EINVAL, -EIO, -EINTR, -EPROTO, -ENOMEM, 1}) {
        for (int group : {0, 1}) {
            fixture f; reset(f); mock.fail_group = group; mock.execute_error = error;
            const bool allowed = group == 0 && (error == -EOPNOTSUPP || error == -ENODEV || error == -EBUSY);
            assert(run(f) == (allowed ? result::declined : result::failed));
            assert(mock.executes == group + 1 && mock.closes == 1);
            f.unchanged();
        }
    }
    {
        fixture f; reset(f); mock.fail_group = 1; mock.execute_error = -EFAULT;
        bool aborted = false;
        try { (void) ggml_akv_linux::flash_attn(&f.params, &f.dst); }
        catch (const std::runtime_error &) { aborted = true; }
        assert(aborted && !mock.live && mock.executes == 2);
        f.unchanged();
    }
    assert(ggml_akv_linux::counts.portal_calls > 0 && ggml_akv_linux::counts.portal_groups > 0);
    assert(ggml_akv_linux::counts.portal_declines > 0 && ggml_akv_linux::counts.portal_failed > 0);
    std::puts("PASS: canonical ioctl packing, shape/stride/alias/overflow, mask prefix, state ordering, atomic publication, terminal errors, no retry, per-call fd");
    return 0;
}
