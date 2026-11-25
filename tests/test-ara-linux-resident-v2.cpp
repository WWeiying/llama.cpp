// Reuse the sealed shape/packing/retry tests without changing their source.
#define main ara_linux_resident_v1_cases
#define ara_resident_test_syscall ara_resident_test_syscall_v1
#include "test-ara-linux-resident.cpp"
#undef ara_resident_test_syscall
#undef main

static constexpr long hwprobe_nr = 258;
static constexpr uint64_t required_isa = UINT64_C(5) | (UINT64_C(1) << 27) | (UINT64_C(1) << 30);

struct admission_mock {
    long policy = 2, probe_rc = 0;
    int64_t key = 4;
    uint64_t isa = required_isa;
    unsigned policy_calls = 0, probe_calls = 0;
    bool policy_ok = false, probe_ok = false;
} admission;

extern "C" long ara_resident_test_syscall(long nr, long a, long b, long c) {
    const bool session = ggml_riscv_accel_true(std::getenv("GGML_RISCV_AKV_LINUX_SESSION"));
    if (nr == __NR_prctl) {
        assert(a == 70 && b == 0 && c == 0 && (session || !mock.live) &&
               (session || !mock.reads) && !mock.conversions);
        ++admission.policy_calls;
        admission.policy_ok = admission.policy >= 0 && (admission.policy & 3) == 2;
        admission.probe_ok = false;
        return admission.policy;
    }
    if (nr == hwprobe_nr) {
        assert(admission.policy_ok && (session || !mock.live) &&
               (session || !mock.reads) && !mock.conversions);
        assert(a && b == 1 && c == 0);
        struct { int64_t key; uint64_t value; } probe;
        std::memcpy(&probe, reinterpret_cast<const void *>(a), sizeof(probe));
        assert(probe.key == 4 && probe.value == 0);
        ++admission.probe_calls;
        probe.key = admission.key;
        probe.value = admission.isa;
        std::memcpy(reinterpret_cast<void *>(a), &probe, sizeof(probe));
        admission.probe_ok = !admission.probe_rc && probe.key == 4 && (probe.value & required_isa) == required_isa;
        return admission.probe_rc;
    }
    if (nr == __NR_openat) assert(admission.policy_ok && admission.probe_ok);
    return ara_resident_test_syscall_v1(nr, a, b, c);
}

static void reset_v2(fixture & f) {
    reset(f);
    admission = {};
}

static void admission_declines(fixture & f, bool probed) {
    assert(run(f) == result::declined);
    assert(admission.policy_calls == 1 && admission.probe_calls == unsigned(probed));
    assert(!mock.opens && !mock.infos && !mock.closes && !mock.reads && !mock.conversions && !mock.executes);
    f.unchanged();
}

static void storage_reuse_cases() {
    // Each call is still independently admitted. Only allocation capacity is retained.
    for (unsigned contiguous = 0; contiguous < 4; ++contiguous) {
        uintptr_t previous_query = 0, previous_output = 0;
        for (unsigned iteration = 0; iteration < 4; ++iteration) {
            fixture f(128, 2, 1, 256, iteration % 2 ? 65 : 137);
            if (contiguous & 1) f.k.nb[1] = f.dim * 2;
            if (contiguous & 2) f.v.nb[1] = f.dim * 2;
            // Mutate live inputs between calls: stale cached contents must not pass.
            f.keys[0] = ggml_compute_fp32_to_fp16(float(iteration + 1));
            f.values[0] = ggml_compute_fp32_to_fp16(float(iteration + 2));
            f.queries[0] = float(iteration + 3);
            reset_v2(f);
            unsetenv("GGML_RISCV_AKV_LINUX_REUSE");
            assert(run(f) == result::complete);
            assert(admission.policy_calls == 1 && admission.probe_calls == 1);
            assert(mock.opens == 1 && mock.closes == 1 && mock.executes == 1);
            // Query offset changes with packed token count. Direct K/V has no such offset.
            if (iteration && contiguous == 3) assert(mock.last_query == previous_query);
            if (iteration) assert(mock.last_output == previous_output);
            previous_query = mock.last_query; previous_output = mock.last_output;
        }
    }
    fixture f; reset_v2(f);
    setenv("GGML_RISCV_AKV_LINUX_REUSE", "0", 1);
    assert(run(f) == result::complete);
    unsetenv("GGML_RISCV_AKV_LINUX_REUSE");
    // Failure after a completed group must not publish cached partial output or retry.
    fixture bad(96, 9, 1); reset_v2(bad);
    mock.fail_group = 1; mock.execute_error = -EOPNOTSUPP;
    assert(run(bad) == result::failed); bad.unchanged();
    fixture fresh(96, 9, 1); reset_v2(fresh);
    assert(run(fresh) == result::complete);
    std::puts("PASS storage: four K/V stride combinations; refreshed inputs; allocation reuse; failure then reuse; no partial publication");
}

#ifndef ARA_RESIDENT_V2_ENTRY
#define ARA_RESIDENT_V2_ENTRY main
#endif
int ARA_RESIDENT_V2_ENTRY() {
    assert(ara_linux_resident_v1_cases() == 0);
    for (long policy : {2L, 2L | (1L << 2) | 16L, 2L | (2L << 2)}) {
        fixture f; reset_v2(f); admission.policy = policy;
        assert(run(f) == result::complete);
        assert(admission.policy_calls == 1 && admission.probe_calls == 1);
        assert(mock.opens == 1 && mock.infos == 1 && mock.closes == 1 && mock.reads == 1);
    }
    for (long policy : {0L, 1L, 3L, 1L | (2L << 2) | 16L, long(-ENOSYS), long(-EINVAL), long(-EPERM)}) {
        fixture f; reset_v2(f); admission.policy = policy;
        admission_declines(f, false);
    }
    for (uint64_t bit : {UINT64_C(1), UINT64_C(4), UINT64_C(1) << 27, UINT64_C(1) << 30}) {
        fixture f; reset_v2(f); admission.isa &= ~bit;
        admission_declines(f, true);
    }
    for (long error : {long(-ENOSYS), long(-EINVAL), long(-EFAULT), 1L}) {
        fixture f; reset_v2(f); admission.probe_rc = error;
        admission_declines(f, true);
    }
    for (int64_t key : {int64_t(-1), int64_t(0), int64_t(5)}) {
        fixture f; reset_v2(f); admission.key = key;
        admission_declines(f, true);
    }
    for (uint32_t bits : {0x3f800001U, 0x40000000U, 0x7f7fffffU}) {
        fixture f; reset_v2(f); f.dst.op_params[0] = int32_t(bits);
        assert(run(f) == result::declined);
        assert(!admission.policy_calls && !admission.probe_calls && !mock.opens && !mock.reads && !mock.conversions);
        f.unchanged();
    }
    for (uint32_t bits : {1U, 0x3e000000U, 0x3f800000U}) {
        fixture f; reset_v2(f); f.dst.op_params[0] = int32_t(bits);
        ggml_akv_linux::shape shape;
        assert(ggml_akv_linux::inspect(&f.params, &f.dst, shape) && shape.scale_bits == bits);
        assert(!admission.policy_calls && !mock.reads);
    }
    storage_reuse_cases();
    std::puts("PASS v2: current policy ON and FD/V/Zfh/Zvfh admission; OFF/ENOSYS/missing ISA/malformed key decline before open or CSR; scale <= 1");
    return 0;
}
