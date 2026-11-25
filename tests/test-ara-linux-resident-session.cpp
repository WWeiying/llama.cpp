// The existing per-call admission, packing and publication tests still run.
#define ARA_RESIDENT_V2_ENTRY ara_linux_resident_previous_cases
#include "test-ara-linux-resident-v2.cpp"
#undef ARA_RESIDENT_V2_ENTRY
#include <thread>

static void session_case(fixture & f, bool retain = false) {
    if (!retain) ggml_akv_linux::thread_session().reset();
    const bool live = mock.live;
    const int infos = mock.infos;
    reset_v2(f);
    if (retain) { mock.live = live; mock.infos = infos; }
    setenv("GGML_RISCV_AKV_LINUX_SESSION", "1", 1);
}

static void session_cases() {
    fixture cold(128, 2, 1, 256, 137);
    session_case(cold);
    assert(run(cold) == result::complete);
    assert(mock.opens == 1 && mock.infos == 1 && !mock.closes && mock.executes == 1);
    assert(admission.policy_calls == 2 && admission.probe_calls == 2);
    for (unsigned i = 0; i < 8; ++i) {
        fixture warm(i & 1 ? 96 : 128, 2, 1, 256, i & 2 ? 65 : 137);
        warm.queries[0] = float(i + 2);
        warm.keys[0] = ggml_compute_fp32_to_fp16(float(i + 3));
        warm.values[0] = ggml_compute_fp32_to_fp16(float(i + 4));
        session_case(warm, true);
        assert(run(warm) == result::complete);
        assert(!mock.opens && !mock.closes && mock.executes == 1 && mock.reads == 1);
        assert(admission.policy_calls == 1 && admission.probe_calls == 1);
    }
    for (bool policy : {true, false}) {
        fixture disabled;
        session_case(disabled, true);
        if (policy) admission.policy = 1;
        else admission.isa &= ~(UINT64_C(1) << 30);
        assert(run(disabled) == result::declined);
        assert(!mock.opens && !mock.closes && !mock.reads && !mock.conversions && !mock.executes);
        disabled.unchanged();
    }
    for (long rc : {long(-ENODEV), long(-EOPNOTSUPP), long(-EBUSY), long(-EFAULT), long(-EINTR), long(-EIO)}) {
        fixture bad;
        session_case(bad, true);
        mock.fail_group = 0; mock.execute_error = rc;
        const bool decline = rc == -ENODEV || rc == -EOPNOTSUPP || rc == -EBUSY;
        assert(run(bad) == (decline ? result::declined : result::failed));
        assert(mock.executes == 1 && mock.closes == 1 && !mock.live);
        assert(ggml_akv_linux::thread_session().fd == -1);
        bad.unchanged();
        fixture recovered;
        session_case(recovered);
        assert(run(recovered) == result::complete);
        assert(mock.opens == 1 && mock.executes == 2 && mock.closes == 0);
    }
    fixture partial(96, 9, 1);
    session_case(partial, true);
    mock.fail_group = 1; mock.execute_error = -ENODEV;
    assert(run(partial) == result::failed && mock.executes == 2 && mock.closes == 1);
    partial.unchanged();
    for (int error : {-ERANGE, -EOPNOTSUPP}) {
        fixture invalid;
        session_case(invalid);
        mock.state_error = error;
        assert(run(invalid) == (error == -EOPNOTSUPP ? result::declined : result::failed));
        assert(mock.reads == 1 && !mock.opens && !mock.conversions && !mock.executes);
        invalid.unchanged();
    }
    fixture unavailable;
    session_case(unavailable);
    mock.info_error = -ENODEV;
    assert(run(unavailable) == result::declined && mock.opens == 1 && mock.closes == 1);
    assert(ggml_akv_linux::thread_session().fd == -1);
    unavailable.unchanged();
    // No concurrent mocks: joining proves the TLS destructor closes the FD.
    std::thread worker([] {
        fixture f;
        session_case(f);
        assert(run(f) == result::complete && mock.live && !mock.closes);
    });
    worker.join();
    assert(!mock.live && mock.closes == 1);
    unsetenv("GGML_RISCV_AKV_LINUX_SESSION");
    std::puts("PASS session: warm FD reuse, fresh policy/ISA and inputs, epoch/error invalidation, no replay, private output, thread-exit close");
}

int main() {
    assert(ara_linux_resident_previous_cases() == 0);
    session_cases();
    return 0;
}
