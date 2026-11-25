#include "client.h"
#include <cstddef>

extern "C" long ara_resident_linux_execute_batch_scalar(int, ara_resident_batch *);

// Host tests cover the scalar client, not the assembly CSR save/restore.
extern "C" long ara_resident_linux_execute_batch(int fd, ara_resident_batch * batch) {
    return ara_resident_linux_execute_batch_scalar(fd, batch);
}

static_assert(offsetof(ara_resident_job, fflags) == 28, "legacy assembly offset");
static_assert(offsetof(ara_resident_batch, fflags) == 28, "batch assembly offset");

#ifndef ARA_RESIDENT_VIEWS_BATCH_WRAPPER_ONLY
#define ARA_RESIDENT_V2_ENTRY ara_linux_resident_v2_cases
#include "test-ara-linux-resident-v2.cpp"
#undef ARA_RESIDENT_V2_ENTRY
#include <map>

struct view_mock_state {
    bool enabled = false, direct = false;
    int caps_calls = 0, registrations = 0, releases = 0, batches = 0;
    int bad_caps = 0, bad_reply = 0, fail_batch = -1;
    long caps_error = 0, register_error = 0, release_error = 0, batch_error = 0;
    bool zero_handle = false;
    uint64_t next_handle = UINT64_C(0x100000000);
    std::map<uint64_t, ara_resident_kv_view> registered;
    std::vector<unsigned> batch_counts;
} views;

static ggml_fp16_t & kv_element(ggml_tensor & tensor, unsigned head, unsigned token, unsigned x) {
    return *reinterpret_cast<ggml_fp16_t *>(static_cast<char *>(tensor.data) + head * tensor.nb[2] + token * tensor.nb[1] + x * 2);
}

static void view_storage(fixture & f, bool key_interleaved, bool value_interleaved, bool padded = false) {
    const auto configure = [&](ggml_tensor & tensor, std::vector<ggml_fp16_t> & storage, bool interleaved, unsigned salt) {
        const size_t row = f.dim * 2;
        tensor.nb[1] = interleaved ? f.kv_heads * row + (padded ? 64 : 0) : row + (padded ? 64 : 0);
        tensor.nb[2] = interleaved ? row : f.capacity * tensor.nb[1] + (padded ? 64 : 0);
        const size_t bytes = (f.capacity - 1) * tensor.nb[1] + (f.kv_heads - 1) * tensor.nb[2] + row;
        storage.resize(bytes / 2 + 32);
        tensor.data = reinterpret_cast<void *>((uintptr_t(storage.data()) + 63) & ~uintptr_t(63));
        for (unsigned h = 0; h < f.kv_heads; ++h)
            for (unsigned t = 0; t < f.capacity; ++t)
                for (unsigned x = 0; x < f.dim; ++x)
                    kv_element(tensor, h, t, x) = ggml_compute_fp32_to_fp16(float((h * 7 + t * 3 + x + salt) % 29) / 32);
    };
    configure(f.k, f.keys, key_interleaved, 1);
    configure(f.v, f.values, value_interleaved, 3);
}

static ara_resident_kv_view layout(fixture & f) {
    ara_resident_kv_view view{};
    view.abi = ARA_RESIDENT_VIEW_ABI;
    view.bytes = sizeof(view);
    view.dimension = f.dim;
    view.kv_heads = f.kv_heads;
    view.capacity = std::min(f.capacity, ARA_RESIDENT_MAX_TOKENS);
    view.key = uintptr_t(f.k.data);
    view.value = uintptr_t(f.v.data);
    view.key_token_stride = uint32_t(f.k.nb[1]);
    view.key_head_stride = uint32_t(f.k.nb[2]);
    view.value_token_stride = uint32_t(f.v.nb[1]);
    view.value_head_stride = uint32_t(f.v.nb[2]);
    return view;
}

static void view_case(fixture & f, bool retain = false) {
    if (!retain) ggml_akv_linux::thread_session().reset();
    const bool live = retain && mock.live;
    auto registered = std::move(views.registered);
    const uint64_t next_handle = views.next_handle;
    views = {};
    views.enabled = true;
    if (retain) {
        views.registered = std::move(registered);
        views.next_handle = next_handle;
    }
    reset_v2(f);
    mock.live = live;
    std::fill(f.outputs.begin(), f.outputs.end(), -99.0f);
    setenv("GGML_RISCV_AKV_LINUX_SESSION", "1", 1);
    setenv("GGML_RISCV_AKV_LINUX_VIEWS", "1", 1);
}

static float expected_value(fixture & f, unsigned head, unsigned x) {
    const unsigned kv = head / f.rows;
    const auto query = ggml_compute_fp32_to_fp16(f.queries[head * (f.dim + 8) + x]);
    return float(head * f.dim + x + 1) + ggml_compute_fp16_to_fp32(query) +
           ggml_compute_fp16_to_fp32(kv_element(f.k, kv, 0, x)) +
           ggml_compute_fp16_to_fp32(kv_element(f.v, kv, f.active - 1, x));
}

static void check_output(fixture & f) {
    for (unsigned h = 0; h < f.rows * f.kv_heads; ++h)
        for (unsigned x = 0; x < f.dim + 12; ++x)
            assert(f.outputs[h * (f.dim + 12) + x] == (x < f.dim ? expected_value(f, h, x) : -99.0f));
}

extern "C" long ara_resident_views_syscall(long nr, long a, long b, long c) {
    if (nr == __NR_close) views.registered.clear();
    if (nr != __NR_ioctl || !views.enabled)
        return ara_resident_test_syscall(nr, a, b, c);
    assert(a == 42 && mock.live);
    const auto request = static_cast<unsigned long>(b);
    if (request == ARA_RESIDENT_GET_INFO) return ara_resident_test_syscall(nr, a, b, c);
    if (request == ARA_RESIDENT_VIEW_CAPS) {
        ++views.caps_calls;
        if (views.caps_error) return views.caps_error;
        auto & caps = *reinterpret_cast<ara_resident_view_caps *>(c);
        const ara_resident_view_caps zero{};
        assert(!std::memcmp(&caps, &zero, sizeof(caps)));
        caps.abi = ARA_RESIDENT_VIEW_ABI; caps.bytes = sizeof(caps);
        caps.max_views = ARA_RESIDENT_MAX_VIEWS; caps.max_groups = ARA_RESIDENT_MAX_BATCH;
        caps.max_kv_heads = ARA_RESIDENT_MAX_KV_HEADS; caps.max_view_bytes = ARA_RESIDENT_MAX_VIEW_BYTES;
        caps.pin_limit_bytes = ARA_RESIDENT_PIN_LIMIT_BYTES;
        switch (views.bad_caps) {
            case 1: ++caps.abi; break;
            case 2: --caps.bytes; break;
            case 3: --caps.max_views; break;
            case 4: --caps.max_groups; break;
            case 5: --caps.max_kv_heads; break;
            case 6: --caps.max_view_bytes; break;
            case 7: --caps.pin_limit_bytes; break;
            case 8: caps.flags = 1; break;
            default:
                if (views.bad_caps >= 9) caps.reserved[views.bad_caps - 9] = 1;
        }
        return 0;
    }
    if (request == ARA_RESIDENT_REGISTER_KV) {
        ++views.registrations;
        auto & view = *reinterpret_cast<ara_resident_kv_view *>(c);
        assert(view.handle == 0);
        const auto expected = layout(*mock.f);
        assert(!std::memcmp(&view, &expected, sizeof(view)));
        if (views.register_error) return views.register_error;
        if (views.zero_handle) return 0;
        view.handle = ++views.next_handle;
        views.registered.emplace(view.handle, view);
        return 0;
    }
    if (request == ARA_RESIDENT_RELEASE_KV) {
        ++views.releases;
        const auto & release = *reinterpret_cast<ara_resident_view_release *>(c);
        assert(release.abi == ARA_RESIDENT_VIEW_ABI && release.bytes == sizeof(release));
        assert(!release.reserved[0] && !release.reserved[1]);
        if (views.release_error) return views.release_error;
        return views.registered.erase(release.handle) ? 0 : -ESTALE;
    }
    assert(request == ARA_RESIDENT_EXECUTE_BATCH);
    auto & batch = *reinterpret_cast<ara_resident_batch *>(c);
    const int index = views.batches++;
    views.batch_counts.push_back(batch.count);
    const auto registered = views.registered.find(batch.handle);
    if (registered == views.registered.end()) return -ESTALE;
    const auto & view = registered->second;
    const auto * groups = reinterpret_cast<const ara_resident_batch_group *>(uintptr_t(batch.groups));
    ara_resident_batch_sizes sizes;
    assert(ara_resident_check_batch(&batch, groups, view.kv_heads, &sizes));
    if (!views.direct) {
        auto & f = *mock.f;
        f.unchanged();
        assert(mock.reads == 1 && !mock.executes && mock.conversions == int(f.rows * f.kv_heads));
        assert(batch.dimension == f.dim && batch.tokens == f.active && batch.scale_bits == 0x3e000000U);
        assert(batch.mask == uintptr_t(f.mask.data));
        const auto * query = reinterpret_cast<const ggml_fp16_t *>(uintptr_t(batch.query));
        const auto * mask = reinterpret_cast<const ggml_fp16_t *>(uintptr_t(batch.mask));
        auto * out = reinterpret_cast<float *>(uintptr_t(batch.output));
        const uintptr_t begin = uintptr_t(f.outputs.data()), end = begin + f.outputs.size() * sizeof(float);
        assert(batch.output < begin || batch.output >= end);
        unsigned row = 0;
        for (unsigned g = 0; g < batch.count; ++g) {
            const unsigned first = mock.first + row;
            assert(groups[g].kv_head == first / f.rows);
            assert(groups[g].heads == std::min(f.rows - first % f.rows, ARA_RESIDENT_MAX_HEADS));
            for (unsigned t = 0; t < f.active; ++t) {
                assert(mask[t] == f.masks[t]);
                for (unsigned x = 0; x < f.dim; ++x) {
                    const auto * key = reinterpret_cast<const ggml_fp16_t *>(uintptr_t(view.key) + groups[g].kv_head * view.key_head_stride + t * view.key_token_stride);
                    const auto * value = reinterpret_cast<const ggml_fp16_t *>(uintptr_t(view.value) + groups[g].kv_head * view.value_head_stride + t * view.value_token_stride);
                    assert(key[x] == kv_element(f.k, groups[g].kv_head, t, x));
                    assert(value[x] == kv_element(f.v, groups[g].kv_head, t, x));
                }
            }
            for (unsigned h = 0; h < groups[g].heads; ++h)
                for (unsigned x = 0; x < f.dim; ++x) {
                    assert(query[(row + h) * f.dim + x] == ggml_compute_fp32_to_fp16(f.queries[(first + h) * (f.dim + 8) + x]));
                    out[(row + h) * f.dim + x] = expected_value(f, first + h, x);
                }
            row += groups[g].heads;
        }
        assert(row == sizes.heads);
        mock.first += row;
    }
    if (index == views.fail_batch) return views.batch_error;
    batch.completed = views.bad_reply == 1 ? batch.count - 1 : batch.count;
    batch.fflags = views.bad_reply == 2 ? 32 : 0x11;
    return 0;
}

static void shape_and_live_cases() {
    for (unsigned dim : {64U, 96U, 128U})
        for (unsigned mode = 0; mode < 4; ++mode) {
            fixture f(dim, 9, 3, 256, 65);
            view_storage(f, mode & 1, mode & 2, mode == 3);
            view_case(f);
            assert(run(f) == result::complete);
            assert(views.caps_calls == 1 && views.registrations == 1 && views.batches == 1);
            assert(views.batch_counts == std::vector<unsigned>{6});
            assert(!mock.executes && mock.opens == 1 && !mock.closes);
            check_output(f);
            const uint64_t handle = views.registered.begin()->first;
            for (unsigned iteration = 0; iteration < 4; ++iteration) {
                ++f.active;
                f.masks[f.active - 1] = 0;
                f.queries[0] = float(iteration + 2);
                kv_element(f.k, 0, 0, 0) = ggml_compute_fp32_to_fp16(float(iteration + 3));
                kv_element(f.v, 0, f.active - 1, 0) = ggml_compute_fp32_to_fp16(float(iteration + 4));
                view_case(f, true);
                assert(run(f) == result::complete);
                assert(!views.caps_calls && !views.registrations && !views.releases && views.batches == 1);
                assert(!mock.opens && !mock.closes && !mock.executes);
                assert(admission.policy_calls == 1 && admission.probe_calls == 1);
                assert(views.registered.count(handle) == 1);
                check_output(f);
            }
        }
    for (unsigned heads : {1U, 16U, 17U, 32U})
        for (unsigned rows : {1U, 8U, 9U, 17U}) {
            fixture f(64, rows, heads, 129, 129);
            view_storage(f, true, false);
            view_case(f);
            assert(run(f) == result::complete);
            const unsigned groups = heads * ((rows + 7) / 8);
            assert(views.batches == int((groups + 15) / 16));
            unsigned remaining = groups;
            for (unsigned count : views.batch_counts) {
                assert(count == std::min(remaining, ARA_RESIDENT_MAX_BATCH));
                remaining -= count;
            }
            assert(!remaining && mock.first == heads * rows && !mock.executes);
            check_output(f);
        }
    std::puts("PASS views: D64/96/128, four KV layouts, live updates without registration copies, 16-group batch boundaries");
}

static void error_cases() {
    for (long error : {long(-ESTALE), long(-EOPNOTSUPP), long(-ENODEV), long(-EBUSY), long(-EFAULT), long(-EINTR), long(-EIO), long(-EINVAL), long(-ENOMEM), 1L})
        for (int batch : {0, 1}) {
            fixture f(64, 9, 9, 65, 65);
            view_storage(f, true, true);
            view_case(f);
            views.fail_batch = batch; views.batch_error = error;
            const bool decline = batch == 0 && (error == -ESTALE || error == -EOPNOTSUPP || error == -ENODEV || error == -EBUSY);
            assert(run(f) == (decline ? result::declined : result::failed));
            assert(views.batches == batch + 1 && !mock.executes && mock.closes == 1);
            assert(ggml_akv_linux::thread_session().fd == -1 && views.registered.empty());
            f.unchanged();
        }
    for (int bad : {1, 2}) {
        fixture f(96, 2, 2, 65, 65);
        view_storage(f, false, true);
        view_case(f); views.bad_reply = bad;
        assert(run(f) == result::failed && views.batches == 1 && mock.closes == 1);
        f.unchanged();
    }
    fixture stale(128, 2, 2, 65, 65);
    view_storage(stale, true, false);
    view_case(stale);
    assert(run(stale) == result::complete);
    view_case(stale, true);
    views.registered.clear();
    assert(run(stale) == result::declined && views.batches == 1 && !views.registrations && mock.closes == 1);
    stale.unchanged();
    view_case(stale);
    assert(run(stale) == result::complete && views.registrations == 1);
    check_output(stale);
    for (long error : {long(-ENOTTY), long(-EOPNOTSUPP), long(-EFAULT), long(-EIO)}) {
        fixture f(64, 1, 1, 1, 1);
        view_storage(f, false, false);
        view_case(f); views.caps_error = error;
        assert(run(f) == (error == -ENOTTY || error == -EOPNOTSUPP ? result::declined : result::failed));
        assert(views.caps_calls == 1 && !views.registrations && !views.batches && !mock.executes);
        f.unchanged();
        if (error == -ENOTTY || error == -EOPNOTSUPP) {
            view_case(f, true);
            assert(run(f) == result::declined && !views.caps_calls && !views.batches);
            f.unchanged();
        }
    }
    for (long error : {long(-ENOMEM), long(-ENOSPC), long(-ESTALE), long(-EINVAL), long(-EFAULT), long(-EIO)}) {
        fixture f(64, 1, 1, 1, 1);
        view_storage(f, false, false);
        view_case(f); views.register_error = error;
        assert(run(f) == (error == -ENOMEM || error == -ENOSPC || error == -ESTALE ? result::declined : result::failed));
        assert(views.registrations == 1 && !views.batches && !mock.executes && mock.closes == 1);
        f.unchanged();
    }
    std::puts("PASS views errors: first/later batch failures, stale handle, malformed completion/fflags, no replay or partial publication");
}

static void cache_cases() {
    std::vector<std::unique_ptr<fixture>> tensors;
    for (unsigned i = 0; i <= ARA_RESIDENT_MAX_VIEWS; ++i) {
        tensors.emplace_back(new fixture(64, 1, 1, 1, 1));
        auto & f = *tensors.back();
        view_storage(f, false, true);
        view_case(f, i != 0);
        assert(run(f) == result::complete && views.registrations == 1);
        assert(views.releases == (i == ARA_RESIDENT_MAX_VIEWS ? 1 : 0));
        assert(views.registered.size() == std::min(i + 1, ARA_RESIDENT_MAX_VIEWS));
        check_output(f);
    }
    ggml_akv_linux::thread_session().reset();
    assert(views.registered.empty() && !mock.live);
    for (long error : {long(-ENODEV), long(-EIO)}) {
        std::vector<std::unique_ptr<fixture>> slots;
        for (unsigned i = 0; i < ARA_RESIDENT_MAX_VIEWS; ++i) {
            slots.emplace_back(new fixture(64, 1, 1, 1, 1));
            auto & f = *slots.back();
            view_storage(f, false, true);
            view_case(f, i != 0);
            assert(run(f) == result::complete && views.registrations == 1 && !views.releases);
            check_output(f);
        }
        fixture candidate(64, 1, 1, 1, 1);
        view_storage(candidate, false, true);
        view_case(candidate, true);
        assert(views.registered.size() == ARA_RESIDENT_MAX_VIEWS);
        assert(ggml_akv_linux::thread_session().next_view == 0);
        views.release_error = error;
        assert(run(candidate) == (error == -ENODEV ? result::declined : result::failed));
        assert(views.releases == 1 && !views.registrations && !views.batches);
        assert(!mock.executes && !mock.conversions && !mock.first);
        assert(mock.closes == 1 && !mock.live && views.registered.empty());
        const auto & session = ggml_akv_linux::thread_session();
        assert(session.fd == -1 && session.view_support == 0 && session.next_view == 0);
        for (const auto & view : session.views) assert(!view.handle);
        candidate.unchanged();
    }
    std::puts("PASS views cache: distinct live mappings, bounded handle eviction and FD cleanup");
    std::puts("PASS views eviction errors: 33rd-view ENODEV declines, EIO is terminal; no registration, conversion, execution, replay or output");
}

static void scalar_client_cases() {
    fixture f(64, 1, 2, 2, 2);
    view_storage(f, true, false);
    view_case(f);
    views.direct = true;
    ara_resident_info info;
    assert(ara_resident_linux_open(&info) == 42);
    ara_resident_view_caps caps;
    assert(ara_resident_linux_view_caps(-1, &caps) == -EINVAL);
    assert(ara_resident_linux_view_caps(42, nullptr) == -EINVAL);
    assert(!views.caps_calls);
    for (int bad = 0; bad <= 12; ++bad) {
        views.bad_caps = bad;
        std::memset(&caps, 0xa5, sizeof(caps));
        assert(ara_resident_linux_view_caps(42, &caps) == (bad ? -EPROTO : 0));
    }
    auto view = layout(f);
    assert(ara_resident_linux_register_kv(-1, &view) == -EINVAL);
    assert(ara_resident_linux_register_kv(42, nullptr) == -EINVAL);
    for (const auto & mutate : std::vector<std::function<void(ara_resident_kv_view &)>>{
            [](ara_resident_kv_view & v) { ++v.abi; },
            [](ara_resident_kv_view & v) { --v.bytes; },
            [](ara_resident_kv_view & v) { v.dimension = 80; },
            [](ara_resident_kv_view & v) { v.kv_heads = 0; },
            [](ara_resident_kv_view & v) { v.capacity = 257; },
            [](ara_resident_kv_view & v) { ++v.key; },
            [](ara_resident_kv_view & v) { v.flags = 1; },
            [](ara_resident_kv_view & v) { v.handle = 1; },
            [](ara_resident_kv_view & v) { v.reserved[3] = 1; },
            [](ara_resident_kv_view & v) { v.key_token_stride = 2; },
            [](ara_resident_kv_view & v) { v.value = UINT64_MAX - 63; }}) {
        auto invalid = view;
        mutate(invalid);
        assert(ara_resident_linux_register_kv(42, &invalid) == -EINVAL);
    }
    assert(!views.registrations);
    views.zero_handle = true;
    assert(ara_resident_linux_register_kv(42, &view) == -EPROTO);
    views.zero_handle = false;
    assert(ara_resident_linux_register_kv(42, &view) == 0 && view.handle);
    const auto handle = view.handle;
    ara_resident_batch_group group{0, 1, 0};
    ggml_fp16_t query[64]{};
    float output[64]{};
    ara_resident_batch batch{};
    batch.abi = ARA_RESIDENT_VIEW_ABI; batch.bytes = sizeof(batch);
    batch.count = 1; batch.dimension = 64; batch.tokens = 2; batch.scale_bits = 0x3e000000;
    batch.handle = handle; batch.groups = uintptr_t(&group); batch.query = uintptr_t(query);
    batch.mask = uintptr_t(f.mask.data); batch.output = uintptr_t(output);
    assert(ara_resident_linux_execute_batch(-1, &batch) == -EINVAL);
    assert(ara_resident_linux_execute_batch(42, nullptr) == -EINVAL);
    for (const auto & mutate : std::vector<std::function<void(ara_resident_batch &)>>{
            [](ara_resident_batch & b) { ++b.abi; },
            [](ara_resident_batch & b) { --b.bytes; },
            [](ara_resident_batch & b) { b.count = 0; },
            [](ara_resident_batch & b) { b.count = 17; },
            [](ara_resident_batch & b) { b.handle = 0; },
            [](ara_resident_batch & b) { b.fflags = 1; },
            [](ara_resident_batch & b) { b.completed = 1; },
            [](ara_resident_batch & b) { b.flags = 1; },
            [](ara_resident_batch & b) { b.reserved0 = 1; },
            [](ara_resident_batch & b) { b.reserved[2] = 1; },
            [](ara_resident_batch & b) { b.groups++; },
            [](ara_resident_batch & b) { b.groups = UINT64_MAX - 7; },
            [](ara_resident_batch & b) { b.query++; },
            [](ara_resident_batch & b) { b.output++; },
            [](ara_resident_batch & b) { b.scale_bits = 0x3f800001; }}) {
        auto invalid = batch;
        mutate(invalid);
        assert(ara_resident_linux_execute_batch(42, &invalid) == -EINVAL);
    }
    assert(!views.batches);
    auto invalid = batch;
    invalid.handle = handle + 1;
    assert(ara_resident_linux_execute_batch(42, &invalid) == -ESTALE && views.batches == 1);
    assert(ara_resident_linux_release_kv(-1, handle) == -EINVAL);
    assert(ara_resident_linux_release_kv(42, 0) == -EINVAL);
    assert(!views.releases);
    assert(ara_resident_linux_execute_batch(42, &batch) == 0 && batch.fflags == 0x11 && batch.completed == 1);
    assert(ara_resident_linux_release_kv(42, handle) == 0);
    assert(ara_resident_linux_release_kv(42, handle) == -ESTALE);
    batch.fflags = batch.completed = 0;
    assert(ara_resident_linux_execute_batch(42, &batch) == -ESTALE);
    ara_resident_linux_close(42);
    assert(!mock.live && views.registered.empty());
    std::puts("PASS views client: actual C ioctl path, capability validation, invalid metadata/handle, release and shared fflags offset");
}

int main() {
    unsetenv("GGML_RISCV_AKV_LINUX_VIEWS");
    assert(ara_linux_resident_v2_cases() == 0);
    scalar_client_cases();
    shape_and_live_cases();
    error_cases();
    cache_cases();
    ggml_akv_linux::thread_session().reset();
    unsetenv("GGML_RISCV_AKV_LINUX_VIEWS");
    unsetenv("GGML_RISCV_AKV_LINUX_SESSION");
    return 0;
}
#endif
