#include "qbs.h"

#include "ggml-quants.h"
#include "qbs/qbs.h"
#include "quants.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct qbs_profile {
    unsigned weight;
    unsigned activation;
};

struct qbs_encoding_pair {
    uint64_t weight;
    uint64_t activation;
};

struct qbs_trace_counters {
    std::atomic<uint64_t>                        gemv_calls{ 0 };
    std::atomic<uint64_t>                        gemm_calls{ 0 };
    std::atomic<uint64_t>                        input_rows{ 0 };
    std::atomic<uint64_t>                        output_rows{ 0 };
    std::atomic<uint64_t>                        dot_elements{ 0 };
    std::atomic<uint64_t>                        split_calls{ 0 };
    std::array<std::atomic<uint64_t>, QBS_MAX_M> commands_by_m{};
    std::atomic<uint64_t>                        native_qbexec{ 0 };
    std::atomic<uint64_t>                        emulated_commands{ 0 };
    std::atomic<uint64_t>                        command_dot_elements{ 0 };
    std::atomic<uint64_t>                        segmented_commands{ 0 };
    std::atomic<uint64_t>                        context_fill_commands{ 0 };
    std::atomic<uint64_t>                        context_reuse_commands{ 0 };
    std::atomic<uint64_t>                        context_release_commands{ 0 };

    std::atomic<uint64_t> candidate_tensors{ 0 };
    std::atomic<uint64_t> selected_tensors{ 0 };
    std::atomic<uint64_t> segmented_tensors{ 0 };
    std::atomic<uint64_t> candidate_elements{ 0 };
    std::atomic<uint64_t> selected_elements{ 0 };
    std::atomic<uint64_t> fallback_runtime{ 0 };
    std::atomic<uint64_t> fallback_format_filter{ 0 };
    std::atomic<uint64_t> fallback_capability{ 0 };
    std::atomic<uint64_t> fallback_dimensions{ 0 };
    std::atomic<uint64_t> fallback_shape{ 0 };
    std::atomic<uint64_t> fallback_layout{ 0 };
    std::atomic<uint64_t> fallback_profile{ 0 };
    std::atomic<uint64_t> fallback_dispatch{ 0 };
};

enum class qbs_support_reason {
    supported,
    not_candidate,
    runtime_disabled,
    format_filter,
    capability,
    dimensions,
    shape,
    layout,
    profile,
    dispatch,
};

constexpr size_t                                         QBS_TRACE_FORMATS = 9;
static std::array<qbs_trace_counters, QBS_TRACE_FORMATS> qbs_trace{};

struct qbs_activation_trace_record {
    uint64_t             sequence;
    const void *         source_tensor_identity;
    const void *         source_data;
    ggml_type            activation_type;
    int64_t              activation_rows;
    int64_t              k_elements;
    uint64_t             digest;
    std::string          weight_name;
    std::vector<uint8_t> quantized_data;
};

struct qbs_activation_trace_state {
    std::mutex                              mutex;
    std::deque<qbs_activation_trace_record> records;
    uint64_t                                sequence               = 0;
    uint64_t                                quantizations           = 0;
    uint64_t                                exact_reuse_candidates  = 0;
    uint64_t                                quantized_bytes          = 0;
    uint64_t                                reusable_quantized_bytes = 0;
    size_t                                  retained_snapshot_bytes  = 0;
};

static qbs_activation_trace_state & activation_trace_state() {
    static auto * state = new qbs_activation_trace_state;
    return *state;
}

static uint64_t activation_digest(const void * data, size_t bytes) {
    const auto * values = static_cast<const uint8_t *>(data);
    uint64_t     hash   = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < bytes; ++index) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static std::string trace_name(const char * name) {
    std::string result = name && name[0] ? name : "<unnamed>";
    for (char & character : result) {
        if (std::isspace(static_cast<unsigned char>(character)) || character == '=') {
            character = '_';
        }
    }
    return result;
}

static void print_activation_trace_summary() {
    auto & state = activation_trace_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::fprintf(stderr,
                 "GGML_RISCV_QBS_LIFETIME_SUMMARY quantizations=%llu "
                 "exact_reuse_candidates=%llu quantized_bytes=%llu "
                 "reusable_quantized_bytes=%llu\n",
                 static_cast<unsigned long long>(state.quantizations),
                 static_cast<unsigned long long>(state.exact_reuse_candidates),
                 static_cast<unsigned long long>(state.quantized_bytes),
                 static_cast<unsigned long long>(state.reusable_quantized_bytes));
}

static void ensure_activation_trace_registered() {
    static const bool registered = [] {
        std::atexit(print_activation_trace_summary);
        return true;
    }();
    (void) registered;
}

static qbs_encoding_pair encoding_for_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return { QBS_WEIGHT_ENCODING_Q4_K, QBS_ACTIVATION_ENCODING_Q8_K };
        case GGML_TYPE_Q5_K:
            return { QBS_WEIGHT_ENCODING_Q5_K, QBS_ACTIVATION_ENCODING_Q8_K };
        case GGML_TYPE_Q6_K:
            return { QBS_WEIGHT_ENCODING_Q6_K, QBS_ACTIVATION_ENCODING_Q8_K };
        case GGML_TYPE_Q3_K:
            return { QBS_WEIGHT_ENCODING_Q3_K, QBS_ACTIVATION_ENCODING_Q8_K };
        case GGML_TYPE_Q8_0:
            return { QBS_WEIGHT_ENCODING_S8_B32_F16_TWOS_COMPLEMENT,
                     QBS_ACTIVATION_ENCODING_S8_B32_F16_TWOS_COMPLEMENT };
        case GGML_TYPE_Q4_0:
            return { QBS_WEIGHT_ENCODING_S4_B32_F16_SPLIT_NIBBLE_OFFSET8,
                     QBS_ACTIVATION_ENCODING_S8_B32_F16_TWOS_COMPLEMENT };
        case GGML_TYPE_Q2_K:
            return { QBS_WEIGHT_ENCODING_Q2_K, QBS_ACTIVATION_ENCODING_Q8_K };
        case GGML_TYPE_Q5_0:
            return { QBS_WEIGHT_ENCODING_S5_B32_F16_NIBBLE_HIGHBIT_OFFSET16,
                     QBS_ACTIVATION_ENCODING_S8_B32_F16_TWOS_COMPLEMENT };
        case GGML_TYPE_IQ4_NL:
            return { QBS_WEIGHT_ENCODING_IQ4_NL, QBS_ACTIVATION_ENCODING_Q8_0 };
        default:
            return {};
    }
}

static qbs_profile profile_for_type(ggml_type type) {
    const qbs_encoding_pair encoding = encoding_for_type(type);
    return { qbs_weight_profile_from_encoding(encoding.weight),
             qbs_activation_profile_from_encoding(encoding.activation) };
}

static bool env_true(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "yes") == 0 || std::strcmp(value, "YES") == 0 || std::strcmp(value, "on") == 0 ||
           std::strcmp(value, "ON") == 0;
}

static bool qbs_emulation_enabled() {
    return env_true("GGML_RISCV_QBS_EMULATE");
}

static bool qbs_runtime_enabled() {
    return env_true("GGML_RISCV_QBS") || qbs_emulation_enabled();
}

static bool qbs_activation_context_enabled() {
    const char * value = std::getenv("GGML_RISCV_QBS_ACTIVATION_CONTEXT");
    return value == nullptr || env_true("GGML_RISCV_QBS_ACTIVATION_CONTEXT");
}

static bool format_name_equal(const char * token, size_t token_size, const char * expected) {
    const size_t expected_size = std::strlen(expected);
    if (token_size != expected_size) {
        return false;
    }
    for (size_t index = 0; index < token_size; ++index) {
        const unsigned char lhs            = static_cast<unsigned char>(token[index]);
        const unsigned char rhs            = static_cast<unsigned char>(expected[index]);
        const char          normalized_lhs = lhs == '-' ? '_' : std::toupper(lhs);
        const char          normalized_rhs = rhs == '-' ? '_' : std::toupper(rhs);
        if (normalized_lhs != normalized_rhs) {
            return false;
        }
    }
    return true;
}

static size_t trace_index(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q3_K:
            return 0;
        case GGML_TYPE_Q4_K:
            return 1;
        case GGML_TYPE_Q5_K:
            return 2;
        case GGML_TYPE_Q6_K:
            return 3;
        case GGML_TYPE_Q8_0:
            return 4;
        case GGML_TYPE_Q4_0:
            return 5;
        case GGML_TYPE_Q2_K:
            return 6;
        case GGML_TYPE_Q5_0:
            return 7;
        case GGML_TYPE_IQ4_NL:
            return 8;
        default:
            return QBS_TRACE_FORMATS;
    }
}

static const char * trace_type_name(size_t index) {
    static constexpr const char * names[QBS_TRACE_FORMATS] = {
        "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "Q2_K", "Q5_0", "IQ4_NL",
    };
    return names[index];
}

static bool qbs_format_enabled(ggml_type type) {
    const char * filter = std::getenv("GGML_RISCV_QBS_FORMATS");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }

    const size_t index = trace_index(type);
    if (index == QBS_TRACE_FORMATS) {
        return false;
    }
    const char * expected = trace_type_name(index);
    const char * cursor   = filter;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',' || *cursor == ';') {
            ++cursor;
        }
        const char * begin = cursor;
        while (*cursor != '\0' && *cursor != ' ' && *cursor != '\t' && *cursor != ',' && *cursor != ';') {
            ++cursor;
        }
        const size_t token_size = static_cast<size_t>(cursor - begin);
        if (format_name_equal(begin, token_size, expected) || format_name_equal(begin, token_size, "ALL") ||
            (token_size == 1 && begin[0] == '*')) {
            return true;
        }
    }
    return false;
}

static const char * support_reason_name(qbs_support_reason reason) {
    switch (reason) {
        case qbs_support_reason::supported:
            return "selected";
        case qbs_support_reason::runtime_disabled:
            return "runtime_disabled";
        case qbs_support_reason::format_filter:
            return "format_filter";
        case qbs_support_reason::capability:
            return "capability";
        case qbs_support_reason::dimensions:
            return "dimensions";
        case qbs_support_reason::shape:
            return "shape";
        case qbs_support_reason::layout:
            return "layout";
        case qbs_support_reason::profile:
            return "profile";
        case qbs_support_reason::dispatch:
            return "dispatch";
        case qbs_support_reason::not_candidate:
            return "not_candidate";
    }
    return "unknown";
}

static void print_trace_summary() {
    for (size_t index = 0; index < qbs_trace.size(); ++index) {
        const auto &   counters          = qbs_trace[index];
        const uint64_t gemv_calls        = counters.gemv_calls.load(std::memory_order_relaxed);
        const uint64_t gemm_calls        = counters.gemm_calls.load(std::memory_order_relaxed);
        const uint64_t candidate_tensors = counters.candidate_tensors.load(std::memory_order_relaxed);
        if (gemv_calls == 0 && gemm_calls == 0 && candidate_tensors == 0) {
            continue;
        }
        if (candidate_tensors != 0) {
            std::fprintf(
                stderr,
                "GGML_RISCV_QBS_COVERAGE type=%s candidate_tensors=%llu "
                "selected_tensors=%llu segmented_tensors=%llu "
                "candidate_elements=%llu selected_elements=%llu "
                "fallback_runtime=%llu fallback_format_filter=%llu "
                "fallback_capability=%llu "
                "fallback_dimensions=%llu fallback_shape=%llu "
                "fallback_layout=%llu fallback_profile=%llu "
                "fallback_dispatch=%llu\n",
                trace_type_name(index), static_cast<unsigned long long>(candidate_tensors),
                static_cast<unsigned long long>(counters.selected_tensors.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.segmented_tensors.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.candidate_elements.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.selected_elements.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.fallback_runtime.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.fallback_format_filter.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.fallback_capability.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.fallback_dimensions.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.fallback_shape.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.fallback_layout.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.fallback_profile.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(counters.fallback_dispatch.load(std::memory_order_relaxed)));
        }
        if (gemv_calls != 0 || gemm_calls != 0) {
            std::fprintf(stderr,
                         "GGML_RISCV_QBS_EXEC type=%s gemv_calls=%llu gemm_calls=%llu "
                         "input_rows=%llu output_rows=%llu dot_elements=%llu "
                         "split_calls=%llu commands_m1=%llu commands_m2=%llu "
                         "commands_m3=%llu commands_m4=%llu native_qbexec=%llu "
                         "emulated_commands=%llu command_dot_elements=%llu "
                         "segmented_commands=%llu context_fill=%llu "
                         "context_reuse=%llu context_release=%llu\n",
                         trace_type_name(index), static_cast<unsigned long long>(gemv_calls),
                         static_cast<unsigned long long>(gemm_calls),
                         static_cast<unsigned long long>(counters.input_rows.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.output_rows.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.dot_elements.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.split_calls.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.commands_by_m[0].load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.commands_by_m[1].load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.commands_by_m[2].load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.commands_by_m[3].load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.native_qbexec.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.emulated_commands.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.command_dot_elements.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(counters.segmented_commands.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(
                             counters.context_fill_commands.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(
                             counters.context_reuse_commands.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(
                             counters.context_release_commands.load(std::memory_order_relaxed)));
        }
    }
}

static void ensure_trace_registered() {
    static const bool registered = [] {
        std::atexit(print_trace_summary);
        return true;
    }();
    (void) registered;
}

static void record_operation(ggml_type type, bool gemm, int n, int input_rows, int output_rows, bool split_k) {
    if (!env_true("GGML_RISCV_QBS_TRACE")) {
        return;
    }
    ensure_trace_registered();
    const size_t index = trace_index(type);
    if (index == QBS_TRACE_FORMATS) {
        return;
    }
    auto & counters = qbs_trace[index];
    (gemm ? counters.gemm_calls : counters.gemv_calls).fetch_add(1, std::memory_order_relaxed);
    counters.input_rows.fetch_add(static_cast<uint64_t>(input_rows), std::memory_order_relaxed);
    counters.output_rows.fetch_add(static_cast<uint64_t>(output_rows), std::memory_order_relaxed);
    counters.dot_elements.fetch_add(
        static_cast<uint64_t>(n) * static_cast<uint64_t>(input_rows) * static_cast<uint64_t>(output_rows),
        std::memory_order_relaxed);
    if (split_k) {
        counters.split_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

static void record_command(ggml_type type,
                           unsigned  m,
                           unsigned  n,
                           unsigned  k_blocks,
                           unsigned  block_elements,
                           bool      emulated,
                           bool      segmented,
                           unsigned  activation_access) {
    if (!env_true("GGML_RISCV_QBS_TRACE")) {
        return;
    }
    ensure_trace_registered();
    const size_t index = trace_index(type);
    if (index == QBS_TRACE_FORMATS || m == 0 || m > QBS_MAX_M) {
        return;
    }
    auto & counters = qbs_trace[index];
    counters.commands_by_m[m - 1].fetch_add(1, std::memory_order_relaxed);
    (emulated ? counters.emulated_commands : counters.native_qbexec).fetch_add(1, std::memory_order_relaxed);
    counters.command_dot_elements.fetch_add(
        static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * static_cast<uint64_t>(k_blocks) * block_elements,
        std::memory_order_relaxed);
    if (segmented) {
        counters.segmented_commands.fetch_add(1, std::memory_order_relaxed);
    }
    switch (activation_access) {
        case QBS_ACTIVATION_ACCESS_FILL:
            counters.context_fill_commands.fetch_add(1, std::memory_order_relaxed);
            break;
        case QBS_ACTIVATION_ACCESS_REUSE:
            counters.context_reuse_commands.fetch_add(1, std::memory_order_relaxed);
            break;
        case QBS_ACTIVATION_ACCESS_RELEASE:
            counters.context_release_commands.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
    }
}

static qbs_device_t detect_device() {
    qbs_device_t device{};
    qbs_status_t status;
    if (qbs_emulation_enabled()) {
        status = qbs_device_init_reference(1024, &device);
        return status == QBS_STATUS_OK ? device : qbs_device_t{};
    }
    // Probing executes a custom opcode, so ordinary RVV systems must opt in.
    if (!qbs_runtime_enabled()) {
        return device;
    }
    status = qbs_device_query(qbs_native_info, nullptr, &device);
    return status == QBS_STATUS_OK ? device : qbs_device_t{};
}

static const qbs_device_t & device() {
    static const qbs_device_t value = detect_device();
    return value;
}

static bool profile_supported(ggml_type type) {
    const qbs_encoding_pair encoding = encoding_for_type(type);
    qbs_profile_binding_t   binding{};
    return encoding.weight != 0 && encoding.activation != 0 &&
           qbs_device_bind_encodings(&device(), encoding.weight, encoding.activation, &binding) == QBS_STATUS_OK;
}

static qbs_support_reason tensor_support_reason(ggml_type type, int64_t k_elements, int64_t output_rows) {
    const qbs_profile profile = profile_for_type(type);
    if (profile.weight == QBS_WEIGHT_PROFILE_INVALID) {
        return qbs_support_reason::not_candidate;
    }
    if (!qbs_runtime_enabled()) {
        return qbs_support_reason::runtime_disabled;
    }
    if (!qbs_format_enabled(type)) {
        return qbs_support_reason::format_filter;
    }

    const qbs_capabilities_t & caps = device().capabilities;
    if (!caps.valid || caps.max_m < QBS_MAX_M || caps.max_n < 4 || caps.max_k_blocks == 0) {
        return qbs_support_reason::capability;
    }
    const unsigned block_elements = qbs_weight_block_elements(profile.weight);
    if (block_elements == 0 || k_elements <= 0 || output_rows <= 0 || k_elements > std::numeric_limits<int>::max() ||
        output_rows > std::numeric_limits<int>::max() || k_elements % block_elements != 0) {
        return qbs_support_reason::shape;
    }
    if ((caps.weight_layouts & (UINT64_C(1) << QBS_WEIGHT_LAYOUT_R4_BLOCK_MAJOR)) == 0 ||
        (caps.activation_layouts & (UINT64_C(1) << QBS_ACTIVATION_LAYOUT_ROW_MAJOR)) == 0 ||
        (caps.activation_layouts & (UINT64_C(1) << QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED)) == 0) {
        return qbs_support_reason::layout;
    }
    if (!profile_supported(type)) {
        return qbs_support_reason::profile;
    }
    return qbs_support_reason::supported;
}

static void count_tensor_fallback(qbs_trace_counters & counters, qbs_support_reason reason) {
    switch (reason) {
        case qbs_support_reason::runtime_disabled:
            counters.fallback_runtime.fetch_add(1, std::memory_order_relaxed);
            break;
        case qbs_support_reason::format_filter:
            counters.fallback_format_filter.fetch_add(1, std::memory_order_relaxed);
            break;
        case qbs_support_reason::capability:
            counters.fallback_capability.fetch_add(1, std::memory_order_relaxed);
            break;
        case qbs_support_reason::dimensions:
            counters.fallback_dimensions.fetch_add(1, std::memory_order_relaxed);
            break;
        case qbs_support_reason::shape:
            counters.fallback_shape.fetch_add(1, std::memory_order_relaxed);
            break;
        case qbs_support_reason::layout:
            counters.fallback_layout.fetch_add(1, std::memory_order_relaxed);
            break;
        case qbs_support_reason::profile:
            counters.fallback_profile.fetch_add(1, std::memory_order_relaxed);
            break;
        case qbs_support_reason::dispatch:
            counters.fallback_dispatch.fetch_add(1, std::memory_order_relaxed);
            break;
        case qbs_support_reason::supported:
        case qbs_support_reason::not_candidate:
            break;
    }
}

static void dequantize_weight_block(ggml_type weight_type, const uint8_t * source, float * destination) {
    switch (weight_type) {
        case GGML_TYPE_Q3_K:
            dequantize_row_q3_K(reinterpret_cast<const block_q3_K *>(source), destination, QBS_Q3_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(source), destination, QBS_Q4_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q5_K:
            dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(source), destination, QBS_Q5_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(source), destination, QBS_Q6_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(source), destination,
                                QBS_Q8_0_WEIGHT_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q4_0:
            dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(source), destination, QBS_Q4_0_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q2_K:
            dequantize_row_q2_K(reinterpret_cast<const block_q2_K *>(source), destination, QBS_Q2_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q5_0:
            dequantize_row_q5_0(reinterpret_cast<const block_q5_0 *>(source), destination, QBS_Q5_0_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_IQ4_NL:
            dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(source), destination,
                                  QBS_IQ4_NL_BLOCK_ELEMENTS);
            return;
        default:
            assert(false);
    }
}

static void dequantize_activation_block(const qbs_profile & profile, const uint8_t * source, float * destination) {
    const unsigned block_elements = qbs_weight_block_elements(profile.weight);
    if (profile.activation == QBS_ACTIVATION_PROFILE_Q8_K) {
        dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(source), destination, block_elements);
    } else {
        assert(profile.activation == QBS_ACTIVATION_PROFILE_Q8_0);
        dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(source), destination, block_elements);
    }
}

// Emulation is a correctness oracle, so keep it independent of RVV dot code.
static float scalar_dot(const float * lhs, const float * rhs, unsigned count) {
    const volatile float * scalar_lhs = lhs;
    const volatile float * scalar_rhs = rhs;
    float                  result     = 0.0f;
    for (unsigned index = 0; index < count; ++index) {
        result += scalar_lhs[index] * scalar_rhs[index];
    }
    return result;
}

static const uint8_t * packed_weight_block(const uint8_t * weights,
                                           unsigned        k_blocks,
                                           unsigned        block_bytes,
                                           unsigned        row,
                                           unsigned        block) {
    const size_t block_index = ((static_cast<size_t>(row) / 4) * k_blocks + block) * 4 + row % 4;
    return weights + block_index * block_bytes;
}

static void extract_interleaved_activation(const qbs_profile & profile,
                                           const uint8_t *     source,
                                           unsigned            context,
                                           uint8_t *           destination) {
    const unsigned scale_bytes        = qbs_activation_scale_bytes(profile.activation);
    const unsigned quant_bytes        = qbs_activation_quant_bytes(profile.activation);
    const unsigned aux_count          = qbs_activation_aux_count(profile.activation);
    const unsigned aux_element_bytes  = qbs_activation_aux_element_bytes(profile.activation);
    const unsigned scale_region_bytes = 4 * scale_bytes;
    const unsigned quant_region_bytes = 4 * quant_bytes;

    std::memcpy(destination, source + context * scale_bytes, scale_bytes);
    for (unsigned byte = 0; byte < quant_bytes; ++byte) {
        destination[scale_bytes + byte] = source[scale_region_bytes + byte * 4 + context];
    }
    for (unsigned item = 0; item < aux_count; ++item) {
        const size_t source_offset =
            scale_region_bytes + quant_region_bytes + (static_cast<size_t>(item) * 4 + context) * aux_element_bytes;
        const size_t destination_offset = scale_bytes + quant_bytes + static_cast<size_t>(item) * aux_element_bytes;
        std::memcpy(destination + destination_offset, source + source_offset, aux_element_bytes);
    }
}

static void emulate_qbs(ggml_type    weight_type,
                        int          n,
                        float *      dst,
                        size_t       dst_stride,
                        const void * weights,
                        const void * activations,
                        int          input_rows,
                        int          output_rows,
                        bool         interleaved_activations) {
    const qbs_profile profile                = profile_for_type(weight_type);
    const unsigned    block_elements         = qbs_weight_block_elements(profile.weight);
    const unsigned    weight_block_bytes     = qbs_weight_block_bytes(profile.weight);
    const unsigned    activation_block_bytes = qbs_activation_block_bytes(profile.activation);
    assert(block_elements != 0 && input_rows > 0 && (!interleaved_activations || input_rows % 4 == 0));
    const unsigned                                      k_blocks         = static_cast<unsigned>(n / block_elements);
    const auto *                                        weight_bytes     = static_cast<const uint8_t *>(weights);
    const auto *                                        activation_bytes = static_cast<const uint8_t *>(activations);
    std::array<uint8_t, QBS_MAX_ACTIVATION_BLOCK_BYTES> activation_block{};
    std::array<float, QBS_Q4_K_BLOCK_ELEMENTS>          dequantized_weight{};
    std::array<float, QBS_Q4_K_BLOCK_ELEMENTS>          dequantized_activation{};

    for (int input = 0; input < input_rows; ++input) {
        for (int output = 0; output < output_rows; ++output) {
            float result = 0.0f;
            for (unsigned block = 0; block < k_blocks; ++block) {
                const uint8_t * activation;
                if (interleaved_activations) {
                    const uint8_t * group = activation_bytes + ((static_cast<size_t>(input) / 4 * k_blocks + block) *
                                                                4 * activation_block_bytes);
                    extract_interleaved_activation(profile, group, input % 4, activation_block.data());
                    activation = activation_block.data();
                } else {
                    activation =
                        activation_bytes + (static_cast<size_t>(input) * k_blocks + block) * activation_block_bytes;
                }
                dequantize_weight_block(weight_type,
                                        packed_weight_block(weight_bytes, k_blocks, weight_block_bytes, output, block),
                                        dequantized_weight.data());
                dequantize_activation_block(profile, activation, dequantized_activation.data());
                result += scalar_dot(dequantized_weight.data(), dequantized_activation.data(), block_elements);
            }
            dst[static_cast<size_t>(input) * dst_stride + output] = result;
        }
    }
}

struct qbs_execution_context {
    ggml_type weight_type;
    bool      emulated;
};

struct qbs_workspace_buffer {
    static constexpr size_t LOCAL_ELEMENTS = QBS_MAX_M * 4u;

    explicit qbs_workspace_buffer(size_t required_bytes) {
        if (required_bytes == 0) {
            return;
        }
        if (required_bytes <= sizeof(local)) {
            data     = local.data();
            capacity = sizeof(local);
            return;
        }
        dynamic.resize(required_bytes / sizeof(float) + (required_bytes % sizeof(float) != 0));
        data     = dynamic.data();
        capacity = dynamic.size() * sizeof(dynamic[0]);
    }

    void * pointer() { return data; }

    size_t size_bytes() const { return capacity; }

    std::array<float, LOCAL_ELEMENTS> local;
    std::vector<float>                dynamic;
    void *                            data     = nullptr;
    size_t                            capacity = 0;
};

static qbs_status_t execute_command(void *                      opaque,
                                    const qbs_descriptor_t *    descriptor,
                                    unsigned                    m,
                                    const void *                activations,
                                    float *                     output,
                                    size_t                      output_stride,
                                    unsigned                    n,
                                    int                         segmented) {
    const auto *                  context        = static_cast<const qbs_execution_context *>(opaque);
    const qbs_descriptor_fields_t fields         = qbs_unpack_descriptor_header(descriptor->header);
    const unsigned                block_elements = qbs_weight_block_elements(fields.weight_profile);
    if (block_elements == 0) {
        return QBS_STATUS_PROFILE;
    }

    qbs_status_t status = QBS_STATUS_OK;
    if (context->emulated) {
        emulate_qbs(context->weight_type, static_cast<int>(fields.k_blocks * block_elements), output, output_stride,
                    reinterpret_cast<const void *>(descriptor->weight_base), activations, static_cast<int>(m),
                    static_cast<int>(n), fields.activation_layout == QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED);
    } else {
        status = qbs_native_execute_command(nullptr, descriptor, m, activations, output, output_stride, n, segmented);
    }
    if (status == QBS_STATUS_OK) {
        record_command(context->weight_type, m, n, fields.k_blocks, block_elements, context->emulated, segmented,
                       fields.activation_access);
    }
    return status;
}

static qbs_status_t execute_plan(const qbs_plan_t &          plan,
                                 const void *                weights,
                                 size_t                      weight_bytes,
                                 const void *                activations,
                                 size_t                      activation_bytes,
                                 float *                     output,
                                 size_t                      output_capacity,
                                 size_t                      output_stride,
                                 void *                      workspace,
                                 size_t                      workspace_bytes,
                                 qbs_execution_context *     context) {
    if (!qbs_activation_context_enabled() || !qbs_plan_supports_activation_context(&plan)) {
        return qbs_execute(&plan, weights, weight_bytes, activations, activation_bytes, output, output_capacity,
                           output_stride, workspace, workspace_bytes, execute_command, context);
    }

    // The current hardware exposes one activation context. Keep FILL through
    // RELEASE atomic with respect to other GGML worker threads.
    static std::mutex context_mutex;
    static uint16_t   next_generation = 0;
    std::lock_guard<std::mutex> lock(context_mutex);

    const uint16_t generation_count = static_cast<uint16_t>(1u << plan.activation_context_generation_bits);
    const qbs_execution_options_t options = {
        1u,
        { 0u, static_cast<uint8_t>(next_generation) },
    };
    next_generation = static_cast<uint16_t>((next_generation + 1u) % generation_count);
    return qbs_execute_with_options(&plan, weights, weight_bytes, activations, activation_bytes, output,
                                    output_capacity, output_stride, workspace, workspace_bytes, &options,
                                    execute_command, context);
}

}  // namespace

bool ggml_riscv_qbs_supports_tensor(ggml_type type, int64_t k_elements, int64_t output_rows) {
    return tensor_support_reason(type, k_elements, output_rows) == qbs_support_reason::supported;
}

void ggml_riscv_qbs_trace_tensor_decision(const void * tensor_identity,
                                          const char * tensor_name,
                                          ggml_type    type,
                                          int          n_dims,
                                          int64_t      k_elements,
                                          int64_t      output_rows,
                                          int64_t      matrix_count,
                                          bool         selected) {
    if (!env_true("GGML_RISCV_QBS_TRACE")) {
        return;
    }

    const size_t index = trace_index(type);
    if (index == QBS_TRACE_FORMATS) {
        return;
    }

    static std::mutex                       trace_mutex;
    static std::unordered_set<const void *> traced_tensors;
    {
        std::lock_guard<std::mutex> lock(trace_mutex);
        if (!traced_tensors.insert(tensor_identity).second) {
            return;
        }
    }

    ensure_trace_registered();
    auto & counters = qbs_trace[index];
    counters.candidate_tensors.fetch_add(1, std::memory_order_relaxed);
    if ((n_dims == 2 || n_dims == 3) && k_elements > 0 && output_rows > 0 && matrix_count > 0) {
        counters.candidate_elements.fetch_add(static_cast<uint64_t>(k_elements) * static_cast<uint64_t>(output_rows) *
                                                  static_cast<uint64_t>(matrix_count),
                                              std::memory_order_relaxed);
    }

    qbs_support_reason reason;
    if (n_dims != 2 && n_dims != 3) {
        reason = qbs_support_reason::dimensions;
    } else if (n_dims == 3 && (matrix_count <= 0 || output_rows % 4 != 0)) {
        // R4 groups must not cross an expert boundary.
        reason = qbs_support_reason::layout;
    } else {
        reason = tensor_support_reason(type, k_elements, output_rows);
    }
    if (selected) {
        counters.selected_tensors.fetch_add(1, std::memory_order_relaxed);
        if (k_elements > 0 && output_rows > 0 && matrix_count > 0) {
            counters.selected_elements.fetch_add(static_cast<uint64_t>(k_elements) *
                                                     static_cast<uint64_t>(output_rows) *
                                                     static_cast<uint64_t>(matrix_count),
                                                 std::memory_order_relaxed);
        }
        const qbs_profile          profile        = profile_for_type(type);
        const qbs_capabilities_t & caps           = device().capabilities;
        const unsigned             block_elements = qbs_weight_block_elements(profile.weight);
        const uint64_t             k_blocks       = static_cast<uint64_t>(k_elements) / block_elements;
        if (k_blocks > std::min<unsigned>(caps.max_k_blocks, QBS_MAX_K_BLOCKS)) {
            counters.segmented_tensors.fetch_add(1, std::memory_order_relaxed);
        }
        if (reason != qbs_support_reason::supported) {
            reason = qbs_support_reason::dispatch;
            count_tensor_fallback(counters, reason);
        }
    } else {
        if (reason == qbs_support_reason::supported) {
            reason = qbs_support_reason::dispatch;
        }
        count_tensor_fallback(counters, reason);
    }

    if (env_true("GGML_RISCV_QBS_TRACE_TENSORS")) {
        std::fprintf(stderr,
                     "GGML_RISCV_QBS_TENSOR name=%s type=%s dims=%d "
                     "k=%lld rows=%lld matrices=%lld decision=%s segmented=%d\n",
                     tensor_name && tensor_name[0] ? tensor_name : "<unnamed>", ggml_type_name(type), n_dims,
                     static_cast<long long>(k_elements), static_cast<long long>(output_rows),
                     static_cast<long long>(matrix_count), selected ? "selected" : support_reason_name(reason),
                     selected && k_elements > 0 &&
                         static_cast<uint64_t>(k_elements) / qbs_weight_block_elements(profile_for_type(type).weight) >
                             std::min<unsigned>(device().capabilities.max_k_blocks, QBS_MAX_K_BLOCKS));
    }
}

void ggml_riscv_qbs_trace_activation_lifetime(const void * op_identity,
                                              const char * op_name,
                                              const void * weight_identity,
                                              const char * weight_name,
                                              ggml_type    weight_type,
                                              const void * source_tensor_identity,
                                              const char * source_name,
                                              const void * source_data,
                                              const void * quantized_data,
                                              size_t       quantized_bytes,
                                              ggml_type    activation_type,
                                              int64_t      activation_rows,
                                              int64_t      k_elements) {
    if (!env_true("GGML_RISCV_QBS_TRACE_LIFETIME") || quantized_data == nullptr || quantized_bytes == 0) {
        return;
    }

    constexpr size_t MAX_RECORDS        = 64;
    constexpr size_t MAX_SNAPSHOT_BYTES = 8u * 1024u * 1024u;
    constexpr size_t MAX_RETAINED_BYTES = 64u * 1024u * 1024u;

    const uint64_t digest = activation_digest(quantized_data, quantized_bytes);
    auto &         state  = activation_trace_state();
    ensure_activation_trace_registered();
    std::lock_guard<std::mutex> lock(state.mutex);

    const uint64_t sequence = ++state.sequence;
    ++state.quantizations;
    state.quantized_bytes += quantized_bytes;

    const qbs_activation_trace_record * exact_match = nullptr;
    for (auto record = state.records.rbegin(); record != state.records.rend(); ++record) {
        const bool same_source = record->source_tensor_identity == source_tensor_identity ||
                                 record->source_data == source_data;
        if (!same_source || record->activation_type != activation_type ||
            record->activation_rows != activation_rows || record->k_elements != k_elements ||
            record->quantized_data.size() != quantized_bytes || record->digest != digest) {
            continue;
        }
        if (std::memcmp(record->quantized_data.data(), quantized_data, quantized_bytes) == 0) {
            exact_match = &*record;
            break;
        }
    }

    const bool reusable = exact_match != nullptr;
    if (reusable) {
        ++state.exact_reuse_candidates;
        state.reusable_quantized_bytes += quantized_bytes;
    }

    std::fprintf(stderr,
                 "GGML_RISCV_QBS_LIFETIME seq=%llu op=%s weight=%s "
                 "weight_type=%s source=%s activation_type=%s rows=%lld k=%lld "
                 "bytes=%zu digest=%016llx match_seq=%llu match_weight=%s "
                 "same_tensor=%d same_data=%d reusable=%d op_id=%p weight_id=%p source_id=%p source_data=%p\n",
                 static_cast<unsigned long long>(sequence), trace_name(op_name).c_str(),
                 trace_name(weight_name).c_str(), ggml_type_name(weight_type), trace_name(source_name).c_str(),
                 ggml_type_name(activation_type), static_cast<long long>(activation_rows),
                 static_cast<long long>(k_elements), quantized_bytes, static_cast<unsigned long long>(digest),
                 static_cast<unsigned long long>(exact_match ? exact_match->sequence : 0),
                 exact_match ? exact_match->weight_name.c_str() : "-",
                 exact_match && exact_match->source_tensor_identity == source_tensor_identity,
                 exact_match && exact_match->source_data == source_data, reusable, op_identity, weight_identity,
                 source_tensor_identity, source_data);

    if (quantized_bytes > MAX_SNAPSHOT_BYTES) {
        return;
    }

    qbs_activation_trace_record record = {
        sequence,
        source_tensor_identity,
        source_data,
        activation_type,
        activation_rows,
        k_elements,
        digest,
        trace_name(weight_name),
        std::vector<uint8_t>(static_cast<const uint8_t *>(quantized_data),
                             static_cast<const uint8_t *>(quantized_data) + quantized_bytes),
    };
    state.retained_snapshot_bytes += record.quantized_data.size();
    state.records.emplace_back(std::move(record));
    while (state.records.size() > MAX_RECORDS || state.retained_snapshot_bytes > MAX_RETAINED_BYTES) {
        state.retained_snapshot_bytes -= state.records.front().quantized_data.size();
        state.records.pop_front();
    }
}

int ggml_riscv_qbs_repack_r4(ggml_type    type,
                             void *       dst,
                             size_t       dst_size,
                             const void * src,
                             size_t       src_size,
                             int64_t      output_rows,
                             int64_t      k_blocks) {
    const qbs_profile profile = profile_for_type(type);
    if (!dst || !src || output_rows <= 0 || k_blocks <= 0 || profile.weight == QBS_WEIGHT_PROFILE_INVALID) {
        return -1;
    }
    return qbs_repack_weight_r4(profile.weight, src, src_size, static_cast<size_t>(output_rows),
                                static_cast<size_t>(k_blocks), dst, dst_size) == QBS_STATUS_OK ?
               0 :
               -1;
}

void ggml_riscv_qbs_gemv(ggml_type    weight_type,
                         int          n,
                         float *      dst,
                         size_t       dst_stride,
                         const void * weights,
                         const void * activations,
                         int          input_rows,
                         int          output_rows) {
    const qbs_profile profile        = profile_for_type(weight_type);
    const unsigned    block_elements = qbs_weight_block_elements(profile.weight);
    GGML_ASSERT(input_rows == 1 && n > 0 && block_elements != 0 && n % static_cast<int>(block_elements) == 0);
    const qbs_problem_t problem = {
        static_cast<uint8_t>(profile.weight),
        static_cast<uint8_t>(profile.activation),
        static_cast<uint8_t>(QBS_WEIGHT_LAYOUT_R4_BLOCK_MAJOR),
        static_cast<uint8_t>(QBS_ACTIVATION_STORAGE_ROW_MAJOR),
        static_cast<uint32_t>(input_rows),
        static_cast<uint32_t>(output_rows),
        static_cast<uint32_t>(n),
    };
    qbs_plan_t         plan{};
    const qbs_status_t plan_status = qbs_plan_create(&device(), &problem, &plan);
    GGML_ASSERT(plan_status == QBS_STATUS_OK);
    record_operation(weight_type, false, n, input_rows, output_rows, plan.split_k);
    qbs_workspace_buffer  workspace(plan.workspace_bytes);
    qbs_execution_context context{ weight_type, qbs_emulation_enabled() };
    const size_t          weight_bytes = qbs_weight_storage_bytes(profile.weight, QBS_WEIGHT_LAYOUT_R4_BLOCK_MAJOR,
                                                                  static_cast<size_t>(output_rows), plan.k_blocks);
    const size_t activation_bytes = qbs_activation_storage_bytes(profile.activation, QBS_ACTIVATION_STORAGE_ROW_MAJOR,
                                                                 static_cast<size_t>(input_rows), plan.k_blocks);
    const size_t output_capacity  = static_cast<size_t>(input_rows - 1) * dst_stride + static_cast<size_t>(output_rows);
    const qbs_status_t status =
        execute_plan(plan, weights, weight_bytes, activations, activation_bytes, dst, output_capacity, dst_stride,
                     workspace.pointer(), workspace.size_bytes(), &context);
    GGML_ASSERT(status == QBS_STATUS_OK);
}

void ggml_riscv_qbs_gemm(ggml_type    weight_type,
                         int          n,
                         float *      dst,
                         size_t       dst_stride,
                         const void * weights,
                         const void * activations,
                         int          input_rows,
                         int          output_rows) {
    const qbs_profile profile        = profile_for_type(weight_type);
    const unsigned    block_elements = qbs_weight_block_elements(profile.weight);
    GGML_ASSERT(input_rows > 0 && n > 0 && block_elements != 0 && n % static_cast<int>(block_elements) == 0);
    const qbs_problem_t problem = {
        static_cast<uint8_t>(profile.weight),
        static_cast<uint8_t>(profile.activation),
        static_cast<uint8_t>(QBS_WEIGHT_LAYOUT_R4_BLOCK_MAJOR),
        static_cast<uint8_t>(QBS_ACTIVATION_STORAGE_M4_GROUPED),
        static_cast<uint32_t>(input_rows),
        static_cast<uint32_t>(output_rows),
        static_cast<uint32_t>(n),
    };
    qbs_plan_t         plan{};
    const qbs_status_t plan_status = qbs_plan_create(&device(), &problem, &plan);
    GGML_ASSERT(plan_status == QBS_STATUS_OK);
    record_operation(weight_type, true, n, input_rows, output_rows, plan.split_k);
    qbs_workspace_buffer  workspace(plan.workspace_bytes);
    qbs_execution_context context{ weight_type, qbs_emulation_enabled() };
    const size_t          weight_bytes = qbs_weight_storage_bytes(profile.weight, QBS_WEIGHT_LAYOUT_R4_BLOCK_MAJOR,
                                                                  static_cast<size_t>(output_rows), plan.k_blocks);
    const size_t activation_bytes = qbs_activation_storage_bytes(profile.activation, QBS_ACTIVATION_STORAGE_M4_GROUPED,
                                                                 static_cast<size_t>(input_rows), plan.k_blocks);
    const size_t output_capacity  = static_cast<size_t>(input_rows - 1) * dst_stride + static_cast<size_t>(output_rows);
    const qbs_status_t status =
        execute_plan(plan, weights, weight_bytes, activations, activation_bytes, dst, output_capacity, dst_stride,
                     workspace.pointer(), workspace.size_bytes(), &context);
    GGML_ASSERT(status == QBS_STATUS_OK);
}
