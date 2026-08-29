#include "qbs.h"
#include "qbs-layout.h"

#include "qbs_abi.h"
#include "ggml-quants.h"
#include "quants.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace {

struct qbs_profile {
    unsigned weight;
    unsigned activation;
    unsigned block_elements;
    unsigned weight_block_bytes;
    unsigned activation_block_bytes;
};

struct qbs_capabilities {
    bool valid;
    unsigned max_m;
    unsigned max_n;
    unsigned max_k_blocks;
    uint64_t weight_layouts;
    uint64_t activation_layouts;
};

struct qbs_trace_counters {
    std::atomic<uint64_t> gemv_calls{0};
    std::atomic<uint64_t> gemm_calls{0};
    std::atomic<uint64_t> input_rows{0};
    std::atomic<uint64_t> output_rows{0};
    std::atomic<uint64_t> dot_elements{0};
    std::atomic<uint64_t> split_calls{0};
    std::array<std::atomic<uint64_t>, QBS_MAX_M> commands_by_m{};
    std::atomic<uint64_t> native_qbexec{0};
    std::atomic<uint64_t> emulated_commands{0};
    std::atomic<uint64_t> command_dot_elements{0};
    std::atomic<uint64_t> segmented_commands{0};

    std::atomic<uint64_t> candidate_tensors{0};
    std::atomic<uint64_t> selected_tensors{0};
    std::atomic<uint64_t> segmented_tensors{0};
    std::atomic<uint64_t> candidate_elements{0};
    std::atomic<uint64_t> selected_elements{0};
    std::atomic<uint64_t> fallback_runtime{0};
    std::atomic<uint64_t> fallback_format_filter{0};
    std::atomic<uint64_t> fallback_capability{0};
    std::atomic<uint64_t> fallback_dimensions{0};
    std::atomic<uint64_t> fallback_shape{0};
    std::atomic<uint64_t> fallback_layout{0};
    std::atomic<uint64_t> fallback_profile{0};
    std::atomic<uint64_t> fallback_dispatch{0};
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

constexpr size_t QBS_TRACE_FORMATS = 9;
static std::array<qbs_trace_counters, QBS_TRACE_FORMATS> qbs_trace{};

static qbs_profile profile_for_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return {QBS_WEIGHT_PROFILE_Q4_K, QBS_ACTIVATION_PROFILE_Q8_K,
                    QBS_Q4_K_BLOCK_ELEMENTS, QBS_Q4_K_BLOCK_BYTES,
                    QBS_Q8_K_BLOCK_BYTES};
        case GGML_TYPE_Q5_K:
            return {QBS_WEIGHT_PROFILE_Q5_K, QBS_ACTIVATION_PROFILE_Q8_K,
                    QBS_Q5_K_BLOCK_ELEMENTS, QBS_Q5_K_BLOCK_BYTES,
                    QBS_Q8_K_BLOCK_BYTES};
        case GGML_TYPE_Q6_K:
            return {QBS_WEIGHT_PROFILE_Q6_K, QBS_ACTIVATION_PROFILE_Q8_K,
                    QBS_Q6_K_BLOCK_ELEMENTS, QBS_Q6_K_BLOCK_BYTES,
                    QBS_Q8_K_BLOCK_BYTES};
        case GGML_TYPE_Q3_K:
            return {QBS_WEIGHT_PROFILE_Q3_K, QBS_ACTIVATION_PROFILE_Q8_K,
                    QBS_Q3_K_BLOCK_ELEMENTS, QBS_Q3_K_BLOCK_BYTES,
                    QBS_Q8_K_BLOCK_BYTES};
        case GGML_TYPE_Q8_0:
            return {QBS_WEIGHT_PROFILE_Q8_0_WEIGHT, QBS_ACTIVATION_PROFILE_Q8_0,
                    QBS_Q8_0_WEIGHT_BLOCK_ELEMENTS, QBS_Q8_0_WEIGHT_BLOCK_BYTES,
                    QBS_Q8_0_BLOCK_BYTES};
        case GGML_TYPE_Q4_0:
            return {QBS_WEIGHT_PROFILE_Q4_0, QBS_ACTIVATION_PROFILE_Q8_0,
                    QBS_Q4_0_BLOCK_ELEMENTS, QBS_Q4_0_BLOCK_BYTES,
                    QBS_Q8_0_BLOCK_BYTES};
        case GGML_TYPE_Q2_K:
            return {QBS_WEIGHT_PROFILE_Q2_K, QBS_ACTIVATION_PROFILE_Q8_K,
                    QBS_Q2_K_BLOCK_ELEMENTS, QBS_Q2_K_BLOCK_BYTES,
                    QBS_Q8_K_BLOCK_BYTES};
        case GGML_TYPE_Q5_0:
            return {QBS_WEIGHT_PROFILE_Q5_0, QBS_ACTIVATION_PROFILE_Q8_0,
                    QBS_Q5_0_BLOCK_ELEMENTS, QBS_Q5_0_BLOCK_BYTES,
                    QBS_Q8_0_BLOCK_BYTES};
        case GGML_TYPE_IQ4_NL:
            return {QBS_WEIGHT_PROFILE_IQ4_NL, QBS_ACTIVATION_PROFILE_Q8_0,
                    QBS_IQ4_NL_BLOCK_ELEMENTS, QBS_IQ4_NL_BLOCK_BYTES,
                    QBS_Q8_0_BLOCK_BYTES};
        default:
            return {};
    }
}

static uint64_t qbs_info(unsigned index) {
    register uintptr_t a0 asm("a0") = index;
    // qbinfo a0, a0 (custom-2, funct3=1).
    asm volatile(".word 0x0005155b" : "+r"(a0) : : "memory");
    return a0;
}

static bool env_true(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "yes") == 0 ||
           std::strcmp(value, "YES") == 0 ||
           std::strcmp(value, "on") == 0 ||
           std::strcmp(value, "ON") == 0;
}

static bool qbs_emulation_enabled() {
    return env_true("GGML_RISCV_QBS_EMULATE");
}

static bool qbs_runtime_enabled() {
    return env_true("GGML_RISCV_QBS") || qbs_emulation_enabled();
}

static bool format_name_equal(const char * token, size_t token_size,
                              const char * expected) {
    const size_t expected_size = std::strlen(expected);
    if (token_size != expected_size) {
        return false;
    }
    for (size_t index = 0; index < token_size; ++index) {
        const unsigned char lhs = static_cast<unsigned char>(token[index]);
        const unsigned char rhs = static_cast<unsigned char>(expected[index]);
        const char normalized_lhs = lhs == '-' ? '_' : std::toupper(lhs);
        const char normalized_rhs = rhs == '-' ? '_' : std::toupper(rhs);
        if (normalized_lhs != normalized_rhs) {
            return false;
        }
    }
    return true;
}

static size_t trace_index(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q3_K: return 0;
        case GGML_TYPE_Q4_K: return 1;
        case GGML_TYPE_Q5_K: return 2;
        case GGML_TYPE_Q6_K: return 3;
        case GGML_TYPE_Q8_0: return 4;
        case GGML_TYPE_Q4_0: return 5;
        case GGML_TYPE_Q2_K: return 6;
        case GGML_TYPE_Q5_0: return 7;
        case GGML_TYPE_IQ4_NL: return 8;
        default: return QBS_TRACE_FORMATS;
    }
}

static const char * trace_type_name(size_t index) {
    static constexpr const char * names[QBS_TRACE_FORMATS] = {
        "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0",
        "Q2_K", "Q5_0", "IQ4_NL",
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
    const char * cursor = filter;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',' ||
               *cursor == ';') {
            ++cursor;
        }
        const char * begin = cursor;
        while (*cursor != '\0' && *cursor != ' ' && *cursor != '\t' &&
               *cursor != ',' && *cursor != ';') {
            ++cursor;
        }
        const size_t token_size = static_cast<size_t>(cursor - begin);
        if (format_name_equal(begin, token_size, expected) ||
            format_name_equal(begin, token_size, "ALL") ||
            (token_size == 1 && begin[0] == '*')) {
            return true;
        }
    }
    return false;
}

static const char * support_reason_name(qbs_support_reason reason) {
    switch (reason) {
        case qbs_support_reason::supported:        return "selected";
        case qbs_support_reason::runtime_disabled: return "runtime_disabled";
        case qbs_support_reason::format_filter:    return "format_filter";
        case qbs_support_reason::capability:       return "capability";
        case qbs_support_reason::dimensions:       return "dimensions";
        case qbs_support_reason::shape:            return "shape";
        case qbs_support_reason::layout:           return "layout";
        case qbs_support_reason::profile:          return "profile";
        case qbs_support_reason::dispatch:         return "dispatch";
        case qbs_support_reason::not_candidate:    return "not_candidate";
    }
    return "unknown";
}

static void print_trace_summary() {
    for (size_t index = 0; index < qbs_trace.size(); ++index) {
        const auto & counters = qbs_trace[index];
        const uint64_t gemv_calls = counters.gemv_calls.load(std::memory_order_relaxed);
        const uint64_t gemm_calls = counters.gemm_calls.load(std::memory_order_relaxed);
        const uint64_t candidate_tensors =
            counters.candidate_tensors.load(std::memory_order_relaxed);
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
                trace_type_name(index),
                static_cast<unsigned long long>(candidate_tensors),
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
            std::fprintf(
                stderr,
                "GGML_RISCV_QBS_EXEC type=%s gemv_calls=%llu gemm_calls=%llu "
                "input_rows=%llu output_rows=%llu dot_elements=%llu "
                "split_calls=%llu commands_m1=%llu commands_m2=%llu "
                "commands_m3=%llu commands_m4=%llu native_qbexec=%llu "
                "emulated_commands=%llu command_dot_elements=%llu "
                "segmented_commands=%llu\n",
                trace_type_name(index),
                static_cast<unsigned long long>(gemv_calls),
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
                static_cast<unsigned long long>(counters.segmented_commands.load(std::memory_order_relaxed)));
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

static void record_operation(ggml_type type, bool gemm, int n,
                             int input_rows, int output_rows,
                             bool split_k) {
    if (!env_true("GGML_RISCV_QBS_TRACE")) {
        return;
    }
    ensure_trace_registered();
    const size_t index = trace_index(type);
    if (index == QBS_TRACE_FORMATS) {
        return;
    }
    auto & counters = qbs_trace[index];
    (gemm ? counters.gemm_calls : counters.gemv_calls)
        .fetch_add(1, std::memory_order_relaxed);
    counters.input_rows.fetch_add(static_cast<uint64_t>(input_rows),
                                  std::memory_order_relaxed);
    counters.output_rows.fetch_add(static_cast<uint64_t>(output_rows),
                                   std::memory_order_relaxed);
    counters.dot_elements.fetch_add(
        static_cast<uint64_t>(n) * static_cast<uint64_t>(input_rows) *
            static_cast<uint64_t>(output_rows),
        std::memory_order_relaxed);
    if (split_k) {
        counters.split_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

static void record_command(ggml_type type, unsigned m, unsigned n,
                           unsigned k_blocks, unsigned block_elements,
                           bool emulated, bool segmented) {
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
    (emulated ? counters.emulated_commands : counters.native_qbexec)
        .fetch_add(1, std::memory_order_relaxed);
    counters.command_dot_elements.fetch_add(
        static_cast<uint64_t>(m) * static_cast<uint64_t>(n) *
            static_cast<uint64_t>(k_blocks) * block_elements,
        std::memory_order_relaxed);
    if (segmented) {
        counters.segmented_commands.fetch_add(1, std::memory_order_relaxed);
    }
}

static qbs_capabilities detect_capabilities() {
    if (qbs_emulation_enabled()) {
        return {
            true,
            QBS_MAX_M,
            QBS_MAX_N,
            QBS_MAX_K_BLOCKS,
            UINT64_C(1) << QBS_WEIGHT_LAYOUT_R4_BLOCK_MAJOR,
            (UINT64_C(1) << QBS_ACTIVATION_LAYOUT_ROW_MAJOR) |
                (UINT64_C(1) << QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED),
        };
    }
    // Probing executes a custom opcode, so ordinary RVV systems must opt in.
    if (!qbs_runtime_enabled()) {
        return {};
    }

    const uint64_t info0 = qbs_info(0);
    const uint64_t info1 = qbs_info(1);
    const bool valid = ((info0 >> 0) & 0xffu) == QBS_ARCH_VERSION &&
                       ((info0 >> 8) & 0xffu) == QBS_DESCRIPTOR_VERSION &&
                       ((info0 >> 16) & 0xffu) == QBS_DESCRIPTOR_BYTES &&
                       ((info0 >> 39) & 0x0fu) == QBS_NUMERICAL_CONTRACT_VERSION;
    if (!valid) {
        return {};
    }

    return {
        true,
        static_cast<unsigned>(((info0 >> 24) & 0x03u) + 1u),
        static_cast<unsigned>(((info0 >> 26) & 0x1fu) + 1u),
        static_cast<unsigned>(((info0 >> 31) & 0xffu) + 1u),
        info1 & 0xffffu,
        (info1 >> 16) & 0xffffu,
    };
}

static const qbs_capabilities & capabilities() {
    static const qbs_capabilities value = detect_capabilities();
    return value;
}

static bool profile_supported(const qbs_profile & profile) {
    if (profile.weight == QBS_WEIGHT_PROFILE_INVALID) {
        return false;
    }
    if (qbs_emulation_enabled()) {
        return true;
    }
    const uint64_t activation_profiles = qbs_info(0x10u + profile.weight);
    return (activation_profiles & (UINT64_C(1) << profile.activation)) != 0;
}

static qbs_support_reason tensor_support_reason(ggml_type type,
                                                int64_t k_elements,
                                                int64_t output_rows) {
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

    const qbs_capabilities & caps = capabilities();
    if (!caps.valid || caps.max_m < QBS_MAX_M || caps.max_n < 4 ||
        caps.max_k_blocks == 0) {
        return qbs_support_reason::capability;
    }
    if (profile.block_elements == 0 || k_elements <= 0 || output_rows <= 0 ||
        k_elements > std::numeric_limits<int>::max() ||
        output_rows > std::numeric_limits<int>::max() ||
        k_elements % profile.block_elements != 0) {
        return qbs_support_reason::shape;
    }
    if ((caps.weight_layouts &
         (UINT64_C(1) << QBS_WEIGHT_LAYOUT_R4_BLOCK_MAJOR)) == 0 ||
        (caps.activation_layouts &
         (UINT64_C(1) << QBS_ACTIVATION_LAYOUT_ROW_MAJOR)) == 0 ||
        (caps.activation_layouts &
         (UINT64_C(1) << QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED)) == 0) {
        return qbs_support_reason::layout;
    }
    if (!profile_supported(profile)) {
        return qbs_support_reason::profile;
    }
    return qbs_support_reason::supported;
}

static void count_tensor_fallback(qbs_trace_counters & counters,
                                  qbs_support_reason reason) {
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

static void dequantize_weight_block(ggml_type weight_type,
                                    const uint8_t * source,
                                    float * destination) {
    switch (weight_type) {
        case GGML_TYPE_Q3_K:
            dequantize_row_q3_K(reinterpret_cast<const block_q3_K *>(source),
                                destination, QBS_Q3_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(source),
                                destination, QBS_Q4_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q5_K:
            dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(source),
                                destination, QBS_Q5_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(source),
                                destination, QBS_Q6_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(source),
                                destination, QBS_Q8_0_WEIGHT_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q4_0:
            dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(source),
                                destination, QBS_Q4_0_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q2_K:
            dequantize_row_q2_K(reinterpret_cast<const block_q2_K *>(source),
                                destination, QBS_Q2_K_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_Q5_0:
            dequantize_row_q5_0(reinterpret_cast<const block_q5_0 *>(source),
                                destination, QBS_Q5_0_BLOCK_ELEMENTS);
            return;
        case GGML_TYPE_IQ4_NL:
            dequantize_row_iq4_nl(
                reinterpret_cast<const block_iq4_nl *>(source), destination,
                QBS_IQ4_NL_BLOCK_ELEMENTS);
            return;
        default:
            assert(false);
    }
}

static void dequantize_activation_block(const qbs_profile & profile,
                                        const uint8_t * source,
                                        float * destination) {
    if (profile.activation == QBS_ACTIVATION_PROFILE_Q8_K) {
        dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(source),
                            destination, profile.block_elements);
    } else {
        assert(profile.activation == QBS_ACTIVATION_PROFILE_Q8_0);
        dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(source),
                            destination, profile.block_elements);
    }
}

// Emulation is a correctness oracle, so keep it independent of RVV dot code.
static float scalar_dot(const float * lhs, const float * rhs, unsigned count) {
    const volatile float * scalar_lhs = lhs;
    const volatile float * scalar_rhs = rhs;
    float result = 0.0f;
    for (unsigned index = 0; index < count; ++index) {
        result += scalar_lhs[index] * scalar_rhs[index];
    }
    return result;
}

static const uint8_t * packed_weight_block(const uint8_t * weights,
                                            unsigned k_blocks,
                                            unsigned block_bytes,
                                            unsigned row, unsigned block) {
    const size_t block_index =
        ((static_cast<size_t>(row) / 4) * k_blocks + block) * 4 + row % 4;
    return weights + block_index * block_bytes;
}

static void extract_interleaved_activation(
        const qbs_profile & profile, const uint8_t * source, unsigned context,
        uint8_t * destination) {
    const unsigned scale_bytes = qbs_activation_scale_bytes(profile.activation);
    const unsigned quant_bytes = qbs_activation_quant_bytes(profile.activation);
    const unsigned aux_count = qbs_activation_aux_count(profile.activation);
    const unsigned aux_element_bytes =
        qbs_activation_aux_element_bytes(profile.activation);
    const unsigned scale_region_bytes = 4 * scale_bytes;
    const unsigned quant_region_bytes = 4 * quant_bytes;

    std::memcpy(destination, source + context * scale_bytes, scale_bytes);
    for (unsigned byte = 0; byte < quant_bytes; ++byte) {
        destination[scale_bytes + byte] =
            source[scale_region_bytes + byte * 4 + context];
    }
    for (unsigned item = 0; item < aux_count; ++item) {
        const size_t source_offset = scale_region_bytes + quant_region_bytes +
            (static_cast<size_t>(item) * 4 + context) * aux_element_bytes;
        const size_t destination_offset =
            scale_bytes + quant_bytes +
            static_cast<size_t>(item) * aux_element_bytes;
        std::memcpy(destination + destination_offset, source + source_offset,
                    aux_element_bytes);
    }
}

static void emulate_qbs(ggml_type weight_type, int n, float * dst,
                        size_t dst_stride, const void * weights,
                        const void * activations, int input_rows,
                        int output_rows, bool interleaved_activations) {
    const qbs_profile profile = profile_for_type(weight_type);
    assert(profile.block_elements != 0 && input_rows > 0 &&
           (!interleaved_activations || input_rows % 4 == 0));
    const unsigned k_blocks = static_cast<unsigned>(n / profile.block_elements);
    const auto * weight_bytes = static_cast<const uint8_t *>(weights);
    const auto * activation_bytes = static_cast<const uint8_t *>(activations);
    std::array<uint8_t, QBS_MAX_ACTIVATION_BLOCK_BYTES> activation_block{};
    std::array<float, QBS_Q4_K_BLOCK_ELEMENTS> dequantized_weight{};
    std::array<float, QBS_Q4_K_BLOCK_ELEMENTS> dequantized_activation{};

    for (int input = 0; input < input_rows; ++input) {
        for (int output = 0; output < output_rows; ++output) {
            float result = 0.0f;
            for (unsigned block = 0; block < k_blocks; ++block) {
                const uint8_t * activation;
                if (interleaved_activations) {
                    const uint8_t * group =
                        ggml::cpu::riscv::qbs::activation_m4_group_block(
                            activation_bytes, static_cast<size_t>(input), block,
                            k_blocks, profile.activation_block_bytes);
                    extract_interleaved_activation(profile, group, input % 4,
                                                   activation_block.data());
                    activation = activation_block.data();
                } else {
                    activation = activation_bytes +
                        (static_cast<size_t>(input) * k_blocks + block) *
                            profile.activation_block_bytes;
                }
                dequantize_weight_block(
                    weight_type,
                    packed_weight_block(weight_bytes, k_blocks,
                                        profile.weight_block_bytes, output, block),
                    dequantized_weight.data());
                dequantize_activation_block(profile, activation,
                                            dequantized_activation.data());
                result += scalar_dot(dequantized_weight.data(),
                                     dequantized_activation.data(),
                                     profile.block_elements);
            }
            dst[static_cast<size_t>(input) * dst_stride + output] = result;
        }
    }
}

static void issue_m1(const qbs_descriptor_v1_t * descriptor,
                     const void * activation, float * output, unsigned n) {
    register uintptr_t a0 asm("a0") = reinterpret_cast<uintptr_t>(descriptor);
    register uintptr_t a1 asm("a1") = reinterpret_cast<uintptr_t>(activation);
    register uintptr_t a2 asm("a2") = reinterpret_cast<uintptr_t>(output);
    asm volatile(
        "fence rw, rw\n"
        "li t0, 32\n"
        "vsetvli zero, t0, e32, m1, ta, ma\n"
        ".word 0x00b5045b\n"
        "mv t0, %[n]\n"
        "vsetvli zero, t0, e32, m1, ta, ma\n"
        "vse32.v v8, (a2)\n"
        : "+r"(a0), "+r"(a1), "+r"(a2)
        : [n] "r"(n)
        : "t0", "v8", "memory");
}

static void issue_m2(const qbs_descriptor_v1_t * descriptor,
                     const void * activation, float * output0,
                     float * output1, unsigned n) {
    register uintptr_t a0 asm("a0") = reinterpret_cast<uintptr_t>(descriptor);
    register uintptr_t a1 asm("a1") = reinterpret_cast<uintptr_t>(activation);
    register uintptr_t a2 asm("a2") = reinterpret_cast<uintptr_t>(output0);
    register uintptr_t a3 asm("a3") = reinterpret_cast<uintptr_t>(output1);
    asm volatile(
        "fence rw, rw\n"
        "li t0, 64\n"
        "vsetvli zero, t0, e32, m2, ta, ma\n"
        ".word 0x02b5045b\n"
        "mv t0, %[n]\n"
        "vsetvli zero, t0, e32, m1, ta, ma\n"
        "vse32.v v8, (a2)\n"
        "vse32.v v9, (a3)\n"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : [n] "r"(n)
        : "t0", "v8", "v9", "memory");
}

static void issue_m3(const qbs_descriptor_v1_t * descriptor,
                     const void * activation, float * output0,
                     float * output1, float * output2, unsigned n) {
    register uintptr_t a0 asm("a0") = reinterpret_cast<uintptr_t>(descriptor);
    register uintptr_t a1 asm("a1") = reinterpret_cast<uintptr_t>(activation);
    register uintptr_t a2 asm("a2") = reinterpret_cast<uintptr_t>(output0);
    register uintptr_t a3 asm("a3") = reinterpret_cast<uintptr_t>(output1);
    register uintptr_t a4 asm("a4") = reinterpret_cast<uintptr_t>(output2);
    asm volatile(
        "fence rw, rw\n"
        "li t0, 128\n"
        "vsetvli zero, t0, e32, m4, ta, ma\n"
        ".word 0x04b5045b\n"
        "mv t0, %[n]\n"
        "vsetvli zero, t0, e32, m1, ta, ma\n"
        "vse32.v v8, (a2)\n"
        "vse32.v v9, (a3)\n"
        "vse32.v v10, (a4)\n"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4)
        : [n] "r"(n)
        : "t0", "v8", "v9", "v10", "v11", "memory");
}

static void issue_m4(const qbs_descriptor_v1_t * descriptor,
                     const void * activation, float * output0,
                     float * output1, float * output2, float * output3,
                     unsigned n) {
    register uintptr_t a0 asm("a0") = reinterpret_cast<uintptr_t>(descriptor);
    register uintptr_t a1 asm("a1") = reinterpret_cast<uintptr_t>(activation);
    register uintptr_t a2 asm("a2") = reinterpret_cast<uintptr_t>(output0);
    register uintptr_t a3 asm("a3") = reinterpret_cast<uintptr_t>(output1);
    register uintptr_t a4 asm("a4") = reinterpret_cast<uintptr_t>(output2);
    register uintptr_t a5 asm("a5") = reinterpret_cast<uintptr_t>(output3);
    asm volatile(
        "fence rw, rw\n"
        "li t0, 128\n"
        "vsetvli zero, t0, e32, m4, ta, ma\n"
        ".word 0x06b5045b\n"
        "mv t0, %[n]\n"
        "vsetvli zero, t0, e32, m1, ta, ma\n"
        "vse32.v v8, (a2)\n"
        "vse32.v v9, (a3)\n"
        "vse32.v v10, (a4)\n"
        "vse32.v v11, (a5)\n"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5)
        : [n] "r"(n)
        : "t0", "v8", "v9", "v10", "v11", "memory");
}

static qbs_descriptor_v1_t make_descriptor(const qbs_profile & profile,
                                            unsigned activation_layout,
                                            unsigned n, unsigned k_blocks,
                                            const void * weights) {
    const qbs_descriptor_fields_t fields = {
        QBS_DESCRIPTOR_VERSION,
        static_cast<uint8_t>(profile.weight),
        static_cast<uint8_t>(profile.activation),
        QBS_WEIGHT_LAYOUT_R4_BLOCK_MAJOR,
        static_cast<uint8_t>(activation_layout),
        static_cast<uint8_t>(n),
        static_cast<uint16_t>(k_blocks),
    };
    return {qbs_pack_descriptor_header(&fields),
            reinterpret_cast<uintptr_t>(weights)};
}

static unsigned command_k_capacity(const qbs_capabilities & caps) {
    return std::min<unsigned>(caps.max_k_blocks, QBS_MAX_K_BLOCKS);
}

static void execute_command(ggml_type weight_type,
                            const qbs_profile & profile,
                            unsigned m, unsigned n, unsigned k_blocks,
                            const void * weights, const void * activations,
                            unsigned activation_layout,
                            float * output, size_t output_stride,
                            bool segmented) {
    assert(m >= 1 && m <= QBS_MAX_M && n >= 1 && n <= QBS_MAX_N &&
           k_blocks >= 1 && k_blocks <= QBS_MAX_K_BLOCKS);

    if (qbs_emulation_enabled()) {
        emulate_qbs(weight_type,
                    static_cast<int>(k_blocks * profile.block_elements),
                    output, output_stride, weights, activations,
                    static_cast<int>(m), static_cast<int>(n),
                    activation_layout == QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED);
    } else {
        const qbs_descriptor_v1_t descriptor __attribute__((aligned(16))) =
            make_descriptor(profile, activation_layout, n, k_blocks, weights);
        switch (m) {
            case 1:
                issue_m1(&descriptor, activations, output, n);
                break;
            case 2:
                issue_m2(&descriptor, activations, output,
                         output + output_stride, n);
                break;
            case 3:
                issue_m3(&descriptor, activations, output,
                         output + output_stride,
                         output + 2 * output_stride, n);
                break;
            case 4:
                issue_m4(&descriptor, activations, output,
                         output + output_stride,
                         output + 2 * output_stride,
                         output + 3 * output_stride, n);
                break;
            default:
                assert(false);
        }
    }
    record_command(weight_type, m, n, k_blocks, profile.block_elements,
                   qbs_emulation_enabled(), segmented);
}

} // namespace

bool ggml_riscv_qbs_supports_tensor(ggml_type type, int64_t k_elements,
                                    int64_t output_rows) {
    return tensor_support_reason(type, k_elements, output_rows) ==
           qbs_support_reason::supported;
}

void ggml_riscv_qbs_trace_tensor_decision(const void * tensor_identity,
                                          const char * tensor_name,
                                          ggml_type type, int n_dims,
                                          int64_t k_elements,
                                          int64_t output_rows,
                                          int64_t matrix_count,
                                          bool selected) {
    if (!env_true("GGML_RISCV_QBS_TRACE")) {
        return;
    }

    const size_t index = trace_index(type);
    if (index == QBS_TRACE_FORMATS) {
        return;
    }

    static std::mutex trace_mutex;
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
    if ((n_dims == 2 || n_dims == 3) && k_elements > 0 &&
        output_rows > 0 && matrix_count > 0) {
        counters.candidate_elements.fetch_add(
            static_cast<uint64_t>(k_elements) *
                static_cast<uint64_t>(output_rows) *
                static_cast<uint64_t>(matrix_count),
            std::memory_order_relaxed);
    }

    qbs_support_reason reason;
    if (n_dims != 2 && n_dims != 3) {
        reason = qbs_support_reason::dimensions;
    } else if (n_dims == 3 &&
               (matrix_count <= 0 || output_rows % 4 != 0)) {
        // R4 groups must not cross an expert boundary.
        reason = qbs_support_reason::layout;
    } else {
        reason = tensor_support_reason(type, k_elements, output_rows);
    }
    if (selected) {
        counters.selected_tensors.fetch_add(1, std::memory_order_relaxed);
        if (k_elements > 0 && output_rows > 0 && matrix_count > 0) {
            counters.selected_elements.fetch_add(
                static_cast<uint64_t>(k_elements) *
                    static_cast<uint64_t>(output_rows) *
                    static_cast<uint64_t>(matrix_count),
                std::memory_order_relaxed);
        }
        const qbs_profile profile = profile_for_type(type);
        const qbs_capabilities & caps = capabilities();
        const uint64_t k_blocks =
            static_cast<uint64_t>(k_elements) / profile.block_elements;
        if (k_blocks > command_k_capacity(caps)) {
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
                     tensor_name && tensor_name[0] ? tensor_name : "<unnamed>",
                     ggml_type_name(type), n_dims,
                     static_cast<long long>(k_elements),
                     static_cast<long long>(output_rows),
                     static_cast<long long>(matrix_count),
                     selected ? "selected" : support_reason_name(reason),
                     selected && k_elements > 0 &&
                         static_cast<uint64_t>(k_elements) /
                                 profile_for_type(type).block_elements >
                             command_k_capacity(capabilities()));
    }
}

int ggml_riscv_qbs_repack_r4(void * dst, size_t dst_size, const void * src,
                             size_t src_size, int64_t output_rows,
                             int64_t k_blocks, size_t block_bytes) {
    if (!dst || !src || output_rows <= 0 || k_blocks <= 0 ||
        block_bytes == 0) {
        return -1;
    }
    const size_t source_size = static_cast<size_t>(output_rows) *
                               static_cast<size_t>(k_blocks) * block_bytes;
    const size_t padded_rows = (static_cast<size_t>(output_rows) + 3) & ~size_t(3);
    const size_t destination_size = padded_rows *
                                    static_cast<size_t>(k_blocks) * block_bytes;
    if (dst_size < destination_size || src_size != source_size) {
        return -1;
    }

    ggml::cpu::riscv::qbs::repack_r4(
        static_cast<uint8_t *>(dst), static_cast<const uint8_t *>(src),
        static_cast<size_t>(output_rows), static_cast<size_t>(k_blocks),
        block_bytes);
    return 0;
}

void ggml_riscv_qbs_gemv(ggml_type weight_type, int n, float * dst,
                          size_t dst_stride, const void * weights,
                          const void * activations, int input_rows,
                          int output_rows) {
    const qbs_profile profile = profile_for_type(weight_type);
    const qbs_capabilities & caps = capabilities();
    assert(input_rows == 1 && n > 0 && n % static_cast<int>(profile.block_elements) == 0);
    const unsigned k_capacity = command_k_capacity(caps);
    assert(k_capacity > 0);
    (void) input_rows;
    const unsigned k_blocks = static_cast<unsigned>(n / profile.block_elements);
    const bool split_k = k_blocks > k_capacity;
    record_operation(weight_type, false, n, input_rows, output_rows, split_k);
    const unsigned tile_capacity = std::min<unsigned>(caps.max_n, QBS_MAX_N) & ~3u;
    const auto * weight_bytes = static_cast<const uint8_t *>(weights);
    const auto * activation_bytes = static_cast<const uint8_t *>(activations);

    if (!split_k) {
        for (int output = 0; output < output_rows; output += tile_capacity) {
            const unsigned tile_n =
                std::min<unsigned>(tile_capacity, output_rows - output);
            const void * tile_weights = weight_bytes +
                static_cast<size_t>(output) * k_blocks * profile.weight_block_bytes;
            execute_command(weight_type, profile, 1, tile_n, k_blocks,
                            tile_weights, activations,
                            QBS_ACTIVATION_LAYOUT_ROW_MAJOR,
                            dst + output, dst_stride, false);
        }
        return;
    }

    // A segment of the existing R4 layout is contiguous only within one
    // four-row weight group. Keep N<=4 on this functional fallback path.
    for (int output = 0; output < output_rows; output += 4) {
        const unsigned tile_n = std::min<unsigned>(4, output_rows - output);
        bool first_segment = true;
        for (unsigned block = 0; block < k_blocks; block += k_capacity) {
            const unsigned segment_blocks =
                std::min<unsigned>(k_capacity, k_blocks - block);
            const void * segment_weights = weight_bytes +
                (static_cast<size_t>(output) * k_blocks +
                 static_cast<size_t>(block) * 4) * profile.weight_block_bytes;
            const void * segment_activations = activation_bytes +
                static_cast<size_t>(block) * profile.activation_block_bytes;
            std::array<float, 4> partial{};
            execute_command(weight_type, profile, 1, tile_n, segment_blocks,
                            segment_weights, segment_activations,
                            QBS_ACTIVATION_LAYOUT_ROW_MAJOR,
                            partial.data(), partial.size(), true);
            for (unsigned column = 0; column < tile_n; ++column) {
                if (first_segment) {
                    dst[output + column] = partial[column];
                } else {
                    dst[output + column] += partial[column];
                }
            }
            first_segment = false;
        }
    }
    (void) dst_stride;
}

void ggml_riscv_qbs_gemm(ggml_type weight_type, int n, float * dst,
                          size_t dst_stride, const void * weights,
                          const void * activations, int input_rows,
                          int output_rows) {
    const qbs_profile profile = profile_for_type(weight_type);
    const qbs_capabilities & caps = capabilities();
    assert(input_rows > 0 && n > 0 &&
           n % static_cast<int>(profile.block_elements) == 0);
    const unsigned k_capacity = command_k_capacity(caps);
    assert(k_capacity > 0);
    const unsigned k_blocks = static_cast<unsigned>(n / profile.block_elements);
    const bool split_k = k_blocks > k_capacity;
    record_operation(weight_type, true, n, input_rows, output_rows, split_k);
    const unsigned tile_capacity = std::min<unsigned>(caps.max_n, QBS_MAX_N) & ~3u;
    const auto * weight_bytes = static_cast<const uint8_t *>(weights);
    const auto * activation_bytes = static_cast<const uint8_t *>(activations);

    for (int input = 0; input < input_rows;) {
        const unsigned m = std::min<unsigned>(4, input_rows - input);
        const unsigned activation_layout = m == 4
            ? QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED
            : QBS_ACTIVATION_LAYOUT_ROW_MAJOR;
        const void * activation_group = activation_bytes +
            static_cast<size_t>(input) * k_blocks * profile.activation_block_bytes;

        if (!split_k) {
            for (int output = 0; output < output_rows; output += tile_capacity) {
                const unsigned tile_n =
                    std::min<unsigned>(tile_capacity, output_rows - output);
                const void * tile_weights = weight_bytes +
                    static_cast<size_t>(output) * k_blocks *
                        profile.weight_block_bytes;
                execute_command(
                    weight_type, profile, m, tile_n, k_blocks,
                    tile_weights, activation_group, activation_layout,
                    dst + static_cast<size_t>(input) * dst_stride + output,
                    dst_stride, false);
            }
            input += m;
            continue;
        }

        std::vector<uint8_t> activation_segment;
        if (m != 4) {
            activation_segment.resize(static_cast<size_t>(m) * k_capacity *
                                      profile.activation_block_bytes);
        }

        bool first_segment = true;
        for (unsigned block = 0; block < k_blocks; block += k_capacity) {
            const unsigned segment_blocks =
                std::min<unsigned>(k_capacity, k_blocks - block);
            const void * segment_activations;
            if (m == 4) {
                segment_activations =
                    static_cast<const uint8_t *>(activation_group) +
                    static_cast<size_t>(block) * 4 *
                        profile.activation_block_bytes;
            } else {
                const size_t segment_bytes =
                    static_cast<size_t>(segment_blocks) *
                    profile.activation_block_bytes;
                for (unsigned context = 0; context < m; ++context) {
                    const auto * source =
                        static_cast<const uint8_t *>(activation_group) +
                        (static_cast<size_t>(context) * k_blocks + block) *
                            profile.activation_block_bytes;
                    std::memcpy(activation_segment.data() +
                                    static_cast<size_t>(context) * segment_bytes,
                                source, segment_bytes);
                }
                segment_activations = activation_segment.data();
            }

            for (int output = 0; output < output_rows; output += 4) {
                const unsigned tile_n =
                    std::min<unsigned>(4, output_rows - output);
                const void * segment_weights = weight_bytes +
                    (static_cast<size_t>(output) * k_blocks +
                     static_cast<size_t>(block) * 4) *
                        profile.weight_block_bytes;
                std::array<float, QBS_MAX_M * 4> partial{};
                execute_command(weight_type, profile, m, tile_n,
                                segment_blocks, segment_weights,
                                segment_activations, activation_layout,
                                partial.data(), 4, true);
                for (unsigned context = 0; context < m; ++context) {
                    for (unsigned column = 0; column < tile_n; ++column) {
                        float & destination =
                            dst[static_cast<size_t>(input + context) *
                                    dst_stride +
                                output + column];
                        const float value = partial[context * 4 + column];
                        if (first_segment) {
                            destination = value;
                        } else {
                            destination += value;
                        }
                    }
                }
            }
            first_segment = false;
        }
        input += m;
    }
}
