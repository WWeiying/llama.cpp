#include "qbs.h"
#include "qbs-layout.h"

#include "qbs_abi.h"
#include "quants.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
};

constexpr size_t QBS_TRACE_FORMATS = 6;
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

static size_t trace_index(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q3_K: return 0;
        case GGML_TYPE_Q4_K: return 1;
        case GGML_TYPE_Q5_K: return 2;
        case GGML_TYPE_Q6_K: return 3;
        case GGML_TYPE_Q8_0: return 4;
        case GGML_TYPE_Q4_0: return 5;
        default: return QBS_TRACE_FORMATS;
    }
}

static const char * trace_type_name(size_t index) {
    static constexpr const char * names[QBS_TRACE_FORMATS] = {
        "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0",
    };
    return names[index];
}

static void print_trace_summary() {
    for (size_t index = 0; index < qbs_trace.size(); ++index) {
        const auto & counters = qbs_trace[index];
        const uint64_t gemv_calls = counters.gemv_calls.load(std::memory_order_relaxed);
        const uint64_t gemm_calls = counters.gemm_calls.load(std::memory_order_relaxed);
        if (gemv_calls == 0 && gemm_calls == 0) {
            continue;
        }
        std::fprintf(stderr,
                     "GGML_RISCV_QBS_EXEC type=%s gemv_calls=%llu "
                     "gemm_calls=%llu input_rows=%llu output_rows=%llu "
                     "dot_elements=%llu\n",
                     trace_type_name(index),
                     static_cast<unsigned long long>(gemv_calls),
                     static_cast<unsigned long long>(gemm_calls),
                     static_cast<unsigned long long>(counters.input_rows.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(counters.output_rows.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(counters.dot_elements.load(std::memory_order_relaxed)));
    }
}

static void record_trace(ggml_type type, bool gemm, int n,
                         int input_rows, int output_rows) {
    if (!env_true("GGML_RISCV_QBS_TRACE")) {
        return;
    }
    static const bool registered = [] {
        std::atexit(print_trace_summary);
        return true;
    }();
    (void) registered;
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

using qbs_vec_dot_t = void (*)(int, float *, size_t, const void *, size_t,
                               const void *, size_t, int);

static qbs_vec_dot_t reference_dot(ggml_type weight_type) {
    switch (weight_type) {
        case GGML_TYPE_Q3_K: return ggml_vec_dot_q3_K_q8_K_generic;
        case GGML_TYPE_Q4_K: return ggml_vec_dot_q4_K_q8_K_generic;
        case GGML_TYPE_Q5_K: return ggml_vec_dot_q5_K_q8_K_generic;
        case GGML_TYPE_Q6_K: return ggml_vec_dot_q6_K_q8_K_generic;
        case GGML_TYPE_Q8_0: return ggml_vec_dot_q8_0_q8_0_generic;
        case GGML_TYPE_Q4_0: return ggml_vec_dot_q4_0_q8_0_generic;
        default: return nullptr;
    }
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
    const qbs_vec_dot_t dot = reference_dot(weight_type);
    assert(dot != nullptr && input_rows > 0 &&
           (!interleaved_activations || input_rows % 4 == 0));
    const unsigned k_blocks = static_cast<unsigned>(n / profile.block_elements);
    const auto * weight_bytes = static_cast<const uint8_t *>(weights);
    const auto * activation_bytes = static_cast<const uint8_t *>(activations);
    std::array<uint8_t, QBS_MAX_ACTIVATION_BLOCK_BYTES> activation_block{};

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
                float block_result = 0.0f;
                dot(profile.block_elements, &block_result, 0,
                    packed_weight_block(weight_bytes, k_blocks,
                                        profile.weight_block_bytes, output, block),
                    0, activation, 0, 1);
                result += block_result;
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

} // namespace

bool ggml_riscv_qbs_supports_tensor(ggml_type type, int64_t k_elements,
                                    int64_t output_rows) {
    const qbs_profile profile = profile_for_type(type);
    const qbs_capabilities & caps = capabilities();
    if (!caps.valid || caps.max_m < QBS_MAX_M || caps.max_n < 4 ||
        profile.block_elements == 0 || k_elements <= 0 || output_rows <= 0 ||
        k_elements % profile.block_elements != 0) {
        return false;
    }

    const uint64_t k_blocks = static_cast<uint64_t>(k_elements / profile.block_elements);
    if (k_blocks == 0 || k_blocks > caps.max_k_blocks ||
        (caps.weight_layouts & (UINT64_C(1) << QBS_WEIGHT_LAYOUT_R4_BLOCK_MAJOR)) == 0 ||
        (caps.activation_layouts & (UINT64_C(1) << QBS_ACTIVATION_LAYOUT_ROW_MAJOR)) == 0 ||
        (caps.activation_layouts & (UINT64_C(1) << QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED)) == 0) {
        return false;
    }
    return profile_supported(profile);
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
    record_trace(weight_type, false, n, input_rows, output_rows);
    (void) input_rows;
    const unsigned k_blocks = static_cast<unsigned>(n / profile.block_elements);
    const unsigned tile_capacity = std::min<unsigned>(caps.max_n, QBS_MAX_N) & ~3u;
    const auto * weight_bytes = static_cast<const uint8_t *>(weights);

    for (int output = 0; output < output_rows; output += tile_capacity) {
        const unsigned tile_n = std::min<unsigned>(tile_capacity, output_rows - output);
        const void * tile_weights = weight_bytes +
            static_cast<size_t>(output) * k_blocks * profile.weight_block_bytes;
        if (qbs_emulation_enabled()) {
            emulate_qbs(weight_type, n, dst + output, dst_stride, tile_weights,
                        activations, 1, tile_n, false);
        } else {
            const qbs_descriptor_v1_t descriptor __attribute__((aligned(16))) =
                make_descriptor(profile, QBS_ACTIVATION_LAYOUT_ROW_MAJOR,
                                tile_n, k_blocks, tile_weights);
            issue_m1(&descriptor, activations, dst + output, tile_n);
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
    assert(input_rows > 0 && input_rows % 4 == 0 && n > 0 &&
           n % static_cast<int>(profile.block_elements) == 0);
    record_trace(weight_type, true, n, input_rows, output_rows);
    const unsigned k_blocks = static_cast<unsigned>(n / profile.block_elements);
    const unsigned tile_capacity = std::min<unsigned>(caps.max_n, QBS_MAX_N) & ~3u;
    const auto * weight_bytes = static_cast<const uint8_t *>(weights);
    const auto * activation_bytes = static_cast<const uint8_t *>(activations);

    for (int input = 0; input < input_rows; input += 4) {
        const void * activation_group = activation_bytes +
            static_cast<size_t>(input) * k_blocks * profile.activation_block_bytes;
        for (int output = 0; output < output_rows; output += tile_capacity) {
            const unsigned tile_n = std::min<unsigned>(tile_capacity, output_rows - output);
            const void * tile_weights = weight_bytes +
                static_cast<size_t>(output) * k_blocks * profile.weight_block_bytes;
            if (qbs_emulation_enabled()) {
                emulate_qbs(
                    weight_type, n,
                    dst + static_cast<size_t>(input) * dst_stride + output,
                    dst_stride, tile_weights, activation_group, 4, tile_n, true);
            } else {
                const qbs_descriptor_v1_t descriptor __attribute__((aligned(16))) =
                    make_descriptor(profile, QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED,
                                    tile_n, k_blocks, tile_weights);
                issue_m4(&descriptor, activation_group,
                         dst + static_cast<size_t>(input + 0) * dst_stride + output,
                         dst + static_cast<size_t>(input + 1) * dst_stride + output,
                         dst + static_cast<size_t>(input + 2) * dst_stride + output,
                         dst + static_cast<size_t>(input + 3) * dst_stride + output,
                         tile_n);
            }
        }
    }
}
