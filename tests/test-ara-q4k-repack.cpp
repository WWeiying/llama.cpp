#include "repack.h"
#include "arch/riscv/qbs-layout.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "quants.h"
#if defined(GGML_TEST_RISCV_QBS)
#include "arch/riscv/qbs.h"
#endif

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__riscv) && !defined(__riscv_zvfh)
extern "C" void ggml_quantize_mat_q8_0_4x1(
    const float * source, void * destination, int64_t k);
#endif

static block_q4_K make_source_block(size_t row, size_t block) {
    block_q4_K result;
    uint8_t * bytes = reinterpret_cast<uint8_t *>(&result);

    for (size_t i = 0; i < sizeof(result); ++i) {
        bytes[i] = (uint8_t) ((row * 97 + block * 53 + i * 29) & 0xff);
    }

    return result;
}

static block_q4_K recover_source_block(const block_q4_Kx32_ara & packed, size_t row) {
    block_q4_K result = {};
    result.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = packed.d[row];
    result.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin = packed.dmin[row];

    for (size_t byte = 0; byte < K_SCALE_SIZE; ++byte) {
        result.scales[byte] = packed.scales[byte * 32 + row];
    }
    for (size_t byte = 0; byte < QK_K / 2; ++byte) {
        result.qs[byte] = packed.qs[byte * 32 + row];
    }

    return result;
}

static void quantize_q8_0_m4(const float * source, void * destination,
                             int64_t k) {
#if defined(__riscv)
    ggml_quantize_mat_q8_0_4x1(source, destination, k);
#else
    ggml_quantize_mat_q8_0_4x1_generic(source, destination, k);
#endif
}

static void test_qbs_r4_layout() {
    constexpr size_t blocks = 65;
    for (const size_t rows : {1, 2, 3, 4, 5, 7, 8, 9}) {
        for (const size_t block_bytes : {
                 sizeof(block_q4_0), sizeof(block_q5_0), sizeof(block_iq4_nl),
                 sizeof(block_q2_K), sizeof(block_q3_K), sizeof(block_q4_K),
                 sizeof(block_q5_K), sizeof(block_q6_K), sizeof(block_q8_0)}) {
            std::vector<uint8_t> source(rows * blocks * block_bytes);
            const size_t padded_rows = (rows + 3) & ~size_t(3);
            std::vector<uint8_t> packed(
                padded_rows * blocks * block_bytes, 0xa5);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t block = 0; block < blocks; ++block) {
                    for (size_t byte = 0; byte < block_bytes; ++byte) {
                        source[(row * blocks + block) * block_bytes + byte] =
                            (uint8_t) ((row * 71 + block * 37 + byte * 13) & 0xff);
                    }
                }
            }

            ggml::cpu::riscv::qbs::repack_r4(
                packed.data(), source.data(), rows, blocks, block_bytes);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t block = 0; block < blocks; ++block) {
                    const size_t source_index = row * blocks + block;
                    const size_t packed_index =
                        ((row / 4) * blocks + block) * 4 + row % 4;
                    assert(std::memcmp(
                               source.data() + source_index * block_bytes,
                               packed.data() + packed_index * block_bytes,
                               block_bytes) == 0);
                }
            }
            for (size_t row = rows; row < padded_rows; ++row) {
                for (size_t block = 0; block < blocks; ++block) {
                    const size_t packed_index =
                        ((row / 4) * blocks + block) * 4 + row % 4;
                    for (size_t byte = 0; byte < block_bytes; ++byte) {
                        assert(packed[packed_index * block_bytes + byte] == 0);
                    }
                }
            }
        }
    }
    std::printf("QBS R4 layout: PASS\n");
}

static void test_qbs_activation_layouts() {
    constexpr size_t ninputs = 8;
    constexpr int64_t q8_0_k = 3 * QK8_0;
    constexpr int64_t q8_k = 2 * QK_K;
    std::vector<float> q8_0_input(ninputs * q8_0_k);
    std::vector<float> q8_k_input(ninputs * q8_k);
    for (size_t i = 0; i < q8_0_input.size(); ++i) {
        q8_0_input[i] = 0.75f * std::sin(float(i * 17) * 0.03125f);
    }
    for (size_t i = 0; i < q8_k_input.size(); ++i) {
        q8_k_input[i] = 0.625f * std::cos(float(i * 11) * 0.015625f);
    }

    const size_t q8_0_blocks = q8_0_k / QK8_0;
    std::vector<block_q8_0x4> actual_q8_0((ninputs / 4) * q8_0_blocks);
    std::vector<block_q8_0> reference_q8_0(ninputs * q8_0_blocks);
    for (size_t group = 0; group < ninputs / 4; ++group) {
        quantize_q8_0_m4(
            q8_0_input.data() + group * 4 * q8_0_k,
            actual_q8_0.data() + group * q8_0_blocks, q8_0_k);
    }
    for (size_t context = 0; context < ninputs; ++context) {
        quantize_row_q8_0_ref(
            q8_0_input.data() + context * q8_0_k,
            reference_q8_0.data() + context * q8_0_blocks, q8_0_k);
    }
    for (size_t context = 0; context < ninputs; ++context) {
        for (size_t block = 0; block < q8_0_blocks; ++block) {
            const auto * group_block = reinterpret_cast<const block_q8_0x4 *>(
                ggml::cpu::riscv::qbs::activation_m4_group_block(
                    reinterpret_cast<const uint8_t *>(actual_q8_0.data()),
                    context, block, q8_0_blocks, sizeof(block_q8_0)));
            const block_q8_0 & reference =
                reference_q8_0[context * q8_0_blocks + block];
            const size_t lane = context % 4;
            assert(group_block->d[lane] == reference.d);
            for (size_t byte = 0; byte < QK8_0; ++byte) {
                assert(group_block->qs[byte * 4 + lane] ==
                       reference.qs[byte]);
            }
        }
    }

    const size_t q8_k_blocks = q8_k / QK_K;
    std::vector<block_q8_Kx4> actual_q8_k((ninputs / 4) * q8_k_blocks);
    std::vector<block_q8_K> reference_q8_k(ninputs * q8_k_blocks);
    for (size_t group = 0; group < ninputs / 4; ++group) {
        ggml_quantize_mat_q8_K_4x1_ara_generic(
            q8_k_input.data() + group * 4 * q8_k,
            actual_q8_k.data() + group * q8_k_blocks, q8_k);
    }
    for (size_t context = 0; context < ninputs; ++context) {
        quantize_row_q8_K_ref(
            q8_k_input.data() + context * q8_k,
            reference_q8_k.data() + context * q8_k_blocks, q8_k);
    }
    for (size_t context = 0; context < ninputs; ++context) {
        for (size_t block = 0; block < q8_k_blocks; ++block) {
            const auto * group_block = reinterpret_cast<const block_q8_Kx4 *>(
                ggml::cpu::riscv::qbs::activation_m4_group_block(
                    reinterpret_cast<const uint8_t *>(actual_q8_k.data()),
                    context, block, q8_k_blocks, sizeof(block_q8_K)));
            const block_q8_K & reference =
                reference_q8_k[context * q8_k_blocks + block];
            const size_t lane = context % 4;
            assert(group_block->d[lane] == reference.d);
            for (size_t byte = 0; byte < QK_K; ++byte) {
                assert(group_block->qs[byte * 4 + lane] ==
                       reference.qs[byte]);
            }
            for (size_t item = 0; item < QK_K / 16; ++item) {
                assert(group_block->bsums[item * 4 + lane] ==
                       reference.bsums[item]);
            }
        }
    }
    std::printf("QBS M4 activation layouts: PASS\n");
}

static void test_numeric_gemv() {
    constexpr int64_t nrows = 64;
    constexpr int64_t n = 3 * QK_K;
    constexpr size_t blocks_per_row = n / QK_K;

    std::vector<float> source_f32(nrows * n);
    std::vector<float> activation_f32(n);
    for (size_t row = 0; row < (size_t) nrows; ++row) {
        for (size_t col = 0; col < (size_t) n; ++col) {
            source_f32[row * n + col] =
                    0.75f * std::sin(float(row * 13 + col * 7) * 0.03125f) -
                    0.25f * std::cos(float(row * 3 + col * 11) * 0.015625f);
        }
    }
    for (size_t col = 0; col < (size_t) n; ++col) {
        activation_f32[col] =
                0.5f * std::sin(float(col * 5) * 0.0234375f) +
                0.375f * std::cos(float(col * 9) * 0.01171875f);
    }

    std::vector<block_q4_K> source(nrows * blocks_per_row);
    for (size_t row = 0; row < (size_t) nrows; ++row) {
        quantize_row_q4_K_ref(source_f32.data() + row * n,
                              source.data() + row * blocks_per_row, n);
    }

    std::vector<block_q8_K> activation(blocks_per_row);
    quantize_row_q8_K_ref(activation_f32.data(), activation.data(), n);

    std::vector<block_q4_Kx32_ara> packed((nrows / 32) * blocks_per_row);
    assert(ggml_repack_q4_K_32x1_ara(
            packed.data(), packed.size() * sizeof(packed[0]),
            source.data(), source.size() * sizeof(source[0]), nrows, n) == 0);

    std::vector<float> expected(nrows);
    std::vector<float> actual(nrows);
    for (size_t row = 0; row < (size_t) nrows; ++row) {
        ggml_vec_dot_q4_K_q8_K_generic(
                n, &expected[row], sizeof(float),
                source.data() + row * blocks_per_row, 0,
                activation.data(), 0, 1);
    }
    ggml_gemv_q4_K_32x1_q8_K_generic(
            n, actual.data(), nrows, packed.data(), activation.data(), 1, nrows);

    float max_error = 0.0f;
    for (size_t row = 0; row < (size_t) nrows; ++row) {
        const float error = std::fabs(actual[row] - expected[row]);
        max_error = std::max(max_error, error);
        assert(std::isfinite(actual[row]));
        assert(error <= 1e-5f + 1e-5f * std::fabs(expected[row]));
    }
    std::printf("Ara Q4_Kx32 GEMV reference: PASS (max abs error %.8g)\n", max_error);

#if defined(__riscv)
    std::fill(actual.begin(), actual.end(), 0.0f);
    ggml_gemv_q4_K_32x1_q8_K(
            n, actual.data(), nrows, packed.data(), activation.data(), 1, nrows);
    max_error = 0.0f;
    for (size_t row = 0; row < (size_t) nrows; ++row) {
        const float error = std::fabs(actual[row] - expected[row]);
        max_error = std::max(max_error, error);
        assert(std::isfinite(actual[row]));
        assert(error <= 1e-4f + 1e-5f * std::fabs(expected[row]));
    }
    std::printf("Ara Q4_Kx32 RVV GEMV: PASS (max abs error %.8g)\n", max_error);
#endif
}

static void test_numeric_gemm() {
    constexpr int64_t nrows = 64;
    constexpr int64_t n = 3 * QK_K;
    constexpr int64_t ninputs = 4;
    constexpr size_t blocks_per_row = n / QK_K;

    std::vector<float> source_f32(nrows * n);
    std::vector<float> activation_f32(ninputs * n);
    for (size_t row = 0; row < (size_t) nrows; ++row) {
        for (size_t col = 0; col < (size_t) n; ++col) {
            source_f32[row * n + col] =
                    0.625f * std::sin(float(row * 17 + col * 5) * 0.02734375f) -
                    0.3125f * std::cos(float(row * 7 + col * 3) * 0.01953125f);
        }
    }
    for (size_t input = 0; input < (size_t) ninputs; ++input) {
        for (size_t col = 0; col < (size_t) n; ++col) {
            activation_f32[input * n + col] =
                    0.4375f * std::sin(float(input * 19 + col * 11) * 0.017578125f) +
                    0.5625f * std::cos(float(input * 23 + col * 13) * 0.009765625f);
        }
    }

    std::vector<block_q4_K> source(nrows * blocks_per_row);
    for (size_t row = 0; row < (size_t) nrows; ++row) {
        quantize_row_q4_K_ref(source_f32.data() + row * n,
                              source.data() + row * blocks_per_row, n);
    }

    std::vector<block_q4_Kx32_ara> packed((nrows / 32) * blocks_per_row);
    assert(ggml_repack_q4_K_32x1_ara(
            packed.data(), packed.size() * sizeof(packed[0]),
            source.data(), source.size() * sizeof(source[0]), nrows, n) == 0);

    std::vector<block_q8_Kx4> activation_packed(blocks_per_row);
    ggml_quantize_mat_q8_K_4x1_ara_generic(
            activation_f32.data(), activation_packed.data(), n);

    std::vector<float> expected(ninputs * nrows);
    std::vector<block_q8_K> activation(blocks_per_row);
    for (size_t input = 0; input < (size_t) ninputs; ++input) {
        quantize_row_q8_K_ref(
                activation_f32.data() + input * n, activation.data(), n);
        for (size_t row = 0; row < (size_t) nrows; ++row) {
            ggml_vec_dot_q4_K_q8_K_generic(
                    n, &expected[input * nrows + row], sizeof(float),
                    source.data() + row * blocks_per_row, 0,
                    activation.data(), 0, 1);
        }
    }

    std::vector<float> actual(ninputs * nrows);
    ggml_gemm_q4_K_32x1_q8_K_generic(
            n, actual.data(), nrows, packed.data(), activation_packed.data(), ninputs, nrows);

    float max_error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);
        max_error = std::max(max_error, error);
        assert(std::isfinite(actual[index]));
        assert(error <= 1e-5f + 1e-5f * std::fabs(expected[index]));
    }
    std::printf("Ara Q4_Kx32 GEMM reference: PASS (max abs error %.8g)\n", max_error);

#if defined(__riscv)
    std::fill(actual.begin(), actual.end(), 0.0f);
    ggml_gemm_q4_K_32x1_q8_K(
            n, actual.data(), nrows, packed.data(), activation_packed.data(), ninputs, nrows);
    max_error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);
        max_error = std::max(max_error, error);
        assert(std::isfinite(actual[index]));
        assert(error <= 1e-4f + 1e-5f * std::fabs(expected[index]));
    }
    std::printf("Ara Q4_Kx32 RVV GEMM: PASS (max abs error %.8g)\n", max_error);
#endif
}

#if defined(GGML_TEST_RISCV_QBS)
using qbs_quantize_row_t = void (*)(const float *, void *, int64_t);

struct qbs_test_profile {
    ggml_type type;
    const char * name;
    size_t block_elements;
    size_t weight_block_bytes;
    size_t activation_block_bytes;
    bool activation_q8_0;
    qbs_quantize_row_t quantize_weight;
};

static void qbs_dequantize_weight(ggml_type type, const void * source,
                                  float * destination, int64_t n) {
    switch (type) {
        case GGML_TYPE_Q3_K:
            dequantize_row_q3_K(static_cast<const block_q3_K *>(source), destination, n);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K(static_cast<const block_q4_K *>(source), destination, n);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_row_q5_K(static_cast<const block_q5_K *>(source), destination, n);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K(static_cast<const block_q6_K *>(source), destination, n);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0(static_cast<const block_q8_0 *>(source), destination, n);
            break;
        case GGML_TYPE_Q4_0:
            dequantize_row_q4_0(static_cast<const block_q4_0 *>(source), destination, n);
            break;
        case GGML_TYPE_Q2_K:
            dequantize_row_q2_K(static_cast<const block_q2_K *>(source), destination, n);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_row_q5_0(static_cast<const block_q5_0 *>(source), destination, n);
            break;
        case GGML_TYPE_IQ4_NL:
            dequantize_row_iq4_nl(static_cast<const block_iq4_nl *>(source), destination, n);
            break;
        default:
            assert(false);
    }
}

// Volatile loads keep this test oracle independent of RVV dot-product code.
static float qbs_scalar_dot(const float * lhs, const float * rhs, int64_t n) {
    const volatile float * scalar_lhs = lhs;
    const volatile float * scalar_rhs = rhs;
    float result = 0.0f;
    for (int64_t index = 0; index < n; ++index) {
        result += scalar_lhs[index] * scalar_rhs[index];
    }
    return result;
}

static void quantize_test_q3_k(const float * source, void * destination,
                               int64_t n) {
    quantize_row_q3_K_ref(source, static_cast<block_q3_K *>(destination), n);
}

static void quantize_test_q4_k(const float * source, void * destination,
                               int64_t n) {
    quantize_row_q4_K_ref(source, static_cast<block_q4_K *>(destination), n);
}

static void quantize_test_q5_k(const float * source, void * destination,
                               int64_t n) {
    quantize_row_q5_K_ref(source, static_cast<block_q5_K *>(destination), n);
}

static void quantize_test_q6_k(const float * source, void * destination,
                               int64_t n) {
    quantize_row_q6_K_ref(source, static_cast<block_q6_K *>(destination), n);
}

static void quantize_test_q8_0(const float * source, void * destination,
                               int64_t n) {
    quantize_row_q8_0_ref(source, static_cast<block_q8_0 *>(destination), n);
}

static void quantize_test_q4_0(const float * source, void * destination,
                               int64_t n) {
    quantize_row_q4_0_ref(source, static_cast<block_q4_0 *>(destination), n);
}

static void quantize_test_q2_k(const float * source, void * destination,
                               int64_t n) {
    quantize_row_q2_K_ref(source, static_cast<block_q2_K *>(destination), n);
}

static void quantize_test_q5_0(const float * source, void * destination,
                               int64_t n) {
    quantize_row_q5_0_ref(source, static_cast<block_q5_0 *>(destination), n);
}

static void quantize_test_iq4_nl(const float * source, void * destination,
                                 int64_t n) {
    quantize_row_iq4_nl_ref(source, static_cast<block_iq4_nl *>(destination), n);
}

static void test_qbs_emulated_profiles() {
    constexpr int64_t nrows = 35;
    constexpr int64_t ninputs = 7;
    static const qbs_test_profile profiles[] = {
        {GGML_TYPE_Q3_K, "Q3_K", QK_K, sizeof(block_q3_K), sizeof(block_q8_K),
         false, quantize_test_q3_k},
        {GGML_TYPE_Q4_K, "Q4_K", QK_K, sizeof(block_q4_K), sizeof(block_q8_K),
         false, quantize_test_q4_k},
        {GGML_TYPE_Q5_K, "Q5_K", QK_K, sizeof(block_q5_K), sizeof(block_q8_K),
         false, quantize_test_q5_k},
        {GGML_TYPE_Q6_K, "Q6_K", QK_K, sizeof(block_q6_K), sizeof(block_q8_K),
         false, quantize_test_q6_k},
        {GGML_TYPE_Q8_0, "Q8_0", QK8_0, sizeof(block_q8_0), sizeof(block_q8_0),
         true, quantize_test_q8_0},
        {GGML_TYPE_Q4_0, "Q4_0", QK4_0, sizeof(block_q4_0), sizeof(block_q8_0),
         true, quantize_test_q4_0},
        {GGML_TYPE_Q2_K, "Q2_K", QK_K, sizeof(block_q2_K), sizeof(block_q8_K),
         false, quantize_test_q2_k},
        {GGML_TYPE_Q5_0, "Q5_0", QK5_0, sizeof(block_q5_0), sizeof(block_q8_0),
         true, quantize_test_q5_0},
        {GGML_TYPE_IQ4_NL, "IQ4_NL", QK4_NL, sizeof(block_iq4_nl), sizeof(block_q8_0),
         true, quantize_test_iq4_nl},
    };

    const bool native = std::getenv("GGML_RISCV_QBS") != nullptr;
    if (!native && std::getenv("GGML_RISCV_QBS_EMULATE") == nullptr) {
        setenv("GGML_RISCV_QBS_EMULATE", "1", 1);
    }
    for (const qbs_test_profile & profile : profiles) {
        const int64_t n = 2 * profile.block_elements;
        const size_t blocks_per_row = 2;
        std::vector<float> source_f32(nrows * n);
        std::vector<float> activation_f32(ninputs * n);
        for (size_t row = 0; row < (size_t) nrows; ++row) {
            for (size_t col = 0; col < (size_t) n; ++col) {
                source_f32[row * n + col] =
                    0.625f * std::sin(float(row * 23 + col * 7) * 0.01953125f);
            }
        }
        for (size_t input = 0; input < (size_t) ninputs; ++input) {
            for (size_t col = 0; col < (size_t) n; ++col) {
                activation_f32[input * n + col] =
                    0.5f * std::cos(float(input * 17 + col * 11) * 0.015625f);
            }
        }

        std::vector<uint8_t> source(
            nrows * blocks_per_row * profile.weight_block_bytes);
        for (size_t row = 0; row < (size_t) nrows; ++row) {
            profile.quantize_weight(
                source_f32.data() + row * n,
                source.data() + row * blocks_per_row * profile.weight_block_bytes,
                n);
        }
        const size_t padded_rows = (nrows + 3) & ~size_t(3);
        std::vector<uint8_t> packed(
            padded_rows * blocks_per_row * profile.weight_block_bytes);
        assert(ggml_riscv_qbs_repack_r4(
                   profile.type, packed.data(), packed.size(), source.data(),
                   source.size(), nrows, blocks_per_row) == 0);

        std::vector<uint8_t> activation_packed(
            ninputs * blocks_per_row * profile.activation_block_bytes);
        for (size_t group = 0; group < (size_t) ninputs / 4; ++group) {
            void * destination = activation_packed.data() +
                group * 4 * blocks_per_row * profile.activation_block_bytes;
            if (profile.activation_q8_0) {
                quantize_q8_0_m4(
                    activation_f32.data() + group * 4 * n, destination, n);
            } else {
                ggml_quantize_mat_q8_K_4x1_ara_generic(
                    activation_f32.data() + group * 4 * n, destination, n);
            }
        }

        std::vector<uint8_t> activation(
            ninputs * blocks_per_row * profile.activation_block_bytes);
        std::vector<float> expected(ninputs * nrows);
        std::vector<float> dequantized_weight(n);
        std::vector<float> dequantized_activation(n);
        for (size_t input = 0; input < (size_t) ninputs; ++input) {
            uint8_t * activation_row = activation.data() +
                input * blocks_per_row * profile.activation_block_bytes;
            if (profile.activation_q8_0) {
                quantize_row_q8_0_ref(
                    activation_f32.data() + input * n,
                    reinterpret_cast<block_q8_0 *>(activation_row), n);
            } else {
                quantize_row_q8_K_ref(
                    activation_f32.data() + input * n,
                    reinterpret_cast<block_q8_K *>(activation_row), n);
            }
            if (profile.activation_q8_0) {
                dequantize_row_q8_0(
                    reinterpret_cast<const block_q8_0 *>(activation_row),
                    dequantized_activation.data(), n);
            } else {
                dequantize_row_q8_K(
                    reinterpret_cast<const block_q8_K *>(activation_row),
                    dequantized_activation.data(), n);
            }
            for (size_t row = 0; row < (size_t) nrows; ++row) {
                qbs_dequantize_weight(
                    profile.type,
                    source.data() + row * blocks_per_row * profile.weight_block_bytes,
                    dequantized_weight.data(), n);
                expected[input * nrows + row] = qbs_scalar_dot(
                    dequantized_weight.data(), dequantized_activation.data(), n);
            }
        }
        const size_t full_groups = static_cast<size_t>(ninputs) / 4;
        for (size_t input = full_groups * 4;
             input < static_cast<size_t>(ninputs); ++input) {
            std::memcpy(
                activation_packed.data() +
                    input * blocks_per_row * profile.activation_block_bytes,
                activation.data() +
                    input * blocks_per_row * profile.activation_block_bytes,
                blocks_per_row * profile.activation_block_bytes);
        }

        std::vector<float> actual(ninputs * nrows);
        ggml_riscv_qbs_gemm(profile.type, n, actual.data(), nrows,
                            packed.data(), activation_packed.data(), ninputs,
                            nrows);
        if (std::getenv("GGML_TEST_QBS_DIAGNOSTIC") != nullptr) {
            std::fprintf(stderr,
                         "QBS %s diagnostic GEMM[0,0]: actual=%.9g expected=%.9g\n",
                         profile.name, actual[0], expected[0]);
        }
        float max_gemm_error = 0.0f;
        for (size_t index = 0; index < actual.size(); ++index) {
            const float error = std::fabs(actual[index] - expected[index]);
            max_gemm_error = std::max(max_gemm_error, error);
            assert(std::isfinite(actual[index]));
            if (error > 1e-4f + 1e-5f * std::fabs(expected[index])) {
                std::fprintf(stderr,
                             "QBS %s GEMM mismatch input=%zu output=%zu "
                             "actual=%.9g expected=%.9g error=%.9g\n",
                             profile.name, index / nrows, index % nrows,
                             actual[index], expected[index], error);
            }
            assert(error <= 1e-4f + 1e-5f * std::fabs(expected[index]));
        }

        std::vector<float> actual_m2(2 * nrows);
        ggml_riscv_qbs_gemm(profile.type, n, actual_m2.data(), nrows,
                            packed.data(), activation.data(), 2, nrows);
        for (size_t index = 0; index < actual_m2.size(); ++index) {
            const float error = std::fabs(actual_m2[index] - expected[index]);
            assert(std::isfinite(actual_m2[index]));
            assert(error <= 1e-4f + 1e-5f * std::fabs(expected[index]));
        }

        std::fill(actual.begin(), actual.end(), 0.0f);
        ggml_riscv_qbs_gemv(profile.type, n, actual.data(), nrows,
                            packed.data(), activation.data(), 1, nrows);
        float max_gemv_error = 0.0f;
        for (size_t row = 0; row < (size_t) nrows; ++row) {
            const float error = std::fabs(actual[row] - expected[row]);
            max_gemv_error = std::max(max_gemv_error, error);
            assert(std::isfinite(actual[row]));
            if (error > 1e-4f + 1e-5f * std::fabs(expected[row])) {
                std::fprintf(stderr,
                             "QBS %s GEMV mismatch output=%zu actual=%.9g "
                             "expected=%.9g error=%.9g\n",
                             profile.name, row, actual[row], expected[row],
                             error);
            }
            assert(error <= 1e-4f + 1e-5f * std::fabs(expected[row]));
        }
        std::printf(
            "QBS %s %s M1/M2/M3/M4,N35: PASS "
            "(GEMV %.8g, GEMM %.8g max abs error)\n",
            profile.name, native ? "native" : "emulated",
            max_gemv_error, max_gemm_error);
    }
}

static void test_qbs_repack_buffer_tail() {
    constexpr int64_t n = 2 * QK_K;
    constexpr int64_t nrows = 35;
    constexpr size_t blocks_per_row = n / QK_K;
    const size_t source_size = nrows * blocks_per_row * sizeof(block_q4_K);
    const size_t padded_size = ((nrows + 3) & ~size_t(3)) *
                               blocks_per_row * sizeof(block_q4_K);

    std::vector<float> source_f32(nrows * n);
    std::vector<block_q4_K> source(nrows * blocks_per_row);
    for (size_t index = 0; index < source_f32.size(); ++index) {
        source_f32[index] = 0.375f * std::sin(float(index * 29) * 0.0078125f);
    }
    for (size_t row = 0; row < (size_t) nrows; ++row) {
        quantize_row_q4_K_ref(source_f32.data() + row * n,
                              source.data() + row * blocks_per_row, n);
    }
    std::vector<uint8_t> expected(padded_size);
    assert(ggml_riscv_qbs_repack_r4(
               GGML_TYPE_Q4_K, expected.data(), expected.size(), source.data(),
               source_size, nrows, blocks_per_row) == 0);

    ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    assert(context != nullptr);
    ggml_tensor * tensor =
        ggml_new_tensor_2d(context, GGML_TYPE_Q4_K, n, nrows);
    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_cpu_repack_buffer_type();
    assert(ggml_backend_buft_get_alloc_size(buffer_type, tensor) == padded_size);
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(context, buffer_type);
    assert(buffer != nullptr && tensor->data != nullptr);
    ggml_backend_tensor_set(tensor, source.data(), 0, source_size);
    assert(std::memcmp(tensor->data, expected.data(), padded_size) == 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    std::printf("QBS CPU_REPACK N35 padded allocation: PASS\n");
}

static void test_qbs_moe_repack() {
    constexpr int64_t n = 2 * QK_K;
    constexpr int64_t rows_per_expert = 8;
    constexpr int64_t experts = 3;
    constexpr size_t blocks_per_row = n / QK_K;
    constexpr size_t total_rows = rows_per_expert * experts;
    const size_t source_size =
        total_rows * blocks_per_row * sizeof(block_q2_K);

    std::vector<uint8_t> source(source_size);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint8_t>((index * 37 + 19) & 0xff);
    }
    std::vector<uint8_t> expected(source_size);
    assert(ggml_riscv_qbs_repack_r4(
               GGML_TYPE_Q2_K, expected.data(), expected.size(), source.data(),
               source.size(), total_rows, blocks_per_row) == 0);

    ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    assert(context != nullptr);
    ggml_tensor * tensor = ggml_new_tensor_3d(
        context, GGML_TYPE_Q2_K, n, rows_per_expert, experts);
    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_cpu_repack_buffer_type();
    assert(ggml_backend_buft_get_alloc_size(buffer_type, tensor) == source_size);
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(context, buffer_type);
    assert(buffer != nullptr && tensor->data != nullptr);
    ggml_backend_tensor_set(tensor, source.data(), 0, source.size());
    assert(std::memcmp(tensor->data, expected.data(), expected.size()) == 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    std::printf("QBS MUL_MAT_ID expert-local R4 repack: PASS\n");
}

static void test_qbs_moe_repack_fallback() {
    constexpr int64_t n = 2 * QK_K;
    constexpr int64_t rows_per_expert = 6;
    constexpr int64_t experts = 3;

    ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    assert(context != nullptr);
    ggml_tensor * tensor = ggml_new_tensor_3d(
        context, GGML_TYPE_Q2_K, n, rows_per_expert, experts);
    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_cpu_repack_buffer_type();

    // R4 groups must remain within one expert. Six rows per expert cannot be
    // globally padded without changing nb02, so this shape must stay in the
    // ordinary, unpacked GGML layout.
    assert(ggml_backend_buft_get_alloc_size(buffer_type, tensor) ==
           ggml_nbytes(tensor));

    ggml_free(context);
    std::printf("QBS MUL_MAT_ID non-R4 expert shape fallback: PASS\n");
}

static void test_qbs_moe_graph(int threads) {
    constexpr int64_t n = 7 * QK_K;
    constexpr int64_t rows_per_expert = 8;
    constexpr int64_t experts = 3;
    constexpr int64_t experts_per_token = 2;
    constexpr int64_t tokens = 3;
    constexpr size_t blocks_per_row = n / QK_K;

    std::vector<float> weights_f32(experts * rows_per_expert * n);
    std::vector<block_q2_K> weights_q2(
        experts * rows_per_expert * blocks_per_row);
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < rows_per_expert; ++row) {
            float * source = weights_f32.data() +
                (expert * rows_per_expert + row) * n;
            for (int64_t col = 0; col < n; ++col) {
                source[col] =
                    0.5f * std::sin(float(expert * 41 + row * 17 + col * 5) *
                                    0.015625f) -
                    0.25f * std::cos(float(expert * 13 + row * 29 + col * 3) *
                                     0.01171875f);
            }
            quantize_row_q2_K_ref(
                source,
                weights_q2.data() +
                    (expert * rows_per_expert + row) * blocks_per_row,
                n);
        }
    }

    std::vector<float> activations(tokens * experts_per_token * n);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t slot = 0; slot < experts_per_token; ++slot) {
            float * row = activations.data() +
                (token * experts_per_token + slot) * n;
            for (int64_t col = 0; col < n; ++col) {
                row[col] =
                    0.375f * std::cos(float(token * 37 + slot * 19 + col * 7) *
                                      0.013671875f);
            }
        }
    }
    const std::vector<int32_t> ids = {
        2, 0,
        1, 2,
        0, 1,
    };

    ggml_init_params weights_params{};
    weights_params.mem_size = 64 * 1024;
    weights_params.no_alloc = true;
    ggml_context * weights_context = ggml_init(weights_params);
    assert(weights_context != nullptr);
    ggml_tensor * weights = ggml_new_tensor_3d(
        weights_context, GGML_TYPE_Q2_K, n, rows_per_expert, experts);
    ggml_set_name(weights, "qbs_moe_weights");
    ggml_backend_buffer_t weights_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(
            weights_context, ggml_backend_cpu_repack_buffer_type());
    assert(weights_buffer != nullptr);
    ggml_backend_tensor_set(weights, weights_q2.data(), 0,
                            weights_q2.size() * sizeof(block_q2_K));

    ggml_init_params graph_params{};
    graph_params.mem_size = 1024 * 1024;
    graph_params.no_alloc = true;
    ggml_context * graph_context = ggml_init(graph_params);
    assert(graph_context != nullptr);
    ggml_tensor * activation = ggml_new_tensor_3d(
        graph_context, GGML_TYPE_F32, n, experts_per_token, tokens);
    ggml_tensor * expert_ids = ggml_new_tensor_2d(
        graph_context, GGML_TYPE_I32, experts_per_token, tokens);
    ggml_tensor * output =
        ggml_mul_mat_id(graph_context, weights, activation, expert_ids);
    ggml_set_name(output, "qbs_moe_output");

    ggml_backend_t backend =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(backend != nullptr);
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_backend_buffer_t graph_buffer =
        ggml_backend_alloc_ctx_tensors(graph_context, backend);
    assert(graph_buffer != nullptr);
    ggml_backend_tensor_set(activation, activations.data(), 0,
                            activations.size() * sizeof(float));
    ggml_backend_tensor_set(expert_ids, ids.data(), 0,
                            ids.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(graph_context);
    ggml_build_forward_expand(graph, output);
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0,
                            actual.size() * sizeof(float));
    std::vector<float> weight_dequantized(n);
    std::vector<float> activation_dequantized(n);
    std::vector<block_q8_K> activation_q8(blocks_per_row);
    float max_error = 0.0f;
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t slot = 0; slot < experts_per_token; ++slot) {
            quantize_row_q8_K_ref(
                activations.data() + (token * experts_per_token + slot) * n,
                activation_q8.data(), n);
            dequantize_row_q8_K(activation_q8.data(),
                                activation_dequantized.data(), n);
            const int64_t expert = ids[token * experts_per_token + slot];
            for (int64_t row = 0; row < rows_per_expert; ++row) {
                dequantize_row_q2_K(
                    weights_q2.data() +
                        (expert * rows_per_expert + row) * blocks_per_row,
                    weight_dequantized.data(), n);
                const float expected = qbs_scalar_dot(
                    weight_dequantized.data(), activation_dequantized.data(), n);
                const size_t output_index =
                    (token * experts_per_token + slot) * rows_per_expert + row;
                const float error = std::fabs(actual[output_index] - expected);
                max_error = std::max(max_error, error);
                assert(std::isfinite(actual[output_index]));
                assert(error <= 2e-4f + 2e-5f * std::fabs(expected));
            }
        }
    }

    ggml_backend_buffer_free(graph_buffer);
    ggml_backend_free(backend);
    ggml_free(graph_context);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(weights_context);
    std::printf("QBS MUL_MAT_ID Q2_K 3-expert graph: PASS "
                "(threads %d, max abs error %.8g)\n", threads, max_error);
}

static void test_qbs_matrix_graphs(bool natural_k = false) {
    const char * requested = std::getenv("GGML_RISCV_QBS_SEGMENT_K_BLOCKS");
    const unsigned segment = requested ? static_cast<unsigned>(std::strtoul(requested, nullptr, 10)) : 0;
    assert(ggml_riscv_qbs_weight_segment_blocks() == segment);
    const ggml_type types[] = {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0,
        GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL};
    constexpr int64_t rows = 37;
    const size_t k_blocks = natural_k ? 280 : 7;
    unsigned cases = 0;
    for (ggml_type type : types) {
        if (natural_k && type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q8_0) {
            continue;
        }
        const bool q8_0 = type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 ||
                          type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL;
        const ggml_type activation_type = q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
        const auto * wt = ggml_get_type_traits(type);
        const int64_t k = k_blocks * ggml_blck_size(type);
        const size_t weight_stride = ggml_row_size(type, k);
        const size_t activation_stride = ggml_row_size(activation_type, k);
        std::vector<uint8_t> packed_weights(rows * weight_stride);
        std::vector<float> weight_values(rows * k), source(k);
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t i = 0; i < k; ++i) {
                source[i] = 0.625f * std::sin(float(row * 23 + i * 7) * 0.01953125f);
            }
            wt->from_float_ref(source.data(), packed_weights.data() + row * weight_stride, k);
            wt->to_float(packed_weights.data() + row * weight_stride, weight_values.data() + row * k, k);
        }
        for (int inputs : {1, 2, 3, 4, 5, 7, 8}) {
            if (natural_k && inputs != 1 && inputs != 3 && inputs != 4) {
                continue;
            }
            std::vector<float> activations(inputs * k), activation_values(inputs * k);
            std::vector<uint8_t> activation_quantized(activation_stride);
            size_t quant_reference_differences = 0;
            for (int input = 0; input < inputs; ++input) {
                for (int64_t i = 0; i < k; ++i) {
                    activations[input * k + i] = 0.5f * std::cos(float(input * 17 + i * 11) * 0.015625f);
                }
                if (q8_0) {
                    auto * quantized = reinterpret_cast<block_q8_0 *>(activation_quantized.data());
                    quantize_row_q8_0_ref(activations.data() + input * k, quantized, k);
                    dequantize_row_q8_0(quantized, activation_values.data() + input * k, k);
                } else {
                    auto * quantized = reinterpret_cast<block_q8_K *>(activation_quantized.data());
                    quantize_row_q8_K_ref(activations.data() + input * k, quantized, k);
                    dequantize_row_q8_K(quantized, activation_values.data() + input * k, k);
                }
                if (natural_k && input >= inputs - inputs % 4) {
                    // Match GGML's tail-row quantizer; _ref can contract rounding into FMA.
                    const auto reference = activation_quantized;
                    ggml_get_type_traits_cpu(activation_type)->from_float(
                        activations.data() + input * k, activation_quantized.data(), k);
                    for (size_t byte = 0; byte < reference.size(); ++byte) {
                        quant_reference_differences += reference[byte] != activation_quantized[byte];
                    }
                    if (q8_0) {
                        dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(activation_quantized.data()),
                            activation_values.data() + input * k, k);
                    } else {
                        dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(activation_quantized.data()),
                            activation_values.data() + input * k, k);
                    }
                }
            }
            for (int threads : {1, 3}) {
                ggml_init_params params{};
                params.mem_size = 1024 * 1024;
                params.no_alloc = true;
                ggml_context * weight_context = ggml_init(params);
                assert(weight_context != nullptr);
                ggml_tensor * weights = ggml_new_tensor_2d(weight_context, type, k, rows);
                ggml_backend_buffer_t weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
                    weight_context, ggml_backend_cpu_repack_buffer_type());
                assert(weight_buffer != nullptr && weights->extra != nullptr);
                ggml_backend_tensor_set(weights, packed_weights.data(), 0, packed_weights.size());
                const size_t block_bytes = ggml_type_size(type);
                const size_t segment_blocks = segment ? segment : k_blocks;
                for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {
                    for (size_t block = 0; block < k_blocks; ++block) {
                        const size_t start = block / segment_blocks * segment_blocks;
                        const size_t local_k = std::min(segment_blocks, k_blocks - start);
                        const size_t index = start * 40 + (row / 4) * local_k * 4 + (block - start) * 4 + row % 4;
                        assert(std::memcmp(static_cast<const uint8_t *>(weights->data) + index * block_bytes,
                                           packed_weights.data() + (row * k_blocks + block) * block_bytes, block_bytes) == 0);
                    }
                }

                ggml_context * context = ggml_init(params);
                assert(context != nullptr);
                ggml_tensor * activation = ggml_new_tensor_2d(context, GGML_TYPE_F32, k, inputs);
                ggml_tensor * output = ggml_mul_mat(context, weights, activation);
                ggml_backend_t backend = ggml_backend_cpu_init();
                assert(backend != nullptr);
                ggml_backend_cpu_set_n_threads(backend, threads);
                ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
                assert(buffer != nullptr);
                ggml_backend_tensor_set(activation, activations.data(), 0, activations.size() * sizeof(float));
                ggml_cgraph * graph = ggml_new_graph(context);
                ggml_build_forward_expand(graph, output);
                std::printf("QBS_MATRIX_BEGIN type=%s m=%d n=%lld k=%lld threads=%d segment=%u\n",
                    ggml_type_name(type), inputs, (long long) rows, (long long) k, threads,
                    ggml_riscv_qbs_weight_segment_blocks());
                std::fflush(stdout);
                assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                std::vector<float> actual(rows * inputs);
                ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
                float max_error = 0;
                double max_scalar_drift = 0;
                for (int input = 0; input < inputs; ++input) {
                    for (int64_t row = 0; row < rows; ++row) {
                        const float scalar = qbs_scalar_dot(weight_values.data() + row * k,
                                                            activation_values.data() + input * k, k);
                        double precise = 0;
                        if (natural_k) {
                            // A long, cancelling FP32 sum is not an independent accuracy oracle.
                            for (int64_t i = 0; i < k; ++i) {
                                precise += double(weight_values[row * k + i]) * double(activation_values[input * k + i]);
                            }
                            max_scalar_drift = std::max(max_scalar_drift, std::fabs(double(scalar) - precise));
                        }
                        const float expected = natural_k ? static_cast<float>(precise) : scalar;
                        const float error = std::fabs(actual[input * rows + row] - expected);
                        max_error = std::max(max_error, error);
                        if (!std::isfinite(actual[input * rows + row]) || !std::isfinite(expected) ||
                            !std::isfinite(error) || error > 2e-4f + 2e-5f * std::fabs(expected)) {
                            if (natural_k) {
                                std::vector<uint8_t> reference(activation_stride), production(activation_stride);
                                std::vector<float> produced_values(k), reference_values(k);
                                const float * values = activations.data() + input * k;
                                ggml_get_type_traits_cpu(activation_type)->from_float(values, production.data(), k);
                                if (q8_0) {
                                    quantize_row_q8_0_ref(values, reinterpret_cast<block_q8_0 *>(reference.data()), k);
                                    dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(reference.data()), reference_values.data(), k);
                                    dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(production.data()), produced_values.data(), k);
                                } else {
                                    quantize_row_q8_K_ref(values, reinterpret_cast<block_q8_K *>(reference.data()), k);
                                    dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(reference.data()), reference_values.data(), k);
                                    dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(production.data()), produced_values.data(), k);
                                }
                                size_t different_bytes = 0;
                                for (size_t byte = 0; byte < reference.size(); ++byte) {
                                    different_bytes += reference[byte] != production[byte];
                                }
                                float reference_blocked = 0, production_blocked = 0;
                                for (size_t start = 0; start < k_blocks; start += 256) {
                                    float ref_partial = 0, prod_partial = 0;
                                    for (size_t block = start; block < std::min(start + 256, k_blocks); ++block) {
                                        const size_t offset = block * ggml_blck_size(type);
                                        ref_partial += qbs_scalar_dot(weight_values.data() + row * k + offset,
                                            reference_values.data() + offset, ggml_blck_size(type));
                                        prod_partial += qbs_scalar_dot(weight_values.data() + row * k + offset,
                                            produced_values.data() + offset, ggml_blck_size(type));
                                    }
                                    reference_blocked += ref_partial;
                                    production_blocked += prod_partial;
                                }
                                std::fprintf(stderr, "QBS_NATURAL_DIAG quant_byte_diffs=%zu scalar=%.9g precise=%.17g ref_blocked=%.9g prod_blocked=%.9g actual=%.9g\n",
                                    different_bytes, double(scalar), precise, double(reference_blocked), double(production_blocked), double(actual[input * rows + row]));
                            }
                            std::fprintf(stderr, "QBS graph mismatch type=%s M=%d N=%lld threads=%d row=%lld input=%d actual=%g expected=%g\n",
                                ggml_type_name(type), inputs, (long long) rows, threads, (long long) row,
                                input, actual[input * rows + row], expected);
                            std::abort();
                        }
                    }
                }
                if (natural_k) {
                    uint64_t hash = UINT64_C(14695981039346656037);
                    for (float value : actual) {
                        uint32_t bits;
                        std::memcpy(&bits, &value, sizeof(bits));
                        for (unsigned byte = 0; byte < 4; ++byte) {
                            hash = (hash ^ ((bits >> (8 * byte)) & 255u)) * UINT64_C(1099511628211);
                        }
                    }
                    std::printf("QBS_NATURAL_ORACLE quant_reference_differences=%zu max_fp32_scalar_drift=%.9g max_qbs_error=%.9g\n",
                        quant_reference_differences, max_scalar_drift, double(max_error));
                    std::printf("QBS_NATURAL_DIGEST value=%016llx\n", (unsigned long long) hash);
                }
                std::printf("QBS MUL_MAT graph: PASS type=%s M=%d N=%lld K=%lld threads=%d segment=%u error=%g\n",
                    ggml_type_name(type), inputs, (long long) rows, (long long) k, threads,
                    ggml_riscv_qbs_weight_segment_blocks(), max_error);
                ++cases;
                ggml_backend_buffer_free(buffer);
                ggml_backend_free(backend);
                ggml_free(context);
                ggml_backend_buffer_free(weight_buffer);
                ggml_free(weight_context);
            }
        }
    }
    assert(cases == (natural_k ? 12 : 126));
    std::printf("QBS matrix graph suite: PASS cases=%u\n", cases);
}

static void test_qbs_context_graphs(bool moe_barrier = false, int mutation = 0) {
    struct scenario { int blocks; int rows; bool shared; int threads; };
    const scenario scenarios[] = {{2, 1, true, 1}, {7, 1, true, 1}, {7, 1, false, 1},
                                  {7, 1, true, 3}, {7, 3, true, 1}, {7, 4, true, 1}};
    constexpr int n = 37;
    constexpr int expert_rows = 36;
    unsigned cases = 0;
    for (const auto & test : scenarios) {
        if ((moe_barrier || mutation != 0) && test.blocks != 2) {
            continue;
        }
        const int k = test.blocks * QK_K;
        const size_t stride = ggml_row_size(GGML_TYPE_Q4_K, k);
        std::vector<uint8_t> packed[2];
        std::vector<float> weights[2];
        std::vector<float> source(k);
        for (int projection = 0; projection < 2; ++projection) {
            packed[projection].resize(n * stride);
            weights[projection].resize(n * k);
            for (int row = 0; row < n; ++row) {
                for (int i = 0; i < k; ++i) {
                    source[i] = 0.375f * std::sin(float(i * 7 + row * 19 + projection * 31) * 0.0234375f);
                }
                auto * block = reinterpret_cast<block_q4_K *>(packed[projection].data() + row * stride);
                quantize_row_q4_K_ref(source.data(), block, k);
                dequantize_row_q4_K(block, weights[projection].data() + row * k, k);
            }
        }
        ggml_init_params params{};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ggml_context * wc = ggml_init(params);
        assert(wc != nullptr);
        ggml_tensor * w[2] = {ggml_new_tensor_2d(wc, GGML_TYPE_Q4_K, k, n),
                             ggml_new_tensor_2d(wc, GGML_TYPE_Q4_K, k, n)};
        ggml_tensor * experts = moe_barrier ? ggml_new_tensor_3d(wc, GGML_TYPE_Q4_K, k, expert_rows, 2) : nullptr;
        ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, ggml_backend_cpu_repack_buffer_type());
        assert(wb != nullptr && w[0]->extra != nullptr && w[1]->extra != nullptr);
        for (int projection = 0; projection < 2; ++projection) {
            ggml_backend_tensor_set(w[projection], packed[projection].data(), 0, packed[projection].size());
        }
        if (moe_barrier) {
            assert(experts->extra != nullptr);
            std::vector<uint8_t> expert_data(2 * expert_rows * stride);
            for (int expert = 0; expert < 2; ++expert) {
                std::memcpy(expert_data.data() + expert * expert_rows * stride, packed[expert].data(), expert_rows * stride);
            }
            ggml_backend_tensor_set(experts, expert_data.data(), 0, expert_data.size());
        }
        ggml_context * ctx = ggml_init(params);
        assert(ctx != nullptr);
        ggml_tensor * x[2];
        x[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, test.rows);
        x[1] = test.shared ? x[0] : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, test.rows);
        ggml_tensor * y[2] = {ggml_mul_mat(ctx, w[0], x[0]), ggml_mul_mat(ctx, w[1], x[1])};
        ggml_tensor * sum = ggml_add(ctx, y[0], y[1]);
        ggml_tensor * ids = moe_barrier ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1) : nullptr;
        ggml_tensor * moe = moe_barrier ? ggml_mul_mat_id(ctx, experts, x[0], ids) : nullptr;
        ggml_tensor * mutation_target = nullptr;
        ggml_tensor * mutation_node = nullptr;
        if (mutation != 0) {
            mutation_target = mutation == 3 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k) : x[0];
            if (mutation == 2 || mutation == 4) {
                mutation_target = ggml_view_1d(ctx, x[0], k / 2, (k / 4) * sizeof(float));
            }
            mutation_node = mutation == 4 ? mutation_target : ggml_scale_inplace(ctx, mutation_target, 0.5f);
        }
        ggml_backend_t backend = ggml_backend_cpu_init();
        assert(backend != nullptr);
        ggml_backend_cpu_set_n_threads(backend, test.threads);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert(buffer != nullptr);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        if (moe_barrier) {
            const int32_t selected = 1;
            ggml_backend_tensor_set(ids, &selected, 0, sizeof(selected));
            ggml_build_forward_expand(graph, y[0]);
            ggml_build_forward_expand(graph, moe);
        }
        if (mutation != 0) {
            ggml_build_forward_expand(graph, y[0]);
            ggml_build_forward_expand(graph, mutation_node);
        }
        ggml_build_forward_expand(graph, sum);
        for (int epoch = 0; epoch < 2; ++epoch) {
            std::vector<float> input(test.rows * k), dequantized(test.rows * k);
            std::vector<block_q8_K> quantized(test.blocks);
            for (int row = 0; row < test.rows; ++row) {
                for (int i = 0; i < k; ++i) {
                    input[row * k + i] = 0.5f * std::cos(float(i * 11 + row * 17 + epoch * 53) * 0.01953125f);
                }
                if (row < test.rows - test.rows % 4) {
                    quantize_row_q8_K_ref(input.data() + row * k, quantized.data(), k);
                } else {
                    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(input.data() + row * k, quantized.data(), k);
                }
                dequantize_row_q8_K(quantized.data(), dequantized.data() + row * k, k);
            }
            ggml_backend_tensor_set(x[0], input.data(), 0, input.size() * sizeof(float));
            std::vector<float> input_after = input;
            std::vector<float> dequantized_after = dequantized;
            if (mutation == 1 || mutation == 2) {
                const int begin = mutation == 1 ? 0 : k / 4;
                const int end = mutation == 1 ? k : 3 * k / 4;
                for (int i = begin; i < end; ++i) {
                    input_after[i] *= 0.5f;
                }
                ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(input_after.data(), quantized.data(), k);
                dequantize_row_q8_K(quantized.data(), dequantized_after.data(), k);
            } else if (mutation == 3) {
                ggml_backend_tensor_set(mutation_target, input.data(), 0, input.size() * sizeof(float));
            }
            if (!test.shared) {
                // Equal bytes at a different tensor identity must not enable reuse.
                ggml_backend_tensor_set(x[1], input.data(), 0, input.size() * sizeof(float));
            }
            std::printf("QBS_CONTEXT_BEGIN blocks=%d m=%d n=%d shared=%d threads=%d epoch=%d segment=%u\n",
                test.blocks, test.rows, n, test.shared, test.threads, epoch, ggml_riscv_qbs_weight_segment_blocks());
            if (moe_barrier) {
                std::printf("QBS_CONTEXT_MOE_BARRIER expert=1 rows=%d\n", expert_rows);
            }
            if (mutation != 0) {
                std::printf("QBS_CONTEXT_MUTATION kind=%d writes_source=%d\n", mutation, mutation <= 2);
            }
            std::fflush(stdout);
            assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            if (mutation != 0) {
                std::vector<float> actual_input(k);
                ggml_backend_tensor_get(x[0], actual_input.data(), 0, k * sizeof(float));
                assert(actual_input == input_after);
                if (mutation == 3) {
                    ggml_backend_tensor_get(mutation_target, actual_input.data(), 0, k * sizeof(float));
                    for (int i = 0; i < k; ++i) {
                        assert(actual_input[i] == input[i] * 0.5f);
                    }
                }
                std::printf("QBS_CONTEXT_MUTATION_INPUT_PASS words=%d\n", k);
            }
            std::vector<float> actual[3];
            for (int output = 0; output < 3; ++output) {
                actual[output].resize(test.rows * n);
                ggml_backend_tensor_get(output < 2 ? y[output] : sum, actual[output].data(), 0, actual[output].size() * sizeof(float));
            }
            double max_error = 0;
            for (int row = 0; row < test.rows; ++row) {
                for (int column = 0; column < n; ++column) {
                    double expected[3] = {};
                    for (int projection = 0; projection < 2; ++projection) {
                        for (int i = 0; i < k; ++i) {
                            const auto & values = projection == 1 ? dequantized_after : dequantized;
                            expected[projection] += double(weights[projection][column * k + i]) * double(values[row * k + i]);
                        }
                    }
                    expected[2] = expected[0] + expected[1];
                    for (int output = 0; output < 3; ++output) {
                        const double value = actual[output][row * n + column];
                        const double error = std::fabs(value - expected[output]);
                        max_error = std::max(max_error, error);
                        if (!std::isfinite(value) || !std::isfinite(expected[output]) || error > 2e-4 + 2e-5 * std::fabs(expected[output])) {
                            std::fprintf(stderr, "QBS context mismatch output=%d row=%d column=%d actual=%.9g expected=%.17g\n",
                                output, row, column, value, expected[output]);
                            std::abort();
                        }
                    }
                }
            }
            std::printf("QBS_CONTEXT_PASS outputs=%d max_error=%.9g\n", 3 * n * test.rows, max_error);
            if (moe_barrier) {
                float values[expert_rows];
                ggml_backend_tensor_get(moe, values, 0, sizeof(values));
                for (int row = 0; row < expert_rows; ++row) {
                    double expected = 0;
                    for (int i = 0; i < k; ++i) {
                        expected += double(weights[1][row * k + i]) * double(dequantized[i]);
                    }
                    assert(std::isfinite(values[row]) && std::fabs(double(values[row]) - expected) <= 2e-4 + 2e-5 * std::fabs(expected));
                }
                std::printf("QBS_CONTEXT_MOE_PASS outputs=%d\n", expert_rows);
            }
            ++cases;
        }
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        ggml_backend_buffer_free(wb);
        ggml_free(wc);
    }
    assert(cases == (moe_barrier || mutation != 0 ? 2 : 12));
    std::printf("QBS context graph suite: PASS cases=%u\n", cases);
}

static void test_qbs_segmented_k() {
    constexpr int64_t k_blocks = 280;
    constexpr int64_t n = k_blocks * QK4_0;
    constexpr int64_t nrows = 5;
    constexpr int64_t ninputs = 7;

    assert(ggml_riscv_qbs_supports_tensor(GGML_TYPE_Q4_0, n, nrows));

    std::vector<float> weights_f32(nrows * n);
    std::vector<float> activations_f32(ninputs * n);
    for (size_t index = 0; index < weights_f32.size(); ++index) {
        weights_f32[index] =
            0.25f * std::sin(float(index % 4093) * 0.00390625f);
    }
    for (size_t index = 0; index < activations_f32.size(); ++index) {
        activations_f32[index] =
            0.375f * std::cos(float(index % 4051) * 0.003125f);
    }

    std::vector<block_q4_0> weights(nrows * k_blocks);
    std::vector<block_q8_0> activations(ninputs * k_blocks);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_q4_0_ref(weights_f32.data() + row * n,
                              weights.data() + row * k_blocks, n);
    }
    for (int64_t input = 0; input < ninputs; ++input) {
        quantize_row_q8_0_ref(activations_f32.data() + input * n,
                              activations.data() + input * k_blocks, n);
    }

    std::vector<block_q8_0> activations_packed(ninputs * k_blocks);
    quantize_q8_0_m4(activations_f32.data(), activations_packed.data(), n);
    for (int64_t input = 4; input < ninputs; ++input) {
        std::memcpy(activations_packed.data() + input * k_blocks,
                    activations.data() + input * k_blocks,
                    k_blocks * sizeof(block_q8_0));
    }

    const size_t padded_rows = (nrows + 3) & ~size_t(3);
    std::vector<block_q4_0> packed(padded_rows * k_blocks);
    assert(ggml_riscv_qbs_repack_r4(
               GGML_TYPE_Q4_0, packed.data(),
               packed.size() * sizeof(block_q4_0), weights.data(),
               weights.size() * sizeof(block_q4_0), nrows, k_blocks) == 0);

    std::vector<float> expected(ninputs * nrows, 0.0f);
    std::vector<float> weight_block(QK4_0);
    std::vector<float> activation_block(QK8_0);
    for (int64_t input = 0; input < ninputs; ++input) {
        for (int64_t row = 0; row < nrows; ++row) {
            float sum = 0.0f;
            for (int64_t block = 0; block < k_blocks; ++block) {
                dequantize_row_q4_0(
                    weights.data() + row * k_blocks + block,
                    weight_block.data(), QK4_0);
                dequantize_row_q8_0(
                    activations.data() + input * k_blocks + block,
                    activation_block.data(), QK8_0);
                sum += qbs_scalar_dot(weight_block.data(),
                                      activation_block.data(), QK4_0);
            }
            expected[input * nrows + row] = sum;
        }
    }

    std::vector<float> actual(ninputs * nrows);
    ggml_riscv_qbs_gemm(GGML_TYPE_Q4_0, n, actual.data(), nrows,
                        packed.data(), activations_packed.data(), ninputs,
                        nrows);
    float max_error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);
        max_error = std::max(max_error, error);
        assert(std::isfinite(actual[index]));
        assert(error <= 5e-3f + 2e-4f * std::fabs(expected[index]));
    }

    std::vector<float> actual_m2(2 * nrows);
    ggml_riscv_qbs_gemm(GGML_TYPE_Q4_0, n, actual_m2.data(), nrows,
                        packed.data(), activations.data(), 2, nrows);
    for (size_t index = 0; index < actual_m2.size(); ++index) {
        const float error = std::fabs(actual_m2[index] - expected[index]);
        max_error = std::max(max_error, error);
        assert(std::isfinite(actual_m2[index]));
        assert(error <= 5e-3f + 2e-4f * std::fabs(expected[index]));
    }

    std::fill(actual.begin(), actual.end(), 0.0f);
    ggml_riscv_qbs_gemv(GGML_TYPE_Q4_0, n, actual.data(), nrows,
                        packed.data(), activations.data(), 1, nrows);
    for (int64_t row = 0; row < nrows; ++row) {
        const float error = std::fabs(actual[row] - expected[row]);
        max_error = std::max(max_error, error);
        assert(std::isfinite(actual[row]));
        assert(error <= 5e-3f + 2e-4f * std::fabs(expected[row]));
    }
    std::printf("QBS Q4_0 segmented K=%lld M1/M2/M3/M4: PASS "
                "(max abs error %.8g)\n",
                (long long) n, max_error);
}
#endif

int main() {
#if defined(GGML_TEST_RISCV_QBS)
    if (const char * suite = std::getenv("GGML_QBS_GRAPH_SUITE")) {
        if (std::strcmp(suite, "alias") == 0) {
            for (int mutation = 1; mutation <= 4; ++mutation) {
                test_qbs_context_graphs(false, mutation);
            }
        } else if (std::strcmp(suite, "context") == 0 || std::strcmp(suite, "barrier") == 0) {
            test_qbs_context_graphs(std::strcmp(suite, "barrier") == 0);
        } else {
            assert(std::strcmp(suite, "natural") == 0);
            test_qbs_matrix_graphs(true);
        }
        return 0;
    }
#endif
    test_qbs_r4_layout();
    test_qbs_activation_layouts();
    constexpr int64_t nrows = 64;
    constexpr int64_t n_per_row = 3 * QK_K;
    constexpr size_t blocks_per_row = n_per_row / QK_K;

    std::vector<block_q4_K> source(nrows * blocks_per_row);
    for (size_t row = 0; row < (size_t) nrows; ++row) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            source[row * blocks_per_row + block] = make_source_block(row, block);
        }
    }

    std::vector<block_q4_Kx32_ara> packed((nrows / 32) * blocks_per_row);
    const int status = ggml_repack_q4_K_32x1_ara(
            packed.data(), packed.size() * sizeof(packed[0]),
            source.data(), source.size() * sizeof(source[0]), nrows, n_per_row);
    assert(status == 0);
    assert(packed.size() * sizeof(packed[0]) == source.size() * sizeof(source[0]));

    for (size_t row = 0; row < (size_t) nrows; ++row) {
        const size_t tile = row / 32;
        const size_t row_in_tile = row % 32;
        for (size_t block = 0; block < blocks_per_row; ++block) {
            const block_q4_K recovered = recover_source_block(
                    packed[tile * blocks_per_row + block], row_in_tile);
            assert(memcmp(&recovered, &source[row * blocks_per_row + block], sizeof(recovered)) == 0);
        }
    }

    assert(ggml_repack_q4_K_32x1_ara(
            packed.data(), packed.size() * sizeof(packed[0]),
            source.data(), source.size() * sizeof(source[0]), 33, n_per_row) == -1);
    assert(ggml_repack_q4_K_32x1_ara(
            packed.data(), packed.size() * sizeof(packed[0]),
            source.data(), source.size() * sizeof(source[0]), nrows, n_per_row + 1) == -1);
    assert(ggml_repack_q4_K_32x1_ara(
            packed.data(), packed.size() * sizeof(packed[0]) - 1,
            source.data(), source.size() * sizeof(source[0]), nrows, n_per_row) == -1);

    std::printf("Ara Q4_Kx32 repack: PASS (%zu source bytes, %zu packed bytes)\n",
                source.size() * sizeof(source[0]), packed.size() * sizeof(packed[0]));
    test_numeric_gemv();
    test_numeric_gemm();
#if defined(GGML_TEST_RISCV_QBS)
    test_qbs_emulated_profiles();
    test_qbs_repack_buffer_tail();
    test_qbs_moe_repack();
    test_qbs_moe_repack_fallback();
    test_qbs_moe_graph(1);
    test_qbs_moe_graph(3);
    test_qbs_matrix_graphs();
    test_qbs_segmented_k();
#endif
    return 0;
}
