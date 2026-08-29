#include "repack.h"
#include "arch/riscv/qbs-layout.h"
#include "ggml-backend.h"
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
                   packed.data(), packed.size(), source.data(), source.size(),
                   nrows, blocks_per_row, profile.weight_block_bytes) == 0);

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
               expected.data(), expected.size(), source.data(), source_size,
               nrows, blocks_per_row, sizeof(block_q4_K)) == 0);

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
               expected.data(), expected.size(), source.data(), source.size(),
               total_rows, blocks_per_row, sizeof(block_q2_K)) == 0);

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

static void test_qbs_moe_graph() {
    constexpr int64_t n = 2 * QK_K;
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
                "(max abs error %.8g)\n", max_error);
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
               packed.data(), packed.size() * sizeof(block_q4_0),
               weights.data(), weights.size() * sizeof(block_q4_0),
               nrows, k_blocks, sizeof(block_q4_0)) == 0);

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
    test_qbs_moe_graph();
    test_qbs_segmented_k();
#endif
    return 0;
}
