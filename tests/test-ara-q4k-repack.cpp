#include "repack.h"
#include "ggml-quants.h"
#include "quants.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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

int main() {
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
    return 0;
}
