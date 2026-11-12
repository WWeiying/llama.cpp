#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ggml_qbs_strided_input {

// Quantization scales are independent per block. Gather only one block of
// logical rows, avoiding a full-K allocation and leaving the dense path alone.
template <size_t Rows, size_t Block, size_t OutputBlockBytes, typename Quantize>
__attribute__((noinline)) void quantize(const char *input, void *output, int64_t k,
                                      size_t element_stride, size_t row_stride,
                                      Quantize produce) {
    static_assert(Rows == 1 || Rows == 4, "unsupported row tile");
    static_assert(Block == 32 || Block == 256, "unsupported quantization block");
    assert(k >= 0 && k % Block == 0);
    alignas(64) float gathered[Rows * Block];
    auto *bytes = static_cast<unsigned char *>(output);
    for (size_t block = 0; block < size_t(k) / Block; ++block) {
        for (size_t row = 0; row < Rows; ++row) {
            const char *source = input + row*row_stride + block*Block*element_stride;
            if (element_stride == sizeof(float)) {
                std::memcpy(gathered + row*Block, source, Block*sizeof(float));
            } else {
                for (size_t col = 0; col < Block; ++col) {
                    std::memcpy(gathered + row*Block + col, source + col*element_stride, sizeof(float));
                }
            }
        }
        produce(gathered, bytes + block*OutputBlockBytes, int64_t(Block));
    }
}

} // namespace ggml_qbs_strided_input
