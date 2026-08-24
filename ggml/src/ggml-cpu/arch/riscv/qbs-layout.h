#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ggml::cpu::riscv::qbs {

// Convert GGML row-major blocks into the QBS four-row, block-major layout.
inline void repack_r4(uint8_t * destination, const uint8_t * source,
                      size_t output_rows, size_t k_blocks,
                      size_t block_bytes) {
    const size_t padded_rows = (output_rows + 3) & ~size_t(3);
    std::memset(destination, 0, padded_rows * k_blocks * block_bytes);
    for (size_t row = 0; row < output_rows; ++row) {
        for (size_t block = 0; block < k_blocks; ++block) {
            const size_t source_index = row * k_blocks + block;
            const size_t destination_index =
                ((row / 4) * k_blocks + block) * 4 + row % 4;
            std::memcpy(destination + destination_index * block_bytes,
                        source + source_index * block_bytes, block_bytes);
        }
    }
}

// Return the start of one four-context activation block. Consecutive groups
// are laid out as [group][k_block][four interleaved contexts].
inline const uint8_t * activation_m4_group_block(
        const uint8_t * source, size_t input_row, size_t k_block,
        size_t k_blocks, size_t activation_block_bytes) {
    const size_t group = input_row / 4;
    return source + (group * k_blocks + k_block) * 4 * activation_block_bytes;
}

} // namespace ggml::cpu::riscv::qbs
