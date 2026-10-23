#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace diffusion {

bool repack_tensor_data(
    uint32_t version,
    int32_t ggml_type,
    const int64_t* ne,
    int32_t n_dims,
    const uint8_t* in_data,
    size_t in_size,
    std::vector<uint8_t>& out_buffer,
    const uint8_t*& out_data,
    size_t& out_size
);

} // namespace diffusion
