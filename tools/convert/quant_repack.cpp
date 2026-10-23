#include "quant_repack.h"
#include <cstring>
#include <iostream>

namespace diffusion {

bool repack_tensor_data(
    uint32_t version,
    int32_t ggml_type,
    const int64_t* /*ne*/,
    int32_t /*n_dims*/,
    const uint8_t* in_data,
    size_t in_size,
    std::vector<uint8_t>& out_buffer,
    const uint8_t*& out_data,
    size_t& out_size
) {
    if (version == 3) {
        // v3 blocks are directly compatible with modern GGML
        out_data = in_data;
        out_size = in_size;
        return true;
    }

    if (version == 1 || version == 2) {
        if (ggml_type == GGML_TYPE_F32 || ggml_type == GGML_TYPE_F16) {
            out_data = in_data;
            out_size = in_size;
            return true;
        }

        if (ggml_type == GGML_TYPE_Q4_0) {
            if (in_size % 20 != 0) {
                std::cerr << "Invalid Q4_0 input size for v" << version << ": " << in_size << "\n";
                return false;
            }
            size_t n_blocks = in_size / 20;
            out_buffer.resize(n_blocks * 18);

            for (size_t b = 0; b < n_blocks; ++b) {
                float d = 0.0f;
                std::memcpy(&d, in_data + b * 20, 4);
                ggml_fp16_t d16 = ggml_fp32_to_fp16(d);
                std::memcpy(out_buffer.data() + b * 18, &d16, 2);

                const uint8_t* qs_in = in_data + b * 20 + 4;
                uint8_t* qs_out = out_buffer.data() + b * 18 + 2;

                if (version == 1) {
                    // v1 swapped nibbles
                    for (int j = 0; j < 16; ++j) {
                        uint8_t x = qs_in[j];
                        qs_out[j] = static_cast<uint8_t>(((x >> 4) & 0x0F) | ((x & 0x0F) << 4));
                    }
                } else {
                    // v2 already has modern low-nibble first ordering
                    std::memcpy(qs_out, qs_in, 16);
                }
            }

            out_data = out_buffer.data();
            out_size = out_buffer.size();
            return true;
        }

        if (ggml_type == GGML_TYPE_Q4_1) {
            if (in_size % 24 != 0) return false;
            size_t n_blocks = in_size / 24;
            out_buffer.resize(n_blocks * 20);

            for (size_t b = 0; b < n_blocks; ++b) {
                float d = 0.0f, m = 0.0f;
                std::memcpy(&d, in_data + b * 24 + 0, 4);
                std::memcpy(&m, in_data + b * 24 + 4, 4);
                ggml_fp16_t d16 = ggml_fp32_to_fp16(d);
                ggml_fp16_t m16 = ggml_fp32_to_fp16(m);

                std::memcpy(out_buffer.data() + b * 20 + 0, &d16, 2);
                std::memcpy(out_buffer.data() + b * 20 + 2, &m16, 2);
                std::memcpy(out_buffer.data() + b * 20 + 4, in_data + b * 24 + 8, 16);
            }

            out_data = out_buffer.data();
            out_size = out_buffer.size();
            return true;
        }

        if (ggml_type == GGML_TYPE_Q8_0) {
            if (in_size % 36 != 0) return false;
            size_t n_blocks = in_size / 36;
            out_buffer.resize(n_blocks * 34);

            for (size_t b = 0; b < n_blocks; ++b) {
                float d = 0.0f;
                std::memcpy(&d, in_data + b * 36, 4);
                ggml_fp16_t d16 = ggml_fp32_to_fp16(d);

                std::memcpy(out_buffer.data() + b * 34, &d16, 2);
                std::memcpy(out_buffer.data() + b * 34 + 2, in_data + b * 36 + 4, 32);
            }

            out_data = out_buffer.data();
            out_size = out_buffer.size();
            return true;
        }
    }

    std::cerr << "Unsupported GGJT tensor type: " << ggml_type << " for version " << version << "\n";
    return false;
}

} // namespace diffusion
