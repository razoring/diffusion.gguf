#pragma once

#include "ggjt_reader.h"
#include <string>

namespace diffusion {

struct ConvertOptions {
    std::string name;
    int32_t ctx_len = 2048;
    float eps = 1e-6f;
    int32_t gqa = 1;
};

bool emit_gguf_file(
    const GgjtReader& reader,
    const std::string& out_path,
    const ConvertOptions& options
);

} // namespace diffusion
