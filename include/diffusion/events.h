#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace diffusion {

enum class StopReason {
    NONE = 0,
    EOG,
    MAX_TOKENS,
    STOP_STRING,
    CANCELLED
};

struct TokenDiff {
    int pos = 0;
    int32_t old_id = -1;
    int32_t new_id = -1;
    std::string old_text;
    std::string new_text;
    float confidence = 0.0f;
};

struct StepEvent {
    int step = 0;
    int committed_upto = 0;
    std::vector<int32_t> committed_new;
    std::vector<int32_t> canvas;
    std::vector<TokenDiff> diffs;
    bool done = false;
    StopReason reason = StopReason::NONE;
    double tok_per_s = 0.0;
    double tok_per_iter = 0.0;
};

using StepCallback = std::function<bool(const StepEvent&)>;

} // namespace diffusion
