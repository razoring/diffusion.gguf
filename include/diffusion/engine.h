#pragma once

#include "events.h"
#include <cstdint>
#include <string>
#include <vector>

namespace diffusion {

enum class DecodeMode {
    BLOCK = 0,
    WHOLE,
    AR
};

enum class InitStrategy {
    NGRAM = 0,
    RANDOM,
    PAD,
    DRAFT
};

enum class RngMode {
    POSITIONAL = 0,
    LLAMA_COMPAT
};

struct SamplerParams {
    float temp = 0.8f;
    float dynatemp_range = 0.0f;
    float dynatemp_exponent = 1.0f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float typical_p = 1.0f;
    float xtc_probability = 0.0f;
    float xtc_threshold = 0.1f;

    // Penalties
    float repeat_penalty = 1.0f;
    int32_t repeat_last_n = 64;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;

    // DRY
    float dry_multiplier = 0.0f;
    float dry_base = 1.75f;
    int32_t dry_allowed_length = 2;
    int32_t dry_penalty_last_n = -1;

    // Mirostat
    int32_t mirostat = 0; // 0=disabled, 1=v1, 2=v2
    float mirostat_tau = 5.0f;
    float mirostat_eta = 0.1f;

    // Logit bias (token_id -> bias value)
    std::vector<std::pair<int32_t, float>> logit_bias;

    // Additional samplers
    float top_n_sigma = 0.0f;
    float adaptive_p_target = -1.0f;
    float adaptive_p_decay = 0.9f;
    bool infill = false;

    // Grammar & constraints
    std::string grammar;
    std::string json_schema;

    uint32_t seed = 42;
    RngMode rng_mode = RngMode::POSITIONAL;
};

struct LogprobItem {
    int32_t token = 0;
    std::string piece;
    float logprob = 0.0f;
};

struct TokenLogprobs {
    int32_t token = 0;
    std::string piece;
    float logprob = 0.0f;
    std::vector<LogprobItem> top_logprobs;
};

struct GenParams {
    int32_t n_predict = 128;
    DecodeMode mode = DecodeMode::BLOCK;
    int32_t block_size = 32;
    int32_t whole_max = 512;
    InitStrategy init_strategy = InitStrategy::NGRAM;
    int32_t ngram_n = 3;
    int32_t ngram_pool_max = 4096;
    int32_t max_iters = 0; // 0 = default (n_predict + 8)

    // Speculative draft model options
    std::string model_draft;
    int32_t draft_n_gpu_layers = 0;
    int32_t draft_n_predict = 4;

    // Prompt cache options
    std::string prompt_cache_path;
    bool prompt_cache_ro = false;
    bool prompt_cache_all = false;

    // Context management
    bool ctx_shift = true;
    int32_t n_keep = 0;
    int32_t n_discard = 0;

    // Logprobs
    bool logprobs = false;
    int32_t top_logprobs = 0;

    // Reasoning extraction (<think> ... </think>)
    bool extract_reasoning = false;
    int32_t reasoning_budget = 0;

    // Multi-sequence slot id
    int32_t seq_id = 0;

    std::vector<std::string> stop_strings;
    SamplerParams sampling;
};

struct GenResult {
    std::vector<int32_t> tokens;
    std::string text;
    std::string reasoning_content;
    std::vector<TokenLogprobs> token_logprobs;
    int iterations = 0;
    double tok_per_iter = 0.0;
    double tok_per_s = 0.0;
    double total_time_ms = 0.0;
    StopReason stop_reason = StopReason::NONE;
};

} // namespace diffusion
