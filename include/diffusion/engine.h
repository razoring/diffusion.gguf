#pragma once

#include "events.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace diffusion {

enum class DecodeMode {
    BLOCK = 0,
    WHOLE
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

enum class ActiveSetMode {
    DISABLED = 0,    // Uniform block refilling (baseline, W = B_target fixed)
    NATURAL_COLLAPSE // Active-set refinement (W naturally collapses as prefix commits)
};

enum class TailPruningMode {
    OFF = 0,
    ADAPTIVE,
    AGGRESSIVE
};

enum class CommitmentPolicy {
    FULL_PREFIX_FREEZE = 0,
    SOFT_FREEZE,            // Primary focus: provisional commitment with rollback safety
    UNCOMMITTED_FRONTIER    // Roving window: commit only at block convergence
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

struct FrontierTelemetry {
    uint64_t total_forward_passes = 0;
    uint64_t total_tokens_committed = 0;
    uint64_t total_width_sum = 0;
    int32_t max_width = 0;
    int32_t min_width = 100000;
    uint64_t degenerate_w1_passes = 0;

    // Frontier Mobility
    uint64_t expansions = 0;           // Passes where W(t) > W(t-1)
    uint64_t contractions = 0;         // Passes where W(t) < W(t-1)
    uint64_t shifts = 0;               // Passes where start_index shifted
    uint64_t stationary_passes = 0;    // Passes where window stayed fixed
    double avg_lifetime_passes = 0.0;  // Average iterations a window persists before shift

    // Distributions
    std::map<int32_t, uint64_t> width_pass_count;       // W -> total passes executed
    std::map<int32_t, int64_t> width_runtime_us;        // W -> total execution time (us)
    std::map<int32_t, uint64_t> width_tokens_committed; // W -> tokens accepted/committed in passes of width W
    std::vector<int32_t> position_flips;                // Relative position within candidate block -> flip count

    // Derived Computations
    double avg_frontier_width() const {
        return total_forward_passes > 0 ? static_cast<double>(total_width_sum) / total_forward_passes : 0.0;
    }
    double effective_parallelism_ratio(int32_t max_block) const {
        return max_block > 0 ? avg_frontier_width() / max_block : 0.0;
    }
    std::map<int32_t, double> time_weighted_width_histogram() const {
        int64_t total_us = 0;
        for (const auto& kv : width_runtime_us) total_us += kv.second;
        std::map<int32_t, double> dist;
        if (total_us > 0) {
            for (const auto& kv : width_runtime_us) {
                dist[kv.first] = (static_cast<double>(kv.second) / total_us) * 100.0;
            }
        }
        return dist;
    }
    std::map<int32_t, double> empirical_throughput_by_width() const {
        std::map<int32_t, double> tp;
        for (const auto& kv : width_runtime_us) {
            int32_t w = kv.first;
            int64_t us = kv.second;
            auto it = width_tokens_committed.find(w);
            if (us > 0 && it != width_tokens_committed.end()) {
                tp[w] = (static_cast<double>(it->second) / us) * 1e6;
            }
        }
        return tp;
    }
    int32_t compute_w_critical(double retention_ratio = 0.90) const {
        auto tp = empirical_throughput_by_width();
        if (tp.empty()) return 1;
        double peak = 0.0;
        for (const auto& kv : tp) {
            if (kv.second > peak) peak = kv.second;
        }
        if (peak <= 0.0) return 1;
        int32_t w_crit = 1;
        for (const auto& kv : tp) {
            if (kv.second >= peak * retention_ratio) {
                w_crit = kv.first;
                break; // smallest W that meets threshold
            }
        }
        return w_crit;
    }
};

struct GenParams {
    int32_t n_predict = 128;
    DecodeMode mode = DecodeMode::BLOCK;
    int32_t block_size = 1; // 1 = exact standard single-token decode; >1 requires explicit speculative opt-in
    int32_t whole_max = 512;
    InitStrategy init_strategy = InitStrategy::NGRAM;
    int32_t ngram_n = 3;
    int32_t ngram_pool_max = 4096;
    int32_t max_iters = 0; // 0 = default (n_predict + 8)

    // Speculative draft model options
    std::string model_draft;
    int32_t draft_n_gpu_layers = 0;
    int32_t draft_n_predict = 4;

    // Active-Set & Diffusion options
    ActiveSetMode active_set_mode = ActiveSetMode::NATURAL_COLLAPSE;
    TailPruningMode tail_pruning = TailPruningMode::OFF;
    CommitmentPolicy commitment_policy = CommitmentPolicy::SOFT_FREEZE;

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
    FrontierTelemetry telemetry;
};

} // namespace diffusion
