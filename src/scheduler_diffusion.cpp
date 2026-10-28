#include "scheduler_diffusion.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace diffusion {

SchedulerDiffusion::SchedulerDiffusion(
    IModelBackend& backend,
    const ITokenizer& tokenizer,
    ISamplerChain& sampler,
    int default_block_size,
    int default_max_diffusion_steps
) : backend_(backend),
    tokenizer_(tokenizer),
    sampler_(sampler),
    default_block_size_(default_block_size > 0 ? default_block_size : 8),
    default_max_diffusion_steps_(default_max_diffusion_steps > 0 ? default_max_diffusion_steps : 4) {
    if (diff_device_alloc(device_bufs_, static_cast<size_t>(default_block_size_ * 2))) {
        device_bufs_allocated_ = true;
    }
}

SchedulerDiffusion::~SchedulerDiffusion() {
    if (device_bufs_allocated_) {
        kernels::diff_device_free(device_bufs_);
        device_bufs_allocated_ = false;
    }
}

GenResult SchedulerDiffusion::run(
    const std::vector<int32_t>& prompt,
    const GenParams& params,
    int reuse_prefix_len,
    const StepCallback& callback
) {
    GenResult result;
    if (prompt.empty()) return result;

    std::vector<int32_t> effective_prompt = prompt;
    if (backend_.n_ctx() > 0 && static_cast<int>(effective_prompt.size()) >= backend_.n_ctx()) {
        int max_prompt = std::max(1, backend_.n_ctx() - std::max(64, params.n_predict > 0 ? std::min(params.n_predict, 512) : 128));
        if (static_cast<int>(effective_prompt.size()) > max_prompt) {
            std::vector<int32_t> trimmed;
            trimmed.push_back(effective_prompt.front());
            int offset = static_cast<int>(effective_prompt.size()) - (max_prompt - 1);
            trimmed.insert(trimmed.end(), effective_prompt.begin() + offset, effective_prompt.end());
            effective_prompt = std::move(trimmed);
            reuse_prefix_len = 0;
        }
    }

    int prompt_len = static_cast<int>(effective_prompt.size());
    if (reuse_prefix_len < 0) reuse_prefix_len = 0;
    if (reuse_prefix_len > prompt_len) reuse_prefix_len = prompt_len;

    auto t_start = std::chrono::steady_clock::now();

    int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int B_target = (params.block_size > 1) ? params.block_size : default_block_size_;
    if (!backend_.supports_kv_truncate() || B_target < 1) {
        B_target = 1;
    }
    int M_max = (params.max_iters > 0 && params.max_iters <= 16) ? params.max_iters : default_max_diffusion_steps_;
    int max_iters = (params.max_iters > 0 && params.max_iters > 16) ? params.max_iters : (n_predict + 16);

    // Ensure device buffer capacity accommodates block size
    if (!device_bufs_allocated_ || device_bufs_.capacity < static_cast<size_t>(B_target)) {
        if (device_bufs_allocated_) {
            kernels::diff_device_free(device_bufs_);
            device_bufs_allocated_ = false;
        }
        if (kernels::diff_device_alloc(device_bufs_, static_cast<size_t>(B_target * 2))) {
            device_bufs_allocated_ = true;
        }
    }

    // 1. Prefill prompt (with optional KV prefix reuse)
    LogitsView prefill_logits;
    if (reuse_prefix_len == 0) {
        backend_.kv_clear(params.seq_id);
        prefill_logits = backend_.eval(effective_prompt.data(), prompt_len, 0, false, params.seq_id);
        for (int32_t tok : effective_prompt) {
            sampler_.accept(tok);
        }
    } else {
        backend_.kv_truncate(reuse_prefix_len, params.seq_id);
        int new_len = prompt_len - reuse_prefix_len;
        if (new_len > 0) {
            prefill_logits = backend_.eval(effective_prompt.data() + reuse_prefix_len, new_len, reuse_prefix_len, false, params.seq_id);
            std::vector<int32_t> new_prompt_slice(effective_prompt.begin() + reuse_prefix_len, effective_prompt.end());
            for (int32_t tok : new_prompt_slice) {
                sampler_.accept(tok);
            }
        } else {
            backend_.kv_truncate(prompt_len - 1, params.seq_id);
            prefill_logits = backend_.eval(effective_prompt.data() + prompt_len - 1, 1, prompt_len - 1, false, params.seq_id);
        }
    }

    int P = prompt_len;
    StopReason stop_reason = StopReason::NONE;

    // Buffer prefix logits predicting position P
    int n_vocab = backend_.n_vocab();
    std::vector<float> prefix_logits(n_vocab, 0.0f);
    const float* prefill_row = prefill_logits.row(prefill_logits.n_rows - 1);
    if (prefill_row) {
        std::copy(prefill_row, prefill_row + n_vocab, prefix_logits.begin());
    }

    if (B_target == 1) {
        // Causal single-token decode path
        float p0_conf = 1.0f;
        int32_t current_token = sampler_.sample(prefix_logits.data(), P, &p0_conf);

        while (static_cast<int>(result.tokens.size()) < n_predict && result.iterations < max_iters) {
            result.iterations++;
            result.total_diffusion_passes++;
            result.cumulative_active_work += 1;

            if (P + 1 >= backend_.n_ctx()) {
                if (params.ctx_shift && backend_.supports_kv_shift()) {
                    int n_keep = params.n_keep > 0 ? params.n_keep : 1;
                    int n_discard = params.n_discard > 0 ? params.n_discard : std::max(1, (backend_.n_ctx() - n_keep) / 2);
                    if (n_discard > 0 && P - n_discard >= n_keep) {
                        backend_.kv_truncate(n_keep + n_discard, params.seq_id);
                        backend_.kv_shift(params.seq_id, n_keep + n_discard, P, -n_discard);
                        P -= n_discard;
                    } else {
                        stop_reason = StopReason::MAX_TOKENS;
                        break;
                    }
                } else {
                    stop_reason = StopReason::MAX_TOKENS;
                    break;
                }
            }

            LogitsView L = backend_.eval(&current_token, 1, P, params.logprobs, params.seq_id);

            float next_conf = 1.0f;
            int32_t next_tok = sampler_.sample(L.row(0), P + 1, &next_conf);

            sampler_.accept(current_token);
            result.tokens.push_back(current_token);
            result.total_blocks++;

            if (tokenizer_.is_eog(current_token)) {
                stop_reason = StopReason::EOG;
                if (callback) {
                    StepEvent ev;
                    ev.step = result.iterations;
                    ev.committed_upto = static_cast<int>(result.tokens.size());
                    ev.committed_new = { current_token };
                    ev.done = true;
                    ev.reason = stop_reason;
                    callback(ev);
                }
                break;
            }

            result.text += tokenizer_.piece(current_token);

            bool should_stop = false;
            for (const auto& s : params.stop_strings) {
                if (!s.empty() && result.text.size() >= s.size()) {
                    if (result.text.compare(result.text.size() - s.size(), s.size(), s) == 0) {
                        result.text.resize(result.text.size() - s.size());
                        stop_reason = StopReason::STOP_STRING;
                        should_stop = true;
                        break;
                    }
                }
            }

            if (!should_stop && static_cast<int>(result.tokens.size()) >= n_predict) {
                stop_reason = StopReason::MAX_TOKENS;
                should_stop = true;
            }

            if (callback) {
                StepEvent ev;
                ev.step = result.iterations;
                ev.committed_upto = static_cast<int>(result.tokens.size());
                ev.committed_new = { current_token };
                ev.canvas = { next_tok };
                ev.done = should_stop;
                ev.reason = stop_reason;
                if (!callback(ev)) {
                    stop_reason = StopReason::CANCELLED;
                    break;
                }
            }

            if (should_stop) break;

            P += 1;
            current_token = next_tok;
        }
    } else {
        // Discrete Block Diffusion Inference Loop
        while (static_cast<int>(result.tokens.size()) < n_predict && result.iterations < max_iters) {
            int B_cur = std::min(B_target, n_predict - static_cast<int>(result.tokens.size()));
            if (B_cur <= 0) break;

            // Context limit handling
            if (P + B_cur >= backend_.n_ctx()) {
                if (params.ctx_shift && backend_.supports_kv_shift()) {
                    int n_keep = params.n_keep > 0 ? params.n_keep : 1;
                    int n_discard = params.n_discard > 0 ? params.n_discard : std::max(1, (backend_.n_ctx() - n_keep) / 2);
                    if (n_discard > 0 && P - n_discard >= n_keep) {
                        backend_.kv_truncate(n_keep + n_discard, params.seq_id);
                        backend_.kv_shift(params.seq_id, n_keep + n_discard, P, -n_discard);
                        P -= n_discard;
                    } else {
                        stop_reason = StopReason::MAX_TOKENS;
                        break;
                    }
                } else {
                    stop_reason = StopReason::MAX_TOKENS;
                    break;
                }
            }

            // Phase 1: Model-Derived Prior Seeding
            // Token 0 at position P is sampled from prefix logits
            float conf0 = 1.0f;
            int32_t x0 = sampler_.sample(prefix_logits.data(), P, &conf0);

            std::vector<int32_t> canvas(B_cur, x0);
            std::vector<float> confidences(B_cur, 0.0f);
            std::vector<uint8_t> frozen_mask(B_cur, 0);

            canvas[0] = x0;
            confidences[0] = conf0;
            if (conf0 >= 0.88f) {
                frozen_mask[0] = 1;
            }

            LogitsView L;

            // Phase 2: Iterative Discrete Denoising Loop (m = 1 ... M_max)
            for (int m = 1; m <= M_max; ++m) {
                result.iterations++;
                result.total_diffusion_passes++;

                // Formal freeze threshold schedule: theta_freeze(m) = theta0 + (1 - theta0) * (m - 1) / (M_max - 1)
                float theta0 = 0.88f;
                float theta_freeze = theta0 + (1.0f - theta0) * (static_cast<float>(m - 1) / std::max(1, M_max - 1));

                // Partition Active Query Set vs Frozen KV Anchor Set
                int W_active = 0;
                std::vector<int32_t> active_indices;
                active_indices.reserve(B_cur);

                if (m == 1) {
                    // Pass 1: Full block evaluation to establish initial representations
                    W_active = B_cur;
                    for (int i = 0; i < B_cur; ++i) {
                        active_indices.push_back(i);
                    }
                } else {
                    for (int i = 0; i < B_cur; ++i) {
                        if (confidences[i] >= theta_freeze) {
                            frozen_mask[i] = 1;
                        } else {
                            frozen_mask[i] = 0;
                            active_indices.push_back(i);
                        }
                    }
                    W_active = static_cast<int>(active_indices.size());
                    if (W_active == 0) {
                        // Fixed point reached early: all tokens locked
                        break;
                    }
                }

                // Primary KPI: Cumulative Active Work tracking
                result.cumulative_active_work += W_active;

                // Build batch active mask: row i-1 needs logits if position i is active.
                // Row B_cur - 1 needs logits for next block's prefix seed.
                std::vector<uint8_t> batch_active_mask(B_cur, 0);
                for (int i = 1; i < B_cur; ++i) {
                    if (frozen_mask[i] == 0) {
                        batch_active_mask[i - 1] = 1;
                    }
                }
                batch_active_mask[B_cur - 1] = 1;

                // Forward pass with sparse active queries (skips lm_head for frozen anchors)
                L = backend_.eval_active(canvas.data(), B_cur, P, batch_active_mask.data(), params.seq_id);

                // Denoise active positions
                bool changed = false;
                for (int i = 1; i < B_cur; ++i) {
                    if (frozen_mask[i] != 0) {
                        continue; // Frozen token: remains static anchor
                    }

                    // Position i is predicted by position i - 1 in the batch
                    const float* row = backend_.get_logits_ith(i - 1);
                    if (!row) row = L.row(i - 1);
                    if (!row) continue;

                    // Compute argmax and top softmax probability
                    float max_val = row[0];
                    int32_t best_tok = 0;
                    for (int v = 1; v < n_vocab; ++v) {
                        if (row[v] > max_val) {
                            max_val = row[v];
                            best_tok = v;
                        }
                    }

                    double sum_exp = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        sum_exp += std::exp(row[v] - max_val);
                    }
                    float conf = (sum_exp > 0.0) ? static_cast<float>(1.0 / sum_exp) : 1.0f;

                    if (best_tok != canvas[i]) {
                        canvas[i] = best_tok;
                        changed = true;
                    }
                    confidences[i] = conf;

                    if (conf >= theta_freeze) {
                        frozen_mask[i] = 1;
                    }
                }

                // Check early exit stationarity: no active token changed state
                if (m > 1 && !changed) {
                    break;
                }
            }

            // Phase 3: Commit Block & Stream
            result.total_blocks++;
            std::vector<int32_t> committed_this_block;
            bool should_stop = false;
            int committed_count = 0;

            for (int j = 0; j < B_cur; ++j) {
                int32_t tok = canvas[j];
                sampler_.accept(tok);
                result.tokens.push_back(tok);
                committed_this_block.push_back(tok);
                committed_count++;

                if (tokenizer_.is_eog(tok)) {
                    stop_reason = StopReason::EOG;
                    should_stop = true;
                    break;
                }

                result.text += tokenizer_.piece(tok);

                for (const auto& s : params.stop_strings) {
                    if (!s.empty() && result.text.size() >= s.size()) {
                        if (result.text.compare(result.text.size() - s.size(), s.size(), s) == 0) {
                            result.text.resize(result.text.size() - s.size());
                            stop_reason = StopReason::STOP_STRING;
                            should_stop = true;
                            break;
                        }
                    }
                }
                if (should_stop) break;

                if (static_cast<int>(result.tokens.size()) >= n_predict) {
                    stop_reason = StopReason::MAX_TOKENS;
                    should_stop = true;
                    break;
                }
            }

            if (callback) {
                StepEvent ev;
                ev.step = result.iterations;
                ev.committed_upto = static_cast<int>(result.tokens.size());
                ev.committed_new = committed_this_block;
                ev.canvas = canvas;
                ev.done = should_stop;
                ev.reason = stop_reason;
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double, std::milli>(now - t_start).count();
                ev.tok_per_s = elapsed > 0.0 ? (result.tokens.size() * 1000.0 / elapsed) : 0.0;
                ev.tok_per_iter = result.iterations > 0 ? (static_cast<double>(result.tokens.size()) / result.iterations) : 0.0;
                if (!callback(ev)) {
                    stop_reason = StopReason::CANCELLED;
                    should_stop = true;
                }
            }

            if (should_stop) {
                if (committed_count < B_cur) {
                    backend_.kv_truncate(P + committed_count, params.seq_id);
                }
                break;
            }

            // Advance position P and update prefix logits for next block from row B_cur - 1
            P += B_cur;
            const float* next_p_row = backend_.get_logits_ith(B_cur - 1);
            if (!next_p_row) next_p_row = L.row(B_cur - 1);
            if (next_p_row) {
                std::copy(next_p_row, next_p_row + n_vocab, prefix_logits.begin());
            }
        }
    }

    if (result.iterations >= max_iters && stop_reason == StopReason::NONE) {
        stop_reason = StopReason::MAX_TOKENS;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.tok_per_iter = result.iterations > 0 ? (static_cast<double>(result.tokens.size()) / result.iterations) : 0.0;
    result.tok_per_s = result.total_time_ms > 0.0 ? (result.tokens.size() * 1000.0 / result.total_time_ms) : 0.0;
    result.stop_reason = stop_reason;

    if (result.total_diffusion_passes > 0) {
        result.avg_active_tokens_per_pass = static_cast<double>(result.cumulative_active_work) / result.total_diffusion_passes;
    }
    if (result.total_blocks > 0) {
        result.avg_passes_per_block = static_cast<double>(result.total_diffusion_passes) / result.total_blocks;
    }

    return result;
}

} // namespace diffusion
