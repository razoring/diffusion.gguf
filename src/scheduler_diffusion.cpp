#include "scheduler_diffusion.h"
#include "init_guess.h"
#include "fast_math.h"
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
    int default_max_diffusion_steps,
    InitGuesser* guesser
) : backend_(backend),
    tokenizer_(tokenizer),
    sampler_(sampler),
    default_block_size_(default_block_size > 0 ? default_block_size : 8),
    default_max_diffusion_steps_(default_max_diffusion_steps > 0 ? default_max_diffusion_steps : 3),
    guesser_(guesser) {
    if (kernels::diff_device_alloc(device_bufs_, static_cast<size_t>(default_block_size_ * 2), backend_.n_vocab())) {
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
        if (kernels::diff_device_alloc(device_bufs_, static_cast<size_t>(B_target * 2), backend_.n_vocab())) {
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

    // Ensure device buffer capacity accommodates block size and matches the active output GPU device
    int cur_device = kernels::diff_device_get_current_device();
    if (!device_bufs_allocated_ || device_bufs_.capacity < static_cast<size_t>(B_target) || device_bufs_.device_id != cur_device) {
        if (device_bufs_allocated_) {
            kernels::diff_device_free(device_bufs_);
            device_bufs_allocated_ = false;
        }
        if (kernels::diff_device_alloc(device_bufs_, static_cast<size_t>(B_target * 2), backend_.n_vocab())) {
            device_bufs_allocated_ = true;
        }
    }

    int P = prompt_len;
    StopReason stop_reason = StopReason::NONE;

    std::vector<int32_t> running_history = effective_prompt;
    if (guesser_) {
        guesser_->update_ngram(effective_prompt);
    }

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
                    ev.committed_new = {};
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
        std::vector<int32_t> leftover_canvas;
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

            // Phase 1: Model-Derived Prior Seeding & Entropy Gating
            float conf0 = 1.0f;
            int32_t x0 = sampler_.sample(prefix_logits.data(), P, &conf0);

            // Compute Shannon entropy H0 of prefix logits (only when K=2 branch budget enabled)
            float H0 = 0.0f;
            if (params.multi_candidate_k == 2) {
                float max_l0 = prefix_logits[0];
                for (int v = 1; v < n_vocab; ++v) {
                    if (prefix_logits[v] > max_l0) max_l0 = prefix_logits[v];
                }
                double sum_exp0 = 0.0, sum_z_exp0 = 0.0;
                for (int v = 0; v < n_vocab; ++v) {
                    double e = std::exp(static_cast<double>(prefix_logits[v] - max_l0));
                    sum_exp0 += e;
                    sum_z_exp0 += static_cast<double>(prefix_logits[v] - max_l0) * e;
                }
                if (sum_exp0 > 0.0) {
                    H0 = static_cast<float>(std::log(sum_exp0) - (sum_z_exp0 / sum_exp0));
                }
            }

            // K=2 Branch Budget Controller (moving 100-block window)
            bool should_branch = (params.multi_candidate_k == 2) &&
                                 (H0 >= params.entropy_fork_threshold) &&
                                 (get_branch_frequency_100() < params.max_branch_ratio);
            record_branch_event(should_branch ? 1 : 0);

            std::vector<int32_t> canvas(B_cur, x0);
            std::vector<float> confidences(B_cur, 0.0f);
            std::vector<uint8_t> frozen_mask(B_cur, 0);

            canvas[0] = x0;
            confidences[0] = conf0;
            frozen_mask[0] = 1; // Position 0 is already conditioned on prompt prefix

            int filled = 1;
            // 1. Rollover uncommitted lookahead predictions from previous block
            for (size_t i = 0; i < leftover_canvas.size() && filled < B_cur; ++i) {
                canvas[filled++] = leftover_canvas[i];
            }

            // 2. If slots remain, query guesser from running history
            if (filled < B_cur && guesser_) {
                running_history.push_back(x0);
                auto guesses = guesser_->make_guesses(B_cur - filled, running_history);
                running_history.pop_back();
                for (int32_t g : guesses) {
                    if (filled >= B_cur) break;
                    canvas[filled++] = g;
                }
            }

            // 3. Fallback for open slots: pad token
            int32_t fallback_pad = tokenizer_.pad_or_fallback();
            if (fallback_pad < 0) fallback_pad = x0;
            while (filled < B_cur) {
                canvas[filled++] = fallback_pad;
            }

            LogitsView L;
            bool converged = false;
            int M_max_cur = (M_max > 0) ? M_max : 3;

            int last_pass_m = 1;
            int prev_frozen = 1;
            // Phase 2: Hybrid Annealed Discrete Denoising Loop (m = 1 ... M_max)
            for (int m = 1; m <= M_max_cur; ++m) {
                last_pass_m = m;
                result.iterations++;
                result.total_diffusion_passes++;

                // Partition Active Query Set vs Frozen KV Anchor Set
                int W_active = 0;
                for (int i = 0; i < B_cur; ++i) {
                    if (frozen_mask[i] == 0) W_active++;
                }

                if (m > 1 && W_active == 0) {
                    converged = true;
                    break;
                }
                result.cumulative_active_work += (m == 1 ? B_cur : W_active);

                // Hybrid Annealed Attention Position Schedule
                std::vector<int32_t> custom_pos(B_cur);
                if (params.annealed_attention) {
                    if (m == 1) {
                        // Pass 1: Full diffusion attention (all canvas tokens share position P)
                        for (int i = 0; i < B_cur; ++i) custom_pos[i] = P;
                    } else if (m == 2) {
                        // Pass 2: Partially annealed attention (pairwise lookahead)
                        for (int i = 0; i < B_cur; ++i) custom_pos[i] = P + (i / 2);
                    } else {
                        // Pass 3+: Strict causal attention before commitment
                        for (int i = 0; i < B_cur; ++i) custom_pos[i] = P + i;
                    }
                } else {
                    for (int i = 0; i < B_cur; ++i) custom_pos[i] = P + i;
                }

                // Rewind uncommitted KV cache before re-evaluating block in pass m > 1
                if (m > 1) {
                    backend_.kv_truncate(P, params.seq_id);
                }

                std::vector<int32_t> canvas_in_pass = canvas;
                // Forward pass with resident KV anchors
                L = backend_.eval_custom_pos(canvas.data(), B_cur, custom_pos.data(), true, params.seq_id);

                // Quadratic freeze schedule
                float theta0 = 0.85f;
                float progress = static_cast<float>(m - 1) / std::max(1, M_max_cur - 1);
                float theta_freeze = theta0 + (1.0f - theta0) * (progress * progress);

                // High-performance SIMD Categorical Reduction on Host Logits
                for (int i = 1; i < B_cur; ++i) {
                    const float* row = L.row(i - 1);
                    if (!row) row = backend_.get_logits_ith(i - 1);
                    if (!row) break;

                    int32_t best_tok = fast_argmax_simd(row, n_vocab);
                    float conf = fast_confidence(row, n_vocab, best_tok);
                    confidences[i] = conf;
                    canvas[i] = best_tok;
                }

                // Advance dynamic frontier strictly along verified stationary tokens (canvas_in == canvas_out)
                frozen_mask[0] = 1; // Preserve prefix token 0
                int frontier = 1;
                while (frontier < B_cur && frozen_mask[frontier] == 1) frontier++;
                while (frontier < B_cur) {
                    int b = frontier;
                    bool stable = (canvas_in_pass[b] == canvas[b]);
                    if (stable) {
                        frozen_mask[b] = 1;
                        frontier++;
                    } else {
                        break; // Stop frontier at first unverified token
                    }
                }

                bool all_frozen = (frontier >= B_cur);
                if (all_frozen || (m > 1 && frontier <= prev_frozen)) {
                    converged = all_frozen;
                    break;
                }
                prev_frozen = frontier;
            }


            // Phase 3: Dynamic Frontier Commitment
            int k_commit = 1;
            while (k_commit < B_cur && frozen_mask[k_commit] != 0) {
                k_commit++;
            }

            // Ensure committed tokens are encoded under strict causal attention before permanent KV commitment
            bool last_pass_was_causal = (last_pass_m >= 3) || !params.annealed_attention;
            if (last_pass_was_causal) {
                if (k_commit < B_cur) {
                    backend_.kv_truncate(P + k_commit, params.seq_id);
                }
            } else {
                backend_.kv_truncate(P, params.seq_id);
                std::vector<int32_t> causal_pos(k_commit);
                for (int i = 0; i < k_commit; ++i) causal_pos[i] = P + i;
                L = backend_.eval_custom_pos(canvas.data(), k_commit, causal_pos.data(), true, params.seq_id);
            }

            result.total_blocks++;
            std::vector<int32_t> committed_this_block;
            bool should_stop = false;
            int committed_count = 0;

            for (int j = 0; j < k_commit; ++j) {
                int32_t tok = canvas[j];
                sampler_.accept(tok);
                result.tokens.push_back(tok);
                running_history.push_back(tok);
                if (tokenizer_.is_eog(tok)) {
                    stop_reason = StopReason::EOG;
                    should_stop = true;
                    break;
                }

                committed_this_block.push_back(tok);
                committed_count++;
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

            if (guesser_ && !committed_this_block.empty()) {
                guesser_->update_ngram(committed_this_block);
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
                if (last_pass_was_causal && committed_count < B_cur) {
                    backend_.kv_truncate(P + committed_count, params.seq_id);
                }
                break;
            }

            // Clear leftover canvas to ensure all lookahead slots come from verified ContextTrie history
            leftover_canvas.clear();

            // Advance position P by committed tokens and update prefix logits for next block from row k_commit - 1
            P += k_commit;
            const float* next_p_row = L.row(k_commit - 1);
            if (!next_p_row) next_p_row = backend_.get_logits_ith(k_commit - 1);
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
