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

    int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int B_target = (params.block_size > 1) ? params.block_size : default_block_size_;
    if (!backend_.supports_kv_truncate() || B_target < 1) {
        B_target = 1;
    }
    int M_max = (params.max_iters > 0 && params.max_iters <= 16) ? params.max_iters : default_max_diffusion_steps_;
    int max_iters = (params.max_iters > 0 && params.max_iters > 16) ? params.max_iters : (n_predict + 16);

    backend_.set_sampler(params.seq_id, nullptr);

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
            for (int i = reuse_prefix_len; i < prompt_len; ++i) {
                sampler_.accept(effective_prompt[i]);
            }
        } else {
            backend_.kv_truncate(prompt_len - 1, params.seq_id);
            prefill_logits = backend_.eval(effective_prompt.data() + prompt_len - 1, 1, prompt_len - 1, false, params.seq_id);
        }
    }

    int P = prompt_len;
    StopReason stop_reason = StopReason::NONE;
    std::chrono::steady_clock::time_point t_start;

    std::vector<int32_t> running_history = effective_prompt;
    if (guesser_) {
        guesser_->update_ngram(effective_prompt);
    }

    int n_vocab = backend_.n_vocab();
    std::vector<float> prefix_logits(n_vocab, 0.0f);
    const float* prefill_row = prefill_logits.row(prefill_logits.n_rows - 1);
    if (prefill_row) {
        std::copy(prefill_row, prefill_row + n_vocab, prefix_logits.begin());
    }

    if (B_target == 1) {
        // Causal single-token decode path (host-logit evaluation)
        t_start = std::chrono::steady_clock::now();
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
            const float* r0 = L.row(0);
            float next_conf = 1.0f;
            int32_t next_tok = sampler_.sample(r0, P + 1, &next_conf);

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
        // Discrete Block Diffusion Inference Loop (Two-Stage Block-Parallel Diffusion)
        std::vector<int32_t> leftover_canvas;
        std::vector<float> frontier_prefix_logits(n_vocab, 0.0f);
        int32_t pending_x0 = -1;
        bool has_fresh_host_logits = (prefill_row != nullptr);
        bool can_gpu_sample = sampler_.is_pure_greedy() && !params.logprobs &&
                              backend_.set_sampler(params.seq_id, sampler_.raw_chain());
        bool gpu_sampler_active = can_gpu_sample;

        // Pre-warm graph buckets for W in {2, 3} to eliminate mid-generation graph allocation latency
        if (can_gpu_sample && B_target > 1) {
            int max_w_warmup = (B_target <= 8) ? 2 : 3;
            int32_t dummy_tok = tokenizer_.pad_or_fallback();
            std::vector<int32_t> dummy_canvas(max_w_warmup, dummy_tok);
            for (int w = 2; w <= max_w_warmup; ++w) {
                std::vector<int32_t> dummy_pos(w);
                for (int i = 0; i < w; ++i) dummy_pos[i] = P + i;
                backend_.eval_custom_pos(dummy_canvas.data(), w, dummy_pos.data(), true, params.seq_id);
                backend_.kv_truncate(P, params.seq_id);
            }
        }

        t_start = std::chrono::steady_clock::now();

        while (static_cast<int>(result.tokens.size()) < n_predict && result.iterations < max_iters) {
            int B_cur = std::min(B_target, n_predict - static_cast<int>(result.tokens.size()));
            if (B_cur <= 0) break;

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

            // Phase 1: Block-Boundary Model Prior
            float conf0 = 1.0f;
            int32_t x0 = (pending_x0 >= 0) ? pending_x0 : sampler_.sample(prefix_logits.data(), P, &conf0);
            pending_x0 = -1;

            if (params.multi_candidate_k == 2 && has_fresh_host_logits) {
                float max_l0 = prefix_logits[0];
                for (int v = 1; v < n_vocab; ++v) {
                    if (prefix_logits[v] > max_l0) max_l0 = prefix_logits[v];
                }
                double sum_exp0 = 0.0, sum_z_exp0 = 0.0;
                for (int v = 0; v < n_vocab; ++v) {
                    double z = static_cast<double>(prefix_logits[v] - max_l0);
                    double e = std::exp(z);
                    sum_exp0 += e;
                    sum_z_exp0 += z * e;
                }
                float H0 = 0.0f;
                if (sum_exp0 > 0.0) {
                    H0 = static_cast<float>(std::log(sum_exp0) - (sum_z_exp0 / sum_exp0));
                }
                bool should_branch = (H0 >= params.entropy_fork_threshold) &&
                                     (get_branch_frequency_100() < params.max_branch_ratio);
                record_branch_event(should_branch ? 1 : 0);
            }

            // Phase 2: Parallel Diffusion Across Block Slots [0 .. B_cur - 1]
            std::vector<int32_t> committed_this_block;
            committed_this_block.reserve(B_cur);
            std::vector<int32_t> last_canvas;
            bool should_stop = false;
            int committed_in_block = 0;

            while (committed_in_block < B_cur && result.iterations < max_iters && !should_stop) {
                int max_W = (B_target <= 8) ? 2 : 3;
                int rem = std::min(B_cur - committed_in_block, max_W);
                if (rem <= 0) break;

                std::vector<int32_t> canvas;
                canvas.reserve(rem);
                canvas.push_back(x0);

                for (size_t i = 0; i < leftover_canvas.size() && static_cast<int>(canvas.size()) < rem; ++i) {
                    canvas.push_back(leftover_canvas[i]);
                }
                leftover_canvas.clear();

                if (static_cast<int>(canvas.size()) < rem && guesser_) {
                    running_history.push_back(x0);
                    auto guesses = guesser_->make_guesses(rem - static_cast<int>(canvas.size()), running_history);
                    running_history.pop_back();
                    for (int32_t g : guesses) {
                        if (static_cast<int>(canvas.size()) >= rem) break;
                        canvas.push_back(g);
                    }
                }

                int W = static_cast<int>(canvas.size());
                last_canvas = canvas;

                if (W == 1) {
                    result.iterations++;
                    result.total_diffusion_passes++;
                    result.cumulative_active_work += 1;

                    LogitsView L1 = backend_.eval_custom_pos(&x0, 1, &P, false, params.seq_id);
                    int32_t next_tok = -1;
                    if (gpu_sampler_active) {
                        next_tok = backend_.get_sampled_token(-1);
                    } else {
                        const float* row = L1.row(0);
                        if (!row) row = backend_.get_logits_ith(-1);
                        if (row) {
                            float conf = 1.0f;
                            next_tok = sampler_.sample(row, P + 1, &conf);
                        }
                    }

                    sampler_.accept(x0);
                    result.tokens.push_back(x0);
                    running_history.push_back(x0);
                    committed_this_block.push_back(x0);
                    committed_in_block++;
                    result.text += tokenizer_.piece(x0);

                    if (tokenizer_.is_eog(x0)) {
                        stop_reason = StopReason::EOG;
                        should_stop = true;
                        break;
                    }
                    for (const auto& s : params.stop_strings) {
                        if (!s.empty() && result.text.size() >= s.size() &&
                            result.text.compare(result.text.size() - s.size(), s.size(), s) == 0) {
                            result.text.resize(result.text.size() - s.size());
                            stop_reason = StopReason::STOP_STRING;
                            should_stop = true;
                            break;
                        }
                    }
                    if (should_stop) break;
                    if (static_cast<int>(result.tokens.size()) >= n_predict) {
                        stop_reason = StopReason::MAX_TOKENS;
                        should_stop = true;
                        break;
                    }

                    P += 1;
                    x0 = next_tok;
                    continue;
                }

                result.iterations++;
                result.total_diffusion_passes++;
                result.cumulative_active_work += W;

                std::vector<int32_t> custom_pos(W);
                for (int i = 0; i < W; ++i) custom_pos[i] = P + i;

                LogitsView L1 = backend_.eval_custom_pos(canvas.data(), W, custom_pos.data(), true, params.seq_id);

                std::vector<int32_t> sampled(W, -1);
                if (gpu_sampler_active) {
                    for (int j = 0; j < W; ++j) {
                        sampled[j] = backend_.get_sampled_token(j);
                    }
                } else {
                    for (int j = 0; j < W; ++j) {
                        const float* row = L1.row(j);
                        if (!row) row = backend_.get_logits_ith(j);
                        if (row) {
                            float conf = 1.0f;
                            sampled[j] = sampler_.sample(row, P + j + 1, &conf);
                        }
                    }
                }

                int m = 0;
                while (m + 1 < W && sampled[m] >= 0) {
                    int32_t cand = canvas[m + 1];
                    int32_t top_tok = sampled[m];

                    if (cand == top_tok) {
                        m++;
                        continue;
                    }

                    break;
                }

                if (m < W - 1) {
                    backend_.kv_truncate(P + m + 1, params.seq_id);
                }

                std::vector<int32_t> tokens_to_commit;
                tokens_to_commit.reserve(m + 1);
                for (int j = 0; j <= m; ++j) {
                    tokens_to_commit.push_back(canvas[j]);
                }
                int32_t x_next = sampled[m];

                int step_committed = 0;
                for (int32_t tok : tokens_to_commit) {
                    sampler_.accept(tok);
                    result.tokens.push_back(tok);
                    running_history.push_back(tok);
                    committed_this_block.push_back(tok);
                    committed_in_block++;
                    step_committed++;
                    result.text += tokenizer_.piece(tok);

                    if (tokenizer_.is_eog(tok)) {
                        stop_reason = StopReason::EOG;
                        should_stop = true;
                        break;
                    }

                    for (const auto& s : params.stop_strings) {
                        if (!s.empty() && result.text.size() >= s.size() &&
                            result.text.compare(result.text.size() - s.size(), s.size(), s) == 0) {
                            result.text.resize(result.text.size() - s.size());
                            stop_reason = StopReason::STOP_STRING;
                            should_stop = true;
                            break;
                        }
                    }
                    if (should_stop) break;

                    if (static_cast<int>(result.tokens.size()) >= n_predict) {
                        stop_reason = StopReason::MAX_TOKENS;
                        should_stop = true;
                        break;
                    }
                }

                P += step_committed;
                x0 = x_next;

                if (step_committed < static_cast<int>(tokens_to_commit.size()) || should_stop) {
                    backend_.kv_truncate(P, params.seq_id);
                }
            }

            pending_x0 = x0;

            result.total_blocks++;

            if (guesser_ && !committed_this_block.empty()) {
                guesser_->update_ngram(committed_this_block);
            }

            if (callback) {
                StepEvent ev;
                ev.step = result.iterations;
                ev.committed_upto = static_cast<int>(result.tokens.size());
                ev.committed_new = committed_this_block;
                ev.canvas = last_canvas;
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
                break;
            }
        }

        if (gpu_sampler_active) {
            backend_.set_sampler(params.seq_id, nullptr);
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
