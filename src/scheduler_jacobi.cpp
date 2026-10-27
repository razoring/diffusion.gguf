#include "scheduler_jacobi.h"
#include "kernels/reduce_logits.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace diffusion {

SchedulerJacobi::SchedulerJacobi(
    IModelBackend& backend,
    const ITokenizer& tokenizer,
    ISamplerChain& sampler,
    InitGuesser& guesser
) : backend_(backend),
    tokenizer_(tokenizer),
    sampler_(sampler),
    guesser_(guesser) {}

GenResult SchedulerJacobi::run(
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

    // 1. Prefill prompt (with optional KV prefix reuse)
    LogitsView prefill_logits;
    if (reuse_prefix_len == 0) {
        backend_.kv_clear(params.seq_id);
        prefill_logits = backend_.eval(effective_prompt.data(), prompt_len, 0, false, params.seq_id);
        for (int32_t tok : effective_prompt) {
            sampler_.accept(tok);
        }
        guesser_.update_ngram(effective_prompt);
    } else {
        backend_.kv_truncate(reuse_prefix_len, params.seq_id);
        int new_len = prompt_len - reuse_prefix_len;
        if (new_len > 0) {
            prefill_logits = backend_.eval(effective_prompt.data() + reuse_prefix_len, new_len, reuse_prefix_len, false, params.seq_id);
            std::vector<int32_t> new_prompt_slice(effective_prompt.begin() + reuse_prefix_len, effective_prompt.end());
            for (int32_t tok : new_prompt_slice) {
                sampler_.accept(tok);
            }
            guesser_.update_ngram(new_prompt_slice);
        } else {
            backend_.kv_truncate(prompt_len - 1, params.seq_id);
            prefill_logits = backend_.eval(effective_prompt.data() + prompt_len - 1, 1, prompt_len - 1, false, params.seq_id);
        }
    }

    // Ensure prompt context is fully indexed
    guesser_.update_ngram(effective_prompt);

    int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int B_target = (params.mode == DecodeMode::WHOLE) ? params.whole_max : params.block_size;
    if (!backend_.supports_kv_truncate() || B_target < 1) {
        B_target = 1;
    }

    int max_iters = params.max_iters > 0 ? params.max_iters : (n_predict + 8);
    int P = prompt_len;

    std::vector<int32_t> running_history = effective_prompt;
    running_history.reserve(effective_prompt.size() + n_predict + B_target + 32);

    // 2. Sample exact first token c[0]
    float p0_conf = 1.0f;
    int32_t first_tok = sampler_.sample(prefill_logits.row(0), P, &p0_conf);

    StopReason stop_reason = StopReason::NONE;

    if (B_target == 1) {
        int32_t current_token = first_tok;
        while (static_cast<int>(result.tokens.size()) < n_predict && result.iterations < max_iters) {
            result.iterations++;

            // Context shifting if context limit approached
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

            // 1. Parallel evaluation of 1 token (only single output logit row computed)
            LogitsView L = backend_.eval(&current_token, 1, P, false, params.seq_id);

            // 2. Sample next token from position P + 1
            float next_conf = 1.0f;
            int32_t next_tok = sampler_.sample(L.row(0), P + 1, &next_conf);

            // 3. Commit current token
            sampler_.accept(current_token);
            result.tokens.push_back(current_token);

            // Check EOG
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

            // Optional logprob collection
            if (params.logprobs && L.row(0)) {
                TokenLogprobs tlp;
                tlp.token = current_token;
                tlp.piece = tokenizer_.piece(current_token);
                const float* row = L.row(0);
                int n_vocab = backend_.n_vocab();
                float max_l = row[0];
                for (int v = 1; v < n_vocab; ++v) {
                    if (row[v] > max_l) max_l = row[v];
                }
                double sum_exp = 0.0;
                for (int v = 0; v < n_vocab; ++v) {
                    sum_exp += std::exp(row[v] - max_l);
                }
                float log_sum_exp = max_l + static_cast<float>(std::log(sum_exp));
                tlp.logprob = (current_token >= 0 && current_token < n_vocab) ? (row[current_token] - log_sum_exp) : -999.0f;
                result.token_logprobs.push_back(std::move(tlp));
            }

            // Check stop strings
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

            if (should_stop) {
                break;
            }

            // 4. Advance position and prepare next iteration
            P += 1;
            current_token = next_tok;
        }
    } else {
        // Multi-token Jacobi refinement loop
        int B_initial = std::min(B_target, n_predict);
        std::vector<int32_t> c;
        c.push_back(first_tok);

        running_history.push_back(first_tok);
        auto guesses = guesser_.make_guesses(B_initial - 1, running_history);
        running_history.pop_back();
        for (int32_t g : guesses) {
            c.push_back(g);
            if (static_cast<int>(c.size()) >= B_initial) break;
        }
        while (static_cast<int>(c.size()) < B_initial) {
            c.push_back(c.back());
        }

        while (!c.empty() && static_cast<int>(result.tokens.size()) < n_predict && result.iterations < max_iters) {
            result.iterations++;
            int B = static_cast<int>(c.size());

            if (P + B >= backend_.n_ctx()) {
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

            LogitsView L = backend_.eval(c.data(), B, P, (B > 1), params.seq_id);

            std::vector<kernels::LogitsReduction> reductions(B);
            kernels::reduce_logits_cpu(L.row(0), backend_.n_vocab(), B, reductions.data());

            std::vector<int32_t> p(B + 1);
            p[0] = c[0];
            std::vector<float> conf(B + 1, 0.0f);
            conf[0] = 1.0f;

            auto st = sampler_.clone();
            st->accept(c[0]);

            int k = 0;
            if (params.sampling.temp == 0.0f && !reductions.empty() && reductions[0].top_token >= 0) {
                p[1] = reductions[0].top_token;
                conf[1] = 1.0f;
            } else {
                p[1] = st->sample(L.row(0), P + 1, &conf[1]);
            }

            for (int i = 0; i + 1 < B; ++i) {
                bool matched = (c[i + 1] == p[i + 1]);
                if (!matched || !st->can_accept(c[i + 1])) {
                    break;
                }

                k++;
                st->accept(c[i + 1]);
                if (params.sampling.temp == 0.0f && i + 1 < static_cast<int>(reductions.size()) && reductions[i + 1].top_token >= 0) {
                    p[i + 2] = reductions[i + 1].top_token;
                    conf[i + 2] = 1.0f;
                } else {
                    p[i + 2] = st->sample(L.row(i + 1), P + i + 2, &conf[i + 2]);
                }
            }

            std::vector<int32_t> committed_this_step;
            bool should_stop = false;

            for (int j = 0; j <= k; ++j) {
                int32_t tok = c[j];
                sampler_.accept(tok);
                result.tokens.push_back(tok);

                if (tokenizer_.is_eog(tok)) {
                    stop_reason = StopReason::EOG;
                    should_stop = true;
                    break;
                }

                committed_this_step.push_back(tok);
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

            guesser_.update_ngram(committed_this_step);
            running_history.insert(running_history.end(), committed_this_step.begin(), committed_this_step.end());

            auto t_now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(t_now - t_start).count();
            double tok_per_s = elapsed_s > 0.0 ? (result.tokens.size() / elapsed_s) : 0.0;
            double tok_per_iter = static_cast<double>(result.tokens.size()) / result.iterations;

            int committed_count = static_cast<int>(committed_this_step.size());
            P += committed_count;
            if (committed_count < B) {
                backend_.kv_truncate(P, params.seq_id);
            }

            if (should_stop) {
                if (callback) {
                    StepEvent ev;
                    ev.step = result.iterations;
                    ev.committed_upto = static_cast<int>(result.tokens.size());
                    ev.committed_new = committed_this_step;
                    ev.canvas = {};
                    ev.diffs = {};
                    ev.done = true;
                    ev.reason = stop_reason;
                    ev.tok_per_s = tok_per_s;
                    ev.tok_per_iter = tok_per_iter;
                    callback(ev);
                }
                break;
            }

            int remaining_needed = B_target;
            if (remaining_needed <= 0) {
                stop_reason = StopReason::MAX_TOKENS;
                if (callback) {
                    StepEvent ev;
                    ev.step = result.iterations;
                    ev.committed_upto = static_cast<int>(result.tokens.size());
                    ev.committed_new = committed_this_step;
                    ev.canvas = {};
                    ev.diffs = {};
                    ev.done = true;
                    ev.reason = stop_reason;
                    ev.tok_per_s = tok_per_s;
                    ev.tok_per_iter = tok_per_iter;
                    callback(ev);
                }
                break;
            }

            std::vector<int32_t> next_c;
            if (k + 1 < static_cast<int>(p.size())) {
                next_c.push_back(p[k + 1]);
            } else {
                next_c.push_back(tokenizer_.pad_or_fallback());
            }

            for (int r = k + 1; r < B && static_cast<int>(next_c.size()) < remaining_needed; ++r) {
                if (r < static_cast<int>(reductions.size()) && reductions[r].top_token >= 0 &&
                    reductions[r].top_token < backend_.n_vocab()) {
                    next_c.push_back(reductions[r].top_token);
                }
            }

            int open_slots = remaining_needed - static_cast<int>(next_c.size());
            if (open_slots > 0) {
                running_history.push_back(next_c.back());
                auto probe = guesser_.make_guesses(open_slots, running_history);
                running_history.pop_back();

                for (int32_t g : probe) {
                    next_c.push_back(g);
                    if (static_cast<int>(next_c.size()) >= remaining_needed) break;
                }
            }

            while (static_cast<int>(next_c.size()) < remaining_needed) {
                next_c.push_back(next_c.back());
            }

            std::vector<TokenDiff> diffs;
            for (size_t j = 0; j < next_c.size(); ++j) {
                size_t old_idx = committed_count + j;
                if (old_idx >= c.size() || c[old_idx] != next_c[j]) {
                    diffs.push_back(TokenDiff{
                        static_cast<int>(old_idx),
                        old_idx < c.size() ? c[old_idx] : -1,
                        next_c[j],
                        "",
                        "",
                        1.0f
                    });
                }
            }

            if (callback) {
                StepEvent ev;
                ev.step = result.iterations;
                ev.committed_upto = static_cast<int>(result.tokens.size());
                ev.committed_new = committed_this_step;
                ev.canvas = next_c;
                ev.diffs = diffs;
                ev.done = false;
                ev.reason = StopReason::NONE;
                ev.tok_per_s = tok_per_s;
                ev.tok_per_iter = tok_per_iter;

                if (!callback(ev)) {
                    stop_reason = StopReason::CANCELLED;
                    break;
                }
            }

            c = std::move(next_c);
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

    return result;
}

} // namespace diffusion
