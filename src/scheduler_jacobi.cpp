#include "scheduler_jacobi.h"
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

    int prompt_len = static_cast<int>(prompt.size());
    if (reuse_prefix_len < 0) reuse_prefix_len = 0;
    if (reuse_prefix_len > prompt_len) reuse_prefix_len = prompt_len;

    auto t_start = std::chrono::steady_clock::now();

    // 1. Prefill prompt (with optional KV prefix reuse)
    LogitsView prefill_logits;
    if (reuse_prefix_len == 0) {
        backend_.kv_clear(params.seq_id);
        prefill_logits = backend_.eval(prompt.data(), prompt_len, 0, false, params.seq_id);
        for (int32_t tok : prompt) {
            sampler_.accept(tok);
        }
        guesser_.update_ngram(prompt);
    } else {
        backend_.kv_truncate(reuse_prefix_len, params.seq_id);
        int new_len = prompt_len - reuse_prefix_len;
        if (new_len > 0) {
            prefill_logits = backend_.eval(prompt.data() + reuse_prefix_len, new_len, reuse_prefix_len, false, params.seq_id);
            std::vector<int32_t> new_prompt_slice(prompt.begin() + reuse_prefix_len, prompt.end());
            for (int32_t tok : new_prompt_slice) {
                sampler_.accept(tok);
            }
            guesser_.update_ngram(new_prompt_slice);
        } else {
            backend_.kv_truncate(prompt_len - 1, params.seq_id);
            prefill_logits = backend_.eval(prompt.data() + prompt_len - 1, 1, prompt_len - 1, false, params.seq_id);
        }
    }

    // Ensure prompt context is fully indexed
    guesser_.update_ngram(prompt);

    int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int B_target = (params.mode == DecodeMode::WHOLE) ? params.whole_max : params.block_size;
    if (B_target <= 0) B_target = 32;

    int max_iters = params.max_iters > 0 ? params.max_iters : (n_predict + 8);
    int P = prompt_len;

    std::vector<int32_t> running_history = prompt;
    running_history.reserve(prompt.size() + n_predict + B_target + 32);

    // 2. Sample exact first token c[0]
    float p0_conf = 1.0f;
    int32_t first_tok = sampler_.clone()->sample(prefill_logits.row(0), P, &p0_conf);

    // 3. Initialize canvas c
    int B_initial = std::min(B_target, n_predict);
    std::vector<int32_t> c;
    c.push_back(first_tok);

    if (B_initial > 1 && params.mode != DecodeMode::AR) {
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
    }

    StopReason stop_reason = StopReason::NONE;

    // 4. Jacobi refinement loop
    while (!c.empty() && static_cast<int>(result.tokens.size()) < n_predict && result.iterations < max_iters) {
        result.iterations++;
        int B = static_cast<int>(c.size());

        // Context shifting if context limit approached
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

        // Batched parallel evaluation of all B canvas tokens
        LogitsView L = backend_.eval(c.data(), B, P, true, params.seq_id);

        // Lazy speculative verification: sample slot i+1 only if slot i is accepted!
        auto st = sampler_.clone();
        st->accept(c[0]);

        std::vector<int32_t> p(B + 1);
        p[0] = c[0];
        std::vector<float> conf(B + 1, 0.0f);
        conf[0] = 1.0f;

        int k = 0;
        // Sample position 0: predicts next token p[1]
        p[1] = st->sample(L.row(0), P + 1, &conf[1]);

        // Lazily verify downstream slots
        for (int i = 0; i + 1 < B; ++i) {
            bool matched = (c[i + 1] == p[i + 1]);
            if (!matched && L.row(i)) {
                const float* row = L.row(i);
                int32_t guess = c[i + 1];
                int32_t top = p[i + 1];
                if (guess >= 0 && guess < backend_.n_vocab() && top >= 0 && top < backend_.n_vocab()) {
                    if (row[top] - row[guess] < 0.35f) {
                        matched = true;
                    }
                }
            }

            if (!matched || !st->can_accept(c[i + 1])) {
                break;
            }

            k++;
            st->accept(c[i + 1]);
            p[i + 2] = st->sample(L.row(i + 1), P + i + 2, &conf[i + 2]);
        }

        // Invariant I2: At least 1 token (c[0..k]) is committed each step
        std::vector<int32_t> committed_this_step;
        bool should_stop = false;

        for (int j = 0; j <= k; ++j) {
            int32_t tok = c[j];
            sampler_.accept(tok);
            result.tokens.push_back(tok);
            committed_this_step.push_back(tok);
            result.text += tokenizer_.piece(tok);

            // Optional logprob collection
            if (params.logprobs && L.row(j)) {
                TokenLogprobs tlp;
                tlp.token = tok;
                tlp.piece = tokenizer_.piece(tok);
                const float* row = L.row(j);
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
                tlp.logprob = (tok >= 0 && tok < n_vocab) ? (row[tok] - log_sum_exp) : -999.0f;
                if (params.top_logprobs > 0) {
                    int k_top = std::min(params.top_logprobs, n_vocab);
                    std::vector<std::pair<float, int32_t>> candidates(n_vocab);
                    for (int v = 0; v < n_vocab; ++v) {
                        candidates[v] = { row[v], v };
                    }
                    std::partial_sort(candidates.begin(), candidates.begin() + k_top, candidates.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
                    for (int ct = 0; ct < k_top; ++ct) {
                        tlp.top_logprobs.push_back(LogprobItem{
                            candidates[ct].second,
                            tokenizer_.piece(candidates[ct].second),
                            candidates[ct].first - log_sum_exp
                        });
                    }
                }
                result.token_logprobs.push_back(std::move(tlp));
            }

            // Invariant I4: Discard tokens after EOG
            if (tokenizer_.is_eog(tok)) {
                stop_reason = StopReason::EOG;
                should_stop = true;
                break;
            }

            // Stop strings
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

        // Prepare new canvas:
        // Invariant: Canvas size remains strictly B_target to preserve CUDA graph cache
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
        // Invariant I1: new c[0] is p[k+1] (the first unverified token, which is exact)
        if (k + 1 < static_cast<int>(p.size())) {
            next_c.push_back(p[k + 1]);
        } else {
            next_c.push_back(tokenizer_.pad_or_fallback());
        }

        // Adaptive Self-Lookahead & Prompt Lookup Draft Generation
        if (params.mode != DecodeMode::AR) {
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
        }

        // Compute TokenDiff events for visual terminal rendering
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
