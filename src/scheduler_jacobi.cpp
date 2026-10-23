#include "scheduler_jacobi.h"
#include <algorithm>
#include <chrono>

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
        backend_.kv_clear();
        prefill_logits = backend_.eval(prompt.data(), prompt_len, 0, false);
        for (int32_t tok : prompt) {
            sampler_.accept(tok);
        }
        guesser_.update_ngram(prompt);
    } else {
        backend_.kv_truncate(reuse_prefix_len);
        int new_len = prompt_len - reuse_prefix_len;
        if (new_len > 0) {
            prefill_logits = backend_.eval(prompt.data() + reuse_prefix_len, new_len, reuse_prefix_len, false);
            std::vector<int32_t> new_prompt_slice(prompt.begin() + reuse_prefix_len, prompt.end());
            for (int32_t tok : new_prompt_slice) {
                sampler_.accept(tok);
            }
            guesser_.update_ngram(new_prompt_slice);
        } else {
            backend_.kv_truncate(prompt_len - 1);
            prefill_logits = backend_.eval(prompt.data() + prompt_len - 1, 1, prompt_len - 1, false);
        }
    }

    int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int B_target = (params.mode == DecodeMode::WHOLE) ? params.whole_max : params.block_size;
    if (B_target <= 0) B_target = 32;

    int max_iters = params.max_iters > 0 ? params.max_iters : (n_predict + 8);
    int P = prompt_len;

    // 2. Sample exact first token c[0]
    float p0_conf = 1.0f;
    int32_t first_tok = sampler_.clone()->sample(prefill_logits.row(0), P, &p0_conf);

    // 3. Initialize canvas c
    int B_initial = std::min(B_target, n_predict);
    std::vector<int32_t> c;
    c.push_back(first_tok);

    if (B_initial > 1) {
        std::vector<int32_t> running_ctx = prompt;
        running_ctx.push_back(first_tok);
        auto guesses = guesser_.make_guesses(B_initial - 1, running_ctx);
        c.insert(c.end(), guesses.begin(), guesses.end());
    }

    StopReason stop_reason = StopReason::NONE;

    // 4. Jacobi refinement loop
    while (!c.empty() && static_cast<int>(result.tokens.size()) < n_predict && result.iterations < max_iters) {
        result.iterations++;
        int B = static_cast<int>(c.size());

        // Invariant I3: Truncate KV for uncommitted positions >= P
        backend_.kv_truncate(P);

        // Batched parallel evaluation of all B canvas tokens
        LogitsView L = backend_.eval(c.data(), B, P, true);

        // Walk hypothesized prefix sequentially on CPU to obtain predictions p
        auto st = sampler_.clone();
        st->accept(c[0]);

        std::vector<int32_t> p(B + 1);
        p[0] = c[0];
        std::vector<float> conf(B + 1, 0.0f);
        conf[0] = 1.0f;

        int valid = B;
        for (int i = 0; i < B; ++i) {
            p[i + 1] = st->sample(L.row(i), P + i + 1, &conf[i + 1]);
            if (i + 1 < B) {
                if (!st->can_accept(c[i + 1])) {
                    valid = i + 1;
                    break;
                }
                st->accept(c[i + 1]);
            }
        }

        // Longest verified prefix k where c matches p
        int k = 0;
        while (k + 1 < valid && c[k + 1] == p[k + 1]) {
            k++;
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

        auto t_now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_now - t_start).count();
        double tok_per_s = elapsed_s > 0.0 ? (result.tokens.size() / elapsed_s) : 0.0;
        double tok_per_iter = static_cast<double>(result.tokens.size()) / result.iterations;

        int committed_count = static_cast<int>(committed_this_step.size());
        P += committed_count;
        backend_.kv_truncate(P);

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
        // Invariant I1: new c[0] is p[k+1] (the first unverified token, which is exact)
        int remaining_needed = std::min(B_target, n_predict - static_cast<int>(result.tokens.size()));
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

        // Retain unverified speculative tokens from p[k+2 .. valid]
        for (int idx = k + 2; idx < valid && static_cast<int>(next_c.size()) < remaining_needed; ++idx) {
            next_c.push_back(p[idx]);
        }

        // Fill remaining canvas positions with new initial guesses
        int fill_needed = remaining_needed - static_cast<int>(next_c.size());
        if (fill_needed > 0) {
            std::vector<int32_t> running_ctx = prompt;
            running_ctx.insert(running_ctx.end(), result.tokens.begin(), result.tokens.end());
            running_ctx.insert(running_ctx.end(), next_c.begin(), next_c.end());
            auto fill_guesses = guesser_.make_guesses(fill_needed, running_ctx);
            next_c.insert(next_c.end(), fill_guesses.begin(), fill_guesses.end());
        }

        // Compute TokenDiff events for visual terminal shimmering and callbacks
        std::vector<TokenDiff> diffs;
        for (size_t j = 0; j < next_c.size(); ++j) {
            size_t old_idx = committed_count + j;
            if (old_idx >= c.size() || c[old_idx] != next_c[j]) {
                diffs.push_back(TokenDiff{
                    static_cast<int>(j),
                    (old_idx < c.size()) ? c[old_idx] : -1,
                    next_c[j],
                    (old_idx < c.size()) ? tokenizer_.piece(c[old_idx]) : "",
                    tokenizer_.piece(next_c[j]),
                    (j < conf.size()) ? conf[j] : 0.0f
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
                should_stop = true;
            }
        }

        if (should_stop) break;

        c = std::move(next_c);
    }


    if (stop_reason == StopReason::NONE && static_cast<int>(result.tokens.size()) >= n_predict) {
        stop_reason = StopReason::MAX_TOKENS;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.tok_per_iter = result.iterations > 0 ? (static_cast<double>(result.tokens.size()) / result.iterations) : 0.0;
    result.tok_per_s = result.total_time_ms > 0.0 ? (result.tokens.size() / (result.total_time_ms / 1000.0)) : 0.0;
    result.stop_reason = stop_reason;

    return result;
}

} // namespace diffusion
