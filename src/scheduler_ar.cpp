#include "scheduler_ar.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace diffusion {

SchedulerAR::SchedulerAR(IModelBackend& backend, const ITokenizer& tokenizer, ISamplerChain& sampler)
    : backend_(backend), tokenizer_(tokenizer), sampler_(sampler) {}

GenResult SchedulerAR::run(
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
    LogitsView logits;
    if (reuse_prefix_len == 0) {
        backend_.kv_clear(params.seq_id);
        logits = backend_.eval(prompt.data(), prompt_len, 0, false, params.seq_id);
        for (int32_t tok : prompt) {
            sampler_.accept(tok);
        }
    } else {
        backend_.kv_truncate(reuse_prefix_len, params.seq_id);
        int new_len = prompt_len - reuse_prefix_len;
        if (new_len > 0) {
            logits = backend_.eval(prompt.data() + reuse_prefix_len, new_len, reuse_prefix_len, false, params.seq_id);
            for (int i = reuse_prefix_len; i < prompt_len; ++i) {
                sampler_.accept(prompt[i]);
            }
        } else {
            backend_.kv_truncate(prompt_len - 1, params.seq_id);
            logits = backend_.eval(prompt.data() + prompt_len - 1, 1, prompt_len - 1, false, params.seq_id);
        }
    }

    int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int pos = prompt_len;
    StopReason stop_reason = StopReason::NONE;

    for (int i = 0; i < n_predict; ++i) {
        // Context shifting if context limit reached
        if (pos >= backend_.n_ctx()) {
            if (params.ctx_shift && backend_.supports_kv_shift()) {
                int n_keep = params.n_keep > 0 ? params.n_keep : 1;
                int n_discard = params.n_discard > 0 ? params.n_discard : std::max(1, (backend_.n_ctx() - n_keep) / 2);
                if (n_discard > 0 && pos - n_discard >= n_keep) {
                    backend_.kv_truncate(n_keep + n_discard, params.seq_id);
                    backend_.kv_shift(params.seq_id, n_keep + n_discard, pos, -n_discard);
                    pos -= n_discard;
                } else {
                    stop_reason = StopReason::MAX_TOKENS;
                    break;
                }
            } else {
                stop_reason = StopReason::MAX_TOKENS;
                break;
            }
        }

        float prob = 0.0f;
        int32_t next_tok = sampler_.sample(logits.row(0), pos, &prob);

        // Optional logprobs calculation
        if (params.logprobs && logits.row(0)) {
            TokenLogprobs tlp;
            tlp.token = next_tok;
            tlp.piece = tokenizer_.piece(next_tok);
            const float* row = logits.row(0);
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
            tlp.logprob = (next_tok >= 0 && next_tok < n_vocab) ? (row[next_tok] - log_sum_exp) : -999.0f;
            if (params.top_logprobs > 0) {
                int k = std::min(params.top_logprobs, n_vocab);
                std::vector<std::pair<float, int32_t>> candidates(n_vocab);
                for (int v = 0; v < n_vocab; ++v) {
                    candidates[v] = { row[v], v };
                }
                std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                for (int c = 0; c < k; ++c) {
                    tlp.top_logprobs.push_back(LogprobItem{
                        candidates[c].second,
                        tokenizer_.piece(candidates[c].second),
                        candidates[c].first - log_sum_exp
                    });
                }
            }
            result.token_logprobs.push_back(std::move(tlp));
        }

        if (tokenizer_.is_eog(next_tok)) {
            stop_reason = StopReason::EOG;
        }

        sampler_.accept(next_tok);
        result.tokens.push_back(next_tok);
        std::string piece = tokenizer_.piece(next_tok);
        result.text += piece;

        // Check stop strings
        if (stop_reason == StopReason::NONE) {
            for (const auto& s : params.stop_strings) {
                if (!s.empty() && result.text.size() >= s.size()) {
                    if (result.text.compare(result.text.size() - s.size(), s.size(), s) == 0) {
                        result.text.resize(result.text.size() - s.size());
                        stop_reason = StopReason::STOP_STRING;
                        break;
                    }
                }
            }
        }

        if (i == n_predict - 1 && stop_reason == StopReason::NONE) {
            stop_reason = StopReason::MAX_TOKENS;
        }

        bool done = (stop_reason != StopReason::NONE);
        result.iterations++;

        auto t_now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_now - t_start).count();
        double tok_per_s = elapsed_s > 0.0 ? (result.tokens.size() / elapsed_s) : 0.0;

        if (callback) {
            StepEvent ev;
            ev.step = i + 1;
            ev.committed_upto = static_cast<int>(result.tokens.size());
            ev.committed_new = {next_tok};
            ev.done = done;
            ev.reason = stop_reason;
            ev.tok_per_s = tok_per_s;
            ev.tok_per_iter = 1.0;

            if (!callback(ev)) {
                stop_reason = StopReason::CANCELLED;
                done = true;
            }
        }

        if (done) break;

        // Evaluate next single token
        logits = backend_.eval(&next_tok, 1, pos, false, params.seq_id);
        pos++;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.tok_per_iter = 1.0;
    result.tok_per_s = result.total_time_ms > 0.0 ? (result.tokens.size() * 1000.0 / result.total_time_ms) : 0.0;
    result.stop_reason = stop_reason;

    return result;
}

} // namespace diffusion
