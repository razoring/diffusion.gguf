#include "scheduler_ar.h"
#include <chrono>

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
        backend_.kv_clear();
        logits = backend_.eval(prompt.data(), prompt_len, 0, false);
        for (int32_t tok : prompt) {
            sampler_.accept(tok);
        }
    } else {
        backend_.kv_truncate(reuse_prefix_len);
        int new_len = prompt_len - reuse_prefix_len;
        if (new_len > 0) {
            logits = backend_.eval(prompt.data() + reuse_prefix_len, new_len, reuse_prefix_len, false);
            for (int i = reuse_prefix_len; i < prompt_len; ++i) {
                sampler_.accept(prompt[i]);
            }
        } else {
            backend_.kv_truncate(prompt_len - 1);
            logits = backend_.eval(prompt.data() + prompt_len - 1, 1, prompt_len - 1, false);
        }
    }

    int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    int pos = prompt_len;
    StopReason stop_reason = StopReason::NONE;

    for (int i = 0; i < n_predict; ++i) {
        float prob = 0.0f;
        int32_t next_tok = sampler_.sample(logits.row(0), pos, &prob);

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

        // Eval next token
        logits = backend_.eval(&next_tok, 1, pos, false);
        pos++;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.tok_per_iter = result.iterations > 0 ? (static_cast<double>(result.tokens.size()) / result.iterations) : 0.0;
    result.tok_per_s = result.total_time_ms > 0.0 ? (result.tokens.size() / (result.total_time_ms / 1000.0)) : 0.0;
    result.stop_reason = stop_reason;

    return result;
}

} // namespace diffusion
