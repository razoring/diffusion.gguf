#include "engine.h"
#include "init_guess.h"
#include "scheduler_ar.h"
#include "scheduler_jacobi.h"
#include <algorithm>
#include <stdexcept>

namespace diffusion {

Engine::Engine(const BackendConfig& config) {
    backend_ = std::make_unique<BackendLlama>(config);
    tokenizer_ = std::make_unique<TokenizerLlama>(backend_->raw_vocab());
    chat_formatter_ = std::make_unique<ChatLlama>(backend_->raw_model());
}

Engine::~Engine() = default;

GenResult Engine::generate(
    const std::string& prompt,
    const GenParams& params,
    const StepCallback& callback
) {
    if (prompt.empty()) return GenResult{};
    std::vector<int32_t> prompt_tokens = tokenizer_->tokenize(prompt, true, true);
    return generate(prompt_tokens, params, 0, callback);
}

GenResult Engine::generate(
    const std::vector<int32_t>& prompt_tokens,
    const GenParams& params,
    int reuse_prefix_len,
    const StepCallback& callback
) {
    if (prompt_tokens.empty()) return GenResult{};

    // Handle prompt cache restoration
    std::vector<int32_t> session_tokens;
    if (!params.prompt_cache_path.empty()) {
        if (backend_->load_prompt_cache(params.prompt_cache_path, session_tokens)) {
            size_t match_len = 0;
            while (match_len < prompt_tokens.size() && match_len < session_tokens.size() &&
                   prompt_tokens[match_len] == session_tokens[match_len]) {
                match_len++;
            }
            reuse_prefix_len = std::max(reuse_prefix_len, static_cast<int>(match_len));
        }
    }

    auto sampler = SamplerLlama::create(
        backend_->raw_vocab(),
        backend_->n_vocab(),
        params.sampling
    );

    if (!sampler) {
        throw std::runtime_error("Failed to construct sampler chain");
    }

    std::unique_ptr<BackendLlama> draft_backend;
    std::unique_ptr<SamplerLlama> draft_sampler;
    InitStrategy strat = params.init_strategy;

    if (!params.model_draft.empty()) {
        try {
            BackendConfig dconfig;
            dconfig.model_path = params.model_draft;
            dconfig.n_gpu_layers = params.draft_n_gpu_layers;
            dconfig.n_ctx = params.n_predict + 512;
            draft_backend = std::make_unique<BackendLlama>(dconfig);
            draft_sampler = SamplerLlama::create(
                draft_backend->raw_vocab(),
                draft_backend->n_vocab(),
                params.sampling
            );
            strat = InitStrategy::DRAFT;
        } catch (...) {
            // Gracefully fallback to n-gram if draft fails to load
        }
    }

    InitGuesser guesser(
        strat,
        *tokenizer_,
        backend_->n_vocab(),
        params.sampling.seed,
        params.ngram_n,
        params.ngram_pool_max,
        draft_backend.get(),
        draft_sampler.get()
    );

    GenResult result;
    if (params.mode == DecodeMode::AR) {
        SchedulerAR scheduler(*backend_, *tokenizer_, *sampler);
        result = scheduler.run(prompt_tokens, params, reuse_prefix_len, callback);
    } else {
        SchedulerJacobi scheduler(*backend_, *tokenizer_, *sampler, guesser);
        result = scheduler.run(prompt_tokens, params, reuse_prefix_len, callback);
    }

    // Save prompt cache if requested
    if (!params.prompt_cache_path.empty() && !params.prompt_cache_ro) {
        if (params.prompt_cache_all) {
            std::vector<int32_t> all_tokens = prompt_tokens;
            all_tokens.insert(all_tokens.end(), result.tokens.begin(), result.tokens.end());
            backend_->save_prompt_cache(params.prompt_cache_path, all_tokens);
        } else {
            backend_->save_prompt_cache(params.prompt_cache_path, prompt_tokens);
        }
    }

    // Extract DeepSeek reasoning tags (<think> ... </think>)
    if (params.extract_reasoning) {
        const std::string think_start = "<think>";
        const std::string think_end = "</think>";
        size_t s_pos = result.text.find(think_start);
        if (s_pos != std::string::npos) {
            size_t content_start = s_pos + think_start.size();
            size_t e_pos = result.text.find(think_end, content_start);
            if (e_pos != std::string::npos) {
                result.reasoning_content = result.text.substr(content_start, e_pos - content_start);
                std::string before = result.text.substr(0, s_pos);
                std::string after = result.text.substr(e_pos + think_end.size());
                size_t first_non_ws = after.find_first_not_of(" \t\r\n");
                if (first_non_ws != std::string::npos) {
                    after = after.substr(first_non_ws);
                }
                result.text = before + after;
            } else {
                result.reasoning_content = result.text.substr(content_start);
                result.text = result.text.substr(0, s_pos);
            }
        }
    }

    return result;
}

} // namespace diffusion
