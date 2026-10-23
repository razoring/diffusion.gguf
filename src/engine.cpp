#include "engine.h"
#include "init_guess.h"
#include "scheduler_ar.h"
#include "scheduler_jacobi.h"
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

    if (params.mode == DecodeMode::AR) {
        SchedulerAR scheduler(*backend_, *tokenizer_, *sampler);
        return scheduler.run(prompt_tokens, params, reuse_prefix_len, callback);
    } else {
        SchedulerJacobi scheduler(*backend_, *tokenizer_, *sampler, guesser);
        return scheduler.run(prompt_tokens, params, reuse_prefix_len, callback);
    }
}


} // namespace diffusion
