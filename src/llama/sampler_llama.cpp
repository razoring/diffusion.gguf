#include "sampler_llama.h"
#include "dist_positional.h"
#include <cmath>
#include <stdexcept>

namespace diffusion {

SamplerLlama::SamplerLlama(llama_sampler* chain, const llama_vocab* vocab, llama_sampler* pos_sampler)
    : chain_(chain), vocab_(vocab), pos_sampler_(pos_sampler) {
    if (vocab_) {
        n_vocab_ = llama_vocab_n_tokens(vocab_);
        cur_data_.resize(n_vocab_);
    }
}

SamplerLlama::~SamplerLlama() {
    if (chain_) {
        llama_sampler_free(chain_);
        chain_ = nullptr;
    }
}

std::unique_ptr<ISamplerChain> SamplerLlama::clone() const {
    if (!chain_) return nullptr;
    llama_sampler* cloned = llama_sampler_clone(chain_);
    // Locate positional sampler within cloned chain if present
    llama_sampler* cloned_pos = nullptr;
    if (pos_sampler_) {
        int n = llama_sampler_chain_n(cloned);
        for (int i = 0; i < n; ++i) {
            llama_sampler* s = llama_sampler_chain_get(cloned, i);
            const char* name = llama_sampler_name(s);
            if (name && std::string_view(name) == "dist_positional") {
                cloned_pos = s;
                break;
            }
        }
    }
    return std::make_unique<SamplerLlama>(cloned, vocab_, cloned_pos);
}

void SamplerLlama::accept(int32_t tok) {
    if (chain_) {
        llama_sampler_accept(chain_, tok);
    }
}

bool SamplerLlama::can_accept(int32_t /*tok*/) const {
    // True by default; grammar filtering is evaluated during apply/sampling
    return true;
}

int32_t SamplerLlama::sample(const float* logits, int abs_pos, float* out_prob) {
    if (!chain_ || !logits) return 0;

    if (pos_sampler_) {
        llama_sampler_dist_positional_set_pos(pos_sampler_, abs_pos);
    }

    if (cur_data_.size() < static_cast<size_t>(n_vocab_)) {
        cur_data_.resize(n_vocab_);
    }

    for (int32_t i = 0; i < n_vocab_; ++i) {
        cur_data_[i] = llama_token_data{i, logits[i], 0.0f};
    }

    llama_token_data_array cur_p = {
        cur_data_.data(),
        static_cast<size_t>(n_vocab_),
        -1,
        false
    };

    llama_sampler_apply(chain_, &cur_p);

    int32_t sel = cur_p.selected;
    if (sel < 0 || sel >= static_cast<int32_t>(cur_p.size)) {
        sel = 0;
    }

    if (out_prob) {
        *out_prob = cur_p.data[sel].p;
    }

    return cur_p.data[sel].id;
}

std::unique_ptr<SamplerLlama> SamplerLlama::create(
    const llama_vocab* vocab,
    int32_t n_vocab,
    const SamplerParams& params
) {
    llama_sampler_chain_params cparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(cparams);
    llama_sampler* pos_sampler = nullptr;

    // 1. Grammar
    if (!params.grammar.empty()) {
        auto* smpl_grammar = llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
        if (smpl_grammar) {
            llama_sampler_chain_add(chain, smpl_grammar);
        }
    }

    // 2. Penalties
    if (params.repeat_penalty != 1.0f || params.presence_penalty != 0.0f || params.frequency_penalty != 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            params.repeat_last_n,
            params.repeat_penalty,
            params.frequency_penalty,
            params.presence_penalty
        ));
    }

    // 3. DRY
    if (params.dry_multiplier > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_dry(
            vocab,
            2048,
            params.dry_multiplier,
            params.dry_base,
            params.dry_allowed_length,
            params.dry_penalty_last_n,
            nullptr,
            0
        ));
    }

    // 4. Top-K
    if (params.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
    }

    // 5. Typical
    if (params.typical_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_typical(params.typical_p, 1));
    }

    // 6. Top-P
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1));
    }

    // 7. Min-P
    if (params.min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(params.min_p, 1));
    }

    // 8. XTC
    if (params.xtc_probability > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_xtc(
            params.xtc_probability,
            params.xtc_threshold,
            1,
            params.seed
        ));
    }

    // 9. Temperature
    if (params.temp > 0.0f) {
        if (params.dynatemp_range > 0.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(
                params.temp,
                params.dynatemp_range,
                params.dynatemp_exponent
            ));
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temp));
        }
    }

    // 10. Terminal sampler
    if (params.temp <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else if (params.mirostat == 1) {
        llama_sampler_chain_add(chain, llama_sampler_init_mirostat(
            n_vocab,
            params.seed,
            params.mirostat_tau,
            params.mirostat_eta,
            100
        ));
    } else if (params.mirostat == 2) {
        llama_sampler_chain_add(chain, llama_sampler_init_mirostat_v2(
            params.seed,
            params.mirostat_tau,
            params.mirostat_eta
        ));
    } else if (params.rng_mode == RngMode::POSITIONAL) {
        pos_sampler = llama_sampler_init_dist_positional(params.seed);
        llama_sampler_chain_add(chain, pos_sampler);
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed));
    }

    return std::make_unique<SamplerLlama>(chain, vocab, pos_sampler);
}

} // namespace diffusion
