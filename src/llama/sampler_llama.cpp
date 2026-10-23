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
    if (is_pure_greedy_) {
        auto result = std::make_unique<SamplerLlama>(nullptr, vocab_, nullptr);
        result->set_pure_greedy(true);
        return result;
    }
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
    auto result = std::make_unique<SamplerLlama>(cloned, vocab_, cloned_pos);
    result->set_pure_greedy(is_pure_greedy_);
    return result;
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

#if defined(__AVX2__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>

static inline int32_t fast_argmax_simd(const float* data, int n) {
    if (n <= 0) return 0;
    int i = 0;
    __m256 max_val = _mm256_set1_ps(-1e30f);
    __m256i best_idx = _mm256_setzero_si256();
    __m256i step = _mm256_set1_epi32(8);
    __m256i cur_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 mask = _mm256_cmp_ps(v, max_val, _CMP_GT_OQ);
        max_val = _mm256_blendv_ps(max_val, v, mask);
        best_idx = _mm256_blendv_epi8(best_idx, cur_idx, _mm256_castps_si256(mask));
        cur_idx = _mm256_add_epi32(cur_idx, step);
    }

    alignas(32) float vals[8];
    alignas(32) int32_t idxs[8];
    _mm256_storeu_ps(vals, max_val);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(idxs), best_idx);

    float global_max = vals[0];
    int32_t global_best = idxs[0];
    for (int j = 1; j < 8; ++j) {
        if (vals[j] > global_max) {
            global_max = vals[j];
            global_best = idxs[j];
        }
    }

    for (; i < n; ++i) {
        if (data[i] > global_max) {
            global_max = data[i];
            global_best = i;
        }
    }
    return global_best;
}
#endif

int32_t SamplerLlama::sample(const float* logits, int abs_pos, float* out_prob) {
    if (!logits) return 0;

    // Fast-path: SIMD vectorized argmax for pure greedy decoding
    if (is_pure_greedy_ && n_vocab_ > 0) {
#if defined(__AVX2__) || defined(_M_AMD64) || defined(_M_X64)
        int32_t best_id = fast_argmax_simd(logits, n_vocab_);
#else
        int32_t best_id = 0;
        float max_val = logits[0];
        for (int32_t i = 1; i < n_vocab_; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                best_id = i;
            }
        }
#endif
        if (out_prob) *out_prob = 1.0f;
        return best_id;
    }

    if (!chain_) return 0;

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

    // 0. Logit bias
    if (!params.logit_bias.empty()) {
        std::vector<llama_logit_bias> biases;
        biases.reserve(params.logit_bias.size());
        for (const auto& b : params.logit_bias) {
            biases.push_back(llama_logit_bias{b.first, b.second});
        }
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(
            n_vocab,
            static_cast<int32_t>(biases.size()),
            biases.data()
        ));
    }

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

    // 7b. Top-n-sigma
    if (params.top_n_sigma > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(params.top_n_sigma));
    }

    // 7c. Infill
    if (params.infill) {
        llama_sampler_chain_add(chain, llama_sampler_init_infill(vocab));
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
    } else if (params.adaptive_p_target >= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_adaptive_p(
            params.adaptive_p_target,
            params.adaptive_p_decay,
            params.seed
        ));
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
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed));
    }

    bool is_pure_greedy = (params.temp <= 0.0f) &&
                           params.grammar.empty() &&
                           params.logit_bias.empty() &&
                           (params.repeat_penalty == 1.0f) &&
                           (params.presence_penalty == 0.0f) &&
                           (params.frequency_penalty == 0.0f) &&
                           (params.dry_multiplier <= 0.0f);

    auto result = std::make_unique<SamplerLlama>(chain, vocab, pos_sampler);
    result->set_pure_greedy(is_pure_greedy);
    return result;
}

} // namespace diffusion
