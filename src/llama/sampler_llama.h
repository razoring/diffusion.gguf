#pragma once

#include "diffusion/engine.h"
#include "diffusion/sampler.h"
#include "llama.h"
#include <memory>
#include <vector>

namespace diffusion {

class SamplerLlama : public ISamplerChain {
public:
    SamplerLlama(llama_sampler* chain, const llama_vocab* vocab, llama_sampler* pos_sampler = nullptr);
    ~SamplerLlama() override;

    std::unique_ptr<ISamplerChain> clone() const override;
    void accept(int32_t tok) override;
    bool can_accept(int32_t tok) const override;
    int32_t sample(const float* logits, int abs_pos, float* out_prob = nullptr) override;

    llama_sampler* raw_chain() const { return chain_; }
    bool is_pure_greedy() const { return is_pure_greedy_; }
    void set_pure_greedy(bool val) { is_pure_greedy_ = val; }

    static std::unique_ptr<SamplerLlama> create(
        const llama_vocab* vocab,
        int32_t n_vocab,
        const SamplerParams& params
    );

private:
    llama_sampler* chain_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    llama_sampler* pos_sampler_ = nullptr;
    int32_t n_vocab_ = 0;
    bool is_pure_greedy_ = false;
    mutable std::vector<llama_token_data> cur_data_;
};

} // namespace diffusion
