#pragma once

#include "diffusion/backend.h"
#include "diffusion/engine.h"
#include "diffusion/sampler.h"
#include "diffusion/tokenizer.h"
#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include "context_trie.h"

namespace diffusion {

class NgramPool {
public:
    NgramPool(int n = 3, int max_entries = 4096);

    void train(const std::vector<int32_t>& tokens);
    int32_t predict(const std::vector<int32_t>& context, int32_t fallback) const;

private:
    int n_ = 3;
    int max_entries_ = 4096;
    // Map prefix hash -> list of follow tokens
    std::unordered_map<uint64_t, std::vector<int32_t>> table_;

    uint64_t hash_prefix(const int32_t* toks, int len) const;
};

class InitGuesser {
public:
    InitGuesser(
        InitStrategy strategy,
        const ITokenizer& tokenizer,
        int32_t n_vocab,
        uint32_t seed,
        int32_t ngram_n = 3,
        int32_t ngram_pool_max = 4096,
        IModelBackend* draft_backend = nullptr,
        ISamplerChain* draft_sampler = nullptr
    );

    std::vector<int32_t> make_guesses(
        int count,
        const std::vector<int32_t>& context
    );

    void update_ngram(const std::vector<int32_t>& tokens);

private:
    InitStrategy strategy_;
    const ITokenizer& tokenizer_;
    int32_t n_vocab_;
    std::mt19937 rng_;
    NgramPool ngram_pool_;
    ContextTrie context_trie_;
    IModelBackend* draft_backend_ = nullptr;
    ISamplerChain* draft_sampler_ = nullptr;
};

} // namespace diffusion
