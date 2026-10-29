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
    NgramPool(int n = 3, int max_entries = 16384);

    void train(const std::vector<int32_t>& tokens);
    int32_t predict(const std::vector<int32_t>& context, int32_t fallback) const;

private:
    struct Transition {
        int32_t best_tok = -1;
        int32_t best_count = 0;
        std::unordered_map<int32_t, int32_t> counts;
    };

    int n_ = 3;
    int max_entries_ = 16384;
    std::unordered_map<uint64_t, Transition> table_;

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
        int32_t ngram_pool_max = 4096
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
    std::unordered_map<uint64_t, int32_t> stem_next_;
};

} // namespace diffusion
