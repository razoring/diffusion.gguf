#include "init_guess.h"
#include <algorithm>

namespace diffusion {

NgramPool::NgramPool(int n, int max_entries)
    : n_(std::max(2, n)), max_entries_(std::max(64, max_entries)) {}

uint64_t NgramPool::hash_prefix(const int32_t* toks, int len) const {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(toks[i]);
        h *= 0x100000001b3ULL;
    }
    return h;
}

void NgramPool::train(const std::vector<int32_t>& tokens) {
    if (static_cast<int>(tokens.size()) < n_) return;

    for (size_t i = 0; i + n_ <= tokens.size(); ++i) {
        if (static_cast<int>(table_.size()) >= max_entries_) break;
        uint64_t h = hash_prefix(&tokens[i], n_ - 1);
        int32_t next_tok = tokens[i + n_ - 1];
        auto& list = table_[h];
        if (std::find(list.begin(), list.end(), next_tok) == list.end()) {
            list.push_back(next_tok);
        }
    }
}

int32_t NgramPool::predict(const std::vector<int32_t>& context, int32_t fallback) const {
    int prefix_len = n_ - 1;
    if (static_cast<int>(context.size()) < prefix_len) return fallback;

    const int32_t* ptr = context.data() + (context.size() - prefix_len);
    uint64_t h = hash_prefix(ptr, prefix_len);
    auto it = table_.find(h);
    if (it != table_.end() && !it->second.empty()) {
        return it->second.back();
    }
    return fallback;
}

InitGuesser::InitGuesser(
    InitStrategy strategy,
    const ITokenizer& tokenizer,
    int32_t n_vocab,
    uint32_t seed,
    int32_t ngram_n,
    int32_t ngram_pool_max
) : strategy_(strategy),
    tokenizer_(tokenizer),
    n_vocab_(n_vocab),
    rng_(seed),
    ngram_pool_(ngram_n, ngram_pool_max) {}

void InitGuesser::update_ngram(const std::vector<int32_t>& tokens) {
    ngram_pool_.train(tokens);
}

std::vector<int32_t> InitGuesser::make_guesses(int count, const std::vector<int32_t>& context) {
    std::vector<int32_t> guesses;
    if (count <= 0) return guesses;
    guesses.reserve(count);

    int32_t fallback_pad = tokenizer_.pad_or_fallback();
    std::vector<int32_t> running_ctx = context;

    for (int i = 0; i < count; ++i) {
        int32_t guess = fallback_pad;

        switch (strategy_) {
            case InitStrategy::RANDOM: {
                std::uniform_int_distribution<int32_t> dist(0, std::max(0, n_vocab_ - 1));
                guess = dist(rng_);
                // Avoid picking control / EOG tokens as random canvas noise
                if (tokenizer_.is_eog(guess)) {
                    guess = fallback_pad;
                }
                break;
            }
            case InitStrategy::NGRAM: {
                guess = ngram_pool_.predict(running_ctx, fallback_pad);
                break;
            }
            case InitStrategy::PAD:
            default: {
                guess = fallback_pad;
                break;
            }
        }

        guesses.push_back(guess);
        running_ctx.push_back(guess);
    }

    return guesses;
}

} // namespace diffusion
