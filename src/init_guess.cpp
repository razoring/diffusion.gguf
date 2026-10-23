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
    if (tokens.size() < 2) return;

    for (size_t i = 0; i < tokens.size(); ++i) {
        for (int order = 2; order <= std::min(8, n_); ++order) {
            if (i + order <= tokens.size()) {
                if (static_cast<int>(table_.size()) >= max_entries_) break;
                uint64_t h = hash_prefix(&tokens[i], order - 1);
                int32_t next_tok = tokens[i + order - 1];
                auto& list = table_[h];
                if (std::find(list.begin(), list.end(), next_tok) == list.end()) {
                    list.push_back(next_tok);
                }
            }
        }
    }
}

int32_t NgramPool::predict(const std::vector<int32_t>& context, int32_t fallback) const {
    if (context.empty()) return fallback;

    int ctx_len = static_cast<int>(context.size());
    for (int order = std::min(8, n_); order >= 2; --order) {
        int prefix_len = order - 1;
        if (ctx_len >= prefix_len) {
            const int32_t* ptr = context.data() + (ctx_len - prefix_len);
            uint64_t h = hash_prefix(ptr, prefix_len);
            auto it = table_.find(h);
            if (it != table_.end() && !it->second.empty()) {
                return it->second.back();
            }
        }
    }
    return fallback;
}

InitGuesser::InitGuesser(
    InitStrategy strategy,
    const ITokenizer& tokenizer,
    int32_t n_vocab,
    uint32_t seed,
    int32_t ngram_n,
    int32_t ngram_pool_max,
    IModelBackend* draft_backend,
    ISamplerChain* draft_sampler
) : strategy_(strategy),
    tokenizer_(tokenizer),
    n_vocab_(n_vocab),
    rng_(seed),
    ngram_pool_(std::max(4, ngram_n), ngram_pool_max),
    context_trie_(12),
    draft_backend_(draft_backend),
    draft_sampler_(draft_sampler) {}

void InitGuesser::update_ngram(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return;
    context_trie_.update(tokens);
    const auto& hist = context_trie_.full_history();
    if (hist.size() >= 2) {
        size_t window = std::min(static_cast<size_t>(16), hist.size());
        std::vector<int32_t> slice(hist.end() - window, hist.end());
        ngram_pool_.train(slice);
    }
}

std::vector<int32_t> InitGuesser::make_guesses(int count, const std::vector<int32_t>& context) {
    std::vector<int32_t> guesses;
    if (count <= 0 || context.empty()) return guesses;
    guesses.reserve(count);

    int32_t fallback_pad = tokenizer_.pad_or_fallback();

    // Strategy: DRAFT model speculative generation
    if (strategy_ == InitStrategy::DRAFT && draft_backend_ && draft_sampler_ && !context.empty()) {
        try {
            draft_backend_->kv_clear();
            int ctx_len = static_cast<int>(context.size());
            LogitsView dlogits = draft_backend_->eval(context.data(), ctx_len, 0, false);
            auto dsampler = draft_sampler_->clone();
            for (int32_t tok : context) {
                dsampler->accept(tok);
            }

            int pos = ctx_len;
            for (int i = 0; i < count; ++i) {
                float prob = 0.0f;
                int32_t tok = dsampler->sample(dlogits.row(0), pos, &prob);
                if (tokenizer_.is_eog(tok)) break;
                guesses.push_back(tok);
                dsampler->accept(tok);
                pos++;
                if (i + 1 < count) {
                    dlogits = draft_backend_->eval(&tok, 1, pos - 1, false);
                }
            }
            if (static_cast<int>(guesses.size()) >= count) {
                return guesses;
            }
        } catch (...) {
            // Fallback gracefully on draft eval error
        }
    }

    // In NGRAM mode (native self-diffusion), match high-confidence chained phrases
    if (strategy_ == InitStrategy::NGRAM && !context.empty()) {
        // 1. High-confidence chained phrase match from ContextTrie (order down to 1)
        auto phrase = context_trie_.chain_match(context, count, 1);
        for (int32_t tok : phrase) {
            if (tokenizer_.is_eog(tok)) break;
            guesses.push_back(tok);
            if (static_cast<int>(guesses.size()) >= count) break;
        }

        // 2. Iterative N-gram prediction from learned pool (order >= 2)
        if (static_cast<int>(guesses.size()) < count) {
            std::vector<int32_t> running_ctx = context;
            running_ctx.insert(running_ctx.end(), guesses.begin(), guesses.end());
            while (static_cast<int>(guesses.size()) < count) {
                int32_t pred = ngram_pool_.predict(running_ctx, -1);
                if (pred < 0 || tokenizer_.is_eog(pred)) {
                    break;
                }
                guesses.push_back(pred);
                running_ctx.push_back(pred);
            }
        }

        // 3. Fallback backoff continuation from parent syntactic context
        if (static_cast<int>(guesses.size()) < count && context.size() >= 2) {
            std::vector<int32_t> backoff_ctx(context.end() - 2, context.end() - 1);
            auto backoff_phrase = context_trie_.chain_match(backoff_ctx, count - static_cast<int>(guesses.size()), 1);
            for (int32_t tok : backoff_phrase) {
                if (tokenizer_.is_eog(tok)) break;
                guesses.push_back(tok);
                if (static_cast<int>(guesses.size()) >= count) break;
            }
        }

        // 4. Keep batch size strictly static: pad open slots so CUDA graph warmup never resets
        int needed = count - static_cast<int>(guesses.size());
        if (needed > 0) {
            int32_t pad_tok = guesses.empty() ? context.back() : guesses.back();
            if (tokenizer_.is_eog(pad_tok)) pad_tok = fallback_pad;
            for (int i = 0; i < needed; ++i) {
                guesses.push_back(pad_tok);
            }
        }
        return guesses;
    }

    // For RANDOM / PAD strategies (testing modes)
    int needed = count - static_cast<int>(guesses.size());
    for (int i = 0; i < needed; ++i) {
        int32_t guess = fallback_pad;
        if (strategy_ == InitStrategy::RANDOM) {
            std::uniform_int_distribution<int32_t> dist(0, std::max(0, n_vocab_ - 1));
            guess = dist(rng_);
            if (tokenizer_.is_eog(guess)) guess = fallback_pad;
        }
        guesses.push_back(guess);
    }

    return guesses;
}

} // namespace diffusion
