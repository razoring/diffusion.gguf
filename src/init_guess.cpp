#include "init_guess.h"
#include <algorithm>
#include <cctype>

namespace diffusion {

static uint64_t hash_stem_piece(std::string_view piece, int& out_len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    int len = 0;
    for (unsigned char c : piece) {
        if (std::isalnum(c)) {
            unsigned char lc = static_cast<unsigned char>(std::tolower(c));
            h ^= static_cast<uint64_t>(lc);
            h *= 0x100000001b3ULL;
            len++;
        }
    }
    out_len = len;
    return h;
}

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
    if (tokens.size() < 3) return;

    for (size_t i = 0; i < tokens.size(); ++i) {
        for (int order = 2; order <= std::min(8, n_); ++order) {
            if (i + order <= tokens.size()) {
                if (static_cast<int>(table_.size()) >= max_entries_) break;
                uint64_t h = hash_prefix(&tokens[i], order - 1);
                int32_t next_tok = tokens[i + order - 1];
                auto& tr = table_[h];
                int cnt = ++tr.counts[next_tok];
                if (cnt >= tr.best_count) {
                    tr.best_count = cnt;
                    tr.best_tok = next_tok;
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
            if (it != table_.end() && it->second.best_tok >= 0 && (order >= 3 || it->second.best_count >= 1)) {
                return it->second.best_tok;
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
    int32_t ngram_pool_max
) : strategy_(strategy),
    tokenizer_(tokenizer),
    n_vocab_(n_vocab),
    rng_(seed),
    ngram_pool_(std::max(4, ngram_n), ngram_pool_max),
    context_trie_(12) {}

void InitGuesser::update_ngram(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return;
    size_t old_size = context_trie_.full_history().size();
    context_trie_.update(tokens);
    const auto& hist = context_trie_.full_history();
    if (hist.size() >= 2) {
        size_t start_idx = (old_size >= 7) ? (old_size - 7) : 0;
        std::vector<int32_t> slice(hist.begin() + start_idx, hist.end());
        ngram_pool_.train(slice);
        for (size_t i = (old_size > 0 ? old_size - 1 : 0); i + 1 < hist.size(); ++i) {
            int stem_len = 0;
            uint64_t sh = hash_stem_piece(tokenizer_.piece(hist[i]), stem_len);
            if (stem_len >= 4 && !tokenizer_.is_eog(hist[i + 1])) {
                stem_next_[sh] = hist[i + 1];
            }
        }
    }
}

std::vector<int32_t> InitGuesser::make_guesses(int count, const std::vector<int32_t>& context) {
    std::vector<int32_t> guesses;
    if (count <= 0 || context.empty()) return guesses;
    guesses.reserve(count);

    int32_t fallback_pad = tokenizer_.pad_or_fallback();

    if ((strategy_ == InitStrategy::NGRAM || strategy_ == InitStrategy::ADAPTIVE_ENTROPY) && !context.empty()) {
        std::vector<int32_t> cur_ctx = context;
        cur_ctx.reserve(context.size() + count);

        while (static_cast<int>(guesses.size()) < count) {
            int rem = count - static_cast<int>(guesses.size());

            // 1. Multi-token phrase match from ContextTrie (order >= 3)
            int alnum_cnt = 0;
            for (size_t k = (cur_ctx.size() >= 3 ? cur_ctx.size() - 3 : 0); k < cur_ctx.size(); ++k) {
                int slen = 0;
                hash_stem_piece(tokenizer_.piece(cur_ctx[k]), slen);
                if (slen > 0) alnum_cnt++;
            }
            int min_trie_len = (alnum_cnt >= 2) ? 1 : 2;
            auto phrase = context_trie_.match_continuation(cur_ctx, rem, min_trie_len);
            if (!phrase.empty()) {
                bool added = false;
                for (int32_t tok : phrase) {
                    if (tokenizer_.is_eog(tok)) break;
                    guesses.push_back(tok);
                    cur_ctx.push_back(tok);
                    added = true;
                    if (static_cast<int>(guesses.size()) >= count) break;
                }
                if (added) continue;
            }

            // 2. Highest-frequency N-gram continuation (order >= 3)
            if (cur_ctx.size() >= 1) {
                int32_t pred = ngram_pool_.predict(cur_ctx, -1);
                if (pred >= 0 && !tokenizer_.is_eog(pred) && pred != cur_ctx.back()) {
                    guesses.push_back(pred);
                    cur_ctx.push_back(pred);
                    continue;
                }
            }

            // 3. Stem continuation match (only for first lookahead slot)
            if (guesses.empty()) {
                int stem_len = 0;
                uint64_t sh = hash_stem_piece(tokenizer_.piece(cur_ctx.back()), stem_len);
                if (stem_len >= 5) {
                    auto it = stem_next_.find(sh);
                    if (it != stem_next_.end() && it->second >= 0 && !tokenizer_.is_eog(it->second) && it->second != cur_ctx.back()) {
                        guesses.push_back(it->second);
                        cur_ctx.push_back(it->second);
                        continue;
                    }
                }
            }

            break;
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
