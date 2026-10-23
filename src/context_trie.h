#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace diffusion {

class ContextTrie {
public:
    explicit ContextTrie(int max_order = 12);

    // Updates trie with new tokens
    void update(const std::vector<int32_t>& tokens);

    // Searches context for the longest suffix match of length min_order..max_order
    // and returns up to 'count' continuation tokens.
    std::vector<int32_t> match_continuation(
        const std::vector<int32_t>& context,
        int count,
        int min_match_len = 1
    ) const;

    // Chained continuation matching: iteratively extends context until 'count' tokens are collected
    std::vector<int32_t> chain_match(
        const std::vector<int32_t>& context,
        int count,
        int min_match_len = 1
    ) const;

    const std::vector<int32_t>& full_history() const { return full_history_; }

private:
    int max_order_ = 12;
    std::vector<int32_t> full_history_;

    // Fast hash table mapping n-gram prefix hash to array of occurrence offsets in full_history_
    std::unordered_map<uint64_t, std::vector<int32_t>> index_;

    static uint64_t hash_ngram(const int32_t* ptr, int len) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (int i = 0; i < len; ++i) {
            h ^= static_cast<uint64_t>(ptr[i]);
            h *= 0x100000001b3ULL;
        }
        return h;
    }
};

} // namespace diffusion
