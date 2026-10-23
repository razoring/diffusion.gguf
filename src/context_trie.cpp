#include "context_trie.h"

namespace diffusion {

ContextTrie::ContextTrie(int max_order)
    : max_order_(std::max(2, max_order)) {}

void ContextTrie::update(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return;

    size_t old_size = full_history_.size();
    full_history_.insert(full_history_.end(), tokens.begin(), tokens.end());

    // Index new n-grams spanning across boundary and newly added tokens
    size_t start_idx = (old_size >= static_cast<size_t>(max_order_)) ? (old_size - max_order_ + 1) : 0;
    for (size_t i = start_idx; i < full_history_.size(); ++i) {
        for (int order = 2; order <= max_order_; ++order) {
            if (i + order <= full_history_.size()) {
                uint64_t h = hash_ngram(&full_history_[i], order);
                auto& offsets = index_[h];
                if (offsets.empty() || offsets.back() != static_cast<int32_t>(i)) {
                    offsets.push_back(static_cast<int32_t>(i));
                }
            }
        }
    }
}

std::vector<int32_t> ContextTrie::match_continuation(
    const std::vector<int32_t>& context,
    int count,
    int min_match_len
) const {
    std::vector<int32_t> result;
    if (count <= 0 || context.empty()) return result;

    int ctx_len = static_cast<int>(context.size());

    // Try longest n-gram match first from max_order down to min_match_len
    for (int order = std::min(max_order_, ctx_len); order >= min_match_len; --order) {
        const int32_t* suffix_ptr = context.data() + (ctx_len - order);
        uint64_t h = hash_ngram(suffix_ptr, order);

        auto it = index_.find(h);
        if (it != index_.end() && !it->second.empty()) {
            // Pick latest occurrence before the current match location if possible
            for (auto r_it = it->second.rbegin(); r_it != it->second.rend(); ++r_it) {
                int32_t match_offset = *r_it;
                int32_t next_pos = match_offset + order;
                if (next_pos < static_cast<int32_t>(full_history_.size())) {
                    // Found valid follow tokens
                    int available = static_cast<int>(full_history_.size()) - next_pos;
                    int take = std::min(count, available);
                    result.assign(full_history_.begin() + next_pos, full_history_.begin() + next_pos + take);
                    return result;
                }
            }
        }
    }

    return result;
}

} // namespace diffusion
