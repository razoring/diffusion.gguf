#pragma once

#include <cstdint>
#include <vector>

namespace diffusion {

struct LogitsView {
    const float* data = nullptr;
    int n_vocab = 0;
    int n_rows = 0;

    const float* row(int i) const {
        return (data && i >= 0 && i < n_rows) ? (data + static_cast<size_t>(i) * n_vocab) : nullptr;
    }
};

class IModelBackend {
public:
    virtual ~IModelBackend() = default;

    virtual int  n_vocab() const = 0;
    virtual int  n_ctx() const = 0;
    virtual bool supports_kv_truncate() const = 0;

    // Evaluates tokens at absolute positions [pos0, pos0 + n).
    // want_all=true -> returns logits for all n rows; want_all=false -> last row only.
    virtual LogitsView eval(const int32_t* toks, int n, int pos0, bool want_all) = 0;

    // Drops KV cache for absolute positions >= pos
    virtual void kv_truncate(int pos) = 0;

    // Clears entire KV cache
    virtual void kv_clear() = 0;
};

} // namespace diffusion
