#pragma once

#include "diffusion/tokenizer.h"
#include "llama.h"

namespace diffusion {

class TokenizerLlama : public ITokenizer {
public:
    explicit TokenizerLlama(const llama_vocab* vocab);

    std::vector<int32_t> tokenize(std::string_view text, bool add_bos, bool parse_special) const override;
    std::string piece(int32_t tok) const override;
    bool is_eog(int32_t tok) const override;
    int32_t pad_or_fallback() const override;
    int32_t bos() const override;
    int32_t eos() const override;

private:
    const llama_vocab* vocab_ = nullptr;
};

} // namespace diffusion
