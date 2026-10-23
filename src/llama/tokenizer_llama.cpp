#include "tokenizer_llama.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace diffusion {

TokenizerLlama::TokenizerLlama(const llama_vocab* vocab) : vocab_(vocab) {
    if (!vocab_) {
        throw std::invalid_argument("TokenizerLlama requires non-null llama_vocab pointer");
    }
}

std::vector<int32_t> TokenizerLlama::tokenize(std::string_view text, bool add_bos, bool parse_special) const {
    if (text.empty()) {
        if (add_bos) {
            llama_token b = llama_vocab_bos(vocab_);
            if (b >= 0) return {b};
        }
        return {};
    }

    // Initial estimate for token count
    int32_t n_max = static_cast<int32_t>(text.size()) + (add_bos ? 2 : 1) + 16;
    std::vector<llama_token> tokens(n_max);

    int32_t n = llama_tokenize(
        vocab_,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_bos,
        parse_special
    );

    if (n < 0) {
        // Buffer was too small, n is negative required capacity
        tokens.resize(-n);
        n = llama_tokenize(
            vocab_,
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_bos,
            parse_special
        );
    }

    if (n < 0) {
        throw std::runtime_error("llama_tokenize failed to tokenize string");
    }

    tokens.resize(n);
    return std::vector<int32_t>(tokens.begin(), tokens.end());
}

std::string TokenizerLlama::piece(int32_t tok) const {
    char buf[64];
    int32_t n = llama_token_to_piece(vocab_, tok, buf, sizeof(buf), 0, true);
    if (n >= 0 && n <= static_cast<int32_t>(sizeof(buf))) {
        return std::string(buf, n);
    }
    if (n < 0) {
        std::string dyn_buf;
        dyn_buf.resize(-n);
        int32_t n2 = llama_token_to_piece(vocab_, tok, &dyn_buf[0], -n, 0, true);
        if (n2 >= 0) {
            dyn_buf.resize(n2);
            return dyn_buf;
        }
    }
    return {};
}

bool TokenizerLlama::is_eog(int32_t tok) const {
    return llama_vocab_is_eog(vocab_, tok);
}

int32_t TokenizerLlama::pad_or_fallback() const {
    llama_token pad = llama_vocab_pad(vocab_);
    if (pad >= 0) return pad;
    llama_token eos = llama_vocab_eos(vocab_);
    if (eos >= 0) return eos;
    return 0;
}

int32_t TokenizerLlama::bos() const {
    return llama_vocab_bos(vocab_);
}

int32_t TokenizerLlama::eos() const {
    return llama_vocab_eos(vocab_);
}

} // namespace diffusion
