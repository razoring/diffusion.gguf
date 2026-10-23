#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace diffusion {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::vector<int32_t> tokenize(std::string_view text, bool add_bos, bool parse_special) const = 0;
    virtual std::string piece(int32_t tok) const = 0;
    virtual bool is_eog(int32_t tok) const = 0;
    virtual int32_t pad_or_fallback() const = 0;
    virtual int32_t bos() const = 0;
    virtual int32_t eos() const = 0;
};

} // namespace diffusion
