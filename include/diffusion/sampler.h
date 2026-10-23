#pragma once

#include <cstdint>
#include <memory>

namespace diffusion {

class ISamplerChain {
public:
    virtual ~ISamplerChain() = default;

    virtual std::unique_ptr<ISamplerChain> clone() const = 0;
    virtual void accept(int32_t tok) = 0;
    virtual bool can_accept(int32_t tok) const = 0;
    virtual int32_t sample(const float* logits, int abs_pos, float* out_prob = nullptr) = 0;
};

} // namespace diffusion
