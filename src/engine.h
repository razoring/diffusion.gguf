#pragma once

#include "diffusion/engine.h"
#include "diffusion/events.h"
#include "llama/backend_llama.h"
#include "llama/chat_llama.h"
#include "llama/sampler_llama.h"
#include "llama/tokenizer_llama.h"
#include <memory>
#include <string>

namespace diffusion {

class Engine {
public:
    explicit Engine(const BackendConfig& config);
    ~Engine();

    GenResult generate(
        const std::string& prompt,
        const GenParams& params,
        const StepCallback& callback = nullptr
    );

    GenResult generate(
        const std::vector<int32_t>& prompt_tokens,
        const GenParams& params,
        int reuse_prefix_len = 0,
        const StepCallback& callback = nullptr
    );

    const ITokenizer& tokenizer() const { return *tokenizer_; }
    const IChatFormatter& chat_formatter() const { return *chat_formatter_; }
    IModelBackend& backend() { return *backend_; }

private:
    std::unique_ptr<BackendLlama> backend_;
    std::unique_ptr<TokenizerLlama> tokenizer_;
    std::unique_ptr<ChatLlama> chat_formatter_;
};

} // namespace diffusion
