#pragma once

#include "diffusion/chat.h"
#include "llama.h"
#include <memory>
#include <string>

struct common_chat_templates;

namespace diffusion {

class ChatLlama : public IChatFormatter {
public:
    ChatLlama(const llama_model* model, const std::string& custom_template = "");
    ~ChatLlama() override;

    std::string apply(const std::vector<ChatMsg>& msgs, bool add_generation_prompt) const override {
        return apply(msgs, {}, add_generation_prompt);
    }
    std::string apply(const std::vector<ChatMsg>& msgs, const std::vector<ToolDef>& tools, bool add_generation_prompt) const override;

private:
    struct common_chat_templates* templates_ = nullptr;
};

} // namespace diffusion
