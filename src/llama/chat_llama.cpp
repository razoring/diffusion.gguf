#include "chat_llama.h"
#include "chat.h"
#include <iostream>

namespace diffusion {

ChatLlama::ChatLlama(const llama_model* model, const std::string& custom_template) {
    try {
        const char* tmpl = custom_template.empty() ? nullptr : custom_template.c_str();
        auto ptr = common_chat_templates_init(model, tmpl ? tmpl : "");
        templates_ = ptr.release();
    } catch (const std::exception& e) {
        // Models without templates (like base models) gracefully fall back
        templates_ = nullptr;
    } catch (...) {
        templates_ = nullptr;
    }
}

ChatLlama::~ChatLlama() {
    if (templates_) {
        common_chat_templates_free(templates_);
        templates_ = nullptr;
    }
}

std::string ChatLlama::apply(const std::vector<ChatMsg>& msgs, bool add_generation_prompt) const {
    if (!templates_) {
        // Fallback for base models
        std::string res;
        for (const auto& m : msgs) {
            res += m.role + ": " + m.content + "\n";
        }
        if (add_generation_prompt) res += "assistant: ";
        return res;
    }

    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja = true;

    for (const auto& m : msgs) {
        common_chat_msg msg;
        msg.role = m.role;
        msg.content = m.content;
        inputs.messages.push_back(std::move(msg));
    }

    common_chat_params params = common_chat_templates_apply(templates_, inputs);
    return params.prompt;
}

} // namespace diffusion
