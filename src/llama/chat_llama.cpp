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

std::string ChatLlama::apply(const std::vector<ChatMsg>& msgs, const std::vector<ToolDef>& tools, bool add_generation_prompt) const {
    if (!templates_) {
        // Fallback for base models
        std::string res;
        for (const auto& m : msgs) {
            res += m.role + ": " + m.content + "\n";
        }
        if (add_generation_prompt) res += "assistant: ";
        return res;
    }

    try {
        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.use_jinja = true;

        for (const auto& t : tools) {
            common_chat_tool ct;
            ct.name = t.name;
            ct.description = t.description;
            ct.parameters = t.parameters;
            inputs.tools.push_back(std::move(ct));
        }

        for (const auto& m : msgs) {
            common_chat_msg msg;
            msg.role = m.role;
            msg.content = m.content;
            msg.reasoning_content = m.reasoning_content;
            msg.tool_call_id = m.tool_call_id;
            for (const auto& tc : m.tool_calls) {
                common_chat_tool_call ctc;
                ctc.id = tc.id;
                ctc.name = tc.name;
                ctc.arguments = tc.arguments;
                msg.tool_calls.push_back(std::move(ctc));
            }
            inputs.messages.push_back(std::move(msg));
        }

        common_chat_params params = common_chat_templates_apply(templates_, inputs);
        return params.prompt;
    } catch (const std::exception& e) {
        std::cerr << "Warning: chat template apply failed: " << e.what() << ", using fallback formatting\n";
        std::string res;
        for (const auto& m : msgs) {
            res += "<start_of_turn>" + m.role + "\n" + m.content + "<end_of_turn>\n";
        }
        if (add_generation_prompt) res += "<start_of_turn>assistant\n";
        return res;
    } catch (...) {
        std::cerr << "Warning: unknown error in chat template apply, using fallback formatting\n";
        std::string res;
        for (const auto& m : msgs) {
            res += m.role + ": " + m.content + "\n";
        }
        if (add_generation_prompt) res += "assistant: ";
        return res;
    }
}

} // namespace diffusion
