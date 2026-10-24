#pragma once

#include <string>
#include <vector>

namespace diffusion {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

struct ToolDef {
    std::string name;
    std::string description;
    std::string parameters;
};

struct ChatMsg {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;
};

class IChatFormatter {
public:
    virtual ~IChatFormatter() = default;

    virtual std::string apply(const std::vector<ChatMsg>& msgs,
                              bool add_generation_prompt = true) const {
        return apply(msgs, {}, add_generation_prompt);
    }

    virtual std::string apply(const std::vector<ChatMsg>& msgs,
                              const std::vector<ToolDef>& tools,
                              bool add_generation_prompt = true) const = 0;
};

} // namespace diffusion
