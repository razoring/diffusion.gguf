#pragma once

#include <string>
#include <vector>

namespace diffusion {

struct ChatMsg {
    std::string role;
    std::string content;
};

class IChatFormatter {
public:
    virtual ~IChatFormatter() = default;

    virtual std::string apply(const std::vector<ChatMsg>& msgs, bool add_generation_prompt) const = 0;
};

} // namespace diffusion
