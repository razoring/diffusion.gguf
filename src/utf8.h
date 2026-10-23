#pragma once

#include <cstddef>
#include <string_view>

namespace diffusion {

// Returns true if s is a valid complete UTF-8 byte sequence.
bool is_valid_utf8(std::string_view s);

// Returns length in bytes of the valid UTF-8 prefix, holding back any trailing incomplete multibyte sequence.
size_t valid_utf8_prefix_len(std::string_view s);

// Returns terminal column display width for UTF-8 string (wcwidth style, CJK / wide = 2, ASCII = 1, control = 0).
int utf8_display_width(std::string_view s);

} // namespace diffusion
