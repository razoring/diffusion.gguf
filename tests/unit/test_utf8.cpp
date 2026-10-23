#include "utf8.h"
#include <cassert>
#include <iostream>
#include <string>

int main() {
    using namespace diffusion;

    // ASCII test
    assert(is_valid_utf8("Hello, World!"));
    assert(utf8_display_width("Hello, World!") == 13);
    assert(valid_utf8_prefix_len("Hello, World!") == 13);

    // Multibyte test (CJK: "你好" is 6 bytes, 2 chars, width 4)
    std::string s_cjk = "\xE4\xBD\xA0\xE5\xA5\xBD"; // 你好
    assert(is_valid_utf8(s_cjk));
    assert(utf8_display_width(s_cjk) == 4);
    assert(valid_utf8_prefix_len(s_cjk) == 6);

    // Incomplete trailing multibyte
    std::string s_incomplete = s_cjk.substr(0, 4); // "你" (3 bytes) + 1st byte of "好"
    assert(!is_valid_utf8(s_incomplete));
    assert(valid_utf8_prefix_len(s_incomplete) == 3);

    // Emoji test (4-byte utf8, width 2)
    std::string s_emoji = "\xF0\x9F\x9A\x80"; // 🚀
    assert(is_valid_utf8(s_emoji));
    assert(utf8_display_width(s_emoji) == 2);
    assert(valid_utf8_prefix_len(s_emoji) == 4);

    // Incomplete emoji (3 bytes of 4)
    std::string s_emoji_incomplete = s_emoji.substr(0, 3);
    assert(!is_valid_utf8(s_emoji_incomplete));
    assert(valid_utf8_prefix_len(s_emoji_incomplete) == 0);

    std::cout << "All test_utf8 assertions passed!\n";
    return 0;
}
