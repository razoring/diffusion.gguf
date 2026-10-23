#include "utf8.h"
#include <cstdint>

namespace diffusion {

static bool decode_codepoint(const uint8_t*& p, const uint8_t* end, uint32_t& cp) {
    if (p >= end) return false;
    uint8_t c0 = *p++;
    if (c0 < 0x80) {
        cp = c0;
        return true;
    }
    if ((c0 & 0xE0) == 0xC0) {
        if (p >= end) return false;
        uint8_t c1 = *p++;
        if ((c1 & 0xC0) != 0x80) return false;
        cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        return cp >= 0x80;
    }
    if ((c0 & 0xF0) == 0xE0) {
        if (p + 1 >= end) return false;
        uint8_t c1 = *p++;
        uint8_t c2 = *p++;
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return false;
        cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        return true;
    }
    if ((c0 & 0xF8) == 0xF0) {
        if (p + 2 >= end) return false;
        uint8_t c1 = *p++;
        uint8_t c2 = *p++;
        uint8_t c3 = *p++;
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return false;
        cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        return cp >= 0x10000 && cp <= 0x10FFFF;
    }
    return false;
}

bool is_valid_utf8(std::string_view s) {
    const auto* p = reinterpret_cast<const uint8_t*>(s.data());
    const auto* end = p + s.size();
    uint32_t cp = 0;
    while (p < end) {
        if (!decode_codepoint(p, end, cp)) {
            return false;
        }
    }
    return true;
}

size_t valid_utf8_prefix_len(std::string_view s) {
    const auto* start = reinterpret_cast<const uint8_t*>(s.data());
    const auto* p = start;
    const auto* end = start + s.size();
    const auto* last_valid = start;
    uint32_t cp = 0;

    while (p < end) {
        const auto* prev = p;
        if (decode_codepoint(p, end, cp)) {
            last_valid = p;
        } else {
            // Failed: could be trailing incomplete byte sequence
            break;
        }
    }
    return static_cast<size_t>(last_valid - start);
}

static bool is_wide_char(uint32_t cp) {
    if (cp < 0x1100) return false;
    return (cp >= 0x1100 && cp <= 0x115F) ||
           (cp >= 0x2329 && cp <= 0x232A) ||
           (cp >= 0x2E80 && cp <= 0x303E) ||
           (cp >= 0x3040 && cp <= 0xA4CF) ||
           (cp >= 0xAC00 && cp <= 0xD7A3) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0xFE10 && cp <= 0xFE19) ||
           (cp >= 0xFE30 && cp <= 0xFE6F) ||
           (cp >= 0xFF00 && cp <= 0xFF60) ||
           (cp >= 0xFFE0 && cp <= 0xFFE6) ||
           (cp >= 0x1F300 && cp <= 0x1F9FF) ||
           (cp >= 0x20000 && cp <= 0x3FFFD);
}

int utf8_display_width(std::string_view s) {
    const auto* p = reinterpret_cast<const uint8_t*>(s.data());
    const auto* end = p + s.size();
    int width = 0;
    uint32_t cp = 0;

    while (p < end) {
        if (decode_codepoint(p, end, cp)) {
            if (cp == 0 || cp < 32 || (cp >= 0x7F && cp < 0xA0)) {
                // Control character: width 0
                continue;
            }
            width += is_wide_char(cp) ? 2 : 1;
        } else {
            // Invalid byte: count as 1 column fallback
            p++;
            width += 1;
        }
    }
    return width;
}

} // namespace diffusion
