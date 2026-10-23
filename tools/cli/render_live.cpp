#include "render_live.h"
#include "utf8.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_set>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#define is_terminal(fd) (_isatty(fd) != 0)
#else
#include <unistd.h>
#include <sys/ioctl.h>
#define is_terminal(fd) (isatty(fd) != 0)
#endif

namespace diffusion {

RenderLive::RenderLive(const CliArgs& args, const ITokenizer& tokenizer)
    : args_(args), tokenizer_(tokenizer) {
#ifdef _WIN32
    is_tty_ = is_terminal(_fileno(stdout));
    if (is_tty_) {
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD dwMode = 0;
            if (GetConsoleMode(hOut, &dwMode)) {
                dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                SetConsoleMode(hOut, dwMode);
            }
        }
    }
#else
    is_tty_ = is_terminal(fileno(stdout));
#endif

    color_enabled_ = is_tty_ && !args_.no_color && (std::getenv("NO_COLOR") == nullptr);
    last_render_time_ = std::chrono::steady_clock::now();
}

RenderLive::~RenderLive() {
    clear_tail();
}

void RenderLive::get_terminal_size(int& cols, int& rows) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
        if (cols > 0 && rows > 0) return;
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        cols = ws.ws_col;
        rows = ws.ws_row;
        return;
    }
#endif
    cols = 80;
    rows = 24;
}

void RenderLive::clear_tail() {
    if (last_lines_down_ > 0) {
        std::cout << "\x1b[" << last_lines_down_ << "A"
                  << "\x1b[" << (last_start_col_ + 1) << "G"
                  << "\x1b[J";
        last_lines_down_ = 0;
    }
}

bool RenderLive::on_step(const StepEvent& ev) {
    if (!is_tty_) {
        for (int32_t tok : ev.committed_new) {
            std::cout << tokenizer_.piece(tok);
        }
        std::cout.flush();
        return true;
    }

    auto now = std::chrono::steady_clock::now();
    int min_interval_ms = (args_.render_fps > 0) ? (1000 / args_.render_fps) : 33;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_render_time_).count();

    if (!ev.done && elapsed_ms < min_interval_ms && ev.committed_new.empty()) {
        return true;
    }
    last_render_time_ = now;

    int cols = 80, rows = 24;
    get_terminal_size(cols, rows);

    // 1. Clear active tail and status line
    clear_tail();

    // 2. Output committed tokens
    for (int32_t tok : ev.committed_new) {
        std::string piece = tokenizer_.piece(tok);
        std::cout << piece;

        for (size_t i = 0; i < piece.size(); ++i) {
            if (piece[i] == '\n') {
                current_col_ = 0;
            } else {
                current_col_++;
                if (cols > 0 && current_col_ >= cols) {
                    current_col_ = 0;
                }
            }
        }
    }
    std::cout.flush();

    if (ev.done) {
        std::cout << "\n";
        std::cout.flush();
        return true;
    }

    // 3. Render unconverged canvas tail and status bar
    last_start_col_ = current_col_;
    render_canvas_and_status(ev, cols, rows);

    return true;
}

void RenderLive::render_canvas_and_status(const StepEvent& ev, int cols, int rows) {
    if (ev.canvas.empty()) return;

    std::unordered_set<int> diff_set;
    for (const auto& d : ev.diffs) {
        diff_set.insert(d.pos);
    }

    int line_count = 0;
    int col = last_start_col_;
    int max_tail_lines = (args_.params.mode == DecodeMode::WHOLE) ? std::max(1, rows - 3) : 999;

    for (size_t i = 0; i < ev.canvas.size(); ++i) {
        if (line_count >= max_tail_lines) {
            std::cout << (color_enabled_ ? "\x1b[2m …\x1b[0m" : " …");
            break;
        }

        int32_t tok = ev.canvas[i];
        std::string piece = tokenizer_.piece(tok);
        bool is_diff = diff_set.count(static_cast<int>(i)) > 0;

        if (color_enabled_) {
            if (is_diff) {
                std::cout << "\x1b[1;36m"; // Bold cyan
            } else {
                std::cout << "\x1b[2m";    // Dim unconverged
            }
        }

        std::cout << piece;

        if (color_enabled_) {
            std::cout << "\x1b[0m";
        }

        for (size_t j = 0; j < piece.size(); ++j) {
            if (piece[j] == '\n') {
                line_count++;
                col = 0;
            } else {
                col++;
                if (cols > 0 && col >= cols) {
                    line_count++;
                    col = 0;
                }
            }
        }
    }

    // Status line on next row
    std::cout << "\n";
    line_count++;

    if (color_enabled_) {
        std::cout << "\x1b[90m"; // Dim gray
    }

    int n_predict = args_.params.n_predict > 0 ? args_.params.n_predict : 128;
    float pct = (n_predict > 0) ? (100.0f * ev.committed_upto / n_predict) : 0.0f;
    if (pct > 100.0f) pct = 100.0f;

    std::cout << "step " << ev.step
              << " · " << ev.committed_upto << "/" << n_predict << " tok (" << static_cast<int>(pct) << "%)"
              << " · " << std::fixed << std::setprecision(2) << ev.tok_per_iter << " tok/it"
              << " · " << std::fixed << std::setprecision(1) << ev.tok_per_s << " tok/s";

    if (color_enabled_) {
        std::cout << "\x1b[0m";
    }

    std::cout.flush();
    last_lines_down_ = line_count;
}

void RenderLive::finish(const GenResult& result) {
    clear_tail();
    std::cout.flush();

    if (!args_.tokens_out_file.empty()) {
        std::ofstream f(args_.tokens_out_file);
        if (f) {
            for (size_t i = 0; i < result.tokens.size(); ++i) {
                if (i > 0) f << ",";
                f << result.tokens[i];
            }
            f << "\n";
        }
    }

    if (args_.show_stats) {
        std::cerr << "{\n"
                  << "  \"tokens\": " << result.tokens.size() << ",\n"
                  << "  \"iterations\": " << result.iterations << ",\n"
                  << "  \"tokens_per_iteration\": " << result.tok_per_iter << ",\n"
                  << "  \"tokens_per_second\": " << result.tok_per_s << ",\n"
                  << "  \"total_time_ms\": " << result.total_time_ms << "\n"
                  << "}\n";
    }
}

} // namespace diffusion
