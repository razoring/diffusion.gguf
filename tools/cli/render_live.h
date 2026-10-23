#pragma once

#include "args.h"
#include "diffusion/engine.h"
#include "diffusion/events.h"
#include "diffusion/tokenizer.h"
#include <chrono>
#include <string>

namespace diffusion {

class RenderLive {
public:
    RenderLive(const CliArgs& args, const ITokenizer& tokenizer);
    ~RenderLive();

    bool on_step(const StepEvent& event);
    void finish(const GenResult& result);

private:
    void get_terminal_size(int& cols, int& rows);
    void clear_tail();
    void render_canvas_and_status(const StepEvent& ev, int cols, int rows);

    CliArgs args_;
    const ITokenizer& tokenizer_;
    bool color_enabled_ = true;
    bool is_tty_ = true;
    int current_col_ = 0;
    int last_lines_down_ = 0;
    int last_start_col_ = 0;
    std::chrono::steady_clock::time_point last_render_time_;
};

} // namespace diffusion
