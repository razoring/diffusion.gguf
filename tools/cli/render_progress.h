#pragma once

#include "args.h"
#include "diffusion/engine.h"
#include "diffusion/events.h"
#include <chrono>

namespace diffusion {

class RenderProgress {
public:
    explicit RenderProgress(const CliArgs& args);

    bool on_step(const StepEvent& event);
    void finish(const GenResult& result);

private:
    CliArgs args_;
    std::chrono::steady_clock::time_point last_render_time_;
};

} // namespace diffusion
