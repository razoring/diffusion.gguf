#pragma once

#include "args.h"
#include "diffusion/engine.h"
#include "diffusion/events.h"

namespace diffusion {

class RenderFinal {
public:
    explicit RenderFinal(const CliArgs& args);

    bool on_step(const StepEvent& event);
    void finish(const GenResult& result);

private:
    CliArgs args_;
};

} // namespace diffusion
