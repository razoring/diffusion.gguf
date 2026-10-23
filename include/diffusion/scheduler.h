#pragma once

#include "engine.h"
#include "events.h"
#include <cstdint>
#include <vector>

namespace diffusion {

class IScheduler {
public:
    virtual ~IScheduler() = default;

    virtual GenResult run(
        const std::vector<int32_t>& prompt,
        const GenParams& params,
        const StepCallback& callback = nullptr
    ) {
        return run(prompt, params, 0, callback);
    }

    virtual GenResult run(
        const std::vector<int32_t>& prompt,
        const GenParams& params,
        int reuse_prefix_len,
        const StepCallback& callback = nullptr
    ) = 0;
};

} // namespace diffusion

