#pragma once

#include "diffusion/backend.h"
#include "diffusion/engine.h"
#include "diffusion/events.h"
#include "diffusion/sampler.h"
#include "diffusion/scheduler.h"
#include "diffusion/tokenizer.h"
#include <memory>

namespace diffusion {

class SchedulerAR : public IScheduler {
public:
    SchedulerAR(IModelBackend& backend, const ITokenizer& tokenizer, ISamplerChain& sampler);

    GenResult run(
        const std::vector<int32_t>& prompt,
        const GenParams& params,
        const StepCallback& callback = nullptr
    ) override {
        return run(prompt, params, 0, callback);
    }

    GenResult run(
        const std::vector<int32_t>& prompt,
        const GenParams& params,
        int reuse_prefix_len,
        const StepCallback& callback = nullptr
    ) override;

private:
    IModelBackend& backend_;
    const ITokenizer& tokenizer_;
    ISamplerChain& sampler_;
};

} // namespace diffusion
