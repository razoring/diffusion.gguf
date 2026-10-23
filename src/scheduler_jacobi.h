#pragma once

#include "diffusion/backend.h"
#include "diffusion/engine.h"
#include "diffusion/events.h"
#include "diffusion/sampler.h"
#include "diffusion/scheduler.h"
#include "diffusion/tokenizer.h"
#include "init_guess.h"

namespace diffusion {

class SchedulerJacobi : public IScheduler {
public:
    SchedulerJacobi(
        IModelBackend& backend,
        const ITokenizer& tokenizer,
        ISamplerChain& sampler,
        InitGuesser& guesser
    );

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
    InitGuesser& guesser_;
};

} // namespace diffusion
