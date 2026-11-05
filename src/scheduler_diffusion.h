#pragma once

#include "diffusion/backend.h"
#include "diffusion/engine.h"
#include "diffusion/events.h"
#include "diffusion/sampler.h"
#include "diffusion/scheduler.h"
#include "diffusion/tokenizer.h"
#include "kernels/diff_device.h"
#include "kernels/reduce_logits.h"
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace diffusion {

class InitGuesser;

class SchedulerDiffusion : public IScheduler {
public:
    SchedulerDiffusion(
        IModelBackend& backend,
        const ITokenizer& tokenizer,
        ISamplerChain& sampler,
        int default_max_diffusion_steps = 4,
        InitGuesser* guesser = nullptr
    );

    ~SchedulerDiffusion() override;

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

    int default_max_diffusion_steps() const { return default_max_diffusion_steps_; }

private:
    IModelBackend& backend_;
    const ITokenizer& tokenizer_;
    ISamplerChain& sampler_;
    int default_max_diffusion_steps_;
    InitGuesser* guesser_ = nullptr;

    kernels::DeviceDiffusionBuffers device_bufs_;
    bool device_bufs_allocated_ = false;

    void* d_reductions_ = nullptr;
    size_t d_reductions_cap_ = 0;

    std::deque<uint8_t> branch_history_100_;
    int branch_count_100_ = 0;

    float get_branch_frequency_100() const {
        return branch_history_100_.empty() ? 0.0f : (static_cast<float>(branch_count_100_) / static_cast<float>(branch_history_100_.size()));
    }

    void record_branch_event(uint8_t branched) {
        if (branch_history_100_.size() >= 100) {
            branch_count_100_ -= branch_history_100_.front();
            branch_history_100_.pop_front();
        }
        branch_history_100_.push_back(branched);
        branch_count_100_ += branched;
    }
};

} // namespace diffusion
