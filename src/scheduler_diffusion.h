#pragma once

#include "diffusion/backend.h"
#include "diffusion/engine.h"
#include "diffusion/events.h"
#include "diffusion/sampler.h"
#include "diffusion/scheduler.h"
#include "diffusion/tokenizer.h"
#include "kernels/diff_device.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace diffusion {

/**
 * SchedulerDiffusion implements pure discrete block diffusion inference directly on
 * standard GGUF weights with zero retraining and zero speculative rejection loops.
 *
 * Mathematical mechanics:
 * 1. Model-Derived Prior Seeding: seeds canvas from prefix and initial model predictions.
 * 2. Active Query Set (Q_active) vs Frozen KV Anchor Set (K_frozen) execution partitioning.
 * 3. Dynamic confidence-ordered denoising with adaptive freeze threshold theta_freeze(m).
 * 4. In-VRAM device reduction and warp-ballot fixed-point stationarity check.
 * 5. Cumulative active work C_effective KPI measurement.
 */
class InitGuesser;

class SchedulerDiffusion : public IScheduler {
public:
    SchedulerDiffusion(
        IModelBackend& backend,
        const ITokenizer& tokenizer,
        ISamplerChain& sampler,
        int default_block_size = 8,
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

    // Accessors for tuning / debugging
    int default_block_size() const { return default_block_size_; }
    int default_max_diffusion_steps() const { return default_max_diffusion_steps_; }

private:
    IModelBackend& backend_;
    const ITokenizer& tokenizer_;
    ISamplerChain& sampler_;
    int default_block_size_;
    int default_max_diffusion_steps_;
    InitGuesser* guesser_ = nullptr;

    kernels::DeviceDiffusionBuffers device_bufs_;
    bool device_bufs_allocated_ = false;
};

} // namespace diffusion
