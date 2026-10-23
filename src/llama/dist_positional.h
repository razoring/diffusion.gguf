#pragma once

#include "llama.h"
#include <cstdint>

namespace diffusion {

struct llama_sampler_dist_positional_ctx {
    uint32_t seed = 0;
    int abs_pos = 0;
};

// Creates a custom llama_sampler that re-seeds its RNG deterministically for each abs_pos
llama_sampler* llama_sampler_init_dist_positional(uint32_t seed);

// Sets the target absolute position for the next sample call
void llama_sampler_dist_positional_set_pos(llama_sampler* smpl, int abs_pos);

} // namespace diffusion
