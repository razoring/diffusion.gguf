#pragma once

#include "diffusion/engine.h"
#include "llama/backend_llama.h"
#include <string>

namespace diffusion {
namespace os {

struct SystemProfile {
    std::string os_name;
    int cpu_threads = 4;
    size_t total_ram_bytes = 0;
    size_t free_ram_bytes = 0;

    int gpu_count = 0;
    std::string primary_gpu_name;
    size_t primary_gpu_vram_bytes = 0;
    size_t primary_gpu_free_vram_bytes = 0;
    size_t total_vram_bytes = 0;
    size_t free_vram_bytes = 0;
};

// Probe operating system, CPU, RAM, and CUDA GPUs
SystemProfile detect_system_profile();

struct ExplicitFlags {
    bool n_gpu_layers = false;
    bool split_mode = false;
    bool n_threads = false;
    bool flash_attn = false;
    bool n_ctx = false;
    bool block_size = false;
    bool diffusion_steps = false;
};

// Automatically adjust BackendConfig and GenParams based on detected hardware
// Only modifies fields where explicit_flags is false.
void auto_tune_system(
    BackendConfig& backend,
    GenParams& params,
    const ExplicitFlags& flags,
    bool verbose = true
);

} // namespace os
} // namespace diffusion
