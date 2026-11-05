#pragma once

#include "diffusion/engine.h"
#include "llama/backend_llama.h"
#include <string>
#include <vector>

namespace diffusion {

enum class RenderMode {
    AUTO = 0,
    LIVE,
    PROGRESS,
    FINAL
};

struct SpeculativeArgs {
    std::string draft_model_path;
    int32_t draft_gpu_layers = -1;
    int32_t draft_max = 0; // 0 = disabled by default
    int32_t draft_min = 0;
    float draft_p_min = 0.0f;
    float draft_p_split = 0.0f;
    std::string lookup_cache_static;
    std::string lookup_cache_dynamic;
};

struct CliArgs {
    BackendConfig backend;
    GenParams params;
    SpeculativeArgs speculative;
    std::string prompt;
    std::string prompt_file;
    std::string tokens_out_file;
    RenderMode render_mode = RenderMode::AUTO;
    bool conversation = false;
    bool single_turn = false;
    std::string system_prompt;
    std::string custom_chat_template;
    bool show_stats = false;
    bool show_version = false;
    bool show_help = false;
    bool no_color = false;
    int render_fps = 30;
    std::string numa_mode;
};

CliArgs parse_cli_args(int argc, char** argv);
void print_cli_help();

} // namespace diffusion
