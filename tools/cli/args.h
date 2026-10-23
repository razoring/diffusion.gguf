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

struct CliArgs {
    BackendConfig backend;
    GenParams params;
    std::string prompt;
    std::string prompt_file;
    std::string tokens_out_file;
    RenderMode render_mode = RenderMode::AUTO;
    bool conversation = false;
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
