#include "args.h"
#include "render_final.h"
#include "render_live.h"
#include "render_progress.h"
#include "repl.h"
#include "engine.h"
#include "ggml-cpu.h"
#include <iostream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#define is_stdout_tty() (_isatty(_fileno(stdout)) != 0)
#define is_stdin_tty() (_isatty(_fileno(stdin)) != 0)
#else
#include <unistd.h>
#define is_stdout_tty() (isatty(fileno(stdout)) != 0)
#define is_stdin_tty() (isatty(fileno(stdin)) != 0)
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    diffusion::CliArgs args;
    try {
        args = diffusion::parse_cli_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (args.show_version) {
        std::cout << "diffusion-cli (diffusion.gguf) version 0.1.0\n";
        return 0;
    }

    if (args.show_help) {
        diffusion::print_cli_help();
        return 0;
    }

    if (args.backend.model_path.empty()) {
        std::cerr << "Error: --model PATH is required.\n";
        diffusion::print_cli_help();
        return 1;
    }

    if (args.render_mode == diffusion::RenderMode::AUTO) {
        args.render_mode = is_stdout_tty() ? diffusion::RenderMode::LIVE : diffusion::RenderMode::FINAL;
    }

    if (!args.numa_mode.empty()) {
        ggml_numa_strategy strategy = GGML_NUMA_STRATEGY_DISTRIBUTE;
        if (args.numa_mode == "isolate") strategy = GGML_NUMA_STRATEGY_ISOLATE;
        else if (args.numa_mode == "numactl") strategy = GGML_NUMA_STRATEGY_NUMACTL;
        else if (args.numa_mode == "mirror") strategy = GGML_NUMA_STRATEGY_MIRROR;
        ggml_numa_init(strategy);
    }

    try {
        diffusion::Engine engine(args.backend);

        if (args.conversation || (args.prompt.empty() && args.prompt_file.empty() && is_stdin_tty())) {
            diffusion::run_repl(engine, args);
            return 0;
        }

        if (args.prompt.empty()) {
            args.prompt = "Once upon a time";
        }

        if (args.render_mode == diffusion::RenderMode::PROGRESS) {
            diffusion::RenderProgress renderer(args);
            auto callback = [&](const diffusion::StepEvent& ev) {
                return renderer.on_step(ev);
            };
            diffusion::GenResult result = engine.generate(args.prompt, args.params, callback);
            renderer.finish(result);
        } else if (args.render_mode == diffusion::RenderMode::LIVE) {
            diffusion::RenderLive renderer(args, engine.tokenizer());
            auto callback = [&](const diffusion::StepEvent& ev) {
                return renderer.on_step(ev);
            };
            diffusion::GenResult result = engine.generate(args.prompt, args.params, callback);
            renderer.finish(result);
        } else {
            diffusion::RenderFinal renderer(args);
            auto callback = [&](const diffusion::StepEvent& ev) {
                return renderer.on_step(ev);
            };
            diffusion::GenResult result = engine.generate(args.prompt, args.params, callback);
            renderer.finish(result);
        }

    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Fatal unknown error\n";
        return 1;
    }

    return 0;
}
