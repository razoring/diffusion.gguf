#include "repl.h"
#include "diffusion/chat.h"
#include "engine.h"
#include "render_final.h"
#include "render_live.h"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <signal.h>
#endif


namespace diffusion {

static std::atomic<bool> g_in_generation{false};
static std::atomic<bool> g_cancel_generation{false};

#ifdef _WIN32
static BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        if (g_in_generation.load()) {
            g_cancel_generation.store(true);
            return TRUE;
        }
    }
    return FALSE;
}
#else
static void signal_handler(int signum) {
    if (signum == SIGINT) {
        if (g_in_generation.load()) {
            g_cancel_generation.store(true);
            return;
        }
        std::exit(0);
    }
}
#endif

void run_repl(Engine& engine, CliArgs& args) {
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
#endif

    std::vector<ChatMsg> history;
    if (!args.system_prompt.empty()) {
        history.push_back(ChatMsg{"system", args.system_prompt});
    }

    std::vector<int32_t> cached_prompt_tokens;

    const char* mode_name = (args.params.mode == DecodeMode::WHOLE) ? "whole" : "block";

    std::cout << "=== diffusion.gguf REPL (mode: " << mode_name
              << ", block: " << args.params.block_size << ") ===\n"
              << "Type /help for commands, Ctrl+C to cancel generation, /exit to quit.\n\n";

    std::string line;
    while (true) {
        std::cout << "> ";
        std::cout.flush();

        if (!std::getline(std::cin, line)) {
            break;
        }

        // Trim leading/trailing whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        std::string input = line.substr(start, end - start + 1);

        if (input == "/exit" || input == "/quit") {
            break;
        }

        if (input == "/help") {
            std::cout << "Available commands:\n"
                      << "  /exit, /quit             Exit REPL\n"
                      << "  /reset, /clear           Clear conversation history and KV cache\n"
                      << "  /mode <block|whole|ar>   Change generation mode\n"
                      << "  /block <N>               Change block size\n"
                      << "  /init <ngram|random|pad> Change initial canvas guess strategy\n"
                      << "  /temp <FLOAT>            Set sampling temperature\n"
                      << "  /stats                   Toggle statistics output\n"
                      << "  /help                    Show this help\n\n";
            continue;
        }

        if (input == "/reset" || input == "/clear") {
            history.clear();
            if (!args.system_prompt.empty()) {
                history.push_back(ChatMsg{"system", args.system_prompt});
            }
            cached_prompt_tokens.clear();
            engine.backend().kv_clear();
            std::cout << "[Conversation and KV cache reset]\n\n";
            continue;
        }

        if (input.rfind("/mode ", 0) == 0) {
            std::string m = input.substr(6);
            if (m == "ar") {
                std::cout << "[Error: Autoregressive mode ('ar') has been completely removed. Use 'block' or 'whole']\n\n";
            } else if (m == "whole") {
                args.params.mode = DecodeMode::WHOLE;
                std::cout << "[Mode set to whole]\n\n";
            } else {
                args.params.mode = DecodeMode::BLOCK;
                std::cout << "[Mode set to block]\n\n";
            }
            continue;
        }

        if (input.rfind("/block ", 0) == 0) {
            try {
                args.params.block_size = std::stoi(input.substr(7));
                std::cout << "[Block size set to " << args.params.block_size << "]\n\n";
            } catch (...) {
                std::cout << "[Invalid block size]\n\n";
            }
            continue;
        }

        if (input.rfind("/init ", 0) == 0) {
            std::string strat = input.substr(6);
            if (strat == "random") args.params.init_strategy = InitStrategy::RANDOM;
            else if (strat == "pad") args.params.init_strategy = InitStrategy::PAD;
            else args.params.init_strategy = InitStrategy::NGRAM;
            std::cout << "[Init strategy set to " << strat << "]\n\n";
            continue;
        }

        if (input.rfind("/temp ", 0) == 0) {
            try {
                args.params.sampling.temp = std::stof(input.substr(6));
                std::cout << "[Temperature set to " << args.params.sampling.temp << "]\n\n";
            } catch (...) {
                std::cout << "[Invalid temperature]\n\n";
            }
            continue;
        }

        if (input == "/stats") {
            args.show_stats = !args.show_stats;
            std::cout << "[Stats " << (args.show_stats ? "enabled" : "disabled") << "]\n\n";
            continue;
        }

        // Add user message to conversation history
        history.push_back(ChatMsg{"user", input});

        // Format prompt via Jinja chat template
        std::string full_prompt = engine.chat_formatter().apply(history, true);
        auto prompt_tokens = engine.tokenizer().tokenize(full_prompt, true, true);

        // Find longest common prefix with KV cache
        int reuse_prefix_len = 0;
        while (reuse_prefix_len < static_cast<int>(prompt_tokens.size()) &&
               reuse_prefix_len < static_cast<int>(cached_prompt_tokens.size()) &&
               prompt_tokens[reuse_prefix_len] == cached_prompt_tokens[reuse_prefix_len]) {
            reuse_prefix_len++;
        }

        g_cancel_generation.store(false);
        g_in_generation.store(true);

        GenResult result;
        if (args.render_mode == RenderMode::FINAL) {
            RenderFinal renderer(args);
            auto callback = [&](const StepEvent& ev) {
                if (g_cancel_generation.load()) return false;
                return renderer.on_step(ev);
            };
            result = engine.generate(prompt_tokens, args.params, reuse_prefix_len, callback);
            renderer.finish(result);
        } else {
            RenderLive renderer(args, engine.tokenizer());
            auto callback = [&](const StepEvent& ev) {
                if (g_cancel_generation.load()) return false;
                return renderer.on_step(ev);
            };
            result = engine.generate(prompt_tokens, args.params, reuse_prefix_len, callback);
            renderer.finish(result);
        }

        g_in_generation.store(false);

        if (g_cancel_generation.exchange(false) || result.stop_reason == StopReason::CANCELLED) {
            std::cout << "\n[Generation cancelled]\n";
        }

        // Store assistant response in history
        history.push_back(ChatMsg{"assistant", result.text});

        // Update KV cached tokens: prompt tokens + generated tokens
        cached_prompt_tokens = prompt_tokens;
        cached_prompt_tokens.insert(cached_prompt_tokens.end(), result.tokens.begin(), result.tokens.end());

        std::cout << "\n";
    }
}

} // namespace diffusion
