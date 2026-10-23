#include "args.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>

namespace diffusion {

void print_cli_help() {
    std::cout << "Usage: diffusion-cli -m <model.gguf> [options]\n\n"
              << "General Options:\n"
              << "  -m, --model PATH         Path to GGUF model file (required)\n"
              << "  -p, --prompt TEXT        Input prompt text\n"
              << "  -f, --file PATH          Read prompt from file\n"
              << "  -n, --predict N          Number of tokens to generate (default: 128)\n"
              << "  -c, --ctx-size N         Context size (default: 2048)\n"
              << "  -t, --threads N          Number of CPU threads (default: 4)\n"
              << "  -ngl, --n-gpu-layers N   Number of layers to offload to GPU (default: 0)\n"
              << "  -sm, --split-mode MODE   Multi-GPU split mode: none, layer (default), row\n"
              << "  -mg, --main-gpu N        Main GPU index (default: 0)\n"
              << "  -ts, --tensor-split S    Fraction of model to offload per GPU (e.g. '3,2')\n"
              << "  -fa, --flash-attn        Enable Flash Attention\n"
              << "  --no-mmap                Do not memory-map model file\n"
              << "  --mlock                  Lock model in RAM (prevent swap)\n"
              << "  --check-tensors          Validate tensor data integrity\n"
              << "  --numa MODE              NUMA optimization: distribute, isolate, numactl, mirror\n"
              << "  -s, --seed N             RNG seed (default: 42)\n\n"
              << "Model Adaptation & KV Quantization:\n"
              << "  --lora PATH              Apply LoRA adapter\n"
              << "  --lora-scaled PATH SCALE Apply LoRA adapter with user scale\n"
              << "  --control-vector PATH    Apply control vector file\n"
              << "  --cache-type-k TYPE      KV cache data type for K (f16, q8_0, q4_0, etc.)\n"
              << "  --cache-type-v TYPE      KV cache data type for V (f16, q8_0, q4_0, etc.)\n\n"
              << "Context & RoPE Options:\n"
              << "  --prompt-cache PATH      File to cache prompt KV state for instant startup\n"
              << "  --prompt-cache-ro        Open prompt cache read-only\n"
              << "  --prompt-cache-all       Cache both prompt and generated tokens\n"
              << "  --ctx-shift              Enable rolling context shifting on overflow (default: true)\n"
              << "  --no-ctx-shift           Disable context shifting (stop at max context)\n"
              << "  --n-keep N               Number of prompt tokens to preserve during shift (default: 0)\n"
              << "  --n-discard N            Number of tokens to discard during shift\n"
              << "  --rope-scaling TYPE      RoPE scaling type: none, linear, yarn\n"
              << "  --rope-freq-base FLOAT   RoPE base frequency\n"
              << "  --rope-freq-scale FLOAT  RoPE frequency scaling factor\n"
              << "  --yarn-orig-ctx N        YaRN original context size\n"
              << "  --yarn-ext-factor FLOAT  YaRN extrapolation factor\n"
              << "  --yarn-attn-factor FLOAT YaRN attention magnitude factor\n"
              << "  --yarn-beta-fast FLOAT   YaRN low correction dim\n"
              << "  --yarn-beta-slow FLOAT   YaRN high correction dim\n\n"
              << "Sampling & Reasoning Options:\n"
              << "  --temp FLOAT             Temperature (default: 0.8, 0 = greedy)\n"
              << "  --top-k N                Top-k sampling (default: 40)\n"
              << "  --top-p FLOAT            Top-p sampling (default: 0.95)\n"
              << "  --min-p FLOAT            Min-p sampling (default: 0.05)\n"
              << "  --top-n-sigma FLOAT      Top-n sigma sampling\n"
              << "  --adaptive-p-target F    Adaptive-p target probability\n"
              << "  --adaptive-p-decay F     Adaptive-p EMA decay (default: 0.9)\n"
              << "  -l, --logit-bias ID:BIAS Add bias to token ID (can repeat)\n"
              << "  --logprobs               Compute and output token logprobs\n"
              << "  --top-logprobs N         Output top N candidate logprobs per token\n"
              << "  --extract-reasoning      Extract <think> tags as reasoning content\n"
              << "  --reasoning-budget N     Maximum tokens allowed in reasoning block\n"
              << "  --repeat-penalty FLOAT   Repeat penalty (default: 1.0)\n"
              << "  --repeat-last-n N        Repeat last n tokens (default: 64)\n"
              << "  --presence-penalty FLOAT Presence penalty (default: 0.0)\n"
              << "  --frequency-penalty F    Frequency penalty (default: 0.0)\n"
              << "  --mirostat N             Mirostat (0=disabled, 1=v1, 2=v2)\n\n"
              << "Diffusion / Speculative Options:\n"
              << "  --mode MODE              Decoding mode: block (default), whole, ar\n"
              << "  --block N                Jacobi block size (default: 32)\n"
              << "  --whole-max N            Max canvas size for whole mode (default: 512)\n"
              << "  --init STRAT             Canvas init strategy: ngram (default), random, pad, draft\n"
              << "  -md, --model-draft PATH  Path to speculative draft model GGUF\n"
              << "  -ngld, --draft-gpu N     Number of layers to offload for draft model (default: 0)\n"
              << "  --ngram-n N              N-gram context length for init guesser (default: 3)\n"
              << "  --ngram-pool N           N-gram pool max capacity (default: 4096)\n"
              << "  --max-iters N            Maximum iterations cap (default: n_predict + 8)\n"
              << "  --rng-mode MODE          positional (default), llama-compat\n\n"
              << "Display & Output Options:\n"
              << "  --render STYLE           auto (default), live, progress, final\n"
              << "  --render-fps N           Live terminal render framerate (default: 30)\n"
              << "  --no-color               Disable ANSI color codes\n"
              << "  --stats                  Print JSON timing and token stats to stderr\n"
              << "  --tokens-out PATH        Dump generated token IDs to file (for test oracle)\n"
              << "  -cnv, --conversation     Run interactive chat REPL mode\n"
              << "  -sys, --system-prompt S  System prompt for chat mode\n"
              << "  --chat-template JINJA    Override model Jinja chat template\n"
              << "  -r, --reverse-prompt S   Stop string (can be specified multiple times)\n"
              << "  -v, --version            Show version\n"
              << "  -h, --help               Show this help message\n";
}

CliArgs parse_cli_args(int argc, char** argv) {
    CliArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];

        if (arg == "-v" || arg == "--version") {
            args.show_version = true;
            return args;
        }
        if (arg == "-h" || arg == "--help") {
            args.show_help = true;
            return args;
        }

        auto next_str = [&](const char* opt_name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: option " << opt_name << " requires an argument\n";
                exit(1);
            }
            return argv[++i];
        };

        auto next_int = [&](const char* opt_name) -> int {
            return std::stoi(next_str(opt_name));
        };

        auto next_float = [&](const char* opt_name) -> float {
            return std::stof(next_str(opt_name));
        };

        if (arg == "-m" || arg == "--model") {
            args.backend.model_path = next_str("-m/--model");
        } else if (arg == "-p" || arg == "--prompt") {
            args.prompt = next_str("-p/--prompt");
        } else if (arg == "-f" || arg == "--file") {
            args.prompt_file = next_str("-f/--file");
        } else if (arg == "-n" || arg == "--predict") {
            args.params.n_predict = next_int("-n/--predict");
        } else if (arg == "-c" || arg == "--ctx-size") {
            args.backend.n_ctx = next_int("-c/--ctx-size");
        } else if (arg == "-t" || arg == "--threads") {
            args.backend.n_threads = next_int("-t/--threads");
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            args.backend.n_gpu_layers = next_int("-ngl/--n-gpu-layers");
        } else if (arg == "-sm" || arg == "--split-mode") {
            std::string sm = next_str("-sm/--split-mode");
            if (sm == "none") args.backend.split_mode = 0;
            else if (sm == "row") args.backend.split_mode = 2;
            else args.backend.split_mode = 1;
        } else if (arg == "-mg" || arg == "--main-gpu") {
            args.backend.main_gpu = next_int("-mg/--main-gpu");
        } else if (arg == "-ts" || arg == "--tensor-split") {
            args.backend.tensor_split_str = next_str("-ts/--tensor-split");
        } else if (arg == "-fa" || arg == "--flash-attn") {
            args.backend.flash_attn = true;
        } else if (arg == "--no-mmap") {
            args.backend.use_mmap = false;
        } else if (arg == "--mlock") {
            args.backend.use_mlock = true;
        } else if (arg == "--check-tensors") {
            args.backend.check_tensors = true;
        } else if (arg == "--numa") {
            args.numa_mode = next_str("--numa");
        } else if (arg == "--lora") {
            args.backend.loras.push_back({next_str("--lora"), 1.0f});
        } else if (arg == "--lora-scaled") {
            std::string path = next_str("--lora-scaled path");
            float scale = next_float("--lora-scaled scale");
            args.backend.loras.push_back({path, scale});
        } else if (arg == "--control-vector") {
            args.backend.control_vector_path = next_str("--control-vector");
        } else if (arg == "--cache-type-k") {
            args.backend.cache_type_k = next_str("--cache-type-k");
        } else if (arg == "--cache-type-v") {
            args.backend.cache_type_v = next_str("--cache-type-v");
        } else if (arg == "--prompt-cache") {
            args.params.prompt_cache_path = next_str("--prompt-cache");
            args.backend.prompt_cache_path = args.params.prompt_cache_path;
        } else if (arg == "--prompt-cache-ro") {
            args.params.prompt_cache_ro = true;
            args.backend.prompt_cache_ro = true;
        } else if (arg == "--prompt-cache-all") {
            args.params.prompt_cache_all = true;
            args.backend.prompt_cache_all = true;
        } else if (arg == "--ctx-shift") {
            args.params.ctx_shift = true;
        } else if (arg == "--no-ctx-shift") {
            args.params.ctx_shift = false;
        } else if (arg == "--n-keep") {
            args.params.n_keep = next_int("--n-keep");
        } else if (arg == "--n-discard") {
            args.params.n_discard = next_int("--n-discard");
        } else if (arg == "--rope-scaling") {
            std::string rs = next_str("--rope-scaling");
            if (rs == "none") args.backend.rope_scaling_type = 0;
            else if (rs == "linear") args.backend.rope_scaling_type = 1;
            else if (rs == "yarn") args.backend.rope_scaling_type = 2;
        } else if (arg == "--rope-freq-base") {
            args.backend.rope_freq_base = next_float("--rope-freq-base");
        } else if (arg == "--rope-freq-scale") {
            args.backend.rope_freq_scale = next_float("--rope-freq-scale");
        } else if (arg == "--yarn-orig-ctx") {
            args.backend.yarn_orig_ctx = static_cast<uint32_t>(next_int("--yarn-orig-ctx"));
        } else if (arg == "--yarn-ext-factor") {
            args.backend.yarn_ext_factor = next_float("--yarn-ext-factor");
        } else if (arg == "--yarn-attn-factor") {
            args.backend.yarn_attn_factor = next_float("--yarn-attn-factor");
        } else if (arg == "--yarn-beta-fast") {
            args.backend.yarn_beta_fast = next_float("--yarn-beta-fast");
        } else if (arg == "--yarn-beta-slow") {
            args.backend.yarn_beta_slow = next_float("--yarn-beta-slow");
        } else if (arg == "--logprobs") {
            args.params.logprobs = true;
        } else if (arg == "--top-logprobs") {
            args.params.top_logprobs = next_int("--top-logprobs");
            args.params.logprobs = true;
        } else if (arg == "-l" || arg == "--logit-bias") {
            std::string lb_str = next_str("-l/--logit-bias");
            size_t colon = lb_str.find(':');
            if (colon != std::string::npos) {
                int32_t tok_id = std::stoi(lb_str.substr(0, colon));
                float bias = std::stof(lb_str.substr(colon + 1));
                args.params.sampling.logit_bias.push_back({tok_id, bias});
            }
        } else if (arg == "--top-n-sigma") {
            args.params.sampling.top_n_sigma = next_float("--top-n-sigma");
        } else if (arg == "--adaptive-p-target") {
            args.params.sampling.adaptive_p_target = next_float("--adaptive-p-target");
        } else if (arg == "--adaptive-p-decay") {
            args.params.sampling.adaptive_p_decay = next_float("--adaptive-p-decay");
        } else if (arg == "--extract-reasoning") {
            args.params.extract_reasoning = true;
        } else if (arg == "--reasoning-budget") {
            args.params.reasoning_budget = next_int("--reasoning-budget");
            args.params.extract_reasoning = true;
        } else if (arg == "-md" || arg == "--model-draft") {
            args.params.model_draft = next_str("-md/--model-draft");
            args.params.init_strategy = InitStrategy::DRAFT;
        } else if (arg == "-ngld" || arg == "--draft-gpu") {
            args.params.draft_n_gpu_layers = next_int("-ngld/--draft-gpu");
        } else if (arg == "-s" || arg == "--seed") {
            args.params.sampling.seed = static_cast<uint32_t>(next_int("-s/--seed"));
        } else if (arg == "--temp") {
            args.params.sampling.temp = next_float("--temp");
        } else if (arg == "--top-k") {
            args.params.sampling.top_k = next_int("--top-k");
        } else if (arg == "--top-p") {
            args.params.sampling.top_p = next_float("--top-p");
        } else if (arg == "--min-p") {
            args.params.sampling.min_p = next_float("--min-p");
        } else if (arg == "--repeat-penalty") {
            args.params.sampling.repeat_penalty = next_float("--repeat-penalty");
        } else if (arg == "--repeat-last-n") {
            args.params.sampling.repeat_last_n = next_int("--repeat-last-n");
        } else if (arg == "--presence-penalty") {
            args.params.sampling.presence_penalty = next_float("--presence-penalty");
        } else if (arg == "--frequency-penalty") {
            args.params.sampling.frequency_penalty = next_float("--frequency-penalty");
        } else if (arg == "--mirostat") {
            args.params.sampling.mirostat = next_int("--mirostat");
        } else if (arg == "--mode") {
            std::string mode_str = next_str("--mode");
            if (mode_str == "ar") args.params.mode = DecodeMode::AR;
            else if (mode_str == "whole" || mode_str == "jacobi-whole") args.params.mode = DecodeMode::WHOLE;
            else args.params.mode = DecodeMode::BLOCK;
        } else if (arg == "--block" || arg == "--block-size" || arg == "-b") {
            args.params.block_size = next_int("--block/--block-size");
        } else if (arg == "--whole-max") {
            args.params.whole_max = next_int("--whole-max");
        } else if (arg == "--init") {
            std::string s = next_str("--init");
            if (s == "random") args.params.init_strategy = InitStrategy::RANDOM;
            else if (s == "pad") args.params.init_strategy = InitStrategy::PAD;
            else if (s == "draft") args.params.init_strategy = InitStrategy::DRAFT;
            else args.params.init_strategy = InitStrategy::NGRAM;
        } else if (arg == "--ngram-n") {
            args.params.ngram_n = next_int("--ngram-n");
        } else if (arg == "--ngram-pool") {
            args.params.ngram_pool_max = next_int("--ngram-pool");
        } else if (arg == "--max-iters") {
            args.params.max_iters = next_int("--max-iters");
        } else if (arg == "--rng-mode") {
            std::string s = next_str("--rng-mode");
            if (s == "llama-compat") args.params.sampling.rng_mode = RngMode::LLAMA_COMPAT;
            else args.params.sampling.rng_mode = RngMode::POSITIONAL;
        } else if (arg == "--render") {
            std::string r = next_str("--render");
            if (r == "live") args.render_mode = RenderMode::LIVE;
            else if (r == "progress") args.render_mode = RenderMode::PROGRESS;
            else if (r == "final") args.render_mode = RenderMode::FINAL;
            else args.render_mode = RenderMode::AUTO;
        } else if (arg == "--render-fps") {
            args.render_fps = next_int("--render-fps");
        } else if (arg == "--no-color") {
            args.no_color = true;
        } else if (arg == "--stats") {
            args.show_stats = true;
        } else if (arg == "--tokens-out") {
            args.tokens_out_file = next_str("--tokens-out");
        } else if (arg == "-cnv" || arg == "--conversation" || arg == "--chat" || arg == "-i") {
            args.conversation = true;
        } else if (arg == "-sys" || arg == "--system-prompt") {
            args.system_prompt = next_str("-sys/--system-prompt");
        } else if (arg == "--chat-template") {
            args.custom_chat_template = next_str("--chat-template");
        } else if (arg == "-r" || arg == "--reverse-prompt" || arg == "--stop") {
            args.params.stop_strings.push_back(next_str("-r/--stop"));
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            exit(1);
        }
    }

    if (!args.prompt_file.empty()) {
        std::ifstream f(args.prompt_file);
        if (f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            args.prompt = ss.str();
        } else {
            std::cerr << "Error: could not open prompt file: " << args.prompt_file << "\n";
            exit(1);
        }
    }

    return args;
}

} // namespace diffusion
