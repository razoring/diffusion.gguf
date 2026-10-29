#include "llama.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct SimpleLlamaConfig {
    std::string model_path;
    std::string prompt;
    int32_t n_ctx = 32000;
    int32_t n_predict = 2048;
    float temp = 0.0f;
    bool json_output = true;
};

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage: %s -m <model> [-c <ctx>] [-temp <temp>] [-n <predict>] -p <prompt> [--json]\n", prog);
}

static SimpleLlamaConfig parse_args(int argc, char** argv) {
    SimpleLlamaConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_str = [&](const char* opt) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option %s\n", opt);
                std::exit(1);
            }
            return argv[++i];
        };
        auto next_int = [&](const char* opt) -> int32_t {
            return std::stoi(next_str(opt));
        };
        auto next_float = [&](const char* opt) -> float {
            return std::stof(next_str(opt));
        };

        if (arg == "-m" || arg == "--model") {
            cfg.model_path = next_str("-m/--model");
        } else if (arg == "-c" || arg == "-ctx" || arg == "--ctx" || arg == "--ctx-size") {
            cfg.n_ctx = next_int("-c");
        } else if (arg == "-temp" || arg == "--temp") {
            cfg.temp = next_float("-temp");
        } else if (arg == "-n" || arg == "--predict") {
            cfg.n_predict = next_int("-n");
        } else if (arg == "-p" || arg == "--prompt") {
            cfg.prompt = next_str("-p");
        } else if (arg == "--json") {
            cfg.json_output = true;
        }
    }
    return cfg;
}

int main(int argc, char** argv) {
    SimpleLlamaConfig cfg = parse_args(argc, argv);
    if (cfg.model_path.empty() || cfg.prompt.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;

    llama_model* model = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "Error: Failed to load model from %s\n", cfg.model_path.c_str());
        llama_backend_free();
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = cfg.n_ctx;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 512;
    cparams.swa_full = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "Error: Failed to create context with ctx_size=%d\n", cfg.n_ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Tokenize prompt
    std::vector<llama_token> prompt_tokens(cfg.prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, cfg.prompt.c_str(), (int32_t)cfg.prompt.size(), prompt_tokens.data(), (int32_t)prompt_tokens.size(), true, true);
    if (n_tokens < 0) {
        prompt_tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, cfg.prompt.c_str(), (int32_t)cfg.prompt.size(), prompt_tokens.data(), (int32_t)prompt_tokens.size(), true, true);
    }
    prompt_tokens.resize(n_tokens);

    // Sampler setup
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    if (cfg.temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
    }

    // Prefill prompt
    llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
    auto t_prefill_start = std::chrono::steady_clock::now();

    for (int p = 0; p < n_tokens; p += cparams.n_batch) {
        int cur_n = std::min<int>(cparams.n_batch, n_tokens - p);
        bool is_last = (p + cur_n == n_tokens);
        for (int i = 0; i < cur_n; ++i) {
            batch.token[i] = prompt_tokens[p + i];
            batch.pos[i] = p + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (is_last && i == cur_n - 1) ? 1 : 0;
        }
        batch.n_tokens = cur_n;
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "Error: llama_decode failed during prefill\n");
            llama_batch_free(batch);
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    auto t_prefill_end = std::chrono::steady_clock::now();
    double prefill_time_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    double prefill_tok_per_s = (n_tokens > 0 && prefill_time_ms > 0.0) ? (n_tokens / (prefill_time_ms / 1000.0)) : 0.0;

    for (llama_token tok : prompt_tokens) {
        llama_sampler_accept(smpl, tok);
    }

    // Autoregressive single-token generation loop
    std::vector<llama_token> generated_tokens;
    std::string generated_text;
    generated_tokens.reserve(cfg.n_predict);

    int n_past = n_tokens;
    auto t_gen_start = std::chrono::steady_clock::now();

    for (int i = 0; i < cfg.n_predict; ++i) {
        if (n_past >= cfg.n_ctx) break;

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        generated_tokens.push_back(new_token_id);

        char buf[256];
        int n_piece = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n_piece > 0) {
            generated_text.append(buf, n_piece);
        }

        batch.token[0] = new_token_id;
        batch.pos[0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;

        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "Error: llama_decode failed at pos=%d\n", n_past);
            break;
        }
        n_past++;
    }

    auto t_gen_end = std::chrono::steady_clock::now();
    double gen_time_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
    double gen_tok_per_s = (generated_tokens.size() > 0 && gen_time_ms > 0.0) ? (generated_tokens.size() / (gen_time_ms / 1000.0)) : 0.0;

    if (cfg.json_output) {
        std::printf("{\n");
        std::printf("  \"engine\": \"llama.cpp\",\n");
        std::printf("  \"prompt_tokens\": %d,\n", n_tokens);
        std::printf("  \"completion_tokens\": %zu,\n", generated_tokens.size());
        std::printf("  \"prefill_time_ms\": %.2f,\n", prefill_time_ms);
        std::printf("  \"prefill_tok_per_s\": %.2f,\n", prefill_tok_per_s);
        std::printf("  \"decode_time_ms\": %.2f,\n", gen_time_ms);
        std::printf("  \"decode_tok_per_s\": %.2f\n", gen_tok_per_s);
        std::printf("}\n");
    } else {
        std::printf("\n================================================================================\n");
        std::printf(" [Benchmark] llama.cpp Pure Causal Baseline\n");
        std::printf("================================================================================\n");
        std::printf(" Prompt tokens:     %d (%.2f tok/s in %.2f ms)\n", n_tokens, prefill_tok_per_s, prefill_time_ms);
        std::printf(" Completion tokens: %zu (%.2f tok/s in %.2f ms)\n", generated_tokens.size(), gen_tok_per_s, gen_time_ms);
        std::printf("================================================================================\n");
        std::printf("%.*s\n", (int)std::min<size_t>(generated_text.size(), 200), generated_text.c_str());
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

