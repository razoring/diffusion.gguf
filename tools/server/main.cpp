#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "engine.h"
#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>
#include "json-schema-to-grammar.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace diffusion {

struct ServerConfig {
    BackendConfig backend;
    GenParams default_params;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
    int n_parallel = 1;
    std::string numa_mode;
};

struct ServerSlot {
    int id = 0;
    std::atomic<bool> is_processing{false};
    std::mutex mtx;
    std::string prompt;
    int n_predict = 0;
    int tokens_generated = 0;
    std::chrono::steady_clock::time_point start_time;
};

class SlotManager {
public:
    explicit SlotManager(int n_slots) : n_slots_(n_slots > 0 ? n_slots : 1) {
        slots_.reserve(n_slots_);
        for (int i = 0; i < n_slots_; ++i) {
            auto slot = std::make_unique<ServerSlot>();
            slot->id = i;
            slots_.push_back(std::move(slot));
        }
    }

    ServerSlot* acquire_slot(int timeout_ms = 10000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            for (auto& s : slots_) {
                bool expected = false;
                if (s->is_processing.compare_exchange_strong(expected, true)) {
                    s->tokens_generated = 0;
                    s->start_time = std::chrono::steady_clock::now();
                    return s.get();
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return nullptr;
    }

    void release_slot(ServerSlot* slot) {
        if (slot) {
            slot->is_processing.store(false);
        }
    }

    json get_slots_status() const {
        json arr = json::array();
        for (const auto& s : slots_) {
            bool proc = s->is_processing.load();
            double duration_s = 0.0;
            if (proc) {
                duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s->start_time).count();
            }
            arr.push_back({
                {"id", s->id},
                {"state", proc ? "processing" : "idle"},
                {"prompt", proc ? s->prompt : ""},
                {"n_predict", proc ? s->n_predict : 0},
                {"tokens_generated", proc ? s->tokens_generated : 0},
                {"duration_s", duration_s}
            });
        }
        return arr;
    }

    int active_slots_count() const {
        int cnt = 0;
        for (const auto& s : slots_) {
            if (s->is_processing.load()) cnt++;
        }
        return cnt;
    }

    int total_slots() const { return n_slots_; }

private:
    int n_slots_ = 1;
    std::vector<std::unique_ptr<ServerSlot>> slots_;
};

struct ServerMetrics {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> total_prompt_tokens{0};
    std::atomic<uint64_t> total_completion_tokens{0};
};

static void print_server_help() {
    std::cout << "Usage: diffusion-server -m <model.gguf> [options]\n\n"
              << "Server Options:\n"
              << "  -m, --model PATH         Path to GGUF model file (required)\n"
              << "  --host HOST              Host address to listen on (default: 127.0.0.1)\n"
              << "  --port PORT              Port to listen on (default: 8080)\n"
              << "  -np, --parallel N        Number of concurrent slots / sequences (default: 1)\n"
              << "  --api-key KEY            Optional Bearer API key authentication\n"
              << "  -c, --ctx-size N         Context size (default: 2048)\n"
              << "  -t, --threads N          CPU threads (default: 4)\n"
              << "  -ngl, --n-gpu-layers N   GPU offload layers (default: 0)\n"
              << "  -fa, --flash-attn        Enable Flash Attention\n"
              << "  --no-mmap                Disable mmap\n"
              << "  --mlock                  Force lock in RAM\n"
              << "  --numa MODE              NUMA optimization\n"
              << "  --cache-type-k TYPE      KV cache data type for K (f16, q8_0, q4_0)\n"
              << "  --cache-type-v TYPE      KV cache data type for V (f16, q8_0, q4_0)\n"
              << "  --lora PATH              Load LoRA adapter\n\n"
              << "Default Generation Options:\n"
              << "  --mode MODE              Decoding mode: block (default), whole, ar\n"
              << "  --block N                Jacobi block size (default: 32)\n"
              << "  --whole-max N            Max canvas size for whole mode (default: 512)\n"
              << "  --init STRAT             Initial guess: ngram (default), random, pad\n"
              << "  --temp FLOAT             Temperature (default: 0.8)\n"
              << "  --top-p FLOAT            Top-p (default: 0.95)\n"
              << "  --top-k N                Top-k (default: 40)\n"
              << "  -v, --version            Show version\n"
              << "  -h, --help               Show help message\n";
}

static ServerConfig parse_server_args(int argc, char** argv) {
    ServerConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-v" || arg == "--version") {
            std::cout << "diffusion-server (diffusion.gguf) version 0.1.0\n";
            std::exit(0);
        }
        if (arg == "-h" || arg == "--help") {
            print_server_help();
            std::exit(0);
        }

        auto next_str = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << name << " requires an argument\n";
                std::exit(1);
            }
            return argv[++i];
        };
        auto next_int = [&](const char* name) -> int {
            return std::stoi(next_str(name));
        };
        auto next_float = [&](const char* name) -> float {
            return std::stof(next_str(name));
        };

        if (arg == "-m" || arg == "--model") cfg.backend.model_path = next_str("-m/--model");
        else if (arg == "--host") cfg.host = next_str("--host");
        else if (arg == "--port") cfg.port = next_int("--port");
        else if (arg == "-np" || arg == "--parallel") cfg.n_parallel = next_int("-np/--parallel");
        else if (arg == "--api-key") cfg.api_key = next_str("--api-key");
        else if (arg == "-c" || arg == "--ctx-size") cfg.backend.n_ctx = next_int("-c/--ctx-size");
        else if (arg == "-t" || arg == "--threads") cfg.backend.n_threads = next_int("-t/--threads");
        else if (arg == "-ngl" || arg == "--n-gpu-layers") cfg.backend.n_gpu_layers = next_int("-ngl/--n-gpu-layers");
        else if (arg == "-fa" || arg == "--flash-attn") cfg.backend.flash_attn = true;
        else if (arg == "--no-mmap") cfg.backend.use_mmap = false;
        else if (arg == "--mlock") cfg.backend.use_mlock = true;
        else if (arg == "--numa") cfg.numa_mode = next_str("--numa");
        else if (arg == "--cache-type-k") cfg.backend.cache_type_k = next_str("--cache-type-k");
        else if (arg == "--cache-type-v") cfg.backend.cache_type_v = next_str("--cache-type-v");
        else if (arg == "--lora") cfg.backend.loras.push_back({next_str("--lora"), 1.0f});
        else if (arg == "--mode") {
            std::string m = next_str("--mode");
            if (m == "ar") cfg.default_params.mode = DecodeMode::AR;
            else if (m == "whole") cfg.default_params.mode = DecodeMode::WHOLE;
            else cfg.default_params.mode = DecodeMode::BLOCK;
        } else if (arg == "--block") cfg.default_params.block_size = next_int("--block");
        else if (arg == "--whole-max") cfg.default_params.whole_max = next_int("--whole-max");
        else if (arg == "--init") {
            std::string s = next_str("--init");
            if (s == "random") cfg.default_params.init_strategy = InitStrategy::RANDOM;
            else if (s == "pad") cfg.default_params.init_strategy = InitStrategy::PAD;
            else cfg.default_params.init_strategy = InitStrategy::NGRAM;
        } else if (arg == "--temp") cfg.default_params.sampling.temp = next_float("--temp");
        else if (arg == "--top-p") cfg.default_params.sampling.top_p = next_float("--top-p");
        else if (arg == "--top-k") cfg.default_params.sampling.top_k = next_int("--top-k");
    }

    cfg.backend.n_seq_max = cfg.n_parallel;
    return cfg;
}

static bool check_auth(const httplib::Request& req, const std::string& api_key, httplib::Response& res) {
    if (api_key.empty()) return true;
    std::string auth = req.get_header_value("Authorization");
    std::string expected = "Bearer " + api_key;
    if (auth != expected) {
        json err = {
            {"error", {
                {"message", "Invalid API key"},
                {"type", "invalid_request_error"},
                {"code", "invalid_api_key"}
            }}
        };
        res.status = 401;
        res.set_content(err.dump(), "application/json");
        return false;
    }
    return true;
}

static size_t compute_safe_prefix_len(const std::string& text, const std::vector<std::string>& stop_strings) {
    size_t holdback = 0;
    for (const auto& stop : stop_strings) {
        if (stop.empty()) continue;
        size_t max_check = std::min(text.size(), stop.size() - 1);
        for (size_t len = 1; len <= max_check; ++len) {
            std::string_view suffix(text.data() + text.size() - len, len);
            std::string_view prefix(stop.data(), len);
            if (suffix == prefix) {
                holdback = std::max(holdback, len);
            }
        }
    }
    return text.size() >= holdback ? (text.size() - holdback) : 0;
}

static json format_logprobs_json(const std::vector<TokenLogprobs>& tlp) {
    json content = json::array();
    for (const auto& item : tlp) {
        json top_arr = json::array();
        for (const auto& top : item.top_logprobs) {
            std::vector<uint8_t> bytes(top.piece.begin(), top.piece.end());
            top_arr.push_back({
                {"token", top.piece},
                {"logprob", top.logprob},
                {"bytes", bytes}
            });
        }
        std::vector<uint8_t> item_bytes(item.piece.begin(), item.piece.end());
        content.push_back({
            {"token", item.piece},
            {"logprob", item.logprob},
            {"bytes", item_bytes},
            {"top_logprobs", top_arr}
        });
    }
    return json{{"content", content}};
}

static GenParams parse_request_params(const json& body, const GenParams& default_params) {
    GenParams params = default_params;

    if (body.contains("max_tokens")) params.n_predict = body["max_tokens"].get<int>();
    else if (body.contains("n_predict")) params.n_predict = body["n_predict"].get<int>();

    if (body.contains("temperature")) params.sampling.temp = body["temperature"].get<float>();
    if (body.contains("top_p")) params.sampling.top_p = body["top_p"].get<float>();
    if (body.contains("top_k")) params.sampling.top_k = body["top_k"].get<int>();
    if (body.contains("min_p")) params.sampling.min_p = body["min_p"].get<float>();
    if (body.contains("typical_p")) params.sampling.typical_p = body["typical_p"].get<float>();
    if (body.contains("presence_penalty")) params.sampling.presence_penalty = body["presence_penalty"].get<float>();
    if (body.contains("frequency_penalty")) params.sampling.frequency_penalty = body["frequency_penalty"].get<float>();
    if (body.contains("repeat_penalty")) params.sampling.repeat_penalty = body["repeat_penalty"].get<float>();
    if (body.contains("repeat_last_n")) params.sampling.repeat_last_n = body["repeat_last_n"].get<int>();
    if (body.contains("seed")) params.sampling.seed = body["seed"].get<uint32_t>();

    // Logprobs
    if (body.contains("logprobs")) {
        if (body["logprobs"].is_boolean()) {
            params.logprobs = body["logprobs"].get<bool>();
        } else if (body["logprobs"].is_number_integer()) {
            params.logprobs = true;
            params.top_logprobs = body["logprobs"].get<int>();
        }
    }
    if (body.contains("top_logprobs")) {
        params.top_logprobs = body["top_logprobs"].get<int>();
        params.logprobs = true;
    }

    // Logit bias
    if (body.contains("logit_bias")) {
        if (body["logit_bias"].is_object()) {
            for (auto& el : body["logit_bias"].items()) {
                try {
                    int32_t tid = std::stoi(el.key());
                    float bias = el.value().get<float>();
                    params.sampling.logit_bias.push_back({tid, bias});
                } catch (...) {}
            }
        } else if (body["logit_bias"].is_array()) {
            for (const auto& item : body["logit_bias"]) {
                if (item.is_array() && item.size() >= 2) {
                    params.sampling.logit_bias.push_back({item[0].get<int32_t>(), item[1].get<float>()});
                }
            }
        }
    }

    // Reasoning budget
    if (body.contains("reasoning_budget")) {
        params.reasoning_budget = body["reasoning_budget"].get<int>();
        params.extract_reasoning = true;
    }
    if (body.value("extract_reasoning", false)) {
        params.extract_reasoning = true;
    }

    // Stop strings
    if (body.contains("stop")) {
        if (body["stop"].is_string()) {
            params.stop_strings.push_back(body["stop"].get<std::string>());
        } else if (body["stop"].is_array()) {
            for (const auto& s : body["stop"]) {
                if (s.is_string()) params.stop_strings.push_back(s.get<std::string>());
            }
        }
    }

    // Response format (Grammar / JSON Schema)
    if (body.contains("response_format") && body["response_format"].is_object()) {
        auto rf = body["response_format"];
        std::string type = rf.value("type", "");
        if (type == "json_object") {
            json schema = {{"type", "object"}};
            params.sampling.grammar = json_schema_to_grammar(schema);
        } else if (type == "json_schema" && rf.contains("json_schema") && rf["json_schema"].contains("schema")) {
            params.sampling.grammar = json_schema_to_grammar(rf["json_schema"]["schema"]);
        }
    }

    // Diffusion extension object
    if (body.contains("diffusion") && body["diffusion"].is_object()) {
        auto diff = body["diffusion"];
        std::string m = diff.value("mode", "");
        if (m == "ar") params.mode = DecodeMode::AR;
        else if (m == "whole") params.mode = DecodeMode::WHOLE;
        else if (m == "block") params.mode = DecodeMode::BLOCK;

        if (diff.contains("block_size")) params.block_size = diff["block_size"].get<int>();
        if (diff.contains("whole_max")) params.whole_max = diff["whole_max"].get<int>();
        if (diff.contains("max_iters")) params.max_iters = diff["max_iters"].get<int>();
        if (diff.contains("init")) {
            std::string init = diff["init"].get<std::string>();
            if (init == "random") params.init_strategy = InitStrategy::RANDOM;
            else if (init == "pad") params.init_strategy = InitStrategy::PAD;
            else params.init_strategy = InitStrategy::NGRAM;
        }
    }

    return params;
}

} // namespace diffusion

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    diffusion::ServerConfig config = diffusion::parse_server_args(argc, argv);

    if (config.backend.model_path.empty()) {
        std::cerr << "Error: --model PATH is required.\n";
        diffusion::print_server_help();
        return 1;
    }

    if (!config.numa_mode.empty()) {
        ggml_numa_strategy strategy = GGML_NUMA_STRATEGY_DISTRIBUTE;
        if (config.numa_mode == "isolate") strategy = GGML_NUMA_STRATEGY_ISOLATE;
        else if (config.numa_mode == "numactl") strategy = GGML_NUMA_STRATEGY_NUMACTL;
        else if (config.numa_mode == "mirror") strategy = GGML_NUMA_STRATEGY_MIRROR;
        ggml_numa_init(strategy);
    }

    std::unique_ptr<diffusion::Engine> engine;
    try {
        engine = std::make_unique<diffusion::Engine>(config.backend);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize engine: " << e.what() << "\n";
        return 1;
    }

    httplib::Server svr;
    diffusion::SlotManager slot_manager(config.n_parallel);
    diffusion::ServerMetrics metrics;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // Model list
    svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;
        json resp = {
            {"object", "list"},
            {"data", json::array({
                {
                    {"id", "diffusion-model"},
                    {"object", "model"},
                    {"created", 1700000000},
                    {"owned_by", "diffusion.gguf"}
                }
            })}
        };
        res.set_content(resp.dump(), "application/json");
    });

    // Slots status (GET /slots)
    svr.Get("/slots", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;
        res.set_content(slot_manager.get_slots_status().dump(), "application/json");
    });

    // Model properties (GET /props)
    svr.Get("/props", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;
        json p = {
            {"default_generation_settings", {
                {"n_ctx", engine->backend().n_ctx()},
                {"n_vocab", engine->backend().n_vocab()},
                {"n_embd", engine->backend().n_embd()},
                {"n_parallel", slot_manager.total_slots()}
            }},
            {"total_slots", slot_manager.total_slots()}
        };
        res.set_content(p.dump(), "application/json");
    });

    // Metrics (GET /metrics)
    svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
        std::string m = "# HELP diffusion_requests_total Total number of HTTP requests processed\n"
                        "# TYPE diffusion_requests_total counter\n"
                        "diffusion_requests_total " + std::to_string(metrics.total_requests.load()) + "\n"
                        "# HELP diffusion_prompt_tokens_total Total prompt tokens evaluated\n"
                        "# TYPE diffusion_prompt_tokens_total counter\n"
                        "diffusion_prompt_tokens_total " + std::to_string(metrics.total_prompt_tokens.load()) + "\n"
                        "# HELP diffusion_completion_tokens_total Total completion tokens generated\n"
                        "# TYPE diffusion_completion_tokens_total counter\n"
                        "diffusion_completion_tokens_total " + std::to_string(metrics.total_completion_tokens.load()) + "\n"
                        "# HELP diffusion_slots_active Number of slots currently processing\n"
                        "# TYPE diffusion_slots_active gauge\n"
                        "diffusion_slots_active " + std::to_string(slot_manager.active_slots_count()) + "\n"
                        "# HELP diffusion_slots_total Total number of concurrency slots\n"
                        "# TYPE diffusion_slots_total gauge\n"
                        "diffusion_slots_total " + std::to_string(slot_manager.total_slots()) + "\n";
        res.set_content(m, "text/plain; version=0.0.4");
    });

    // Tokenize endpoint (POST /tokenize)
    svr.Post("/tokenize", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content(R"({"error":{"message":"Invalid JSON"}})", "application/json"); return;
        }
        std::string content = body.value("content", body.value("text", ""));
        bool add_special = body.value("add_special", false);
        auto toks = engine->tokenizer().tokenize(content, add_special, true);
        json out = {{"tokens", toks}};
        res.set_content(out.dump(), "application/json");
    });

    // Detokenize endpoint (POST /detokenize)
    svr.Post("/detokenize", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content(R"({"error":{"message":"Invalid JSON"}})", "application/json"); return;
        }
        std::string text;
        if (body.contains("tokens") && body["tokens"].is_array()) {
            for (const auto& tok : body["tokens"]) {
                text += engine->tokenizer().piece(tok.get<int32_t>());
            }
        }
        json out = {{"content", text}};
        res.set_content(out.dump(), "application/json");
    });

    // LoRA Adapters (GET /lora-adapters, POST /lora-adapters)
    svr.Get("/lora-adapters", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;
        auto loras = engine->backend().get_lora_adapters();
        json arr = json::array();
        for (size_t i = 0; i < loras.size(); ++i) {
            arr.push_back({{"id", i}, {"path", loras[i].first}, {"scale", loras[i].second}});
        }
        res.set_content(arr.dump(), "application/json");
    });

    svr.Post("/lora-adapters", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content(R"({"error":{"message":"Invalid JSON"}})", "application/json"); return;
        }
        std::vector<std::pair<std::string, float>> to_apply;
        if (body.is_array()) {
            for (const auto& el : body) {
                to_apply.push_back({el.value("path", ""), el.value("scale", 1.0f)});
            }
        } else if (body.is_object()) {
            to_apply.push_back({body.value("path", ""), body.value("scale", 1.0f)});
        }
        bool ok = engine->backend().apply_lora_adapters(to_apply);
        res.set_content(ok ? R"({"status":"ok"})" : R"({"error":"Failed to apply adapters"})", "application/json");
    });

    // Embeddings endpoint (POST /v1/embeddings)
    svr.Post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content(R"({"error":{"message":"Invalid JSON"}})", "application/json"); return;
        }

        std::vector<std::string> inputs;
        if (body.contains("input")) {
            if (body["input"].is_string()) inputs.push_back(body["input"].get<std::string>());
            else if (body["input"].is_array()) {
                for (const auto& item : body["input"]) {
                    if (item.is_string()) inputs.push_back(item.get<std::string>());
                }
            }
        }
        if (inputs.empty()) inputs.push_back("");

        auto* slot = slot_manager.acquire_slot(10000);
        if (!slot) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"All slots busy, try again later"}})", "application/json");
            return;
        }

        json data_arr = json::array();
        int total_prompt_tokens = 0;

        engine->backend().set_embeddings(true);
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            auto toks = engine->tokenizer().tokenize(inputs[idx], true, true);
            total_prompt_tokens += static_cast<int>(toks.size());

            engine->backend().kv_clear(slot->id);
            if (!toks.empty()) {
                engine->backend().eval(toks.data(), static_cast<int>(toks.size()), 0, false, slot->id);
            }
            auto embd = engine->backend().get_embeddings_seq(slot->id);

            data_arr.push_back({
                {"object", "embedding"},
                {"index", idx},
                {"embedding", embd}
            });
        }
        engine->backend().set_embeddings(false);

        engine->backend().kv_clear(slot->id);
        slot_manager.release_slot(slot);

        metrics.total_requests++;
        metrics.total_prompt_tokens += total_prompt_tokens;

        json resp = {
            {"object", "list"},
            {"data", data_arr},
            {"model", "diffusion-model"},
            {"usage", {
                {"prompt_tokens", total_prompt_tokens},
                {"total_tokens", total_prompt_tokens}
            }}
        };
        res.set_content(resp.dump(), "application/json");
    });

    // Infill endpoint (POST /infill)
    svr.Post("/infill", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content(R"({"error":{"message":"Invalid JSON"}})", "application/json"); return;
        }

        std::string prefix = body.value("input_prefix", "");
        std::string suffix = body.value("input_suffix", "");
        std::string prompt = body.value("prompt", "");
        if (prompt.empty()) {
            prompt = "<fim_prefix>" + prefix + "<fim_suffix>" + suffix + "<fim_middle>";
        }

        auto* slot = slot_manager.acquire_slot(10000);
        if (!slot) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"All slots busy, try again later"}})", "application/json");
            return;
        }

        diffusion::GenParams params = diffusion::parse_request_params(body, config.default_params);
        params.seq_id = slot->id;
        params.sampling.infill = true;

        auto result = engine->generate(prompt, params);
        engine->backend().kv_clear(slot->id);
        slot_manager.release_slot(slot);

        metrics.total_requests++;
        metrics.total_completion_tokens += result.tokens.size();

        json resp = {
            {"content", result.text},
            {"tokens", result.tokens},
            {"stop", true}
        };
        res.set_content(resp.dump(), "application/json");
    });

    // POST /v1/completions
    svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"Invalid JSON body"}})", "application/json");
            return;
        }

        std::string prompt;
        if (body.contains("prompt")) {
            if (body["prompt"].is_string()) {
                prompt = body["prompt"].get<std::string>();
            } else if (body["prompt"].is_array() && !body["prompt"].empty()) {
                prompt = body["prompt"][0].get<std::string>();
            }
        }
        if (prompt.empty()) prompt = "Once upon a time";

        diffusion::GenParams params = diffusion::parse_request_params(body, config.default_params);
        bool stream = body.value("stream", false);

        auto* slot = slot_manager.acquire_slot(10000);
        if (!slot) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"All concurrency slots busy"}})", "application/json");
            return;
        }
        params.seq_id = slot->id;
        slot->prompt = prompt;
        slot->n_predict = params.n_predict;

        std::string id = "cmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        uint64_t created = static_cast<uint64_t>(std::time(nullptr));
        metrics.total_requests++;

        if (!stream) {
            auto prompt_tokens = engine->tokenizer().tokenize(prompt, true, true);
            diffusion::GenResult result = engine->generate(prompt, params);
            engine->backend().kv_clear(slot->id);
            slot_manager.release_slot(slot);

            metrics.total_prompt_tokens += prompt_tokens.size();
            metrics.total_completion_tokens += result.tokens.size();

            std::string finish_reason = (result.stop_reason == diffusion::StopReason::EOG) ? "stop"
                : (result.stop_reason == diffusion::StopReason::MAX_TOKENS ? "length" : "stop");

            json choices_obj = {
                {"text", result.text},
                {"index", 0},
                {"finish_reason", finish_reason}
            };
            if (params.logprobs) {
                choices_obj["logprobs"] = diffusion::format_logprobs_json(result.token_logprobs);
            } else {
                choices_obj["logprobs"] = nullptr;
            }

            json resp = {
                {"id", id},
                {"object", "text_completion"},
                {"created", created},
                {"model", "diffusion-model"},
                {"choices", json::array({choices_obj})},
                {"usage", {
                    {"prompt_tokens", prompt_tokens.size()},
                    {"completion_tokens", result.tokens.size()},
                    {"total_tokens", prompt_tokens.size() + result.tokens.size()},
                    {"diffusion_iterations", result.iterations},
                    {"tokens_per_iteration", result.tok_per_iter}
                }}
            };
            res.set_content(resp.dump(), "application/json");
        } else {
            // Streaming mode via SSE
            res.set_chunked_content_provider(
                "text/event-stream",
                [&, prompt, params, id, created, slot](size_t, httplib::DataSink& sink) {
                    std::string pending_text;

                    auto callback = [&](const diffusion::StepEvent& ev) -> bool {
                        if (!sink.is_writable()) return false;

                        for (int32_t tok : ev.committed_new) {
                            pending_text += engine->tokenizer().piece(tok);
                        }
                        slot->tokens_generated = ev.committed_upto;

                        if (!ev.done) {
                            size_t safe_len = diffusion::compute_safe_prefix_len(pending_text, params.stop_strings);
                            if (safe_len > 0) {
                                std::string delta_text = pending_text.substr(0, safe_len);
                                pending_text.erase(0, safe_len);

                                json chunk = {
                                    {"id", id},
                                    {"object", "text_completion"},
                                    {"created", created},
                                    {"model", "diffusion-model"},
                                    {"choices", json::array({
                                        {
                                            {"text", delta_text},
                                            {"index", 0},
                                            {"logprobs", nullptr},
                                            {"finish_reason", nullptr}
                                        }
                                    })}
                                };
                                std::string sse = "data: " + chunk.dump() + "\n\n";
                                sink.write(sse.data(), sse.size());
                            }
                        } else {
                            // Final flush
                            for (const auto& stop : params.stop_strings) {
                                if (!stop.empty() && pending_text.size() >= stop.size()) {
                                    if (pending_text.compare(pending_text.size() - stop.size(), stop.size(), stop) == 0) {
                                        pending_text.resize(pending_text.size() - stop.size());
                                        break;
                                    }
                                }
                            }
                            if (!pending_text.empty()) {
                                json chunk = {
                                    {"id", id},
                                    {"object", "text_completion"},
                                    {"created", created},
                                    {"model", "diffusion-model"},
                                    {"choices", json::array({
                                        {
                                            {"text", pending_text},
                                            {"index", 0},
                                            {"logprobs", nullptr},
                                            {"finish_reason", nullptr}
                                        }
                                    })}
                                };
                                std::string sse = "data: " + chunk.dump() + "\n\n";
                                sink.write(sse.data(), sse.size());
                            }
                        }
                        return sink.is_writable();
                    };

                    diffusion::GenResult result = engine->generate(prompt, params, callback);
                    engine->backend().kv_clear(slot->id);
                    slot_manager.release_slot(slot);

                    metrics.total_completion_tokens += result.tokens.size();

                    std::string finish_reason = (result.stop_reason == diffusion::StopReason::EOG) ? "stop"
                        : (result.stop_reason == diffusion::StopReason::MAX_TOKENS ? "length" : "stop");

                    json final_chunk = {
                        {"id", id},
                        {"object", "text_completion"},
                        {"created", created},
                        {"model", "diffusion-model"},
                        {"choices", json::array({
                            {
                                {"text", ""},
                                {"index", 0},
                                {"logprobs", nullptr},
                                {"finish_reason", finish_reason}
                            }
                        })}
                    };
                    std::string sse = "data: " + final_chunk.dump() + "\n\ndata: [DONE]\n\n";
                    sink.write(sse.data(), sse.size());
                    sink.done();
                    return false;
                }
            );
        }
    });

    // POST /v1/chat/completions
    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"Invalid JSON body"}})", "application/json");
            return;
        }

        std::vector<diffusion::ChatMsg> msgs;
        if (body.contains("messages") && body["messages"].is_array()) {
            for (const auto& m : body["messages"]) {
                if (m.is_object() && m.contains("role") && m.contains("content")) {
                    msgs.push_back(diffusion::ChatMsg{
                        m["role"].get<std::string>(),
                        m["content"].get<std::string>()
                    });
                }
            }
        }
        if (msgs.empty()) {
            msgs.push_back(diffusion::ChatMsg{"user", "Hello"});
        }

        std::string prompt = engine->chat_formatter().apply(msgs, true);
        diffusion::GenParams params = diffusion::parse_request_params(body, config.default_params);
        bool stream = body.value("stream", false);

        auto* slot = slot_manager.acquire_slot(10000);
        if (!slot) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"All concurrency slots busy"}})", "application/json");
            return;
        }
        params.seq_id = slot->id;
        slot->prompt = prompt;
        slot->n_predict = params.n_predict;

        std::string id = "chatcmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        uint64_t created = static_cast<uint64_t>(std::time(nullptr));
        metrics.total_requests++;

        if (!stream) {
            auto prompt_tokens = engine->tokenizer().tokenize(prompt, true, true);
            diffusion::GenResult result = engine->generate(prompt, params);
            engine->backend().kv_clear(slot->id);
            slot_manager.release_slot(slot);

            metrics.total_prompt_tokens += prompt_tokens.size();
            metrics.total_completion_tokens += result.tokens.size();

            std::string finish_reason = (result.stop_reason == diffusion::StopReason::EOG) ? "stop"
                : (result.stop_reason == diffusion::StopReason::MAX_TOKENS ? "length" : "stop");

            json msg_obj = {
                {"role", "assistant"},
                {"content", result.text}
            };
            if (!result.reasoning_content.empty()) {
                msg_obj["reasoning_content"] = result.reasoning_content;
            }

            json choice = {
                {"index", 0},
                {"message", msg_obj},
                {"finish_reason", finish_reason}
            };
            if (params.logprobs) {
                choice["logprobs"] = diffusion::format_logprobs_json(result.token_logprobs);
            }

            json resp = {
                {"id", id},
                {"object", "chat.completion"},
                {"created", created},
                {"model", "diffusion-model"},
                {"choices", json::array({choice})},
                {"usage", {
                    {"prompt_tokens", prompt_tokens.size()},
                    {"completion_tokens", result.tokens.size()},
                    {"total_tokens", prompt_tokens.size() + result.tokens.size()},
                    {"diffusion_iterations", result.iterations},
                    {"tokens_per_iteration", result.tok_per_iter}
                }}
            };
            res.set_content(resp.dump(), "application/json");
        } else {
            // Streaming mode via SSE
            res.set_chunked_content_provider(
                "text/event-stream",
                [&, prompt, params, id, created, slot](size_t, httplib::DataSink& sink) {
                    // Initial role announcement chunk
                    json init_chunk = {
                        {"id", id},
                        {"object", "chat.completion.chunk"},
                        {"created", created},
                        {"model", "diffusion-model"},
                        {"choices", json::array({
                            {
                                {"index", 0},
                                {"delta", {{"role", "assistant"}, {"content", ""}}},
                                {"finish_reason", nullptr}
                            }
                        })}
                    };
                    std::string init_sse = "data: " + init_chunk.dump() + "\n\n";
                    sink.write(init_sse.data(), init_sse.size());

                    std::string pending_text;
                    bool in_thinking = false;

                    auto callback = [&](const diffusion::StepEvent& ev) -> bool {
                        if (!sink.is_writable()) return false;

                        for (int32_t tok : ev.committed_new) {
                            pending_text += engine->tokenizer().piece(tok);
                        }
                        slot->tokens_generated = ev.committed_upto;

                        if (!ev.done) {
                            size_t safe_len = diffusion::compute_safe_prefix_len(pending_text, params.stop_strings);
                            if (safe_len > 0) {
                                std::string delta_text = pending_text.substr(0, safe_len);
                                pending_text.erase(0, safe_len);

                                // Check reasoning tags
                                json delta_obj = json::object();
                                size_t t_start = delta_text.find("<think>");
                                size_t t_end = delta_text.find("</think>");

                                if (t_start != std::string::npos) {
                                    in_thinking = true;
                                    delta_text.erase(t_start, 7);
                                }
                                if (t_end != std::string::npos) {
                                    in_thinking = false;
                                    delta_text.erase(t_end, 8);
                                }

                                if (in_thinking) {
                                    delta_obj["reasoning_content"] = delta_text;
                                } else {
                                    delta_obj["content"] = delta_text;
                                }

                                json chunk = {
                                    {"id", id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", created},
                                    {"model", "diffusion-model"},
                                    {"choices", json::array({
                                        {
                                            {"index", 0},
                                            {"delta", delta_obj},
                                            {"finish_reason", nullptr}
                                        }
                                    })}
                                };
                                std::string sse = "data: " + chunk.dump() + "\n\n";
                                sink.write(sse.data(), sse.size());
                            }
                        } else {
                            // Final flush
                            for (const auto& stop : params.stop_strings) {
                                if (!stop.empty() && pending_text.size() >= stop.size()) {
                                    if (pending_text.compare(pending_text.size() - stop.size(), stop.size(), stop) == 0) {
                                        pending_text.resize(pending_text.size() - stop.size());
                                        break;
                                    }
                                }
                            }
                            if (!pending_text.empty()) {
                                json delta_obj = json::object();
                                if (in_thinking) delta_obj["reasoning_content"] = pending_text;
                                else delta_obj["content"] = pending_text;

                                json chunk = {
                                    {"id", id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", created},
                                    {"model", "diffusion-model"},
                                    {"choices", json::array({
                                        {
                                            {"index", 0},
                                            {"delta", delta_obj},
                                            {"finish_reason", nullptr}
                                        }
                                    })}
                                };
                                std::string sse = "data: " + chunk.dump() + "\n\n";
                                sink.write(sse.data(), sse.size());
                            }
                        }
                        return sink.is_writable();
                    };

                    diffusion::GenResult result = engine->generate(prompt, params, callback);
                    engine->backend().kv_clear(slot->id);
                    slot_manager.release_slot(slot);

                    metrics.total_completion_tokens += result.tokens.size();

                    std::string finish_reason = (result.stop_reason == diffusion::StopReason::EOG) ? "stop"
                        : (result.stop_reason == diffusion::StopReason::MAX_TOKENS ? "length" : "stop");

                    json final_chunk = {
                        {"id", id},
                        {"object", "chat.completion.chunk"},
                        {"created", created},
                        {"model", "diffusion-model"},
                        {"choices", json::array({
                            {
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", finish_reason}
                            }
                        })}
                    };
                    std::string sse = "data: " + final_chunk.dump() + "\n\ndata: [DONE]\n\n";
                    sink.write(sse.data(), sse.size());
                    sink.done();
                    return false;
                }
            );
        }
    });

    std::cout << "Starting diffusion-server at http://" << config.host << ":" << config.port
              << " (" << slot_manager.total_slots() << " slots)\n";
    std::cout.flush();
    if (!svr.listen(config.host.c_str(), config.port)) {
        std::cerr << "Error: failed to bind to " << config.host << ":" << config.port << "\n";
        return 1;
    }

    return 0;
}
