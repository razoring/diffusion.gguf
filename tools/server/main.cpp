#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "engine.h"
#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>
#include "json-schema-to-grammar.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::json;

namespace diffusion {

struct ServerConfig {
    BackendConfig backend;
    GenParams default_params;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
};

static void print_server_help() {
    std::cout << "Usage: diffusion-server -m <model.gguf> [options]\n\n"
              << "Server Options:\n"
              << "  -m, --model PATH         Path to GGUF model file (required)\n"
              << "  --host HOST              Host address to listen on (default: 127.0.0.1)\n"
              << "  --port PORT              Port to listen on (default: 8080)\n"
              << "  --api-key KEY            Optional Bearer API key authentication\n"
              << "  -c, --ctx-size N         Context size (default: 2048)\n"
              << "  -t, --threads N          CPU threads (default: 4)\n"
              << "  -ngl, --n-gpu-layers N   GPU offload layers (default: 0)\n"
              << "  -fa, --flash-attn        Enable Flash Attention\n\n"
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
        else if (arg == "--api-key") cfg.api_key = next_str("--api-key");
        else if (arg == "-c" || arg == "--ctx-size") cfg.backend.n_ctx = next_int("-c/--ctx-size");
        else if (arg == "-t" || arg == "--threads") cfg.backend.n_threads = next_int("-t/--threads");
        else if (arg == "-ngl" || arg == "--n-gpu-layers") cfg.backend.n_gpu_layers = next_int("-ngl/--n-gpu-layers");
        else if (arg == "-fa" || arg == "--flash-attn") cfg.backend.flash_attn = true;
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

    std::unique_ptr<diffusion::Engine> engine;
    try {
        engine = std::make_unique<diffusion::Engine>(config.backend);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize engine: " << e.what() << "\n";
        return 1;
    }

    httplib::Server svr;
    std::mutex slot_mutex;

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

        if (body.contains("n") && body["n"].get<int>() > 1) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"n > 1 not supported in v1"}})", "application/json");
            return;
        }

        if (body.contains("logprobs") && !body["logprobs"].is_null()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"logprobs not supported in v1"}})", "application/json");
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

        std::lock_guard<std::mutex> lock(slot_mutex);
        std::string id = "cmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        uint64_t created = static_cast<uint64_t>(std::time(nullptr));

        if (!stream) {
            auto prompt_tokens = engine->tokenizer().tokenize(prompt, true, true);
            diffusion::GenResult result = engine->generate(prompt, params);

            std::string finish_reason = (result.stop_reason == diffusion::StopReason::EOG) ? "stop"
                : (result.stop_reason == diffusion::StopReason::MAX_TOKENS ? "length" : "stop");

            json resp = {
                {"id", id},
                {"object", "text_completion"},
                {"created", created},
                {"model", "diffusion-model"},
                {"choices", json::array({
                    {
                        {"text", result.text},
                        {"index", 0},
                        {"logprobs", nullptr},
                        {"finish_reason", finish_reason}
                    }
                })},
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
                [&, prompt, params, id, created](size_t, httplib::DataSink& sink) {
                    std::string pending_text;

                    auto callback = [&](const diffusion::StepEvent& ev) -> bool {
                        if (!sink.is_writable()) return false;

                        for (int32_t tok : ev.committed_new) {
                            pending_text += engine->tokenizer().piece(tok);
                        }

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

        if (body.contains("n") && body["n"].get<int>() > 1) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"n > 1 not supported in v1"}})", "application/json");
            return;
        }

        if (body.contains("logprobs") && !body["logprobs"].is_null()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"logprobs not supported in v1"}})", "application/json");
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

        std::lock_guard<std::mutex> lock(slot_mutex);
        std::string id = "chatcmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        uint64_t created = static_cast<uint64_t>(std::time(nullptr));

        if (!stream) {
            auto prompt_tokens = engine->tokenizer().tokenize(prompt, true, true);
            diffusion::GenResult result = engine->generate(prompt, params);

            std::string finish_reason = (result.stop_reason == diffusion::StopReason::EOG) ? "stop"
                : (result.stop_reason == diffusion::StopReason::MAX_TOKENS ? "length" : "stop");

            json resp = {
                {"id", id},
                {"object", "chat.completion"},
                {"created", created},
                {"model", "diffusion-model"},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"message", {
                            {"role", "assistant"},
                            {"content", result.text}
                        }},
                        {"finish_reason", finish_reason}
                    }
                })},
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
                [&, prompt, params, id, created](size_t, httplib::DataSink& sink) {
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

                    auto callback = [&](const diffusion::StepEvent& ev) -> bool {
                        if (!sink.is_writable()) return false;

                        for (int32_t tok : ev.committed_new) {
                            pending_text += engine->tokenizer().piece(tok);
                        }

                        if (!ev.done) {
                            size_t safe_len = diffusion::compute_safe_prefix_len(pending_text, params.stop_strings);
                            if (safe_len > 0) {
                                std::string delta_text = pending_text.substr(0, safe_len);
                                pending_text.erase(0, safe_len);

                                json chunk = {
                                    {"id", id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", created},
                                    {"model", "diffusion-model"},
                                    {"choices", json::array({
                                        {
                                            {"index", 0},
                                            {"delta", {{"content", delta_text}}},
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
                                    {"object", "chat.completion.chunk"},
                                    {"created", created},
                                    {"model", "diffusion-model"},
                                    {"choices", json::array({
                                        {
                                            {"index", 0},
                                            {"delta", {{"content", pending_text}}},
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

    std::cout << "Starting diffusion-server at http://" << config.host << ":" << config.port << "\n";
    std::cout.flush();
    if (!svr.listen(config.host.c_str(), config.port)) {
        std::cerr << "Error: failed to bind to " << config.host << ":" << config.port << "\n";
        return 1;
    }

    return 0;
}

