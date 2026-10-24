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
#include <regex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace diffusion {

inline bool parse_tool_call(const std::string& raw, std::string& name, std::string& args_json) {
    std::string s = raw;
    while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\r' || s.front() == '\t')) s.erase(0, 1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t')) s.pop_back();

    // 1. JSON format: {"name": ..., "arguments": ...}
    if (s.rfind("{\"", 0) == 0 || s.rfind("{\n", 0) == 0 || s.rfind("{ ", 0) == 0) {
        try {
            json j = json::parse(s);
            if (j.contains("name")) {
                name = j["name"].get<std::string>();
                if (j.contains("arguments")) {
                    if (j["arguments"].is_string()) args_json = j["arguments"].get<std::string>();
                    else args_json = j["arguments"].dump();
                } else {
                    args_json = "{}";
                }
                return true;
            }
        } catch (...) {}
    }

    // 2. XML format:
    // <function=glob>
    // <parameter=pattern>
    // **/todo.py
    // </parameter>
    // </function>
    size_t fn_tag = s.find("<function=");
    size_t fn_prefix_len = 10;
    if (fn_tag == std::string::npos) {
        fn_tag = s.find("<function name=\"");
        fn_prefix_len = 16;
    }
    if (fn_tag != std::string::npos) {
        size_t name_start = fn_tag + fn_prefix_len;
        size_t name_end = s.find_first_of("\">\r\n ", name_start);
        if (name_end != std::string::npos) {
            name = s.substr(name_start, name_end - name_start);

            json j_args = json::object();
            size_t p_pos = name_end;
            while (p_pos < s.size()) {
                size_t param_tag = s.find("<parameter=", p_pos);
                size_t prefix_len = 11;
                if (param_tag == std::string::npos) {
                    param_tag = s.find("<parameter name=\"", p_pos);
                    prefix_len = 17;
                }
                if (param_tag == std::string::npos) break;

                size_t p_name_start = param_tag + prefix_len;
                size_t p_name_end = s.find_first_of("\">\r\n ", p_name_start);
                if (p_name_end == std::string::npos) break;
                std::string p_name = s.substr(p_name_start, p_name_end - p_name_start);

                size_t tag_close = s.find('>', p_name_end);
                if (tag_close == std::string::npos) break;
                size_t val_start = tag_close + 1;

                size_t val_end = s.find("</parameter>", val_start);
                if (val_end == std::string::npos) val_end = s.find("</function>", val_start);
                if (val_end == std::string::npos) val_end = s.size();

                std::string val = s.substr(val_start, val_end - val_start);
                while (!val.empty() && (val.front() == '\n' || val.front() == '\r')) val.erase(0, 1);
                while (!val.empty() && (val.back() == '\n' || val.back() == '\r')) val.pop_back();

                try {
                    j_args[p_name] = json::parse(val);
                } catch (...) {
                    j_args[p_name] = val;
                }

                p_pos = val_end + (s.substr(val_end, 12) == "</parameter>" ? 12 : 0);
            }

            args_json = j_args.dump();
            std::cout << "[Tool Parsed XML] fn=" << name << " args=" << args_json << "\n";
            std::cout.flush();
            return true;
        }
    }

    // 3. Gemma format: call:func_name{...}
    size_t call_pos = s.find("call:");
    if (call_pos != std::string::npos) {
        size_t name_start = call_pos + 5;
        while (name_start < s.size() && (s[name_start] == ' ' || s[name_start] == '\t')) name_start++;

        size_t brace = s.find('{', name_start);
        if (brace != std::string::npos) {
            name = s.substr(name_start, brace - name_start);
            while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.pop_back();

            size_t end_brace = s.rfind('}');
            std::string body;
            if (end_brace != std::string::npos && end_brace > brace) {
                body = s.substr(brace, end_brace - brace + 1);
            } else {
                body = s.substr(brace) + "}";
            }

            // Try robust Gemma key-value parser first (handles <|"|> string wrappers and multiline code)
            size_t inner_start = body.find('{');
            size_t inner_end = body.rfind('}');
            if (inner_start != std::string::npos && inner_end != std::string::npos && inner_end > inner_start) {
                std::string inner = body.substr(inner_start + 1, inner_end - inner_start - 1);
                json j = json::object();
                size_t pos = 0;
                while (pos < inner.size()) {
                    while (pos < inner.size() && (inner[pos] == ' ' || inner[pos] == '\n' || inner[pos] == '\r' || inner[pos] == '\t' || inner[pos] == ',')) pos++;
                    if (pos >= inner.size()) break;

                    size_t colon = inner.find(':', pos);
                    if (colon == std::string::npos) break;

                    std::string key = inner.substr(pos, colon - pos);
                    while (!key.empty() && (key.front() == ' ' || key.front() == '"')) key.erase(0, 1);
                    while (!key.empty() && (key.back() == ' ' || key.back() == '"')) key.pop_back();

                    pos = colon + 1;
                    while (pos < inner.size() && (inner[pos] == ' ' || inner[pos] == '\n' || inner[pos] == '\r' || inner[pos] == '\t')) pos++;
                    if (pos >= inner.size()) break;

                    if (inner.compare(pos, 5, "<|\"|>") == 0) {
                        size_t val_start = pos + 5;
                        size_t val_end = inner.find("<|\"|>", val_start);
                        if (val_end == std::string::npos) val_end = inner.size();
                        std::string val = inner.substr(val_start, val_end - val_start);
                        j[key] = val;
                        pos = val_end + (val_end < inner.size() ? 5 : 0);
                    } else if (inner[pos] == '"') {
                        size_t val_start = pos + 1;
                        size_t val_end = val_start;
                        while (val_end < inner.size()) {
                            if (inner[val_end] == '"' && inner[val_end - 1] != '\\') break;
                            val_end++;
                        }
                        std::string val = inner.substr(val_start, val_end - val_start);
                        j[key] = val;
                        pos = val_end + (val_end < inner.size() ? 1 : 0);
                    } else {
                        size_t val_end = inner.find_first_of(",}\n\r", pos);
                        if (val_end == std::string::npos) val_end = inner.size();
                        std::string val = inner.substr(pos, val_end - pos);
                        while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
                        if (val == "true" || val == "True") j[key] = true;
                        else if (val == "false" || val == "False") j[key] = false;
                        else if (val == "null" || val == "None") j[key] = nullptr;
                        else {
                            try {
                                j[key] = json::parse(val);
                            } catch (...) {
                                j[key] = val;
                            }
                        }
                        pos = val_end;
                    }
                }

                if (!j.empty()) {
                    args_json = j.dump();
                    std::cout << "[Tool Parsed Gemma KV] fn=" << name << " args=" << args_json << "\n";
                    std::cout.flush();
                    return true;
                }
            }

            size_t q_pos = 0;
            while ((q_pos = body.find("<|\"|>", q_pos)) != std::string::npos) {
                body.replace(q_pos, 5, "\"");
                q_pos += 1;
            }

            size_t tq = 0;
            while ((tq = body.find("\"\"\"", tq)) != std::string::npos) {
                size_t tq_end = body.find("\"\"\"", tq + 3);
                if (tq_end != std::string::npos) {
                    std::string inner = body.substr(tq + 3, tq_end - (tq + 3));
                    json j_inner = inner;
                    std::string escaped = j_inner.dump();
                    body.replace(tq, (tq_end + 3) - tq, escaped);
                    tq += escaped.size();
                } else {
                    break;
                }
            }

            std::regex key_regex("([{\\s,])([a-zA-Z_][a-zA-Z0-9_]*)\\s*:");
            body = std::regex_replace(body, key_regex, "$1\"$2\":");
            body = std::regex_replace(body, std::regex("\\bTrue\\b"), "true");
            body = std::regex_replace(body, std::regex("\\bFalse\\b"), "false");
            body = std::regex_replace(body, std::regex("\\bNone\\b"), "null");

            try {
                json j = json::parse(body);
                args_json = j.dump();
                std::cout << "[Tool Parsed Gemma] fn=" << name << " args=" << args_json << "\n";
                std::cout.flush();
                return true;
            } catch (...) {
                json fallback = {{"raw", body}};
                args_json = fallback.dump();
                std::cout << "[Tool Parsed Fallback] fn=" << name << " args=" << args_json << "\n";
                std::cout.flush();
                return true;
            }
        }
    }

    return false;
}

inline bool prompt_ends_with_thinking(const std::string& prompt) {
    std::string s = prompt;
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t')) {
        s.pop_back();
    }
    static const std::string_view think_tails[] = {
        "<think>",
        "<|think|>",
        "<|channel>thought",
        "<channel>thought"
    };
    for (const auto& tail : think_tails) {
        if (s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0) {
            return true;
        }
    }
    return false;
}

struct StreamFilter {
    bool in_thinking = false;
    bool in_tool_call = false;
    std::string tool_call_buf;
    std::string carry_buf_;
    bool had_tool_call = false;

    struct OutputItem {
        enum Type { CONTENT, REASONING, TOOL_CALL } type;
        std::string text;
        std::string fn_name;
        std::string fn_args;
    };

    static bool is_potential_tag_prefix(const std::string& s) {
        static const std::string_view all_tags[] = {
            "<|im_end|>\n",
            "<|im_end|>",
            "<|endoftext|>\n",
            "<|endoftext|>",
            "<end_of_turn>\n",
            "<end_of_turn>",
            "<|channel>thought\n",
            "<|channel>thought",
            "<channel>thought\n",
            "<channel>thought",
            "<think>\n",
            "<think>",
            "<|tool_call>call:",
            "<|tool_call>",
            "<tool_call>",
            "<tool_call|>",
            "</tool_call>",
            "<|tool_response>",
            "<tool_response>",
            "|tool_response>",
            "<channel|>\n",
            "<channel|>",
            "</channel>\n",
            "</channel>",
            "<|channel|>\n",
            "<|channel|>",
            "</think>\n",
            "</think>"
        };
        for (const auto& t : all_tags) {
            if (t.rfind(s, 0) == 0) return true;
        }
        return false;
    }

    std::vector<OutputItem> process(std::string chunk, bool is_done) {
        std::vector<OutputItem> out;
        std::string text = carry_buf_ + chunk;
        carry_buf_.clear();

        if (!is_done) {
            size_t last_bracket = text.rfind('<');
            if (last_bracket != std::string::npos && (text.size() - last_bracket) <= 25) {
                std::string candidate = text.substr(last_bracket);
                if (is_potential_tag_prefix(candidate)) {
                    carry_buf_ = candidate;
                    text.erase(last_bracket);
                }
            }
        }

        while (!text.empty()) {
            if (in_thinking) {
                size_t end_pos = std::string::npos;
                size_t end_len = 0;
                static const std::string_view think_ends[] = {
                    "<channel|>\n",
                    "<channel|>",
                    "</channel>\n",
                    "</channel>",
                    "<|channel|>\n",
                    "<|channel|>",
                    "</think>\n",
                    "</think>"
                };
                for (const auto& te : think_ends) {
                    size_t pos = text.find(te);
                    if (pos != std::string::npos && (end_pos == std::string::npos || pos < end_pos)) {
                        end_pos = pos;
                        end_len = te.size();
                    }
                }

                if (end_pos != std::string::npos) {
                    if (end_pos > 0) {
                        out.push_back({OutputItem::REASONING, text.substr(0, end_pos), "", ""});
                    }
                    in_thinking = false;
                    text.erase(0, end_pos + end_len);
                } else {
                    size_t eog_pos = text.find("<|im_end|>");
                    if (eog_pos == std::string::npos) eog_pos = text.find("<|endoftext|>");
                    if (eog_pos == std::string::npos) eog_pos = text.find("<end_of_turn>");
                    if (eog_pos != std::string::npos) {
                        if (eog_pos > 0) {
                            out.push_back({OutputItem::REASONING, text.substr(0, eog_pos), "", ""});
                        }
                        in_thinking = false;
                        text.clear();
                        break;
                    }
                    out.push_back({OutputItem::REASONING, text, "", ""});
                    text.clear();
                }
            } else if (in_tool_call) {
                size_t end_pos = std::string::npos;
                size_t end_len = 0;
                static const std::string_view tool_ends[] = {
                    "<tool_call|>",
                    "</tool_call>",
                    "<|tool_response>",
                    "<tool_response>"
                };
                for (const auto& te : tool_ends) {
                    size_t pos = text.find(te);
                    if (pos != std::string::npos && (end_pos == std::string::npos || pos < end_pos)) {
                        end_pos = pos;
                        end_len = te.size();
                    }
                }

                if (end_pos != std::string::npos) {
                    tool_call_buf += text.substr(0, end_pos);
                    in_tool_call = false;
                    text.erase(0, end_pos + end_len);

                    std::string fn_name, fn_args;
                    if (parse_tool_call(tool_call_buf, fn_name, fn_args)) {
                        had_tool_call = true;
                        out.push_back({OutputItem::TOOL_CALL, "", fn_name, fn_args});
                    }
                    tool_call_buf.clear();
                } else {
                    tool_call_buf += text;
                    text.clear();
                }
            } else {
                size_t eog_pos = text.find("<|im_end|>");
                if (eog_pos == std::string::npos) eog_pos = text.find("<|endoftext|>");
                if (eog_pos == std::string::npos) eog_pos = text.find("<end_of_turn>");

                size_t resp_pos = text.find("<|tool_response>");
                if (resp_pos == std::string::npos) resp_pos = text.find("<tool_response>");
                if (resp_pos == std::string::npos) resp_pos = text.find("|tool_response>");

                size_t tool_pos = std::string::npos;
                size_t tool_len = 0;
                std::string_view matched_tool_start;
                static const std::string_view tool_starts[] = {
                    "<|tool_call>call:",
                    "<|tool_call>",
                    "<tool_call>"
                };
                for (const auto& ts : tool_starts) {
                    size_t pos = text.find(ts);
                    if (pos != std::string::npos && (tool_pos == std::string::npos || pos < tool_pos)) {
                        tool_pos = pos;
                        tool_len = ts.size();
                        matched_tool_start = ts;
                    }
                }

                size_t think_pos = std::string::npos;
                size_t think_len = 0;
                static const std::string_view think_starts[] = {
                    "<|channel>thought\n",
                    "<|channel>thought",
                    "<channel>thought\n",
                    "<channel>thought",
                    "<think>\n",
                    "<think>"
                };
                for (const auto& ts : think_starts) {
                    size_t pos = text.find(ts);
                    if (pos != std::string::npos && (think_pos == std::string::npos || pos < think_pos)) {
                        think_pos = pos;
                        think_len = ts.size();
                    }
                }

                size_t earliest = std::string::npos;
                enum NextAction { ACTION_NONE, ACTION_EOG, ACTION_RESP, ACTION_TOOL, ACTION_THINK } action = ACTION_NONE;

                if (eog_pos != std::string::npos && (earliest == std::string::npos || eog_pos < earliest)) {
                    earliest = eog_pos;
                    action = ACTION_EOG;
                }
                if (resp_pos != std::string::npos && (earliest == std::string::npos || resp_pos < earliest)) {
                    earliest = resp_pos;
                    action = ACTION_RESP;
                }
                if (tool_pos != std::string::npos && (earliest == std::string::npos || tool_pos < earliest)) {
                    earliest = tool_pos;
                    action = ACTION_TOOL;
                }
                if (think_pos != std::string::npos && (earliest == std::string::npos || think_pos < earliest)) {
                    earliest = think_pos;
                    action = ACTION_THINK;
                }

                if (action == ACTION_NONE) {
                    out.push_back({OutputItem::CONTENT, text, "", ""});
                    text.clear();
                } else if (action == ACTION_EOG) {
                    if (eog_pos > 0) {
                        out.push_back({OutputItem::CONTENT, text.substr(0, eog_pos), "", ""});
                    }
                    text.clear();
                    break;
                } else if (action == ACTION_RESP) {
                    if (resp_pos > 0) {
                        out.push_back({OutputItem::CONTENT, text.substr(0, resp_pos), "", ""});
                    }
                    text.clear();
                    break;
                } else if (action == ACTION_TOOL) {
                    if (tool_pos > 0) {
                        out.push_back({OutputItem::CONTENT, text.substr(0, tool_pos), "", ""});
                    }
                    in_tool_call = true;
                    tool_call_buf.clear();
                    if (matched_tool_start == "<|tool_call>call:") {
                        tool_call_buf = "call:";
                    }
                    text.erase(0, tool_pos + tool_len);
                } else if (action == ACTION_THINK) {
                    if (think_pos > 0) {
                        out.push_back({OutputItem::CONTENT, text.substr(0, think_pos), "", ""});
                    }
                    in_thinking = true;
                    text.erase(0, think_pos + think_len);
                }
            }
        }

        if (is_done && in_tool_call && !tool_call_buf.empty()) {
            std::string fn_name, fn_args;
            if (parse_tool_call(tool_call_buf, fn_name, fn_args)) {
                had_tool_call = true;
                out.push_back({OutputItem::TOOL_CALL, "", fn_name, fn_args});
            }
            tool_call_buf.clear();
            in_tool_call = false;
        }

        return out;
    }
};

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
            if (m == "ar") throw std::runtime_error("Autoregressive mode ('ar') has been completely removed. diffusion.gguf only supports diffusion generation ('block' or 'whole').");
            else if (m == "whole") cfg.default_params.mode = DecodeMode::WHOLE;
            else cfg.default_params.mode = DecodeMode::BLOCK;
        } else if (arg == "-n" || arg == "--n-predict") cfg.default_params.n_predict = next_int("-n/--n-predict");
        else if (arg == "--block") cfg.default_params.block_size = next_int("--block");
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

    if (cfg.default_params.n_predict <= 128) {
        cfg.default_params.n_predict = 2048;
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

    if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number_integer()) {
        params.n_predict = body["max_completion_tokens"].get<int>();
    } else if (body.contains("max_tokens") && body["max_tokens"].is_number_integer()) {
        params.n_predict = body["max_tokens"].get<int>();
    } else if (body.contains("n_predict") && body["n_predict"].is_number_integer()) {
        params.n_predict = body["n_predict"].get<int>();
    }

    if (body.contains("temperature") && body["temperature"].is_number()) params.sampling.temp = body["temperature"].get<float>();
    if (body.contains("top_p") && body["top_p"].is_number()) params.sampling.top_p = body["top_p"].get<float>();
    if (body.contains("top_k") && body["top_k"].is_number_integer()) params.sampling.top_k = body["top_k"].get<int>();
    if (body.contains("min_p") && body["min_p"].is_number()) params.sampling.min_p = body["min_p"].get<float>();
    if (body.contains("typical_p") && body["typical_p"].is_number()) params.sampling.typical_p = body["typical_p"].get<float>();
    if (body.contains("presence_penalty") && body["presence_penalty"].is_number()) params.sampling.presence_penalty = body["presence_penalty"].get<float>();
    if (body.contains("frequency_penalty") && body["frequency_penalty"].is_number()) params.sampling.frequency_penalty = body["frequency_penalty"].get<float>();
    if (body.contains("repeat_penalty") && body["repeat_penalty"].is_number()) params.sampling.repeat_penalty = body["repeat_penalty"].get<float>();
    if (body.contains("repeat_last_n") && body["repeat_last_n"].is_number_integer()) params.sampling.repeat_last_n = body["repeat_last_n"].get<int>();
    if (body.contains("seed") && body["seed"].is_number_unsigned()) params.sampling.seed = body["seed"].get<uint32_t>();

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
    static const std::string default_stops[] = {
        "<|im_end|>",
        "<|endoftext|>",
        "<end_of_turn>"
    };
    for (const auto& ds : default_stops) {
        if (std::find(params.stop_strings.begin(), params.stop_strings.end(), ds) == params.stop_strings.end()) {
            params.stop_strings.push_back(ds);
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
        if (m == "ar") {
            throw std::runtime_error("Autoregressive mode ('ar') has been completely removed. diffusion.gguf only supports diffusion generation ('block' or 'whole').");
        } else if (m == "whole") params.mode = DecodeMode::WHOLE;
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
    std::mutex engine_mutex;

    svr.set_exception_handler([](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            std::cerr << "[Server Error] " << req.path << ": " << e.what() << "\n";
            json err = {{"error", {{"message", e.what()}, {"type", "internal_server_error"}}}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
        } catch (...) {
            std::cerr << "[Server Error] " << req.path << ": unknown exception\n";
            json err = {{"error", {{"message", "Unknown internal server error"}, {"type", "internal_server_error"}}}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
        }
    });

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // Model list
    auto models_handler = [&](const httplib::Request& req, httplib::Response& res) {
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
    };
    svr.Get("/v1/models", models_handler);
    svr.Get("/models", models_handler);

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

        diffusion::GenResult result;
        std::vector<int32_t> prompt_tokens;
        if (!stream) {
            {
                std::lock_guard<std::mutex> lock(engine_mutex);
                prompt_tokens = engine->tokenizer().tokenize(prompt, true, true);
                result = engine->generate(prompt, params);
                engine->backend().kv_clear(slot->id);
            }
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
    auto chat_handler = [&](const httplib::Request& req, httplib::Response& res) {
        if (!diffusion::check_auth(req, config.api_key, res)) return;

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"Invalid JSON body"}})", "application/json");
            return;
        }

        std::cout << "[Request Keys] ";
        for (auto& el : body.items()) {
            std::cout << el.key() << " ";
        }
        std::cout << "\n";
        std::cout.flush();

        std::vector<diffusion::ToolDef> tool_defs;
        if (body.contains("tools") && body["tools"].is_array()) {
            for (const auto& t : body["tools"]) {
                if (!t.is_object()) continue;
                std::string type = t.value("type", "function");
                if (type == "function" && t.contains("function") && t["function"].is_object()) {
                    auto fn = t["function"];
                    diffusion::ToolDef td;
                    td.name = fn.value("name", "");
                    td.description = fn.value("description", "");
                    if (fn.contains("parameters")) {
                        if (fn["parameters"].is_string()) td.parameters = fn["parameters"].get<std::string>();
                        else td.parameters = fn["parameters"].dump();
                    }
                    tool_defs.push_back(std::move(td));
                }
            }
        }

        std::vector<diffusion::ChatMsg> msgs;
        if (body.contains("messages") && body["messages"].is_array()) {
            for (const auto& m : body["messages"]) {
                if (!m.is_object() || !m.contains("role")) continue;
                diffusion::ChatMsg msg;
                msg.role = m.value("role", "user");
                if (m.contains("content")) {
                    if (m["content"].is_string()) {
                        msg.content = m["content"].get<std::string>();
                    } else if (m["content"].is_array()) {
                        for (const auto& part : m["content"]) {
                            if (part.is_string()) {
                                msg.content += part.get<std::string>();
                            } else if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                                msg.content += part["text"].get<std::string>();
                            }
                        }
                    }
                }
                if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                    msg.reasoning_content = m["reasoning_content"].get<std::string>();
                }
                if (m.contains("tool_call_id") && m["tool_call_id"].is_string()) {
                    msg.tool_call_id = m["tool_call_id"].get<std::string>();
                }
                if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                    for (const auto& tc : m["tool_calls"]) {
                        if (!tc.is_object()) continue;
                        diffusion::ToolCall call;
                        call.id = tc.value("id", "");
                        if (tc.contains("function") && tc["function"].is_object()) {
                            call.name = tc["function"].value("name", "");
                            if (tc["function"].contains("arguments")) {
                                if (tc["function"]["arguments"].is_string()) call.arguments = tc["function"]["arguments"].get<std::string>();
                                else call.arguments = tc["function"]["arguments"].dump();
                            }
                        }
                        msg.tool_calls.push_back(std::move(call));
                    }
                }
                msgs.push_back(std::move(msg));
            }
        }
        if (msgs.empty()) {
            msgs.push_back(diffusion::ChatMsg{"user", "Hello"});
        }

        std::string prompt = engine->chat_formatter().apply(msgs, tool_defs, true);
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

        diffusion::GenResult result;
        std::vector<int32_t> prompt_tokens;
        if (!stream) {
            {
                std::lock_guard<std::mutex> lock(engine_mutex);
                prompt_tokens = engine->tokenizer().tokenize(prompt, true, true);
                result = engine->generate(prompt, params);
                engine->backend().kv_clear(slot->id);
            }
            slot_manager.release_slot(slot);

            metrics.total_prompt_tokens += prompt_tokens.size();
            metrics.total_completion_tokens += result.tokens.size();

            diffusion::StreamFilter filter;
            filter.in_thinking = diffusion::prompt_ends_with_thinking(prompt);
            auto items = filter.process(result.text, true);
            std::string final_content;
            std::string final_reasoning = result.reasoning_content;
            json tool_calls = json::array();
            int tool_idx = 0;
            for (const auto& it : items) {
                if (it.type == diffusion::StreamFilter::OutputItem::CONTENT) {
                    final_content += it.text;
                } else if (it.type == diffusion::StreamFilter::OutputItem::REASONING) {
                    final_reasoning += it.text;
                } else if (it.type == diffusion::StreamFilter::OutputItem::TOOL_CALL) {
                    tool_calls.push_back({
                        {"id", "call_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "_" + std::to_string(tool_idx++)},
                        {"type", "function"},
                        {"function", {
                            {"name", it.fn_name},
                            {"arguments", it.fn_args}
                        }}
                    });
                }
            }

            std::string finish_reason = filter.had_tool_call ? "tool_calls"
                : ((result.stop_reason == diffusion::StopReason::EOG) ? "stop"
                : ((result.stop_reason == diffusion::StopReason::MAX_TOKENS) ? "length" : "stop"));

            json msg_obj = {
                {"role", "assistant"},
                {"content", final_content}
            };
            if (!final_reasoning.empty()) {
                msg_obj["reasoning_content"] = final_reasoning;
            }
            if (!tool_calls.empty()) {
                msg_obj["tool_calls"] = tool_calls;
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
                [&, prompt, params, id, created, slot, tool_defs](size_t, httplib::DataSink& sink) {
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
                    diffusion::StreamFilter filter;
                    filter.in_thinking = diffusion::prompt_ends_with_thinking(prompt);
                    int tool_call_idx = 0;

                    auto emit_items = [&](const std::vector<diffusion::StreamFilter::OutputItem>& items) -> bool {
                        for (const auto& it : items) {
                            if (!sink.is_writable()) return false;
                            std::cout << "[Emit Item] type=" << (int)it.type << " fn=" << it.fn_name << " text=[" << it.text << "]\n";
                            std::cout.flush();
                            json delta_obj = json::object();
                            if (it.type == diffusion::StreamFilter::OutputItem::CONTENT) {
                                if (it.text.empty()) continue;
                                delta_obj["content"] = it.text;
                            } else if (it.type == diffusion::StreamFilter::OutputItem::REASONING) {
                                if (it.text.empty()) continue;
                                delta_obj["reasoning_content"] = it.text;
                            } else if (it.type == diffusion::StreamFilter::OutputItem::TOOL_CALL) {
                                delta_obj["tool_calls"] = json::array({
                                    {
                                        {"index", tool_call_idx++},
                                        {"id", "call_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "_" + std::to_string(tool_call_idx)},
                                        {"type", "function"},
                                        {"function", {
                                            {"name", it.fn_name},
                                            {"arguments", it.fn_args}
                                        }}
                                    }
                                });
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
                        return sink.is_writable();
                    };

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

                                auto items = filter.process(delta_text, false);
                                if (!emit_items(items)) return false;
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
                                auto items = filter.process(pending_text, true);
                                if (!emit_items(items)) return false;
                            } else {
                                auto items = filter.process("", true);
                                if (!emit_items(items)) return false;
                            }
                        }
                        return sink.is_writable();
                    };

                    try {
                        std::cout << "[Request] stream=true prompt_len=" << prompt.size()
                                  << " tools=" << tool_defs.size() << "\n";
                        std::cout.flush();

                        diffusion::GenResult result;
                        {
                            std::lock_guard<std::mutex> lock(engine_mutex);
                            result = engine->generate(prompt, params, callback);
                            engine->backend().kv_clear(slot->id);
                        }
                        slot_manager.release_slot(slot);

                        std::cout << "[Completed] tok=" << result.tokens.size()
                                  << " stop=" << (int)result.stop_reason
                                  << " had_tool=" << filter.had_tool_call << "\n";
                        std::cout << "[RAW TEXT] >>>" << result.text << "<<<\n";
                        std::cout.flush();

                        metrics.total_completion_tokens += result.tokens.size();

                        std::string finish_reason = filter.had_tool_call ? "tool_calls"
                            : ((result.stop_reason == diffusion::StopReason::EOG) ? "stop"
                            : ((result.stop_reason == diffusion::StopReason::MAX_TOKENS) ? "length" : "stop"));

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
                    } catch (const std::exception& e) {
                        std::cerr << "[Stream Error] Generation error: " << e.what() << "\n";
                        engine->backend().kv_clear(slot->id);
                        slot_manager.release_slot(slot);
                        sink.done();
                    } catch (...) {
                        std::cerr << "[Stream Error] Unknown generation error\n";
                        engine->backend().kv_clear(slot->id);
                        slot_manager.release_slot(slot);
                        sink.done();
                    }
                    return false;
                }
            );
        }
    };
    svr.Post("/v1/chat/completions", chat_handler);
    svr.Post("/chat/completions", chat_handler);

    std::cout << "Starting diffusion-server at http://" << config.host << ":" << config.port
              << " (" << slot_manager.total_slots() << " slots)\n";
    std::cout.flush();
    if (!svr.listen(config.host.c_str(), config.port)) {
        std::cerr << "Error: failed to bind to " << config.host << ":" << config.port << "\n";
        return 1;
    }

    return 0;
}
