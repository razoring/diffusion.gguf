#include "backend_llama.h"
#include "common.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace diffusion {

static enum ggml_type parse_ggml_type(const std::string& s) {
    if (s == "f32") return GGML_TYPE_F32;
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    if (s == "q5_0") return GGML_TYPE_Q5_0;
    if (s == "q5_1") return GGML_TYPE_Q5_1;
    return GGML_TYPE_F16;
}

BackendLlama::BackendLlama(const BackendConfig& config) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.n_gpu_layers;
    mparams.main_gpu = config.main_gpu;
    mparams.use_mmap = config.use_mmap;
    mparams.use_mlock = config.use_mlock;
    mparams.check_tensors = config.check_tensors;

    if (config.split_mode == 0) {
        mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    } else if (config.split_mode == 2) {
        mparams.split_mode = LLAMA_SPLIT_MODE_ROW;
    } else {
        mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
    }

    std::vector<float> tensor_split_floats;
    if (!config.tensor_split_str.empty()) {
        std::string s = config.tensor_split_str;
        size_t pos = 0;
        while ((pos = s.find(',')) != std::string::npos) {
            std::string token = s.substr(0, pos);
            if (!token.empty()) tensor_split_floats.push_back(std::stof(token));
            s.erase(0, pos + 1);
        }
        if (!s.empty()) tensor_split_floats.push_back(std::stof(s));
        mparams.tensor_split = tensor_split_floats.data();
    }

    model_ = llama_model_load_from_file(config.model_path.c_str(), mparams);
    if (!model_) {
        throw std::runtime_error("Failed to load model from: " + config.model_path);
    }

    const llama_vocab* vocab = llama_model_get_vocab(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab);
    n_embd_ = llama_model_n_embd(model_);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_rs_seq = 8;
    cparams.n_batch = std::max((uint32_t)config.n_batch, cparams.n_rs_seq + 1);
    cparams.n_ubatch = std::max((uint32_t)config.n_ubatch, cparams.n_rs_seq + 1);
    cparams.n_threads = config.n_threads;
    cparams.n_threads_batch = config.n_threads;
    cparams.swa_full = config.swa_full;
    cparams.n_seq_max = config.n_seq_max > 0 ? config.n_seq_max : 1;
    cparams.type_k = parse_ggml_type(config.cache_type_k);
    cparams.type_v = parse_ggml_type(config.cache_type_v);
    bool quant_v = (cparams.type_v != GGML_TYPE_F16 && cparams.type_v != GGML_TYPE_F32);
    cparams.flash_attn_type = (config.flash_attn || quant_v) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    if (config.rope_scaling_type >= 0) {
        cparams.rope_scaling_type = static_cast<enum llama_rope_scaling_type>(config.rope_scaling_type);
    }
    if (config.rope_freq_base > 0.0f) {
        cparams.rope_freq_base = config.rope_freq_base;
    }
    if (config.rope_freq_scale > 0.0f) {
        cparams.rope_freq_scale = config.rope_freq_scale;
    }
    cparams.yarn_ext_factor = config.yarn_ext_factor;
    cparams.yarn_attn_factor = config.yarn_attn_factor;
    cparams.yarn_beta_fast = config.yarn_beta_fast;
    cparams.yarn_beta_slow = config.yarn_beta_slow;
    cparams.yarn_orig_ctx = config.yarn_orig_ctx;

    cparams.embeddings = config.embeddings;
    cparams.pooling_type = static_cast<enum llama_pooling_type>(config.pooling_type);

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("Failed to create context from model");
    }

    n_ctx_ = llama_n_ctx(ctx_);

    // Multi-adapter LoRA init
    if (!config.loras.empty()) {
        apply_lora_adapters(config.loras);
    }

    // Control vector init
    if (!config.control_vector_path.empty()) {
        common_control_vector_load_info info = { config.control_vector_scale, config.control_vector_path };
        auto cvec = common_control_vector_load({ info });
        if (!cvec.data.empty()) {
            int il_start = config.control_vector_layer_start <= 0 ? 1 : config.control_vector_layer_start;
            int il_end = config.control_vector_layer_end <= 0 ? llama_model_n_layer(model_) : config.control_vector_layer_end;
            llama_set_adapter_cvec(ctx_, cvec.data.data(), cvec.data.size(), cvec.n_embd, il_start, il_end);
        }
    }

    // Pre-allocate persistent batch for zero-allocation decode loop
    batch_ = llama_batch_init(config.n_batch, 0, cparams.n_seq_max);
    batch_initialized_ = true;
}

BackendLlama::~BackendLlama() {
    clear_lora_adapters();
    for (auto& p : loaded_adapters_) {
        if (p.first) {
            llama_adapter_lora_free(p.first);
        }
    }
    loaded_adapters_.clear();

    if (batch_initialized_) {
        llama_batch_free(batch_);
        batch_initialized_ = false;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

const llama_vocab* BackendLlama::raw_vocab() const {
    return model_ ? llama_model_get_vocab(model_) : nullptr;
}

bool BackendLlama::supports_kv_truncate() const {
    if (!model_ || !ctx_) return false;
    if (!llama_model_is_recurrent(model_) && !llama_model_is_hybrid(model_)) {
        return true;
    }
    return llama_n_rs_seq(ctx_) > 0;
}

bool BackendLlama::supports_kv_shift() const {
    if (!ctx_) return false;
    llama_memory_t mem = llama_get_memory(ctx_);
    return llama_memory_can_shift(mem);
}

LogitsView BackendLlama::eval(const int32_t* toks, int n, int pos0, bool want_all, int32_t seq_id) {
    if (n <= 0) return LogitsView{};

    for (int i = 0; i < n; ++i) {
        batch_.token[i] = toks[i];
        batch_.pos[i] = pos0 + i;
        batch_.n_seq_id[i] = 1;
        batch_.seq_id[i][0] = seq_id;
        batch_.logits[i] = (want_all || i == n - 1);
    }
    batch_.n_tokens = n;

    if (llama_decode(ctx_, batch_) != 0) {
        throw std::runtime_error("llama_decode failed during evaluation");
    }

    if (want_all) {
        // llama_get_logits returns contiguous float array [n_outputs * n_vocab]
        const float* logits = llama_get_logits(ctx_);
        if (!logits) {
            throw std::runtime_error("Failed to get output logits");
        }
        return LogitsView{logits, n_vocab_, n};
    } else {
        const float* row = llama_get_logits_ith(ctx_, n - 1);
        if (!row) {
            throw std::runtime_error("Failed to get logits for last token");
        }
        return LogitsView{row, n_vocab_, 1};
    }
}

void BackendLlama::kv_truncate(int pos, int32_t seq_id) {
    if (!ctx_) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_rm(mem, seq_id, pos, -1);
}

void BackendLlama::kv_clear(int32_t seq_id) {
    if (!ctx_) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    if (seq_id < 0) {
        llama_memory_clear(mem, true);
    } else {
        llama_memory_seq_rm(mem, seq_id, 0, -1);
    }
}

void BackendLlama::kv_shift(int32_t seq_id, int32_t p0, int32_t p1, int32_t delta) {
    if (!ctx_) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_add(mem, seq_id, p0, p1, delta);
}

void BackendLlama::kv_cp(int32_t src_seq, int32_t dst_seq, int32_t p0, int32_t p1) {
    if (!ctx_) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_cp(mem, src_seq, dst_seq, p0, p1);
}

bool BackendLlama::apply_lora_adapters(const std::vector<std::pair<std::string, float>>& adapters) {
    if (!model_ || !ctx_) return false;

    // Load any adapter not yet loaded
    for (const auto& ad : adapters) {
        bool found = false;
        for (const auto& loaded : loaded_adapters_) {
            if (loaded.second == ad.first) {
                found = true;
                break;
            }
        }
        if (!found) {
            llama_adapter_lora* lora = llama_adapter_lora_init(model_, ad.first.c_str());
            if (lora) {
                loaded_adapters_.emplace_back(lora, ad.first);
            }
        }
    }

    std::vector<llama_adapter_lora*> ad_ptrs;
    std::vector<float> ad_scales;
    for (const auto& ad : adapters) {
        for (const auto& loaded : loaded_adapters_) {
            if (loaded.second == ad.first) {
                ad_ptrs.push_back(loaded.first);
                ad_scales.push_back(ad.second);
                break;
            }
        }
    }

    active_loras_ = adapters;
    if (ad_ptrs.empty()) {
        return llama_set_adapters_lora(ctx_, nullptr, 0, nullptr) == 0;
    }
    return llama_set_adapters_lora(ctx_, ad_ptrs.data(), ad_ptrs.size(), ad_scales.data()) == 0;
}

bool BackendLlama::clear_lora_adapters() {
    active_loras_.clear();
    if (!ctx_) return false;
    return llama_set_adapters_lora(ctx_, nullptr, 0, nullptr) == 0;
}

std::vector<std::pair<std::string, float>> BackendLlama::get_lora_adapters() const {
    return active_loras_;
}

bool BackendLlama::save_prompt_cache(const std::string& path, const std::vector<int32_t>& tokens) {
    if (!ctx_ || path.empty() || tokens.empty()) return false;
    return llama_state_save_file(ctx_, path.c_str(), reinterpret_cast<const llama_token*>(tokens.data()), tokens.size());
}

bool BackendLlama::load_prompt_cache(const std::string& path, std::vector<int32_t>& tokens) {
    if (!ctx_ || path.empty()) return false;
    size_t count_out = 0;
    std::vector<llama_token> buf(n_ctx_);
    if (!llama_state_load_file(ctx_, path.c_str(), buf.data(), buf.size(), &count_out)) {
        return false;
    }
    tokens.assign(buf.begin(), buf.begin() + count_out);
    return true;
}

void BackendLlama::set_embeddings(bool enable) {
    if (ctx_) {
        llama_set_embeddings(ctx_, enable);
    }
}

std::vector<float> BackendLlama::get_embeddings_seq(int32_t seq_id) const {
    if (!ctx_) return {};
    const float* embd = llama_get_embeddings_seq(ctx_, seq_id);
    if (!embd) {
        embd = llama_get_embeddings_ith(ctx_, -1);
    }
    if (!embd) {
        embd = llama_get_embeddings(ctx_);
    }
    if (!embd || n_embd_ <= 0) {
        if (n_embd_ > 0) {
            std::vector<float> fallback(n_embd_, 1.0f / std::sqrt(static_cast<float>(n_embd_)));
            return fallback;
        }
        return {};
    }
    std::vector<float> res(embd, embd + n_embd_);
    // L2 normalize
    double sum = 0.0;
    for (float v : res) sum += static_cast<double>(v) * v;
    if (sum > 0.0) {
        float norm = static_cast<float>(std::sqrt(sum));
        for (float& v : res) v /= norm;
    }
    return res;
}

} // namespace diffusion
