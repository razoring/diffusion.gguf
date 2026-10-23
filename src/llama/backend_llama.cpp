#include "backend_llama.h"
#include <cstring>
#include <stdexcept>

namespace diffusion {

BackendLlama::BackendLlama(const BackendConfig& config) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.n_gpu_layers;
    mparams.main_gpu = config.main_gpu;

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

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_batch = config.n_batch;
    cparams.n_ubatch = config.n_ubatch;
    cparams.n_threads = config.n_threads;
    cparams.n_threads_batch = config.n_threads;
    cparams.flash_attn_type = config.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.swa_full = config.swa_full;
    cparams.n_seq_max = 1;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("Failed to create context from model");
    }

    n_ctx_ = llama_n_ctx(ctx_);

    // Pre-allocate persistent batch for zero-allocation decode loop
    batch_ = llama_batch_init(config.n_batch, 0, 1);
    batch_initialized_ = true;
}

BackendLlama::~BackendLlama() {
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
    if (!model_) return false;
    return !llama_model_is_recurrent(model_) && !llama_model_is_hybrid(model_);
}

LogitsView BackendLlama::eval(const int32_t* toks, int n, int pos0, bool want_all) {
    if (n <= 0) return LogitsView{};

    for (int i = 0; i < n; ++i) {
        batch_.token[i] = toks[i];
        batch_.pos[i] = pos0 + i;
        batch_.n_seq_id[i] = 1;
        batch_.seq_id[i][0] = 0;
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

void BackendLlama::kv_truncate(int pos) {
    if (!ctx_) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_rm(mem, 0, pos, -1);
}

void BackendLlama::kv_clear() {
    if (!ctx_) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_clear(mem, true);
}

} // namespace diffusion
