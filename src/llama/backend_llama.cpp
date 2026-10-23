#include "backend_llama.h"
#include <cstring>
#include <stdexcept>

namespace diffusion {

BackendLlama::BackendLlama(const BackendConfig& config) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.n_gpu_layers;

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
}

BackendLlama::~BackendLlama() {
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

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (want_all || i == n - 1);
    }
    batch.n_tokens = n;

    if (llama_decode(ctx_, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed during evaluation");
    }

    if (want_all) {
        logits_buf_.resize(static_cast<size_t>(n) * n_vocab_);
        for (int i = 0; i < n; ++i) {
            const float* row = llama_get_logits_ith(ctx_, i);
            if (!row) {
                llama_batch_free(batch);
                throw std::runtime_error("Failed to get logits for batch index " + std::to_string(i));
            }
            std::memcpy(logits_buf_.data() + static_cast<size_t>(i) * n_vocab_, row, n_vocab_ * sizeof(float));
        }
        llama_batch_free(batch);
        return LogitsView{logits_buf_.data(), n_vocab_, n};
    } else {
        logits_buf_.resize(n_vocab_);
        const float* row = llama_get_logits_ith(ctx_, n - 1);
        if (!row) {
            llama_batch_free(batch);
            throw std::runtime_error("Failed to get logits for last token");
        }
        std::memcpy(logits_buf_.data(), row, n_vocab_ * sizeof(float));
        llama_batch_free(batch);
        return LogitsView{logits_buf_.data(), n_vocab_, 1};
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
