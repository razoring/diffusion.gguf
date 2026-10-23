#pragma once

#include "diffusion/backend.h"
#include "llama.h"
#include <string>
#include <vector>

namespace diffusion {

struct BackendConfig {
    std::string model_path;
    int32_t n_ctx = 2048;
    int32_t n_batch = 512;
    int32_t n_ubatch = 512;
    int32_t n_gpu_layers = 0;
    int32_t n_threads = 4;
    bool flash_attn = false;
    bool swa_full = true;
};

class BackendLlama : public IModelBackend {
public:
    explicit BackendLlama(const BackendConfig& config);
    ~BackendLlama() override;

    int n_vocab() const override { return n_vocab_; }
    int n_ctx() const override { return n_ctx_; }
    bool supports_kv_truncate() const override;

    LogitsView eval(const int32_t* toks, int n, int pos0, bool want_all) override;
    void kv_truncate(int pos) override;
    void kv_clear() override;

    llama_model* raw_model() const { return model_; }
    llama_context* raw_context() const { return ctx_; }
    const llama_vocab* raw_vocab() const;

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    int n_vocab_ = 0;
    int n_ctx_ = 0;
    llama_batch batch_{};
    bool batch_initialized_ = false;
};

} // namespace diffusion
