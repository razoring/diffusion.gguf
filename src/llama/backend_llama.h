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

    // Multi-GPU options
    int32_t split_mode = 1; // 0=none, 1=layer (default), 2=row
    int32_t main_gpu = 0;
    std::string tensor_split_str; // comma-separated proportions, e.g. "3,2" or "1,1"

    // Model adaptation (Phase 1)
    std::vector<std::pair<std::string, float>> loras;
    std::string control_vector_path;
    float control_vector_scale = 1.0f;
    int32_t control_vector_layer_start = -1;
    int32_t control_vector_layer_end = -1;

    // KV Cache Quantization (Phase 1)
    std::string cache_type_k = "f16";
    std::string cache_type_v = "f16";

    // Memory locking & mapping (Phase 2)
    bool use_mmap = true;
    bool use_mlock = false;
    bool check_tensors = false;

    // RoPE and YaRN (Phase 2)
    int32_t rope_scaling_type = -1; // -1 = default / from model
    float rope_freq_base = 0.0f;
    float rope_freq_scale = 0.0f;
    float yarn_ext_factor = -1.0f;
    float yarn_attn_factor = 1.0f;
    float yarn_beta_fast = 32.0f;
    float yarn_beta_slow = 1.0f;
    uint32_t yarn_orig_ctx = 0;

    // Prompt cache (Phase 2)
    std::string prompt_cache_path;
    bool prompt_cache_ro = false;
    bool prompt_cache_all = false;

    // Concurrency / slots (Phase 4)
    int32_t n_seq_max = 1;

    // Embeddings (Phase 4)
    bool embeddings = false;
    int32_t pooling_type = 0; // 0=none, 1=mean, 2=cls, 3=last
};

class BackendLlama : public IModelBackend {
public:
    explicit BackendLlama(const BackendConfig& config);
    ~BackendLlama() override;

    int n_vocab() const override { return n_vocab_; }
    int n_ctx() const override { return n_ctx_; }
    int n_embd() const override { return n_embd_; }
    bool supports_kv_truncate() const override;
    bool supports_kv_shift() const override;

    LogitsView eval(const int32_t* toks, int n, int pos0, bool want_all, int32_t seq_id = 0) override;
    void kv_truncate(int pos, int32_t seq_id = 0) override;
    void kv_clear(int32_t seq_id = -1) override;
    void kv_shift(int32_t seq_id, int32_t p0, int32_t p1, int32_t delta) override;
    void kv_cp(int32_t src_seq, int32_t dst_seq, int32_t p0, int32_t p1) override;

    bool apply_lora_adapters(const std::vector<std::pair<std::string, float>>& adapters) override;
    bool clear_lora_adapters() override;
    std::vector<std::pair<std::string, float>> get_lora_adapters() const override;

    bool save_prompt_cache(const std::string& path, const std::vector<int32_t>& tokens) override;
    bool load_prompt_cache(const std::string& path, std::vector<int32_t>& tokens) override;

    void set_embeddings(bool enable) override;
    std::vector<float> get_embeddings_seq(int32_t seq_id = 0) const override;

    llama_model* raw_model() const { return model_; }
    llama_context* raw_context() const { return ctx_; }
    const llama_vocab* raw_vocab() const;

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    int n_vocab_ = 0;
    int n_ctx_ = 0;
    int n_embd_ = 0;
    llama_batch batch_{};
    bool batch_initialized_ = false;

    std::vector<std::pair<std::string, float>> active_loras_;
    std::vector<std::pair<llama_adapter_lora*, std::string>> loaded_adapters_;
};

} // namespace diffusion
