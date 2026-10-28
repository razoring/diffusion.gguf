#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace diffusion {

struct LogitsView {
    const float* data = nullptr;
    int n_vocab = 0;
    int n_rows = 0;

    const float* row(int i) const {
        return (data && i >= 0 && i < n_rows) ? (data + static_cast<size_t>(i) * n_vocab) : nullptr;
    }
};

class IModelBackend {
public:
    virtual ~IModelBackend() = default;

    virtual int  n_vocab() const = 0;
    virtual int  n_ctx() const = 0;
    virtual bool supports_kv_truncate() const = 0;

    // Evaluates tokens at absolute positions [pos0, pos0 + n).
    // want_all=true -> returns logits for all n rows; want_all=false -> last row only.
    virtual LogitsView eval(const int32_t* toks, int n, int pos0, bool want_all, int32_t seq_id = 0) = 0;

    // Evaluates tokens at absolute positions [pos0, pos0 + n) with active query mask.
    // active_mask[i] != 0 indicates position i requires output logits.
    virtual LogitsView eval_active(const int32_t* toks, int n, int pos0, const uint8_t* active_mask, int32_t seq_id = 0) {
        (void)active_mask;
        return eval(toks, n, pos0, true, seq_id);
    }

    // Evaluates tokens with custom position mapping (e.g. for annealed diffusion attention schedules)
    virtual LogitsView eval_custom_pos(const int32_t* toks, int n, const int32_t* custom_pos, bool want_all = true, int32_t seq_id = 0) {
        (void)custom_pos;
        return eval(toks, n, custom_pos ? custom_pos[0] : 0, want_all, seq_id);
    }

    // Direct access to logits of the i-th token in the last batch (or nullptr if logits[i] == 0)
    virtual const float* get_logits_ith(int32_t i) const {
        (void)i;
        return nullptr;
    }

    // Backend GPU Sampler Offload
    virtual bool set_sampler(int32_t seq_id, void* sampler) { (void)seq_id; (void)sampler; return false; }
    virtual int32_t get_sampled_token(int32_t idx = 0) const { (void)idx; return -1; }

    // Drops KV cache for absolute positions >= pos
    virtual void kv_truncate(int pos, int32_t seq_id = 0) = 0;

    // Clears entire KV cache, or specific sequence if seq_id >= 0
    virtual void kv_clear(int32_t seq_id = -1) = 0;

    // KV Cache shifting and sequence management
    virtual bool supports_kv_shift() const { return false; }
    virtual void kv_shift(int32_t seq_id, int32_t p0, int32_t p1, int32_t delta) { (void)seq_id; (void)p0; (void)p1; (void)delta; }
    virtual void kv_cp(int32_t src_seq, int32_t dst_seq, int32_t p0, int32_t p1) { (void)src_seq; (void)dst_seq; (void)p0; (void)p1; }

    // LoRA Adapters
    virtual bool apply_lora_adapters(const std::vector<std::pair<std::string, float>>& adapters) { (void)adapters; return false; }
    virtual bool clear_lora_adapters() { return false; }
    virtual std::vector<std::pair<std::string, float>> get_lora_adapters() const { return {}; }

    // Prompt Cache State
    virtual bool save_prompt_cache(const std::string& path, const std::vector<int32_t>& tokens) { (void)path; (void)tokens; return false; }
    virtual bool load_prompt_cache(const std::string& path, std::vector<int32_t>& tokens) { (void)path; (void)tokens; return false; }

    // Embeddings
    virtual void set_embeddings(bool enable) { (void)enable; }
    virtual int n_embd() const { return 0; }
    virtual std::vector<float> get_embeddings_seq(int32_t seq_id = 0) const { (void)seq_id; return {}; }
};

} // namespace diffusion
