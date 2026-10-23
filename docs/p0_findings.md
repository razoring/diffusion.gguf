# Phase 0: Research & Verification Findings

## 1. Pinned llama.cpp Submodule
- Pinned tag: `b9999` (commit `47c786924ad1ab7e91da2cdc72fcdb563780c2bd`).
- Location: `third_party/llama.cpp`.

## 2. API Signature Verification in `include/llama.h`
- **KV Truncation**:
  `llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1)` (line 725)
  with `llama_get_memory(const struct llama_context * ctx)` (line 559).
  `p0 >= 0, p1 = -1` drops all tokens with position `>= p0`.
- **Batch Evaluation Logits**:
  `llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max)` allocates batch.
  `batch.logits[i] = true` requests logits for token index `i`.
  `llama_get_logits_ith(struct llama_context * ctx, int32_t i)` retrieves float logits for the i-th token in the batch (line 1017).
- **Sampler Chain & Cloning**:
  `llama_sampler_chain_init` / `llama_sampler_chain_add` / `llama_sampler_clone(const struct llama_sampler * smpl)` (line 1299).
  `llama_sampler_i` interface (line 1249): `name`, `accept`, `apply`, `reset`, `clone`, `free`.
  Custom positional RNG sampler wraps this interface directly via `llama_sampler_init`.
- **Chat Templates**:
  `common/chat.h`: `common_chat_templates_init(const struct llama_model * model, const char * custom_template)`
  `common_chat_templates_apply(const struct common_chat_templates * tmpls, const struct common_chat_templates_inputs & inputs)` (minja Jinja parser).
- **JSON Schema to Grammar**:
  `common/json-schema-to-grammar.h`: `std::string json_schema_to_grammar(const nlohmann::ordered_json & schema, ...)`.
- **Sliding-Window Attention**:
  `llama_context_params.swa_full` (line 383) enables full-size SWA cache so tail truncation via `llama_memory_seq_rm` is always safe for Gemma models.

## 3. GGJT Legacy Format Specification
Reference: `third_party/llama.cpp/convert_llama_ggml_to_gguf.py`
- Magic: `tjgg` (`0x67676a74` little-endian).
- Versions: 1, 2, 3.
- Header:
  - Magic (4B)
  - Version (4B)
  - 7 uint32 hyperparameters: `n_vocab`, `n_embd`, `n_mult`, `n_head`, `n_layer`, `n_rot`, `ftype` (28B)
- Vocab:
  - `n_vocab` items: `len` (uint32), `bytes` (`len` bytes), `score` (float32, for version >= 2).
- Tensors (until EOF):
  - `n_dims` (uint32, 1..4)
  - `name_len` (uint32)
  - `dtype` (uint32, ggml type)
  - `dims[n_dims]` (uint32)
  - `name` (`name_len` bytes)
  - 32-byte alignment padding
  - Tensor payload: `prod(dims) * type_size / block_size` bytes.
- Quantization evolution:
  - GGJTv1: legacy Q4_0/Q4_1 without nibble interleaving adjustments.
  - GGJTv2: updated nibble ordering (PR #1405).
  - GGJTv3: half-precision fp16 scales for Q4_0/Q4_1/Q8_0 (PR #1508) matching modern ggml memory layouts.
