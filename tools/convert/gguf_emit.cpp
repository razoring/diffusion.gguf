#include "gguf_emit.h"
#include "quant_repack.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace diffusion {

static std::string map_tensor_name(const std::string& name) {
    if (name == "tok_embeddings.weight") return "token_embd.weight";
    if (name == "norm.weight") return "output_norm.weight";
    if (name == "output.weight") return "output.weight";

    // layers.N.attention.wq.weight -> blk.N.attn_q.weight
    if (name.rfind("layers.", 0) == 0) {
        size_t dot1 = name.find('.', 7);
        if (dot1 != std::string::npos) {
            std::string layer_num = name.substr(7, dot1 - 7);
            std::string rest = name.substr(dot1 + 1);
            if (rest == "attention.wq.weight") return "blk." + layer_num + ".attn_q.weight";
            if (rest == "attention.wk.weight") return "blk." + layer_num + ".attn_k.weight";
            if (rest == "attention.wv.weight") return "blk." + layer_num + ".attn_v.weight";
            if (rest == "attention.wo.weight") return "blk." + layer_num + ".attn_output.weight";
            if (rest == "attention_norm.weight") return "blk." + layer_num + ".attn_norm.weight";
            if (rest == "feed_forward.w1.weight") return "blk." + layer_num + ".ffn_gate.weight";
            if (rest == "feed_forward.w2.weight") return "blk." + layer_num + ".ffn_down.weight";
            if (rest == "feed_forward.w3.weight") return "blk." + layer_num + ".ffn_up.weight";
            if (rest == "ffn_norm.weight") return "blk." + layer_num + ".ffn_norm.weight";
        }
    }
    return name;
}

static int32_t classify_token(const std::string& s) {
    if (s == "<unk>") return 2; // UNKNOWN
    if (s == "<s>" || s == "</s>") return 3; // CONTROL
    if (s.size() == 6 && s.rfind("<0x", 0) == 0 && s.back() == '>') return 6; // BYTE
    return 1; // NORMAL
}

bool emit_gguf_file(
    const GgjtReader& reader,
    const std::string& out_path,
    const ConvertOptions& options
) {
    struct gguf_context* ctx = gguf_init_empty();
    if (!ctx) {
        std::cerr << "Failed to initialize GGUF context\n";
        return false;
    }

    // 1. General model metadata
    gguf_set_val_str(ctx, "general.architecture", "llama");
    gguf_set_val_str(ctx, "general.name", options.name.empty() ? "llama" : options.name.c_str());
    gguf_set_val_u32(ctx, "general.file_type", reader.hparams().ftype);

    // 2. LLaMA architecture hyperparameters
    gguf_set_val_u32(ctx, "llama.context_length", options.ctx_len);
    gguf_set_val_u32(ctx, "llama.embedding_length", reader.hparams().n_embd);
    gguf_set_val_u32(ctx, "llama.block_count", reader.hparams().n_layer);
    gguf_set_val_u32(ctx, "llama.attention.head_count", reader.hparams().n_head);

    int gqa = options.gqa > 0 ? options.gqa : 1;
    gguf_set_val_u32(ctx, "llama.attention.head_count_kv", reader.hparams().n_head / gqa);

    int32_t n_mult = reader.hparams().n_mult;
    int32_t n_embd = reader.hparams().n_embd;
    int32_t n_ff = (n_mult > 0) ? (((2 * (4 * n_embd) / 3 + n_mult - 1) / n_mult) * n_mult) : (4 * n_embd);
    gguf_set_val_u32(ctx, "llama.feed_forward_length", n_ff);

    gguf_set_val_u32(ctx, "llama.rope.dimension_count", reader.hparams().n_rot);
    gguf_set_val_f32(ctx, "llama.attention.layer_norm_rms_epsilon", options.eps);
    gguf_set_val_f32(ctx, "llama.rope.freq_base", 10000.0f);

    // 3. Tokenizer
    std::vector<const char*> token_ptrs;
    std::vector<float> scores;
    std::vector<int32_t> types;
    token_ptrs.reserve(reader.vocab().size());
    scores.reserve(reader.vocab().size());
    types.reserve(reader.vocab().size());

    for (const auto& item : reader.vocab()) {
        token_ptrs.push_back(item.text.c_str());
        scores.push_back(item.score);
        types.push_back(classify_token(item.text));
    }

    gguf_set_val_str(ctx, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(ctx, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(ctx, "tokenizer.ggml.unknown_token_id", 0);

    // 4. Repack & Add Tensors
    std::vector<std::vector<uint8_t>> repacked_buffers;
    repacked_buffers.reserve(reader.tensors().size());

    for (const auto& raw_t : reader.tensors()) {
        const uint8_t* in_data = reader.tensor_data(raw_t);
        if (!in_data) {
            std::cerr << "Missing data for tensor: " << raw_t.name << "\n";
            gguf_free(ctx);
            return false;
        }

        std::vector<uint8_t> out_buf;
        const uint8_t* final_data = nullptr;
        size_t final_size = 0;

        if (!repack_tensor_data(
            reader.version(),
            raw_t.ggml_type,
            raw_t.ne,
            raw_t.n_dims,
            in_data,
            raw_t.data_size,
            out_buf,
            final_data,
            final_size
        )) {
            std::cerr << "Repack failed for tensor: " << raw_t.name << "\n";
            gguf_free(ctx);
            return false;
        }

        if (!out_buf.empty()) {
            repacked_buffers.push_back(std::move(out_buf));
            final_data = repacked_buffers.back().data();
        }

        struct ggml_tensor t;
        std::memset(&t, 0, sizeof(t));
        std::string new_name = map_tensor_name(raw_t.name);
        ggml_set_name(&t, new_name.c_str());
        t.type = static_cast<enum ggml_type>(raw_t.ggml_type);
        for (int i = 0; i < 4; ++i) {
            t.ne[i] = raw_t.ne[i];
        }

        t.nb[0] = ggml_type_size(t.type);
        t.nb[1] = t.nb[0] * (t.ne[0] / ggml_blck_size(t.type));
        t.nb[2] = t.nb[1] * t.ne[1];
        t.nb[3] = t.nb[2] * t.ne[2];
        t.data = const_cast<uint8_t*>(final_data);

        gguf_add_tensor(ctx, &t);
        gguf_set_tensor_data(ctx, new_name.c_str(), final_data);
    }

    std::cout << "Writing GGUF output to " << out_path << " (" << reader.tensors().size() << " tensors)...\n";
    bool ok = gguf_write_to_file(ctx, out_path.c_str(), false);
    gguf_free(ctx);

    return ok;
}

} // namespace diffusion
