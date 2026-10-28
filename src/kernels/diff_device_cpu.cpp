#include "diff_device.h"

#if !defined(GGML_USE_CUDA) && !defined(DIFFUSION_USE_CUDA)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

namespace diffusion {
namespace kernels {

bool diff_device_alloc(DeviceDiffusionBuffers& bufs, size_t max_block_size, size_t n_vocab) {
    diff_device_free(bufs);
    bufs.capacity = max_block_size;
    bufs.n_vocab = n_vocab;
    bufs.d_canvas_in = (int32_t*)calloc(max_block_size, sizeof(int32_t));
    bufs.d_canvas_out = (int32_t*)calloc(max_block_size, sizeof(int32_t));
    bufs.d_confidences = (float*)calloc(max_block_size, sizeof(float));
    bufs.d_entropies = (float*)calloc(max_block_size, sizeof(float));
    bufs.d_frozen_mask = (uint8_t*)calloc(max_block_size, sizeof(uint8_t));
    bufs.d_active_indices = (int32_t*)calloc(max_block_size, sizeof(int32_t));
    bufs.d_converged = (int32_t*)calloc(1, sizeof(int32_t));
    bufs.d_active_count = (int32_t*)calloc(1, sizeof(int32_t));
    if (n_vocab > 0) {
        bufs.d_logits_ema = (float*)calloc(max_block_size * n_vocab, sizeof(float));
    }
    return true;
}

void diff_device_free(DeviceDiffusionBuffers& bufs) {
    if (bufs.d_canvas_in) { free(bufs.d_canvas_in); bufs.d_canvas_in = nullptr; }
    if (bufs.d_canvas_out) { free(bufs.d_canvas_out); bufs.d_canvas_out = nullptr; }
    if (bufs.d_confidences) { free(bufs.d_confidences); bufs.d_confidences = nullptr; }
    if (bufs.d_entropies) { free(bufs.d_entropies); bufs.d_entropies = nullptr; }
    if (bufs.d_frozen_mask) { free(bufs.d_frozen_mask); bufs.d_frozen_mask = nullptr; }
    if (bufs.d_active_indices) { free(bufs.d_active_indices); bufs.d_active_indices = nullptr; }
    if (bufs.d_converged) { free(bufs.d_converged); bufs.d_converged = nullptr; }
    if (bufs.d_active_count) { free(bufs.d_active_count); bufs.d_active_count = nullptr; }
    if (bufs.d_logits_ema) { free(bufs.d_logits_ema); bufs.d_logits_ema = nullptr; }
    bufs.capacity = 0;
    bufs.n_vocab = 0;
}

void diff_device_set_canvas(DeviceDiffusionBuffers& bufs, const int32_t* h_canvas, size_t B, void*) {
    memcpy(bufs.d_canvas_in, h_canvas, B * sizeof(int32_t));
    memset(bufs.d_frozen_mask, 0, B * sizeof(uint8_t));
    *bufs.d_converged = 0;
    *bufs.d_active_count = (int32_t)B;
}

void diff_device_get_canvas(const DeviceDiffusionBuffers& bufs, int32_t* h_canvas, float* h_conf, uint8_t* h_frozen_mask, size_t B, void*, float* h_entropy) {
    if (h_canvas) memcpy(h_canvas, bufs.d_canvas_out, B * sizeof(int32_t));
    if (h_conf) memcpy(h_conf, bufs.d_confidences, B * sizeof(float));
    if (h_entropy && bufs.d_entropies) memcpy(h_entropy, bufs.d_entropies, B * sizeof(float));
    if (h_frozen_mask) memcpy(h_frozen_mask, bufs.d_frozen_mask, B * sizeof(uint8_t));
}

bool diff_device_is_converged(const DeviceDiffusionBuffers& bufs, void*) {
    return bufs.d_converged ? (*bufs.d_converged != 0) : false;
}

int32_t diff_device_get_active_count(const DeviceDiffusionBuffers& bufs, void*) {
    return bufs.d_active_count ? *bufs.d_active_count : 0;
}

void diff_device_sample_and_anchor(
    const float* d_logits,
    const int32_t* d_active_indices,
    size_t W_active,
    size_t n_vocab,
    float freeze_threshold,
    DeviceDiffusionBuffers& bufs,
    void*,
    float momentum_beta
) {
    for (size_t i = 0; i < bufs.capacity; ++i) {
        if (bufs.d_frozen_mask[i] == 1) {
            bufs.d_canvas_out[i] = bufs.d_canvas_in[i];
        }
    }

    for (size_t r = 0; r < W_active; ++r) {
        int32_t b = d_active_indices ? d_active_indices[r] : static_cast<int32_t>(r);
        const float* row = d_logits + (r * n_vocab);
        float* row_ema = bufs.d_logits_ema ? (bufs.d_logits_ema + (b * n_vocab)) : nullptr;

        float max_val = -1e30f;
        int32_t best_tok = 0;
        for (size_t v = 0; v < n_vocab; ++v) {
            float val = row[v];
            if (row_ema && momentum_beta > 0.0f) {
                val = momentum_beta * row_ema[v] + (1.0f - momentum_beta) * val;
                row_ema[v] = val;
            }
            if (val > max_val) {
                max_val = val;
                best_tok = (int32_t)v;
            }
        }

        double sum_exp = 0.0;
        double sum_z_exp = 0.0;
        for (size_t v = 0; v < n_vocab; ++v) {
            float val = (row_ema && momentum_beta > 0.0f) ? row_ema[v] : row[v];
            double e = std::exp((double)(val - max_val));
            sum_exp += e;
            sum_z_exp += (double)(val - max_val) * e;
        }

        float conf = sum_exp > 0.0 ? (float)(1.0 / sum_exp) : 1.0f;
        float ent = sum_exp > 0.0 ? (float)(std::log(sum_exp) - (sum_z_exp / sum_exp)) : 0.0f;

        bufs.d_canvas_out[b] = best_tok;
        bufs.d_confidences[b] = conf;
        if (bufs.d_entropies) bufs.d_entropies[b] = ent;

        if (conf >= freeze_threshold) {
            bufs.d_frozen_mask[b] = 1;
        }
    }
}

void diff_device_check_convergence(DeviceDiffusionBuffers& bufs, size_t B, void*) {
    int all_match = 1;
    int active_cnt = 0;
    for (size_t i = 0; i < B; ++i) {
        if (bufs.d_canvas_in[i] != bufs.d_canvas_out[i]) {
            all_match = 0;
        }
        if (bufs.d_frozen_mask[i] == 0) {
            active_cnt++;
        }
    }
    if (bufs.d_converged) *bufs.d_converged = all_match;
    if (bufs.d_active_count) *bufs.d_active_count = active_cnt;
}

void diff_device_swap_canvas(DeviceDiffusionBuffers& bufs) {
    std::swap(bufs.d_canvas_in, bufs.d_canvas_out);
}

} // namespace kernels
} // namespace diffusion

#endif // !GGML_USE_CUDA && !DIFFUSION_USE_CUDA
