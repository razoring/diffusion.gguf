#pragma once

#include <cstdint>
#include <cstddef>

namespace diffusion {
namespace kernels {

struct DeviceDiffusionBuffers {
    int32_t* d_canvas_in = nullptr;     // [B] input tokens for current pass
    int32_t* d_canvas_out = nullptr;    // [B] updated tokens produced by pass
    float* d_confidences = nullptr;     // [B] top token confidence
    float* d_entropies = nullptr;       // [B] Shannon entropy of distribution
    uint8_t* d_frozen_mask = nullptr;   // [B] 1 if frozen anchor, 0 if active query
    int32_t* d_active_indices = nullptr;// [B] indices of currently active positions
    int32_t* d_converged = nullptr;     // [1] flag: 1 if all active positions unchanged
    int32_t* d_active_count = nullptr;  // [1] count of remaining active positions
    float* d_logits = nullptr;          // [max_block_size * n_vocab] device logits buffer
    float* d_logits_ema = nullptr;      // [max_block_size * n_vocab] momentum-smoothed logits
    float* d_branch_logprobs = nullptr; // [multi_candidate_k] cumulative branch logprobs
    int32_t* d_winning_branch = nullptr;// [1] winning branch index
    size_t capacity = 0;
    size_t n_vocab = 0;
};

// Allocate VRAM buffers for block size max_block_size and vocabulary size n_vocab
bool diff_device_alloc(DeviceDiffusionBuffers& bufs, size_t max_block_size, size_t n_vocab = 262144);
void diff_device_free(DeviceDiffusionBuffers& bufs);

// Upload initial canvas to device
void diff_device_set_canvas(
    DeviceDiffusionBuffers& bufs,
    const int32_t* h_canvas,
    size_t B,
    void* stream = nullptr
);

// Download canvas, confidences, and optional entropies from device to host
void diff_device_get_canvas(
    const DeviceDiffusionBuffers& bufs,
    int32_t* h_canvas,
    float* h_conf,
    uint8_t* h_frozen_mask,
    size_t B,
    void* stream = nullptr,
    float* h_entropy = nullptr
);

// Download convergence flag (synchronizes stream)
bool diff_device_is_converged(const DeviceDiffusionBuffers& bufs, void* stream = nullptr);

// Download count of active tokens remaining
int32_t diff_device_get_active_count(const DeviceDiffusionBuffers& bufs, void* stream = nullptr);

// Launch device-resident categorical reduction across active query set
// Supports optional momentum smoothing (momentum_beta > 0)
void diff_device_sample_and_anchor(
    const float* d_logits,
    const int32_t* d_active_indices,
    size_t W_active,
    size_t n_vocab,
    float freeze_threshold,
    DeviceDiffusionBuffers& bufs,
    void* stream = nullptr,
    float momentum_beta = 0.0f
);

// Launch device warp ballot convergence check comparing d_canvas_in vs d_canvas_out
void diff_device_check_convergence(
    DeviceDiffusionBuffers& bufs,
    size_t B,
    void* stream = nullptr
);

// Swap d_canvas_in and d_canvas_out pointers
void diff_device_swap_canvas(DeviceDiffusionBuffers& bufs);

} // namespace kernels
} // namespace diffusion

