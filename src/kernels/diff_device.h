#pragma once

#include <cstdint>
#include <cstddef>

namespace diffusion {
namespace kernels {

struct DeviceDiffusionBuffers {
    int32_t* d_canvas_in = nullptr;     // [B] input tokens for current pass
    int32_t* d_canvas_out = nullptr;    // [B] updated tokens produced by pass
    float* d_confidences = nullptr;     // [B] top token confidence
    uint8_t* d_frozen_mask = nullptr;   // [B] 1 if frozen anchor, 0 if active query
    int32_t* d_active_indices = nullptr;// [B] indices of currently active positions
    int32_t* d_converged = nullptr;     // [1] flag: 1 if all active positions unchanged
    int32_t* d_active_count = nullptr;  // [1] count of remaining active positions
    size_t capacity = 0;
};

// Allocate VRAM buffers for block size max_block_size
bool diff_device_alloc(DeviceDiffusionBuffers& bufs, size_t max_block_size);
void diff_device_free(DeviceDiffusionBuffers& bufs);

// Upload initial canvas to device
void diff_device_set_canvas(
    DeviceDiffusionBuffers& bufs,
    const int32_t* h_canvas,
    size_t B,
    void* stream = nullptr
);

// Download canvas and confidences from device to host
void diff_device_get_canvas(
    const DeviceDiffusionBuffers& bufs,
    int32_t* h_canvas,
    float* h_conf,
    uint8_t* h_frozen_mask,
    size_t B,
    void* stream = nullptr
);

// Download convergence flag (synchronizes stream)
bool diff_device_is_converged(const DeviceDiffusionBuffers& bufs, void* stream = nullptr);

// Download count of active tokens remaining
int32_t diff_device_get_active_count(const DeviceDiffusionBuffers& bufs, void* stream = nullptr);

// Launch device-resident categorical reduction across active query set
// W_active: number of active rows in d_logits [W_active, n_vocab]
// d_active_indices: maps row r in [0, W_active-1] to block position b in [0, B-1]
void diff_device_sample_and_anchor(
    const float* d_logits,
    const int32_t* d_active_indices,
    size_t W_active,
    size_t n_vocab,
    float freeze_threshold,
    DeviceDiffusionBuffers& bufs,
    void* stream = nullptr
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
