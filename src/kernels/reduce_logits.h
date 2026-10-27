#pragma once

#include <cstdint>
#include <cstddef>

namespace diffusion {
namespace kernels {

struct LogitsReduction {
    int32_t top_token;
    int32_t second_token;
    float max_logit;
    float second_logit;
    float entropy_estimate;
};

// Evaluates whether a GPU reduction backend is available for current platform
bool has_gpu_reduction();

// Host fallback SIMD reduction
void reduce_logits_cpu(
    const float* logits,
    size_t n_vocab,
    size_t batch_size,
    LogitsReduction* out_reductions
);

#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
void reduce_logits_cuda(
    const float* d_logits,
    size_t n_vocab,
    size_t batch_size,
    LogitsReduction* d_out_reductions,
    void* cuda_stream
);
#endif

#if defined(GGML_USE_HIP) || defined(DIFFUSION_USE_HIP)
void reduce_logits_hip(
    const float* d_logits,
    size_t n_vocab,
    size_t batch_size,
    LogitsReduction* d_out_reductions,
    void* hip_stream
);
#endif

#if defined(GGML_USE_METAL) || defined(DIFFUSION_USE_METAL)
void reduce_logits_metal(
    const void* d_logits_buffer,
    size_t n_vocab,
    size_t batch_size,
    void* d_out_buffer,
    void* command_queue
);
#endif

#if defined(GGML_USE_VULKAN) || defined(DIFFUSION_USE_VULKAN)
void reduce_logits_vulkan(
    const void* d_logits_buf,
    size_t n_vocab,
    size_t batch_size,
    void* d_out_buf,
    void* vk_queue
);
#endif

} // namespace kernels
} // namespace diffusion
