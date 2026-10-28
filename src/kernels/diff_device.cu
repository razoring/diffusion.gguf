#include "diff_device.h"

#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

namespace diffusion {
namespace kernels {

// Kernel 1: Categorical reduction over active query positions
__global__ void diff_sample_and_anchor_kernel(
    const float* __restrict__ logits,          // [W_active, n_vocab]
    const int32_t* __restrict__ active_indices,// [W_active] -> maps to canvas index b
    int32_t* __restrict__ canvas_out,          // [B]
    float* __restrict__ confidences,           // [B]
    uint8_t* __restrict__ frozen_mask,         // [B]
    size_t W_active,
    size_t n_vocab,
    float freeze_threshold
) {
    size_t r = blockIdx.x; // Row index in active queries
    if (r >= W_active) return;

    int32_t b = active_indices[r]; // Target position in canvas [0, B-1]
    const float* row = logits + (r * n_vocab);
    int tid = threadIdx.x;

    // Phase 1: Block tree reduction for max logit and argmax token ID
    __shared__ float s_max_val[512];
    __shared__ int32_t s_max_idx[512];

    float local_max = -1e30f;
    int32_t local_idx = 0;

    for (size_t v = tid; v < n_vocab; v += blockDim.x) {
        float val = row[v];
        if (val > local_max) {
            local_max = val;
            local_idx = static_cast<int32_t>(v);
        }
    }

    s_max_val[tid] = local_max;
    s_max_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max_val[tid + stride] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + stride];
                s_max_idx[tid] = s_max_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    float max_logit = s_max_val[0];
    int32_t best_token = s_max_idx[0];

    // Phase 2: Compute softmax denominator for confidence estimate
    float local_sum_exp = 0.0f;
    for (size_t v = tid; v < n_vocab; v += blockDim.x) {
        local_sum_exp += __expf(row[v] - max_logit);
    }

    __shared__ float s_sum_exp[512];
    s_sum_exp[tid] = local_sum_exp;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_exp[tid] += s_sum_exp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum_exp = s_sum_exp[0];
        float confidence = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1.0f;
        canvas_out[b] = best_token;
        confidences[b] = confidence;

        if (confidence >= freeze_threshold) {
            frozen_mask[b] = 1; // Newly frozen anchor
        }
    }
}

// Kernel 2: Copy frozen anchors from canvas_in to canvas_out
__global__ void diff_copy_frozen_kernel(
    const int32_t* __restrict__ canvas_in,
    int32_t* __restrict__ canvas_out,
    const uint8_t* __restrict__ frozen_mask,
    size_t B
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B && frozen_mask[idx] == 1) {
        canvas_out[idx] = canvas_in[idx];
    }
}

// Kernel 3: Warp-ballot convergence check and active token count
__global__ void diff_check_convergence_kernel(
    const int32_t* __restrict__ canvas_in,
    const int32_t* __restrict__ canvas_out,
    const uint8_t* __restrict__ frozen_mask,
    int32_t* __restrict__ converged_flag,
    int32_t* __restrict__ active_count,
    size_t B
) {
    int tid = threadIdx.x;
    int is_match = 1;
    int is_active = 0;

    if (tid < static_cast<int>(B)) {
        is_match = (canvas_in[tid] == canvas_out[tid]) ? 1 : 0;
        is_active = (frozen_mask[tid] == 0) ? 1 : 0;
    }

    // Warp ballot for match
    unsigned int mask = __activemask();
    int all_match = __all_sync(mask, (tid < static_cast<int>(B)) ? is_match : 1);

    // Warp reduction for active count
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        is_active += __shfl_down_sync(mask, is_active, offset);
    }

    if (tid == 0) {
        *converged_flag = all_match;
        *active_count = is_active;
    }
}

bool diff_device_alloc(DeviceDiffusionBuffers& bufs, size_t max_block_size) {
    diff_device_free(bufs);
    bufs.capacity = max_block_size;

    cudaError_t err;
    err = cudaMalloc(&bufs.d_canvas_in, max_block_size * sizeof(int32_t));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&bufs.d_canvas_out, max_block_size * sizeof(int32_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_confidences, max_block_size * sizeof(float));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_frozen_mask, max_block_size * sizeof(uint8_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_active_indices, max_block_size * sizeof(int32_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_converged, sizeof(int32_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_active_count, sizeof(int32_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    cudaMemset(bufs.d_frozen_mask, 0, max_block_size * sizeof(uint8_t));
    cudaMemset(bufs.d_converged, 0, sizeof(int32_t));
    cudaMemset(bufs.d_active_count, 0, sizeof(int32_t));
    return true;
}

void diff_device_free(DeviceDiffusionBuffers& bufs) {
    if (bufs.d_canvas_in) { cudaFree(bufs.d_canvas_in); bufs.d_canvas_in = nullptr; }
    if (bufs.d_canvas_out) { cudaFree(bufs.d_canvas_out); bufs.d_canvas_out = nullptr; }
    if (bufs.d_confidences) { cudaFree(bufs.d_confidences); bufs.d_confidences = nullptr; }
    if (bufs.d_frozen_mask) { cudaFree(bufs.d_frozen_mask); bufs.d_frozen_mask = nullptr; }
    if (bufs.d_active_indices) { cudaFree(bufs.d_active_indices); bufs.d_active_indices = nullptr; }
    if (bufs.d_converged) { cudaFree(bufs.d_converged); bufs.d_converged = nullptr; }
    if (bufs.d_active_count) { cudaFree(bufs.d_active_count); bufs.d_active_count = nullptr; }
    bufs.capacity = 0;
}

void diff_device_set_canvas(
    DeviceDiffusionBuffers& bufs,
    const int32_t* h_canvas,
    size_t B,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(bufs.d_canvas_in, h_canvas, B * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemsetAsync(bufs.d_frozen_mask, 0, B * sizeof(uint8_t), s);
    cudaMemsetAsync(bufs.d_converged, 0, sizeof(int32_t), s);
    int32_t b_init = static_cast<int32_t>(B);
    cudaMemcpyAsync(bufs.d_active_count, &b_init, sizeof(int32_t), cudaMemcpyHostToDevice, s);
}

void diff_device_get_canvas(
    const DeviceDiffusionBuffers& bufs,
    int32_t* h_canvas,
    float* h_conf,
    uint8_t* h_frozen_mask,
    size_t B,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (h_canvas) {
        cudaMemcpyAsync(h_canvas, bufs.d_canvas_out, B * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    }
    if (h_conf) {
        cudaMemcpyAsync(h_conf, bufs.d_confidences, B * sizeof(float), cudaMemcpyDeviceToHost, s);
    }
    if (h_frozen_mask) {
        cudaMemcpyAsync(h_frozen_mask, bufs.d_frozen_mask, B * sizeof(uint8_t), cudaMemcpyDeviceToHost, s);
    }
    if (s) {
        cudaStreamSynchronize(s);
    } else {
        cudaDeviceSynchronize();
    }
}

bool diff_device_is_converged(const DeviceDiffusionBuffers& bufs, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int32_t h_conv = 0;
    cudaMemcpyAsync(&h_conv, bufs.d_converged, sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    if (s) cudaStreamSynchronize(s);
    else cudaDeviceSynchronize();
    return h_conv != 0;
}

int32_t diff_device_get_active_count(const DeviceDiffusionBuffers& bufs, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int32_t h_count = 0;
    cudaMemcpyAsync(&h_count, bufs.d_active_count, sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    if (s) cudaStreamSynchronize(s);
    else cudaDeviceSynchronize();
    return h_count;
}

void diff_device_sample_and_anchor(
    const float* d_logits,
    const int32_t* d_active_indices,
    size_t W_active,
    size_t n_vocab,
    float freeze_threshold,
    DeviceDiffusionBuffers& bufs,
    void* stream
) {
    if (W_active == 0) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // 1. Copy frozen tokens to canvas_out
    int threads = 32;
    int blocks = (static_cast<int>(bufs.capacity) + threads - 1) / threads;
    diff_copy_frozen_kernel<<<blocks, threads, 0, s>>>(
        bufs.d_canvas_in,
        bufs.d_canvas_out,
        bufs.d_frozen_mask,
        bufs.capacity
    );

    // 2. Reduce active query set positions
    diff_sample_and_anchor_kernel<<<static_cast<int>(W_active), 512, 0, s>>>(
        d_logits,
        d_active_indices,
        bufs.d_canvas_out,
        bufs.d_confidences,
        bufs.d_frozen_mask,
        W_active,
        n_vocab,
        freeze_threshold
    );
}

void diff_device_check_convergence(
    DeviceDiffusionBuffers& bufs,
    size_t B,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    diff_check_convergence_kernel<<<1, 32, 0, s>>>(
        bufs.d_canvas_in,
        bufs.d_canvas_out,
        bufs.d_frozen_mask,
        bufs.d_converged,
        bufs.d_active_count,
        B
    );
}

void diff_device_swap_canvas(DeviceDiffusionBuffers& bufs) {
    std::swap(bufs.d_canvas_in, bufs.d_canvas_out);
}

} // namespace kernels
} // namespace diffusion

#endif // GGML_USE_CUDA || DIFFUSION_USE_CUDA
