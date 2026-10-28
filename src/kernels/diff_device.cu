#include "diff_device.h"

#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

namespace diffusion {
namespace kernels {

// Kernel 1: Categorical reduction over active query positions with PTX warp-shuffle & entropy
__global__ void diff_sample_and_anchor_kernel(
    const float* __restrict__ logits,          // [B, n_vocab]
    const int32_t* __restrict__ active_indices,// [W_active] -> maps to canvas index b
    const int32_t* __restrict__ canvas_in,     // [B]
    int32_t* __restrict__ canvas_out,          // [B]
    float* __restrict__ confidences,           // [B]
    float* __restrict__ entropies,             // [B]
    uint8_t* __restrict__ frozen_mask,         // [B]
    float* __restrict__ logits_ema,            // [B, n_vocab] optional EMA buffer
    float momentum_beta,
    size_t W_active,
    size_t B,
    size_t n_vocab,
    float freeze_threshold,
    bool is_final_pass
) {
    size_t r = blockIdx.x;
    if (r >= W_active) return;

    int32_t b = active_indices ? active_indices[r] : static_cast<int32_t>(r + 1);
    if (b <= 0 || b >= static_cast<int32_t>(B)) return;

    const float* row = logits + (static_cast<size_t>(b - 1) * n_vocab);
    float* row_ema = logits_ema ? (logits_ema + (static_cast<size_t>(b) * n_vocab)) : nullptr;
    int tid = threadIdx.x;

    // Phase 1: Local register scan for max logit and argmax index
    float local_max = -1e30f;
    int32_t local_idx = 0;

    for (size_t v = tid; v < n_vocab; v += blockDim.x) {
        float val = row[v];
        if (row_ema && momentum_beta > 0.0f) {
            val = momentum_beta * row_ema[v] + (1.0f - momentum_beta) * val;
            row_ema[v] = val;
        }
        if (val > local_max) {
            local_max = val;
            local_idx = static_cast<int32_t>(v);
        }
    }

    // Warp-level reduction using PTX __shfl_down_sync
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(mask, local_max, offset);
        int32_t other_idx = __shfl_down_sync(mask, local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    // Block-level reduction across warps (up to 32 warps = 1024 threads)
    __shared__ float s_max_val[32];
    __shared__ int32_t s_max_idx[32];

    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        s_max_val[warp_id] = local_max;
        s_max_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        float b_max = (lane < num_warps) ? s_max_val[lane] : -1e30f;
        int32_t b_idx = (lane < num_warps) ? s_max_idx[lane] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(mask, b_max, offset);
            int32_t other_idx = __shfl_down_sync(mask, b_idx, offset);
            if (other_max > b_max) {
                b_max = other_max;
                b_idx = other_idx;
            }
        }
        if (lane == 0) {
            s_max_val[0] = b_max;
            s_max_idx[0] = b_idx;
        }
    }
    __syncthreads();

    float max_logit = s_max_val[0];
    int32_t best_token = s_max_idx[0];

    // Phase 2: Compute softmax denominator and Shannon entropy
    float local_sum_exp = 0.0f;
    float local_sum_z_exp = 0.0f;

    for (size_t v = tid; v < n_vocab; v += blockDim.x) {
        float val = (row_ema && momentum_beta > 0.0f) ? row_ema[v] : row[v];
        float exp_diff = __expf(val - max_logit);
        local_sum_exp += exp_diff;
        local_sum_z_exp += (val - max_logit) * exp_diff;
    }

    // Warp-level sum reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum_exp += __shfl_down_sync(mask, local_sum_exp, offset);
        local_sum_z_exp += __shfl_down_sync(mask, local_sum_z_exp, offset);
    }

    __shared__ float s_sum_exp[32];
    __shared__ float s_sum_z_exp[32];

    if (lane == 0) {
        s_sum_exp[warp_id] = local_sum_exp;
        s_sum_z_exp[warp_id] = local_sum_z_exp;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        float b_sum_exp = (lane < num_warps) ? s_sum_exp[lane] : 0.0f;
        float b_sum_z_exp = (lane < num_warps) ? s_sum_z_exp[lane] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            b_sum_exp += __shfl_down_sync(mask, b_sum_exp, offset);
            b_sum_z_exp += __shfl_down_sync(mask, b_sum_z_exp, offset);
        }
        if (lane == 0) {
            s_sum_exp[0] = b_sum_exp;
            s_sum_z_exp[0] = b_sum_z_exp;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float sum_exp = s_sum_exp[0];
        float sum_z_exp = s_sum_z_exp[0];
        float confidence = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1.0f;
        float entropy = (sum_exp > 0.0f) ? (__logf(sum_exp) - (sum_z_exp / sum_exp)) : 0.0f;

        canvas_out[b] = best_token;
        confidences[b] = confidence;
        if (entropies) entropies[b] = entropy;
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

// Kernel 3: Dynamic frontier advance, convergence check, and active token count
__global__ void diff_check_convergence_kernel(
    const int32_t* __restrict__ canvas_in,
    const int32_t* __restrict__ canvas_out,
    const float* __restrict__ confidences,
    uint8_t* __restrict__ frozen_mask,
    int32_t* __restrict__ converged_flag,
    int32_t* __restrict__ active_count,
    size_t B,
    float freeze_threshold,
    int pass_m,
    bool is_final_pass
) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Frontier starts at 0 (prefix token is always frozen)
        frozen_mask[0] = 1;
        int frontier = 1;
        while (frontier < static_cast<int>(B) && frozen_mask[frontier] == 1) {
            frontier++;
        }

        // Advance frontier contiguously along stabilized or high-confidence tokens
        while (frontier < static_cast<int>(B)) {
            // In pass 1, frontier cannot advance past token 1
            if (pass_m == 1 && frontier > 1) {
                break;
            }

            int b = frontier;
            bool stable = (pass_m > 1) && (canvas_in[b] == canvas_out[b]);
            bool confident = (confidences[b] >= freeze_threshold);

            if (stable) {
                frozen_mask[b] = 1;
                frontier++;
            } else if (confident) {
                // Token b verified from frozen prefix, but successor b+1 was evaluated on stale canvas_in[b].
                // Freeze b and stop frontier to prevent hallucination propagation.
                frozen_mask[b] = 1;
                frontier++;
                break;
            } else {
                break; // Frontier stops at first unverified token
            }
        }

        // Count active (unfrozen) tokens and check overall stationarity
        int active_cnt = 0;
        int all_match = 1;
        for (size_t i = 0; i < B; ++i) {
            if (frozen_mask[i] == 0) active_cnt++;
            if (canvas_in[i] != canvas_out[i]) all_match = 0;
        }

        *converged_flag = all_match;
        *active_count = active_cnt;
    }
}

struct ScopedCudaDevice {
    int prev_dev = -1;
    explicit ScopedCudaDevice(int target_dev) {
        if (target_dev >= 0) {
            cudaGetDevice(&prev_dev);
            if (prev_dev != target_dev) {
                cudaSetDevice(target_dev);
            }
        }
    }
    ~ScopedCudaDevice() {
        if (prev_dev >= 0) {
            cudaSetDevice(prev_dev);
        }
    }
};

bool diff_device_alloc(DeviceDiffusionBuffers& bufs, size_t max_block_size, size_t n_vocab) {
    cudaGetDevice(&bufs.device_id);
    diff_device_free(bufs);
    ScopedCudaDevice scope(bufs.device_id);
    bufs.capacity = max_block_size;
    bufs.n_vocab = n_vocab;

    cudaError_t err;
    err = cudaMalloc(&bufs.d_canvas_in, max_block_size * sizeof(int32_t));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&bufs.d_canvas_out, max_block_size * sizeof(int32_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_confidences, max_block_size * sizeof(float));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_entropies, max_block_size * sizeof(float));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_frozen_mask, max_block_size * sizeof(uint8_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_active_indices, max_block_size * sizeof(int32_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_converged, sizeof(int32_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    err = cudaMalloc(&bufs.d_active_count, sizeof(int32_t));
    if (err != cudaSuccess) { diff_device_free(bufs); return false; }

    if (n_vocab > 0) {
        err = cudaMalloc(&bufs.d_logits, max_block_size * n_vocab * sizeof(float));
        if (err != cudaSuccess) { diff_device_free(bufs); return false; }

        err = cudaMalloc(&bufs.d_logits_ema, max_block_size * n_vocab * sizeof(float));
        if (err != cudaSuccess) { diff_device_free(bufs); return false; }
        cudaMemset(bufs.d_logits_ema, 0, max_block_size * n_vocab * sizeof(float));
    }

    cudaMemset(bufs.d_frozen_mask, 0, max_block_size * sizeof(uint8_t));
    cudaMemset(bufs.d_converged, 0, sizeof(int32_t));
    cudaMemset(bufs.d_active_count, 0, sizeof(int32_t));
    return true;
}

void diff_device_free(DeviceDiffusionBuffers& bufs) {
    ScopedCudaDevice scope(bufs.device_id);
    if (bufs.d_canvas_in) { cudaFree(bufs.d_canvas_in); bufs.d_canvas_in = nullptr; }
    if (bufs.d_canvas_out) { cudaFree(bufs.d_canvas_out); bufs.d_canvas_out = nullptr; }
    if (bufs.d_confidences) { cudaFree(bufs.d_confidences); bufs.d_confidences = nullptr; }
    if (bufs.d_entropies) { cudaFree(bufs.d_entropies); bufs.d_entropies = nullptr; }
    if (bufs.d_frozen_mask) { cudaFree(bufs.d_frozen_mask); bufs.d_frozen_mask = nullptr; }
    if (bufs.d_active_indices) { cudaFree(bufs.d_active_indices); bufs.d_active_indices = nullptr; }
    if (bufs.d_converged) { cudaFree(bufs.d_converged); bufs.d_converged = nullptr; }
    if (bufs.d_active_count) { cudaFree(bufs.d_active_count); bufs.d_active_count = nullptr; }
    if (bufs.d_logits) { cudaFree(bufs.d_logits); bufs.d_logits = nullptr; }
    if (bufs.d_logits_ema) { cudaFree(bufs.d_logits_ema); bufs.d_logits_ema = nullptr; }
    if (bufs.d_branch_logprobs) { cudaFree(bufs.d_branch_logprobs); bufs.d_branch_logprobs = nullptr; }
    if (bufs.d_winning_branch) { cudaFree(bufs.d_winning_branch); bufs.d_winning_branch = nullptr; }
    bufs.capacity = 0;
    bufs.n_vocab = 0;
}

void diff_device_set_canvas(
    DeviceDiffusionBuffers& bufs,
    const int32_t* h_canvas,
    const uint8_t* h_frozen_mask,
    size_t B,
    void* stream
) {
    ScopedCudaDevice scope(bufs.device_id);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(bufs.d_canvas_in, h_canvas, B * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(bufs.d_canvas_out, h_canvas, B * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    if (h_frozen_mask) {
        cudaMemcpyAsync(bufs.d_frozen_mask, h_frozen_mask, B * sizeof(uint8_t), cudaMemcpyHostToDevice, s);
    } else {
        cudaMemsetAsync(bufs.d_frozen_mask, 0, B * sizeof(uint8_t), s);
        uint8_t one = 1;
        cudaMemcpyAsync(bufs.d_frozen_mask, &one, sizeof(uint8_t), cudaMemcpyHostToDevice, s);
    }
    cudaMemsetAsync(bufs.d_converged, 0, sizeof(int32_t), s);
    int32_t b_init = static_cast<int32_t>(B > 1 ? B - 1 : 0);
    cudaMemcpyAsync(bufs.d_active_count, &b_init, sizeof(int32_t), cudaMemcpyHostToDevice, s);
}

void diff_device_get_canvas(
    const DeviceDiffusionBuffers& bufs,
    int32_t* h_canvas,
    float* h_conf,
    uint8_t* h_frozen_mask,
    size_t B,
    void* stream,
    float* h_entropy
) {
    ScopedCudaDevice scope(bufs.device_id);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (h_canvas) {
        cudaMemcpyAsync(h_canvas, bufs.d_canvas_out, B * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    }
    if (h_conf) {
        cudaMemcpyAsync(h_conf, bufs.d_confidences, B * sizeof(float), cudaMemcpyDeviceToHost, s);
    }
    if (h_entropy && bufs.d_entropies) {
        cudaMemcpyAsync(h_entropy, bufs.d_entropies, B * sizeof(float), cudaMemcpyDeviceToHost, s);
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
    ScopedCudaDevice scope(bufs.device_id);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int32_t h_conv = 0;
    cudaMemcpyAsync(&h_conv, bufs.d_converged, sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    if (s) cudaStreamSynchronize(s);
    else cudaDeviceSynchronize();
    return h_conv != 0;
}

int32_t diff_device_get_active_count(const DeviceDiffusionBuffers& bufs, void* stream) {
    ScopedCudaDevice scope(bufs.device_id);
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
    size_t B,
    size_t n_vocab,
    float freeze_threshold,
    DeviceDiffusionBuffers& bufs,
    void* stream,
    float momentum_beta,
    int pass_m,
    bool is_final_pass
) {
    if (W_active == 0 || B <= 1) return;
    ScopedCudaDevice scope(bufs.device_id);
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // 1. Copy frozen tokens to canvas_out
    int threads = 256;
    int blocks = static_cast<int>((B + threads - 1) / threads);
    diff_copy_frozen_kernel<<<blocks, threads, 0, s>>>(
        bufs.d_canvas_in,
        bufs.d_canvas_out,
        bufs.d_frozen_mask,
        B
    );

    // 2. If logits are in host memory and d_logits buffer is present, copy asynchronously
    const float* actual_d_logits = d_logits;
    cudaPointerAttributes ptr_attr;
    cudaError_t p_err = cudaPointerGetAttributes(&ptr_attr, d_logits);
    if (p_err != cudaSuccess) cudaGetLastError();
    bool is_device_ptr = (p_err == cudaSuccess && (ptr_attr.type == cudaMemoryTypeDevice || ptr_attr.type == cudaMemoryTypeManaged));

    if (!is_device_ptr && bufs.d_logits) {
        cudaMemcpyAsync(bufs.d_logits, d_logits, B * n_vocab * sizeof(float), cudaMemcpyHostToDevice, s);
        actual_d_logits = bufs.d_logits;
    }

    // 3. Launch PTX warp-shuffle categorical reduction kernel (1024 threads per row block)
    dim3 grid(static_cast<unsigned int>(W_active), 1, 1);
    dim3 block(1024, 1, 1);
    diff_sample_and_anchor_kernel<<<grid, block, 0, s>>>(
        actual_d_logits,
        d_active_indices,
        bufs.d_canvas_in,
        bufs.d_canvas_out,
        bufs.d_confidences,
        bufs.d_entropies,
        bufs.d_frozen_mask,
        bufs.d_logits_ema,
        momentum_beta,
        W_active,
        B,
        n_vocab,
        freeze_threshold,
        is_final_pass
    );

    // 4. Advance dynamic frontier and check convergence across canvas
    diff_check_convergence_kernel<<<1, 32, 0, s>>>(
        bufs.d_canvas_in,
        bufs.d_canvas_out,
        bufs.d_confidences,
        bufs.d_frozen_mask,
        bufs.d_converged,
        bufs.d_active_count,
        B,
        freeze_threshold,
        pass_m,
        is_final_pass
    );

    if (s) {
        cudaStreamSynchronize(s);
    } else {
        cudaDeviceSynchronize();
    }
}

void diff_device_check_convergence(
    DeviceDiffusionBuffers& bufs,
    size_t B,
    float freeze_threshold,
    int pass_m,
    bool is_final_pass,
    void* stream
) {
    ScopedCudaDevice scope(bufs.device_id);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    diff_check_convergence_kernel<<<1, 32, 0, s>>>(
        bufs.d_canvas_in,
        bufs.d_canvas_out,
        bufs.d_confidences,
        bufs.d_frozen_mask,
        bufs.d_converged,
        bufs.d_active_count,
        B,
        freeze_threshold,
        pass_m,
        is_final_pass
    );
}

void diff_device_swap_canvas(DeviceDiffusionBuffers& bufs) {
    ScopedCudaDevice scope(bufs.device_id);
    std::swap(bufs.d_canvas_in, bufs.d_canvas_out);
}

int diff_device_get_current_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

} // namespace kernels
} // namespace diffusion

#endif // GGML_USE_CUDA || DIFFUSION_USE_CUDA
