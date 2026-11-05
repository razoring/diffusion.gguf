#include "reduce_logits.h"

#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace diffusion {
namespace kernels {

__device__ __forceinline__ void update_pair(
    float& max1, int32_t& tok1,
    float& max2, int32_t& tok2,
    float new_val, int32_t new_idx
) {
    if (new_val > max1) {
        max2 = max1;
        tok2 = tok1;
        max1 = new_val;
        tok1 = new_idx;
    } else if (new_val > max2) {
        max2 = new_val;
        tok2 = new_idx;
    }
}

__device__ __forceinline__ void warp_reduce_top2(
    float& max1, int32_t& tok1,
    float& max2, int32_t& tok2
) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max1 = __shfl_down_sync(mask, max1, offset);
        int32_t other_tok1 = __shfl_down_sync(mask, tok1, offset);
        float other_max2 = __shfl_down_sync(mask, max2, offset);
        int32_t other_tok2 = __shfl_down_sync(mask, tok2, offset);

        update_pair(max1, tok1, max2, tok2, other_max1, other_tok1);
        update_pair(max1, tok1, max2, tok2, other_max2, other_tok2);
    }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void reduce_logits_cuda_kernel(
    const float* __restrict__ logits,
    size_t n_vocab,
    LogitsReduction* __restrict__ out
) {
    const size_t b = blockIdx.x;
    const float* row = logits + (b * n_vocab);

    float l_max1 = -1e30f;
    float l_max2 = -1e30f;
    int32_t l_tok1 = 0;
    int32_t l_tok2 = 0;

    const size_t n_vec = n_vocab / 4;
    const float4* row4 = reinterpret_cast<const float4*>(row);
    for (size_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
        float4 v = row4[i];
        int32_t base = static_cast<int32_t>(i * 4);
        update_pair(l_max1, l_tok1, l_max2, l_tok2, v.x, base + 0);
        update_pair(l_max1, l_tok1, l_max2, l_tok2, v.y, base + 1);
        update_pair(l_max1, l_tok1, l_max2, l_tok2, v.z, base + 2);
        update_pair(l_max1, l_tok1, l_max2, l_tok2, v.w, base + 3);
    }
    for (size_t i = n_vec * 4 + threadIdx.x; i < n_vocab; i += blockDim.x) {
        update_pair(l_max1, l_tok1, l_max2, l_tok2, row[i], static_cast<int32_t>(i));
    }

    warp_reduce_top2(l_max1, l_tok1, l_max2, l_tok2);

    __shared__ float s_max1[32];
    __shared__ int32_t s_tok1[32];
    __shared__ float s_max2[32];
    __shared__ int32_t s_tok2[32];
    __shared__ float s_row_max;
    __shared__ float s_row_max2;
    __shared__ int32_t s_row_tok1;
    __shared__ int32_t s_row_tok2;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    if (lane == 0) {
        s_max1[warp_id] = l_max1;
        s_tok1[warp_id] = l_tok1;
        s_max2[warp_id] = l_max2;
        s_tok2[warp_id] = l_tok2;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b_max1 = (lane < num_warps) ? s_max1[lane] : -1e30f;
        int32_t b_tok1 = (lane < num_warps) ? s_tok1[lane] : 0;
        float b_max2 = (lane < num_warps) ? s_max2[lane] : -1e30f;
        int32_t b_tok2 = (lane < num_warps) ? s_tok2[lane] : 0;

        warp_reduce_top2(b_max1, b_tok1, b_max2, b_tok2);

        if (lane == 0) {
            s_row_max  = b_max1;
            s_row_max2 = b_max2;
            s_row_tok1 = b_tok1;
            s_row_tok2 = b_tok2;
        }
    }
    __syncthreads();

    // Pass 2 (L2-hot): exact softmax partition Z and Shannon entropy sum
    const float row_max = s_row_max;
    float l_sum_exp = 0.0f;
    float l_sum_z_exp = 0.0f;

    for (size_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
        float4 v = row4[i];
        float d0 = v.x - row_max; float e0 = (d0 > -20.0f) ? __expf(d0) : 0.0f;
        float d1 = v.y - row_max; float e1 = (d1 > -20.0f) ? __expf(d1) : 0.0f;
        float d2 = v.z - row_max; float e2 = (d2 > -20.0f) ? __expf(d2) : 0.0f;
        float d3 = v.w - row_max; float e3 = (d3 > -20.0f) ? __expf(d3) : 0.0f;
        l_sum_exp   += (e0 + e1) + (e2 + e3);
        l_sum_z_exp += (d0 * e0 + d1 * e1) + (d2 * e2 + d3 * e3);
    }
    for (size_t i = n_vec * 4 + threadIdx.x; i < n_vocab; i += blockDim.x) {
        float d = row[i] - row_max;
        float e = (d > -20.0f) ? __expf(d) : 0.0f;
        l_sum_exp   += e;
        l_sum_z_exp += d * e;
    }

    l_sum_exp   = warp_reduce_sum(l_sum_exp);
    l_sum_z_exp = warp_reduce_sum(l_sum_z_exp);

    if (lane == 0) {
        s_max1[warp_id] = l_sum_exp;
        s_max2[warp_id] = l_sum_z_exp;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b_sum_exp   = (lane < num_warps) ? s_max1[lane] : 0.0f;
        float b_sum_z_exp = (lane < num_warps) ? s_max2[lane] : 0.0f;
        b_sum_exp   = warp_reduce_sum(b_sum_exp);
        b_sum_z_exp = warp_reduce_sum(b_sum_z_exp);

        if (lane == 0) {
            float inv_z = 1.0f / fmaxf(b_sum_exp, 1e-12f);
            float p1 = fminf(1.0f, fmaxf(0.0f, inv_z));
            float p2 = fminf(1.0f - p1, fmaxf(0.0f, __expf(s_row_max2 - row_max) * inv_z));
            float raw_h = fmaxf(0.0f, logf(fmaxf(b_sum_exp, 1.0f)) - b_sum_z_exp * inv_z);
            float norm_h = fminf(1.0f, raw_h / fmaxf(logf(static_cast<float>(n_vocab)), 1.0f));

            out[b].top_token        = s_row_tok1;
            out[b].second_token     = s_row_tok2;
            out[b].max_logit        = row_max;
            out[b].second_logit     = s_row_max2;
            out[b].top_prob         = p1;
            out[b].second_prob      = p2;
            out[b].entropy_estimate = norm_h;
        }
    }
}

void reduce_logits_cuda(
    const float* d_logits,
    size_t n_vocab,
    size_t batch_size,
    LogitsReduction* d_out_reductions,
    void* cuda_stream
) {
    if (!d_logits || !d_out_reductions || batch_size == 0 || n_vocab == 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
    constexpr int block_threads = 512;
    dim3 grid(static_cast<unsigned int>(batch_size));
    dim3 block(block_threads);

    reduce_logits_cuda_kernel<<<grid, block, 0, stream>>>(
        d_logits,
        n_vocab,
        d_out_reductions
    );
}

void reduce_logits_cuda_to_host(const float* d_logits, size_t n_vocab, size_t batch_size, LogitsReduction* h_out, void** d_buf, size_t* d_cap) {
    if (!d_logits || !h_out || batch_size == 0 || n_vocab == 0 || !d_buf || !d_cap) return;
    if (batch_size > *d_cap || !*d_buf) {
        if (*d_buf) cudaFree(*d_buf);
        *d_cap = batch_size > 64 ? batch_size : 64;
        if (cudaMalloc(d_buf, (*d_cap) * sizeof(LogitsReduction)) != cudaSuccess) {
            *d_buf = nullptr;
            *d_cap = 0;
            return;
        }
    }
    reduce_logits_cuda(d_logits, n_vocab, batch_size, static_cast<LogitsReduction*>(*d_buf), nullptr);
    cudaMemcpy(h_out, *d_buf, batch_size * sizeof(LogitsReduction), cudaMemcpyDeviceToHost);
}

void reduce_logits_cuda_free(void* d_buf) {
    if (d_buf) cudaFree(d_buf);
}

} // namespace kernels
} // namespace diffusion
#endif
