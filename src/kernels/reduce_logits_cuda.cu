#include "reduce_logits.h"

#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

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

__device__ __forceinline__ void update_online_val(
    float& m1, int32_t& t1,
    float& m2, int32_t& t2,
    float& Z, float& S,
    float x, int32_t idx
) {
    if (x > m1) {
        float d = m1 - x;
        float e = (d > -20.0f) ? __expf(d) : 0.0f;
        S = (S + Z * d) * e;
        Z = Z * e + 1.0f;
        m2 = m1;
        t2 = t1;
        m1 = x;
        t1 = idx;
    } else {
        if (x > m2) {
            m2 = x;
            t2 = idx;
        }
        float d = x - m1;
        float e = (d > -20.0f) ? __expf(d) : 0.0f;
        Z += e;
        S += d * e;
    }
}

__device__ __forceinline__ void warp_reduce_online(
    float& m1, int32_t& t1,
    float& m2, int32_t& t2,
    float& Z, float& S
) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_m1 = __shfl_down_sync(mask, m1, offset);
        int32_t other_t1 = __shfl_down_sync(mask, t1, offset);
        float other_m2 = __shfl_down_sync(mask, m2, offset);
        int32_t other_t2 = __shfl_down_sync(mask, t2, offset);
        float other_Z = __shfl_down_sync(mask, Z, offset);
        float other_S = __shfl_down_sync(mask, S, offset);

        float orig_m1 = m1;
        float m = fmaxf(orig_m1, other_m1);
        float d_a = orig_m1 - m;
        float d_b = other_m1 - m;
        float e_a = (d_a > -20.0f) ? __expf(d_a) : 0.0f;
        float e_b = (d_b > -20.0f) ? __expf(d_b) : 0.0f;
        float new_S = e_a * (S + d_a * Z) + e_b * (other_S + d_b * other_Z);
        float new_Z = Z * e_a + other_Z * e_b;
        S = new_S;
        Z = new_Z;

        update_pair(m1, t1, m2, t2, other_m1, other_t1);
        update_pair(m1, t1, m2, t2, other_m2, other_t2);
    }
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
    float l_sum_exp = 0.0f;
    float l_sum_z_exp = 0.0f;

    const size_t n_vec = n_vocab / 4;
    const float4* row4 = reinterpret_cast<const float4*>(row);
    for (size_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
        float4 v = row4[i];
        int32_t base = static_cast<int32_t>(i * 4);
        update_online_val(l_max1, l_tok1, l_max2, l_tok2, l_sum_exp, l_sum_z_exp, v.x, base + 0);
        update_online_val(l_max1, l_tok1, l_max2, l_tok2, l_sum_exp, l_sum_z_exp, v.y, base + 1);
        update_online_val(l_max1, l_tok1, l_max2, l_tok2, l_sum_exp, l_sum_z_exp, v.z, base + 2);
        update_online_val(l_max1, l_tok1, l_max2, l_tok2, l_sum_exp, l_sum_z_exp, v.w, base + 3);
    }
    for (size_t i = n_vec * 4 + threadIdx.x; i < n_vocab; i += blockDim.x) {
        update_online_val(l_max1, l_tok1, l_max2, l_tok2, l_sum_exp, l_sum_z_exp, row[i], static_cast<int32_t>(i));
    }

    warp_reduce_online(l_max1, l_tok1, l_max2, l_tok2, l_sum_exp, l_sum_z_exp);

    __shared__ float s_max1[32];
    __shared__ int32_t s_tok1[32];
    __shared__ float s_max2[32];
    __shared__ int32_t s_tok2[32];
    __shared__ float s_sum_exp[32];
    __shared__ float s_sum_z[32];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    if (lane == 0) {
        s_max1[warp_id] = l_max1;
        s_tok1[warp_id] = l_tok1;
        s_max2[warp_id] = l_max2;
        s_tok2[warp_id] = l_tok2;
        s_sum_exp[warp_id] = l_sum_exp;
        s_sum_z[warp_id] = l_sum_z_exp;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b_max1 = (lane < num_warps) ? s_max1[lane] : -1e30f;
        int32_t b_tok1 = (lane < num_warps) ? s_tok1[lane] : 0;
        float b_max2 = (lane < num_warps) ? s_max2[lane] : -1e30f;
        int32_t b_tok2 = (lane < num_warps) ? s_tok2[lane] : 0;
        float b_sum_exp = (lane < num_warps) ? s_sum_exp[lane] : 0.0f;
        float b_sum_z = (lane < num_warps) ? s_sum_z[lane] : 0.0f;

        warp_reduce_online(b_max1, b_tok1, b_max2, b_tok2, b_sum_exp, b_sum_z);

        if (lane == 0) {
            float inv_z = 1.0f / fmaxf(b_sum_exp, 1e-12f);
            float p1 = fminf(1.0f, fmaxf(0.0f, inv_z));
            float p2 = fminf(1.0f - p1, fmaxf(0.0f, __expf(b_max2 - b_max1) * inv_z));
            float raw_h = fmaxf(0.0f, logf(fmaxf(b_sum_exp, 1.0f)) - b_sum_z * inv_z);
            float norm_h = fminf(1.0f, raw_h / fmaxf(logf(static_cast<float>(n_vocab)), 1.0f));

            out[b].top_token        = b_tok1;
            out[b].second_token     = b_tok2;
            out[b].max_logit        = b_max1;
            out[b].second_logit     = b_max2;
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
    constexpr int block_threads = 1024;
    dim3 grid(static_cast<unsigned int>(batch_size));
    dim3 block(block_threads);

    reduce_logits_cuda_kernel<<<grid, block, 0, stream>>>(
        d_logits,
        n_vocab,
        d_out_reductions
    );
}

void reduce_logits_cuda_to_host(const float* d_logits, size_t n_vocab, size_t batch_size, LogitsReduction* h_out, void** d_buf, size_t* d_cap, int device_id) {
    if (!d_logits || !h_out || batch_size == 0 || n_vocab == 0 || !d_buf || !d_cap) return;

    int orig_device = 0;
    cudaGetDevice(&orig_device);
    if (device_id >= 0) {
        cudaSetDevice(device_id);
    } else {
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, d_logits) == cudaSuccess && attr.device >= 0) {
            cudaSetDevice(attr.device);
        }
    }

    if (batch_size > *d_cap || !*d_buf) {
        if (*d_buf) cudaFree(*d_buf);
        *d_cap = batch_size > 64 ? batch_size : 64;
        if (cudaMalloc(d_buf, (*d_cap) * sizeof(LogitsReduction)) != cudaSuccess) {
            *d_buf = nullptr;
            *d_cap = 0;
            cudaSetDevice(orig_device);
            return;
        }
    }
    reduce_logits_cuda(d_logits, n_vocab, batch_size, static_cast<LogitsReduction*>(*d_buf), nullptr);
    cudaMemcpy(h_out, *d_buf, batch_size * sizeof(LogitsReduction), cudaMemcpyDeviceToHost);
    cudaSetDevice(orig_device);
}

void reduce_logits_cuda_free(void* d_buf, int device_id) {
    if (d_buf) {
        int orig_device = 0;
        if (device_id >= 0) {
            cudaGetDevice(&orig_device);
            cudaSetDevice(device_id);
        }
        cudaFree(d_buf);
        if (device_id >= 0) {
            cudaSetDevice(orig_device);
        }
    }
}

} // namespace kernels
} // namespace diffusion
#endif
