#include "reduce_logits.h"

#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace diffusion {
namespace kernels {

struct ValIdx {
    float val;
    int32_t idx;
};

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

__global__ void reduce_logits_cuda_kernel(
    const float* __restrict__ logits,
    size_t n_vocab,
    LogitsReduction* __restrict__ out
) {
    // Each block reduces one batch element
    const size_t b = blockIdx.x;
    const float* row = logits + (b * n_vocab);

    float l_max1 = -1e30f;
    float l_max2 = -1e30f;
    int32_t l_tok1 = 0;
    int32_t l_tok2 = -1;

    for (size_t i = threadIdx.x; i < n_vocab; i += blockDim.x) {
        float val = row[i];
        update_pair(l_max1, l_tok1, l_max2, l_tok2, val, static_cast<int32_t>(i));
    }

    // Warp-level reduction
    warp_reduce_top2(l_max1, l_tok1, l_max2, l_tok2);

    // Shared memory for 32 warps (1024 threads max)
    __shared__ float s_max1[32];
    __shared__ int32_t s_tok1[32];
    __shared__ float s_max2[32];
    __shared__ int32_t s_tok2[32];

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) {
        s_max1[warp_id] = l_max1;
        s_tok1[warp_id] = l_tok1;
        s_max2[warp_id] = l_max2;
        s_tok2[warp_id] = l_tok2;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        float b_max1 = (lane < num_warps) ? s_max1[lane] : -1e30f;
        int32_t b_tok1 = (lane < num_warps) ? s_tok1[lane] : 0;
        float b_max2 = (lane < num_warps) ? s_max2[lane] : -1e30f;
        int32_t b_tok2 = (lane < num_warps) ? s_tok2[lane] : -1;

        warp_reduce_top2(b_max1, b_tok1, b_max2, b_tok2);

        if (lane == 0) {
            float margin = b_max1 - b_max2;
            float entropy = (margin > 0.0f) ? log1pf(expf(-margin)) : 1.0f;

            out[b].top_token = b_tok1;
            out[b].second_token = b_tok2;
            out[b].max_logit = b_max1;
            out[b].second_logit = b_max2;
            out[b].entropy_estimate = entropy;
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

} // namespace kernels
} // namespace diffusion
#endif
