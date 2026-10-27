#include <metal_stdlib>
using namespace metal;

struct LogitsReduction {
    int32_t top_token;
    int32_t second_token;
    float max_logit;
    float second_logit;
    float entropy_estimate;
};

inline void update_pair_metal(
    thread float& max1, thread int32_t& tok1,
    thread float& max2, thread int32_t& tok2,
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

kernel void reduce_logits_metal_kernel(
    device const float* logits          [[buffer(0)]],
    constant uint32_t& n_vocab          [[buffer(1)]],
    device LogitsReduction* out         [[buffer(2)]],
    uint3 threadgroup_pos               [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_tg              [[thread_position_in_threadgroup]],
    uint3 threads_per_tg                [[threads_per_threadgroup]],
    uint simd_lane_id                   [[thread_index_in_simdgroup]],
    uint simd_group_id                  [[simdgroup_index_in_threadgroup]]
) {
    uint b = threadgroup_pos.x;
    device const float* row = logits + (b * n_vocab);

    float l_max1 = -1e30f;
    float l_max2 = -1e30f;
    int32_t l_tok1 = 0;
    int32_t l_tok2 = -1;

    for (uint i = thread_pos_in_tg.x; i < n_vocab; i += threads_per_tg.x) {
        float val = row[i];
        update_pair_metal(l_max1, l_tok1, l_max2, l_tok2, val, static_cast<int32_t>(i));
    }

    // SIMD-group shuffle reduction
    for (ushort offset = 16; offset > 0; offset /= 2) {
        float o_max1 = simd_shuffle_down(l_max1, offset);
        int32_t o_tok1 = simd_shuffle_down(l_tok1, offset);
        float o_max2 = simd_shuffle_down(l_max2, offset);
        int32_t o_tok2 = simd_shuffle_down(l_tok2, offset);

        update_pair_metal(l_max1, l_tok1, l_max2, l_tok2, o_max1, o_tok1);
        update_pair_metal(l_max1, l_tok1, l_max2, l_tok2, o_max2, o_tok2);
    }

    threadgroup float s_max1[32];
    threadgroup int32_t s_tok1[32];
    threadgroup float s_max2[32];
    threadgroup int32_t s_tok2[32];

    if (simd_lane_id == 0) {
        s_max1[simd_group_id] = l_max1;
        s_tok1[simd_group_id] = l_tok1;
        s_max2[simd_group_id] = l_max2;
        s_tok2[simd_group_id] = l_tok2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        uint num_simd = threads_per_tg.x / 32;
        float b_max1 = (simd_lane_id < num_simd) ? s_max1[simd_lane_id] : -1e30f;
        int32_t b_tok1 = (simd_lane_id < num_simd) ? s_tok1[simd_lane_id] : 0;
        float b_max2 = (simd_lane_id < num_simd) ? s_max2[simd_lane_id] : -1e30f;
        int32_t b_tok2 = (simd_lane_id < num_simd) ? s_tok2[simd_lane_id] : -1;

        for (ushort offset = 16; offset > 0; offset /= 2) {
            float o_max1 = simd_shuffle_down(b_max1, offset);
            int32_t o_tok1 = simd_shuffle_down(b_tok1, offset);
            float o_max2 = simd_shuffle_down(b_max2, offset);
            int32_t o_tok2 = simd_shuffle_down(b_tok2, offset);

            update_pair_metal(b_max1, b_tok1, b_max2, b_tok2, o_max1, o_tok1);
            update_pair_metal(b_max1, b_tok1, b_max2, b_tok2, o_max2, o_tok2);
        }

        if (simd_lane_id == 0) {
            float margin = b_max1 - b_max2;
            float entropy = (margin > 0.0f) ? log(1.0f + exp(-margin)) : 1.0f;

            out[b].top_token = b_tok1;
            out[b].second_token = b_tok2;
            out[b].max_logit = b_max1;
            out[b].second_logit = b_max2;
            out[b].entropy_estimate = entropy;
        }
    }
}
