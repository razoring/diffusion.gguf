#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

#if defined(__AVX2__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>

static inline int32_t fast_argmax_simd(const float* data, int n) {
    if (n <= 0) return 0;
    int i = 0;
    __m256 max_val = _mm256_set1_ps(-1e30f);
    __m256i best_idx = _mm256_setzero_si256();
    __m256i step = _mm256_set1_epi32(8);
    __m256i cur_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 mask = _mm256_cmp_ps(v, max_val, _CMP_GT_OQ);
        max_val = _mm256_blendv_ps(max_val, v, mask);
        best_idx = _mm256_blendv_epi8(best_idx, cur_idx, _mm256_castps_si256(mask));
        cur_idx = _mm256_add_epi32(cur_idx, step);
    }

    alignas(32) float vals[8];
    alignas(32) int32_t idxs[8];
    _mm256_storeu_ps(vals, max_val);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(idxs), best_idx);

    float global_max = vals[0];
    int32_t global_best = idxs[0];
    for (int j = 1; j < 8; ++j) {
        if (vals[j] > global_max) {
            global_max = vals[j];
            global_best = idxs[j];
        }
    }

    for (; i < n; ++i) {
        if (data[i] > global_max) {
            global_max = data[i];
            global_best = i;
        }
    }
    return global_best;
}
#else
static inline int32_t fast_argmax_simd(const float* data, int n) {
    if (n <= 0) return 0;
    int32_t best_id = 0;
    float max_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            best_id = i;
        }
    }
    return best_id;
}
#endif

// Fast confidence computation for top-1 token using near-peak thresholding
static inline float fast_confidence(const float* data, int n, int32_t best_id) {
    if (!data || n <= 0 || best_id < 0 || best_id >= n) return 1.0f;
    float max_val = data[best_id];
    float sum_exp = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = data[i] - max_val;
        if (diff > -10.0f) {
            sum_exp += std::exp(diff);
        }
    }
    return sum_exp > 0.0f ? (1.0f / sum_exp) : 1.0f;
}
