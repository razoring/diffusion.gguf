#include "reduce_logits.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace diffusion {
namespace kernels {

bool has_gpu_reduction() {
#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA) || \
    defined(GGML_USE_HIP) || defined(DIFFUSION_USE_HIP) || \
    defined(GGML_USE_METAL) || defined(DIFFUSION_USE_METAL)
    return true;
#else
    return false;
#endif
}

void reduce_logits_cpu(
    const float* logits,
    size_t n_vocab,
    size_t batch_size,
    LogitsReduction* out_reductions
) {
    if (!logits || !out_reductions || n_vocab == 0) return;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* row = logits + (b * n_vocab);

        float max1 = -std::numeric_limits<float>::infinity();
        float max2 = -std::numeric_limits<float>::infinity();
        int32_t tok1 = 0;
        int32_t tok2 = -1;

        // 1. Single pass to find top-1 and top-2
        for (size_t v = 0; v < n_vocab; ++v) {
            float val = row[v];
            if (val > max1) {
                max2 = max1;
                tok2 = tok1;
                max1 = val;
                tok1 = static_cast<int32_t>(v);
            } else if (val > max2) {
                max2 = val;
                tok2 = static_cast<int32_t>(v);
            }
        }

        // 2. Shannon entropy estimation via log-sum-exp on top logits
        // For fast confidence evaluation, margin = max1 - max2 is primary indicator:
        // High margin implies low entropy (quasi-deterministic next token)
        float margin = max1 - max2;
        // Entropy approximation: H ~ log(1 + exp(-margin))
        float entropy = (margin > 0.0f) ? std::log1pf(std::exp(-margin)) : 1.0f;

        out_reductions[b].top_token = tok1;
        out_reductions[b].second_token = tok2;
        out_reductions[b].max_logit = max1;
        out_reductions[b].second_logit = max2;
        out_reductions[b].entropy_estimate = entropy;
    }
}

} // namespace kernels
} // namespace diffusion
