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

    const float inv_log_v = 1.0f / std::max(1.0f, std::log(static_cast<float>(n_vocab)));

    for (size_t b = 0; b < batch_size; ++b) {
        const float* row = logits + (b * n_vocab);

        float max1 = -std::numeric_limits<float>::infinity();
        float max2 = -std::numeric_limits<float>::infinity();
        int32_t tok1 = 0;
        int32_t tok2 = 0;

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

        double sum_exp = 0.0;
        double sum_z_exp = 0.0;
        for (size_t v = 0; v < n_vocab; ++v) {
            float d = row[v] - max1;
            if (d > -20.0f) {
                double e = std::exp(static_cast<double>(d));
                sum_exp += e;
                sum_z_exp += static_cast<double>(d) * e;
            }
        }

        float inv_z = static_cast<float>(1.0 / std::max(sum_exp, 1e-12));
        float p1 = std::min(1.0f, std::max(0.0f, inv_z));
        float p2 = std::min(1.0f - p1, std::max(0.0f, std::exp(max2 - max1) * inv_z));
        float raw_h = std::max(0.0f, static_cast<float>(std::log(std::max(sum_exp, 1.0)) - sum_z_exp / std::max(sum_exp, 1e-12)));
        float norm_h = std::min(1.0f, raw_h * inv_log_v);

        out_reductions[b].top_token        = tok1;
        out_reductions[b].second_token     = tok2;
        out_reductions[b].max_logit        = max1;
        out_reductions[b].second_logit     = max2;
        out_reductions[b].top_prob         = p1;
        out_reductions[b].second_prob      = p2;
        out_reductions[b].entropy_estimate = norm_h;
    }
}

} // namespace kernels
} // namespace diffusion
