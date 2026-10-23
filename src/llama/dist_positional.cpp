#include "dist_positional.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <random>

namespace diffusion {

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static const char* dist_pos_name(const struct llama_sampler* /*smpl*/) {
    return "dist_positional";
}

static void dist_pos_apply(struct llama_sampler* smpl, llama_token_data_array* cur_p) {
    auto* ctx = reinterpret_cast<llama_sampler_dist_positional_ctx*>(smpl->ctx);
    if (!cur_p || cur_p->size == 0) return;

    cur_p->selected = 0;
    if (cur_p->size == 1) {
        cur_p->data[0].p = 1.0f;
        return;
    }

    // Numerical stability
    float max_l = cur_p->data[0].logit;
    if (!cur_p->sorted) {
        for (size_t i = 1; i < cur_p->size; ++i) {
            max_l = std::max(max_l, cur_p->data[i].logit);
        }
    }

    double sum_cum = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float p = std::exp(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        sum_cum += p;
    }

    // Seed RNG deterministically for (seed, abs_pos)
    uint64_t combined = static_cast<uint64_t>(ctx->seed) ^ (static_cast<uint64_t>(ctx->abs_pos) * 0x517cc1b727220a95ULL);
    uint64_t rng_seed = splitmix64(combined);
    std::mt19937_64 rng(rng_seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double rnd = dist(rng);

    double sum_run = 0.0;
    const double sum_tgt = sum_cum * rnd;
    bool found = false;

    for (size_t i = 0; i < cur_p->size; ++i) {
        if (!found) {
            sum_run += cur_p->data[i].p;
            if (sum_run >= sum_tgt) {
                cur_p->selected = static_cast<int32_t>(i);
                found = true;
            }
        }
        cur_p->data[i].p = static_cast<float>(cur_p->data[i].p / sum_cum);
    }

    if (!found) {
        cur_p->selected = static_cast<int32_t>(cur_p->size - 1);
    }
}

static struct llama_sampler* dist_pos_clone(const struct llama_sampler* smpl) {
    const auto* ctx = reinterpret_cast<const llama_sampler_dist_positional_ctx*>(smpl->ctx);
    auto* cloned = llama_sampler_init_dist_positional(ctx->seed);
    auto* cloned_ctx = reinterpret_cast<llama_sampler_dist_positional_ctx*>(cloned->ctx);
    cloned_ctx->abs_pos = ctx->abs_pos;
    return cloned;
}

static void dist_pos_free(struct llama_sampler* smpl) {
    delete reinterpret_cast<llama_sampler_dist_positional_ctx*>(smpl->ctx);
}

static struct llama_sampler_i dist_pos_i = {
    /* .name              = */ dist_pos_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ dist_pos_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ dist_pos_clone,
    /* .free              = */ dist_pos_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

llama_sampler* llama_sampler_init_dist_positional(uint32_t seed) {
    auto* ctx = new llama_sampler_dist_positional_ctx{seed, 0};
    return llama_sampler_init(&dist_pos_i, ctx);
}

void llama_sampler_dist_positional_set_pos(llama_sampler* smpl, int abs_pos) {
    if (!smpl || !smpl->ctx) return;
    auto* ctx = reinterpret_cast<llama_sampler_dist_positional_ctx*>(smpl->ctx);
    ctx->abs_pos = abs_pos;
}

} // namespace diffusion
