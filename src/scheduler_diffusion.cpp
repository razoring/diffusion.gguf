#include "scheduler_diffusion.h"
#include "init_guess.h"
#include "fast_math.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <unordered_map>

namespace diffusion {

namespace {

enum class SlotState : uint8_t {
    MASKED = 0,
    ACTIVE = 1,
    SOFT_LOCKED = 2,
    HARD_LOCKED = 3
};

struct CanvasSlot {
    int32_t tok1 = -1;
    int32_t tok2 = -1;
    int32_t kv_tok = -1;
    float p1 = 0.0f;
    float p2 = 0.0f;
    float entropy = 1.0f;
    float energy = 0.0f;
    SlotState state = SlotState::MASKED;
    bool in_kv = false;
    uint8_t passes = 0;
};

static inline void compute_annealed_weights(
    const CanvasSlot& slot,
    float alpha_mask,
    float& w1,
    float& w2,
    float& wm
) {
    if (slot.state == SlotState::SOFT_LOCKED || slot.state == SlotState::HARD_LOCKED) {
        w1 = 1.0f;
        w2 = 0.0f;
        wm = 0.0f;
        return;
    }
    if (slot.state == SlotState::MASKED && slot.tok1 < 0) {
        w1 = 0.0f;
        w2 = 0.0f;
        wm = 1.0f;
        return;
    }
    const float u_u = std::clamp(slot.entropy * (1.0f - slot.energy), 0.0f, 1.0f);
    wm = alpha_mask * u_u;
    const float w_cand = std::max(0.0f, 1.0f - wm);

    if (slot.tok2 < 0 || slot.tok2 == slot.tok1 || slot.p2 <= 1e-4f) {
        w1 = w_cand;
        w2 = 0.0f;
        return;
    }

    const float tau = std::max(0.15f, u_u);
    const float inv_tau = 1.0f / tau;
    const float p1_safe = std::max(1e-4f, slot.p1);
    const float p2_safe = std::max(1e-4f, slot.p2);
    const float tp1 = std::pow(p1_safe, inv_tau);
    const float tp2 = std::pow(p2_safe, inv_tau);
    const float sum_tp = tp1 + tp2 + 1e-6f;

    w1 = w_cand * (tp1 / sum_tp);
    w2 = std::max(0.0f, w_cand - w1);
}

struct ModelTransition {
    int32_t tok1 = -1;
    int32_t tok2 = -1;
    float p1 = 0.0f;
    float p2 = 0.0f;
    uint16_t count = 0;
    uint16_t misses = 0;
};

static inline uint64_t hash_pair(int32_t a, int32_t b) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(a));
    h *= 0x100000001b3ULL;
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(b));
    h *= 0x100000001b3ULL;
    return h;
}

static inline uint64_t hash_triple(int32_t a, int32_t b, int32_t c) {
    uint64_t h = hash_pair(a, b);
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(c));
    h *= 0x100000001b3ULL;
    return h;
}

struct SkipLogitsGuard {
    IModelBackend& backend;
    bool active = false;
    explicit SkipLogitsGuard(IModelBackend& b, bool enable) : backend(b), active(enable) {
        if (active) backend.set_skip_host_logits_copy(true);
    }
    ~SkipLogitsGuard() {
        if (active) backend.set_skip_host_logits_copy(false);
    }
};

} // namespace

SchedulerDiffusion::SchedulerDiffusion(
    IModelBackend& backend,
    const ITokenizer& tokenizer,
    ISamplerChain& sampler,
    int default_max_diffusion_steps,
    InitGuesser* guesser
) : backend_(backend),
    tokenizer_(tokenizer),
    sampler_(sampler),
    default_max_diffusion_steps_(default_max_diffusion_steps > 0 ? default_max_diffusion_steps : 3),
    guesser_(guesser) {
}

SchedulerDiffusion::~SchedulerDiffusion() {
    if (device_bufs_allocated_) {
        kernels::diff_device_free(device_bufs_);
        device_bufs_allocated_ = false;
    }
#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
    if (d_reductions_) {
        kernels::reduce_logits_cuda_free(d_reductions_, d_reductions_dev_);
        d_reductions_ = nullptr;
        d_reductions_cap_ = 0;
        d_reductions_dev_ = -1;
    }
#endif
}

GenResult SchedulerDiffusion::run(
    const std::vector<int32_t>& prompt,
    const GenParams& params,
    int reuse_prefix_len,
    const StepCallback& callback
) {
    GenResult result;
    if (prompt.empty()) return result;

    std::vector<int32_t> effective_prompt = prompt;
    if (backend_.n_ctx() > 0 && static_cast<int>(effective_prompt.size()) >= backend_.n_ctx()) {
        int max_prompt = std::max(1, backend_.n_ctx() - std::max(64, params.n_predict > 0 ? std::min(params.n_predict, 512) : 128));
        if (static_cast<int>(effective_prompt.size()) > max_prompt) {
            std::vector<int32_t> trimmed;
            trimmed.push_back(effective_prompt.front());
            int offset = static_cast<int>(effective_prompt.size()) - (max_prompt - 1);
            trimmed.insert(trimmed.end(), effective_prompt.begin() + offset, effective_prompt.end());
            effective_prompt = std::move(trimmed);
            reuse_prefix_len = 0;
        }
    }

    int prompt_len = static_cast<int>(effective_prompt.size());
    if (reuse_prefix_len < 0) reuse_prefix_len = 0;
    if (reuse_prefix_len > prompt_len) reuse_prefix_len = prompt_len;

    int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    // Dynamic whole canvas evaluation wave width (bounded by hardware-optimal batch headroom)
    int B_target = 16;
    if (!backend_.supports_kv_truncate()) {
        B_target = 1;
    }
    int max_iters = (params.max_iters > 16) ? params.max_iters : (n_predict * 2 + 32);

    backend_.set_sampler(params.seq_id, nullptr);

    const bool pure_greedy = sampler_.is_pure_greedy() && !params.logprobs;
    int n_vocab = backend_.n_vocab();

    auto reduce_rows = [&](const LogitsView& L, int n_rows) -> std::vector<kernels::LogitsReduction> {
        std::vector<kernels::LogitsReduction> out(std::max(0, n_rows), kernels::LogitsReduction{-1, -1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        if (n_rows <= 0 || L.data == nullptr) return out;
#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
        const float* d_logits = backend_.get_logits_device_ptr();
        int dev_id = backend_.get_logits_device_id();
        if (d_logits != nullptr && L.data == d_logits && dev_id >= 0) {
            if (d_reductions_ && d_reductions_dev_ != dev_id) {
                kernels::reduce_logits_cuda_free(d_reductions_, d_reductions_dev_);
                d_reductions_ = nullptr;
                d_reductions_cap_ = 0;
            }
            d_reductions_dev_ = dev_id;
            kernels::reduce_logits_cuda_to_host(d_logits, static_cast<size_t>(n_vocab), static_cast<size_t>(n_rows), out.data(), &d_reductions_, &d_reductions_cap_, dev_id);
            return out;
        }
#endif
        kernels::reduce_logits_cpu(L.data, static_cast<size_t>(n_vocab), static_cast<size_t>(n_rows), out.data());
        return out;
    };

    SkipLogitsGuard skip_guard(backend_, pure_greedy && kernels::has_gpu_reduction());

    // In-context LLM transition cache (populated from prefill + every diffusion wave)
    std::unordered_map<int32_t, ModelTransition> unigram_table;
    std::unordered_map<uint64_t, ModelTransition> bigram_table;
    std::unordered_map<uint64_t, ModelTransition> trigram_table;
    unigram_table.reserve(512);
    bigram_table.reserve(1024);
    trigram_table.reserve(1024);

    auto record_transition = [&](int32_t prev2, int32_t prev1, int32_t cur_tok, const kernels::LogitsReduction& red) {
        if (cur_tok < 0 || red.top_token < 0 || tokenizer_.is_eog(red.top_token)) return;
        auto update_entry = [&](ModelTransition& entry) {
            if (entry.tok1 == red.top_token) {
                entry.count = static_cast<uint16_t>(std::min<int>(65535, entry.count + 1));
                entry.p1 = std::max(entry.p1, red.top_prob);
                entry.tok2 = red.second_token;
                entry.p2 = red.second_prob;
            } else if (entry.tok1 < 0) {
                entry = ModelTransition{red.top_token, red.second_token, red.top_prob, red.second_prob, 1, 0};
            } else {
                uint16_t m = static_cast<uint16_t>(std::min<int>(65535, entry.misses + 1));
                entry = ModelTransition{red.top_token, red.second_token, red.top_prob, red.second_prob, 1, m};
            }
        };
        if (red.top_prob >= 0.25f || unigram_table.find(cur_tok) != unigram_table.end()) {
            update_entry(unigram_table[cur_tok]);
        }
        if (prev1 >= 0 && (red.top_prob >= 0.20f || bigram_table.find(hash_pair(prev1, cur_tok)) != bigram_table.end())) {
            update_entry(bigram_table[hash_pair(prev1, cur_tok)]);
        }
        if (prev2 >= 0 && prev1 >= 0 && (red.top_prob >= 0.15f || trigram_table.find(hash_triple(prev2, prev1, cur_tok)) != trigram_table.end())) {
            update_entry(trigram_table[hash_triple(prev2, prev1, cur_tok)]);
        }
    };

    // 1. Prefill prompt
    LogitsView prefill_logits;
    bool prefill_all = pure_greedy && (B_target > 1) && (reuse_prefix_len == 0) && (prompt_len <= 64);
    if (reuse_prefix_len == 0) {
        backend_.kv_clear(params.seq_id);
        prefill_logits = backend_.eval(effective_prompt.data(), prompt_len, 0, prefill_all, params.seq_id);
        for (int32_t tok : effective_prompt) {
            sampler_.accept(tok);
        }
    } else {
        backend_.kv_truncate(reuse_prefix_len, params.seq_id);
        int new_len = prompt_len - reuse_prefix_len;
        if (new_len > 0) {
            prefill_logits = backend_.eval(effective_prompt.data() + reuse_prefix_len, new_len, reuse_prefix_len, false, params.seq_id);
            for (int i = reuse_prefix_len; i < prompt_len; ++i) {
                sampler_.accept(effective_prompt[i]);
            }
        } else {
            backend_.kv_truncate(prompt_len - 1, params.seq_id);
            prefill_logits = backend_.eval(effective_prompt.data() + prompt_len - 1, 1, prompt_len - 1, false, params.seq_id);
        }
    }

    int P_0 = prompt_len;
    StopReason stop_reason = StopReason::NONE;
    std::vector<int32_t> running_history = effective_prompt;
    if (guesser_) {
        guesser_->update_ngram(effective_prompt);
    }

    kernels::LogitsReduction first_red{-1, -1, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    if (pure_greedy) {
        auto preds = reduce_rows(prefill_logits, prefill_logits.n_rows);
        if (prefill_all && static_cast<int>(preds.size()) == prompt_len) {
            for (int i = 0; i < prompt_len - 1; ++i) {
                int32_t p2 = (i >= 2) ? effective_prompt[i - 2] : -1;
                int32_t p1 = (i >= 1) ? effective_prompt[i - 1] : -1;
                record_transition(p2, p1, effective_prompt[i], preds[i]);
            }
        }
        if (!preds.empty()) {
            first_red = preds.back();
        }
    } else {
        const float* prefill_row = prefill_logits.row(prefill_logits.n_rows - 1);
        if (prefill_row) {
            float conf = 1.0f;
            first_red.top_token = sampler_.sample(prefill_row, P_0, &conf);
            first_red.top_prob = conf;
        }
    }

    auto t_start = std::chrono::steady_clock::now();

    if (B_target == 1) {
        // Fast single-token decode path (GPU-reduced when greedy)
        int P = P_0;
        int32_t current_token = first_red.top_token;

        while (static_cast<int>(result.tokens.size()) < n_predict && result.iterations < max_iters && current_token >= 0) {
            result.iterations++;
            result.total_diffusion_passes++;
            result.cumulative_active_work += 1;

            if (P + 1 >= backend_.n_ctx()) {
                if (params.ctx_shift && backend_.supports_kv_shift()) {
                    int n_keep = params.n_keep > 0 ? params.n_keep : 1;
                    int n_discard = params.n_discard > 0 ? params.n_discard : std::max(1, (backend_.n_ctx() - n_keep) / 2);
                    if (n_discard > 0 && P - n_discard >= n_keep) {
                        backend_.kv_truncate(n_keep + n_discard, params.seq_id);
                        backend_.kv_shift(params.seq_id, n_keep + n_discard, P, -n_discard);
                        P -= n_discard;
                    } else {
                        stop_reason = StopReason::MAX_TOKENS;
                        break;
                    }
                } else {
                    stop_reason = StopReason::MAX_TOKENS;
                    break;
                }
            }

            LogitsView L = backend_.eval(&current_token, 1, P, params.logprobs, params.seq_id);
            int32_t next_tok = -1;
            if (pure_greedy) {
                auto reds = reduce_rows(L, 1);
                if (!reds.empty()) next_tok = reds[0].top_token;
            } else {
                const float* r0 = L.row(0);
                float next_conf = 1.0f;
                if (r0) next_tok = sampler_.sample(r0, P + 1, &next_conf);
            }

            sampler_.accept(current_token);
            result.tokens.push_back(current_token);
            result.total_blocks++;

            if (tokenizer_.is_eog(current_token)) {
                stop_reason = StopReason::EOG;
                if (callback) {
                    StepEvent ev;
                    ev.step = result.iterations;
                    ev.committed_upto = static_cast<int>(result.tokens.size());
                    ev.done = true;
                    ev.reason = stop_reason;
                    callback(ev);
                }
                break;
            }

            result.text += tokenizer_.piece(current_token);

            bool should_stop = false;
            for (const auto& s : params.stop_strings) {
                if (!s.empty() && result.text.size() >= s.size() &&
                    result.text.compare(result.text.size() - s.size(), s.size(), s) == 0) {
                    result.text.resize(result.text.size() - s.size());
                    stop_reason = StopReason::STOP_STRING;
                    should_stop = true;
                    break;
                }
            }

            if (!should_stop && static_cast<int>(result.tokens.size()) >= n_predict) {
                stop_reason = StopReason::MAX_TOKENS;
                should_stop = true;
            }

            if (callback) {
                StepEvent ev;
                ev.step = result.iterations;
                ev.committed_upto = static_cast<int>(result.tokens.size());
                ev.committed_new = { current_token };
                ev.canvas = { next_tok };
                ev.done = should_stop;
                ev.reason = stop_reason;
                if (!callback(ev)) {
                    stop_reason = StopReason::CANCELLED;
                    break;
                }
            }

            if (should_stop) break;

            P += 1;
            current_token = next_tok;
        }
    } else {
        // =========================================================================
        // Pure Field-Driven Wave Diffusion Engine
        // (Annealed S3 + Bidirectional Free-Energy Ripple + Island Crystallization)
        // =========================================================================
        constexpr float THETA_LOCK = 1.0f;
        constexpr float THETA_WAKE = 0.85f;
        constexpr float LAMBDA_FWD = 0.35f;
        constexpr float LAMBDA_BACK = 0.45f;
        constexpr float TAU_H = 0.72f;
        constexpr float TAU_P = 0.20f;
        constexpr float ETA_REPULSIVE = 0.50f;
        constexpr int R_BACK = 2;
        constexpr float ALPHA_MASK = 0.25f;

        std::vector<CanvasSlot> canvas(n_predict);
        canvas[0].tok1 = first_red.top_token;
        canvas[0].tok2 = first_red.second_token;
        canvas[0].p1 = first_red.top_prob;
        canvas[0].p2 = first_red.second_prob;
        canvas[0].entropy = first_red.entropy_estimate;
        canvas[0].energy = 1.0f;
        canvas[0].state = SlotState::SOFT_LOCKED;
        canvas[0].in_kv = false;
        canvas[0].passes = 1;

        const int32_t mask_id = tokenizer_.pad_or_fallback();
        const bool is_hybrid_recr = backend_.is_recurrent_or_hybrid();

        // Wave width: W=8 on pure attention, W=6 on recurrent/hybrid models (n_rs_seq = 7)
        const int max_W = is_hybrid_recr ? 6 : 8;

        int q_hard = 0;       // First slot not yet finalized to HARD_LOCKED
        int q_active_min = 0; // Earliest slot not yet SOFT_LOCKED or HARD_LOCKED
        bool should_stop = false;

        std::vector<int32_t> wave_canvas_pos(max_W);
        std::vector<int32_t> wave_tok1(max_W);
        std::vector<int32_t> wave_tok2(max_W);
        std::vector<int32_t> wave_pos(max_W);
        std::vector<float> wave_w1(max_W);
        std::vector<float> wave_w2(max_W);
        std::vector<float> wave_wm(max_W);

        auto get_tok_at = [&](int canvas_idx) -> int32_t {
            if (canvas_idx >= 0 && canvas_idx < n_predict) {
                return canvas[canvas_idx].tok1;
            }
            int prompt_idx = static_cast<int>(effective_prompt.size()) + canvas_idx;
            if (prompt_idx >= 0 && prompt_idx < static_cast<int>(effective_prompt.size())) {
                return effective_prompt[prompt_idx];
            }
            return -1;
        };

        while (q_hard < n_predict && result.iterations < max_iters && !should_stop) {
            // Find earliest unconverged slot
            q_active_min = q_hard;
            while (q_active_min < n_predict && (canvas[q_active_min].state == SlotState::SOFT_LOCKED || canvas[q_active_min].state == SlotState::HARD_LOCKED)) {
                q_active_min++;
            }

            if (q_active_min >= n_predict) {
                // All slots up to n_predict reached locked state
                while (q_hard < n_predict) {
                    canvas[q_hard].state = SlotState::HARD_LOCKED;
                    int32_t tok = canvas[q_hard].tok1;
                    if (tok >= 0) {
                        sampler_.accept(tok);
                        result.tokens.push_back(tok);
                        result.text += tokenizer_.piece(tok);
                    }
                    q_hard++;
                }
                break;
            }

            // Termination guarantee: if slot has undergone too many passes without locking, force lock to top candidate
            int check_slot = is_hybrid_recr ? q_hard : q_active_min;
            if (canvas[check_slot].passes >= default_max_diffusion_steps_ && canvas[check_slot].tok1 >= 0) {
                canvas[check_slot].energy = 1.0f;
                canvas[check_slot].state = SlotState::SOFT_LOCKED;
                if (!is_hybrid_recr) continue;
            }

            // Context window shift if needed
            if (P_0 + q_active_min + max_W >= backend_.n_ctx()) {
                if (params.ctx_shift && backend_.supports_kv_shift()) {
                    int n_keep = params.n_keep > 0 ? params.n_keep : 1;
                    int n_discard = params.n_discard > 0 ? params.n_discard : std::max(1, (backend_.n_ctx() - n_keep) / 2);
                    int cur_total = P_0 + q_active_min;
                    if (n_discard > 0 && cur_total - n_discard >= n_keep) {
                        backend_.kv_truncate(n_keep + n_discard, params.seq_id);
                        backend_.kv_shift(params.seq_id, n_keep + n_discard, cur_total, -n_discard);
                        P_0 -= n_discard;
                    } else {
                        stop_reason = StopReason::MAX_TOKENS;
                        break;
                    }
                } else {
                    stop_reason = StopReason::MAX_TOKENS;
                    break;
                }
            }

            // 1. Assemble active diffusion wave slots
            int W = 0;
            bool any_superpos = false;

            if (is_hybrid_recr) {
                // Recurrent / hybrid models: sequential evaluation starting at q_hard
                int start_pos = q_hard;
                int target_W = std::min(max_W, n_predict - start_pos);
                for (int r = 0; r < target_W; ++r) {
                    int pos = start_pos + r;
                    wave_canvas_pos[r] = pos;
                    int32_t cand1 = -1;
                    int32_t cand2 = -1;

                    if (r == 0 || canvas[pos].state == SlotState::SOFT_LOCKED || canvas[pos].state == SlotState::HARD_LOCKED) {
                        cand1 = canvas[pos].tok1;
                        cand2 = canvas[pos].tok2;
                    } else {
                        // Chained N-gram / Guesser continuation from previous slot in wave
                        int32_t prev1 = wave_tok1[r - 1];
                        int32_t prev2 = (r >= 2) ? wave_tok1[r - 2] : get_tok_at(pos - 2);
                        int32_t prev3 = (r >= 3) ? wave_tok1[r - 3] : get_tok_at(pos - 3);

                        if (prev3 >= 0 && prev2 >= 0 && prev1 >= 0) {
                            auto it3 = trigram_table.find(hash_triple(prev3, prev2, prev1));
                            if (it3 != trigram_table.end() && (it3->second.p1 >= 0.20f || it3->second.count >= 1)) {
                                cand1 = it3->second.tok1;
                                cand2 = it3->second.tok2;
                            }
                        }
                        if (cand1 < 0 && prev2 >= 0 && prev1 >= 0) {
                            auto it2 = bigram_table.find(hash_pair(prev2, prev1));
                            if (it2 != bigram_table.end() && (it2->second.count >= 1 || it2->second.p1 >= 0.20f)) {
                                cand1 = it2->second.tok1;
                                cand2 = it2->second.tok2;
                            }
                        }
                        if (cand1 < 0 && guesser_) {
                            std::vector<int32_t> local_ctx;
                            for (int k = std::min(15, pos); k >= 1; --k) {
                                int wave_idx = (pos - k) - start_pos;
                                if (wave_idx >= 0 && wave_idx < r) {
                                    local_ctx.push_back(wave_tok1[wave_idx]);
                                } else {
                                    local_ctx.push_back(get_tok_at(pos - k));
                                }
                            }
                            auto trie_g = guesser_->make_guesses(1, local_ctx);
                            if (!trie_g.empty()) cand1 = trie_g[0];
                        }
                        if (cand1 < 0 && prev1 >= 0) {
                            auto it1 = unigram_table.find(prev1);
                            if (it1 != unigram_table.end()) {
                                cand1 = it1->second.tok1;
                                cand2 = it1->second.tok2;
                            }
                        }
                        // Fall back to neural posterior from previous wave if available
                        if (cand1 < 0 && canvas[pos].tok1 >= 0) {
                            cand1 = canvas[pos].tok1;
                            cand2 = canvas[pos].tok2;
                        }

                        if (cand1 >= 0 && !tokenizer_.is_eog(cand1)) {
                            canvas[pos].tok1 = cand1;
                            canvas[pos].tok2 = cand2;
                            canvas[pos].p1 = 0.50f;
                            canvas[pos].p2 = (cand2 >= 0 && cand2 != cand1) ? 0.30f : 0.0f;
                            canvas[pos].entropy = 0.80f;
                            canvas[pos].energy = 0.0f;
                            canvas[pos].state = SlotState::ACTIVE;
                        } else {
                            cand1 = mask_id;
                            cand2 = -1;
                            canvas[pos].tok1 = mask_id;
                            canvas[pos].tok2 = -1;
                            canvas[pos].p1 = 0.0f;
                            canvas[pos].p2 = 0.0f;
                            canvas[pos].entropy = 1.0f;
                            canvas[pos].energy = 0.0f;
                            canvas[pos].state = SlotState::ACTIVE;
                        }
                    }

                    wave_tok1[r] = cand1;
                    wave_tok2[r] = (cand2 >= 0 && cand2 != cand1) ? cand2 : cand1;
                    wave_pos[r] = P_0 + pos;

                    float w1 = 1.0f, w2 = 0.0f, wm = 0.0f;
                    if (r <= 1) {
                        w1 = 1.0f;
                        w2 = 0.0f;
                        wm = 0.0f;
                    } else {
                        compute_annealed_weights(canvas[pos], ALPHA_MASK, w1, w2, wm);
                    }
                    wave_w1[r] = w1;
                    wave_w2[r] = w2;
                    wave_wm[r] = wm;
                    if (w2 > 0.0f || wm > 0.0f) any_superpos = true;
                    W++;
                }
            } else {
                // Pure Attention models: non-sequential sparse leapfrogging of in_kv SOFT_LOCKED islands
                int curr = (q_active_min > 0 && !canvas[q_active_min - 1].in_kv) ? (q_active_min - 1) : q_active_min;
                while (W < max_W && curr < n_predict) {
                    if (canvas[curr].state == SlotState::SOFT_LOCKED && canvas[curr].in_kv) {
                        // If island is in KV and the next token is also in KV and SOFT_LOCKED, leapfrog!
                        if (curr + 1 < n_predict && canvas[curr + 1].state == SlotState::SOFT_LOCKED && canvas[curr + 1].in_kv) {
                            curr++;
                            continue;
                        }
                    }

                    int r = W;
                    int pos = curr;
                    wave_canvas_pos[r] = pos;
                    int32_t cand1 = -1;
                    int32_t cand2 = -1;

                    if (r == 0 || canvas[pos].state == SlotState::SOFT_LOCKED || canvas[pos].state == SlotState::HARD_LOCKED) {
                        cand1 = canvas[pos].tok1;
                        cand2 = canvas[pos].tok2;
                    } else {
                        // Chained N-gram / Guesser continuation from previous slot in wave
                        int32_t prev1 = wave_tok1[r - 1];
                        int32_t prev2 = (r >= 2) ? wave_tok1[r - 2] : get_tok_at(pos - 2);
                        int32_t prev3 = (r >= 3) ? wave_tok1[r - 3] : get_tok_at(pos - 3);

                        if (prev3 >= 0 && prev2 >= 0 && prev1 >= 0) {
                            auto it3 = trigram_table.find(hash_triple(prev3, prev2, prev1));
                            if (it3 != trigram_table.end() && (it3->second.p1 >= 0.20f || it3->second.count >= 1)) {
                                cand1 = it3->second.tok1;
                                cand2 = it3->second.tok2;
                            }
                        }
                        if (cand1 < 0 && prev2 >= 0 && prev1 >= 0) {
                            auto it2 = bigram_table.find(hash_pair(prev2, prev1));
                            if (it2 != bigram_table.end() && (it2->second.count >= 1 || it2->second.p1 >= 0.20f)) {
                                cand1 = it2->second.tok1;
                                cand2 = it2->second.tok2;
                            }
                        }
                        if (cand1 < 0 && guesser_) {
                            std::vector<int32_t> local_ctx;
                            for (int k = std::min(15, pos); k >= 1; --k) {
                                int wave_idx = (pos - k) - wave_canvas_pos[0];
                                if (wave_idx >= 0 && wave_idx < r) {
                                    local_ctx.push_back(wave_tok1[wave_idx]);
                                } else {
                                    local_ctx.push_back(get_tok_at(pos - k));
                                }
                            }
                            auto trie_g = guesser_->make_guesses(1, local_ctx);
                            if (!trie_g.empty()) cand1 = trie_g[0];
                        }
                        if (cand1 < 0 && prev1 >= 0) {
                            auto it1 = unigram_table.find(prev1);
                            if (it1 != unigram_table.end()) {
                                cand1 = it1->second.tok1;
                                cand2 = it1->second.tok2;
                            }
                        }
                        // Fall back to neural posterior from previous wave if available
                        if (cand1 < 0 && canvas[pos].tok1 >= 0) {
                            cand1 = canvas[pos].tok1;
                            cand2 = canvas[pos].tok2;
                        }

                        if (cand1 >= 0 && !tokenizer_.is_eog(cand1)) {
                            canvas[pos].tok1 = cand1;
                            canvas[pos].tok2 = cand2;
                            canvas[pos].p1 = 0.50f;
                            canvas[pos].p2 = (cand2 >= 0 && cand2 != cand1) ? 0.30f : 0.0f;
                            canvas[pos].entropy = 0.80f;
                            canvas[pos].energy = 0.0f;
                            canvas[pos].state = SlotState::ACTIVE;
                        } else {
                            cand1 = mask_id;
                            cand2 = -1;
                            canvas[pos].tok1 = mask_id;
                            canvas[pos].tok2 = -1;
                            canvas[pos].p1 = 0.0f;
                            canvas[pos].p2 = 0.0f;
                            canvas[pos].entropy = 1.0f;
                            canvas[pos].energy = 0.0f;
                            canvas[pos].state = SlotState::ACTIVE;
                        }
                    }

                    wave_tok1[r] = cand1;
                    wave_tok2[r] = (cand2 >= 0 && cand2 != cand1) ? cand2 : cand1;
                    wave_pos[r] = P_0 + pos;

                    float w1 = 1.0f, w2 = 0.0f, wm = 0.0f;
                    if (r <= 1) {
                        w1 = 1.0f;
                        w2 = 0.0f;
                        wm = 0.0f;
                    } else {
                        compute_annealed_weights(canvas[pos], ALPHA_MASK, w1, w2, wm);
                    }
                    wave_w1[r] = w1;
                    wave_w2[r] = w2;
                    wave_wm[r] = wm;
                    if (w2 > 0.0f || wm > 0.0f) any_superpos = true;
                    W++;
                    curr++;
                }
            }

            if (W <= 0) break;

            result.iterations++;
            result.total_diffusion_passes++;
            result.cumulative_active_work += W;

            // 2. Evaluate active wave on GPU
            // Use eval_superpos to maintain invariant batch.embd presence and 100% graph reuse
            LogitsView L = backend_.eval_superpos(
                wave_tok1.data(),
                wave_tok2.data(),
                mask_id,
                wave_w1.data(),
                wave_w2.data(),
                wave_wm.data(),
                W,
                wave_pos.data(),
                true,
                params.seq_id
            );

            // 3. GPU Logits Reduction / Sampling
            std::vector<kernels::LogitsReduction> reds = reduce_rows(L, W);
            if (!pure_greedy) {
                for (int r = 0; r < W && r < static_cast<int>(reds.size()); ++r) {
                    const float* row = L.row(r);
                    if (!row) row = backend_.get_logits_ith(r);
                    if (row) {
                        float conf = 1.0f;
                        reds[r].top_token = sampler_.sample(row, wave_pos[r] + 1, &conf);
                        reds[r].top_prob = conf;
                    }
                }
            }

            // 4. In-Place Field Update & Bidirectional Free-Energy Ripple
            std::vector<TokenDiff> step_diffs;

            for (int k = 0; k < W && k < static_cast<int>(reds.size()); ++k) {
                int eval_pos = wave_canvas_pos[k];
                canvas[eval_pos].in_kv = true;
                canvas[eval_pos].kv_tok = wave_tok1[k];
                canvas[eval_pos].passes++;

                int target_pos = eval_pos + 1;
                if (target_pos >= n_predict) continue;

                const auto& red = reds[k];
                if (red.top_token < 0) continue;

                bool matched_wave_cand = (k + 1 < W && wave_canvas_pos[k + 1] == target_pos && wave_tok1[k + 1] == red.top_token);

                if (matched_wave_cand) {
                    // Neural prediction verified the candidate token fed at target_pos!
                    // This satisfies exact mathematical fixed-point equivalence: f(x) = x.
                    canvas[target_pos].tok1 = red.top_token;
                    canvas[target_pos].tok2 = red.second_token;
                    canvas[target_pos].p1 = red.top_prob;
                    canvas[target_pos].p2 = red.second_prob;
                    canvas[target_pos].entropy = red.entropy_estimate;
                    canvas[target_pos].energy = 1.0f;
                    canvas[target_pos].state = SlotState::SOFT_LOCKED;
                    canvas[target_pos].passes = 1;
                } else if (canvas[target_pos].state == SlotState::MASKED || canvas[target_pos].passes == 0) {
                    // First neural evaluation of target_pos
                    canvas[target_pos].tok1 = red.top_token;
                    canvas[target_pos].tok2 = red.second_token;
                    canvas[target_pos].p1 = red.top_prob;
                    canvas[target_pos].p2 = red.second_prob;
                    canvas[target_pos].entropy = red.entropy_estimate;
                    canvas[target_pos].energy = (k == 0) ? 1.0f : (red.top_prob * (1.0f - red.entropy_estimate));
                    canvas[target_pos].state = (canvas[target_pos].energy >= THETA_LOCK) ? SlotState::SOFT_LOCKED : SlotState::ACTIVE;
                    canvas[target_pos].passes = 1;
                } else {
                    // Multi-pass refinement of target_pos
                    if (red.top_token == canvas[target_pos].tok1) {
                        // Neural attractor agreement!
                        canvas[target_pos].p1 = 0.65f * canvas[target_pos].p1 + 0.35f * red.top_prob;
                        canvas[target_pos].p2 = 0.65f * canvas[target_pos].p2 + 0.35f * red.second_prob;
                        canvas[target_pos].entropy = 0.65f * canvas[target_pos].entropy + 0.35f * red.entropy_estimate;
                        float delta_agree = 0.35f * (1.0f - red.entropy_estimate);
                        canvas[target_pos].energy = std::min(1.0f, 0.65f * canvas[target_pos].energy + 0.35f * red.top_prob + delta_agree);
                        if (canvas[target_pos].energy >= THETA_LOCK) {
                            canvas[target_pos].state = SlotState::SOFT_LOCKED;
                        }
                    } else {
                        // Mode transition!
                        float delta_f_mode = canvas[target_pos].energy + (1.0f - red.top_prob) * red.entropy_estimate;
                        if (canvas[target_pos].passes >= 2) {
                            TokenDiff diff;
                            diff.pos = target_pos;
                            diff.old_id = canvas[target_pos].tok1;
                            diff.new_id = red.top_token;
                            diff.old_text = tokenizer_.piece(canvas[target_pos].tok1);
                            diff.new_text = tokenizer_.piece(red.top_token);
                            diff.confidence = red.top_prob;
                            step_diffs.push_back(std::move(diff));
                        }

                        canvas[target_pos].tok1 = red.top_token;
                        canvas[target_pos].tok2 = red.second_token;
                        canvas[target_pos].p1 = red.top_prob;
                        canvas[target_pos].p2 = red.second_prob;
                        canvas[target_pos].entropy = red.entropy_estimate;
                        canvas[target_pos].energy = 0.35f * red.top_prob * (1.0f - red.entropy_estimate);
                        canvas[target_pos].state = SlotState::ACTIVE;
                        if (canvas[target_pos].in_kv) {
                            backend_.kv_rm_range(P_0 + target_pos, P_0 + target_pos + 1);
                            canvas[target_pos].in_kv = false;
                        }

                        // Forward Ripple
                        int ripple_end = std::min(n_predict, target_pos + 16);
                        for (int d = target_pos + 1; d < ripple_end; ++d) {
                            if (canvas[d].state != SlotState::MASKED) {
                                float damp = std::exp(-LAMBDA_FWD * delta_f_mode / static_cast<float>(d - target_pos));
                                canvas[d].energy *= damp;
                                if (canvas[d].state == SlotState::SOFT_LOCKED && canvas[d].energy < THETA_WAKE) {
                                    canvas[d].state = SlotState::ACTIVE;
                                    if (canvas[d].in_kv) {
                                        backend_.kv_rm_range(P_0 + d, P_0 + d + 1);
                                        canvas[d].in_kv = false;
                                    }
                                }
                            }
                        }
                    }
                }

                // Continuation Free-Energy Check & Backward Ripple
                if (red.entropy_estimate > TAU_H && red.top_prob < TAU_P) {
                    float delta_f_back = (red.entropy_estimate - TAU_H) + (TAU_P - red.top_prob);
                    for (int up = target_pos - 1; up >= std::max(q_hard, target_pos - R_BACK); --up) {
                        float damp = std::exp(-LAMBDA_BACK * delta_f_back / static_cast<float>(target_pos - up));
                        canvas[up].energy *= damp;
                        if (canvas[up].state == SlotState::SOFT_LOCKED && canvas[up].energy < THETA_WAKE) {
                            canvas[up].state = SlotState::ACTIVE;
                            if (canvas[up].in_kv) {
                                backend_.kv_rm_range(P_0 + up, P_0 + up + 1);
                                canvas[up].in_kv = false;
                            }
                        }
                    }
                    if (target_pos - 1 >= q_hard) {
                        canvas[target_pos - 1].p1 *= std::exp(-ETA_REPULSIVE * delta_f_back);
                    }
                }
            }

            // 5. Finalize Verified HARD_LOCKED Prefix
            std::vector<int32_t> newly_locked_tokens;

            // Recompute q_active_min after ripples
            q_active_min = q_hard;
            while (q_active_min < n_predict && (canvas[q_active_min].state == SlotState::SOFT_LOCKED || canvas[q_active_min].state == SlotState::HARD_LOCKED)) {
                q_active_min++;
            }

            while (q_hard < n_predict) {
                if (canvas[q_hard].state != SlotState::SOFT_LOCKED || !canvas[q_hard].in_kv) {
                    break;
                }
                // Holdback check: for non-greedy sampling, slots within R_BACK wait if downstream entropy is high
                if (!pure_greedy && q_hard >= q_active_min - R_BACK) {
                    if (q_active_min < n_predict && canvas[q_active_min].entropy > TAU_H) {
                        break;
                    }
                }

                int32_t tok = canvas[q_hard].tok1;
                if (tok < 0) break;

                canvas[q_hard].state = SlotState::HARD_LOCKED;
                sampler_.accept(tok);
                result.tokens.push_back(tok);
                newly_locked_tokens.push_back(tok);

                int32_t prev1 = (q_hard >= 1) ? canvas[q_hard - 1].tok1 : get_tok_at(q_hard - 1);
                int32_t prev2 = (q_hard >= 2) ? canvas[q_hard - 2].tok1 : get_tok_at(q_hard - 2);
                int32_t next_tok = (q_hard + 1 < n_predict) ? canvas[q_hard + 1].tok1 : -1;
                if (next_tok >= 0) {
                    kernels::LogitsReduction red_info{next_tok, canvas[q_hard + 1].tok2, canvas[q_hard + 1].p1, canvas[q_hard + 1].p2, canvas[q_hard + 1].entropy, 0.0f, 0.0f};
                    record_transition(prev2, prev1, tok, red_info);
                }

                if (tokenizer_.is_eog(tok)) {
                    stop_reason = StopReason::EOG;
                    should_stop = true;
                    break;
                }

                result.text += tokenizer_.piece(tok);

                for (const auto& s : params.stop_strings) {
                    if (!s.empty() && result.text.size() >= s.size() &&
                        result.text.compare(result.text.size() - s.size(), s.size(), s) == 0) {
                        result.text.resize(result.text.size() - s.size());
                        stop_reason = StopReason::STOP_STRING;
                        should_stop = true;
                        break;
                    }
                }
                if (should_stop) break;

                if (static_cast<int>(result.tokens.size()) >= n_predict) {
                    stop_reason = StopReason::MAX_TOKENS;
                    should_stop = true;
                    break;
                }

                q_hard++;
            }

            // Recurrent state rollback for hybrid models
            if (is_hybrid_recr) {
                if (q_hard < n_predict) {
                    backend_.kv_truncate(P_0 + q_hard, params.seq_id);
                    for (int u = q_hard; u < n_predict; ++u) {
                        canvas[u].in_kv = false;
                    }
                }
            } else {
                // Pure attention: remove any uncommitted active KV cells
                for (int k = 0; k < W; ++k) {
                    int p = wave_canvas_pos[k];
                    if (canvas[p].state == SlotState::ACTIVE && canvas[p].in_kv) {
                        backend_.kv_rm_range(P_0 + p, P_0 + p + 1, params.seq_id);
                        canvas[p].in_kv = false;
                    }
                }
            }

            if (guesser_ && !newly_locked_tokens.empty()) {
                guesser_->update_ngram(newly_locked_tokens);
            }

            result.total_blocks++;

            if (callback && (!newly_locked_tokens.empty() || !step_diffs.empty() || should_stop)) {
                StepEvent ev;
                ev.step = result.iterations;
                ev.committed_upto = static_cast<int>(result.tokens.size());
                ev.committed_new = newly_locked_tokens;
                ev.diffs = std::move(step_diffs);
                ev.canvas.assign(wave_tok1.begin(), wave_tok1.begin() + W);
                ev.done = should_stop;
                ev.reason = stop_reason;
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double, std::milli>(now - t_start).count();
                ev.tok_per_s = elapsed > 0.0 ? (result.tokens.size() * 1000.0 / elapsed) : 0.0;
                ev.tok_per_iter = result.iterations > 0 ? (static_cast<double>(result.tokens.size()) / result.iterations) : 0.0;
                if (!callback(ev)) {
                    stop_reason = StopReason::CANCELLED;
                    should_stop = true;
                }
            }
        }
    }

    if (result.iterations >= max_iters && stop_reason == StopReason::NONE) {
        stop_reason = StopReason::MAX_TOKENS;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.tok_per_iter = result.iterations > 0 ? (static_cast<double>(result.tokens.size()) / result.iterations) : 0.0;
    result.tok_per_s = result.total_time_ms > 0.0 ? (result.tokens.size() * 1000.0 / result.total_time_ms) : 0.0;
    result.stop_reason = stop_reason;

    if (result.total_diffusion_passes > 0) {
        result.avg_active_tokens_per_pass = static_cast<double>(result.cumulative_active_work) / result.total_diffusion_passes;
    }
    if (result.total_blocks > 0) {
        result.avg_passes_per_block = static_cast<double>(result.total_diffusion_passes) / result.total_blocks;
    }

    return result;
}

} // namespace diffusion
