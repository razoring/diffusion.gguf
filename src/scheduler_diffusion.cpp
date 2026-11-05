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
    SOFT_LOCKED = 2
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
        // Whole-Canvas Diffusion with Hysteresis Soft-Locking & S3 Superposition
        // (W1 + S1 + S3 + R1)
        // =========================================================================
        constexpr float THETA_LOCK = 0.95f;
        constexpr float THETA_WAKE = 0.85f;
        constexpr float RIPPLE_LAMBDA = 0.35f;

        std::vector<CanvasSlot> canvas(n_predict);
        canvas[0].tok1 = first_red.top_token;
        canvas[0].tok2 = first_red.second_token;
        canvas[0].p1 = first_red.top_prob;
        canvas[0].p2 = first_red.second_prob;
        canvas[0].entropy = first_red.entropy_estimate;
        canvas[0].energy = first_red.top_prob;
        canvas[0].state = SlotState::ACTIVE;
        canvas[0].in_kv = false;

        const int32_t mask_id = tokenizer_.pad_or_fallback();
        const bool is_hybrid_recr = backend_.is_recurrent_or_hybrid();

        // Maximum wave width scales with B_target; for recurrent/hybrid models cap at 3 (bounded by n_rs_seq = 2)
        const int max_W = is_hybrid_recr ? 4 : ((B_target <= 4) ? 2 : ((B_target <= 8) ? 3 : 4));

        int q0 = 0;           // Earliest non-soft-locked / non-KV canvas index
        bool should_stop = false;

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

        while (q0 < n_predict && result.iterations < max_iters && !should_stop) {
            if (canvas[q0].tok1 < 0) break;

            // Context window shift if needed
            int rem_canvas = n_predict - q0;
            int W_cap = std::min(max_W, rem_canvas);
            if (P_0 + q0 + W_cap >= backend_.n_ctx()) {
                if (params.ctx_shift && backend_.supports_kv_shift()) {
                    int n_keep = params.n_keep > 0 ? params.n_keep : 1;
                    int n_discard = params.n_discard > 0 ? params.n_discard : std::max(1, (backend_.n_ctx() - n_keep) / 2);
                    int cur_total = P_0 + q0;
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

            // 1. Assemble active diffusion wave [q0 .. q0 + W - 1]
            // Slot r=0 (at q0) has the exact model prediction from q0-1; keep discrete (w1=1.0)
            wave_tok1[0] = canvas[q0].tok1;
            wave_tok2[0] = canvas[q0].tok2 >= 0 ? canvas[q0].tok2 : canvas[q0].tok1;
            wave_pos[0] = P_0 + q0;
            wave_w1[0] = 1.0f;
            wave_w2[0] = 0.0f;
            wave_wm[0] = 0.0f;
            int W = 1;
            bool any_superpos = false;
            bool allow_superpos = true;
            float chain_init_w = 1.0f;

            if (!tokenizer_.is_eog(wave_tok1[0]) && W_cap > 1) {
                // Build local suffix context ending at q0
                std::vector<int32_t> local_ctx;
                local_ctx.reserve(16);
                for (int k = std::min(15, P_0 + q0); k >= 1; --k) { local_ctx.push_back(get_tok_at(q0 - k)); }
                local_ctx.push_back(wave_tok1[0]);
                std::vector<int32_t> trie_guesses;
                if (guesser_) {
                    trie_guesses = guesser_->make_guesses(W_cap - 1, local_ctx);
                }

                for (int r = 1; r < W_cap; ++r) {
                    int pos = q0 + r;
                    int32_t prev1 = wave_tok1[r - 1];
                    int32_t prev2 = (r >= 2) ? wave_tok1[r - 2] : get_tok_at(pos - 2);
                    int32_t prev3 = (r >= 3) ? wave_tok1[r - 3] : get_tok_at(pos - 3);

                    int32_t cand1 = -1;
                    int32_t cand2 = -1;
                    float cp1 = 0.0f;
                    float cp2 = 0.0f;

                    // Priority A: LLM's own 3-gram transition table
                    if (prev3 >= 0 && prev2 >= 0 && prev1 >= 0) {
                        auto it3 = trigram_table.find(hash_triple(prev3, prev2, prev1));
                        if (it3 != trigram_table.end() && (it3->second.p1 >= 0.20f || it3->second.count >= 1)) {
                            cand1 = it3->second.tok1;
                            cand2 = it3->second.tok2;
                            cp1 = std::max(0.75f, it3->second.p1);
                            cp2 = it3->second.p2;
                        }
                    }

                    // Priority B: LLM's own 2-gram transition table
                    if (cand1 < 0 && prev2 >= 0 && prev1 >= 0) {
                        auto it2 = bigram_table.find(hash_pair(prev2, prev1));
                        if (it2 != bigram_table.end() && (it2->second.count >= 1 || it2->second.p1 >= 0.20f)) {
                            cand1 = it2->second.tok1;
                            cand2 = it2->second.tok2;
                            cp1 = std::max(0.65f, it2->second.p1);
                            cp2 = it2->second.p2;
                        }
                    }

                    // Priority C: ContextTrie / N-gram multi-token match
                    if (cand1 < 0 && r - 1 < static_cast<int>(trie_guesses.size())) {
                        cand1 = trie_guesses[r - 1];
                        cp1 = (r == 1) ? 0.78f : 0.80f;
                        if (canvas[pos].tok1 >= 0 && canvas[pos].tok1 != cand1) {
                            cand2 = canvas[pos].tok1;
                            cp2 = canvas[pos].p1 * 0.5f;
                        }
                    }

                    // Priority D: Existing active canvas token from previous diffusion pass
                    if (cand1 < 0 && canvas[pos].tok1 >= 0 && !tokenizer_.is_eog(canvas[pos].tok1) && canvas[pos].p1 >= 0.20f) {
                        cand1 = canvas[pos].tok1;
                        cand2 = canvas[pos].tok2;
                        cp1 = std::max(0.50f, canvas[pos].p1);
                        cp2 = canvas[pos].p2;
                    }

                    // Priority E: LLM's own 1-gram transition table
                    if (cand1 < 0 && prev1 >= 0) {
                        auto it1 = unigram_table.find(prev1);
                        if (it1 != unigram_table.end() && (it1->second.count >= 1 || it1->second.p1 >= 0.20f)) {
                            cand1 = it1->second.tok1;
                            cand2 = it1->second.tok2;
                            cp1 = std::max(0.55f, it1->second.p1);
                            cp2 = it1->second.p2;
                        }
                    }

                    // Compute-budget heuristic only: controls GPU work allocation.
                    // Never affects correctness; omitted slots remain reachable in subsequent passes.
                    if (cand1 < 0 || tokenizer_.is_eog(cand1)) {
                        break;
                    }
                    float init_w1 = cp1;
                    chain_init_w *= init_w1;
                    if (r == 1 && init_w1 < 0.20f) {
                        break;
                    }
                    if (r >= 2 && (init_w1 < 0.20f || chain_init_w < 0.04f)) {
                        break;
                    }

                    // Pure initialization of editable slot - DO NOT leak initialization priors into canvas confidence
                    canvas[pos].tok1 = cand1;
                    if (canvas[pos].state == SlotState::MASKED) {
                        canvas[pos].state = SlotState::ACTIVE;
                    }

                    wave_tok1[r] = cand1;
                    wave_tok2[r] = (cand2 >= 0 && cand2 != cand1) ? cand2 : cand1;
                    wave_pos[r] = P_0 + pos;

                    // S3 Continuous Embedding Superposition on far-horizon slots (r >= 2) with w1=0.45, w2=0.45, wm=0.10
                    if (allow_superpos && r >= 2 && cand2 >= 0 && cand2 != cand1) {
                        wave_w1[r] = 0.45f;
                        wave_w2[r] = 0.45f;
                        wave_wm[r] = 0.10f;
                        any_superpos = true;
                    } else {
                        wave_w1[r] = 1.0f;
                        wave_w2[r] = 0.0f;
                        wave_wm[r] = 0.0f;
                    }

                    W++;
                }
            }

            result.iterations++;
            result.total_diffusion_passes++;
            result.cumulative_active_work += W;

            // 2. Evaluate active wave on GPU
            LogitsView L;
            if (any_superpos) {
                L = backend_.eval_superpos(
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
            } else {
                L = backend_.eval_custom_pos(
                    wave_tok1.data(),
                    W,
                    wave_pos.data(),
                    (W > 1) || params.logprobs,
                    params.seq_id
                );
            }

            // 3. Reduce logits on GPU (or sample on host if non-greedy)
            std::vector<kernels::LogitsReduction> reds;
            if (pure_greedy) {
                reds = reduce_rows(L, W);
            } else {
                reds = reduce_rows(L, W);
                for (int r = 0; r < W && r < static_cast<int>(reds.size()); ++r) {
                    const float* row = L.row(r);
                    if (!row) row = backend_.get_logits_ith(r);
                    if (row) {
                        float conf = 1.0f;
                        reds[r].top_token = sampler_.sample(row, P_0 + q0 + r + 1, &conf);
                        reds[r].top_prob = conf;
                    }
                }
            }

            // 4. Update LLM transition tables from verified rows only (defer until K_lock is verified below)
            std::vector<TokenDiff> step_diffs;
            int K_lock = 1; // Slot q0 is always fed with exact discrete token (wave_w1[0] == 1.0f)
            while (K_lock < W && K_lock - 1 < static_cast<int>(reds.size())) {
                const auto& prev_red = reds[K_lock - 1];
                int32_t expected_tok = prev_red.top_token;
                bool tok_matched = false;
                int32_t chosen_tok = -1;

                if (expected_tok >= 0) {
                    if (wave_w1[K_lock] >= 1.0f) {
                        if (wave_tok1[K_lock] == expected_tok) {
                            tok_matched = true;
                            chosen_tok = expected_tok;
                        }
                    } else {
                        // Superposition slot: verified if base model preferred either candidate
                        if (wave_tok1[K_lock] == expected_tok) {
                            tok_matched = true;
                            chosen_tok = wave_tok1[K_lock];
                        } else if (wave_tok2[K_lock] == expected_tok) {
                            tok_matched = true;
                            chosen_tok = wave_tok2[K_lock];
                        }
                    }
                }

                if (tok_matched && !tokenizer_.is_eog(wave_tok1[K_lock - 1])) {
                    int verified_pos = q0 + K_lock;
                    canvas[verified_pos].tok1 = chosen_tok;
                    canvas[verified_pos].tok2 = prev_red.second_token;
                    canvas[verified_pos].p1   = prev_red.top_prob;
                    canvas[verified_pos].p2   = prev_red.second_prob;
                    canvas[verified_pos].entropy = prev_red.entropy_estimate;
                    wave_tok1[K_lock] = chosen_tok;
                    K_lock++;
                } else {
                    break;
                }
            }

            // Update LLM transition tables only from base-model verified prefix
            for (int r = 0; r < K_lock && r < static_cast<int>(reds.size()); ++r) {
                int pos = q0 + r;
                int32_t prev2 = (r >= 2) ? wave_tok1[r - 2] : get_tok_at(pos - 2);
                int32_t prev1 = (r >= 1) ? wave_tok1[r - 1] : get_tok_at(pos - 1);
                record_transition(prev2, prev1, wave_tok1[r], reds[r]);
                canvas[pos].passes++;
            }

            // Bidirectional Continuation Energy Check on the tip of the wave:
            // If an ambiguous slot u = q0 + K_lock - 1 led into a high-entropy continuation collapse,
            // wake u and revise it in-place to its runner-up tok2!
            int tip_r = K_lock - 1;
            int tip_pos = q0 + tip_r;
            bool can_back_revise = !pure_greedy && ((!is_hybrid_recr && K_lock >= 1) || (is_hybrid_recr && K_lock >= 2));
            if (can_back_revise &&
                tip_r < static_cast<int>(reds.size()) &&
                canvas[tip_pos].passes == 1 &&
                canvas[tip_pos].p1 > 0.0f && canvas[tip_pos].p1 < 0.42f &&
                canvas[tip_pos].tok2 >= 0 && canvas[tip_pos].tok2 != canvas[tip_pos].tok1 &&
                canvas[tip_pos].p2 >= 0.25f &&
                reds[tip_r].top_prob < 0.18f && reds[tip_r].entropy_estimate > 0.75f &&
                !tokenizer_.is_eog(canvas[tip_pos].tok2)) {

                int32_t old_tok = canvas[tip_pos].tok1;
                int32_t new_tok = canvas[tip_pos].tok2;
                TokenDiff diff;
                diff.pos = tip_pos;
                diff.old_id = old_tok;
                diff.new_id = new_tok;
                diff.old_text = tokenizer_.piece(old_tok);
                diff.new_text = tokenizer_.piece(new_tok);
                diff.confidence = canvas[tip_pos].p2;
                step_diffs.push_back(std::move(diff));

                canvas[tip_pos].tok1 = new_tok;
                canvas[tip_pos].p1 = canvas[tip_pos].p2;
                canvas[tip_pos].energy = canvas[tip_pos].p2;
                canvas[tip_pos].state = SlotState::ACTIVE;
                canvas[tip_pos].in_kv = false;
                K_lock--;
            }

            // Soft-lock the validated prefix [q0 .. q0 + K_lock - 1]
            std::vector<int32_t> newly_locked_tokens;
            newly_locked_tokens.reserve(std::max(1, K_lock));
            int actual_locked = 0;

            for (int r = 0; r < K_lock; ++r) {
                int pos = q0 + r;
                int32_t tok = wave_tok1[r];
                float fwd_prob = (r < static_cast<int>(reds.size())) ? reds[r].top_prob : 1.0f;

                canvas[pos].tok1 = tok;
                canvas[pos].kv_tok = tok;
                canvas[pos].in_kv = true;
                canvas[pos].energy = std::min(1.0f, std::max(THETA_LOCK, 0.65f * canvas[pos].p1 + 0.35f * fwd_prob + 0.25f));
                canvas[pos].state = SlotState::SOFT_LOCKED;
                actual_locked++;

                if (pos >= static_cast<int>(result.tokens.size())) {
                    sampler_.accept(tok);
                    result.tokens.push_back(tok);
                    newly_locked_tokens.push_back(tok);

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
                }
            }

            // Update horizon & downstream active slots from rows [actual_locked - 1 .. W - 1]
            if (!should_stop && actual_locked > 0) {
                int frontier_r = actual_locked - 1;
                int frontier_pos = q0 + actual_locked;
                if (frontier_r < static_cast<int>(reds.size()) && frontier_pos < n_predict) {
                    const auto& red = reds[frontier_r];
                    if (red.top_token >= 0) {
                        int32_t old_tok = canvas[frontier_pos].tok1;
                        if (old_tok >= 0 && old_tok != red.top_token && canvas[frontier_pos].passes > 0) {
                            TokenDiff diff;
                            diff.pos = frontier_pos;
                            diff.old_id = old_tok;
                            diff.new_id = red.top_token;
                            diff.old_text = tokenizer_.piece(old_tok);
                            diff.new_text = tokenizer_.piece(red.top_token);
                            diff.confidence = red.top_prob;
                            step_diffs.push_back(std::move(diff));

                            float delta_e = std::fabs(red.top_prob - canvas[frontier_pos].p1) + 0.35f;
                            int ripple_end = std::min(n_predict, frontier_pos + 16);
                            for (int u = frontier_pos + 1; u < ripple_end; ++u) {
                                if (canvas[u].state != SlotState::MASKED) {
                                    float damp = std::exp(-RIPPLE_LAMBDA * delta_e / static_cast<float>(u - frontier_pos));
                                    canvas[u].energy *= damp;
                                    if (canvas[u].state == SlotState::SOFT_LOCKED && canvas[u].energy < THETA_WAKE) {
                                        canvas[u].state = SlotState::ACTIVE;
                                        canvas[u].in_kv = false;
                                    }
                                }
                            }
                        }

                        canvas[frontier_pos].tok1 = red.top_token;
                        canvas[frontier_pos].tok2 = red.second_token;
                        canvas[frontier_pos].p1 = red.top_prob;
                        canvas[frontier_pos].p2 = red.second_prob;
                        canvas[frontier_pos].entropy = red.entropy_estimate;
                        canvas[frontier_pos].energy = red.top_prob;
                        canvas[frontier_pos].state = SlotState::ACTIVE;
                        canvas[frontier_pos].in_kv = false;
                    }
                }

                // Propagate downstream model predictions onto canvas for subsequent passes
                for (int r = actual_locked; r < W && r < static_cast<int>(reds.size()); ++r) {
                    int target_pos = q0 + r + 1;
                    if (target_pos >= n_predict) break;
                    const auto& red = reds[r];
                    if (red.top_token >= 0 && !tokenizer_.is_eog(red.top_token)) {
                        canvas[target_pos].tok1 = red.top_token;
                        canvas[target_pos].tok2 = red.second_token;
                        canvas[target_pos].p1 = red.top_prob * 0.50f;
                        canvas[target_pos].p2 = red.second_prob * 0.50f;
                        canvas[target_pos].entropy = red.entropy_estimate;
                        canvas[target_pos].energy = canvas[target_pos].p1;
                        canvas[target_pos].state = SlotState::ACTIVE;
                        canvas[target_pos].in_kv = false;
                    }
                }
            }

            // R1: Roll back KV cache & recurrent state to exact soft-locked frontier
            if (actual_locked < W || should_stop) {
                backend_.kv_truncate(P_0 + q0 + actual_locked, params.seq_id);
                for (int r = actual_locked; r < W; ++r) {
                    if (q0 + r < n_predict) {
                        canvas[q0 + r].in_kv = false;
                    }
                }
            }

            if (guesser_ && !newly_locked_tokens.empty()) {
                guesser_->update_ngram(newly_locked_tokens);
            }

            q0 += actual_locked;
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
