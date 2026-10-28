#include "diffusion/backend.h"
#include "diffusion/engine.h"
#include "diffusion/sampler.h"
#include "diffusion/tokenizer.h"
#include "scheduler_diffusion.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using namespace diffusion;

class MockTokenizer : public ITokenizer {
public:
    std::vector<int32_t> tokenize(std::string_view, bool, bool) const override {
        return {1}; // BOS
    }
    std::string piece(int32_t tok) const override {
        return "t" + std::to_string(tok) + " ";
    }
    bool is_eog(int32_t tok) const override {
        return tok == 999;
    }
    int32_t pad_or_fallback() const override { return 0; }
    int32_t bos() const override { return 1; }
    int32_t eos() const override { return 999; }
};

class MockSampler : public ISamplerChain {
public:
    MockSampler() = default;

    std::unique_ptr<ISamplerChain> clone() const override {
        auto copy = std::make_unique<MockSampler>();
        copy->history_ = history_;
        return copy;
    }

    void accept(int32_t tok) override {
        history_.push_back(tok);
    }

    bool can_accept(int32_t) const override {
        return true;
    }

    int32_t sample(const float* logits, int, float* out_prob) override {
        // Greedy argmax over logits
        int best_idx = 0;
        float best_val = logits[0];
        for (int i = 1; i < 1000; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        if (out_prob) *out_prob = 1.0f;
        return best_idx;
    }

private:
    std::vector<int32_t> history_;
};

class MockBackend : public IModelBackend {
public:
    // Fixed target trajectory: 1 -> 10 -> 20 -> 30 -> 40 -> 50 -> 60 -> 70 -> 80 -> 90 -> 999 (EOS)
    std::vector<int32_t> target = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 999};

    int n_vocab() const override { return 1000; }
    int n_ctx() const override { return 2048; }
    bool supports_kv_truncate() const override { return true; }

    LogitsView eval(const int32_t* toks, int n, int pos0, bool want_all, int32_t seq_id = 0) override {
        (void)toks;
        (void)seq_id;
        int n_rows = want_all ? n : 1;
        buf_.assign(n_rows * 1000, -100.0f);

        for (int r = 0; r < n_rows; ++r) {
            int token_pos = want_all ? (pos0 + r) : (pos0 + n - 1);
            int next_target_pos = token_pos + 1;
            int32_t correct_tok = 0;
            if (next_target_pos < static_cast<int>(target.size())) {
                correct_tok = target[next_target_pos];
            } else {
                correct_tok = 999; // EOS
            }
            buf_[r * 1000 + correct_tok] = 100.0f; // High confidence for target
        }
        return LogitsView{buf_.data(), 1000, n_rows};
    }

    void kv_truncate(int pos, int32_t seq_id = 0) override {
        (void)seq_id;
        last_truncate_pos = pos;
    }

    void kv_clear(int32_t seq_id = -1) override {
        (void)seq_id;
        last_truncate_pos = 0;
    }

    int last_truncate_pos = -1;

private:
    std::vector<float> buf_;
};

int main() {
    MockTokenizer tokenizer;
    MockSampler sampler;
    MockBackend backend;

    std::cout << "[test_diffusion_mock] Starting unit tests for SchedulerDiffusion...\n";

    // Test 1: B=1 AR baseline
    {
        SchedulerDiffusion scheduler(backend, tokenizer, sampler, 1, 1);
        GenParams params;
        params.block_size = 1;
        params.n_predict = 10;
        std::vector<int32_t> prompt = {1}; // BOS

        GenResult res = scheduler.run(prompt, params);
        std::cout << "[Test 1: B=1] Generated " << res.tokens.size() << " tokens. Stop reason: " << (int)res.stop_reason << "\n";
        assert(!res.tokens.empty());
        assert(res.tokens.front() == 10);
        assert(res.tokens.back() == 999);
        assert(res.stop_reason == StopReason::EOG);
        assert(res.cumulative_active_work == static_cast<int>(res.tokens.size()));
        std::cout << "  Passed!\n";
    }

    // Test 2: B=8 Discrete Block Diffusion
    {
        MockSampler sampler2;
        SchedulerDiffusion scheduler(backend, tokenizer, sampler2, 8, 4);
        GenParams params;
        params.block_size = 8;
        params.n_predict = 10;
        std::vector<int32_t> prompt = {1}; // BOS

        GenResult res = scheduler.run(prompt, params);
        std::cout << "[Test 2: B=8] Generated " << res.tokens.size() << " tokens in " << res.iterations << " iterations.\n";
        std::cout << "  C_effective (cumulative active work): " << res.cumulative_active_work << "\n";
        std::cout << "  Total blocks: " << res.total_blocks << ", Total passes: " << res.total_diffusion_passes << "\n";
        std::cout << "  Avg active tokens/pass: " << res.avg_active_tokens_per_pass << "\n";
        std::cout << "  Avg passes/block: " << res.avg_passes_per_block << "\n";

        assert(!res.tokens.empty());
        assert(res.tokens.front() == 10);
        assert(res.tokens.back() == 999);
        assert(res.stop_reason == StopReason::EOG);
        assert(res.cumulative_active_work > 0);
        assert(res.total_blocks > 0);
        assert(res.total_diffusion_passes > 0);
        assert(res.avg_active_tokens_per_pass > 0.0);
        assert(res.avg_passes_per_block >= 1.0);
        std::cout << "  Passed!\n";
    }

    std::cout << "[test_diffusion_mock] ALL TESTS PASSED!\n";
    return 0;
}
