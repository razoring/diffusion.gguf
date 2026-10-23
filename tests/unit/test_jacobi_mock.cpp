#include "diffusion/backend.h"
#include "diffusion/engine.h"
#include "diffusion/sampler.h"
#include "diffusion/tokenizer.h"
#include "init_guess.h"
#include "scheduler_jacobi.h"
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

    LogitsView eval(const int32_t* toks, int n, int pos0, bool want_all) override {
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

    void kv_truncate(int pos) override {
        last_truncate_pos = pos;
    }

    void kv_clear() override {
        last_truncate_pos = 0;
    }

    int last_truncate_pos = -1;
private:
    std::vector<float> buf_;
};

int main() {
    MockBackend backend;
    MockTokenizer tokenizer;
    MockSampler sampler;

    // Test 1: Block mode Jacobi
    {
        InitGuesser guesser(InitStrategy::PAD, tokenizer, 1000, 42);
        SchedulerJacobi jacobi(backend, tokenizer, sampler, guesser);

        GenParams params;
        params.n_predict = 15;
        params.mode = DecodeMode::BLOCK;
        params.block_size = 4;

        int step_count = 0;
        int max_committed_per_step = 0;

        GenResult res = jacobi.run({1}, params, [&](const StepEvent& ev) {
            step_count++;
            max_committed_per_step = std::max(max_committed_per_step, static_cast<int>(ev.committed_new.size()));
            assert(!ev.committed_new.empty()); // Invariant I2: >= 1 token committed per step
            return true;
        });

        // Invariant I1 & Convergence: Must reach exact target sequence
        std::vector<int32_t> expected_tokens = {10, 20, 30, 40, 50, 60, 70, 80, 90, 999};
        assert(res.tokens == expected_tokens);
        assert(res.stop_reason == StopReason::EOG); // Invariant I4: stopped on EOG
        assert(res.iterations <= params.n_predict); // Invariant I2: iterations <= n_predict
        std::cout << "Test 1 (Block Jacobi): Passed. Iterations = " << res.iterations << ", Tokens = " << res.tokens.size() << "\n";
    }

    // Test 2: Whole mode Jacobi
    {
        InitGuesser guesser(InitStrategy::RANDOM, tokenizer, 1000, 123);
        SchedulerJacobi jacobi(backend, tokenizer, sampler, guesser);

        GenParams params;
        params.n_predict = 15;
        params.mode = DecodeMode::WHOLE;
        params.whole_max = 32;

        GenResult res = jacobi.run({1}, params, nullptr);
        std::vector<int32_t> expected_tokens = {10, 20, 30, 40, 50, 60, 70, 80, 90, 999};
        assert(res.tokens == expected_tokens);
        assert(res.stop_reason == StopReason::EOG);
        std::cout << "Test 2 (Whole Jacobi): Passed. Iterations = " << res.iterations << ", Tokens = " << res.tokens.size() << "\n";
    }

    std::cout << "All Jacobi mock invariant unit tests passed!\n";
    return 0;
}
