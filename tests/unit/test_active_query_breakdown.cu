#include "llama/backend_llama.h"
#include "fast_math.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <cuda_runtime.h>

using namespace diffusion;

const char* MODEL_PATH = "C:\\Users\\conta\\.ollama\\models\\blobs\\sha256-370c2879f17648d337b0ab4e208fdce348e24d1060064e93d7645daafd56d39c";

int main() {
    std::cout << "================================================================================\n";
    std::cout << " ACTIVE-QUERY EXECUTION, FLOP BREAKDOWN & CUDA GRAPH ANALYSIS\n";
    std::cout << " Comparing W_active in {8, 4, 1} for Block Size B = 8\n";
    std::cout << "================================================================================\n";

    if (!std::filesystem::exists(MODEL_PATH)) {
        std::cerr << "[SKIP] Model blob not found at: " << MODEL_PATH << "\n";
        return 0;
    }

    BackendConfig cfg;
    cfg.model_path = MODEL_PATH;
    cfg.n_gpu_layers = 99;
    cfg.n_ctx = 2048;
    cfg.flash_attn = true;
    cfg.swa_full = true;
    cfg.split_mode = 1;

    std::unique_ptr<BackendLlama> backend;
    try {
        backend = std::make_unique<BackendLlama>(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[!] Failed to load model: " << e.what() << "\n";
        return 1;
    }

    int n_vocab = backend->n_vocab();
    int n_embd = backend->n_embd();
    llama_model* model = backend->raw_model();
    int n_layer = llama_model_n_layer(model);
    int n_head = llama_model_n_head(model);
    int n_head_kv = llama_model_n_head_kv(model);

    std::cout << "Model Architecture Parameters:\n";
    std::cout << "  n_layer:   " << n_layer << "\n";
    std::cout << "  n_embd:    " << n_embd << "\n";
    std::cout << "  n_vocab:   " << n_vocab << "\n";
    std::cout << "  n_head:    " << n_head << "\n";
    std::cout << "  n_head_kv: " << n_head_kv << "\n\n";

    // Prefill 128 tokens
    std::vector<int32_t> prompt(128);
    for (int i = 0; i < 128; ++i) prompt[i] = 100 + (i % 500);
    backend->kv_clear();
    backend->eval(prompt.data(), static_cast<int>(prompt.size()), 0, false);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    const int B = 8;
    std::vector<int32_t> canvas(B);
    for (int i = 0; i < B; ++i) canvas[i] = 1000 + i;

    // Test three scenarios for Block B = 8:
    // Case 1: W_active = 8 (all 8 tokens active, batch.n_tokens = 8, 8 logits)
    // Case 2: W_active = 4 (tokens 4..7 active, batch.n_tokens = 8, 4 logits)
    // Case 3: W_active = 1 (token 7 active, batch.n_tokens = 8, 1 logit)
    // Case 4: Actual batch collapse N = 4 (batch.n_tokens = 4, 4 logits)
    // Case 5: Actual batch collapse N = 1 (batch.n_tokens = 1, 1 logit)

    struct CaseConfig {
        std::string name;
        int n_tokens;
        int w_active;
        std::vector<uint8_t> mask;
    };

    std::vector<CaseConfig> cases = {
        { "Fixed B=8, W_active=8 (All logits)", 8, 8, {1,1,1,1, 1,1,1,1} },
        { "Fixed B=8, W_active=4 (Masked logits)", 8, 4, {0,0,0,0, 1,1,1,1} },
        { "Fixed B=8, W_active=1 (Single logit)", 8, 1, {0,0,0,0, 0,0,0,1} },
        { "Batch N=4 (Actual query shrink)", 4, 4, {1,1,1,1} },
        { "Batch N=1 (Single token decode)", 1, 1, {1} }
    };

    const int N_WARMUP = 4;
    const int N_MEASURE = 20;

    std::cout << std::string(105, '=') << "\n";
    std::cout << "             EMPIRICAL TIMING UNDER VARYING W_active AND BATCH SIZES\n";
    std::cout << std::string(105, '=') << "\n";
    std::cout << std::left << std::setw(38) << "Execution Strategy"
              << " | " << std::setw(10) << "GPU Total"
              << " | " << std::setw(10) << "Host Pass"
              << " | " << std::setw(12) << "DtoH / Sync"
              << " | " << std::setw(12) << "CUDA Graph State\n";
    std::cout << std::string(105, '-') << "\n";

    for (const auto& c : cases) {
        // Warmup
        for (int w = 0; w < N_WARMUP; ++w) {
            backend->kv_truncate(128);
            if (c.n_tokens == B) {
                backend->eval_active(canvas.data(), c.n_tokens, 128, c.mask.data());
            } else {
                backend->eval(canvas.data(), c.n_tokens, 128, true);
            }
            cudaDeviceSynchronize();
        }

        double sum_gpu = 0.0;
        double sum_host = 0.0;
        double sum_sync = 0.0;

        for (int it = 0; it < N_MEASURE; ++it) {
            backend->kv_truncate(128);
            cudaDeviceSynchronize();

            auto t_start = std::chrono::high_resolution_clock::now();
            cudaEventRecord(ev_start);

            if (c.n_tokens == B) {
                backend->eval_active(canvas.data(), c.n_tokens, 128, c.mask.data());
            } else {
                backend->eval(canvas.data(), c.n_tokens, 128, true);
            }

            cudaEventRecord(ev_stop);
            auto t_host_end = std::chrono::high_resolution_clock::now();

            auto t_sync_start = std::chrono::high_resolution_clock::now();
            cudaEventSynchronize(ev_stop);
            auto t_sync_end = std::chrono::high_resolution_clock::now();

            float gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);

            double host_ms = std::chrono::duration<double, std::milli>(t_host_end - t_start).count();
            double sync_ms = std::chrono::duration<double, std::milli>(t_sync_end - t_sync_start).count();

            sum_gpu += gpu_ms;
            sum_host += host_ms;
            sum_sync += sync_ms;
        }

        double avg_gpu = sum_gpu / N_MEASURE;
        double avg_host = sum_host / N_MEASURE;
        double avg_sync = sum_sync / N_MEASURE;

        std::cout << std::left << std::setw(38) << c.name
                  << " | " << std::setw(10) << std::fixed << std::setprecision(3) << avg_gpu << " ms"
                  << " | " << std::setw(10) << std::fixed << std::setprecision(3) << avg_host << " ms"
                  << " | " << std::setw(12) << std::fixed << std::setprecision(3) << avg_sync << " ms"
                  << " | Warmup cached\n";
    }

    std::cout << std::string(105, '=') << "\n\n";

    // Now test alternating / shrinking W_active to observe CUDA Graph Invalidation directly
    std::cout << ">>> Testing Dynamic W_active Transitions: 8 -> 4 -> 1 on consecutive passes...\n";
    std::vector<int> dynamic_w = { 8, 4, 1, 8, 4, 1 };
    for (int w : dynamic_w) {
        backend->kv_truncate(128);
        std::vector<uint8_t> mask(B, 0);
        for (int i = B - w; i < B; ++i) mask[i] = 1;

        auto t0 = std::chrono::high_resolution_clock::now();
        backend->eval_active(canvas.data(), B, 128, mask.data());
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "    Step W_active = " << w << ": Pass latency = " << ms << " ms\n";
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
