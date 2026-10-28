#include "llama/backend_llama.h"
#include "fast_math.h"
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
    std::cout << " TEST SPARSE ACTIVE-QUERY EXECUTION ACROSS CONSECUTIVE PASSES\n";
    std::cout << " Ingest N=8 Anchors -> Refine N=4 -> Refine N=1\n";
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

    // Prefill 128 tokens
    std::vector<int32_t> prompt(128);
    for (int i = 0; i < 128; ++i) prompt[i] = 100 + (i % 500);
    backend->kv_clear();
    backend->eval(prompt.data(), static_cast<int>(prompt.size()), 0, false);
    std::cout << ">>> Prefilled 128 tokens in KV cache.\n\n";

    const int B = 8;
    std::vector<int32_t> canvas(B);
    std::vector<int32_t> custom_pos(B);
    for (int i = 0; i < B; ++i) {
        canvas[i] = 2000 + i;
        custom_pos[i] = 128 + i;
    }

    // Simulate 3 diffusion blocks with shrinking active queries:
    // Block 0: pos 128..135
    //   Pass 1: N=8 (tokens 0..7 active, pos 128..135) -> Ingest K/V anchors
    //   Pass 2: N=4 (tokens 4..7 active, pos 132..135) -> Refine active queries
    //   Pass 3: N=1 (token 7 active,   pos 135)        -> Commit tail
    // Repeat for Block 1 (pos 136..143) and Block 2 (pos 144..151)

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    std::cout << std::string(98, '=') << "\n";
    std::cout << "Step | Block | Pass | N (Active) | Pos Range  | GPU (ms)  | Host (ms) | Top Token @ Tail\n";
    std::cout << std::string(98, '-') << "\n";

    int cur_base_pos = 128;
    int step = 0;

    for (int block = 0; block < 4; ++block) {
        // Pass 1: Full block N=8 (anchor write)
        {
            step++;
            std::vector<int32_t> block_pos(B);
            for (int i = 0; i < B; ++i) block_pos[i] = cur_base_pos + i;

            auto t0 = std::chrono::high_resolution_clock::now();
            cudaEventRecord(ev_start);
            auto view = backend->eval_custom_pos(canvas.data(), B, block_pos.data(), true);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            auto t1 = std::chrono::high_resolution_clock::now();

            float gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
            double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int32_t top_tok = fast_argmax_simd(view.row(B - 1), backend->n_vocab());
            std::cout << std::left << std::setw(5) << step
                      << "| " << std::setw(6) << block
                      << "| " << std::setw(5) << "1"
                      << "| " << std::setw(11) << B
                      << "| [" << cur_base_pos << ".." << (cur_base_pos + B - 1) << "]  "
                      << "| " << std::setw(10) << std::fixed << std::setprecision(3) << gpu_ms
                      << "| " << std::setw(10) << std::fixed << std::setprecision(3) << host_ms
                      << "| " << top_tok << "\n";
        }

        // Pass 2: Half block N=4 (active queries 4..7)
        {
            step++;
            std::vector<int32_t> active_pos = { cur_base_pos + 4, cur_base_pos + 5, cur_base_pos + 6, cur_base_pos + 7 };
            std::vector<int32_t> active_toks = { canvas[4], canvas[5], canvas[6], canvas[7] };

            backend->kv_truncate(cur_base_pos + 4); // Rewind active slots, preserve frozen anchors
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaEventRecord(ev_start);
            auto view = backend->eval_custom_pos(active_toks.data(), 4, active_pos.data(), true);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            auto t1 = std::chrono::high_resolution_clock::now();

            float gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
            double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int32_t top_tok = fast_argmax_simd(view.row(3), backend->n_vocab());
            std::cout << std::left << std::setw(5) << step
                      << "| " << std::setw(6) << block
                      << "| " << std::setw(5) << "2"
                      << "| " << std::setw(11) << 4
                      << "| [" << (cur_base_pos + 4) << ".." << (cur_base_pos + B - 1) << "]  "
                      << "| " << std::setw(10) << std::fixed << std::setprecision(3) << gpu_ms
                      << "| " << std::setw(10) << std::fixed << std::setprecision(3) << host_ms
                      << "| " << top_tok << "\n";
        }

        // Pass 3: Tail N=1 (active query 7)
        {
            step++;
            std::vector<int32_t> active_pos = { cur_base_pos + 7 };
            std::vector<int32_t> active_toks = { canvas[7] };

            backend->kv_truncate(cur_base_pos + 7); // Rewind tail slot, preserve frozen anchors
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaEventRecord(ev_start);
            auto view = backend->eval_custom_pos(active_toks.data(), 1, active_pos.data(), true);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            auto t1 = std::chrono::high_resolution_clock::now();

            float gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
            double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int32_t top_tok = fast_argmax_simd(view.row(0), backend->n_vocab());
            std::cout << std::left << std::setw(5) << step
                      << "| " << std::setw(6) << block
                      << "| " << std::setw(5) << "3"
                      << "| " << std::setw(11) << 1
                      << "| [" << (cur_base_pos + 7) << "]      "
                      << "| " << std::setw(10) << std::fixed << std::setprecision(3) << gpu_ms
                      << "| " << std::setw(10) << std::fixed << std::setprecision(3) << host_ms
                      << "| " << top_tok << "\n";
        }

        cur_base_pos += B; // Commit block and advance
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    std::cout << std::string(98, '=') << "\n";

    return 0;
}
