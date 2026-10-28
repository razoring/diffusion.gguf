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

struct LatencyBreakdown {
    int B;
    double attn_ms;      // 1. Attention & transformer computation (GPU kernel time)
    double sync_ms;      // 2. Synchronization & DtoH logit transfer
    double ptx_ms;       // 3. PTX / SIMD categorical reduction (argmax + confidence)
    double active_ms;    // 4. Active-query extraction & dynamic frontier tracking
    double branch_ms;    // 5. Branch evaluation (Shannon entropy H0 + budget check)
    double kv_ms;        // 6. KV cache updates (kv_truncate / seq_rm)
    double graph_ms;     // 7. Graph dispatch & launch overhead
    double total_ms;     // Total measured pass latency
};

int main() {
    std::cout << "================================================================================\n";
    std::cout << " DIFFUSION ENGINE LATENCY PROFILING & SUBSYSTEM SCALING ANALYSIS\n";
    std::cout << " Testing B = {1, 2, 3, 4, 8} on Dual NVIDIA GeForce RTX 3060 12GB\n";
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
    cfg.split_mode = 1; // layer split across dual RTX 3060

    std::unique_ptr<BackendLlama> backend;
    try {
        backend = std::make_unique<BackendLlama>(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[!] Failed to load model: " << e.what() << "\n";
        return 1;
    }

    int n_vocab = backend->n_vocab();
    std::cout << ">>> Model loaded. Vocab size: " << n_vocab << "\n";

    // Prefill a realistic 128-token context
    std::vector<int32_t> prompt(128);
    for (int i = 0; i < 128; ++i) prompt[i] = 100 + (i % 500);

    backend->kv_clear();
    backend->eval(prompt.data(), static_cast<int>(prompt.size()), 0, false);
    std::cout << ">>> Prompt prefilled (128 tokens in KV cache).\n\n";

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    std::vector<int> test_blocks = { 1, 2, 3, 4, 8 };
    std::vector<LatencyBreakdown> results;

    const int N_WARMUP = 5;
    const int N_MEASURE = 25;

    for (int B : test_blocks) {
        std::cout << ">>> Profiling Block Size B = " << B << " (" << N_MEASURE << " passes)..." << std::flush;

        std::vector<int32_t> canvas(B);
        std::vector<int32_t> custom_pos(B);
        for (int i = 0; i < B; ++i) {
            canvas[i] = 1000 + i;
            custom_pos[i] = 128 + i;
        }

        std::vector<uint8_t> frozen_mask(B, 0);
        frozen_mask[0] = 1;

        // Warmup passes
        for (int w = 0; w < N_WARMUP; ++w) {
            backend->kv_truncate(128);
            backend->eval_custom_pos(canvas.data(), B, custom_pos.data(), true);
            cudaDeviceSynchronize();
        }

        double sum_attn = 0.0;
        double sum_sync = 0.0;
        double sum_ptx = 0.0;
        double sum_active = 0.0;
        double sum_branch = 0.0;
        double sum_kv = 0.0;
        double sum_graph = 0.0;
        double sum_total = 0.0;

        for (int it = 0; it < N_MEASURE; ++it) {
            backend->kv_truncate(128);
            auto t_pass_start = std::chrono::high_resolution_clock::now();

            // 1. Attention & Transformer backbone computation
            // Evaluate backbone without all-logits projection
            cudaEventRecord(ev_start);
            backend->eval_custom_pos(canvas.data(), B, custom_pos.data(), false);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float gpu_backbone_ms = 0.0f;
            cudaEventElapsedTime(&gpu_backbone_ms, ev_start, ev_stop);

            // Rewind KV cache to isolate full diffusion pass with all B logits
            backend->kv_truncate(128);

            // 2. Full pass with B logits: measure CUDA graph replay launch & total GPU time
            cudaEventRecord(ev_start);
            auto t_graph_start = std::chrono::high_resolution_clock::now();
            auto logits_view = backend->eval_custom_pos(canvas.data(), B, custom_pos.data(), true);
            auto t_graph_end = std::chrono::high_resolution_clock::now();
            cudaEventRecord(ev_stop);

            // 3. Synchronization & logit fetch
            auto t_sync_start = std::chrono::high_resolution_clock::now();
            cudaEventSynchronize(ev_stop);
            const float* logits = backend->get_logits_ith(0);
            if (!logits) logits = logits_view.row(0);
            auto t_sync_end = std::chrono::high_resolution_clock::now();

            float gpu_full_ms = 0.0f;
            cudaEventElapsedTime(&gpu_full_ms, ev_start, ev_stop);

            double graph_replay_ms = std::chrono::duration<double, std::milli>(t_graph_end - t_graph_start).count();
            double sync_ms = std::chrono::duration<double, std::milli>(t_sync_end - t_sync_start).count();
            // Sync also includes the PCIe DtoH / LM head bottleneck delta over backbone:
            double sync_dtoh_ms = sync_ms + std::max(0.0, static_cast<double>(gpu_full_ms - gpu_backbone_ms));

            // 4. PTX / SIMD Categorical Reduction
            auto t_ptx_start = std::chrono::high_resolution_clock::now();
            std::vector<int32_t> canvas_out(B);
            std::vector<float> confs(B);
            for (int i = 0; i < B; ++i) {
                const float* row = logits + (i * n_vocab);
                canvas_out[i] = fast_argmax_simd(row, n_vocab);
                confs[i] = fast_confidence(row, n_vocab, canvas_out[i]);
            }
            auto t_ptx_end = std::chrono::high_resolution_clock::now();
            double ptx_ms = std::chrono::duration<double, std::milli>(t_ptx_end - t_ptx_start).count();

            // 5. Active-query extraction & dynamic frontier tracking
            auto t_act_start = std::chrono::high_resolution_clock::now();
            int frontier = 1;
            while (frontier < B && frozen_mask[frontier] == 1) frontier++;
            while (frontier < B) {
                if (canvas[frontier] == canvas_out[frontier]) {
                    frozen_mask[frontier] = 1;
                    frontier++;
                } else {
                    break;
                }
            }
            int W_active = 0;
            for (int i = 0; i < B; ++i) if (frozen_mask[i] == 0) W_active++;
            auto t_act_end = std::chrono::high_resolution_clock::now();
            double active_ms = std::chrono::duration<double, std::milli>(t_act_end - t_act_start).count();

            // 6. Branch evaluation (Shannon entropy H0 + sliding window budget check)
            auto t_br_start = std::chrono::high_resolution_clock::now();
            const float* p_row = logits;
            float max_l0 = p_row[0];
            for (int v = 1; v < n_vocab; ++v) {
                if (p_row[v] > max_l0) max_l0 = p_row[v];
            }
            double sum_exp0 = 0.0, sum_z_exp0 = 0.0;
            for (int v = 0; v < n_vocab; ++v) {
                double e = std::exp(static_cast<double>(p_row[v] - max_l0));
                sum_exp0 += e;
                sum_z_exp0 += static_cast<double>(p_row[v] - max_l0) * e;
            }
            float H0 = (sum_exp0 > 0.0) ? static_cast<float>(std::log(sum_exp0) - (sum_z_exp0 / sum_exp0)) : 0.0f;
            bool should_branch = (H0 >= 0.85f);
            (void)should_branch;
            auto t_br_end = std::chrono::high_resolution_clock::now();
            double branch_ms = std::chrono::duration<double, std::milli>(t_br_end - t_br_start).count();

            // 7. KV cache updates (kv_truncate / seq_rm)
            auto t_kv_start = std::chrono::high_resolution_clock::now();
            backend->kv_truncate(128 + frontier);
            backend->kv_truncate(128); // rewind for next pass benchmark
            auto t_kv_end = std::chrono::high_resolution_clock::now();
            double kv_ms = std::chrono::duration<double, std::milli>(t_kv_end - t_kv_start).count();

            auto t_pass_end = std::chrono::high_resolution_clock::now();
            double pass_total_ms = std::chrono::duration<double, std::milli>(t_pass_end - t_pass_start).count();

            sum_attn += gpu_backbone_ms;
            sum_sync += sync_dtoh_ms;
            sum_ptx += ptx_ms;
            sum_active += active_ms;
            sum_branch += branch_ms;
            sum_kv += kv_ms;
            sum_graph += graph_replay_ms;
            sum_total += pass_total_ms;
        }

        LatencyBreakdown lb;
        lb.B = B;
        lb.attn_ms = sum_attn / N_MEASURE;
        lb.sync_ms = sum_sync / N_MEASURE;
        lb.ptx_ms = sum_ptx / N_MEASURE;
        lb.active_ms = sum_active / N_MEASURE;
        lb.branch_ms = sum_branch / N_MEASURE;
        lb.kv_ms = sum_kv / N_MEASURE;
        lb.graph_ms = sum_graph / N_MEASURE;
        lb.total_ms = sum_total / N_MEASURE;
        results.push_back(lb);

        std::cout << " Done. Total pass: " << lb.total_ms << " ms (Backbone: " << lb.attn_ms << " ms, Sync: " << lb.sync_ms << " ms)\n";
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // Print Detailed Breakdown Table for all 7 subsystems
    std::cout << "\n" << std::string(115, '=') << "\n";
    std::cout << "                                  PER-PASS LATENCY BREAKDOWN BY SUBSYSTEM (ms)\n";
    std::cout << std::string(115, '=') << "\n";
    std::cout << std::left << std::setw(4) << "B"
              << " | " << std::setw(12) << "Attention"
              << " | " << std::setw(10) << "KV Update"
              << " | " << std::setw(12) << "Active-Query"
              << " | " << std::setw(10) << "PTX/SIMD"
              << " | " << std::setw(12) << "Branch Eval"
              << " | " << std::setw(14) << "Graph Replay"
              << " | " << std::setw(14) << "Sync & DtoH"
              << " | " << std::setw(10) << "Total (ms)\n";
    std::cout << std::string(115, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(4) << r.B
                  << " | " << std::setw(12) << std::fixed << std::setprecision(3) << r.attn_ms
                  << " | " << std::setw(10) << std::fixed << std::setprecision(3) << r.kv_ms
                  << " | " << std::setw(12) << std::fixed << std::setprecision(4) << r.active_ms
                  << " | " << std::setw(10) << std::fixed << std::setprecision(3) << r.ptx_ms
                  << " | " << std::setw(12) << std::fixed << std::setprecision(4) << r.branch_ms
                  << " | " << std::setw(14) << std::fixed << std::setprecision(3) << r.graph_ms
                  << " | " << std::setw(14) << std::fixed << std::setprecision(3) << r.sync_ms
                  << " | " << std::setw(10) << std::fixed << std::setprecision(3) << r.total_ms << "\n";
    }
    std::cout << std::string(115, '=') << "\n";

    // Scaling Exponent Analysis: y = c * B^beta  =>  beta = ln(y8 / y1) / ln(8)
    std::cout << "\n" << std::string(85, '=') << "\n";
    std::cout << "                      SUBSYSTEM SCALING EXPONENTS (B=1 to B=8)\n";
    std::cout << std::string(85, '=') << "\n";
    std::cout << "Subsystem                     | B=1 (ms) | B=8 (ms) | Ratio (8x) | Exponent beta | Scaling Type\n";
    std::cout << std::string(85, '-') << "\n";

    auto print_scaling = [](const char* name, double v1, double v8) {
        double ratio = (v1 > 1e-6) ? (v8 / v1) : 1.0;
        double beta = (v1 > 1e-6 && ratio > 1e-6) ? (std::log(ratio) / std::log(8.0)) : 0.0;
        const char* cls = "Sublinear O(B^<1)";
        if (beta > 1.15) cls = "SUPERLINEAR O(B^>1) [BOTTLENECK]";
        else if (beta >= 0.85) cls = "Linear O(B)";
        else if (beta < 0.20) cls = "Constant O(1)";

        std::cout << std::left << std::setw(30) << name
                  << " | " << std::setw(8) << std::fixed << std::setprecision(3) << v1
                  << " | " << std::setw(8) << std::fixed << std::setprecision(3) << v8
                  << " | " << std::setw(10) << std::fixed << std::setprecision(2) << ratio << "x"
                  << " | " << std::setw(13) << std::fixed << std::setprecision(2) << beta
                  << " | " << cls << "\n";
    };

    const auto& r1 = results[0];
    const auto& r8 = results.back();

    print_scaling("Attention Computation", r1.attn_ms, r8.attn_ms);
    print_scaling("KV Updates (kv_truncate)", r1.kv_ms, r8.kv_ms);
    print_scaling("Active-Query Extraction", r1.active_ms, r8.active_ms);
    print_scaling("PTX / SIMD Denoising", r1.ptx_ms, r8.ptx_ms);
    print_scaling("Branch Eval (Entropy)", r1.branch_ms, r8.branch_ms);
    print_scaling("CUDA Graph Replay", r1.graph_ms, r8.graph_ms);
    print_scaling("Synchronization & DtoH", r1.sync_ms, r8.sync_ms);
    print_scaling("Total Pass Latency", r1.total_ms, r8.total_ms);

    std::cout << std::string(85, '=') << "\n";

    return 0;
}
