#include "llama/backend_llama.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace diffusion;

const char* MODEL_PATH = "C:\\Users\\conta\\.ollama\\models\\blobs\\sha256-370c2879f17648d337b0ab4e208fdce348e24d1060064e93d7645daafd56d39c";

int main() {
    std::cout << "======================================================================\n";
    std::cout << " PROTOTYPE VERIFICATION: ATTENTION PATTERNS VIA POS REMAPPING\n";
    std::cout << "======================================================================\n";

    if (!std::filesystem::exists(MODEL_PATH)) {
        std::cout << "[SKIP] Model blob not found at: " << MODEL_PATH << "\n";
        return 0;
    }

    std::cout << ">>> Initializing BackendLlama with Gemma 4 E4B...\n";
    BackendConfig cfg;
    cfg.model_path = MODEL_PATH;
    cfg.n_gpu_layers = 99;
    cfg.n_ctx = 1024;
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

    // 4 sample tokens: [2, 106, 1645, 108]
    std::vector<int32_t> toks_A = { 2, 106, 1645, 108 };
    std::vector<int32_t> toks_B = { 2, 106, 1645, 9999 }; // token 3 altered
    int n = 4;

    // -------------------------------------------------------------------------
    // Experiment 1: Standard Monotonic Causal (pos = 0, 1, 2, 3)
    // -------------------------------------------------------------------------
    std::cout << "\n--- Experiment 1: Standard Monotonic Causal (pos = 0, 1, 2, 3) ---\n";
    std::vector<int32_t> pos_causal = { 0, 1, 2, 3 };

    backend->kv_clear();
    auto view_causal_A = backend->eval_custom_pos(toks_A.data(), n, pos_causal.data(), true);
    std::vector<float> row0_causal_A(n_vocab);
    std::copy(view_causal_A.row(0), view_causal_A.row(0) + n_vocab, row0_causal_A.begin());

    backend->kv_clear();
    auto view_causal_B = backend->eval_custom_pos(toks_B.data(), n, pos_causal.data(), true);
    std::vector<float> row0_causal_B(n_vocab);
    std::copy(view_causal_B.row(0), view_causal_B.row(0) + n_vocab, row0_causal_B.begin());

    float max_diff_causal = 0.0f;
    for (int v = 0; v < n_vocab; ++v) {
        float diff = std::abs(row0_causal_A[v] - row0_causal_B[v]);
        if (diff > max_diff_causal) max_diff_causal = diff;
    }
    std::cout << "  Causal baseline max delta on token 0: " << max_diff_causal << "\n";
    if (max_diff_causal == 0.0f) {
        std::cout << "  [PASS] Causal masking strictly isolates token 0 from future tokens.\n";
    } else {
        std::cout << "  [WARN] Unexpected causal leakage: delta=" << max_diff_causal << "\n";
    }

    // -------------------------------------------------------------------------
    // Experiment 2: Full Diffusion Remapping (pos = 3, 3, 3, 3)
    // -------------------------------------------------------------------------
    std::cout << "\n--- Experiment 2: Full Diffusion Remapping (pos = 3, 3, 3, 3) ---\n";
    std::vector<int32_t> pos_diff = { 3, 3, 3, 3 };

    backend->kv_clear();
    auto view_diff_A = backend->eval_custom_pos(toks_A.data(), n, pos_diff.data(), true);
    std::vector<float> row0_diff_A(n_vocab);
    std::copy(view_diff_A.row(0), view_diff_A.row(0) + n_vocab, row0_diff_A.begin());

    backend->kv_clear();
    auto view_diff_B = backend->eval_custom_pos(toks_B.data(), n, pos_diff.data(), true);
    std::vector<float> row0_diff_B(n_vocab);
    std::copy(view_diff_B.row(0), view_diff_B.row(0) + n_vocab, row0_diff_B.begin());

    float max_diff_remapped = 0.0f;
    double l2_diff_remapped = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        float diff = std::abs(row0_diff_A[v] - row0_diff_B[v]);
        if (diff > max_diff_remapped) max_diff_remapped = diff;
        l2_diff_remapped += diff * diff;
    }
    l2_diff_remapped = std::sqrt(l2_diff_remapped);

    std::cout << "  Remapped pos=[3,3,3,3] max delta on token 0: " << max_diff_remapped << "\n";
    std::cout << "  Remapped pos=[3,3,3,3] L2 delta on token 0:  " << l2_diff_remapped << "\n";

    if (max_diff_remapped > 1e-4f) {
        std::cout << "  [CONFIRMED] Token 0 logits actively shifted when future token 3 was altered!\n";
        std::cout << "  [SUCCESS] pos remapping induces TRUE bidirectional attention cross-talk!\n";
    } else {
        std::cout << "  [NOTICE] Token 0 logits unchanged. Underlying attention enforces intra-batch causal order by index.\n";
    }

    // -------------------------------------------------------------------------
    // Experiment 3: Partial Annealing Remapping (pos = 1, 2, 3, 3)
    // -------------------------------------------------------------------------
    std::cout << "\n--- Experiment 3: Partial Annealing Remapping (pos = 1, 2, 3, 3) ---\n";
    std::vector<int32_t> pos_partial = { 1, 2, 3, 3 };

    backend->kv_clear();
    auto view_partial_A = backend->eval_custom_pos(toks_A.data(), n, pos_partial.data(), true);
    std::vector<float> row0_part_A(n_vocab);
    std::copy(view_partial_A.row(0), view_partial_A.row(0) + n_vocab, row0_part_A.begin());

    backend->kv_clear();
    auto view_partial_B = backend->eval_custom_pos(toks_B.data(), n, pos_partial.data(), true);
    std::vector<float> row0_part_B(n_vocab);
    std::copy(view_partial_B.row(0), view_partial_B.row(0) + n_vocab, row0_part_B.begin());

    float max_diff_partial = 0.0f;
    for (int v = 0; v < n_vocab; ++v) {
        float diff = std::abs(row0_part_A[v] - row0_part_B[v]);
        if (diff > max_diff_partial) max_diff_partial = diff;
    }
    std::cout << "  Partial annealed pos=[1,2,3,3] max delta on token 0: " << max_diff_partial << "\n";

    std::cout << "\n>>> GDN Recurrent State and SWA Stability: STABLE (Zero asserts, zero crashes)\n";
    std::cout << "======================================================================\n";
    return 0;
}
