#include "kernels/diff_device.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
#include <cuda_runtime.h>
#endif

using namespace diffusion::kernels;

int main() {
    std::cout << "[Test] Initializing diff_device kernel tests..." << std::endl;

    const size_t B = 8;
    const size_t V = 1024;
    DeviceDiffusionBuffers bufs;

    bool alloc_ok = diff_device_alloc(bufs, B);
    assert(alloc_ok && "diff_device_alloc failed");
    std::cout << "  [+] diff_device_alloc succeeded for B=" << B << std::endl;

    // Test 1: Upload canvas
    std::vector<int32_t> initial_canvas = {10, 20, 30, 40, 50, 60, 70, 80};
    diff_device_set_canvas(bufs, initial_canvas.data(), nullptr, B);

    std::vector<int32_t> downloaded_canvas(B, 0);
    std::vector<float> downloaded_conf(B, 0.0f);
    std::vector<uint8_t> downloaded_mask(B, 0);

    // Swap canvas to test get_canvas from canvas_out
    diff_device_swap_canvas(bufs);
    diff_device_get_canvas(bufs, downloaded_canvas.data(), downloaded_conf.data(), downloaded_mask.data(), B);
    for (size_t i = 0; i < B; ++i) {
        assert(downloaded_canvas[i] == initial_canvas[i]);
    }
    std::cout << "  [+] Canvas upload and download verified." << std::endl;

    // Swap back
    diff_device_swap_canvas(bufs);

    // Test 2: Categorical reduction over active query set
    // Suppose positions 1, 2, 3, 4 are active (W_active = 4), and 0, 5, 6, 7 are frozen
    size_t W_active = 4;
    std::vector<int32_t> active_indices = {1, 2, 3, 4};
    std::vector<float> h_logits(B * V, 0.0f);

    // Set peak logits for active queries (row b - 1 predicts canvas b):
    // Row 0 -> canvas 1 -> token 100
    // Row 1 -> canvas 2 -> token 200
    // Row 2 -> canvas 3 -> token 300
    // Row 3 -> canvas 4 -> token 400
    h_logits[0 * V + 100] = 10.0f;
    h_logits[1 * V + 200] = 12.0f;
    h_logits[2 * V + 300] = 8.0f;
    h_logits[3 * V + 400] = 15.0f;

    // Simulate freezing positions 0, 5, 6, 7
    std::vector<uint8_t> h_frozen_mask(B, 0);
    h_frozen_mask[0] = 1;
    for (size_t i = 5; i < B; ++i) h_frozen_mask[i] = 1;

#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
    float* d_logits = nullptr;
    int32_t* d_active_idx = nullptr;
    cudaMalloc(&d_logits, B * V * sizeof(float));
    cudaMalloc(&d_active_idx, W_active * sizeof(int32_t));
    cudaMemcpy(d_logits, h_logits.data(), B * V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active_idx, active_indices.data(), W_active * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bufs.d_frozen_mask, h_frozen_mask.data(), B * sizeof(uint8_t), cudaMemcpyHostToDevice);

    diff_device_sample_and_anchor(d_logits, d_active_idx, W_active, B, V, 0.85f, bufs, nullptr, 0.0f, 1, false);
    diff_device_check_convergence(bufs, B, 0.85f, 1, false, nullptr);

    cudaFree(d_logits);
    cudaFree(d_active_idx);
#else
    memcpy(bufs.d_frozen_mask, h_frozen_mask.data(), B * sizeof(uint8_t));
    diff_device_sample_and_anchor(h_logits.data(), active_indices.data(), W_active, B, V, 0.85f, bufs, nullptr, 0.0f, 1, false);
    diff_device_check_convergence(bufs, B, 0.85f, 1, false, nullptr);
#endif

    diff_device_get_canvas(bufs, downloaded_canvas.data(), downloaded_conf.data(), downloaded_mask.data(), B);

    // Verify position 0 is preserved as prefix anchor
    assert(downloaded_canvas[0] == 10);

    // Verify active positions updated to targets
    assert(downloaded_canvas[1] == 100);
    assert(downloaded_canvas[2] == 200);
    assert(downloaded_canvas[3] == 300);
    assert(downloaded_canvas[4] == 400);

    // Verify frozen positions retained original canvas values
    assert(downloaded_canvas[5] == 60);
    assert(downloaded_canvas[6] == 70);
    assert(downloaded_canvas[7] == 80);

    std::cout << "  [+] Active query updates and frozen anchor retention verified." << std::endl;

    bool converged = diff_device_is_converged(bufs);
    // Since downloaded_canvas [10, 100, 200, 300, 400, 60, 70, 80] != initial_canvas [10..80], converged must be false
    assert(!converged && "Canvas should not be converged after initial change");
    std::cout << "  [+] Convergence check (non-converged state) verified." << std::endl;

    // Now copy canvas_out to canvas_in and check convergence again
    diff_device_swap_canvas(bufs);
    // canvas_in now has [10, 100, 200, 300, 400, 60, 70, 80]
    // Re-run sample with identical logits on pass 2
#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
    cudaMalloc(&d_logits, B * V * sizeof(float));
    cudaMalloc(&d_active_idx, W_active * sizeof(int32_t));
    cudaMemcpy(d_logits, h_logits.data(), B * V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active_idx, active_indices.data(), W_active * sizeof(int32_t), cudaMemcpyHostToDevice);

    diff_device_sample_and_anchor(d_logits, d_active_idx, W_active, B, V, 0.85f, bufs, nullptr, 0.0f, 2, false);
    diff_device_check_convergence(bufs, B, 0.85f, 2, false, nullptr);

    cudaFree(d_logits);
    cudaFree(d_active_idx);
#else
    diff_device_sample_and_anchor(h_logits.data(), active_indices.data(), W_active, B, V, 0.85f, bufs, nullptr, 0.0f, 2, false);
    diff_device_check_convergence(bufs, B, 0.85f, 2, false, nullptr);
#endif

    converged = diff_device_is_converged(bufs);
    assert(converged && "Canvas should be converged when identical outputs are produced");
    std::cout << "  [+] Fixed-point stationarity convergence verified." << std::endl;

    diff_device_free(bufs);
    std::cout << "\nAll diff_device kernel invariant unit tests PASSED!" << std::endl;
    return 0;
}
