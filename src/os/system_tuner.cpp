#include "system_tuner.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

namespace diffusion {
namespace os {

// Defined in system_tuner_cuda.cu when CUDA enabled
void query_cuda_specs(SystemProfile& profile);

SystemProfile detect_system_profile() {
    SystemProfile p;

#if defined(_WIN32)
    p.os_name = "Windows";
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&mem_info)) {
        p.total_ram_bytes = mem_info.ullTotalPhys;
        p.free_ram_bytes = mem_info.ullAvailPhys;
    }
#elif defined(__linux__)
    p.os_name = "Linux";
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        p.total_ram_bytes = static_cast<size_t>(si.totalram) * si.mem_unit;
        p.free_ram_bytes = static_cast<size_t>(si.freeram) * si.mem_unit;
    }
#elif defined(__APPLE__)
    p.os_name = "macOS";
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t size = 0;
    size_t len = sizeof(size);
    if (sysctl(mib, 2, &size, &len, NULL, 0) == 0) {
        p.total_ram_bytes = static_cast<size_t>(size);
        p.free_ram_bytes = static_cast<size_t>(size / 2);
    }
#else
    p.os_name = "Generic POSIX";
#endif

    unsigned int hw = std::thread::hardware_concurrency();
    p.cpu_threads = (hw > 0) ? hw : 4;

    query_cuda_specs(p);

    return p;
}

static bool str_ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

void auto_tune_system(
    BackendConfig& backend,
    GenParams& params,
    const ExplicitFlags& flags,
    bool verbose
) {
    SystemProfile sys = detect_system_profile();

    size_t model_bytes = 0;
    int n_layer_all = 0;
    int n_head_kv = 0;
    int key_length = 0;
    int full_attention_interval = 1;
    uint32_t train_ctx_len = 0;

    if (!backend.model_path.empty()) {
        try {
            if (std::filesystem::exists(backend.model_path)) {
                model_bytes = std::filesystem::file_size(backend.model_path);

                // Quick metadata inspection via GGUF header
                struct gguf_init_params p = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
                struct gguf_context* gctx = gguf_init_from_file(backend.model_path.c_str(), p);
                if (gctx) {
                    int64_t n_kv = gguf_get_n_kv(gctx);
                    uint32_t embd = 0;
                    uint32_t n_head = 0;

                    for (int64_t i = 0; i < n_kv; ++i) {
                        const char* k = gguf_get_key(gctx, i);
                        if (!k) continue;
                        std::string key(k);

                        if (str_ends_with(key, ".block_count") && n_layer_all == 0) {
                            n_layer_all = static_cast<int>(gguf_get_val_u32(gctx, i));
                        } else if (str_ends_with(key, ".attention.head_count_kv") && n_head_kv == 0) {
                            n_head_kv = static_cast<int>(gguf_get_val_u32(gctx, i));
                        } else if (str_ends_with(key, ".attention.key_length") && key_length == 0) {
                            key_length = static_cast<int>(gguf_get_val_u32(gctx, i));
                        } else if (str_ends_with(key, ".full_attention_interval")) {
                            full_attention_interval = static_cast<int>(gguf_get_val_u32(gctx, i));
                        } else if (str_ends_with(key, ".context_length")) {
                            train_ctx_len = gguf_get_val_u32(gctx, i);
                        } else if (str_ends_with(key, ".embedding_length") && embd == 0) {
                            embd = gguf_get_val_u32(gctx, i);
                        } else if (str_ends_with(key, ".attention.head_count") && n_head == 0) {
                            n_head = gguf_get_val_u32(gctx, i);
                        }
                    }

                    if (key_length == 0 && embd > 0 && n_head > 0) {
                        key_length = static_cast<int>(embd / n_head);
                    }
                    gguf_free(gctx);
                }
            }
        } catch (...) {}
    }

    // Fallbacks if metadata reading was not possible
    if (n_layer_all <= 0) {
        n_layer_all = std::clamp(static_cast<int>(model_bytes / (240ULL * 1024 * 1024)), 24, 80);
    }
    if (n_head_kv <= 0) n_head_kv = 8;
    if (key_length <= 0) key_length = 128;
    if (full_attention_interval <= 0) full_attention_interval = 1;

    int num_attn_layers = (full_attention_interval > 0) ? std::max(1, n_layer_all / full_attention_interval) : n_layer_all;
    size_t kv_elems_per_token = static_cast<size_t>(num_attn_layers) * 2ULL * n_head_kv * key_length;

    // 1. Generation defaults: wave diffusion steps
    if (!flags.diffusion_steps) {
        params.diffusion_steps = 3;
    }

    // 2. Usable VRAM Budget Calculation
    size_t avail_vram = (sys.free_vram_bytes > 0) ? sys.free_vram_bytes : sys.total_vram_bytes;
    size_t headroom = (1024ULL * 1024 * 1024); // 1 GB reserve for CUDA context and display compositor
    size_t usable_vram = (avail_vram > headroom) ? (avail_vram - headroom) : (avail_vram / 2);

    // 3. Context Size Tuning
    if (!flags.n_ctx) {
        if (train_ctx_len > 0 && train_ctx_len < 4096) {
            backend.n_ctx = train_ctx_len;
        } else if (usable_vram >= (16ULL * 1024 * 1024 * 1024)) {
            backend.n_ctx = 8192;
        } else if (usable_vram >= (8ULL * 1024 * 1024 * 1024)) {
            backend.n_ctx = 4096;
        } else {
            backend.n_ctx = 2048;
        }
    }

    int ctx = std::max(128, backend.n_ctx);
    size_t kv_bytes_f16  = static_cast<size_t>(ctx * kv_elems_per_token * 2.0);
    size_t kv_bytes_q8_0 = static_cast<size_t>(ctx * kv_elems_per_token * 1.0625);
    size_t kv_bytes_q4_0 = static_cast<size_t>(ctx * kv_elems_per_token * 0.5625);

    size_t scratch_bytes = (512ULL * 1024 * 1024) + static_cast<size_t>(backend.n_ubatch) * 5120ULL * sizeof(float) * 4ULL;

    // 4. KV Cache Quantization Cascade
    // If user did not explicitly set KV cache type, quantize automatically to fit requested context
    bool kv_auto_quantized = false;
    if (!flags.cache_type_k && !flags.cache_type_v) {
        if (model_bytes + kv_bytes_f16 + scratch_bytes <= usable_vram) {
            backend.cache_type_k = "f16";
            backend.cache_type_v = "f16";
        } else if (model_bytes + kv_bytes_q8_0 + scratch_bytes <= usable_vram) {
            backend.cache_type_k = "q8_0";
            backend.cache_type_v = "q8_0";
            backend.flash_attn = true;
            kv_auto_quantized = true;
        } else {
            backend.cache_type_k = "q4_0";
            backend.cache_type_v = "q4_0";
            backend.flash_attn = true;
            kv_auto_quantized = true;
        }
    }

    double bpe = 2.0;
    if (backend.cache_type_k == "q8_0" || backend.cache_type_v == "q8_0") bpe = 1.0625;
    else if (backend.cache_type_k == "q4_0" || backend.cache_type_v == "q4_0" ||
             backend.cache_type_k == "q4_1" || backend.cache_type_v == "q4_1" ||
             backend.cache_type_k == "iq4_nl" || backend.cache_type_v == "iq4_nl") bpe = 0.5625;
    size_t selected_kv_bytes = static_cast<size_t>(ctx * kv_elems_per_token * bpe);

    // 5. GPU Layer Allocation & Overflow Protection
    size_t total_vram_needed = model_bytes + selected_kv_bytes + scratch_bytes;
    bool layers_clamped = false;

    if (sys.gpu_count > 0 && sys.total_vram_bytes > 0) {
        if (total_vram_needed <= usable_vram) {
            if (!flags.n_gpu_layers) {
                backend.n_gpu_layers = 999; // 100% layer offload
            }
        } else {
            // Overflows total GPU VRAM: calculate safe layer budget and offload remainder to System RAM
            size_t vram_for_weights = 0;
            if (usable_vram > (selected_kv_bytes + scratch_bytes)) {
                vram_for_weights = usable_vram - (selected_kv_bytes + scratch_bytes);
            }
            size_t bytes_per_layer = (n_layer_all > 0) ? (model_bytes / n_layer_all) : (240ULL * 1024 * 1024);
            int safe_layers = static_cast<int>(vram_for_weights / bytes_per_layer);
            safe_layers = std::clamp(safe_layers, 0, n_layer_all);

            if (!flags.n_gpu_layers) {
                backend.n_gpu_layers = safe_layers;
            } else if (backend.n_gpu_layers > safe_layers) {
                // User passed explicit n_gpu_layers that exceeds physical hardware capacity
                std::cerr << "[Auto-Tuning] WARNING: Explicit n_gpu_layers=" << backend.n_gpu_layers
                          << " with n_ctx=" << backend.n_ctx << " exceeds physical GPU VRAM. Auto-clamping to "
                          << safe_layers << " layers to prevent CUDA Out-Of-Memory crash.\n";
                backend.n_gpu_layers = safe_layers;
                layers_clamped = true;
            }

            // Mitigate activation memory spikes for extreme contexts
            if (!flags.n_ubatch && backend.n_ctx >= 32768) {
                backend.n_ubatch = std::min(backend.n_ubatch, 256);
            }
            if (!flags.n_batch && backend.n_ctx >= 32768) {
                backend.n_batch = std::min(backend.n_batch, 512);
            }
        }
    } else {
        backend.n_gpu_layers = 0;
    }

    // 6. Flash Attention
    if (!flags.flash_attn) {
        if (sys.gpu_count > 0 && backend.n_gpu_layers > 0) {
            backend.flash_attn = true;
        } else {
            backend.flash_attn = false;
        }
    }

    // 7. Multi-GPU Split Mode & Proportion Balancing
    if (!flags.split_mode) {
        if (sys.gpu_count > 1 && backend.n_gpu_layers > 0) {
            backend.split_mode = 1; // layer split across GPUs
        } else {
            backend.split_mode = 0;
        }
    }

    if (!flags.tensor_split && sys.gpu_count == 2 && backend.split_mode == 1) {
        size_t g0 = (sys.gpu_free_vram_bytes.size() > 0 && sys.gpu_free_vram_bytes[0] > 0)
            ? sys.gpu_free_vram_bytes[0] : 1;
        size_t g1 = (sys.gpu_free_vram_bytes.size() > 1 && sys.gpu_free_vram_bytes[1] > 0)
            ? sys.gpu_free_vram_bytes[1] : 1;
        double ratio0 = static_cast<double>(g0) / (g0 + g1) * 100.0;
        double ratio1 = 100.0 - ratio0;
        int r0 = static_cast<int>(std::round(ratio0));
        int r1 = static_cast<int>(std::round(ratio1));
        backend.tensor_split_str = std::to_string(r0) + "," + std::to_string(r1);
    }

    // 8. Threads
    if (!flags.n_threads) {
        if (backend.n_gpu_layers >= 999) {
            backend.n_threads = 4;
        } else {
            backend.n_threads = std::max(1, sys.cpu_threads / 2);
        }
    }
    if (!flags.n_threads_batch) {
        backend.n_threads_batch = backend.n_threads;
    }

    if (verbose) {
        std::cerr << "\n================================================================================\n"
                  << " [Auto-Tuning] Native Diffusion Engine Boot Diagnostics\n"
                  << "================================================================================\n"
                  << " OS: " << sys.os_name << " | CPU Threads: " << sys.cpu_threads
                  << " | System RAM: " << (sys.total_ram_bytes / (1024 * 1024 * 1024)) << " GB\n";
        if (sys.gpu_count > 0) {
            std::cerr << " Detected GPU(s): " << sys.gpu_count << "x " << sys.primary_gpu_name
                      << " (Total VRAM: " << (sys.total_vram_bytes / (1024 * 1024)) << " MB"
                      << ", Free: " << (sys.free_vram_bytes / (1024 * 1024)) << " MB)\n";
        } else {
            std::cerr << " GPU: None detected -> CPU execution mode\n";
        }
        if (model_bytes > 0) {
            std::cerr << " Model Weights: " << (model_bytes / (1024 * 1024)) << " MB"
                      << " (" << n_layer_all << " layers, " << num_attn_layers << " attn layers)\n";
        }
        std::cerr << " Context & KV Cache:\n"
                  << "  - n_ctx:        " << backend.n_ctx << (flags.n_ctx ? " [user]" : " [auto]") << "\n"
                  << "  - cache_type_k: " << backend.cache_type_k << (flags.cache_type_k ? " [user]" : (kv_auto_quantized ? " [auto-quant]" : " [auto]")) << "\n"
                  << "  - cache_type_v: " << backend.cache_type_v << (flags.cache_type_v ? " [user]" : (kv_auto_quantized ? " [auto-quant]" : " [auto]")) << "\n"
                  << "  - est_kv_size:  " << (selected_kv_bytes / (1024 * 1024)) << " MB\n"
                  << " Configuration Applied:\n"
                  << "  - n_gpu_layers: " << backend.n_gpu_layers << (layers_clamped ? " [clamped]" : (flags.n_gpu_layers ? " [user]" : " [auto]")) << "\n"
                  << "  - split_mode:   " << (backend.split_mode == 1 ? "layer (multi-GPU)" : "none") << (flags.split_mode ? " [user]" : " [auto]");
        if (!backend.tensor_split_str.empty()) {
            std::cerr << " (split: " << backend.tensor_split_str << ")";
        }
        std::cerr << "\n"
                  << "  - flash_attn:   " << (backend.flash_attn ? "on" : "off") << (flags.flash_attn ? " [user]" : " [auto]") << "\n"
                  << "  - n_batch:      " << backend.n_batch << (flags.n_batch ? " [user]" : " [auto]") << "\n"
                  << "  - n_ubatch:     " << backend.n_ubatch << (flags.n_ubatch ? " [user]" : " [auto]") << "\n"
                  << "  - threads:      " << backend.n_threads << (flags.n_threads ? " [user]" : " [auto]") << "\n"
                  << "  - diff_steps:   " << params.diffusion_steps << (flags.diffusion_steps ? " [user]" : " [auto]") << "\n";
        if (backend.enable_mtp) {
            std::cerr << "  - enable_mtp:   on [user]\n";
        }
        std::cerr << "================================================================================\n\n";
        std::cerr.flush();
    }
}

} // namespace os
} // namespace diffusion
