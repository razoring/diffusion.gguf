#include "system_tuner.h"
#include <algorithm>
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

void auto_tune_system(
    BackendConfig& backend,
    GenParams& params,
    const ExplicitFlags& flags,
    bool verbose
) {
    SystemProfile sys = detect_system_profile();

    size_t model_bytes = 0;
    if (!backend.model_path.empty()) {
        try {
            if (std::filesystem::exists(backend.model_path)) {
                model_bytes = std::filesystem::file_size(backend.model_path);
            }
        } catch (...) {}
    }

    // 1. Diffusion defaults: Must be native block diffusion (B=8, M=3)
    if (!flags.block_size) {
        params.block_size = 8;
    }
    if (!flags.diffusion_steps) {
        params.diffusion_steps = 3;
    }

    // 2. GPU Offloading and VRAM budget
    if (!flags.n_gpu_layers) {
        if (sys.gpu_count > 0 && sys.total_vram_bytes > 0) {
            size_t effective_vram = sys.free_vram_bytes > 0 ? sys.free_vram_bytes : sys.total_vram_bytes;
            if (model_bytes == 0 || effective_vram >= static_cast<size_t>(model_bytes * 0.95)) {
                backend.n_gpu_layers = 999; // Offload 100% of layers
            } else {
                // Insufficient VRAM: calculate fractional layer offload
                double ratio = static_cast<double>(effective_vram) / static_cast<double>(model_bytes * 1.15);
                int est_layers = std::clamp(static_cast<int>(ratio * 40.0), 1, 999);
                backend.n_gpu_layers = est_layers;
            }
        } else {
            backend.n_gpu_layers = 0;
        }
    }

    // 3. Multi-GPU split mode
    if (!flags.split_mode) {
        if (sys.gpu_count > 1 && backend.n_gpu_layers > 0) {
            backend.split_mode = 1; // LLAMA_SPLIT_MODE_LAYER across all GPUs
        } else {
            backend.split_mode = 0; // Single GPU or none
        }
    }

    // 4. Flash Attention
    if (!flags.flash_attn) {
        if (sys.gpu_count > 0 && backend.n_gpu_layers > 0) {
            backend.flash_attn = true;
        } else {
            backend.flash_attn = false;
        }
    }

    // 5. Threads
    if (!flags.n_threads) {
        if (backend.n_gpu_layers >= 999) {
            backend.n_threads = 4; // GPU handles compute; lightweight host scheduler
        } else {
            backend.n_threads = std::max(1, sys.cpu_threads / 2);
        }
    }

    // 6. Context Size
    if (!flags.n_ctx) {
        if (sys.gpu_count > 0 && sys.total_vram_bytes >= (16ULL * 1024 * 1024 * 1024)) {
            backend.n_ctx = 8192; // High VRAM capacity
        } else if (sys.gpu_count > 0 && sys.total_vram_bytes >= (8ULL * 1024 * 1024 * 1024)) {
            backend.n_ctx = 4096;
        } else {
            backend.n_ctx = 2048;
        }
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
            std::cerr << " Model Weight Size: " << (model_bytes / (1024 * 1024)) << " MB\n";
        }
        std::cerr << " Configuration Applied:\n"
                  << "  - n_gpu_layers: " << backend.n_gpu_layers << (flags.n_gpu_layers ? " [user]" : " [auto]") << "\n"
                  << "  - split_mode:   " << (backend.split_mode == 1 ? "layer (multi-GPU)" : "none") << (flags.split_mode ? " [user]" : " [auto]") << "\n"
                  << "  - flash_attn:   " << (backend.flash_attn ? "on" : "off") << (flags.flash_attn ? " [user]" : " [auto]") << "\n"
                  << "  - n_ctx:        " << backend.n_ctx << (flags.n_ctx ? " [user]" : " [auto]") << "\n"
                  << "  - threads:      " << backend.n_threads << (flags.n_threads ? " [user]" : " [auto]") << "\n"
                  << "  - block_size:   " << params.block_size << (flags.block_size ? " [user]" : " [auto]") << "\n"
                  << "  - diff_steps:   " << params.diffusion_steps << (flags.diffusion_steps ? " [user]" : " [auto]") << "\n"
                  << "================================================================================\n\n";
        std::cerr.flush();
    }
}

} // namespace os
} // namespace diffusion
