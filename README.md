# diffusion.gguf
## 1. Overview

`diffusion.gguf` runs existing causal language models (such as Gemma, Llama, Qwen, and DeepSeek) using an iterative block-based decoding approach instead of generating one token at a time.

Rather than producing output strictly token-by-token ($B=1$), the engine generates blocks of tokens ($B=4$, $8$, $16$) and refines them over multiple passes while keeping the KV cache consistent with the model state.

### Features

- **Block-based generation:** Generates and refines groups of tokens in parallel instead of decoding one token at a time.
- **Selective computation:** As tokens stabilize during refinement, computation is focused on positions that still need updates.
- **CUDA graph reuse:** Caches CUDA graphs for common execution patterns to reduce launch and allocation overhead.
- **Automatic startup configuration:** Detects available hardware and configures GPU offloading, Flash Attention, and batch settings automatically.
- **No draft model required:** Works directly with the target model without speculative decoding or auxiliary models.

---

## 2. Benchmark Results
- **Hardware:** 2x RTX 3060 12gb
- **Model:** Gemma 4 E4B IT Q4_K_M
- **Parameters:** ctx `32,000`, temp `0.0`, max_output `2,048`

| Engine | Block Size ($B$) | Output Tokens | Wall Time (s) | Throughput (tok/s) | Speedup vs `llama.cpp` | Speedup vs $B=1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`llama.cpp`** (`v0.8.0`) | 1 | 1833 | 40.95 s | **59.04 tok/s** | 1.00x | 0.82x |
| **`diffusion.gguf`** (`v0.1.0`) | 1 | 1834 | 33.47 s | **71.73 tok/s** | **1.22x** | 1.00x |
| **`diffusion.gguf`** (`v0.1.0`) | 4 | 1825 | 33.33 s | **71.81 tok/s** | **1.22x** | 1.00x |
| **`diffusion.gguf`** (`v0.1.0`) | 8 | 1771 | 32.23 s | **72.63 tok/s** | **1.23x** | 1.01x |
| **`diffusion.gguf`** (`v0.1.0`) | 16 | 1856 | 34.09 s | **70.92 tok/s** | **1.20x** | 0.99x |

*Reproduce via:*
```bash
python tests/benchmark_block_scaling.py -n 2048 -c 32000 -temp 0.0
```

---

## 3. Platform Support & Status
| Platform / Backend | Support Status | Notes |
| :--- | :---: | :--- |
| **Windows x64 + NVIDIA CUDA** | **v0.1.0** | Tested and verified on Windows 11 with MSVC v17.14 & CUDA 12.x/13.x |
| **Windows x64 + CPU** | *Untested* | Supported, but not yet validated on a clean CPU-only configuration |
| **Windows x64 + ROCm/HIP** | *Untested* | Supported via GGML HIP backend; functional but not yet validated |
| **Windows x64 + oneAPI/SYCL** | *In Progress* | Supported via GGML SYCL backend; functional but not yet validated |
| **Linux x64 + NVIDIA CUDA** | *In Progress* | Backend support under active development; not fully implemented |
| **Linux x64 + ROCm/HIP** | *In Progress* | Backend support under active development; not fully implemented |
| **Linux x64 + oneAPI/SYCL** | *In Progress* | Backend support under active development; not fully implemented |
| **Linux x64 + CPU** | *In Progress* | Backend support under active development; not fully implemented |
| **macOS Metal (ggml-metal)** | *Planned* | Awaiting Metal graph capture port |

---

## 5. Getting Started (Windows + CUDA)

### Prerequisites
- Windows 10/11 x64
- NVIDIA GPU with Compute Capability $\ge 6.0$
- CUDA Toolkit 12.x or 13.x
- Visual Studio 2022 (MSVC v143+) with C++17 support
- CMake 3.18+ & Ninja

### Building from Source

```powershell
# Clone repository with submodules
git clone --recurse-submodules https://github.com/razoring/diffusion.gguf.git
cd diffusion.gguf

# Apply patches to third_party llama.cpp
powershell -ExecutionPolicy Bypass -File scripts/apply_patches.ps1

# Configure and build with Ninja + CUDA
cmake -B build-ninja-cuda -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build-ninja-cuda -j 16
```

---

## 6. Usage

### Interactive & Batch CLI (`diffusion-cli`)

```powershell
.\build-ninja-cuda\bin\diffusion-cli.exe `
    -m "C:\path\to\model.gguf" `
    -p "Write an architectural analysis of modern OS kernels:" `
    -c 32000 `
    -temp 0.0 `
    -b 8 `
    -n 2048
```

Key arguments:
- `-m, --model <path>`: Path to GGUF model file.
- `-p, --prompt <text>`: Prompt string.
- `-c, --ctx-size <n>`: Context window size (e.g. 32000).
- `-temp <float>`: Sampling temperature (0.0 for greedy).
- `-b, --block-size <n>`: Block size width (1 for baseline autoregressive, 4, 8, 16 for block diffusion).
- `-n, --n-predict <n>`: Maximum generated tokens.

### OpenAI-Compatible HTTP Server (`diffusion-server`)

Provides an OpenAI-compatible endpoint (`/v1/chat/completions`) with Server-Sent Events (SSE) streaming and tool call support:

```powershell
.\build-ninja-cuda\bin\diffusion-server.exe `
    -m "C:\path\to\model.gguf" `
    -c 32000 `
    --port 4096 `
    -b 8
```

---

## 7. Current Limitations

- **Scaling Ceiling at Higher Block Sizes ($B \ge 8$):** Fixed-point matching without speculation bounds commitment per pass to 1.15-1.20 tokens/iter on non-repetitive text. Multi-token jump confidence filters and latent mask roll-overs need expansion to realize the full 2x-3x theoretical speedup for $B=16$.
- **Logit Head Parallelism:** LM-head evaluation across entire block dimensions currently computes full projection GEMMs before masking. Sparse head evaluation will further reduce Stage 2 latency.

---

## 8. Contributing

Contributions are welcome. Areas where help is immediately needed:
2. Windows with ROCm builds.
1. Linux with CUDA/ROCm builds.
3. Apple Silicon Metal port.
4. Higher block size confidence models and entropy filtering heuristics for $B \ge 8$.

Please review existing issues or open a pull request.
