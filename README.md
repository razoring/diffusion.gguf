# diffusion.gguf

High-throughput, block-parallel diffusion inference engine for standard GGUF causal language models.

---

## 1. Overview

`diffusion.gguf` transforms standard causal autoregressive language models (such as Gemma, LLaMA, Qwen, DeepSeek) into parallel block-diffusion generators without retraining or fine-tuning.

Instead of decoding strictly token-by-token ($B=1$), the engine predicts contiguous token blocks ($B=4, 8, 16$) across iterative multi-pass diffusion refinement schedules with exact KV-cache synchronization.

### Key Innovations:
- **Diffusion-Native Block Trajectories:** Block-parallel multi-token generation with strict KV synchronization, avoiding KV desynchronization and token collapse.
- **Sparse Active Query Execution:** When active tokens converge across diffusion passes ($N=8 \to N=4 \to N=1$), backbone GEMMs and attention kernels evaluate only active positions while reusing frozen prefix tokens directly from the KV cache.
- **Dynamic Multi-Bucket CUDA Graph Caching:** Dedicated graph bucket allocation per batch slice, eliminating CUDA graph recreation and reallocation overhead during variable active query passes.
- **Hardware-Aware Boot Auto-Tuning:** Dynamic VRAM and device topology detection, auto-configuring optimal layer offloading, Flash Attention, and single/multi-GPU batch parameters.
- **Zero-Speculation Guarantee:** Pure diffusion trajectory execution without external draft models, speculative tree verification, or autoregressive drafting.

---

## 2. Benchmark Results

Measured against upstream `llama.cpp` using the standardized synthetic multi-domain kernel analysis prompt under identical hardware and generation parameters:
- **Hardware:** Dual NVIDIA GeForce RTX 3060 (24 GB VRAM)
- **Model:** Gemma 4 E4B IT (7.52B parameters, Q4_K_M GGUF)
- **Parameters:** Context size `32,000` tokens, Temperature `0.0` (greedy), Output limit `2,048` tokens
- **Isolation:** Zero additional engine flags passed to either binary

| Engine | Block Size ($B$) | Output Tokens | Wall Time (s) | Throughput (tok/s) | Speedup vs `llama.cpp` | Speedup vs $B=1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`llama.cpp`** (`llama_simple`) | 1 | 1833 | 40.95 s | **59.04 tok/s** | 1.00x | 0.82x |
| **`diffusion.gguf`** | 1 | 1834 | 33.47 s | **71.73 tok/s** | **1.22x** | 1.00x |
| **`diffusion.gguf`** | 4 | 1825 | 33.33 s | **71.81 tok/s** | **1.22x** | 1.00x |
| **`diffusion.gguf`** | 8 | 1771 | 32.23 s | **72.63 tok/s** | **1.23x** | 1.01x |
| **`diffusion.gguf`** | 16 | 1856 | 34.09 s | **70.92 tok/s** | **1.20x** | 0.99x |

*Reproduce via:*
```bash
python tests/benchmark_block_scaling.py -n 2048 -c 32000 -temp 0.0
```

---

## 3. Platform Support & Status

| Platform / Backend | Support Status | Notes |
| :--- | :---: | :--- |
| **Windows x64 + NVIDIA CUDA** | **Full (v0.1.0)** | Tested and verified on Windows 11 with MSVC v17.14 & CUDA 12.x/13.x |
| **Linux x64 + NVIDIA CUDA** | *Coming Soon* | In progress; CI pipeline under configuration |
| **Linux / macOS CPU (ggml-cpu)** | *Untested* | Functional code path, but performance unoptimized |
| **macOS Metal (ggml-metal)** | *Planned* | Awaiting Metal graph capture port |
| **ROCm / AMD GPU** | *Planned* | Backend hooks present via GGML HIP runtime |

---

## 4. Architecture & Directory Layout

```
diffusion.gguf/
├── CMakeLists.txt              # Top-level build configuration
├── include/                    # Public engine headers
├── src/
│   ├── engine.h / .cpp         # High-level generation coordinator
│   ├── scheduler_diffusion.h   # Multi-pass block scheduler
│   ├── context_trie.h / .cpp   # Context prefix lookup
│   ├── init_guess.h / .cpp     # Block initialization heuristics
│   ├── fast_math.h             # Fast log/entropy approximations
│   ├── kernels/                # CUDA / CPU reduction & diff kernels
│   └── llama/                  # llama.cpp C++ backend integration
├── tools/
│   ├── cli/                    # diffusion-cli interactive & batch generator
│   └── server/                 # diffusion-server OpenAI-compatible HTTP API
├── patches/                    # Patches for third_party submodules
├── scripts/                    # Platform scripts (patching, build helpers)
├── tests/
│   ├── benchmark_block_scaling.py # Automated formal benchmarking
│   ├── llama_simple.cpp        # Pure upstream llama.cpp baseline runner
│   └── unit/                   # Unit profilers and regression tests
└── third_party/
    └── llama.cpp               # Upstream GGML inference framework
```

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

## 7. Current Limitations & What Needs to be Implemented

- **Scaling Ceiling at Higher Block Sizes ($B \ge 8$):** Fixed-point matching without speculation bounds commitment per pass to 1.15-1.20 tokens/iter on non-repetitive text. Multi-token jump confidence filters and latent mask roll-overs need expansion to realize the full 2x-3x theoretical speedup for $B=16$.
- **Operating Systems & Backends:** Pre-built binary support currently restricted to Windows x64 + CUDA. Linux CUDA containerization and Metal support are underway.
- **Logit Head Parallelism:** LM-head evaluation across entire block dimensions currently computes full projection GEMMs before masking. Sparse head evaluation will further reduce Stage 2 latency.

---

## 8. Contributing

Contributions are welcome. Areas where help is immediately needed:
1. Linux CI workflows with automated CUDA 12.x builds.
2. Apple Silicon Metal (`ggml-metal`) port for multi-bucket graph caching.
3. Higher block size confidence models and entropy filtering heuristics for $B \ge 8$.

Please review existing issues or open an RFC pull request before introducing major architectural modifications.
