# diffusion.gguf v0.1.0 (0.1)

Initial release of **diffusion.gguf**, a high-performance, block-parallel diffusion inference engine for standard causal autoregressive GGUF language models (Gemma, LLaMA, Qwen, DeepSeek, etc.) without retraining.

> [!NOTE]
> **Platform Support Status:**
> - **Windows x64 + NVIDIA CUDA:** Supported and tested with **standalone plug-and-play executables** (CUDA and all runtime modules statically embedded).
> - **Linux x64 + CUDA:** Coming soon (untested / not fully implemented yet).
> - **macOS Metal / Linux CPU:** Planned in subsequent releases.

---

## Highlights & Features

- **Standalone Plug-and-Play Executables:** Fully monolithic static binaries with all CUDA 12/13 kernels, Flash Attention, and GGML backends embedded directly inside the `.exe` files. Zero dependency on external runtime DLLs.
- **Block-Parallel Diffusion Trajectories:** Supports block sizes $B \in \{1, 4, 8, 16\}$ predicting contiguous token sequences with zero speculation, no drafting models, and no speculative tree structures.
- **Sparse Active Query Execution:** Uncommitted tokens dynamically contract across refinement passes ($N=8 \to N=4 \to N=1$), evaluating only active query positions while reusing prefix representations from the KV cache.
- **Multi-Bucket CUDA Graph Caching:** Allocates dedicated graph buckets per active batch slice, eliminating CUDA graph recreation overhead when switching query batch sizes.
- **Hardware-Aware Boot Auto-Tuning:** Dynamic GPU VRAM and topology inspection, auto-allocating optimal layer offloading, Flash Attention, and single/multi-GPU batch parameters.
- **OpenAI-Compatible HTTP Server:** Full streaming (`/v1/chat/completions`), SSE chunking, and tool call support with multi-turn context preservation.

---

## Formal Benchmarks

Evaluated against upstream `llama.cpp` using a standardized synthetic multi-domain kernel architecture analysis prompt:
- **Test System:** Dual NVIDIA GeForce RTX 3060 (24 GB VRAM total)
- **Model:** Gemma 4 E4B IT (7.52B parameters, Q4_K_M GGUF)
- **Parameters:** Context size `32,000` tokens, Temperature `0.0` (greedy), Output limit `2,048` tokens
- **Isolation:** Zero additional engine flags passed to either binary

| Engine | Block Size ($B$) | Completion Tokens | Wall Time (s) | Throughput (tok/s) | Speedup vs `llama.cpp` | Speedup vs $B=1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`llama.cpp`** (`llama_simple`) | 1 | 1833 | 40.95 s | **59.04 tok/s** | 1.00x | 0.82x |
| **`diffusion.gguf`** | 1 | 1834 | 33.47 s | **71.73 tok/s** | **1.22x** | 1.00x |
| **`diffusion.gguf`** | 4 | 1825 | 33.33 s | **71.81 tok/s** | **1.22x** | 1.00x |
| **`diffusion.gguf`** | 8 | 1771 | 32.23 s | **72.63 tok/s** | **1.23x** | 1.01x |
| **`diffusion.gguf`** | 16 | 1856 | 34.09 s | **70.92 tok/s** | **1.20x** | 0.99x |

---

## Limitations & Upcoming Work

- **Higher Block Size Scaling Ceiling:** Fixed-point token verification without speculation currently bounds multi-token advances to ~1.15-1.20 tokens/iter on non-repetitive prose. Advanced entropy gating and semantic confidence filters are planned to unlock the full scaling potential of $B=16$.
- **Platform Availability:** Pre-built binaries are strictly Windows x64 + CUDA. Linux CI automation and Apple Silicon Metal graphs will be added in upcoming updates.

---

## Release Assets (Plug-and-Play Windows x64 + CUDA)

Download and run directly without any extra DLLs or installer:
- `diffusion-cli.exe` (57.6 MB) — Standalone CLI runner with embedded CUDA & GGML runtime.
- `diffusion-server.exe` (58.0 MB) — Standalone OpenAI-compatible HTTP streaming server.

