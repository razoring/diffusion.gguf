# diffusion.gguf

High-throughput, block-parallel diffusion inference engine for standard GGUF autoregressive language models.

---

## 1. Overview

`diffusion.gguf` transforms standard causal autoregressive language models (such as Gemma, LLaMA, Qwen) into parallel block-diffusion generators without retraining.

Instead of decoding strictly token-by-token ($B=1$), the engine predicts contiguous token blocks ($B=4$ to $B=8$) across iterative multi-pass diffusion refinement schedules.

### Key Performance Innovations:
- **True Sparse Active Query Execution:** When active tokens converge across diffusion passes ($N=8 \to N=4 \to N=1$), backbone GEMMs and attention kernels execute *only* over active positions while reusing frozen prefix tokens directly from the KV cache.
- **KV Anchor Locking:** Zero cache recomputation for stationary prefixes. `kv_truncate` rewinds uncommitted active tail slots before evaluating sparse wavefronts.
- **Multi-Bucket CUDA Graph Caching:** Eliminates CUDA graph recreation overhead when switching between variable active query batch sizes ($N \in \{1, 4, 8\}$).
- **Zero-Speculation Guarantee:** No draft models, no speculative tree verification, no autoregressive batch drafting.

**Measured Result:** On Gemma 4 E4B Q4_K_M (dual RTX 3060), 3-pass block diffusion drops generation latency from $153.0\text{ ms} \to 109.1\text{ ms}$, achieving **73.33 tok/s** (+19.5% faster than baseline autoregressive llama.cpp at 61.35 tok/s).

---

## 2. Architecture & Directory Layout

```
diffusion.gguf/
├── CMakeLists.txt        # Top-level build configuration
├── include/              # Public engine headers
├── src/
│   ├── engine.h / .cpp   # Top-level generation coordinator
│   ├── scheduler_diffusion.h / .cpp # Multi-pass block scheduler
│   ├── scheduler_jacobi.h / .cpp    # Jacobi parallel scheduler
│   ├── context_trie.h / .cpp        # Context n-gram lookup
│   ├── init_guess.h / .cpp          # Block initialization heuristics
│   ├── fast_math.h       # Fast log/entropy approximations
│   ├── kernels/          # CUDA / CPU reduction & diff kernels
│   └── llama/            # Clean backend wrapper for llama.cpp runtime
├── tools/
│   ├── cli/              # diffusion-cli interactive & batch generator
│   └── server/           # diffusion-server HTTP endpoint
├── patches/              # Differential patches for third_party dependencies
│   ├── ggml_cuda_multibucket_graph.patch # Multi-bucket graph key
│   └── llama_context_mrope_update.patch  # Context memory update
├── scripts/
│   ├── apply_patches.ps1 # Windows patch applicator
│   └── apply_patches.sh  # Unix patch applicator
├── tests/
│   └── unit/             # Profilers & regression tests
└── third_party/
    └── llama.cpp         # Clean upstream dependency (submodule)
```

---

## 3. Getting Started

### Prerequisites
- CMake 3.18+
- C++17 compliant compiler (MSVC 2022 / Clang / GCC)
- CUDA Toolkit 12.x+ (for CUDA acceleration)
- Git

### Cloning & Applying Patches
External dependencies in `third_party/` remain pristine in source control. Build-time patches must be applied after checkout:

```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/your-org/diffusion.gguf.git
cd diffusion.gguf

# Apply patches to third_party dependencies:
# On Windows (PowerShell):
powershell -ExecutionPolicy Bypass -File scripts/apply_patches.ps1

# On Linux / macOS (Bash):
bash scripts/apply_patches.sh
```

### Building

Using CMake and Ninja:
```bash
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## 4. Usage

### Interactive CLI (`diffusion-cli`)

```bash
./build/bin/diffusion-cli \
    -m ./models/gemma-4-e4b-it-Q4_K_M.gguf \
    -p "Explain quantum computing in three sentences:" \
    -ngl 99 \
    --block-size 8 \
    --max-passes 3
```

Key arguments:
- `-m, --model <path>`: Path to GGUF model file.
- `-p, --prompt <text>`: Prompt string.
- `-ngl <n>`: Number of layers to offload to GPU.
- `--block-size <n>`: Diffusion block width (recommended: 4 to 8).
- `--max-passes <n>`: Maximum refinement passes per block (default: 3).

### Diffusion Server (`diffusion-server`)

OpenAI-compatible HTTP chat completion endpoint supporting streaming, tool-calling, and block diffusion:

```bash
./build/bin/diffusion-server \
    -m ./models/gemma-4-e4b-it-Q4_K_M.gguf \
    -ngl 99 \
    --host 127.0.0.1 \
    --port 8080 \
    --block-size 8 \
    -c 2048
```

#### Complete Server CLI Arguments Reference:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Model & Hardware** | | | |
| `-m, --model PATH` | `string` | *(Required)* | Path to GGUF model file |
| `-c, --ctx-size N` | `int` | `2048` | KV context size in tokens |
| `-ngl, --n-gpu-layers N` | `int` | `0` | Number of model layers offloaded to GPU |
| `-t, --threads N` | `int` | `4` | Number of CPU compute threads |
| `-b, --batch-size N` | `int` | `2048` | Logical batch size |
| `-ub, --ubatch-size N` | `int` | `512` | Micro-batch size |
| `-fa, --flash-attn [on\|off\|auto]` | `string/flag` | `auto` | Enable Flash Attention |
| `--no-flash-attn` | `flag` | `false` | Explicitly disable Flash Attention |
| `--kv-unified` / `--no-kv-unified`| `flag` | `true` | Enable/disable unified KV buffer |
| `--no-mmap` | `flag` | `false` | Disable memory mapping for model weights |
| `--mlock` | `flag` | `false` | Lock model memory in RAM |
| `--numa MODE` | `string` | `""` | NUMA memory node optimization mode |
| `--cache-type-k TYPE` | `string` | `f16` | KV cache data type for K (`f16`, `q8_0`, `q4_0`) |
| `--cache-type-v TYPE` | `string` | `f16` | KV cache data type for V (`f16`, `q8_0`, `q4_0`) |
| `--lora PATH` | `string` | `""` | Path to LoRA adapter to apply |
| `-sm, --split-mode MODE` | `string` | `layer` | Multi-GPU split mode (`none`, `layer`, `row`, `tensor`) |
| `-ts, --tensor-split SPLIT` | `string` | `""` | Comma-separated GPU VRAM split ratios |
| `--main-gpu INDEX` | `int` | `0` | Device index of primary GPU |
| **Networking & Concurrency** | | | |
| `--host HOST` | `string` | `127.0.0.1` | Network interface address to listen on |
| `--port PORT` | `int` | `8080` | TCP port to bind server |
| `-np, --parallel N` | `int` | `1` | Number of concurrent slots / parallel sequences |
| `--api-key KEY` | `string` | `""` | Optional Bearer API key authentication |
| **Diffusion & Generation Defaults** | | | |
| `--mode MODE` | `string` | `block` | Decoding mode: `block` or `whole` (`ar` is removed) |
| `-n, --n-predict N` | `int` | `2048` | Default max generation tokens per request |
| `--block, --block-size, -b N` | `int` | `32` | Jacobi / diffusion block size |
| `--diffusion-steps, --diff-steps N` | `int` | `3` | Refinement passes per block |
| `--whole-max N` | `int` | `512` | Maximum canvas size in `whole` mode |
| `--init STRATEGY` | `string` | `adaptive` | Initial guess strategy (`ngram`, `random`, `pad`, `adaptive`) |
| `--annealed-attn` / `--no-annealed-attn` | `flag` | `false` | Enable/disable progressive attention annealing |
| `--speculative, --speculative-mode` | `flag` | `false` | Toggle speculative block evaluation |
| `--temp FLOAT` | `float` | `0.8` | Default sampling temperature |
| `--top-p FLOAT` | `float` | `0.95` | Default Top-p nucleus sampling |
| `--top-k N` | `int` | `40` | Default Top-k sampling filter |
| `-v, --version` | `flag` | - | Display version and exit |
| `-h, --help` | `flag` | - | Show help message and exit |

---

## 5. Benchmarks & Testing

Run unit tests and sparse graph transition verification:
```bash
# Verify sparse wavefront execution & numerical stability
./build/bin/test_sparse_graph_transitions

# Measure subsystem latency breakdown across block sizes
./build/bin/test_latency_breakdown
```
