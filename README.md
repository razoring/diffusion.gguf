# diffusion.gguf

A C++17 runtime that decodes standard autoregressive GGUF models using **Jacobi parallel iterative refinement** ("diffusion on autoregressive weights") instead of strict sequential next-token decoding.

Output tokens "denoise" and shimmer in place across a parallel canvas, converging to the **exact sequence** the autoregressive model would produce.

```
       +-------------------------------------------------------+
       |   Hypothesized Prefix Canvas c[0 .. B-1]              |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |   Parallel Forward Pass (eval B tokens in 1 ubatch)   |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |   Prefix Verification: find longest k where c[k]==p[k]|
       +-------------------------------------------------------+
                |                                      |
                v (Committed prefix)                   v (Refinement tail)
     Permanent Output Scrollback             Canvas Shimmers & Iterates
```

---

## Features

- **Direct AR Weights Compatibility**: Runs unmodified GGUF models (Llama 2/3, Mistral, Qwen 2/2.5/3, Gemma 2/3, Phi).
- **Exact Convergence Guarantee**: Mathematically guaranteed to converge to the exact autoregressive greedy output (and exact stochastic output under positional RNG).
- **Live Terminal Shimmering**: ANSI in-place re-rendering with unconverged token dimming and diff highlight frames.
- **Interactive Chat REPL**: Multi-turn conversation with prompt prefix KV cache reuse and `/mode`, `/block`, `/init` commands.
- **OpenAI-Compatible HTTP Server**: `/v1/chat/completions` and `/v1/completions` with Server-Sent Events (SSE) streaming of stable prefixes and stop-string holdback.
- **Offline GGJT Converter**: `diffusion-convert` upgrades legacy GGJT v1, v2, and v3 models to modern GGUF with bit-exact tensor repacking.

---

## Quickstart

### Building

Requirements: CMake 3.18+, C++17 compiler (MSVC 2022, GCC 11+, or Clang 14+).

```bash
git clone --recursive https://github.com/razoring/diffusion.gguf.git
cd diffusion.gguf

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

For GPU acceleration (CUDA, Vulkan, Metal), pass standard GGML flags:
```bash
# NVIDIA CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# Vulkan
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
```

---

## Tools & Usage

### 1. `diffusion-cli`

Run inference with live terminal rendering:

```bash
# Live in-place shimmering terminal mode (default on TTY)
./build/bin/Release/diffusion-cli -m models/stories15M-q4_0.gguf -p "Once upon a time" --mode block --block 8

# Whole-answer canvas mode (entire response iterates in place)
./build/bin/Release/diffusion-cli -m models/stories15M-q4_0.gguf -p "Explain gravity in one sentence" --mode whole

# Clean progress bar to stderr (stdout kept pristine for piping)
./build/bin/Release/diffusion-cli -m models/stories15M-q4_0.gguf -p "Hello world" --render progress > output.txt

# Interactive multi-turn chat REPL
./build/bin/Release/diffusion-cli -m models/stories15M-q4_0.gguf -cnv
```

#### REPL Commands
- `/exit`, `/quit`: Exit the session
- `/reset`, `/clear`: Reset conversation history and clear KV cache
- `/mode <block|whole|ar>`: Switch decoding mode
- `/block <N>`: Adjust Jacobi block size
- `/init <ngram|random|pad>`: Switch canvas initialization strategy
- `/temp <FLOAT>`: Set sampling temperature
- `/stats`: Toggle JSON performance stats display

---

### 2. `diffusion-server`

Run an OpenAI-compatible HTTP inference server:

```bash
./build/bin/Release/diffusion-server -m models/stories15M-q4_0.gguf --port 8080 --mode block --block 8
```

#### Endpoints
- `GET /health` -> `{"status":"ok"}`
- `GET /v1/models` -> Model list
- `POST /v1/completions` -> Text completion (supports both `stream: false` and `stream: true` SSE)
- `POST /v1/chat/completions` -> Chat completion with Jinja chat template formatting

#### Diffusion Extension Payload
Clients can dynamically control Jacobi execution via the optional `diffusion` JSON object:
```json
{
  "model": "diffusion-model",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "diffusion": {
    "mode": "block",
    "block_size": 16,
    "init": "ngram"
  },
  "stream": true
}
```

---

### 3. `diffusion-convert`

Convert legacy GGJT (v1, v2, v3) models to modern GGUF:

```bash
./build/bin/Release/diffusion-convert legacy_llama_7b.ggjt converted_llama_7b.gguf --ctx 2048 --eps 1e-6
```

Supported conversions:
- **GGJT v3**: Direct bit-exact GGUF emission.
- **GGJT v2**: Repacks float32 quantization scales in Q4_0, Q4_1, and Q8_0 to modern `ggml_fp16_t`.
- **GGJT v1**: Unshuffles legacy nibble permutation and repacks quantization scales to modern GGML format.

---

## Benchmark & Performance

Tested on CPU (AMD64, 4 threads, `stories15M-q4_0.gguf`, `n_predict=64`):

| Mode | Block Size | Init Strategy | Iterations | Tok / Iter | Tok / s | Rel Speed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AR (Autoregressive Baseline)** | 1 | n/a | 64 | 1.00 | 919.8 | 1.00x |
| **Jacobi Block** | 4 | ngram | 61 | 1.05 | 602.3 | 0.65x |
| **Jacobi Block** | 8 | ngram | 62 | 1.03 | 362.1 | 0.39x |
| **Jacobi Block** | 16 | ngram | 62 | 1.03 | 211.4 | 0.23x |
| **Jacobi Block** | 32 | ngram | 62 | 1.03 | 145.4 | 0.16x |
| **Jacobi Whole** | 64 | ngram | 62 | 1.03 | 105.0 | 0.11x |

*Note: On CPU-bound execution, evaluating $B$ tokens concurrently in a single forward pass incurs higher FLOPs per step than single-token AR. On parallel accelerators (CUDA/Vulkan) where memory bandwidth dominates and batch arithmetic intensity is high, Block Jacobi achieves parallel wall-clock acceleration.*

---

## Testing & Parity Verification

```bash
# 1. Run CTest unit tests (UTF-8, Jacobi invariants I1-I4 mock)
ctest --test-dir build -C Release --output-on-failure

# 2. Run parity test suite against upstream llama.cpp oracle
python tests/parity/run_parity.py

# 3. Run server integration tests (SSE streaming, Auth, chat)
python tests/server/test_server.py

# 4. Run GGJT v1-v3 conversion tests
python tests/convert/test_convert.py
```

---

## License

MIT
