# Formal Benchmark Report: diffusion.gguf vs. llama.cpp

- **Model**: `C:\Users\conta\.ollama\models\blobs\sha256-a9d0a3e1d8c91732a3812becd31f8d4a8ff57414903f226306069f0437e2ac50`
- **Context**: `8192` tokens
- **Temperature**: `0.0`
- **Max Output**: `256` tokens

| Engine | Block Size | Tokens Generated | Time (s) | Speed (tok/s) | Speedup vs llama.cpp | Speedup vs B=1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| llama.cpp | 1 | 256 | 9.62 | 71.11 | 1.00x | 1.00x |
| diffusion.gguf | whole | 256 | 7.70 | 76.99 | 1.08x | - |
