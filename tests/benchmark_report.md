# Formal Benchmark Report: diffusion.gguf vs. llama.cpp

- **Model**: `C:\Users\conta\.ollama\models\blobs\sha256-370c2879f17648d337b0ab4e208fdce348e24d1060064e93d7645daafd56d39c`
- **Context**: `32000` tokens
- **Temperature**: `0.0`
- **Max Output**: `2048` tokens

| Engine | Block Size | Tokens Generated | Time (s) | Speed (tok/s) | Speedup vs llama.cpp | Speedup vs B=1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| llama.cpp | 1 | 1833 | 39.67 | 60.60 | 1.00x | 1.00x |
| diffusion.gguf | 1 | 1834 | 32.13 | 74.53 | 1.23x | 1.00x |
| diffusion.gguf | 4 | 1825 | 31.96 | 74.97 | 1.24x | 1.01x |
| diffusion.gguf | 8 | 1771 | 31.05 | 76.15 | 1.26x | 1.02x |
| diffusion.gguf | 16 | 1856 | 32.69 | 73.89 | 1.22x | 0.99x |
