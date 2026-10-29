# Formal Benchmark Report: diffusion.gguf vs. llama.cpp

- **Model**: `C:\Users\conta\.ollama\models\blobs\sha256-370c2879f17648d337b0ab4e208fdce348e24d1060064e93d7645daafd56d39c`
- **Context**: `32000` tokens
- **Temperature**: `0.0`
- **Max Output**: `2048` tokens

| Engine | Block Size | Tokens Generated | Time (s) | Speed (tok/s) | Speedup vs llama.cpp | Speedup vs B=1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| llama.cpp | 1 | 1833 | 40.95 | 59.04 | 1.00x | 1.00x |
| diffusion.gguf | 1 | 1834 | 33.47 | 71.73 | 1.22x | 1.00x |
| diffusion.gguf | 4 | 1825 | 33.33 | 71.81 | 1.22x | 1.00x |
| diffusion.gguf | 8 | 1771 | 32.23 | 72.63 | 1.23x | 1.01x |
| diffusion.gguf | 16 | 1856 | 34.09 | 70.92 | 1.20x | 0.99x |
