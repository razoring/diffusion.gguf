#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/third_party/llama.cpp"

echo "Applying patches to third_party/llama.cpp..."
git apply --check "$ROOT_DIR/patches/llama_context_mrope_update.patch" && git apply "$ROOT_DIR/patches/llama_context_mrope_update.patch" || echo "llama_context_mrope_update.patch already applied or conflict."
git apply --check "$ROOT_DIR/patches/ggml_cuda_multibucket_graph.patch" && git apply "$ROOT_DIR/patches/ggml_cuda_multibucket_graph.patch" || echo "ggml_cuda_multibucket_graph.patch already applied or conflict."
echo "Patches applied successfully."
