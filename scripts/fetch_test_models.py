#!/usr/bin/env python3
"""
Downloads test models for diffusion.gguf test and parity suites.
"""
import os
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "stories15M-q4_0.gguf": "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"
}

def download_file(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size} bytes)")
        return
    print(f"[downloading] {dest.name} from {url}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"[done] {dest.name} saved ({dest.stat().st_size} bytes)")

def main():
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    for name, url in MODELS.items():
        download_file(url, models_dir / name)

if __name__ == "__main__":
    main()
