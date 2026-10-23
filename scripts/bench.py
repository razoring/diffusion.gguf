#!/usr/bin/env python3
"""
Benchmark runner for diffusion.gguf comparing AR, Block Jacobi, and Whole Jacobi decoding.
Measures iterations, tok/iter, tok/s, and wall time across init strategies and block sizes.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI_BIN = ROOT / "build" / "bin" / "Release" / "diffusion-cli.exe"
if not CLI_BIN.exists():
    CLI_BIN = ROOT / "build" / "bin" / "diffusion-cli"

MODEL_PATH = ROOT / "models" / "stories15M-q4_0.gguf"

PROMPT = "Once upon a time there was a curious little rabbit who lived in a green valley."
N_PREDICT = 64

def run_bench_case(mode, block_size=32, init_strat="ngram", threads=4):
    cmd = [
        str(CLI_BIN),
        "-m", str(MODEL_PATH),
        "-p", PROMPT,
        "-n", str(N_PREDICT),
        "-t", str(threads),
        "--temp", "0",
        "--mode", mode,
        "--block", str(block_size),
        "--init", init_strat,
        "--render", "final",
        "--stats"
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error running bench {mode}: {res.stderr}")
        return None

    # Parse JSON stats from stderr
    stderr = res.stderr
    json_start = stderr.find("{")
    json_end = stderr.rfind("}")
    if json_start != -1 and json_end != -1:
        stats_str = stderr[json_start : json_end + 1]
        try:
            return json.loads(stats_str)
        except Exception:
            pass
    return None

def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("=========================================================================================")
    print(f"  diffusion.gguf Decoding Benchmark ({MODEL_PATH.name}, n_predict={N_PREDICT})")
    print("=========================================================================================")

    # 1. Autoregressive baseline
    print("Running AR baseline...")
    ar_stats = run_bench_case("ar")
    if not ar_stats:
        print("Failed to run AR baseline")
        sys.exit(1)

    ar_tok_s = ar_stats["tokens_per_second"]
    ar_iters = ar_stats["iterations"]

    results = [
        ("AR (Autoregressive)", 1, "n/a", ar_iters, ar_stats["tokens_per_iteration"], ar_tok_s, 1.0)
    ]

    # 2. Block Jacobi with different block sizes
    for b in [4, 8, 16, 32]:
        print(f"Running Jacobi Block (B={b}, init=ngram)...")
        stats = run_bench_case("block", block_size=b, init_strat="ngram")
        if stats:
            speedup = stats["tokens_per_second"] / ar_tok_s
            results.append((f"Jacobi Block (B={b})", b, "ngram", stats["iterations"], stats["tokens_per_iteration"], stats["tokens_per_second"], speedup))

    # 3. Init strategies comparison (B=16)
    for strat in ["pad", "random"]:
        print(f"Running Jacobi Block (B=16, init={strat})...")
        stats = run_bench_case("block", block_size=16, init_strat=strat)
        if stats:
            speedup = stats["tokens_per_second"] / ar_tok_s
            results.append((f"Jacobi Block (B=16)", 16, strat, stats["iterations"], stats["tokens_per_iteration"], stats["tokens_per_second"], speedup))

    # 4. Whole mode
    print("Running Jacobi Whole Mode (init=ngram)...")
    stats = run_bench_case("whole", init_strat="ngram")
    if stats:
        speedup = stats["tokens_per_second"] / ar_tok_s
        results.append(("Jacobi Whole", 64, "ngram", stats["iterations"], stats["tokens_per_iteration"], stats["tokens_per_second"], speedup))

    # Print Table
    print("\n")
    print(f"{'Mode':<26} | {'Block':<5} | {'Init':<6} | {'Iters':<5} | {'Tok/Iter':<8} | {'Tok/s':<8} | {'Rel Speed':<9}")
    print("-" * 85)
    for row in results:
        name, b, init, iters, tok_it, tok_s, rel = row
        print(f"{name:<26} | {str(b):<5} | {init:<6} | {iters:<5} | {tok_it:<8.2f} | {tok_s:<8.1f} | {rel:<8.2f}x")
    print("=" * 85)

if __name__ == "__main__":
    main()
