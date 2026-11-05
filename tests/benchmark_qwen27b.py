#!/usr/bin/env python3
"""
Formal Benchmark Suite: Qwen3.8-27B GGUF
Compares official llama.cpp (llama-cli.exe / llama-bench.exe) vs diffusion.gguf under identical parameters:
- Model: Qwen3.8-27B (User-selected quant or auto-downloaded from Hugging Face)
- Context: 16384 (or 32768)
- Output limit: 4096 tokens
- Sampling: Greedy (temp=0.0)
- Speculative decoding / MTP: STRICTLY DISABLED (Zero speculative tokens, no draft models)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- Prompts ---
PROMPT_STAGE1_RAW_DECODE = "Write me a detailed history of Computer Science."


def find_binaries(repo_root: Path, user_llama_bin: str = "", user_diff_bin: str = ""):
    diff_bin = Path(user_diff_bin) if user_diff_bin else (repo_root / "build-ninja-cuda" / "bin" / "diffusion-cli.exe")
    if not diff_bin.exists():
        cand = repo_root / "build" / "bin" / "Release" / "diffusion-cli.exe"
        if cand.exists():
            diff_bin = cand

    if user_llama_bin:
        llama_bin = Path(user_llama_bin)
    else:
        # Check standard user install path
        candidates = [
            Path(os.path.expanduser("~")) / ".llama.cpp" / "llama-cli.exe",
            Path(os.path.expanduser("~")) / ".llama.cpp" / "llama-bench.exe",
        ]
        # Check PATH
        which_cli = shutil.which("llama-cli") or shutil.which("llama-cli.exe")
        if which_cli:
            candidates.insert(0, Path(which_cli))

        llama_bin = None
        for c in candidates:
            if c and c.exists():
                llama_bin = c
                break

        if not llama_bin:
            raise FileNotFoundError(
                "Official llama.cpp binary (llama-cli.exe) not found on system! "
                "Please specify path via --llama-bin."
            )

    return diff_bin, llama_bin


def ensure_model(model_path_str: str, repo_root: Path, quant: str = "Q4_K_M") -> Path:
    if model_path_str:
        p = Path(model_path_str)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified model not found: {model_path_str}")

    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    for f in models_dir.glob("Qwen3.8-27B*.gguf"):
        if f.is_file() and f.stat().st_size > 1024 * 1024 * 1024:
            print(f"[*] Found existing local model: {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")
            return f

    quant_map = {
        "Q4_K_M": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "IQ4_XS": "Qwen3.8-27B-UD-IQ4_XS.gguf",
        "IQ3_S":  "Qwen3.8-27B-UD-IQ3_S.gguf",
        "Q4_0":   "Qwen3.8-27B-Q4_0.gguf",
    }
    filename = quant_map.get(quant, f"Qwen3.8-27B-UD-{quant}.gguf")
    dest_path = models_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024 * 1024:
        return dest_path

    print(f"[*] Model not found locally. Initiating download for unsloth/Qwen3.8-27B-GGUF ({filename}) ...")
    hf_cmd = ["hf", "download", "unsloth/Qwen3.8-27B-GGUF", filename, "--local-dir", str(models_dir)]
    try:
        subprocess.run(hf_cmd, check=True)
        if dest_path.exists():
            return dest_path
    except Exception as e:
        print(f"Warning: hf CLI download failed ({e}). Trying curl/direct download fallback...")

    url = f"https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/{filename}"
    curl_cmd = ["curl", "-L", "-o", str(dest_path), url]
    subprocess.run(curl_cmd, check=True)
    return dest_path


def run_cmd_timed(cmd: list, env: dict = None) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, err = p.communicate()
    elapsed = time.perf_counter() - t0
    combined = out + "\n" + err
    return p.returncode, combined, elapsed


def parse_diffusion_output(output_str: str) -> dict:
    res = {"tokens": 0, "tok_per_s": 0.0, "time_ms": 0.0}
    json_match = re.search(r'\{\s*"tokens":\s*(\d+).*?"tokens_per_second":\s*([0-9\.]+).*?\}', output_str, re.DOTALL)
    if json_match:
        res["tokens"] = int(json_match.group(1))
        res["tok_per_s"] = float(json_match.group(2))
    return res


def parse_llama_cli_output(output_str: str) -> dict:
    res = {"tokens": 0, "tok_per_s": 0.0, "prompt_tok_per_s": 0.0}
    
    # Check official llama-cli perf summary line: [ Prompt: 119.9 t/s | Generation: 60.4 t/s ]
    match_perf = re.search(r'\[\s*Prompt:\s*([0-9\.]+)\s*t/s\s*\|\s*Generation:\s*([0-9\.]+)\s*t/s\s*\]', output_str)
    if match_perf:
        res["prompt_tok_per_s"] = float(match_perf.group(1))
        res["tok_per_s"] = float(match_perf.group(2))

    # Also check standard llama_print_timings: llama_perf_context_print: eval time = ... / X runs ( Y ms per token, Z tokens per second)
    match_timings = re.search(r'eval time\s*=\s*[0-9\.]+\s*ms\s*/\s*(\d+)\s*runs.*?\(\s*([0-9\.]+)\s*tokens per second\)', output_str)
    if match_timings:
        res["tokens"] = int(match_timings.group(1))
        if res["tok_per_s"] == 0.0:
            res["tok_per_s"] = float(match_timings.group(2))

    return res


def run_benchmark_stage(
    stage_name: str,
    prompt: str,
    model_path: Path,
    diff_bin: Path,
    llama_bin: Path,
    ctx_size: int,
    predict_max: int,
    env: dict,
    cached_llama_res: dict = None
) -> dict:
    print(f"\n{'='*80}")
    print(f" [STAGE: {stage_name}] Prompt Length: {len(prompt)} chars | Max Tokens: {predict_max}")
    print(f"{'='*80}")

    stage_results = {}

    # 1. Real Official llama.cpp (llama-cli.exe)
    if cached_llama_res and cached_llama_res.get("tok_per_s", 0.0) > 0:
        print(f"\n[*] Reusing existing official llama.cpp baseline from cache ({cached_llama_res['tok_per_s']:.2f} tok/s) ...")
        stage_results["llama.cpp"] = cached_llama_res
    else:
        print(f"\n[*] Running official llama.cpp ({llama_bin.name}) baseline ...")
        cmd_llama = [
            str(llama_bin),
            "-m", str(model_path),
            "-ngl", "all",
            "-c", str(ctx_size),
            "--temp", "0.0",
            "-n", str(predict_max),
            "-p", prompt,
            "--single-turn",
            "--no-warmup",
            "--simple-io",
            "--perf"
        ]
        code, out, elapsed = run_cmd_timed(cmd_llama, env=env)
        if code != 0:
            print(f"[-] llama.cpp execution failed (code {code}):\n{out[-400:]}")
            stage_results["llama.cpp"] = {"error": f"Exit code {code}", "tokens": 0, "tok_per_s": 0.0, "time_s": elapsed}
        else:
            parsed = parse_llama_cli_output(out)
            tok_s = parsed["tok_per_s"] if parsed["tok_per_s"] > 0 else (predict_max / elapsed if elapsed > 0 else 0.0)
            toks = parsed["tokens"] if parsed["tokens"] > 0 else predict_max
            print(f"[+] official llama.cpp completed: ~{toks} tokens in {elapsed:.2f}s ({tok_s:.2f} tok/s)")
            stage_results["llama.cpp"] = {
                "tokens": toks,
                "tok_per_s": tok_s,
                "time_s": elapsed,
                "raw_output": out[-500:]
            }

    # 2. diffusion.gguf (Whole-Canvas Soft-Locking, zero speculative decoding/MTP)
    print(f"\n[*] Running diffusion.gguf (whole canvas) ...")
    cmd_diff = [
        str(diff_bin),
        "-m", str(model_path),
        "-c", str(ctx_size),
        "-temp", "0.0",
        "-n", str(predict_max),
        "-p", prompt,
        "--stats"
    ]
    code, out, elapsed = run_cmd_timed(cmd_diff, env=env)
    if code != 0:
        print(f"[-] diffusion.gguf execution failed (code {code}):\n{out[-400:]}")
        stage_results["diffusion.gguf"] = {"error": f"Exit code {code}", "tokens": 0, "tok_per_s": 0.0, "time_s": elapsed}
    else:
        parsed = parse_diffusion_output(out)
        tok_s = parsed["tok_per_s"] if parsed["tok_per_s"] > 0 else (parsed["tokens"] / elapsed if elapsed > 0 else 0.0)
        llama_tok_s = stage_results.get("llama.cpp", {}).get("tok_per_s", 1.0)
        sp = (tok_s / llama_tok_s) if llama_tok_s > 0 else 1.0
        print(f"[+] diffusion.gguf completed: {parsed['tokens']} tokens in {elapsed:.2f}s ({tok_s:.2f} tok/s) | Speedup vs llama: {sp:.2f}x")
        stage_results["diffusion.gguf"] = {
            "tokens": parsed["tokens"],
            "tok_per_s": tok_s,
            "time_s": elapsed,
            "speedup_vs_llama": round(sp, 3),
            "raw_output": out[-500:]
        }

    return stage_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.8-27B: diffusion.gguf vs official real llama.cpp")
    parser.add_argument("-m", "--model", default="", help="Path to Qwen3.8-27B .gguf file")
    parser.add_argument("--quant", default="Q4_K_M", choices=["Q4_K_M", "IQ4_XS", "IQ3_S", "Q4_0"], help="Quantization variant to fetch if no model supplied")
    parser.add_argument("-c", "--ctx", type=int, default=16384, help="Context size (default: 16384)")
    parser.add_argument("-n", "--predict", type=int, default=4096, help="Output tokens limit (default: 4096)")
    parser.add_argument("--diffusion-bin", default="", help="Custom diffusion-cli.exe binary path")
    parser.add_argument("--llama-bin", default="", help="Custom llama-cli.exe / llama-bench.exe binary path")
    parser.add_argument("--skip-download", action="store_true", help="Do not attempt HF download if model missing")
    parser.add_argument("--output-json", default="tests/benchmark_qwen27b_results.json", help="Path to write JSON benchmark results")
    parser.add_argument("--reuse-llama-baseline", action="store_true", help="Reuse previously recorded llama.cpp baseline from output-json if available")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    diff_bin, llama_bin = find_binaries(repo_root, args.llama_bin, args.diffusion_bin)

    if not diff_bin.exists():
        print(f"Error: diffusion-cli binary not found at: {diff_bin}")
        sys.exit(1)
    if not llama_bin.exists():
        print(f"Error: official llama.cpp binary not found at: {llama_bin}")
        sys.exit(1)

    print("=" * 80)
    print(" FORMAL BENCHMARK: Qwen3.8-27B .gguf (diffusion.gguf vs REAL llama.cpp)")
    print("=" * 80)
    print(f" Diffusion Binary: {diff_bin}")
    print(f" Official llama:   {llama_bin} ({llama_bin.name})")
    print(f" Context Size:     {args.ctx}")
    print(f" Max Output:       {args.predict} tokens")
    print(f" Sampling:         Greedy (temp=0.0)")
    print(f" Speculative/MTP:  OFF (Strictly disabled)")
    print("=" * 80)

    # Resolve / Fetch Model
    model_path = ensure_model(args.model, repo_root, quant=args.quant)
    print(f"[*] Benchmark Target Model: {model_path} ({model_path.stat().st_size / (1024**3):.2f} GB)\n")

    # Ensure CUDA is in PATH
    env = os.environ.copy()
    cuda_paths = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    env["PATH"] = f"{cuda_paths};{env.get('PATH', '')}"

    cached_llama = None
    if args.reuse_llama_baseline and Path(args.output_json).exists():
        try:
            with open(args.output_json, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                cached_llama = prev_data.get("stages", {}).get("raw_decode", {}).get("llama.cpp")
        except Exception:
            pass

    all_results = {
        "model": str(model_path),
        "ctx": args.ctx,
        "max_predict": args.predict,
        "stages": {}
    }

    # Stage 1: Raw Decode Speed Benchmark
    all_results["stages"]["raw_decode"] = run_benchmark_stage(
        stage_name="Raw Decode Speed",
        prompt=PROMPT_STAGE1_RAW_DECODE,
        model_path=model_path,
        diff_bin=diff_bin,
        llama_bin=llama_bin,
        ctx_size=args.ctx,
        predict_max=args.predict,
        env=env,
        cached_llama_res=cached_llama
    )

    # Summary
    print(f"\n{'='*90}")
    print(f" {'Benchmark Stage':<30} | {'Engine':<20} | {'Tokens':<8} | {'Speed (tok/s)':<14} | {'Speedup':<8}")
    print(f"{'-'*90}")
    for stage_name, data in all_results["stages"].items():
        llama_r = data.get("llama.cpp", {})
        diff_r = data.get("diffusion.gguf", {})
        print(f" {stage_name:<30} | {'official llama-cli':<20} | {llama_r.get('tokens', 0):<8} | {llama_r.get('tok_per_s', 0.0):<14.2f} | 1.00x")
        sp = diff_r.get("speedup_vs_llama", 1.0)
        print(f" {'':<30} | {'diffusion.gguf':<20} | {diff_r.get('tokens', 0):<8} | {diff_r.get('tok_per_s', 0.0):<14.2f} | {sp:.2f}x")
        print(f"{'-'*90}")
    print(f"{'='*90}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Results written to {args.output_json}")


if __name__ == "__main__":
    main()
