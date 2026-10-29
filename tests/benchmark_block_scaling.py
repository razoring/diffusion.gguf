#!/usr/bin/env python3
"""
Formal Benchmark Suite: diffusion.gguf (b=1, b=4, b=8, b=16) vs. llama.cpp
Strict parameters: ctx=32000, temp=0.0, max_predict=2048.
Zero additional configuration parameters passed.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PROMPT = (
    "Write a comprehensive, detailed architectural analysis of modern operating system kernels, "
    "comparing monolithic, microkernel, and hybrid designs across memory management, scheduling, and driver isolation."
)

DEFAULT_MODEL_BLOB = Path(r"C:\Users\conta\.ollama\models\blobs\sha256-370c2879f17648d337b0ab4e208fdce348e24d1060064e93d7645daafd56d39c")


def find_default_model() -> str:
    if DEFAULT_MODEL_BLOB.exists():
        return str(DEFAULT_MODEL_BLOB)
    env_m = os.environ.get("MODEL_PATH")
    if env_m and Path(env_m).exists():
        return env_m
    return ""


def find_diffusion_bin(repo_root: Path) -> Path:
    candidates = [
        repo_root / "build-ninja-cuda" / "bin" / "diffusion-cli.exe",
        repo_root / "build-ninja-cuda" / "bin" / "diffusion-cli",
        repo_root / "build" / "bin" / "Release" / "diffusion-cli.exe",
        repo_root / "build" / "bin" / "diffusion-cli.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def find_llama_bin(repo_root: Path, user_llama_bin: str = "") -> tuple[Path, str]:
    """Returns (path, type): type is 'simple' (tests/llama_simple.exe) or 'upstream' (llama-cli / llama-bench)."""
    if user_llama_bin:
        p = Path(user_llama_bin)
        if p.exists():
            return p, "upstream"
        raise FileNotFoundError(f"Specified --llama-bin not found: {user_llama_bin}")

    # Check for built llama_simple
    simple_candidates = [
        repo_root / "build-ninja-cuda" / "bin" / "llama_simple.exe",
        repo_root / "build-ninja-cuda" / "bin" / "llama_simple",
        repo_root / "build" / "bin" / "llama_simple.exe",
    ]
    for c in simple_candidates:
        if c.exists():
            return c, "simple"

    # Search PATH or known locations for upstream llama-cli
    for cmd in ["llama-cli.exe", "llama-cli"]:
        import shutil
        found = shutil.which(cmd)
        if found:
            return Path(found), "upstream"

    return simple_candidates[0], "simple"


def run_process_timed(cmd: list[str], timeout: int = 600) -> tuple[int, str, float]:
    start_t = time.perf_counter()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    stdout, _ = p.communicate(timeout=timeout)
    elapsed = time.perf_counter() - start_t
    return p.returncode, stdout, elapsed


def parse_diffusion_output(stdout: str) -> dict:
    """Extracts completion tokens, throughput, and diffusion passes from diffusion-cli."""
    res = {
        "tokens": 0,
        "tok_per_s": 0.0,
        "iterations": 0,
        "tok_per_iter": 0.0,
        "active_work": 0,
        "time_s": 0.0,
    }

    # Find JSON object containing tokens_per_second
    pos = stdout.rfind('"tokens_per_second"')
    if pos != -1:
        start = stdout.rfind('{', 0, pos)
        end = stdout.find('}', pos)
        if start != -1 and end != -1:
            try:
                data = json.loads(stdout[start:end+1])
                return {
                    "tokens": data.get("tokens", 0),
                    "tok_per_s": data.get("tokens_per_second", 0.0),
                    "iterations": data.get("iterations", 0),
                    "tok_per_iter": data.get("tokens_per_iteration", 0.0),
                    "active_work": data.get("cumulative_active_work", 0),
                    "time_s": (data.get("total_time_ms", 0.0) / 1000.0),
                }
            except Exception:
                pass

    m_tok = re.search(r"Tokens generated:\s*(\d+)", stdout)
    if m_tok:
        res["tokens"] = int(m_tok.group(1))

    m_speed = re.search(r"Generation speed:\s*([\d\.]+)\s*tok/s", stdout)
    if not m_speed:
        m_speed = re.search(r"Throughput:\s*([\d\.]+)\s*tok/s", stdout)
    if m_speed:
        res["tok_per_s"] = float(m_speed.group(1))

    m_iter = re.search(r"Diffusion iterations:\s*(\d+)", stdout)
    if m_iter:
        res["iterations"] = int(m_iter.group(1))

    m_tpi = re.search(r"Tokens / iteration:\s*([\d\.]+)", stdout)
    if m_tpi:
        res["tok_per_iter"] = float(m_tpi.group(1))

    m_work = re.search(r"Cumulative active work:\s*(\d+)", stdout)
    if m_work:
        res["active_work"] = int(m_work.group(1))

    m_time = re.search(r"Generation time:\s*([\d\.]+)\s*s", stdout)
    if m_time:
        res["time_s"] = float(m_time.group(1))

    return res


def parse_llama_simple_output(stdout: str) -> dict:
    """Extracts metrics from tests/llama_simple."""
    res = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prefill_tok_per_s": 0.0,
        "decode_tok_per_s": 0.0,
        "decode_time_ms": 0.0,
    }
    pos = stdout.rfind('"decode_tok_per_s"')
    if pos != -1:
        start = stdout.rfind('{', 0, pos)
        end = stdout.find('}', pos)
        if start != -1 and end != -1:
            try:
                data = json.loads(stdout[start:end+1])
                return {
                    "prompt_tokens": data.get("prompt_tokens", 0),
                    "completion_tokens": data.get("completion_tokens", 0),
                    "prefill_tok_per_s": data.get("prefill_tok_per_s", 0.0),
                    "decode_tok_per_s": data.get("decode_tok_per_s", 0.0),
                    "decode_time_ms": data.get("decode_time_ms", 0.0),
                }
            except Exception:
                pass

    m_prompt = re.search(r"Prompt tokens:\s*(\d+)\s*\(([\d\.]+)\s*tok/s", stdout)
    if m_prompt:
        res["prompt_tokens"] = int(m_prompt.group(1))
        res["prefill_tok_per_s"] = float(m_prompt.group(2))

    m_comp = re.search(r"Completion tokens:\s*(\d+)\s*\(([\d\.]+)\s*tok/s", stdout)
    if m_comp:
        res["completion_tokens"] = int(m_comp.group(1))
        res["decode_tok_per_s"] = float(m_comp.group(2))

    return res


def parse_upstream_llama_cli_output(stdout: str) -> dict:
    """Extracts metrics from standard upstream llama-cli."""
    res = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prefill_tok_per_s": 0.0,
        "decode_tok_per_s": 0.0,
        "decode_time_ms": 0.0,
    }
    # Look for: llama_print_timings: prompt eval time = ... ms / X tokens (Y ms per token, Z tokens per second)
    m_prompt = re.search(r"prompt eval time\s*=\s*[\d\.]+\s*ms\s*/\s*(\d+)\s*tokens.*?\(\s*([\d\.]+)\s*tokens per second\)", stdout)
    if m_prompt:
        res["prompt_tokens"] = int(m_prompt.group(1))
        res["prefill_tok_per_s"] = float(m_prompt.group(2))

    # Look for: llama_print_timings:        eval time = ... ms / X runs (Y ms per token, Z tokens per second)
    m_eval = re.search(r"\beval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*runs.*?\(\s*([\d\.]+)\s*tokens per second\)", stdout)
    if m_eval:
        res["decode_time_ms"] = float(m_eval.group(1))
        res["completion_tokens"] = int(m_eval.group(2))
        res["decode_tok_per_s"] = float(m_eval.group(3))

    return res


def main():
    parser = argparse.ArgumentParser(description="Formal Benchmark: diffusion.gguf vs. llama.cpp")
    parser.add_argument("-m", "--model", default=find_default_model(), help="Path to .gguf model file")
    parser.add_argument("-c", "--ctx", type=int, default=32000, help="Context size (fixed 32000)")
    parser.add_argument("-temp", "--temp", type=float, default=0.0, help="Temperature (fixed 0.0)")
    parser.add_argument("-n", "--predict", type=int, default=2048, help="Max predict tokens (fixed 2048)")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("-f", "--prompt-file", default="", help="Prompt file path")
    parser.add_argument("--llama-bin", default="", help="External llama-cli.exe / llama-bench.exe path")
    parser.add_argument("--diffusion-bin", default="", help="diffusion-cli.exe binary path")
    parser.add_argument("--skip-llama", action="store_true", help="Skip llama.cpp baseline")
    parser.add_argument("--output-json", default="tests/benchmark_results.json", help="Path to save JSON results")
    parser.add_argument("--output-md", default="tests/benchmark_report.md", help="Path to save Markdown report")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if not args.model or not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please specify -m <path_to_model.gguf>")
        sys.exit(1)

    prompt = args.prompt
    if args.prompt_file:
        pf = Path(args.prompt_file)
        if pf.exists():
            prompt = pf.read_text(encoding="utf-8")
        else:
            print(f"Error: Prompt file not found: {args.prompt_file}")
            sys.exit(1)

    diff_bin = Path(args.diffusion_bin) if args.diffusion_bin else find_diffusion_bin(repo_root)
    if not diff_bin.exists():
        print(f"Error: diffusion-cli executable not found at: {diff_bin}")
        print("Please build diffusion-cli first.")
        sys.exit(1)

    results = {}

    print("=" * 80)
    print(" FORMAL BENCHMARK: diffusion.gguf vs. llama.cpp")
    print("=" * 80)
    print(f" Model:       {args.model}")
    print(f" Context:     {args.ctx}")
    print(f" Temperature: {args.temp}")
    print(f" Max Tokens:  {args.predict}")
    print(f" Prompt Chars:{len(prompt)}")
    print(f" Zero additional engine flags.")
    print("=" * 80)

    # 1. Run llama.cpp baseline
    if not args.skip_llama:
        llama_bin_path, llama_type = find_llama_bin(repo_root, args.llama_bin)
        if not llama_bin_path.exists():
            print(f"Warning: llama.cpp binary not found at {llama_bin_path}. Skipping llama.cpp baseline.")
        else:
            print(f"\n[1/5] Running llama.cpp ({llama_type}: {llama_bin_path.name}) ...")
            if llama_type == "simple":
                cmd = [
                    str(llama_bin_path),
                    "-m", args.model,
                    "-c", str(args.ctx),
                    "-temp", str(args.temp),
                    "-n", str(args.predict),
                    "-p", prompt,
                    "--json"
                ]
            else:
                cmd = [
                    str(llama_bin_path),
                    "-m", args.model,
                    "-c", str(args.ctx),
                    "--temp", str(args.temp),
                    "-n", str(args.predict),
                    "-p", prompt
                ]
            code, stdout, elapsed = run_process_timed(cmd)
            if code != 0:
                print(f"Error: llama.cpp run failed with exit code {code}:\n{stdout[-500:]}")
            else:
                parsed = parse_llama_simple_output(stdout) if llama_type == "simple" else parse_upstream_llama_cli_output(stdout)
                tok_s = parsed["decode_tok_per_s"]
                toks = parsed["completion_tokens"]
                print(f"      -> Completed: {toks} tokens in {elapsed:.2f}s ({tok_s:.2f} tok/s)")
                results["llama.cpp"] = {
                    "engine": "llama.cpp",
                    "block_size": 1,
                    "tokens": toks,
                    "tok_per_s": tok_s,
                    "time_s": elapsed,
                    "speedup_vs_llama": 1.0,
                    "speedup_vs_b1": 1.0,
                }

    # 2. Run diffusion.gguf for B in {1, 4, 8, 16}
    block_sizes = [1, 4, 8, 16]
    b1_tok_s = 1.0
    llama_tok_s = results.get("llama.cpp", {}).get("tok_per_s", 0.0)

    for idx, b in enumerate(block_sizes, start=2):
        print(f"\n[{idx}/5] Running diffusion.gguf (b={b}) ...")
        cmd = [
            str(diff_bin),
            "-m", args.model,
            "-c", str(args.ctx),
            "-temp", str(args.temp),
            "-n", str(args.predict),
            "-b", str(b),
            "-p", prompt
        ]
        code, stdout, elapsed = run_process_timed(cmd)
        if code != 0:
            print(f"Error: diffusion-cli b={b} failed with exit code {code}:\n{stdout[-500:]}")
            results[f"diffusion_b{b}"] = {"error": f"Exit code {code}", "tokens": 0, "tok_per_s": 0.0}
            continue

        parsed = parse_diffusion_output(stdout)
        tok_s = parsed["tok_per_s"]
        toks = parsed["tokens"]
        if b == 1:
            b1_tok_s = tok_s if tok_s > 0 else 1.0

        sp_llama = (tok_s / llama_tok_s) if llama_tok_s > 0 else 1.0
        sp_b1 = (tok_s / b1_tok_s) if b1_tok_s > 0 else 1.0

        print(f"      -> Completed: {toks} tokens in {elapsed:.2f}s ({tok_s:.2f} tok/s) | vs llama: {sp_llama:.2f}x | vs b=1: {sp_b1:.2f}x")
        results[f"diffusion_b{b}"] = {
            "engine": "diffusion.gguf",
            "block_size": b,
            "tokens": toks,
            "tok_per_s": tok_s,
            "iterations": parsed["iterations"],
            "tok_per_iter": parsed["tok_per_iter"],
            "time_s": elapsed,
            "speedup_vs_llama": round(sp_llama, 3),
            "speedup_vs_b1": round(sp_b1, 3),
        }

    # Summary Table
    print("\n" + "=" * 92)
    print(f" {'Engine':<18} | {'Block':<6} | {'Tokens':<8} | {'Time (s)':<9} | {'Speed (tok/s)':<14} | {'vs llama.cpp':<13} | {'vs B=1':<8}")
    print("-" * 92)

    rows = []
    if "llama.cpp" in results:
        r = results["llama.cpp"]
        rows.append(("llama.cpp", "1", r["tokens"], f"{r['time_s']:.2f}", f"{r['tok_per_s']:.2f}", "1.00x", "1.00x"))

    for b in block_sizes:
        key = f"diffusion_b{b}"
        if key in results and "tok_per_s" in results[key]:
            r = results[key]
            rows.append((
                "diffusion.gguf",
                str(b),
                r["tokens"],
                f"{r['time_s']:.2f}",
                f"{r['tok_per_s']:.2f}",
                f"{r['speedup_vs_llama']:.2f}x",
                f"{r['speedup_vs_b1']:.2f}x"
            ))

    for row in rows:
        print(f" {row[0]:<18} | {row[1]:<6} | {row[2]:<8} | {row[3]:<9} | {row[4]:<14} | {row[5]:<13} | {row[6]:<8}")
    print("=" * 92)

    # Save JSON results
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": args.model,
                "ctx": args.ctx,
                "temp": args.temp,
                "predict": args.predict,
                "prompt_chars": len(prompt)
            },
            "results": results
        }, f, indent=2)
    print(f"\nJSON results saved to: {out_json}")

    # Save Markdown report
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Formal Benchmark Report: diffusion.gguf vs. llama.cpp\n\n")
        f.write(f"- **Model**: `{args.model}`\n")
        f.write(f"- **Context**: `{args.ctx}` tokens\n")
        f.write(f"- **Temperature**: `{args.temp}`\n")
        f.write(f"- **Max Output**: `{args.predict}` tokens\n\n")
        f.write("| Engine | Block Size | Tokens Generated | Time (s) | Speed (tok/s) | Speedup vs llama.cpp | Speedup vs B=1 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in rows:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |\n")
    print(f"Markdown report saved to: {out_md}\n")


if __name__ == "__main__":
    main()

