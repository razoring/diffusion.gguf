#!/usr/bin/env python3
"""
Parity test runner comparing diffusion runtime against upstream llama-server oracle.
Implements the §9.3 float near-tie validation rule.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = ROOT / "build" / "bin" / "Release"
if not (BIN_DIR / "diffusion-cli.exe").exists():
    BIN_DIR = ROOT / "build" / "bin"

ORACLE_BIN = BIN_DIR / "llama-server.exe" if os.name == "nt" else BIN_DIR / "llama-server"
DIFFUSION_BIN = BIN_DIR / "diffusion-cli.exe" if os.name == "nt" else BIN_DIR / "diffusion-cli"

PORT = 8089
TIE_EPS = 0.025  # Probability gap threshold for float ties per §9.3

PROMPTS = []
prompts_file = ROOT / "tests" / "parity" / "prompts.jsonl"
if prompts_file.exists():
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                PROMPTS.append(json.loads(line)["prompt"])
else:
    PROMPTS = [
        "Once upon a time, there was a little girl named Lily",
        "One day, a little boy went to the park and found a red ball.",
        "A cat and a dog were best friends who loved to run in the garden."
    ]


def wait_for_server(port, timeout=30):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def query_oracle(prompt, port, n_predict=32):
    url = f"http://127.0.0.1:{port}/completion"
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
        "return_tokens": True,
        "n_probs": 2
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    tokens = []
    if "tokens" in data:
        tokens = [t["id"] if isinstance(t, dict) else t for t in data["tokens"]]

    probs_data = data.get("completion_probabilities", [])
    return tokens, data.get("content", ""), probs_data

def run_diffusion_cli(model_path, prompt, n_predict=32, mode="block"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            str(DIFFUSION_BIN),
            "-m", str(model_path),
            "-p", prompt,
            "-n", str(n_predict),
            "--temp", "0",
            "--mode", mode,
            "--tokens-out", tmp_path,
            "--render", "final"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        tokens = []
        if os.path.exists(tmp_path):
            with open(tmp_path, "r", encoding="utf-8") as f:
                line = f.read().strip()
                if line:
                    tokens = [int(x) for x in line.split(",") if x.strip()]
        return tokens, res.stdout.strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def check_near_tie(oracle_probs_data, div_idx):
    if div_idx >= len(oracle_probs_data):
        return False, 0.0
    cp = oracle_probs_data[div_idx]
    top = cp.get("top_logprobs", [])
    if len(top) >= 2:
        p1 = math.exp(top[0]["logprob"])
        p2 = math.exp(top[1]["logprob"])
        gap = abs(p1 - p2)
        return gap <= TIE_EPS, gap
    return False, 0.0

def main():
    model_path = ROOT / "models" / "stories15M-q4_0.gguf"
    if not model_path.exists():
        print(f"Error: model not found at {model_path}. Run scripts/fetch_test_models.py first.")
        sys.exit(1)

    print(f"Starting oracle server on port {PORT}...")
    server_proc = subprocess.Popen([
        str(ORACLE_BIN),
        "-m", str(model_path),
        "-c", "2048",
        "-t", "4",
        "--port", str(PORT)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        if not wait_for_server(PORT):
            print("Error: llama-server failed to start within timeout")
            sys.exit(1)
        print("Oracle server ready. Running parity comparisons...")

        passed = 0
        tie_divergences = 0

        for idx, prompt in enumerate(PROMPTS):
            print(f"\n[{idx+1}/{len(PROMPTS)}] Prompt: \"{prompt}\"")
            oracle_tokens, oracle_text, oracle_probs = query_oracle(prompt, PORT, n_predict=32)
            diff_tokens, diff_text = run_diffusion_cli(model_path, prompt, n_predict=32, mode="block")

            print(f"  Oracle tokens ({len(oracle_tokens)}): {oracle_tokens[:10]}...")
            print(f"  Diffusion tokens ({len(diff_tokens)}): {diff_tokens[:10]}...")

            if oracle_tokens == diff_tokens:
                print("  => EXACT TOKEN-FOR-TOKEN MATCH")
                passed += 1
            else:
                # Find first divergence index
                div_idx = 0
                while div_idx < len(oracle_tokens) and div_idx < len(diff_tokens) and oracle_tokens[div_idx] == diff_tokens[div_idx]:
                    div_idx += 1

                is_tie, gap = check_near_tie(oracle_probs, div_idx)
                if is_tie:
                    print(f"  => ACCEPTED: Near-tie divergence at token {div_idx} (prob gap = {gap:.4f} <= {TIE_EPS}) per §9.3")
                    passed += 1
                    tie_divergences += 1
                else:
                    print(f"  => FAILED: Divergence at token {div_idx} without near-tie (gap = {gap:.4f})")
                    print(f"     Oracle text:    {repr(oracle_text)}")
                    print(f"     Diffusion text: {repr(diff_text)}")
                    sys.exit(1)

        print(f"\nAll {passed}/{len(PROMPTS)} parity tests PASSED! ({tie_divergences} near-tie acceptable)")

    finally:
        server_proc.terminate()
        server_proc.wait()

if __name__ == "__main__":
    main()
