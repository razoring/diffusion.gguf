#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI_BIN = ROOT / "build-ninja-cuda" / "bin" / "diffusion-cli.exe"
MODEL_PATH = Path(r"C:\Users\conta\.ollama\models\blobs\sha256-370c2879f17648d337b0ab4e208fdce348e24d1060064e93d7645daafd56d39c")

CASES = [
    (
        "Production Code Completion (In-File Context)",
        "class DLinkedNode:\n    def __init__(self, key=0, value=0):\n        self.key = key\n        self.value = value\n        self.prev = None\n        self.next = None\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cache = dict()\n        self.head = DLinkedNode()\n        self.tail = DLinkedNode()\n        self.head.next = self.tail\n        self.tail.prev = self.head\n        self.capacity = capacity\n        self.size = 0\n\n    def _add_node(self, node):\n        node.prev = self.head\n        node.next = self.head.next\n        self.head.next.prev = node\n        self.head.next = node\n\n    def _remove_node(self, node):\n        prev = node.prev\n        nxt = node.next\n        prev.next = nxt\n        nxt.prev = prev\n\n    def _move_to_head(self, node):\n        self._remove_node(node)\n        self._add_node(node)\n\n    def _pop_tail(self):\n        res = self.tail.prev\n        self._remove_node(res)\n        return res\n\n    def get(self, key: int) -> int:\n"
    ),
    (
        "CSS Theme Variable Mapping",
        "/* Color Theme Mapping */\n--color-primary-100: rgb(240, 245, 255);\n--color-primary-200: rgb(215, 230, 255);\n--color-primary-300: rgb(180, 210, 255);\n--color-primary-400: rgb(140, 185, 255);\n--color-primary-500:"
    ),
    (
        "JSON Database Fixtures",
        '{\n  "users": [\n    {"id": 1, "username": "alice", "role": "admin", "active": true},\n    {"id": 2, "username": "bob", "role": "developer", "active": true},\n    {"id": 3,'
    )
]

def run_case(prompt, mode, block=4, n_predict=64):
    cmd = [
        str(CLI_BIN),
        "-m", str(MODEL_PATH),
        "-p", prompt,
        "-n", str(n_predict),
        "-ngl", "99",
        "-c", "2048",
        "--temp", "0.0",
        "--mode", mode,
        "--render", "final",
        "--stats"
    ]
    if mode == "block":
        cmd.extend(["--block", str(block)])

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error ({mode}): {res.stderr[:200]}")
        return None

    stderr = res.stderr
    idx0 = stderr.rfind("{")
    idx1 = stderr.rfind("}")
    if idx0 != -1 and idx1 != -1 and idx1 > idx0:
        return json.loads(stderr[idx0 : idx1 + 1])
    return None

def main():
    if not CLI_BIN.exists():
        print(f"Binary not found: {CLI_BIN}")
        sys.exit(1)

    print("=" * 80)
    print("  diffusion.gguf Production Coding Benchmark (Gemma-4B, Dual RTX 3060 CUDA)")
    print("=" * 80)

    summary = []

    for name, prompt in CASES:
        print(f"\nEvaluating: {name}")
        print("  Running Autoregressive (AR) baseline...")
        ar = run_case(prompt, "ar")
        if not ar:
            continue
        print(f"    AR:     {ar['tokens_per_second']:.2f} tok/s  ({ar['total_time_ms']:.1f} ms, {ar['iterations']} iters)")

        print("  Running Jacobi Block (B=4)...")
        jacobi = run_case(prompt, "block", block=4)
        if not jacobi:
            continue
        speedup = (jacobi['tokens_per_second'] / ar['tokens_per_second'] - 1.0) * 100.0
        print(f"    Jacobi: {jacobi['tokens_per_second']:.2f} tok/s  ({jacobi['total_time_ms']:.1f} ms, {jacobi['iterations']} iters, {jacobi['tokens_per_iteration']:.2f} tok/iter) -> {speedup:+.1f}% vs AR")

        summary.append({
            "name": name,
            "ar_tok_s": ar['tokens_per_second'],
            "ar_ms": ar['total_time_ms'],
            "jacobi_tok_s": jacobi['tokens_per_second'],
            "jacobi_ms": jacobi['total_time_ms'],
            "tok_per_iter": jacobi['tokens_per_iteration'],
            "speedup_pct": speedup
        })

    print("\n" + "=" * 80)
    print("  SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'Task Name':<35} | {'AR tok/s':<10} | {'Jacobi tok/s':<12} | {'Tok/Iter':<8} | {'Speedup':<8}")
    print("-" * 80)
    for s in summary:
        print(f"{s['name']:<35} | {s['ar_tok_s']:<10.2f} | {s['jacobi_tok_s']:<12.2f} | {s['tok_per_iter']:<8.2f} | {s['speedup_pct']:>+6.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
