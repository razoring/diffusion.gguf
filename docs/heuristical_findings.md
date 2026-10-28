# Empirical Active-Set Refinement & Frontier Width Telemetry Findings

**Date**: October 2026  
**Hardware Profile**: Dual NVIDIA GeForce RTX 3060 12GB (24GB VRAM Total), PCIe 4.0, CUDA Compute Capability 8.6  
**Model**: Gemma 4 E4B Q4_K (5.1 GB)  
**Branch**: `heuristical`  

---

## 1. Executive Summary & Core Results

We instrumented and executed an empirical comparison between:
1. **Degenerate Frontier Baseline ($B=1$)**: Single-position forward passes (exact autoregressive decoding).
2. **Uniform Jacobi Baseline ($B=16$)**: Static block refilling (`active_set: disabled`), width fixed at $W=16$.
3. **Active-Set Natural Collapse ($B=16$)**: Unconstrained active-set refinement (`active_set: natural_collapse`), width contracts naturally as prefix stabilizes.
4. **Active-Set + Adaptive Tail Pruning ($B=16$)**: Active-set refinement with width-dependent tail pruning.

### Primary Benchmark Results (256 Tokens Generated)

| Configuration | Tokens | Time (s) | Throughput (tok/s) | Passes | Tok/Pass | $W_{\text{avg}}$ | Flips at Tail ($c_{15}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Degenerate Baseline ($B=1$)** | 256 | 4.11s | **62.27** | 256 | 1.000 | 1.00 | 0 |
| **Uniform Jacobi ($B=16$)** | 256 | 13.04s | **19.63** | 249 | 1.028 | 16.00 | 248 |
| **Active-Set Natural Collapse** | 256 | 11.79s | **21.72** | 249 | 1.028 | 11.78 | 28 (-89%) |
| **Active-Set Adaptive Tail** | 256 | 11.49s | **22.29** | 249 | 1.028 | 11.78 | 28 (-89%) |

---

## 2. Key Findings & Empirical Data

### 2.1. Active-Set Refinement Outperforms Uniform Refilling
- **Throughput Gain**: Active-Set Natural Collapse achieved **21.72 tok/s** vs **19.63 tok/s** for uniform Jacobi (**+10.6% speedup**).
- **Adaptive Tail Pruning**: Increased throughput further to **22.29 tok/s** (**+13.5% speedup**).
- **Wall-Clock Latency**: Generation time reduced from $13.04\text{s} \to 11.49\text{s}$ (saved 1.55 seconds).

### 2.2. Thrashing Suppression (Position Flips)
In Uniform Jacobi ($B=16$), speculative downstream tokens oscillate almost every pass because upstream keys/values fluctuate:
```
Uniform Flips:    [248, 165, 213, 226, 222, 210, 222, 225, 233, 222, 224, 223, 215, 221, 235, 248]
Active-Set Flips: [248, 165, 215, 239, 226, 220, 217, 218, 233, 220,  91,  94,  70,  77,  48,  28]
```
At index 15 (tail of block), flips plummeted from **248 down to 28** (**-88.7% reduction** in wasted downstream compute).

### 2.3. Frontier Distribution & Throughput Curve $\Theta(W)$
Under Natural Collapse, active widths clustered in the $W \in [9, 16]$ band with $W_{\text{avg}} = 11.78$:
- $W=9$: 3.4% time | **27.5 tok/s**
- $W=10$: 47.6% time | **26.0 tok/s**
- $W=11$: 2.2% time | **26.1 tok/s**
- $W=12$: 9.9% time | **23.8 tok/s**
- $W=14$: 13.5% time | **21.8 tok/s**
- $W=15$: 10.0% time | **21.7 tok/s**
- $W=16$: 13.3% time | **21.3 tok/s**

**Knee Observation**:
Throughput improves monotonically as $W$ contracts from $16 \to 10$ ($21.3 \to 26.0\text{ tok/s}$) because latency drops faster than parallel token yield.

---

## 3. The Central Architectural Dilemma: Yield vs Latency

### Why Single-Token ($B=1$) Remains Faster (62.27 tok/s vs 22.29 tok/s)
- At $B=1$, forward pass latency is $T(1) = \frac{4.11\text{s}}{256} \approx 16.05\text{ ms}$. Each pass yields exactly 1 token $\implies 62.3\text{ tok/s}$.
- At $B=16$, forward pass latency on dual RTX 3060 is $T(16) \approx 52.37\text{ ms}$ ($3.26\times$ slower than $B=1$).
- **The Yield Deficit**: In standard causal generation on Gemma 4 without draft guidance, speculative tokens converged at an average of only **1.028 tokens per pass**.
- **Breakeven Requirement**:
  For diffusion at $W=16$ ($T=52.4\text{ ms}$) to beat single-token decode ($T=16.1\text{ ms}$), the block yield $\mathbb{E}[Y]$ must satisfy:
  $$\mathbb{E}[Y] \ge \frac{T(16)}{T(1)} = \frac{52.4}{16.1} \approx 3.25\text{ tokens per pass}$$
  With $\mathbb{E}[Y] = 1.028$, the engine is paying a $3.26\times$ latency penalty for a $1.028\times$ yield.

### Conclusions for Branch `heuristical`
1. **Active-Set Refinement is Validated**: Eliminating unneeded block refilling and allowing natural frontier collapse reliably cuts latency by 13.5% and suppresses downstream thrashing by ~89%.
2. **Width Floor Finding**: Because throughput $\Theta(W)$ peaks around $W \approx 9\text{--}10$, clamping $W \ge 16$ is harmful. Natural collapse to $W \in [8, 12]$ is superior to static full-block evaluation.
3. **Next Frontier for Diffusion Throughput**:
   To beat the 62 tok/s single-token ceiling, diffusion must increase yield $\mathbb{E}[Y]$ beyond $3.25\text{ tokens/pass}$. This requires:
   - Soft Freeze (Policy 2) to eliminate KV rollback penalties.
   - Initial guesser enhancements (speculative draft model or higher-order n-gram caches) to raise candidate acceptance probability above 1 token/pass.
