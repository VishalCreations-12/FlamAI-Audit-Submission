# Part B — Capacity Reconciliation & Serving Audit Answers

This document provides formal written answers to questions B1 through B4 based on empirical data from `bench/model_spec.md` and `bench/bench_log.csv`.

---

## B1. KV-Cache & Capacity Calculation

### Question B1 (a): Exact KV-Cache Bytes per Token
**Calculated Value**: **114,688 bytes/token** (**112.0 KiB/token**)

**Mathematical Derivation**:
- FLM-4B-Instruct has 28 layers, 8 KV heads (Grouped-Query Attention), head dimension 128, and `fp16` precision (2 bytes/element).
- Key tensor per token per layer = $8 \times 128 \times 2 = 2,048 \text{ bytes}$.
- Value tensor per token per layer = $8 \times 128 \times 2 = 2,048 \text{ bytes}$.
- KV tensor per layer = $2048 + 2048 = 4,096 \text{ bytes} = 4 \text{ KiB}$.
- Total KV cache size across 28 layers = $28 \times 4096 = \mathbf{114,688 \text{ bytes/token}} = \mathbf{112 \text{ KiB/token}}$.

### Question B1 (b): Approximate Maximum 4096-Token Sequence Concurrency
**Calculated Capacity**: **25 sequences** (decimal GB) to **29 sequences** (binary GiB).

**Derivation**:
- Total Usable VRAM (0.92 utilization on 24 GB L4 GPU) = $22.08 \text{ GB}$.
- Model Weights (4.2B fp16) = $8.40 \text{ GB}$. Non-KV Overhead = $1.60 \text{ GB}$.
- Remaining VRAM available for KV cache = $22.08 - 8.40 - 1.60 = \mathbf{12.08 \text{ GB}}$ ($12,080,000,000 \text{ bytes}$).
- KV cache size for one 4096-token sequence = $4,096 \times 114,688 = 469,762,048 \text{ bytes} \approx 448.0 \text{ MiB}$.
- Maximum concurrent 4096-token sequences = $\lfloor \frac{12,080,000,000}{469,762,048} \rfloor = \mathbf{25 \text{ sequences}}$.

**Log Verification**:
In `bench_log.csv`, for prompt 3584 + gen 512 (total 4096 tokens):
- `batch_size = 24`: `kv_cache_util = 0.93`, `preempted_seqs = 0` (Fits within the 25 sequence limit).
- `batch_size = 32`: `kv_cache_util = 0.97`, `preempted_seqs = 7` ($32 - 25 = 7$ sequences preempted due to KV memory exhaustion).
The theoretical prediction matches the log with exact precision.

---

## B2. Long-Context Throughput Anomaly

### Identification & Row/Column Mechanism
In `bench_log.csv`, for prompt length 3584, throughput (`reported_tok_s`) increases from batch 4 (565.4 tok/s) up to batch 24 (1607.4 tok/s). However, at **batch 32**, throughput drops to **1384.0 tok/s**, and at **batch 48**, throughput drops further to **1298.5 tok/s**, despite batch size doubling!

**Mechanism**:
- As derived in B1, the L4 GPU can hold a maximum of ~25 concurrent 4096-token sequences.
- At `batch_size = 32` and `batch_size = 48`, the total requested token allocation exceeds physical GPU VRAM.
- Column `kv_cache_util` reaches peak saturation (**0.97**).
- The vLLM scheduler is forced to **preempt active sequences**, recorded in column `preempted_seqs` (**7 preemptions** at batch 32, **23 preemptions** at batch 48).
- Preempted sequences have their KV blocks evicted. When rescheduled, the engine must re-compute their prompt KV blocks from scratch. This redundant prefill re-computation wastes GPU tensor core cycles and memory bandwidth, causing p95 request latency (`e2e_ms_p95`) to inflate from 69.2s (batch 24) to 105.4s (batch 48) and overall throughput to crash.

### Proposed Deployment Change & Predicted Effect
- **Proposed Configuration Change**: Set `max_num_seqs = 24` in the vLLM serving configuration to cap queue concurrency strictly at or below physical KV cache memory capacity.
- **Predicted Quantitative Effect** (*PREDICTION*): Eliminates preemptions entirely (`preempted_seqs` drops from 23 to 0 at batch 48 queue depth). Eliminates redundant prefill re-computation, reducing total wall clock time for 48 requests from 151.4s to ~122.3s (~19.2% wall-clock latency reduction) and preventing throughput degradation.

---

## B3. REPORT_v0 Section 2 Correction & Goodput Derivation

### Misread Column & Original Interpretation Flaw
- **Misread Column**: `reported_tok_s`
- **What it actually represents**: `reported_tok_s` is the serving harness's aggregate token processing rate across **both prefill and decode phases**:
  $$\text{reported\_tok\_s} = \frac{\text{num\_requests} \times (\text{prompt\_len} + \text{gen\_len})}{\text{wall\_clock\_s}}$$
- **Why the original interpretation is wrong**:
  1. `REPORT_v0` falsely claimed that prompt 3584 gives better throughput (1311 tok/s) than prompt 512 (883 tok/s). Prompt 3584 processed 7× more prefill prompt tokens, inflating `reported_tok_s`, but users received output tokens at a much slower rate due to memory bandwidth bottlenecking during decode.
  2. `REPORT_v0` extrapolated batch 48 to ~3200 tok/s by assuming linear throughput scaling, completely ignoring KV cache saturation and preemption thrashing.

### Double Independent Derivation of Batch 24 Goodput (Prompt 3584, Gen 512, Wall Clock 61.16s)
1. **Derivation 1 (End-to-End Output Generation Rate)**:
   $$\text{Goodput}_1 = \frac{\text{batch\_size} \times \text{gen\_len}}{\text{wall\_clock\_s}} = \frac{24 \times 512}{61.16} = \frac{12,288 \text{ gen tokens}}{61.16 \text{ s}} = \mathbf{200.92 \text{ gen\_tok/s}}$$
2. **Derivation 2 (Net Decode Rate Subtracting Prefill TTFT)**:
   $$\text{Decode Wall Clock} = 61.16 \text{ s} - \frac{500.5 \text{ ms}}{1000} = 60.6595 \text{ s}$$
   $$\text{Goodput}_{2\text{ (net)}} = \frac{12,288 \text{ gen tokens}}{60.6595 \text{ s}} = \mathbf{202.57 \text{ gen\_tok/s}}$$
   *(Peak decode concurrency rate from ITL: $\frac{1000}{96.07} \times 24 = \mathbf{249.82 \text{ tok/s}}$).*

### Correct Conclusion for Section 2
`REPORT_v0` should have concluded:
*"Longer prompts increase prefill processing overhead but do NOT improve text generation speed. Actual generation goodput for long prompts (batch 24) is 200.9 gen_tok/s—an 8.0× reduction compared to the reported 1607 tok/s harness number. Furthermore, scaling beyond batch 24 triggers severe KV cache preemption thrashing, reducing goodput and inflating user latency. Concurrency must be capped at max 24 sequences."*

---

## B4. Production Counter Recommendation

### Recommended Metric/Counter
**`vllm:num_preemptions_total`** (Counter) paired with **`vllm:gpu_cache_usage_perc`** (Gauge).

### Expected Value & Verification Criteria
- **Expected Value**: Under stable operation ($\text{concurrency} \le 24$), `vllm:num_preemptions_total` rate MUST be **0.0 / sec**, and `vllm:gpu_cache_usage_perc` should remain $\le \mathbf{0.93}$.
- **Confirmation of B2 Mechanism**: If concurrency spikes beyond 25, `vllm:gpu_cache_usage_perc` will saturate at $\sim 0.97-1.0$, and `vllm:num_preemptions_total` will begin increasing monotonically. Observing a non-zero preemption rate confirms that memory-driven sequence eviction is occurring.
- **Hypothesis Weakening/Rejection Criteria**: If throughput drops or latency spikes occur while `vllm:num_preemptions_total` remains 0.0 and cache usage is below saturation, the B2 KV-exhaustion hypothesis is REJECTED (indicating CPU-GPU transfer bottlenecks or compute kernel degradation instead).
