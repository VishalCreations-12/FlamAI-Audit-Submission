# Part B — Mathematical Derivations & Calculations

This document provides the complete, transparent mathematical derivations for Part B (Capacity Reconciliation).

---

## B1 (a): Exact KV-Cache Bytes per Token

### Model Architecture Parameters (`bench/model_spec.md`):
- Model: **FLM-4B-Instruct** (dense)
- Number of layers ($L$): $28$
- Hidden dimension ($d_{\text{model}}$): $3072$
- Query Attention Heads ($N_Q$): $24$
- Key/Value Attention Heads ($N_{\text{KV}}$): $8$ (Grouped-Query Attention / GQA)
- Head dimension ($d_{\text{head}}$): $128$
- Data Precision: `fp16` (2 bytes per element)

### Formula:
For Grouped-Query Attention (GQA), each layer stores Key ($K$) and Value ($V$) tensors for each token in the context:
$$\text{Key Bytes per token per layer} = N_{\text{KV}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
$$\text{Value Bytes per token per layer} = N_{\text{KV}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$

$$\text{KV Bytes per token per layer} = 2 \times (N_{\text{KV}} \times d_{\text{head}} \times \text{bytes\_per\_elem})$$

$$\text{Total KV Bytes per Token} = L \times 2 \times (N_{\text{KV}} \times d_{\text{head}} \times \text{bytes\_per\_elem})$$

### Arithmetic:
$$\text{Key Bytes/layer} = 8 \times 128 \times 2 = 2,048 \text{ bytes}$$
$$\text{Value Bytes/layer} = 8 \times 128 \times 2 = 2,048 \text{ bytes}$$
$$\text{KV Bytes/layer} = 2,048 + 2,048 = 4,096 \text{ bytes} = 4 \text{ KiB}$$

$$\text{Total KV Bytes per Token} = 28 \text{ layers} \times 4,096 \text{ bytes/layer} = \mathbf{114,688 \text{ bytes}} = \mathbf{112.0 \text{ KiB}}$$

---

## B1 (b): Maximum Concurrent 4096-Token Sequence Capacity

### Hardware & Configuration Parameters:
- GPU: $1 \times \text{NVIDIA L4 (24 GB VRAM)}$
- Target GPU Memory Utilization (`gpu_memory_utilization`): $0.92$
- Model Weights (4.2B parameters, `fp16`): $4.2 \times 10^9 \times 2 \text{ bytes} = 8.4 \text{ GB}$
- Non-KV Runtime Overhead (activations, CUDA graphs, etc.): $1.6 \text{ GB}$
- Target Sequence Length ($N_{\text{seq}}$): $4,096 \text{ tokens}$

### 1. Decimal GB Basis ($1 \text{ GB} = 10^9 \text{ bytes}$):
$$\text{Total GPU RAM} = 24.0 \text{ GB} = 24,000,000,000 \text{ bytes}$$
$$\text{Usable VRAM} = 0.92 \times 24.0 \text{ GB} = 22.08 \text{ GB} = 22,080,000,000 \text{ bytes}$$

$$\text{Available KV RAM} = \text{Usable VRAM} - \text{Weights} - \text{Overhead}$$
$$\text{Available KV RAM} = 22.08 \text{ GB} - 8.40 \text{ GB} - 1.60 \text{ GB} = \mathbf{12.08 \text{ GB}} = 12,080,000,000 \text{ bytes}$$

$$\text{KV Bytes per 4096-Token Sequence} = 4,096 \times 114,688 = \mathbf{469,762,048 \text{ bytes}} \approx 448.0 \text{ MiB} = 0.469762 \text{ GB}$$

$$\text{Max Concurrent Sequences} = \lfloor \frac{12,080,000,000}{469,762,048} \rfloor = \lfloor 25.715 \rfloor = \mathbf{25 \text{ sequences}}$$

### 2. Binary GiB Basis ($1 \text{ GiB} = 1,073,741,824 \text{ bytes}$):
$$\text{Total GPU RAM} = 24 \text{ GiB} = 25,769,803,776 \text{ bytes}$$
$$\text{Usable VRAM} = 0.92 \times 25,769,803,776 = 23,708,219,474 \text{ bytes}$$
$$\text{Available KV RAM} = 23,708,219,474 - 8,400,000,000 - 1,600,000,000 = 13,708,219,474 \text{ bytes}$$

$$\text{Max Concurrent Sequences} = \lfloor \frac{13,708,219,474}{469,762,048} \rfloor = \lfloor 29.18 \rfloor = \mathbf{29 \text{ sequences}}$$

---

## B3: Double Derivation of Honest Goodput for Batch 24 Long Prompt

Log Row: `batch_size = 24`, `prompt_len = 3584`, `gen_len = 512`, `wall_clock_s = 61.16`, `reported_tok_s = 1607.4`, `ttft_ms_p50 = 500.5`, `itl_ms_p50 = 96.07`.

### Derivation 1: Total Generated Output Tokens / Wall Clock Time
$$\text{Total Generated Tokens} = \text{batch\_size} \times \text{gen\_len} = 24 \times 512 = 12,288 \text{ tokens}$$

$$\text{Goodput}_1 = \frac{12,288 \text{ generated tokens}}{61.16 \text{ seconds}} = \mathbf{200.92 \text{ gen\_tok/s}}$$

### Derivation 2: Net Decode Phase Throughput
$$\text{Prefill Time} = \frac{\text{ttft\_ms\_p50}}{1000} = \frac{500.5}{1000} = 0.5005 \text{ seconds}$$
$$\text{Decode Wall Clock Time} = 61.16 - 0.5005 = 60.6595 \text{ seconds}$$

$$\text{Goodput}_{2\text{ (net)}} = \frac{12,288 \text{ generated tokens}}{60.6595 \text{ seconds}} = \mathbf{202.57 \text{ gen\_tok/s}}$$

*(Peak concurrency decode throughput from ITL: $\frac{1000}{\text{itl\_ms}} \times \text{batch\_size} = \frac{1000}{96.07} \times 24 = \mathbf{249.82 \text{ tok/s}}$).*

Both independent derivations confirm that output token generation rate is **~201–203 gen_tok/s**, proving that `reported_tok_s = 1607.4` is an **8.0× overstatement** caused by counting prefill prompt tokens.
