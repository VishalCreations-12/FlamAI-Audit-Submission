# Live Defense Session Preparation Guide (DEFENSE_PREP)

This document contains 22 expected defense questions and exact, evidence-grounded answers to prepare for the 30-minute live defense session.

---

### Q1: What exactly was wrong with `fertility.py`?
**Answer**: `fertility.py` contained two primary technical flaws:
1. **Code Bug (`line.split(" ")`)**: Line 62 splits strings strictly on single space `' '`. When double spaces exist (as in `eng_sample.txt` line 7 and `hin_sample.txt` line 10), `split(" ")` creates empty string elements `""` in the word list. This inflates `len(words)` and understates true fertility by ~1.1%.
2. **Metric Flaw (Macro-averaging)**: Lines 64 & 67 compute `sum(per_line_fertility) / n` (unweighted average of per-line ratios), giving equal weight to 1-word lines and 30-word lines, skewing sample means.

---

### Q2: Show the experiment proving the bug.
**Answer**: Run `python partA/code/audit_fertility.py`.
- Baseline `line.split(" ")` on sample: `eng` fertility = 1.2652, `hin` fertility = 7.4485.
- Fixed `line.split()`: `eng` fertility = 1.2831 (delta = +0.0179), `hin` fertility = 7.5985 (delta = +0.1500).
- In `eng_sample.txt` line 7: `"Please keep the books  in the cupboard."` (2 spaces between `books` and `in`). `line.split(" ")` returns 8 elements instead of 7 words.

---

### Q3: What was the conceptual metric problem?
**Answer**:
1. **Tokenizer Bias**: Evaluating Hindi using GPT-2's English-only BPE tokenizer (`gpt2`). GPT-2 lacks Devanagari subwords, forcing byte fallback (~3 tokens per character). REPORT_v0 claimed Hindi is 5.89x worse tokenization due to the script; in reality, using an Indic-aware tokenizer (`xlm-roberta-base`) drops the ratio from **5.89x** to **1.24x**!
2. **Cost Metric Fallacy**: Comparing tokens per whitespace word across languages with different word morphology (agglutinative Dravidian vs English). Serving cost is driven by total tokens per parallel prompt (`tok/sentence`), NOT whitespace words.

---

### Q4: Which suspicious-looking thing was actually correct?
**Answer**: `unicodedata.normalize("NFC", line)` in `read_lines` (Line 49).
- **Why it looks suspicious**: Modifying text strings before tokenization might seem like altering raw input data.
- **Why it is correct**: Indic scripts represent matras (vowel signs) either as precomposed Unicode code points (NFC) or decomposed base consonant + matra code points (NFD). Un-normalized (NFD) text increases token count by decomposing matras into extra tokens. NFC normalization ensures canonical, deterministic tokenization!

---

### Q5: Why is your denominator appropriate?
**Answer**: We evaluated 4 denominators: `tok/word`, `tok/grapheme`, `tok/byte`, and `tok/sentence`.
`tok/sentence` (parallel prompt) is the ONLY denominator that holds semantic meaning strictly constant across languages. Each line in FLORES-200 conveys the exact same semantic content; comparing total tokens per sentence measures the true context length overhead for serving models.

---

### Q6: Why did you choose your tokenizers?
**Answer**:
1. `gpt2`: Used as the direct baseline to replicate and audit `fertility.py` and `REPORT_v0.md`.
2. `xlm-roberta-base`: Open, standard multilingual SentencePiece tokenizer trained on 100+ languages including Devanagari and Dravidian scripts, featuring dedicated Indic subword tokens.

---

### Q7: What single metric should drive routing/cost?
**Answer**: **Tokens per Parallel Sentence / Prompt (`tok/sentence`)** on a multilingual tokenizer.
It directly measures model context consumption ($N_{\text{tokens}}$), which determines KV-cache memory allocation, GPU compute time, and serving throughput.

---

### Q8: What are the major limitations of your corpus?
**Answer**:
1. **Domain Gap**: FLORES-200 is Wikipedia news text; it does not cover informal, conversational queries.
2. **No Code-Switching**: All text is 100% native script; does not measure Latin-script Hinglish/Kanglish.
3. **Sequence Length**: Sentences average ~20 words; does not measure 4k+ long-context document dynamics.

---

### Q9: Derive KV-cache bytes/token.
**Answer**:
- FLM-4B-Instruct: 28 layers, 8 KV heads (GQA), head dim 128, fp16 (2 bytes).
- Key bytes/layer = $8 \times 128 \times 2 = 2048$ bytes.
- Value bytes/layer = $8 \times 128 \times 2 = 2048$ bytes.
- KV bytes/layer = $2048 + 2048 = 4096$ bytes = 4 KiB.
- Total for 28 layers = $28 \times 4096 = \mathbf{114,688 \text{ bytes/token}} = \mathbf{112.0 \text{ KiB/token}}$.

---

### Q10: Derive maximum 4096-token concurrency.
**Answer**:
- Total usable GPU RAM (0.92 util on 24GB L4) = $22.08 \text{ GB}$.
- Model weights (4.2B fp16) = $8.40 \text{ GB}$. Overhead = $1.60 \text{ GB}$.
- Available KV RAM = $22.08 - 8.40 - 1.60 = \mathbf{12.08 \text{ GB}}$ ($12,080,000,000$ bytes).
- KV size for 4096-token seq = $4096 \times 114,688 = 469,762,048 \text{ bytes} \approx 448 \text{ MiB}$.
- Max Concurrency = $\lfloor 12,080,000,000 / 469,762,048 \rfloor = \mathbf{25 \text{ sequences}}$.

---

### Q11: What happens at prompt length 3584?
**Answer**: Prompt 3584 + gen 512 = 4096 total sequence length. Beyond batch size 24 (e.g. batch 32 and 48), requested memory exceeds the 25-sequence GPU capacity limit.

---

### Q12: What is the throughput anomaly?
**Answer**: At prompt 3584, throughput peaks at batch 24 (1607 tok/s) and then **crashes** to 1384 tok/s at batch 32 and 1298 tok/s at batch 48.
**Mechanism**: KV-cache saturation forced vLLM to preempt sequences (7 at batch 32, 23 at batch 48). Rescheduled sequences had to re-compute prefill prompt blocks from scratch, thrashing memory and wasting GPU compute.

---

### Q13: What column did REPORT_v0 misread?
**Answer**: `reported_tok_s`.
REPORT_v0 mistook `reported_tok_s` (total prefill prompt tokens + decode output tokens processed per second) for generation output throughput.

---

### Q14: Derive batch-24 goodput two ways.
**Answer** (Batch 24, prompt 3584, gen 512, wall clock 61.16s):
1. **Way 1 (End-to-End Generation Rate)**: $\frac{24 \times 512}{61.16} = \frac{12,288}{61.16} = \mathbf{200.92 \text{ gen\_tok/s}}$.
2. **Way 2 (Net Decode Rate Subtracting TTFT)**: $\text{Decode Time} = 61.16 - 0.5005 = 60.6595 \text{ s} \implies \frac{12,288}{60.6595} = \mathbf{202.57 \text{ gen\_tok/s}}$.

---

### Q15: Why does batch 48 NOT mean what REPORT_v0 claimed?
**Answer**: REPORT_v0 extrapolated batch 48 to ~3200 tok/s assuming linear scaling. In reality, batch 48 suffered 23 preemptions, wall clock inflated to 151.4s, and actual goodput dropped to $48 \times 512 / 151.41 = \mathbf{162.3 \text{ gen\_tok/s}}$.

---

### Q16: What serving metric would confirm your hypothesis?
**Answer**: `vllm:num_preemptions_total` counter paired with `vllm:gpu_cache_usage_perc` gauge. Preemptions increasing > 0 as cache usage hits 1.0 proves KV-cache exhaustion.

---

### Q17: Why did you choose your Part C approach?
**Answer**: Prompt Engineering (Path c) is the only path that fits 1× A100-80GB, 0 API budget, and 30 hours of Hindi/Kannada reviewer time. It adds 0ms serving latency, 0MB VRAM, and avoids risking un-inspected SFT model weights on the 4 un-reviewed languages.

---

### Q18: What is your success threshold?
**Answer**: $\ge 75\%$ Win-Rate in blind native human preference evaluation (casual prompt vs formal baseline) on Hindi and Kannada at Week 2, with $\le 5\%$ error rate.

---

### Q19: What is your kill criterion?
**Answer**: If by Day 5, casual prompts achieve $< 60\%$ win-rate over baseline or introduce $> 8\%$ ungrammatical/hallucination errors.

---

### Q20: What would change your recommendation?
**Answer**: If we obtained an external API budget (to synthesize 20,000 verified casual pairs) AND hired 4 additional native reviewers for Tamil, Telugu, Bengali, and Marathi, we would upgrade to Path (a) SFT.

---

### Q21: What happens if your denominator changes?
**Answer**: Switching to `tok/word` artificially inflates agglutinative language ratios (Kannada/Tamil look 2.5x worse), whereas switching to `tok/byte` shows Indic script byte expansion (3 bytes/char). `tok/sentence` remains the stable ground truth.

---

### Q22: What is your biggest uncertainty?
**Answer**: Real-world user code-switching (Hinglish/Kanglish) performance under native-script system prompts.
