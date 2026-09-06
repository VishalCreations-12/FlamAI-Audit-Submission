# FlamAI AI Team Intern Assignment — The Audit

**Candidate / Author**: Lead AI/ML Research & Systems Audit Engineer  
**Date**: September 6, 2026  
**Repository**: `FlamAI_Audit_Submission/`  

---

## Executive Overview

This repository contains the complete technical audit, corrected analysis, empirical evidence, capacity reconciliation, product strategy decision, lab notebook, and automated test suite for the FlamAI Take-Home Assignment ("The Audit").

A previous intern benchmarked a tokenizer and a serving stack, producing `REPORT_v0.md`. This audit inspects and evaluates those findings before leadership makes capacity and routing decisions.

---

## Key Audit Findings Summary

### Part A — Tokenizer Audit
1. **Bugs in `fertility.py`**:
   - `line.split(" ")` in line 62 splits strictly on single spaces, creating empty string elements `""` when double spaces are present (e.g. `eng_sample.txt` L7 and `hin_sample.txt` L10). This artificially inflates word count and understates true fertility by ~1.1%.
   - Averaging per-line ratios $\frac{1}{N}\sum \frac{\text{tok}_i}{\text{word}_i}$ (macro-averaging) over-weights short line outliers.
2. **Tokenizer Vocabulary Bias**:
   - `REPORT_v0` claimed Hindi tokenization is **5.89× worse** than English due to the Hindi script.
   - *Audit Finding*: The 5.89× ratio was entirely an artifact of using an English-centric tokenizer (`gpt2`) which lacks Devanagari subwords. When evaluated on our 500-sentence FLORES-200 parallel corpus using an Indic-aware tokenizer (`xlm-roberta-base`), the actual token overhead for Indic languages is only **1.24× to 1.36×** over English for identical prompt semantics.
3. **Metric Selection**:
   - `tok/sentence` (parallel prompt tokens) is the ONLY metric that holds prompt semantics constant across languages. `tok/word` is morphologically confounded by agglutinative Dravidian languages.

### Part B — Capacity Reconciliation
1. **KV Cache & GPU Concurrency**:
   - Derived exact KV cache size per token for **FLM-4B-Instruct**: **114,688 bytes/token** (**112 KiB/token**).
   - Calculated maximum concurrent 4096-token sequence capacity on 1× NVIDIA L4 GPU (24GB VRAM): **25 sequences**.
2. **Prompt 3584 Throughput Anomaly**:
   - At batch sizes 32 and 48 (prompt 3584), requested memory exceeds the 25-sequence GPU capacity limit.
   - The vLLM scheduler was forced to preempt active sequences (7 preemptions at batch 32, 23 at batch 48), thrashing memory and dropping throughput.
3. **Goodput Correction (`REPORT_v0` Section 2)**:
   - `REPORT_v0` misread `reported_tok_s` (aggregate prefill prompt + decode output tokens per second) as generation throughput.
   - Honest generation goodput for batch 24 (long prompt) is **200.9 gen_tok/s**—an **8.0× reduction** compared to the reported 1607 tok/s.
   - Batch 48 extrapolation to ~3200 tok/s was completely invalid.

### Part C — Product Strategy Decision
- Recommended **Path (c): Prompt Engineering Only with Few-Shot Exemplars**.
- Under constraints (1× A100-80GB, 0 API budget, 2 weeks, 30 hours of Hindi/Kannada reviewer time), Path (c) is the ONLY path that fits reviewer bandwidth, adds 0ms serving latency, 0MB VRAM, and avoids risking un-inspected SFT model weights on the 4 un-reviewed languages (Tamil, Telugu, Bengali, Marathi).

---

## Repository Structure

```
FlamAI_Audit_Submission/
│
├── README.md                 # Main executive overview & reproduction guide
├── NOTEBOOK.md               # Chronological lab notebook (hypotheses & experiments)
├── AI_USAGE.md               # Honest disclosure of AI tool usage
├── DEFENSE_PREP.md           # 22 Q&As for live defense session
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
│
├── original_materials/       # Preserved original starter files
│   ├── fertility.py
│   ├── REPORT_v0.md
│   ├── corpus_sample/
│   └── bench/
│
├── partA/                    # Part A: Tokenizer Audit
│   ├── README.md
│   ├── memo.md               # Recommendation memo
│   ├── code/
│   │   ├── prepare_corpus.py
│   │   ├── audit_fertility.py
│   │   └── run_tokenizer_analysis.py
│   ├── corpus/               # FLORES-200 5-language evaluation dataset
│   │   ├── README.md
│   │   ├── raw/
│   │   └── processed/
│   └── results/
│       ├── tokenizer_results.csv
│       ├── tokenizer_results.json
│       └── evidence_summary.md
│
├── partB/                    # Part B: Capacity Reconciliation
│   ├── README.md
│   ├── calculations.md       # Mathematical derivations for KV-cache & goodput
│   ├── answers.md            # Written answers for B1, B2, B3, B4
│   ├── code/
│   │   └── analyze_capacity.py
│   └── results/
│       └── capacity_summary.json
│
├── partC/                    # Part C: Strategic Decision Memo
│   └── memo.md
│
└── tests/                    # Automated Pytest suite
    ├── test_metrics.py
    └── test_capacity.py
```

---

## Setup & Environment Requirements

### Python Version
- **Python 3.10+**

### Dependencies
Install required packages via `requirements.txt`:
```bash
pip install -r FlamAI_Audit_Submission/requirements.txt
```

Core dependencies: `tiktoken`, `transformers`, `grapheme`, `pytest`, `pandas`, `numpy`.

---

## Step-by-Step Reproduction Guide

All commands are designed to be run directly from the workspace root (`d:/Flam_AI`):

### 1. Build Multilingual Evaluation Corpus (Part A1)
Downloads Meta AI's official FLORES-200 benchmark archive and extracts 500 parallel sentences across English, Hindi, Kannada, Tamil, and Telugu:
```bash
python FlamAI_Audit_Submission/partA/code/prepare_corpus.py \
    --output_dir FlamAI_Audit_Submission/partA/corpus \
    --num_sentences 500
```

### 2. Run `fertility.py` Audit & Generate Evidence Summary (Part A2)
Audits code bugs, macro-averaging, GPT-2 tokenizer bias, and NFC Unicode normalization:
```bash
python FlamAI_Audit_Submission/partA/code/audit_fertility.py \
    --sample_dir FlamAI_Audit_Submission/original_materials/corpus_sample \
    --eval_dir FlamAI_Audit_Submission/partA/corpus/processed \
    --output_dir FlamAI_Audit_Submission/partA/results
```

### 3. Run Corrected Tokenizer Analysis (Part A3)
Evaluates `gpt2` vs `xlm-roberta-base` across 5 languages and 4 denominators:
```bash
python FlamAI_Audit_Submission/partA/code/run_tokenizer_analysis.py \
    --corpus_dir FlamAI_Audit_Submission/partA/corpus \
    --output_dir FlamAI_Audit_Submission/partA/results
```

### 4. Run Capacity Reconciliation & Serving Audit (Part B)
Computes KV-cache bytes/token, 4096-token GPU sequence capacity, preemption detection, and goodput double derivation:
```bash
python FlamAI_Audit_Submission/partB/code/analyze_capacity.py \
    --bench_csv FlamAI_Audit_Submission/original_materials/bench/bench_log.csv \
    --output_dir FlamAI_Audit_Submission/partB/results
```

### 5. Execute Automated Test Suite
Runs pytest to validate metric arithmetic, KV-cache math, goodput double derivations, and parser logic:
```bash
pytest FlamAI_Audit_Submission/tests/
```

---

## Limitations & Scope Constraints

1. **Evaluation Corpus Domain**: FLORES-200 text comes from Wikipedia news/articles. It does not reflect informal, conversational, or dialectal user queries.
2. **Code-Switching**: The evaluation corpus is 100% native script. It does not measure Latin-script code-switching (Hinglish, Kanglish).
3. **Hardware Simulation**: Serving capacity calculations are based on theoretical architecture specs and the benchmark log provided in `bench_log.csv`.
