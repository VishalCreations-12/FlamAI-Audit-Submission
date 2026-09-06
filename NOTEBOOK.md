# Chronological Engineering Laboratory Notebook — FlamAI Audit

This notebook records the step-by-step hypotheses, minimal experiments, exact commands executed, empirical results, dead ends, and revisions during the FlamAI take-home audit assignment.

---

## Stage 1: Initial Workspace & Starter Materials Inspection

### Date: September 6, 2026

### 1.1 Initial Workspace Discovery
- **Action**: Inspected workspace root (`d:/Flam_AI`) and starter kit (`starter_kit (1)/starter_kit`).
- **Files Discovered**:
  - `fertility.py` (Tokenizer benchmarking script)
  - `REPORT_v0.md` (Intern draft report)
  - `corpus_sample/eng_sample.txt`, `hin_sample.txt` (10-line sample parallel text)
  - `bench/model_spec.md` (Serving model architecture & hardware specs)
  - `bench/bench_log.csv` (Serving load test execution log)
  - `AI ASSIGNMENT 2026.pdf` & `Software Development Engineering ( Intern ) (2) (1) (1).pdf`

---

## Stage 2: Part A1 — Evaluation Corpus Assembly & Iterations

### Hypothesis 2.1
*H1: We can load FLORES-200 parallel data using Hugging Face `datasets.load_dataset("facebook/flores", ...)` or downloading raw files from HuggingFace dataset hub.*

### Experiment 2.1
- **Exact Command**:
  ```bash
  python FlamAI_Audit_Submission/partA/code/prepare_corpus.py --num_sentences 500
  ```
- **Result**:
  ```
  HTTP Error 401: Unauthorized / Gated Dataset.
  Dataset 'facebook/flores' is a gated dataset on HuggingFace Hub requiring authentication.
  ```
- **Interpretation & Revision**:
  HF Hub restricts unauthenticated direct access to `facebook/flores`.
  *Surprise / Dead End*: Cannot rely on HuggingFace Hub authentication tokens in automated evaluation environments.
  *Revision*: Tested direct public download links from Meta AI's official NLLB/FLORES storage server (`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`).

### Experiment 2.2 (Successful Corpus Fetch)
- **Exact Command**:
  ```bash
  python -c "import urllib.request; req=urllib.request.Request('https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz'); print(len(urllib.request.urlopen(req).read()))"
  ```
- **Result**: Successfully fetched 25,585,843 bytes archive containing full FLORES-200 dev/devtest splits for 200+ languages.
- **Execution Command**:
  ```bash
  python FlamAI_Audit_Submission/partA/code/prepare_corpus.py --output_dir FlamAI_Audit_Submission/partA/corpus --num_sentences 500
  ```
- **Final Result**: Assembled 500 multi-way parallel sentences across 5 languages:
  - English (`eng_Latn`): 500 sentences, 10,467 words, 62,303 chars, 62,363 bytes
  - Hindi (`hin_Deva`): 500 sentences, 12,084 words, 61,775 chars, 158,437 bytes
  - Kannada (`kan_Knda`): 500 sentences, 7,708 words, 65,096 chars, 175,915 bytes
  - Tamil (`tam_Taml`): 500 sentences, 7,872 words, 71,509 chars, 194,336 bytes
  - Telugu (`tel_Telu`): 500 sentences, 8,060 words, 62,693 chars, 165,771 bytes

---

## Stage 3: Part A2 — Audit of `fertility.py`

### Hypothesis 3.1 (Bug 1: Whitespace Splitting Flaw)
*`line.split(" ")` in `fertility.py` line 62 treats double spaces `'  '` as empty strings `""`, inflating word count.*

### Experiment 3.1
- **Exact Command**:
  ```bash
  python FlamAI_Audit_Submission/partA/code/audit_fertility.py
  ```
- **Result**:
  - Original `line.split(" ")`: `eng` fertility = 1.2652, `hin` fertility = 7.4485
  - Fixed `line.split()`: `eng` fertility = 1.2831, `hin` fertility = 7.5985
  - Discovered double space in `eng_sample.txt` line 7 (`"books  in"`) and `hin_sample.txt` line 10 (`"किताबें  अलमारी"`).
- **Interpretation**: `line.split(" ")` artificially inflates `len(words)` by counting `""` as words, understating true fertility by ~1.1%.

### Hypothesis 3.2 (Conceptual Flaw 1: Tokenizer Bias)
*GPT-2 distorts Hindi token fertility because its BPE vocabulary lacks Devanagari subwords. An Indic-aware tokenizer will eliminate the 5.89x ratio penalty.*

### Experiment 3.2
- **Exact Command**:
  ```bash
  python FlamAI_Audit_Submission/partA/code/audit_fertility.py
  ```
- **Result**:
  - `gpt2` Tokenizer Ratio (HIN/ENG): **5.89x** (1.27 vs 7.45 tok/word)
  - `xlm-roberta-base` Tokenizer Ratio (HIN/ENG): **1.13x** (1.28 vs 1.44 tok/word)
- **Interpretation**: Proven! The 5.89x ratio penalty is an artifact of GPT-2's vocabulary, NOT an intrinsic property of the Hindi language or script.

### Hypothesis 3.3 (Suspicious-but-Correct Item: `unicodedata.normalize("NFC", line)`)
*Is NFC normalization in line 49 unnecessary or distortive?*

### Experiment 3.3
- **Tested**: Evaluated token counts on NFC (composed) vs NFD (decomposed matras) Hindi strings.
- **Result**: NFD format increases token counts unnecessarily because matras are split into separate code points. NFC canonical composition ensures deterministic, minimal tokenization.
- **Interpretation**: `unicodedata.normalize("NFC", line)` is 100% CORRECT and essential.

---

## Stage 4: Part A3 — Corrected Tokenizer Benchmarking

### Experiment 4.1
- **Exact Command**:
  ```bash
  python FlamAI_Audit_Submission/partA/code/run_tokenizer_analysis.py --corpus_dir FlamAI_Audit_Submission/partA/corpus --output_dir FlamAI_Audit_Submission/partA/results
  ```
- **Result Output Table**:
  - `gpt2` (English BPE):
    - `eng`: 26.03 tok/sent, 1.24 tok/word, 0.209 tok/byte (Ratio vs ENG: 1.00x)
    - `hin`: 188.03 tok/sent, 7.78 tok/word, 0.593 tok/byte (Ratio vs ENG: **7.22x**)
    - `kan`: 343.85 tok/sent, 22.30 tok/word, 0.977 tok/byte (Ratio vs ENG: **13.21x**)
    - `tam`: 386.41 tok/sent, 24.54 tok/word, 0.994 tok/byte (Ratio vs ENG: **14.84x**)
    - `tel`: 327.40 tok/sent, 20.31 tok/word, 0.988 tok/byte (Ratio vs ENG: **12.58x**)
  - `xlm-roberta-base` (Multilingual SentencePiece):
    - `eng`: 29.47 tok/sent, 1.41 tok/word, 0.236 tok/byte (Ratio vs ENG: 1.00x)
    - `hin`: 36.62 tok/sent, 1.52 tok/word, 0.116 tok/byte (Ratio vs ENG: **1.24x**)
    - `kan`: 40.14 tok/sent, 2.60 tok/word, 0.114 tok/byte (Ratio vs ENG: **1.36x**)
    - `tam`: 39.22 tok/sent, 2.49 tok/word, 0.101 tok/byte (Ratio vs ENG: **1.33x**)
    - `tel`: 38.63 tok/sent, 2.40 tok/word, 0.116 tok/byte (Ratio vs ENG: **1.31x**)

---

## Stage 5: Part B — Capacity Reconciliation & Serving Audit

### Experiment 5.1 (Capacity Arithmetic & Log Validation)
- **Exact Command**:
  ```bash
  python FlamAI_Audit_Submission/partB/code/analyze_capacity.py --bench_csv FlamAI_Audit_Submission/original_materials/bench/bench_log.csv --output_dir FlamAI_Audit_Submission/partB/results
  ```
- **Results**:
  - KV Cache per Token: $28 \times 2 \times (8 \times 128 \times 2) = 114,688 \text{ bytes} = \mathbf{112.0 \text{ KiB}}$.
  - Usable VRAM (0.92 utilization on 24GB L4): $22.08 \text{ GB}$.
  - Available KV VRAM: $22.08 - 8.40 - 1.60 = \mathbf{12.08 \text{ GB}}$.
  - KV Memory per 4096 Sequence: $4096 \times 114,688 = \mathbf{469,762,048 \text{ bytes}} = 448.0 \text{ MiB}$.
  - Maximum Concurrency: $\lfloor 12.08 / 0.46976 \rfloor = \mathbf{25 \text{ sequences}}$.
  - *Bench Log Match*: Batch 24 has 0 preemptions (`kv_cache_util = 0.93`). Batch 32 has 7 preemptions (`kv_cache_util = 0.97`). Matches $32 - 25 = 7$ preemptions!

### Experiment 5.2 (B3 Goodput Double Derivation)
- **Log Row**: `batch_size = 24`, `prompt_len = 3584`, `gen_len = 512`, `wall_clock_s = 61.16`, `reported_tok_s = 1607.4`.
- **Derivation 1 (End-to-End Generation Rate)**:
  $$\text{Goodput}_1 = \frac{24 \times 512}{61.16} = \frac{12,288}{61.16} = \mathbf{200.92 \text{ gen\_tok/s}}$$
- **Derivation 2 (Net Decode Rate Subtracting Prefill TTFT)**:
  $$\text{Decode Wall Clock} = 61.16 - 0.5005 = 60.6595 \text{ s}$$
  $$\text{Goodput}_2 = \frac{12,288}{60.6595} = \mathbf{202.57 \text{ gen\_tok/s}}$$
- **Conclusion**: `REPORT_v0` misread `reported_tok_s` (total prefill prompt + decode output tokens / wall clock), overstating generation speed by **8.0x** (1607 tok/s vs 201 tok/s).

---

## Stage 6: Unit Test Suite & Verification

### Experiment 6.1
- **Exact Command**:
  ```bash
  pytest FlamAI_Audit_Submission/tests/
  ```
- **Result**: All test cases passed with 100% assertion accuracy.
