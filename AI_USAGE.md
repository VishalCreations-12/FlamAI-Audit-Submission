# AI Usage Disclosure — FlamAI Audit Submission

In accordance with the assignment ground rules, this document provides an honest, transparent summary of how AI tools were utilized during the completion of this project.

---

## 1. AI Tools Used
- **Primary AI Agent**: Gemini 3.6 Flash (High) via Antigravity Agentic pair-programming assistant.
- **IDE / Environment**: Visual Studio Code / Windows PowerShell terminal.

---

## 2. Scope of AI Assistance

### A. Code Generation Assistance
- **Data Scraping & Downloading**: Generated Python helper scripts (`prepare_corpus.py`) to fetch and extract Meta AI's FLORES-200 benchmark archive directly from `dl.fbaipublicfiles.com`.
- **Audit Automation**: Wrote execution scripts (`audit_fertility.py`, `run_tokenizer_analysis.py`) to automate tokenization benchmarks across 5 languages, 2 tokenizers, and 4 denominators.
- **Capacity Analysis Script**: Built `analyze_capacity.py` to automate deterministic KV-cache arithmetic, 4096-token sequence capacity formulas, and goodput derivations.
- **Automated Tests**: Generated Pytest unit test suites in `tests/test_metrics.py` and `tests/test_capacity.py`.

### B. Research & Analysis Assistance
- **KV Cache Arithmetic Verification**: Verified Grouped-Query Attention (GQA) memory formulas ($2 \times L \times N_{\text{KV}} \times d_{\text{head}} \times \text{precision}$).
- **vLLM Preemption Mechanism**: Synthesized serving stack preemption dynamics when KV cache utilization reaches saturation ($0.97$).
- **Statistical Metric Analysis**: Formulated mathematical comparison between macro-averaging (average of per-line ratios) and micro-average (total tokens over total words).

### C. Documentation Assistance
- Formatted markdown tables, structured executive memos (`partA/memo.md`, `partC/memo.md`), and organized the lab notebook (`NOTEBOOK.md`).

---

## 3. Incorrect AI Suggestions & How They Were Detected

1. **Gated Dataset Download Suggestion**:
   - *AI Suggestion*: Initially attempted to use HuggingFace `datasets.load_dataset("facebook/flores", ...)` or `huggingface_hub.hf_hub_download`.
   - *Failure / Error*: Failed with `401 Unauthorized` because Meta's `facebook/flores` dataset on HuggingFace Hub is gated and requires user token login.
   - *Detection & Correction*: Detected via terminal error logs. Switched to fetching Meta AI's official open public release archive (`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`) directly via standard HTTP.

2. **Bitwise Operator Syntax Bug**:
   - *AI Suggestion*: Wrote `(8.4 * 10^9)` in Python math expression.
   - *Failure / Error*: `10^9` evaluated bitwise XOR instead of exponentiation `10**9`, causing a `TypeError`.
   - *Detection & Correction*: Caught immediately by running `python analyze_capacity.py` in the terminal; fixed with `10**9`.

3. **RAM Definition Ambiguity (Decimal vs Binary)**:
   - *AI Suggestion*: Mixed decimal GB ($10^9$) and binary GiB ($1024^3$) in GPU RAM capacity calculations without explicitly distinguishing them.
   - *Detection & Correction*: Audit lead noticed the discrepancy between 25 sequences (decimal basis) and 29 sequences (binary basis) and updated `calculations.md` and `answers.md` to present BOTH derivations transparently.

---

## 4. Independent Verification Statement

Every single numerical value, table entry, and technical claim in this submission was **empirically validated** by running code locally in the workspace:
- CSV/JSON outputs (`tokenizer_results.csv`, `capacity_summary.json`) were generated directly by Python scripts.
- Pytest suite (`pytest tests/`) was executed and passed with 0 errors.
- All derivations in Part B were verified by independent arithmetic methods.

---

## 5. Key Defense Checklist (What the Student Understands)

Before entering the live defense, the student personally understands:
1. Exactly why `line.split(" ")` in `fertility.py` creates empty string elements `""` when double spaces are present, understating fertility.
2. Why GPT-2 yields 7.45 tokens/word for Hindi while XLM-RoBERTa yields 1.52 tokens/word.
3. Why `unicodedata.normalize("NFC", line)` is 100% correct and why removing it inflates token counts for decomposed matras.
4. Why `tok/sentence` (parallel prompt tokens) is the only metric that holds semantic meaning constant across languages.
5. How KV cache per token ($114,688 \text{ bytes}$) is derived from FLM-4B-Instruct's 28 layers, 8 KV heads, 128 head dim, and fp16 precision.
6. Why batch 32 at prompt length 3584 causes 7 preemptions (exceeding 25 sequence GPU memory limit).
7. How `REPORT_v0` misread `reported_tok_s` (total prefill+decode throughput) as generation speed, overstating goodput by 8.0× (1607 tok/s vs 201 gen_tok/s).
8. Why Prompt Engineering (Path c) is the only viable product choice under 1× A100-80GB, 0 API budget, and 30 hours of Hindi/Kannada reviewer time.
