# FlamAI AI Team Intern Assignment — The Audit

**Candidate / Author**: Vishal S  
**Date**: September 6, 2026  
**Repository**: `FlamAI-Audit-Submission`

---

## Executive Overview

This repository contains the complete submission for the **FlamAI AI Team Intern Assignment — The Audit**.

The assignment evaluates an existing AI/ML system through:

- Tokenizer behavior and fertility analysis
- Capacity and throughput reconciliation
- Evidence-backed engineering decisions
- Reproducible experiments
- Clear documentation and audit reasoning

The submission is designed to be independently reproducible and includes source materials, analysis code, experimental evidence, calculations, tests, and defense preparation.

---

## Key Audit Findings

### Part A — Tokenizer Audit

The tokenizer audit evaluates tokenization behavior across English and multiple Indian languages.

The analysis includes:

- Corpus preparation
- Tokenization experiments
- Baseline measurements
- Bug-audit experiments
- Corrected analysis
- Metric validation
- Fertility calculations
- Cross-language comparison

The experiments investigate whether observed tokenization differences are genuine tokenizer behavior or artifacts caused by corpus preparation, measurement, or implementation issues.

### Part B — Capacity Reconciliation

The capacity analysis reconciles the benchmark numbers against the expected hardware and serving configuration.

The analysis distinguishes between:

1. **End-to-end output goodput:** `200.92 gen_tok/s`
2. **Prefill-subtracted net decode phase rate:** `202.57 gen_tok/s`
3. **Instantaneous steady-state decode rate from ITL:** `249.82 tok/s`
4. **Reported harness metric:** `1607.33 tok/s`

These values are intentionally kept separate because they represent different measurement definitions and should not be treated as interchangeable throughput metrics.

### Part C — Decision Memo

The decision memo converts the technical findings into an engineering recommendation covering:

- Deployment assumptions
- Hardware considerations
- Quality and reliability risks
- Reviewer effort
- Budget constraints
- Recommended next steps
- Risks and mitigations

---

## Repository Structure

```text
FlamAI-Audit-Submission/
│
├── README.md
├── NOTEBOOK.md
├── AI_USAGE.md
├── DEFENSE_PREP.md
├── requirements.txt
├── .gitignore
│
├── original_materials/
│   ├── REPORT_v0.md
│   ├── fertility.py
│   ├── bench/
│   │   ├── bench_log.csv
│   │   └── model_spec.md
│   └── corpus_sample/
│       ├── eng_sample.txt
│       └── hin_sample.txt
│
├── partA/
│   ├── memo.md
│   ├── code/
│   │   ├── audit_fertility.py
│   │   ├── prepare_corpus.py
│   │   └── run_tokenizer_analysis.py
│   ├── corpus/
│   │   ├── README.md
│   │   ├── corpus_stats.json
│   │   ├── processed/
│   │   ├── raw/
│   │   └── flores200_dataset.tar.gz
│   ├── experiments/
│   │   ├── baseline/
│   │   ├── bug_audit/
│   │   ├── corrected_analysis/
│   │   └── metric_audit/
│   └── results/
│       ├── evidence_summary.md
│       ├── tokenizer_results.csv
│       └── tokenizer_results.json
│
├── partB/
│   ├── README.md
│   ├── answers.md
│   ├── calculations.md
│   ├── code/
│   │   └── analyze_capacity.py
│   └── results/
│       └── capacity_summary.json
│
├── partC/
│   └── memo.md
│
└── tests/
    ├── test_capacity.py
    └── test_metrics.py
