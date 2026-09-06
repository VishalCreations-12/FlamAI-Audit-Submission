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

## Reproducibility

All analysis is designed to be reproducible from the repository root.

### Install dependencies

```bash
pip install -r requirements.txt
