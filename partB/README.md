# Part B — Capacity Reconciliation & Serving Audit

## Overview
This section audits the serving capacity, KV-cache memory requirements, long-context throughput anomalies, and report misinterpretations presented in `REPORT_v0.md` Section 2 based on `bench/model_spec.md` and `bench/bench_log.csv`.

---

## Deliverables in this Directory

- `code/analyze_capacity.py`: Reproducible Python script performing KV cache arithmetic, capacity estimation, preemption detection, and goodput double derivation.
- `calculations.md`: Step-by-step mathematical derivations for KV-cache bytes/token, 4096-token concurrency limit, and goodput formulas.
- `answers.md`: Formal technical answers for questions B1, B2, B3, and B4.
- `results/capacity_summary.json`: Machine-readable summary of derived capacity parameters and benchmark log metrics.

---

## Reproduction Commands

To execute the capacity analysis and verify calculations:

```bash
python FlamAI_Audit_Submission/partB/code/analyze_capacity.py \
    --bench_csv FlamAI_Audit_Submission/original_materials/bench/bench_log.csv \
    --output_dir FlamAI_Audit_Submission/partB/results
```
