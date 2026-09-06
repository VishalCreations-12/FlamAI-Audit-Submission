# Multilingual Evaluation Corpus Documentation (Part A1)

## Overview
This repository section contains the multi-way parallel evaluation corpus assembled for auditing tokenizer fertility and tokenization compression across Indic and Dravidian languages relative to English.

---

## Dataset Details

- **Dataset Name**: FLORES-200 (Meta AI / NLLB Benchmark)
- **Source URL**: `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`
- **License**: CC-BY-SA 4.0
- **Domain**: Professionally translated multi-domain text extracted from Wikipedia (covering science, global news, history, geography, technology, and culture).

---

## Language Coverage

The evaluation set includes **5 languages** covering Indo-Aryan and Dravidian language families:

| Language | ISO Code | FLORES Code | Script | Family | Role in Audit |
|---|---|---|---|---|---|
| English | eng | `eng_Latn` | Latin | Indo-European | Baseline reference |
| Hindi | hin | `hin_Deva` | Devanagari | Indo-Aryan | Primary target Indic language |
| Kannada | kan | `kan_Knda` | Kannada | Dravidian | Primary Dravidian language |
| Tamil | tam | `tam_Taml` | Tamil | Dravidian | Secondary Dravidian language |
| Telugu | tel | `tel_Telu` | Telugu | Dravidian | Additional Dravidian language |

---

## Corpus Statistics & Alignment

- **Number of Sentences**: 500 multi-way parallel sentences per language (2,500 total evaluation sentences).
- **Parallel Alignment**: 100% sentence-level parallel alignment across all 5 files (`eng_eval.txt`, `hin_eval.txt`, `kan_eval.txt`, `tam_eval.txt`, `tel_eval.txt`). Line $i$ in `eng_eval.txt` corresponds to the exact semantic content in line $i$ of all other language files.
- **Preprocessing**:
  1. Extracted raw parallel sentences from the official `dev` split.
  2. Applied Unicode Canonical Composition (`NFC` normalization) via Python `unicodedata.normalize("NFC", line)`.
  3. Stripped trailing whitespace and verified non-empty line integrity.

---

## Limitations: What This Corpus Cannot Tell Us

While FLORES-200 provides high-quality sentence-level parallel alignment, evaluators must recognize its structural boundaries:

1. **Domain & Register Gap**: The corpus consists of formal, professionally translated Wikipedia articles. It does not reflect informal, conversational, colloquial, or dialectal assistant interactions. Real-world user queries frequently feature casual phrasing, slang, and ungrammatical structures.
2. **Code-Switching & Romanized Text**: The corpus is 100% native script (Devanagari, Kannada, Tamil, Telugu). In production, Indian users frequently write Indic languages using the Latin script ("Hinglish", "Tanglish", "Kanglish"). This corpus cannot measure tokenizer behavior on code-switched or Romanized Indic input.
3. **Sequence Length Dynamics**: Sentences average ~20–30 words (100–250 characters). The corpus does not measure tokenization efficiency or KV-cache memory impact on long-context documents, multi-turn conversational histories, or 4000+ token system prompts.
