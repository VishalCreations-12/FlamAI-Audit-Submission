# Executive Recommendation Memo: Multilingual Tokenizer Audit & Routing Strategy

**To**: FlamAI Leadership Team  
**From**: AI/ML Research & Systems Audit Lead  
**Date**: September 6, 2026  
**Subject**: Corrected Tokenizer Audit Findings & Production Routing Recommendation  

---

### Executive Summary & Corrected Headline Numbers

The initial report (`REPORT_v0.md`) claimed that Hindi tokenization is **5.89× worse** than English and concluded that serving Indic traffic will cost **~6× more per request**. Our technical audit demonstrates that this 6× cost penalty claim is **invalid and false**. 

The 5.89× penalty was entirely an artifact of using an English-centric tokenizer (`gpt2`), which lacks Indic subwords and decomposes Hindi script into raw byte fallbacks. When evaluated on our 500-sentence multi-way parallel FLORES-200 corpus across 5 languages using an Indic-aware multilingual tokenizer (`xlm-roberta-base`), the actual token consumption overhead drops dramatically:

| Language | ISO Code | GPT-2 tok/word | GPT-2 Ratio vs ENG | XLM-R tok/word | XLM-R Ratio vs ENG | XLM-R tok/sentence | XLM-R Sentence Ratio vs ENG |
|---|---|---|---|---|---|---|---|
| **English** | `eng` | 1.28 | 1.00× | 2.01 | 1.00× | 42.11 | **1.00×** |
| **Hindi** | `hin` | 7.45 | **5.82×** | 2.65 | 1.32× | 64.08 | **1.52×** |
| **Kannada** | `kan` | 9.81 | **7.66×** | 3.52 | 1.75× | 54.26 | **1.29×** |
| **Tamil** | `tam` | 11.20 | **8.75×** | 4.18 | 2.08× | 65.81 | **1.56×** |
| **Telugu** | `tel` | 9.54 | **7.45×** | 3.61 | 1.80× | 58.19 | **1.38×** |

*Key Correction*: For identical semantic context (parallel sentences), Indic requests require only **1.29× to 1.56×** more tokens than English—**NOT 6×**.

---

### Key Metric Selection: Tokens per Parallel Sentence (`tok/sentence`)

We strongly advise against using `tok/word` or `tok/char` for capacity or routing decisions:
- **`tok/word` Flaw**: Agglutinative Dravidian languages (Kannada, Tamil, Telugu) join multiple morphemes into single long words. Comparing tokens per whitespace word inflates per-word ratios while ignoring the fact that fewer words are used per sentence.
- **`tok/sentence` Superiority**: Serving compute, KV-cache allocation, and API billing are strictly driven by total sequence token length ($N_{\text{tokens}}$). Total tokens per parallel semantic unit (`tok/sentence`) is the ONLY metric that holds prompt semantics constant across languages.

---

### Strategic Routing Recommendation

1. **Do NOT build or deploy a separate Indic-only infrastructure/model stack**: Budgeting 6× capacity for Indic requests is unnecessary and wasteful.
2. **Standardize on a Multilingual Vocabulary (Vocab Size $\ge$ 128k)**: Ensure all primary serving models (e.g. FLM-4B or Gemma/Llama-3 series) utilize a multilingual tokenizer with dedicated Devanagari and Dravidian subword byte-pair encodings.
3. **Capacity Allocation**: Reserve a modest **+30% to +50% token capacity buffer** for Indic user sessions relative to English sessions, rather than +500%.

---

### Biggest Caveat

**Domain & Register Gap**: The evaluation corpus consists of formal Wikipedia articles (`FLORES-200`). Real-world assistant traffic frequently features **informal, code-switched text** (e.g., Hinglish or Kanglish written in Latin script: *"aaj weather kaisa hai?"*). Latin-script code-switching exhibits different subword splitting behavior than pure native scripts.

---

### Production Metric to Monitor

**`prometheus_model_mean_input_tokens_per_request_by_language`**:  
Track the 7-day rolling mean token length per incoming request grouped by detected input language. If real-world Indic request token counts exceed English by more than **1.6×**, trigger an alert to audit prompt engineering templates and system prompt overhead.
