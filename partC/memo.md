# Part C — Product Decision Memo: Conversational Indic Assistant Strategy

**To**: Product & Engineering Leadership  
**From**: AI/ML Lead Research & Systems Engineer  
**Date**: September 6, 2026  
**Subject**: Strategic Recommendation for Conversational Casualization in Indic Languages  

---

### Executive Recommendation

We recommend **Path (c): Prompt Engineering Only with Few-Shot Exemplars**, supplemented by structured persona framing. 

Under the strict constraints of **1× A100-80GB GPU**, **zero API budget**, **two weeks of development**, and **only 10 hours/week of reviewer time covering Hindi and Kannada**, attempting Supervised Fine-Tuning (Path a) or an Inference-Time Rewriter (Path b) across 6 languages introduces catastrophic quality risks for the 4 un-reviewed languages (Tamil, Telugu, Bengali, Marathi) while adding unnecessary serving latency.

---

### Explicitly Labeled Analysis & Back-of-Envelope Arithmetic

#### ASSUMPTIONS
1. Base model (FLM-4B-Instruct or equivalent 8B multilingual model) already possesses latent conversational Indic fluency, but defaults to textbook formality due to standard system prompts.
2. Native speakers evaluate conversational naturalness at a throughput of **30 prompt-response pairs per hour**.
3. Reviewer covers **Hindi and Kannada ONLY** for 10 hours/week over 3 weeks = **30 total reviewer hours** (300 evaluations/week, max 900 total evaluations).
4. No external commercial LLM APIs can be called (all synthetic generation/eval must run locally on the single A100-80GB GPU).

#### DATA VOLUME
- **Few-Shot Exemplar Pool**: 15–20 gold-standard "formal-to-casual" conversation pairs per language.
- Total curated dataset across 6 languages = **90 to 120 high-quality parallel pairs**.
- Local synthetic candidate generation: 500 candidate responses generated on the A100 GPU using local 8B model with diverse temperature sampling ($T=0.7$), filtered down by reviewer.

#### TRAINING/SERVING COST
- **Training Cost**: **\$0** (No GPU fine-tuning required; 1× A100-80GB remains 100% available for local batch evaluation).
- **Serving Overhead**: **0 ms latency penalty** and **0 MB additional VRAM**. System prompt addition adds only ~150 tokens to prefill context, consuming $< 0.017 \text{ GB}$ of KV cache per request.
- *(Comparison: Path b rewriter adds +150ms latency per request and requires 2GB additional VRAM).*

#### REVIEWER THROUGHPUT
- **Reviewer Availability**: 10 hours/week $\times$ 3 weeks = 30 total hours.
- **Reviewer Throughput**: 30 pairs/hour $\implies$ **max 900 pair evaluations total**.
- **Reviewer Allocation Plan**:
  - Week 1: 5 hours Hindi (150 pairs) + 5 hours Kannada (150 pairs) to select gold few-shot exemplars.
  - Week 2: 5 hours Hindi + 5 hours Kannada to evaluate prompt variants against formal baselines.
  - Week 3 (Launch Review): 10 hours final blind A/B preference testing (300 pairs).
  - *Un-reviewed Languages (Tamil, Telugu, Bengali, Marathi)*: Use native-script cross-lingual transfer templates modeled on the validated Hindi/Kannada few-shot structures.

#### SUCCESS METRIC WITH NUMERIC THRESHOLD
- **Primary Metric**: **Win-Rate in Blind A/B Human Preference Evaluation** (Casual Prompt vs Formal Baseline).
- **Numeric Threshold**: **$\ge 75\%$ Win-Rate** on Hindi and Kannada in blind native-reviewer evaluation at the end of Week 2, with **$\le 5\%$ error/ungrammatical rate**.

#### KILL CRITERION
- **Abandonment Condition**: If after Day 5 of prompt engineering, the casualized system prompt achieves **$< 60\%$ win-rate** over the formal baseline in Hindi/Kannada, OR introduces an ungrammatical/hallucination rate **$> 8\%$** due to prompt confusion.

#### KILL DEADLINE
- **Hard Deadline**: **Day 7 (End of Week 1)** at 17:00 IST. If killed, we fall back to a minimal low-risk system instruction (*"Answer concisely in a friendly tone"*) without few-shot exemplars.

#### DAY-1 EXPERIMENT
- **Setup**: On Day 1, author 5 candidate system prompt variations (ranging from persona instructions like *"Talk like a helpful friend"* to 3-shot in-context exemplars).
- **Execution**: Generate 50 response pairs per prompt across 10 standard user query types on the A100 GPU using local 8B model.
- **Validation**: Hand off the top 30 Hindi and 30 Kannada responses to the native reviewer during their first 2-hour shift on Day 2 to establish the baseline Win-Rate.

---

### Comparative Evaluation Summary Table

| Evaluation Criterion | Path (a): Synthetic SFT | Path (b): 1B Rewriter | Path (c): Prompt Engineering (Selected) |
|---|---|---|---|
| **Development Feasibility (2 Wks)** | Low (requires dataset gen + LoRA + eval) | Medium | **High** (immediate iteration) |
| **Serving Latency Impact** | 0 ms | +150 ms (2nd model pass) | **0 ms** (+150 prefill tokens) |
| **GPU VRAM Overhead** | 0 MB | +2.0 GB | **0 MB** |
| **Risk to Un-reviewed Languages** | **High** (un-inspected SFT weights) | **High** (un-inspected rewriter) | **Low** (interpretable text prompt) |
| **Reviewer Hour Requirement** | >100 hours (unfeasible) | >80 hours | **30 hours** (fits constraint) |
| **API Cost** | \$0 | \$0 | **\$0** |
