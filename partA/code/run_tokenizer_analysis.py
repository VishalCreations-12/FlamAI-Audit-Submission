#!/usr/bin/env python3
"""
run_tokenizer_analysis.py -- Corrected Multilingual Tokenizer Analysis for FlamAI Audit

Evaluates tokenization behavior on the 500-sentence multi-way parallel FLORES-200 corpus across:
- 5 Languages: English (eng), Hindi (hin), Kannada (kan), Tamil (tam), Telugu (tel)
- 2 Tokenizers: GPT-2 (English BPE) vs XLM-RoBERTa (Multilingual SentencePiece)
- 4 Denominators:
    1. Whitespace words (tok/word)
    2. Grapheme clusters (tok/grapheme)
    3. UTF-8 bytes (tok/byte)
    4. Parallel sentences (tok/sentence)

Outputs:
- partA/results/tokenizer_results.csv
- partA/results/tokenizer_results.json
"""

import os
import sys
import json
import argparse
import pandas as pd
import grapheme
import tiktoken
from transformers import AutoTokenizer

def count_graphemes(text):
    return grapheme.length(text)

def load_corpus_files(corpus_dir, languages=["eng", "hin", "kan", "tam", "tel"]):
    corpus = {}
    proc_dir = os.path.join(corpus_dir, "processed")
    for lang in languages:
        path = os.path.join(proc_dir, f"{lang}_eval.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing corpus file for {lang}: {path}")
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        corpus[lang] = lines
    return corpus

def analyze_tokenizer_on_corpus(lines, encode_fn, name):
    num_sentences = len(lines)
    total_tokens = 0
    total_words = 0
    total_graphemes = 0
    total_bytes = 0

    for line in lines:
        tokens = encode_fn(line)
        words = line.split()
        graphemes_cnt = count_graphemes(line)
        bytes_cnt = len(line.encode("utf-8"))

        total_tokens += len(tokens)
        total_words += len(words)
        total_graphemes += graphemes_cnt
        total_bytes += bytes_cnt

    return {
        "tokenizer": name,
        "sentences": num_sentences,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_graphemes": total_graphemes,
        "total_bytes": total_bytes,
        "tok_per_sentence": round(total_tokens / num_sentences, 4),
        "tok_per_word": round(total_tokens / total_words, 4),
        "tok_per_grapheme": round(total_tokens / total_graphemes, 4),
        "tok_per_byte": round(total_tokens / total_bytes, 4)
    }

def main():
    parser = argparse.ArgumentParser(description="Run Corrected Tokenizer Analysis")
    parser.add_argument("--corpus_dir", default="FlamAI_Audit_Submission/partA/corpus")
    parser.add_argument("--output_dir", default="FlamAI_Audit_Submission/partA/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading multi-way parallel evaluation corpus...")
    corpus = load_corpus_files(args.corpus_dir)

    print("Loading tokenizers...")
    gpt2_enc = tiktoken.get_encoding("gpt2")
    gpt2_encode = gpt2_enc.encode

    xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    xlm_encode = lambda s: xlm_tok.encode(s, add_special_tokens=False)

    tokenizers = {
        "gpt2": gpt2_encode,
        "xlm-roberta-base": xlm_encode
    }

    records = []

    print("\nRunning multi-denominator analysis across 5 languages...")
    for tok_name, encode_fn in tokenizers.items():
        for lang, lines in corpus.items():
            res = analyze_tokenizer_on_corpus(lines, encode_fn, tok_name)
            res["language"] = lang
            records.append(res)

    df = pd.DataFrame(records)

    # Re-order columns for clarity
    cols = [
        "tokenizer", "language", "sentences", "total_tokens",
        "tok_per_sentence", "tok_per_word", "tok_per_grapheme", "tok_per_byte",
        "total_words", "total_graphemes", "total_bytes"
    ]
    df = df[cols]

    # Calculate ratios relative to English baseline for each tokenizer & metric
    eng_baseline = df[df["language"] == "eng"].set_index("tokenizer")

    df["ratio_tok_per_sentence_vs_eng"] = df.apply(
        lambda r: round(r["tok_per_sentence"] / eng_baseline.loc[r["tokenizer"], "tok_per_sentence"], 2), axis=1
    )
    df["ratio_tok_per_word_vs_eng"] = df.apply(
        lambda r: round(r["tok_per_word"] / eng_baseline.loc[r["tokenizer"], "tok_per_word"], 2), axis=1
    )
    df["ratio_tok_per_byte_vs_eng"] = df.apply(
        lambda r: round(r["tok_per_byte"] / eng_baseline.loc[r["tokenizer"], "tok_per_byte"], 2), axis=1
    )

    csv_path = os.path.join(args.output_dir, "tokenizer_results.csv")
    json_path = os.path.join(args.output_dir, "tokenizer_results.json")

    df.to_csv(csv_path, index=False)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("\nCorrected Tokenizer Results Summary Table:")
    print(df[["tokenizer", "language", "tok_per_sentence", "tok_per_word", "tok_per_grapheme", "tok_per_byte", "ratio_tok_per_sentence_vs_eng"]].to_string())

    print(f"\nMachine-readable results saved to:\n  CSV: {csv_path}\n  JSON: {json_path}")

if __name__ == "__main__":
    main()
