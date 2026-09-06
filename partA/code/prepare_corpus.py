#!/usr/bin/env python3
"""
prepare_corpus.py -- Multilingual Evaluation Corpus Builder for FlamAI Audit

Downloads and prepares the official Meta FLORES-200 multi-way parallel evaluation benchmark:
- English (eng_Latn)
- Hindi (hin_Deva)
- Kannada (kan_Knda)
- Tamil (tam_Taml)
- Telugu (tel_Telu)

Source: Meta AI NLLB / FLORES-200 Benchmark (dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz)
Domain: Multi-domain (Wikipedia news, science, culture, technology, history)
Alignment: Strict 1-to-1 parallel sentence alignment across all 5 languages.
"""

import os
import sys
import argparse
import json
import urllib.request
import tarfile
import unicodedata

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

LANG_MAP = {
    "eng": "eng_Latn",
    "hin": "hin_Deva",
    "kan": "kan_Knda",
    "tam": "tam_Taml",
    "tel": "tel_Telu"
}

def download_and_extract_flores(output_dir, num_sentences=500, split="dev"):
    raw_dir = os.path.join(output_dir, "raw")
    proc_dir = os.path.join(output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    archive_path = os.path.join(raw_dir, "flores200_dataset.tar.gz")
    
    if not os.path.exists(archive_path):
        print(f"Downloading FLORES-200 benchmark archive from Meta AI ({FLORES_URL})...")
        req = urllib.request.Request(FLORES_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(archive_path, "wb") as out_f:
            out_f.write(resp.read())
        print(f"Downloaded FLORES-200 archive ({os.path.getsize(archive_path)} bytes).")

    print(f"Extracting '{split}' split for languages: {list(LANG_MAP.keys())}...")
    corpus_raw = {}
    
    with tarfile.open(archive_path, mode="r:gz") as tar:
        members = tar.getmembers()
        for lang_key, flores_code in LANG_MAP.items():
            target_name = f"flores200_dataset/{split}/{flores_code}.{split}"
            found = False
            for m in members:
                if m.name.endswith(target_name) or target_name in m.name:
                    f = tar.extractfile(m)
                    lines = [line.decode("utf-8").strip() for line in f if line.decode("utf-8").strip()]
                    corpus_raw[lang_key] = lines
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not find {target_name} in FLORES-200 archive!")

    # Verify sentence counts & preserve exact parallel alignment
    min_sentences = min(len(lines) for lines in corpus_raw.values())
    target_count = min(min_sentences, num_sentences)
    print(f"Selected {target_count} multi-way parallel sentences across all {len(LANG_MAP)} languages.")

    stats = {}
    for lang_key, lines in corpus_raw.items():
        subset = lines[:target_count]
        
        # Save raw file
        raw_path = os.path.join(raw_dir, f"{lang_key}_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write("\n".join(subset) + "\n")
            
        # NFC Unicode Normalization & Cleaned saving
        proc_lines = [unicodedata.normalize("NFC", line) for line in subset]
        proc_path = os.path.join(proc_dir, f"{lang_key}_eval.txt")
        with open(proc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(proc_lines) + "\n")

        # Stats
        total_chars = sum(len(line) for line in proc_lines)
        total_words = sum(len(line.split()) for line in proc_lines)
        stats[lang_key] = {
            "language_code": LANG_MAP[lang_key],
            "sentences": target_count,
            "total_words": total_words,
            "total_chars": total_chars,
            "total_bytes": sum(len(line.encode("utf-8")) for line in proc_lines),
            "avg_words_per_sent": round(total_words / target_count, 2),
            "avg_chars_per_sent": round(total_chars / target_count, 2)
        }

    # Save summary stats
    stats_path = os.path.join(output_dir, "corpus_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Corpus prepared successfully!")
    print(json.dumps(stats, indent=2))
    return stats

def main():
    parser = argparse.ArgumentParser(description="Prepare FLORES-200 Evaluation Corpus")
    parser.add_argument("--output_dir", default="partA/corpus", help="Target corpus directory")
    parser.add_argument("--num_sentences", type=int, default=500, help="Number of parallel sentences")
    args = parser.parse_args()

    download_and_extract_flores(args.output_dir, num_sentences=args.num_sentences)

if __name__ == "__main__":
    main()
