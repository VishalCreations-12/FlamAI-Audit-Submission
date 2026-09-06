import pytest
import pandas as pd
from FlamAI_Audit_Submission.partB.code.analyze_capacity import (
    compute_kv_cache_bytes_per_token,
    compute_max_concurrency,
    derive_goodput_batch24
)

def test_kv_cache_calculation():
    """
    Verifies FLM-4B-Instruct KV cache arithmetic:
    28 layers, 8 KV heads, 128 head_dim, fp16 (2 bytes) = 114,688 bytes/token.
    """
    kv_res = compute_kv_cache_bytes_per_token(
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        bytes_per_elem=2
    )
    assert kv_res["total_kv_bytes_per_token"] == 114688
    assert kv_res["total_kv_kib_per_token"] == 112.0

def test_max_concurrency_calculation():
    """
    Verifies maximum 4096-token sequence capacity calculation on 24GB GPU.
    Usable VRAM = 22.08 GB, Weights = 8.4 GB, Overhead = 1.6 GB -> Avail KV RAM = 12.08 GB.
    KV per 4096 seq = 469,762,048 bytes (448 MiB).
    Max seqs = floor(12.08e9 / 469762048) = 25 sequences.
    """
    cap_res = compute_max_concurrency(
        gpu_total_gb=24.0,
        gpu_util=0.92,
        weights_gb=8.4,
        overhead_gb=1.6,
        seq_len=4096,
        kv_bytes_per_token=114688
    )
    assert cap_res["available_kv_ram_gb_decimal"] == 12.08
    assert cap_res["kv_bytes_per_sequence"] == 469762048
    assert cap_res["max_concurrent_4096_sequences_decimal"] == 25

def test_goodput_double_derivation():
    """
    Verifies double derivation of honest goodput for batch 24 (prompt 3584, gen 512, wall_clock 61.16s).
    """
    data = {
        "batch_size": [24],
        "prompt_len": [3584],
        "gen_len": [512],
        "num_requests": [24],
        "wall_clock_s": [61.16],
        "reported_tok_s": [1607.4],
        "ttft_ms_p50": [500.5],
        "itl_ms_p50": [96.07],
        "e2e_ms_p95": [69221.3],
        "preempted_seqs": [0],
        "kv_cache_util": [0.93]
    }
    df = pd.DataFrame(data)
    
    goodput_res = derive_goodput_batch24(df)
    
    # Derivation 1: (24 * 512) / 61.16 = 200.92 gen_tok/s
    assert goodput_res["goodput_way1_gen_tok_s"] == 200.92
    # Derivation 2 net decode: 12288 / (61.16 - 0.5005) = 202.57 gen_tok/s
    assert goodput_res["goodput_way2_net_decode_tok_s"] == 202.57
    # Misread ratio: 1607.4 / 200.92 = 8.0x overstatement
    assert goodput_res["misread_ratio"] == 8.0
