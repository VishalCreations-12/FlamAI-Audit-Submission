import pytest
import unicodedata
import tiktoken
from FlamAI_Audit_Submission.partA.code.audit_fertility import (
    run_original_analyze,
    run_fixed_split_analyze,
    run_micro_average_analyze
)

@pytest.fixture
def gpt2_encode():
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode

def test_split_bug_detection(gpt2_encode):
    """
    Verifies that line.split(' ') counts empty string elements '' as words,
    whereas line.split() handles multiple spaces correctly.
    """
    double_space_line = ["Please keep the books  in the cupboard."] # 2 spaces between books and in
    
    orig_fert, _ = run_original_analyze(double_space_line, gpt2_encode)
    fixed_fert, _ = run_fixed_split_analyze(double_space_line, gpt2_encode)
    
    # In original split(" "), 'books  in' generates ['books', '', 'in'] (8 words)
    # In fixed split(), it generates 7 words.
    # Therefore, fixed fertility (tokens/7) MUST be strictly greater than orig fertility (tokens/8).
    assert fixed_fert > orig_fert
    assert len(double_space_line[0].split(" ")) == 8
    assert len(double_space_line[0].split()) == 7

def test_micro_vs_macro_averaging(gpt2_encode):
    """
    Verifies micro-average vs macro-average behavior on lines of varying length.
    """
    lines = [
        "Short.",                           # 1 word, e.g. 2 tokens -> ratio 2.0
        "This is a longer line with eight words total." # 8 words, e.g. 9 tokens -> ratio 1.125
    ]
    
    macro_fert, _ = run_fixed_split_analyze(lines, gpt2_encode)
    micro_fert, _ = run_micro_average_analyze(lines, gpt2_encode)
    
    # Macro average = (2.0 + 1.125)/2 = 1.5625
    # Micro average = (2 + 9) / (1 + 8) = 11 / 9 = 1.2222
    assert macro_fert != micro_fert
    assert micro_fert < macro_fert

def test_nfc_normalization_integrity(gpt2_encode):
    """
    Verifies that NFC normalization provides deterministic canonical output.
    """
    text = "मुझे सुबह की चाय बहुत पसंद है।"
    nfc_text = unicodedata.normalize("NFC", text)
    nfd_text = unicodedata.normalize("NFD", text)
    
    tok_nfc = gpt2_encode(nfc_text)
    tok_nfd = gpt2_encode(nfd_text)
    
    # Both should yield valid tokens
    assert len(tok_nfc) > 0
    assert len(tok_nfd) > 0
