
"""Multi-reference evaluation for WikiNews 2024 with "/" separator format.
A word counts as correct if it matches ANY alternative (split by "/").

Aggregation is WORD-WEIGHTED (matches eval_der.aggregate): each per-line
rate is multiplied by that line's word count before summing, so the result
is the corpus-level rate, not a mean of per-line fractions.

Caveat: the WikiNews 2024 gate inputs contain "/" alternatives, so models
evaluated on them copy "/" into their predictions. DER is unaffected (it
compares diacritic marks only, and "/" is not a mark); WER/preservation
treat the whole "/"-containing token as the prediction. This condition is
uniform across all models evaluated here, so relative comparisons hold.
"""
import json, re, sys, unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_der

MARK_RANGES = ((0x064B, 0x0652), (0x0670, 0x0670), (0x0653, 0x065F), (0x06D6, 0x06ED))

def strip_marks(text):
    out = []
    for ch in text:
        o = ord(ch)
        if any(lo <= o <= hi for lo, hi in MARK_RANGES):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))

def load_multi_ref(path):
    """Load WikiNews 2024 multi-ref file. Each word may have alternatives separated by /"""
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Split into words, each word may have / alternatives
        words = line.split()
        # For each word, split by / to get alternatives
        word_alts = []
        for w in words:
            alts = w.split("/")
            word_alts.append(alts)
        # Create bare input (strip marks from first alternative of each word)
        bare_words = []
        for alts in word_alts:
            bare_words.append(strip_marks(alts[0]))
        bare = " ".join(bare_words)
        pairs.append((bare, word_alts))
    return pairs

def evaluate_multi_ref(pred_text, word_alts):
    """Evaluate prediction against multi-ref word alternatives.
    A word is correct if its marks match ANY alternative.
    """
    pred_words = pred_text.split()
    n = max(len(pred_words), len(word_alts))
    if n == 0:
        return {"der": 0, "der_nocase": 0, "wer": 0, "preserve": 0, "words": 0}
    
    der_correct = 0
    der_nocase_correct = 0
    wer_correct = 0
    preserve_correct = 0
    
    for i in range(n):
        p = pred_words[i] if i < len(pred_words) else None
        alts = word_alts[i] if i < len(word_alts) else [""]
        
        if p is None:
            continue
        
        # Check if prediction matches ANY alternative
        der_ok = False
        der_nocase_ok = False
        for alt in alts:
            if eval_der._pairs_ok(p, alt, nocase=False):
                der_ok = True
            if eval_der._pairs_ok(p, alt, nocase=True):
                der_nocase_ok = True
        
        # WER: exact match with first alternative
        wer_ok = (p == alts[0])
        
        # Preservation: base letters match first alternative
        preserve_ok = (eval_der.base_of(p) == eval_der.base_of(alts[0]))
        
        der_correct += int(der_ok)
        der_nocase_correct += int(der_nocase_ok)
        wer_correct += int(wer_ok)
        preserve_correct += int(preserve_ok)
    
    return {
        "der": 1.0 - der_correct / n,
        "der_nocase": 1.0 - der_nocase_correct / n,
        "wer": 1.0 - wer_correct / n,
        "preserve": preserve_correct / n,
        "words": n,
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--model", default="unknown")
    args = ap.parse_args()
    
    pairs = load_multi_ref(args.ref)
    pred_lines = Path(args.pred).read_text(encoding="utf-8").splitlines()
    
    total = {"der": 0, "der_nocase": 0, "wer": 0, "preserve": 0, "words": 0}
    n_lines = min(len(pred_lines), len(pairs))
    
    for i in range(n_lines):
        bare, word_alts = pairs[i]
        pred = pred_lines[i] if i < len(pred_lines) else ""
        result = evaluate_multi_ref(pred, word_alts)
        # Word-weighted: accumulate rate*words per line, then divide by
        # total words. Summing raw per-line fractions and dividing by total
        # words would under-report the corpus rate by ~words-per-line.
        w = result["words"]
        for k in ("der", "der_nocase", "wer", "preserve"):
            total[k] += result[k] * w
        total["words"] += w

    n = max(total["words"], 1)
    agg = {k: v / n for k, v in total.items() if k != "words"}
    agg["words"] = total["words"]
    agg["model"] = args.model
    agg["lines"] = n_lines
    
    print(json.dumps(agg, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
