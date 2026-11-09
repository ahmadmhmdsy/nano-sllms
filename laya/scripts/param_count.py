r"""True parameter count BEFORE pre-registration (E-65 reflection N3 / rule R5).

Builds a BERT config from CLI args and prints the measured parameter count with a
per-component breakdown, so an architecture is never registered with a guessed size.
Optional --target_m compares against the user-stated size (PASS = within 15%).

Usage:
  & .\.venv\Scripts\python.exe laya/scripts/param_count.py --layers 6 --hidden 512 --ffn 2048 --heads 8 --vocab 30522 --target_m 50
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=30522)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--target_m", type=float, default=0.0,
                    help="user-stated size in M params; PASS/FAIL printed against it")
    a = ap.parse_args()
    from transformers import BertConfig, BertModel
    cfg = BertConfig(vocab_size=a.vocab, hidden_size=a.hidden, num_hidden_layers=a.layers,
                     num_attention_heads=a.heads, intermediate_size=a.ffn,
                     max_position_embeddings=a.seq + 2)
    m = BertModel(cfg)
    total = sum(p.numel() for p in m.parameters())
    by = {}
    for n, p in m.named_parameters():
        top = n.split(".")[0]
        by[top] = by.get(top, 0) + p.numel()
    out = {"layers": a.layers, "hidden": a.hidden, "ffn": a.ffn, "heads": a.heads,
           "seq": a.seq, "params_m": round(total / 1e6, 2),
           "breakdown_m": {k: round(v / 1e6, 2) for k, v in sorted(by.items())}}
    print(json.dumps(out))
    if a.target_m > 0:
        ratio = (total / 1e6) / a.target_m
        verdict = "PASS" if 0.85 <= ratio <= 1.15 else "FAIL"
        print("TARGET %.1fM: measured %.2fM (%.0f%%) -> %s" % (a.target_m, total / 1e6, ratio * 100, verdict))
        sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()