"""E-13 follow-up: matched-parameter depth-over-width architecture grid.

Builds candidate s7-family configs on CPU and reports parameter counts, so a
depth-over-width arm can be chosen at ~the SAME parameter budget as the current
anchor (18L/64h/256ffn/8H/2KV = 1,083,328 params) instead of drifting upward.

CPU only. No GPU, no training, no files written.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402

BASE = dict(ctx=256, ffn=256, hidden=64, layers=18, n_heads=8, n_kv=2)


def params(cfg):
    m = build_from_config({"model": dict(BASE, **cfg)}, vocab_size=TK.VOCAB_SIZE)
    return sum(p.numel() for p in m.parameters())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, default=None,
                    help="restrict to this head count (isolate depth)")
    ap.add_argument("--n-kv", type=int, default=None)
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()
    heads = (args.n_heads,) if args.n_heads else (4, 5, 6, 7, 8, 10)
    kvs = (args.n_kv,) if args.n_kv else (1, 2)

    base_n = params({})
    print("baseline 18L/64h/256ffn/8H/2KV = %d params" % base_n)
    print()
    rows = []
    for layers in (18, 20, 22, 24, 26, 28, 30, 32, 36):
        for hidden in (40, 44, 48, 52, 56, 60, 64, 72, 80):
            for ffn in (128, 160, 192, 224, 256, 320):
                for n_heads in heads:
                    if hidden % n_heads:
                        continue
                    for n_kv in kvs:
                        if n_heads % n_kv:
                            continue
                        try:
                            n = params(dict(layers=layers, hidden=hidden, ffn=ffn,
                                            n_heads=n_heads, n_kv=n_kv))
                        except Exception:
                            continue
                        rows.append((abs(n - base_n), layers, hidden, ffn,
                                     n_heads, n_kv, n))
    rows.sort()
    print("%-5s %-5s %-5s %-4s %-4s %-10s %s"
          % ("L", "h", "ffn", "H", "KV", "params", "vs base"))
    seen = set()
    shown = 0
    for d, layers, hidden, ffn, n_heads, n_kv, n in rows:
        key = (layers, hidden, ffn, n_heads, n_kv)
        if key in seen or layers <= 18:
            continue
        seen.add(key)
        print("%-5d %-5d %-5d %-4d %-4d %-10d %+.2f%%"
              % (layers, hidden, ffn, n_heads, n_kv, n, 100.0 * (n - base_n) / base_n))
        shown += 1
        if shown >= args.top:
            break


if __name__ == "__main__":
    main()
