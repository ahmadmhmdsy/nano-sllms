"""E-084 triage bed - CPU preflight. Proves the screen's exact code paths BEFORE
any GPU spend. SCREEN-NOT-FULL-TRAIN: this is not a measurement of anything.

Checks, all on CPU:
  1. the anchor loads into BOTH the bidirectional and the causal build with zero
     missing and zero unexpected tensors, so a causal screen is a real
     weight-preserving ablation and not a silent re-init;
  2. the causal build actually changes behaviour - identical outputs would mean
     the causal flag never reached the attention;
  3. bench.py's registered decode path (chunking + arqmax + mark insertion)
     preserves the bare text for both builds, so a screen number is the
     registered metric and not a mangled string;
  4. the SDPA is_causal/attn_mask fold is equivalent to the explicit math path.
"""
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
sys.path.insert(0, str(REPO / "diacritizer" / "src"))

from model import build_from_config        # noqa: E402
import tokenizer as TK                     # noqa: E402
import bench as B                          # noqa: E402
from passthrough import is_arabic_base     # noqa: E402


def sdpa_fold_check():
    q = torch.randn(2, 8, 5, 8)
    k = torch.randn(2, 8, 5, 8)
    v = torch.randn(2, 8, 5, 8)
    mask = torch.zeros(2, 1, 1, 5, dtype=torch.bool)
    mask[:, :, :, :4] = True
    try:
        torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, attn_mask=mask)
        both = "unexpectedly-accepted"
    except RuntimeError as exc:
        both = "raises: " + str(exc)[:70]
    tri = torch.ones(5, 5, dtype=torch.bool).tril()
    folded = mask & tri
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=folded)
    s = torch.matmul(q, k.transpose(-2, -1)) * (8 ** -0.5)
    s = s.masked_fill(~folded, float("-inf"))
    ref = torch.matmul(torch.softmax(s, dim=-1), v)
    return {"torch": torch.__version__, "is_causal_plus_attn_mask": both,
            "fold_vs_math_max_abs_diff": float((out - ref).abs().max())}


def strip_marks(text):
    return B.strip_marks(text)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--src", default="fadel_test")
    ap.add_argument("--lines", type=int, default=3)
    args = ap.parse_args()
    # repo convention (bench.py): cp1252 console cannot encode Arabic samples
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    state = torch.load(REPO / args.anchor, map_location="cpu", weights_only=False)
    sd = state["model"] if (isinstance(state, dict) and "model" in state) else state
    pairs = list(B.load_pairs(args.src, ctx))[:args.lines]

    rep = {"status": "SCREEN-NOT-FULL-TRAIN", "anchor": args.anchor,
           "config": args.config, "sdpa": sdpa_fold_check(), "builds": {}}
    preds = {}
    for name, causal in (("bidirectional", False), ("causal", True)):
        cc = copy.deepcopy(cfg)
        cc["model"]["causal"] = causal
        model = build_from_config(cc, vocab_size=TK.VOCAB_SIZE)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.eval()
        out = B.predict_bares(model, pairs, ctx, "cpu", 2)
        preds[name] = out
        pres = all(strip_marks(p) == strip_marks(b) for p, (b, _) in zip(out, pairs))
        rep["builds"][name] = {"missing": len(missing), "unexpected": len(unexpected),
                               "text_preserved": bool(pres),
                               "sample": out[0][:70] if out else ""}
        del model
    rep["causal_changes_output"] = preds["causal"] != preds["bidirectional"]
    ok = (all(v["missing"] == 0 and v["unexpected"] == 0
              and v["text_preserved"] for v in rep["builds"].values())
          and rep["causal_changes_output"]
          and rep["sdpa"]["fold_vs_math_max_abs_diff"] < 1e-5)
    rep["PREFLIGHT"] = "PASS" if ok else "FAIL"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
