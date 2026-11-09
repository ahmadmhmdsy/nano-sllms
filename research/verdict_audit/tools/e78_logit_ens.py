"""E-078 - cross-basin LOGIT-level ensemble.

E-073 refuted merging the c4b and c4 basins at the WEIGHT level: the uniform
mean of all 24 probes (all24_uniform) was 19.89 percent worse than the same-basin
soup. Weight averaging needs a shared loss basin. Logit averaging does not - it
needs only a shared OUTPUT space, which these models have because they share one
tokenizer and one 15-class label set.

This driver mirrors diacritizer/scripts/bench.py bucket batching EXACTLY (same
length sort, same bool padding mask with True = KEEP, same chunks_of over ctx)
but keeps the fp32 logits instead of taking the argmax, averages them across the
member models, and only then argmaxes. Averaging is in fp32, never fp16, so the
merge is not quantised.

Usage:
  & .venv/Scripts/python.exe research/verdict_audit/tools/e78_logit_ens.py \
      --ckpt runs/diac/e73_soup/c4b_uniform \
      --ckpt runs/diac/e73_soup/c4_uniform \
      --config runs/diac/e67_gen/c4b_ctx256.yaml \
      --src abdou_test --out runs/diac/e78_logitens/L1/abdou_test.txt
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402
import bench as B  # noqa: E402

SOURCES = ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024",
           "abdou_test"]


def load_model(ckpt, cfg, device):
    ck = Path(ckpt)
    if ck.is_file():
        # E-078 L2: a member may be a raw probe wrapper
        # (gate_probe/weights_step*.pt = {model, step, config}) rather than a
        # checkpoint directory. The --ckpt help always promised this.
        state_p = ck
    else:
        state_p = ck / "state.pt" if (ck / "state.pt").exists() else ck / "model.pt"
    state = torch.load(state_p, map_location=device, weights_only=False)
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="member checkpoint dir or .pt; repeat for each member")
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True, choices=SOURCES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    device = args.device
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    models = [load_model(c, cfg, device) for c in args.ckpt]
    n_params = sum(p.numel() for p in models[0].parameters())

    pairs = []
    for bare, refs in B.load_pairs(args.src, ctx):
        pairs.append((bare, refs))
        if args.limit and len(pairs) >= args.limit:
            break

    metas = []
    keep = []
    for si, (bare, _) in enumerate(pairs):
        pieces = B.chunks_of(bare, ctx)
        keep.append(pieces)
        for pi, pc in enumerate(pieces):
            metas.append((si, pi, TK.encode(pc)))
    order = sorted(range(len(metas)), key=lambda i: len(metas[i][2]))
    out_pieces = [None] * len(metas)

    cm = (torch.amp.autocast("cuda", dtype=torch.float16)
          if device == "cuda" else contextlib.nullcontext())
    for k in range(0, len(order), args.batch):
        group = order[k:k + args.batch]
        if not group:
            continue
        T = max(len(metas[i][2]) for i in group)
        x = torch.zeros(len(group), T, dtype=torch.long, device=device)
        mask = torch.zeros(len(group), 1, 1, T, dtype=torch.bool, device=device)
        for r0, i in enumerate(group):
            L0 = len(metas[i][2])
            x[r0, :L0] = torch.tensor(metas[i][2], dtype=torch.long)
            mask[r0, 0, 0, :L0] = True
        acc = None
        with torch.no_grad(), cm:
            for m in models:
                lg = m(x, attn_mask=mask)["logits"].float()
                acc = lg if acc is None else acc + lg
        acc = acc / len(models)
        arg = acc.argmax(-1).tolist()
        for r0, i in enumerate(group):
            si, pi, _ = metas[i]
            out_pieces[i] = B.diacritize_piece(keep[si][pi], arg[r0])

    preds = []
    ptr = 0
    for pieces in keep:
        n_p = len(pieces)
        preds.append("".join(out_pieces[ptr + q] for q in range(n_p)))
        ptr += n_p

    refs_all = ["\t".join(refs) for _, refs in pairs]
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(preds) + "\n", encoding="utf-8")
    p.with_suffix(".ref.txt").write_text("\n".join(refs_all) + "\n",
                                         encoding="utf-8")
    print(json.dumps({
        "tag": args.tag, "src": args.src, "sentences": len(pairs),
        "members": args.ckpt, "n_members": len(models), "params": n_params,
        "ctx": ctx, "batch": args.batch, "device": device,
        "out": str(p),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
