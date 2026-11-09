"""E-083 - WEIGHTED cross-basin LOGIT ensemble (ONE pass, many weightings).

E-078 established that merging the c4b and c4 basins at the LOGIT level recovers
20.9 percent relative against the SAME merge at the WEIGHT level (all24_uniform
0.449353 -> L2 0.355179). But the uniform 24-probe logit mean (L2) still LOST to
the best same-basin soup (c4b_uniform 0.352209 / 0.349044) and to L1
(0.351324 / 0.350703).

The recorded hypothesis for why: L2 weights all 24 members uniformly, so it gives
HALF its mass to the weaker c4 basin and equal weight to every weak early probe.
L1 instead averages two SAME-BASIN soups first (a within-basin ensemble followed
by a cross-basin merge).

This driver tests that in ONE GPU pass: each member's forward pass is computed
once and every DECLARED weight vector is accumulated from those same fp32 logits,
so N weightings cost one 24-member pass (~12.5 min), not N passes. Averaging is
fp32, never fp16.

Bucketing mirrors bench.py EXACTLY (same length sort, same bool padding mask with
True = KEEP, same chunks_of over ctx), like e78_logit_ens.py.

Usage:
  & .venv/Scripts/python.exe research/verdict_audit/tools/e83_logit_weighted.py \
      --config runs/diac/e67_gen/c4b_ctx256.yaml \
      --src abdou_test --out-root runs/diac/e83_weighted
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

STEPS = [5000, 10000, 15000, 20000, 25000, 30000,
         35000, 40000, 45000, 50000, 55000, 60000]
C4B = ["runs/diac/e70_c4b_ctx256/gate_probe/weights_step%d.pt" % s for s in STEPS]
C4 = ["runs/diac/e69_c4_ctx256/gate_probe/weights_step%d.pt" % s for s in STEPS]
MEMBERS = C4B + C4


def basin_weights(mass_c4b, ramp):
    """mass_c4b goes to the 12 c4b probes, the rest to the 12 c4 probes.

    ramp=False: uniform over steps inside a basin (flat).
    ramp=True : linear in step (later probes weigh more), normalised.
    """
    r = [float(j + 1) for j in range(len(STEPS))] if ramp else [1.0] * len(STEPS)
    tot = sum(r)
    w = [x / tot for x in r]
    return [mass_c4b * x for x in w] + [(1.0 - mass_c4b) * x for x in w]


# DECLARED BEFORE LAUNCH. base24 is the instrument self-check (it must reproduce
# E-078 L2); c4b12/c412 are the within-basin logit-vs-weight-soup arms; b2/b4/b8
# are the basin admixtures; ramp_b1 is the only step-weighted arm.
SCHEMES = {
    "base24":  basin_weights(0.5, False),
    "c4b12":   basin_weights(1.0, False),
    "c412":    basin_weights(0.0, False),
    "b2":      basin_weights(2.0 / 3.0, False),
    "b4":      basin_weights(4.0 / 5.0, False),
    "b8":      basin_weights(8.0 / 9.0, False),
    "ramp_b1": basin_weights(0.5, True),
}
for _n, _w in SCHEMES.items():
    assert len(_w) == len(MEMBERS), _n
    assert abs(sum(_w) - 1.0) < 1e-9, (_n, sum(_w))


def load_model(ckpt, cfg, device):
    p = Path(ckpt)
    if not p.is_file():
        raise SystemExit("E-083 expects raw probe files, got %s" % p)
    state = torch.load(p, map_location=device, weights_only=False)
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--schemes", default="")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    names = [s for s in args.schemes.split(",") if s] or list(SCHEMES)
    for n in names:
        if n not in SCHEMES:
            ap.error("unknown scheme %r" % n)

    device = args.device
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    models = [load_model(c, cfg, device) for c in MEMBERS]
    n_params = sum(p.numel() for p in models[0].parameters())

    pairs = []
    for bare, refs in B.load_pairs(args.src, ctx):
        pairs.append((bare, refs))
        if args.limit and len(pairs) >= args.limit:
            break

    metas, keep = [], []
    for si, (bare, _) in enumerate(pairs):
        pieces = B.chunks_of(bare, ctx)
        keep.append(pieces)
        for pi, pc in enumerate(pieces):
            metas.append((si, pi, TK.encode(pc)))
    order = sorted(range(len(metas)), key=lambda i: len(metas[i][2]))
    out_pieces = {n: [None] * len(metas) for n in names}

    cm = (torch.amp.autocast("cuda", dtype=torch.float16)
          if device == "cuda" else contextlib.nullcontext())
    wmap = {n: SCHEMES[n] for n in names}
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
        accs = {n: None for n in names}
        with torch.no_grad(), cm:
            for mi, m in enumerate(models):
                # skip a member only when EVERY selected scheme gives it zero mass
                if all(wmap[n][mi] == 0.0 for n in names):
                    continue
                lg = m(x, attn_mask=mask)["logits"].float()
                for n in names:
                    wi = wmap[n][mi]
                    if wi == 0.0:
                        continue
                    accs[n] = (lg * wi) if accs[n] is None else (accs[n] + lg * wi)
        for n in names:
            arg = accs[n].argmax(-1).tolist()
            for r0, i in enumerate(group):
                si, pi, _ = metas[i]
                out_pieces[n][i] = B.diacritize_piece(keep[si][pi], arg[r0])

    refs_all = ["\t".join(refs) for _, refs in pairs]
    for n in names:
        preds, ptr = [], 0
        for pieces in keep:
            n_p = len(pieces)
            preds.append("".join(out_pieces[n][ptr + q] for q in range(n_p)))
            ptr += n_p
        p = Path(args.out_root) / n / (args.src + ".single.txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(preds) + "\n", encoding="utf-8")
        p.with_suffix(".ref.txt").write_text("\n".join(refs_all) + "\n", encoding="utf-8")
        print(json.dumps({
            "tag": "E083_" + n, "scheme": n, "src": args.src,
            "sentences": len(pairs), "n_members": len(models),
            "n_members_used": sum(1 for mi in range(len(models))
                                  if any(wmap[x][mi] != 0.0 for x in names)),
            "params": n_params, "ctx": ctx, "batch": args.batch,
            "device": device, "out": str(p),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
