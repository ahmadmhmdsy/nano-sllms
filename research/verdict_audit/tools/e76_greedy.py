"""E-076 - greedy (Caruana) weight-soup selection over the 12 C4b probes.

Selection objective: der_registered on the abdou_test DEV subsample (--limit),
the dev set E-074 validated as a rank proxy for the registered gates
(Spearman rho +0.909, 94.4 percent of the 0.963 ceiling). The registered gates
are NEVER consulted here to choose, weight, or stop the search; they are read
once afterwards by the E-075 voting procedure for the reported column.

CPU builds each candidate soup in memory (fp32 weighted mean of member state
dicts, cast back per-key to the member dtype); the GPU only does single-pass
inference. No training, and no soup file per candidate.

Pre-registration: research/experiments.yaml -> diacritizer/E-076.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]  # tools -> verdict_audit -> research -> root
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))

from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402
import bench as B  # noqa: E402
import eval_metrics as em  # noqa: E402


def probe_files(run):
    gp = REPO / "runs" / "diac" / run / "gate_probe"
    return sorted(gp.glob("weights_step*.pt"),
                  key=lambda p: int(p.stem.split("step")[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e70_c4b_ctx256")
    ap.add_argument("--config", default="runs/diac/e67_gen/c4b_ctx256.yaml")
    ap.add_argument("--src", default="abdou_test")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="runs/diac/e76_greedy")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    device = "cuda"

    paths = probe_files(args.run)
    if not paths:
        raise SystemExit("no probes found for " + args.run)
    members = [torch.load(p, map_location="cpu", weights_only=False)["model"]
               for p in paths]
    steps = [int(p.stem.split("step")[1]) for p in paths]
    print("members: %d  steps %s" % (len(steps), steps), flush=True)

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.eval()

    pairs = []
    for bare, refs in B.load_pairs(args.src, ctx):
        pairs.append((bare, refs))
        if args.limit and len(pairs) >= args.limit:
            break
    refs_all = ["\t".join(r) for _, r in pairs]
    print("dev %s: %d sentences (limit %s)"
          % (args.src, len(pairs), args.limit), flush=True)

    keys = list(members[0].keys())
    fkeys = [k for k in keys if members[0][k].is_floating_point()]
    okeys = [k for k in keys if not members[0][k].is_floating_point()]
    if okeys:
        print("non-float keys carried from member 0:", okeys, flush=True)

    def build(counts):
        tot = float(sum(counts))
        acc = {k: torch.zeros_like(members[0][k], dtype=torch.float32)
               for k in fkeys}
        for w, sd in zip(counts, members):
            if w == 0:
                continue
            for k in fkeys:
                acc[k] += sd[k].to(torch.float32) * w
        out_sd = {k: (acc[k] / tot).to(members[0][k].dtype) for k in fkeys}
        for k in okeys:
            out_sd[k] = members[0][k].clone()
        return out_sd

    def score(counts):
        model.load_state_dict(build(counts), strict=True)
        preds = B.predict_bares(model, pairs, ctx, device, batch=args.batch)
        sm = em.summarize(em.score_corpus(preds, refs_all))
        return float(sm["der_registered"])

    counts = [0] * len(members)
    trace = []
    for rnd in range(1, args.rounds + 1):
        cand = []
        for i in range(len(members)):
            trial = list(counts)
            trial[i] += 1
            d = score(trial)
            cand.append(d)
            print("  round %d  probe %6d  der %.6f" % (rnd, steps[i], d),
                  flush=True)
        best = min(range(len(cand)), key=lambda i: cand[i])
        counts[best] += 1
        trace.append({"round": rnd, "picked_step": steps[best],
                      "picked_der": cand[best], "candidates": cand})
        print("ROUND %d PICK step %d  der %.6f  counts %s"
              % (rnd, steps[best], cand[best], counts), flush=True)

    out = REPO / args.out
    (out / "greedy").mkdir(parents=True, exist_ok=True)
    torch.save({"model": build(counts),
                "step": "soup_greedy_%d" % sum(counts), "config": cfg},
               out / "greedy" / "model.pt")
    (out / "weights.json").write_text(json.dumps({
        "run": args.run, "src": args.src, "limit": args.limit,
        "rounds": args.rounds, "steps": steps, "counts": counts,
        "weights": [c / sum(counts) for c in counts], "trace": trace,
        "final_dev_der": trace[-1]["picked_der"],
    }, indent=2), encoding="utf-8")
    print("saved " + str(out / "greedy" / "model.pt"), flush=True)
    print("final counts " + str(counts), flush=True)


if __name__ == "__main__":
    main()
