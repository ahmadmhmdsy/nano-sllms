"""E-084 triage bed - few-minute SCREEN scorer for ONE arm.

*** THIS IS A SCREEN, NOT A FULL TRAINING RUN. ***

Every artifact this script writes carries status SCREEN-NOT-FULL-TRAIN, for one
reason: a screen trains for a few minutes at a tiny fixed step budget and one
seed. It can hard-KILL a catastrophic arm, prove smoke validity and measure
speed. It canNOT rank a sub-2-percent effect, because the trainer's own 5.72
percent probe band and the unseeded init draw dominate anything that small.
Anything inside +/-2 percent relative is INCONCLUSIVE and must never be promoted
on a screen alone. A promotion still needs a full pre-registered run.

What it does, per arm:
  * SELECTOR  = abdou_test stride slice, scored at EVERY surviving checkpoint
                (der_registered is the registered metric; lower is better).
                The selector decides; it is the dev selector per house rule.
  * DIAGNOSTICS = fadel_test / sadeed25 / wikinews2014 at the FINAL checkpoint
                only, printed for context and NEVER used for promotion - the
                registered gates stay score-once.
  * VERDICT   = paired delta against a --baseline control screen of the SAME
                seed and SAME step budget, plus a McNemar paired test on the
                selector when the control's predictions are on disk.

Decode path is bench.py itself (import bench), so the screen sees the exact
registered chunking / argmax / mark-insertion routing. No reimplementation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
sys.path.insert(0, str(REPO / "diacritizer" / "src"))

import torch                                                    # noqa: E402
import yaml                                                     # noqa: E402
from model import build_from_config                             # noqa: E402
import tokenizer as TK                                          # noqa: E402
import bench as B                                               # noqa: E402
import eval_metrics as em                                       # noqa: E402

STAMP = "SCREEN-NOT-FULL-TRAIN"
DISCLAIMER = (
    "FEW-MINUTE SCREEN, NOT A FULL TRAINING RUN. Small fixed step budget, single "
    "seed, paired against a same-seed control. A screen hard-KILLS and measures "
    "speed only; anything inside +/-2 percent relative is INCONCLUSIVE and must "
    "not be promoted on a screen. A promotion needs a full pre-registered run."
)
RULE = (
    "SELECTOR only. rel = (arm_der - control_der)/control_der at matched steps. "
    "KILL if rel >= +0.02 at the final matched step. PROMOTE if rel <= -0.02 AND "
    "the last two matched steps agree in sign. INCONCLUSIVE otherwise (this "
    "includes every result inside the +/-2 percent band). Diagnostic corpora are "
    "printed for context only and never decide."
)
DEGEN_ANYALT = 0.7554      # e67_p6_vote.py collapse signature
DEGEN_F1 = 0.0953          # e67_p6_vote.py collapse signature
PROMOTE_REL = -0.02
KILL_REL = 0.02
DIAG_SRC = ("fadel_test", "sadeed25", "wikinews2014")


def _lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def slice_pairs(src, ctx, stride, limit):
    """Stride-slice then cap. stride=1 limit=None means the whole corpus."""
    want = (limit * stride) if (limit and stride and stride > 1) else (limit or 0)
    raw = []
    for bare, ref in B.load_pairs(src, ctx):
        raw.append((bare, ref))
        if want and len(raw) >= want:
            break
    raw = raw[::stride]
    return raw[:limit] if limit else raw


def load_model(cfg, device, ckpt_file):
    state = torch.load(ckpt_file, map_location=device, weights_only=False)
    sd = state["model"] if (isinstance(state, dict) and "model" in state) else state
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def score_pairs(model, pairs, ctx, device, batch, pred_out=None, ref_out=None):
    t0 = time.time()
    preds = B.predict_bares(model, pairs, ctx, device, batch)
    refs = ["\t".join(r) for _, r in pairs]
    if pred_out:
        Path(pred_out).write_text("\n".join(preds) + "\n", encoding="utf-8")
        Path(ref_out).write_text("\n".join(refs) + "\n", encoding="utf-8")
    sm = em.summarize(em.score_corpus(preds, refs))
    out = {"der_registered": sm.get("der_registered"),
           "der_canonical": sm.get("der_canonical"),
           "der_anyalt": sm.get("der_anyalt"),
           "macro_f1": sm.get("macro_f1"),
           "lines": len(preds), "seconds": round(time.time() - t0, 1)}
    degen = []
    if out["der_anyalt"] is not None and abs(out["der_anyalt"] - DEGEN_ANYALT) < 1e-3:
        degen.append("der_anyalt~0.7554")
    if out["macro_f1"] is not None and abs(out["macro_f1"] - DEGEN_F1) < 1e-3:
        degen.append("macro_f1~0.0953")
    if degen:
        out["DEGENERATE"] = degen
    return out, preds, refs


def mcnemar_p(a_pred, b_pred, refs, nocase=False):
    """Paired significance on the identical selector slice. Never fatal."""
    try:
        res = em.mcnemar(a_pred, b_pred, refs, nocase=nocase)
        return json.loads(json.dumps(res, default=str))
    except Exception as exc:                                   # noqa: BLE001
        return {"error": str(exc)[-200:]}


def find_steps(rundir):
    steps = []
    for d in sorted(Path(rundir).glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1])):
        sp = d / "state.pt"
        if sp.exists():
            steps.append((int(d.name.split("-")[-1]), sp))
    return steps


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True,
                    help="run dir holding checkpoint-<step>/state.pt (rotation keeps 3)")
    ap.add_argument("--model", default=None,
                    help="explicit single checkpoint/state dict to score as one point")
    ap.add_argument("--config", required=True, help="arch yaml (must match the run)")
    ap.add_argument("--out", required=True, help="screen JSON to write")
    ap.add_argument("--arm", required=True, help="arm label")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--baseline", default=None,
                    help="control screen JSON from the SAME seed+steps (paired delta)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--diag-limit", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ctx = cfg["model"]["ctx"]
    device = args.device

    points = find_steps(args.rundir)
    if args.model:
        points = [(0, Path(args.model))] + points
    if not points:
        raise SystemExit("[SCREEN-FATAL] no checkpoints under " + str(args.rundir))

    t_all = time.time()
    sel = slice_pairs("abdou_test", ctx, args.stride, args.limit)
    print({"screen": STAMP, "arm": args.arm, "seed": args.seed,
           "selector_lines": len(sel), "points": [s for s, _ in points]}, flush=True)

    results, last_pred = [], None
    for step, ckf in points:
        model = load_model(cfg, device, ckf)
        pp = out_p.parent / ("sel_step%d.pred.txt" % step)
        rr = out_p.parent / ("sel_step%d.ref.txt" % step)
        sc, preds, refs = score_pairs(model, sel, ctx, device, args.batch, pp, rr)
        rec = {"step": step, "selector": sc, "selector_pred": str(pp),
               "selector_ref": str(rr)}
        if step == points[-1][0]:
            for src in DIAG_SRC:
                dpairs = slice_pairs(src, ctx, 1, args.diag_limit)
                dsc, _, _ = score_pairs(model, dpairs, ctx, device, args.batch)
                rec.setdefault("diagnostics", {})[src] = dsc
            last_pred = (preds, refs)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        results.append(rec)
        print({"step": step, "sel_der": sc["der_registered"],
               "sel_der_can": sc["der_canonical"],
               "sec": sc["seconds"]}, flush=True)

    vd = {"rule": RULE, "label": "NO-BASELINE"}
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        bmap = {p["step"]: p for p in base.get("points", [])}
        rels = []
        for r in results:
            q = bmap.get(r["step"])
            if not q:
                continue
            a = r["selector"].get("der_registered")
            c = q["selector"].get("der_registered")
            if a and c:
                rels.append({"step": r["step"], "arm": a, "control": c,
                             "rel": (a - c) / c})
        vd["matched"] = rels
        if rels:
            tail = rels[-2:] if len(rels) >= 2 else rels
            last = rels[-1]["rel"]
            if last <= PROMOTE_REL and all(x["rel"] <= PROMOTE_REL for x in tail):
                vd["label"] = "PROMOTE"
            elif last >= KILL_REL:
                vd["label"] = "KILL"
            else:
                vd["label"] = "INCONCLUSIVE"
            bp = bmap.get(results[-1]["step"], {}).get("selector_pred")
            if bp and Path(bp).exists() and last_pred is not None:
                vd["mcnemar_selector"] = mcnemar_p(
                    last_pred[0], _lines(bp), last_pred[1])
                vd["mcnemar_selector"]["note"] = (
                    "paired, identical selector slice, arm=first vs control=second")
    vd["label_meaning"] = {
        "KILL": "arm is >=2 percent relative worse at matched steps - drop it",
        "PROMOTE": "arm is >=2 percent relative better at 2 matched steps - "
                   "still needs a FULL pre-registered run before any promotion",
        "INCONCLUSIVE": "inside the +/-2 percent band - promote NOTHING from this",
        "NO-BASELINE": "no same-seed control supplied - screen cannot judge",
    }

    diag_mean = None
    d = results[-1].get("diagnostics") or {}
    vals = [d[s]["der_registered"] for s in DIAG_SRC if s in d
            and d[s].get("der_registered") is not None]
    if len(vals) == 3:
        diag_mean = sum(vals) / 3
        diag_mean = diag_mean if diag_mean < 0.05 else None   # score_vote_dir guard

    doc = {"status": STAMP, "disclaimer": DISCLAIMER, "arm": args.arm,
           "seed": args.seed, "config": str(args.config),
           "selector_corpus": "abdou_test stride=%d limit=%d (%d lines)"
                              % (args.stride, args.limit, len(sel)),
           "diagnostics_note": "context only - never decides, never promoted",
           "gate_mean3_diagnostic": diag_mean,
           "points": results, "verdict": vd,
           "total_seconds": round(time.time() - t_all, 1)}
    out_p.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print("=" * 72)
    print("[%s] %s" % (STAMP, DISCLAIMER))
    print("verdict: %s" % vd["label"])
    if vd.get("matched"):
        print("matched steps: " + json.dumps(vd["matched"]))
    if vd.get("mcnemar_selector"):
        print("mcnemar: " + json.dumps(vd["mcnemar_selector"]))
    print("wrote %s in %.1fs" % (out_p, doc["total_seconds"]))


if __name__ == "__main__":
    main()
