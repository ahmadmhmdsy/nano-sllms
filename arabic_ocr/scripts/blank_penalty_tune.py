"""Tune the CTC blank penalty (E-98) on muharaf_val, then apply it to every other set.

Discipline: delta is selected ONLY on data/hf/muharaf/validation logits and then
applied unchanged to the held-out sets. CPU only - it reads the cached log-softmax
npz written by decode_ctc_lm.py --cache, so a sweep never touches the GPU.

Reports raw CER, norm_ar CER (E-89/E-99) and the emitted-character ratio. The
emission ratio is the diagnostic that matters here: E-97 showed the failure mode
is under-emission (65% of all edits are deletions), and the penalty trades
deletions against insertions, so a CER win with no emission move would be suspect.

Usage:
  python arabic_ocr/scripts/blank_penalty_tune.py --tags teacher,adapt11700,stage05f
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "arabic_ocr" / "models" / "muharaf_ocr"
sys.path.insert(0, str(MODEL_DIR))

from submission_code.ocr_helpers import greedy_ctc_decode          # noqa: E402
from submission_code.torch_models import load_ctc_backbone          # noqa: E402

DEFAULT_CKPT = MODEL_DIR / "models" / "ctc_backbone" / "model.pt"
DEFAULT_LOGITS = ROOT / "arabic_ocr" / "runs" / "stage05e" / "logits"
DEV = "muharaf_val"
ORDER = ["muharaf_val", "muharaf_test", "khatt_teklia_test", "isippt_bench", "khatt_bench"]
DELTAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def _load_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_M = _load_sibling("ocr_eval_metrics", Path(__file__).with_name("eval_muharaf_hflines.py"))
edit_distance, norm_ar = _M.edit_distance, _M.norm_ar


def score(path: Path, charset, delta: float) -> dict:
    """Greedy CTC over one cached set with the blank logit shifted by -delta."""
    npz = np.load(path, allow_pickle=False)
    data, offs, refs = npz["data"], npz["offsets"], [str(r) for r in npz["refs"]]
    logp = data.astype(np.float32)
    if delta:
        logp[:, charset.blank_id] -= delta
    hyps = []
    for i in range(len(offs) - 1):
        raw = np.argmax(logp[offs[i]:offs[i + 1]], axis=-1).tolist()
        hyps.append(charset.decode(greedy_ctc_decode(raw, blank_id=charset.blank_id)))
    nref = max(1, sum(len(r) for r in refs))
    nref_n = max(1, sum(len(norm_ar(r)) for r in refs))
    return {"cer": round(100.0 * sum(edit_distance(h, r) for h, r in zip(hyps, refs)) / nref, 2),
            "cer_norm": round(100.0 * sum(edit_distance(norm_ar(h), norm_ar(r))
                                          for h, r in zip(hyps, refs)) / nref_n, 2),
            "emit": round(100.0 * sum(len(h) for h in hyps) / nref, 1),
            "n": len(refs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits-dir", type=Path, default=DEFAULT_LOGITS)
    ap.add_argument("--tags", default="teacher,adapt11700")
    ap.add_argument("--deltas", default=",".join(str(d) for d in DELTAS))
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",") if x != ""]
    _, charset, _ = load_ctc_backbone(str(args.ckpt), device="cpu")
    report = {}
    for tag in [x for x in args.tags.split(",") if x]:
        d = args.logits_dir / tag
        vp = d / (DEV + ".npz")
        if not vp.exists():
            print("%s: no %s cache - skipped" % (tag, DEV), flush=True)
            continue
        sweep = {delta: score(vp, charset, delta) for delta in deltas}
        best = min(deltas, key=lambda x: sweep[x]["cer"])
        print("\n%s  %s sweep: %s" % (tag, DEV, " ".join(
            "d=%.2f:%.2f" % (x, sweep[x]["cer"]) for x in deltas)), flush=True)
        print("%s -> chosen delta %.2f (val raw %.2f norm %.2f emit %.1f%%)"
              % (tag, best, sweep[best]["cer"], sweep[best]["cer_norm"], sweep[best]["emit"]),
              flush=True)
        rows = {}
        for st in ORDER:
            p = d / (st + ".npz")
            if not p.exists():
                continue
            a, b = score(p, charset, 0.0), score(p, charset, best)
            rows[st] = {"delta0": a, "delta_star": b,
                        "delta_pp": round(b["cer"] - a["cer"], 2)}
            print("   %-18s raw %5.2f -> %5.2f (%+.2f)  norm %5.2f -> %5.2f  emit %5.1f%% -> %5.1f%%"
                  % (st, a["cer"], b["cer"], b["cer"] - a["cer"],
                     a["cer_norm"], b["cer_norm"], a["emit"], b["emit"]), flush=True)
        report[tag] = {"chosen_delta": best, "dev": DEV, "sweep": sweep, "sets": rows}
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
