"""E-085 probe: does EXCEEDING the trained context help? (inference only)

WHY. bench.chunks_of() splits a sentence into HARD NON-OVERLAPPING pieces of
exactly ctx chars: [(bare[o:o+size]) for o in range(0, len(bare), size)]. A
sentence longer than ctx therefore never has its per-position context seen
across the cut, and this is not a corner case:

    fadel_test    30.0 pct of sentences, 68 pct of all scored characters
    sadeed25      29.5 pct of sentences, 60 pct of characters
    wikinews2014  38.7 pct of sentences, 67 pct of characters
    wikinews2024  13.5 pct of sentences, 28 pct of characters
    abdou_test    63.3 pct of sentences, 93 pct of characters   (max 2234)

E-077 swept the inference window and found shrinking it hurts MONOTONICALLY
(256 > 192 > 128 on abdou), i.e. more context is better for this model. But
E-077 bounded itself explicitly: "Every window is strictly <= ctx, so no arm
exceeds the model's trained context." The direction the monotone trend points
at - LONGER than the trained ctx - is the one arm never run.

RoPE (model.rope_freqs) is computed per forward from seq_len, and there are no
learned positional parameters, so a ctx512/768 build loads the ctx256 anchor
with ZERO key mismatch. This probe is therefore training-free and exact: a full
corpus scored through the REGISTERED metric, not a screen.

THE HONEST CAVEAT, pre-registered: this reads positions 256..767 that the model
never trained on, so it measures RoPE EXTRAPOLATION, not a trained longer ctx. A
loss here does NOT close a trained ctx-extension; a WIN here does NOT promote
the deployed model - it justifies the TRAINED ctx-extension run as the next arm.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402
import bench as B  # noqa: E402
import eval_metrics as em  # noqa: E402

SRCS = ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024", "abdou_test"]


def score(pred, ref):
    sm = em.summarize(em.score_corpus(pred, ref))
    return {"der_registered": sm.get("der_registered"),
            "der_anyalt": sm.get("der_anyalt"),
            "macro_f1": sm.get("macro_f1")}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/diac/e73_soup/c4b_uniform/model.pt")
    ap.add_argument("--config", default="runs/diac/e67_gen/c4b_ctx256.yaml")
    ap.add_argument("--refdir", default="runs/diac/e75_vote")
    ap.add_argument("--srcs", default=",".join(SRCS))
    ap.add_argument("--ctxs", default="256,512,768")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/diac/_screen_out/e85_ctx_extrapolation.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(REPO / args.config).read_text(encoding="utf-8"))
    device = args.device
    state = torch.load(REPO / args.ckpt, map_location=device, weights_only=False)
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"status": "EXACT-CPU-GPU-CENSUS", "ckpt": args.ckpt,
                      "params": n_params, "batch": args.batch}), flush=True)

    res = {"status": "EXACT-FULL-CORPUS-NOT-A-SCREEN", "ckpt": args.ckpt,
           "config": args.config, "params": n_params,
           "trained_ctx": cfg["model"]["ctx"], "batch": args.batch, "corpora": {}}
    for src in [s for s in args.srcs.split(",") if s]:
        pairs = []
        for bare, refs in B.load_pairs(src, 256):
            pairs.append((bare, refs))
            if args.limit and len(pairs) >= args.limit:
                break
        # REFS COME FROM load_pairs, IN MEMORY. The on-disk
        # <corpus>.single.ref.txt CANNOT be read line-by-line: a gold field may
        # contain embedded newlines, so abdou_test's file holds 41378 PHYSICAL
        # lines for 15091 sentences. The registered scorer takes its refs from
        # load_pairs, and so must this probe. The file is kept only as a
        # cross-check, and only meaningful when the counts agree.
        ref = ["/".join(r) for _, r in pairs]
        rf = REPO / args.refdir / (src + ".single.ref.txt")
        rf_lines = rf.read_text(encoding="utf-8").splitlines() if rf.exists() else []
        rec = {"lines": len(pairs), "ref_in_memory": len(ref),
               "ref_file_lines": len(rf_lines),
               "ref_file_agrees": len(rf_lines) == len(pairs)}
        # instrument self-check: the registered ctx256 single pass must reproduce
        # the on-disk E-073/E-075 artifact, else the probe is INSTRUMENT-UNKNOWN.
        disk = (REPO / "runs/diac/e73_soup/c4b_uniform/pred" /
                (src + ".txt"))
        disk_lines = disk.read_text(encoding="utf-8").splitlines() if disk.exists() else []
        rec["chars"] = sum(len(b) for b, _ in pairs)
        rec["over_ctx"] = sum(1 for b, _ in pairs if len(b) > 256)
        rec["chars_over_ctx"] = sum(len(b) for b, _ in pairs if len(b) > 256)
        long_idx = [i for i, (b, _) in enumerate(pairs) if len(b) > 256]
        rec["trained_ctx"] = cfg["model"]["ctx"]
        rec["arms"] = {}
        for ctx in [int(c) for c in args.ctxs.split(",") if c]:
            t0 = time.time()
            pred = B.predict_bares(model, pairs, ctx, device, batch=args.batch)
            sec = round(time.time() - t0, 1)
            a = {"ctx": ctx, "seconds": sec,
                 "chars_per_s": round(rec["chars"] / max(sec, 1e-9), 1),
                 "full": score(pred, ref)}
            if long_idx:
                a["over_ctx_only"] = score([pred[i] for i in long_idx],
                                           [ref[i] for i in long_idx])
                a["over_ctx_lines"] = len(long_idx)
            rec["arms"]["ctx%d" % ctx] = a
            if ctx == 256 and disk_lines:
                same = sum(1 for p, d in zip(pred, disk_lines) if p == d)
                a["selfcheck_vs_disk_single"] = {
                    "matched": same, "of": min(len(pred), len(disk_lines)),
                    "byte_identical": same == len(pred) == len(disk_lines)}
            print(json.dumps({src: {"ctx": ctx, "seconds": sec,
                                    "der": a["full"]["der_registered"],
                                    "der_over": a.get("over_ctx_only", {}).get("der_registered"),
                                    "selfcheck": a.get("selfcheck_vs_disk_single")}},
                             ensure_ascii=False), flush=True)
        # reference arms read from disk, no re-run
        for nm, fn in (("disk_single_ctx256", args.refdir + "/%s.single.txt"),
                       ("disk_voted_256_128", args.refdir + "/%s.vote.txt")):
            p = REPO / (fn % src)
            if p.exists():
                rec["arms"][nm] = {
                    "full": score(p.read_text(encoding="utf-8").splitlines(), ref),
                    "note": "read from disk; the deployed E-075 reference"}
                if long_idx:
                    pl = p.read_text(encoding="utf-8").splitlines()
                    rec["arms"][nm]["over_ctx_only"] = score(
                        [pl[i] for i in long_idx], [ref[i] for i in long_idx])
        res["corpora"][src] = rec
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote " + str(out), flush=True)


if __name__ == "__main__":
    main()
