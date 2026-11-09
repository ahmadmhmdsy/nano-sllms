"""Does the model under-emit on its OWN training lines? (step 4, E-99 question)

E-97 showed the failure mode is under-emission (65% of edits are deletions) and E-99
localised it to small glyphs. That leaves two explanations:

  (a) DATA  - the labels contain characters that are not visible in the image
              (orthographic annotation vs calligraphic rendering), so CTC optimum is
              to drop them; the model under-emits even where it was trained.
  (b) MODEL - the labels are learnable but the model fails to generalise; it emits
              ~100% on its own training lines and under-emits only off-distribution.

The discriminator is the emission ratio measured ON THE TRAINING SPLIT, with the
held-out validation split as the control. Preprocessing is imported from
finetune_muharaf so the comparison is apples-to-apples with how the model was trained.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arabic_ocr" / "scripts"))

from finetune_muharaf import (CKPT_PATH, TARGET_H, collate, encode_label,  # noqa: E402
                              estimate_width, load_ctc_backbone, norm_ar,
                              edit_distance, preprocess_image, ImageConfig)
from submission_code.ocr_helpers import greedy_ctc_decode                  # noqa: E402

SPLITS = {"muharaf_train": ROOT / "data" / "hf" / "muharaf" / "train",
          "muharaf_val": ROOT / "data" / "hf" / "muharaf" / "validation",
          "muharaf_test": ROOT / "data" / "hf" / "muharaf" / "test"}


def stride_sample(labels, n):
    """Evenly strided sample, not a prefix - a prefix would follow file order."""
    if n and len(labels) > n:
        step = len(labels) / float(n)
        return [labels[int(i * step)] for i in range(n)]
    return labels


def build(charset, folder: Path, n: int, target_h: int = TARGET_H):
    labels = stride_sample(json.loads((folder / "labels.json").read_text(encoding="utf-8")), n)
    items, skipped = [], 0
    for rec in labels:
        ids, _repaired = encode_label(charset, rec.get("text", ""))
        if ids is None:
            skipped += 1
            continue
        items.append({"path": str(folder / rec["image"]), "ids": ids,
                      "est_w": estimate_width(rec, folder, target_h), "raw": rec.get("text", "")})
    return items, skipped


@torch.no_grad()
def run_split(model, charset, items, device, batch, delta, target_h: int = TARGET_H):
    refs, hyps, n_edit, n_ref, n_edit_norm, n_ref_norm, n_raw, n_edit_raw = [], [], 0, 0, 0, 0, 0, 0
    t0 = time.time()
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        tensors = []
        for it in chunk:
            with Image.open(it["path"]) as im:
                arr = preprocess_image(im, ImageConfig(target_height=target_h,
                                                       grayscale=True, normalize=True))
            tensors.append({"img": torch.from_numpy(arr[None].astype("float32")),
                            "ids": torch.tensor(it["ids"], dtype=torch.long),
                            "w": int(arr.shape[1])})
        b = collate(tensors)
        widths = b["input_widths"]
        logits = model(b["images"].to(device), input_widths=widths.to(device))
        if delta:
            logits[:, :, charset.blank_id] -= delta
        lens = model.output_lengths(widths)
        for i, it in enumerate(chunk):
            n = int(lens[i])
            # CNNTransformerCTC.forward returns (T, B, C) - see its permute(1, 0, 2)
            ids = logits[:n, i, :].argmax(-1).tolist()
            hyp = charset.decode(greedy_ctc_decode(ids, blank_id=charset.blank_id))
            ref = charset.decode(it["ids"])          # the label as TRAINED
            hyps.append(hyp)
            refs.append(ref)
            n_edit += edit_distance(hyp, ref)
            n_ref += len(ref)
            n_edit_norm += edit_distance(norm_ar(hyp), norm_ar(it["raw"]))
            n_ref_norm += len(norm_ar(it["raw"]))
            n_raw += len(it["raw"])
            n_edit_raw += edit_distance(hyp, it["raw"])
    emit = 100.0 * sum(len(h) for h in hyps) / max(1, sum(len(r) for r in refs))
    raw_emit = 100.0 * sum(len(h) for h in hyps) / max(1, n_raw)
    return {"n": len(items), "emit_vs_repaired_ref": round(emit, 1),
            "emit_vs_raw_ref": round(raw_emit, 1),
            "cer_vs_repaired_ref": round(100.0 * n_edit / max(1, n_ref), 2),
            "cer_raw": round(100.0 * n_edit_raw / max(1, n_raw), 2),
            "cer_norm": round(100.0 * n_edit_norm / max(1, n_ref_norm), 2),
            "secs": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda. An explicit cpu is HONOURED - the old "
                         "code silently used CUDA whenever it existed, so a CPU-only "
                         "run would have competed with a live training run for the GPU")
    ap.add_argument("--target-height", type=int, default=0,
                    help="0 = the 96 default. MUST match the height the checkpoint "
                         "was trained at, or the probe measures the wrong geometry")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--deltas", default="0,1.0")
    ap.add_argument("--splits", default="muharaf_train,muharaf_val,muharaf_test")
    ap.add_argument("--self-check", action="store_true",
                    help="run the FULL muharaf_test at delta 0 and verify it reproduces "
                         "the cached-logit numbers (raw CER 22.11, emission 86.7%)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    target_h = args.target_height or TARGET_H
    model, charset, _cfg = load_ctc_backbone(str(args.ckpt), device=device)
    model.eval()
    print("ckpt %s  device %s  height %d  n/split %d" % (args.ckpt, device, target_h, args.n),
          flush=True)

    if args.self_check:
        if target_h != 96:
            raise SystemExit("--self-check calibrates against height-96 numbers; "
                             "do not combine it with --target-height")
        items, _ = build(charset, SPLITS["muharaf_test"], 0, target_h)
        res = run_split(model, charset, items, device, args.batch, 0.0, target_h)
        ok = abs(res["cer_raw"] - 22.11) <= 0.30 and abs(res["emit_vs_raw_ref"] - 86.7) <= 1.5
        print("SELF-CHECK muharaf_test full: raw CER %.2f (expect 22.11), emit %.1f%% "
              "(expect 86.7) -> %s" % (res["cer_raw"], res["emit_vs_raw_ref"],
                                       "PASS" if ok else "FAIL"), flush=True)
        if not ok:
            raise SystemExit("self-check failed - batched probe disagrees with the cache path")

    report = {}
    for name in [s for s in args.splits.split(",") if s]:
        items, skipped = build(charset, SPLITS[name], args.n, target_h)
        report[name] = {"skipped_unencodable": skipped}
        for delta in [float(x) for x in args.deltas.split(",") if x != ""]:
            res = run_split(model, charset, items, device, args.batch, delta, target_h)
            report[name]["delta_%.2f" % delta] = res
            print("  %-14s delta %.2f  emit_raw %5.1f%%  emit_vs_trained_ref %5.1f%%  "
                  "CER(raw) %5.2f  CER(trained ref) %5.2f  norm %5.2f  (%.1fs)"
                  % (name, delta, res["emit_vs_raw_ref"], res["emit_vs_repaired_ref"],
                     res["cer_raw"], res["cer_vs_repaired_ref"], res["cer_norm"],
                     res["secs"]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
