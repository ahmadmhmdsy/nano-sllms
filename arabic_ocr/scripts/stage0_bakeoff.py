
r"""Stage 0 bake-off - zero-shot FREE-RUNNING CER/WER for the two PaddleOCR
Arabic recognizers, against our recorded from-scratch control.

Design notes:
- Both candidate repos ship PaddleOCR *inference* format (inference.json +
  inference.pdiparams + inference.yml), so no paddleocr/paddlex install is
  needed - this loads them with paddle directly.
- Decoding is greedy CTC. That is free-running BY CONSTRUCTION: no ground-truth
  token is ever fed back, so this cannot repeat the teacher-forced mistake that
  made the earlier OCR numbers meaningless.
- The v6 model was trained with label_order: visual, so CER/WER are reported
  BOTH as-is and with the prediction reversed (the README's "mirror" question).
- Arabic is written to files, never judged from the console.

Usage:
  & .\.venv\Scripts\python.exe arabic_ocr\scripts\stage0_bakeoff.py --probe
  & .\.venv\Scripts\python.exe arabic_ocr\scripts\stage0_bakeoff.py --sets all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import paddle
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
MODELS = {
    "v5_mobile_rec": REPO / "arabic_ocr/models/ppocrv5_mobile_rec",
    "v6_small_rec": REPO / "arabic_ocr/models/ppocrv6_small_rec/inference",
}
SETS = {
    "hflines_val": (REPO / "data/hflines/val", "mixed: printed + handwritten"),
    "isippt": (REPO / "data/benchmark/isippt", "PRINTED (ISIPPT)"),
    "khatt": (REPO / "data/benchmark/khatt", "HANDWRITTEN (KHATT)"),
}
# Recorded from-scratch control (free-running, 2026-10-01) - no new compute.
CONTROL = {"isippt": 144.8, "khatt": 97.8}

DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640\u06D6-\u06ED]")


def norm_loose(s: str) -> str:
    s = DIACRITICS.sub("", s)
    s = s.replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
    s = s.replace("\u0649", "\u064A").replace("\u0629", "\u0647")
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_chars(yml_path: Path):
    """PaddleOCR CTC alphabet: ['blank'] + character_dict + [' '] (use_space_char).
    The exported model's class count is 749 = 1 + 747 + 1, which confirms the
    trailing space; omitting it silently drops every space from the decode."""
    cfg = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    chars = cfg["PostProcess"]["character_dict"]
    return ["blank"] + list(chars) + [" "]


def load_model(model_dir: Path):
    """Return (callable, kind, meta). callable takes [HWC-float32 CHW array]."""
    prefix = model_dir / "inference"
    try:
        m = paddle.jit.load(str(prefix))
        m.eval()
        return m, "jit", {}
    except Exception as exc:  # noqa: BLE001
        print("   jit.load failed: %s: %s" % (type(exc).__name__, exc))
    import paddle.inference as pi

    cfg = pi.Config(str(model_dir / "inference.json"),
                    str(model_dir / "inference.pdiparams"))
    cfg.disable_gpu()
    cfg.switch_ir_optim(True)
    pred = pi.create_predictor(cfg)
    meta = {
        "inputs": pred.get_input_names(),
        "outputs": pred.get_output_names(),
    }
    return pred, "predictor", meta


def call_model(model, kind, arr):
    if kind == "jit":
        out = model(arr)  # PirTranslatedLayer takes a numpy array, not a list
    else:
        names = model.get_input_names()
        handle = model.get_input_handle(names[0])
        handle.copy_from_cpu(arr)
        model.run()
        out = model.get_output_handle(model.get_output_names()[0]).copy_to_cpu()
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, paddle.Tensor):
        out = out.numpy()
    return np.asarray(out)


def preprocess(path: Path, img_h: int = 48, max_w: int = 3200):
    """PaddleOCR rec preprocessing: BGR, aspect-preserving resize to height 48,
    (x/255 - 0.5)/0.5, CHW. Width is dynamic (the exported models allow it)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if h <= 0 or w <= 0:
        raise ValueError("bad image size")
    resized_w = int(round(img_h * w / h))
    resized_w = max(8, min(max_w, resized_w))
    im = im.resize((resized_w, img_h), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
    arr = arr.transpose(2, 0, 1) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.ascontiguousarray(arr[None, ...])


def ctc_greedy(logits, chars):
    """logits: (B, T, C) -> list of strings. Greedy, repeat-collapsed, blank=0."""
    idx = logits.argmax(axis=2)
    out = []
    for row in idx:
        prev = -1
        cs = []
        for k in row:
            k = int(k)
            if k != prev and k != 0:
                if 0 <= k < len(chars):
                    cs.append(chars[k])
            prev = k
        out.append("".join(cs))
    return out


# Visual-order (mirrored) decodes must be un-mirrored before scoring: reversing
# the whole string is not enough, because bidi display keeps LTR runs (Latin
# words, numbers) left-to-right and swaps bracket pairs. This is the artifact the
# v6 README calls "decode-mirror" (13.5% strict vs 10.9% no-mirror on isippt).
MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{", "<": ">", ">": "<"}
LTR_RUN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.,:/+%\-]*")


def unmirror(s: str) -> str:
    r = s[::-1]
    out, i = [], 0
    while i < len(r):
        mt = LTR_RUN.match(r, i)
        if mt:
            out.append(mt.group(0)[::-1])
            i = mt.end()
        else:
            out.append(MIRROR.get(r[i], r[i]))
            i += 1
    return "".join(out)


def arabic_pct(s: str) -> float:
    if not s:
        return 0.0
    ar = sum(1 for ch in s if "؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ"
             or "ﭐ" <= ch <= "﷿" or "ﹰ" <= ch <= "﻿")
    return 100.0 * ar / len(s)


def load_set(name):
    d, note = SETS[name]
    data = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    items = []
    for r in data:
        img = r.get("image") or r.get("img") or r.get("file")
        txt = r.get("text", r.get("label", ""))
        items.append((d / img, txt))
    return items, note


def evaluate(model, kind, chars, items, limit=None):
    if limit:
        items = items[:limit]
    preds, refs, paths = [], [], []
    t0 = time.perf_counter()
    for i, (path, ref) in enumerate(items):
        paths.append(str(path))
        arr = preprocess(path)
        logits = call_model(model, kind, arr)
        if logits.ndim == 2:
            logits = logits[None, ...]
        preds.append(ctc_greedy(logits, chars)[0])
        refs.append(ref)
        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print("      %d/%d  %.1f img/s" % (i + 1, len(items), (i + 1) / el))
    elapsed = time.perf_counter() - t0
    m = {"n": len(items), "seconds": round(elapsed, 1),
         "img_per_s": round(len(items) / elapsed, 2)}
    for tag, fn in (("as_is", lambda s: s), ("reversed", lambda s: s[::-1]),
                    ("unmirrored", unmirror)):
        p = [fn(x) for x in preds]
        ed = sum(edit_distance(a, b) for a, b in zip(p, refs))
        rc = sum(len(r) for r in refs) or 1
        ew = sum(edit_distance(a.split(), b.split()) for a, b in zip(p, refs))
        rw = sum(max(1, len(r.split())) for r in refs) or 1
        m["cer_" + tag] = round(100.0 * ed / rc, 2)
        m["wer_" + tag] = round(100.0 * ew / rw, 2)
        m["exact_" + tag] = round(100.0 * sum(a == b for a, b in zip(p, refs)) / max(1, len(refs)), 2)
        m["cer_loose_" + tag] = round(
            100.0 * sum(edit_distance(norm_loose(a), norm_loose(b)) for a, b in zip(p, refs))
            / (sum(len(norm_loose(r)) for r in refs) or 1), 2)
    m["empty_pred_pct"] = round(100.0 * sum(1 for x in preds if not x.strip()) / max(1, len(preds)), 1)
    m["distinct_preds"] = len(set(preds))
    # Console-safe proof that the decode really is Arabic (never judge by eye on Windows).
    m["arabic_pct_pred"] = round(sum(arabic_pct(p) for p in preds) / max(1, len(preds)), 1)
    m["arabic_pct_ref"] = round(sum(arabic_pct(r) for r in refs) / max(1, len(refs)), 1)
    m["cp_hex_first_pred"] = " ".join("%04X" % ord(c) for c in preds[0][:10]) if preds else ""
    return m, [unmirror(x) for x in preds], refs, paths


def probe():
    items, _ = load_set("hflines_val")
    path, ref = items[0]
    arr = preprocess(path)
    print("input array:", arr.shape, arr.dtype)
    for name, d in MODELS.items():
        print("== %s  (%s)" % (name, d))
        chars = load_chars(d / "inference.yml")
        print("   dict size: %d" % len(chars))
        model, kind, meta = load_model(d)
        print("   loader: %s %s" % (kind, meta))
        raw = model(arr) if kind == "jit" else None
        if raw is not None:
            if isinstance(raw, (list, tuple)):
                print("   raw outputs: %d -> %s" % (len(raw), [getattr(x, "shape", type(x).__name__) for x in raw]))
            else:
                print("   raw output: %s %s" % (getattr(raw, "shape", "?"), type(raw).__name__))
        out = call_model(model, kind, arr)
        print("   output: ndim=%d shape=%s dtype=%s" % (out.ndim, out.shape, out.dtype))
        if out.ndim == 3:
            print("   greedy: %r" % (ctc_greedy(out, chars)[0][:60],))
        elif out.ndim == 2:
            print("   greedy: %r" % (ctc_greedy(out[None, ...], chars)[0][:60],))
        print("   ref (first 40 chars): %r" % (ref[:40],))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sets", default="hflines_val")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="runs/ocr_bakeoff")
    args = ap.parse_args()
    if args.probe:
        probe()
        return 0

    set_names = list(SETS) if args.sets == "all" else args.sets.split(",")
    model_names = args.models.split(",")
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    samples = []
    per_image = []
    for mn in model_names:
        d = MODELS[mn]
        chars = load_chars(d / "inference.yml")
        print("== %s (dict %d)" % (mn, len(chars)))
        model, kind, meta = load_model(d)
        for sn in set_names:
            items, note = load_set(sn)
            print("   -- %s (%s) n=%d" % (sn, note, len(items)))
            m, preds, refs, paths = evaluate(model, kind, chars, items, args.limit)
            m["note"] = note
            # Per-line CER distribution - the aggregate hides whether a set is
            # uniformly mediocre or bimodal (printed vs handwritten).
            cers = [100.0 * edit_distance(p, r) / max(1, len(r)) for p, r in zip(preds, refs)]
            n = max(1, len(cers))
            m["cer_median"] = round(sorted(cers)[len(cers) // 2], 2) if cers else None
            m["cer_lt10_pct"] = round(100.0 * sum(1 for c in cers if c < 10) / n, 1)
            m["cer_lt20_pct"] = round(100.0 * sum(1 for c in cers if c < 20) / n, 1)
            m["cer_lt50_pct"] = round(100.0 * sum(1 for c in cers if c < 50) / n, 1)
            results["%s|%s" % (mn, sn)] = m
            print("      CER unmirrored %.2f%% | reversed %.2f%% | as-is %.2f%%"
                  % (m["cer_unmirrored"], m["cer_reversed"], m["cer_as_is"]))
            print("      WER(unmir) %.2f%% | exact(unmir) %.2f%% | pred-arabic %.1f%% | %.1f img/s"
                  % (m["wer_unmirrored"], m["exact_unmirrored"], m["arabic_pct_pred"], m["img_per_s"]))
            print("      per-line CER: median %.1f%% | <10%%: %.1f%% | <20%%: %.1f%% | <50%%: %.1f%%"
                  % (m["cer_median"], m["cer_lt10_pct"], m["cer_lt20_pct"], m["cer_lt50_pct"]))
            for p, r in list(zip(preds, refs))[:5]:
                samples.append({"model": mn, "set": sn, "pred": p, "ref": r})
            for p, r, pa, ce in zip(preds, refs, paths, cers):
                per_image.append({"model": mn, "set": sn, "image": pa,
                                  "cer": round(ce, 2), "pred": p, "ref": r})

    (out_dir / "stage0_results.json").write_text(
        json.dumps({"results": results, "control": CONTROL}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with (out_dir / "stage0_samples.txt").open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write("[%s | %s]\nPRED: %s\nREF : %s\n\n" % (s["model"], s["set"], s["pred"], s["ref"]))
    with (out_dir / "per_image.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for rec in per_image:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote %s" % (out_dir / "per_image.jsonl"))
    print("wrote %s" % (out_dir / "stage0_results.json"))
    print("wrote %s" % (out_dir / "stage0_samples.txt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
