"""Stage 0.5c - zero-shot evaluation of FixFips/arabicocr-khatt on our corpora.

Evaluates the KHATT-trained CRNN (CNN + BiLSTM + CTC) published as
FixFips/arabicocr-khatt against the Stage 0 control (PaddleOCR
arabic_PP-OCRv6_small_rec, free-running un-mirrored CER 57.80% on
data/hflines/val).

Protocol (matches the control):
  * greedy CTC decode over the model's own forward logits
  * NO language model (the shipped bigram_lm.json is deliberately unused)
  * CER = sum(edit_distance(hyp, ref)) / sum(len(ref))
  * WER = sum(token edit distance) / sum(len(ref tokens))
  * both raw strings and the author's Arabic orthographic normalization

Model code is the author's own (github.com/FixFips/ArabicOCR_KHATT,
arabicocr_khatt/model.py + preprocess.py), imported from a local copy under
arabic_ocr/models/fixfips_khatt/author_code/ - nothing is pip-installed.

The author's training/eval code encodes labels reversed (t[::-1]) and flips
predictions back with h[::-1]; the headline number follows that protocol.
The no-flip CER is also recorded as a direction diagnostic.

Usage:
  & .venv/Scripts/python.exe arabic_ocr/scripts/eval_fixfips_khatt.py --set all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
import unicodedata as ud
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "arabic_ocr" / "models" / "fixfips_khatt"
AUTHOR_DIR = MODEL_DIR / "author_code"
CKPT_PATH = MODEL_DIR / "crnn_best.pt"
OUT_DIR = ROOT / "runs" / "stage05c"

HEIGHT = 96
MAX_W = 1536

# Stage 0.5c control: PaddleOCR PP-OCRv6_small_rec, same corpus, same protocol.
CONTROL_CER_HFLINES = 57.80
CONTROL_CER_KHATT = 44.41
CONTROL_CER_ISIPPT = 13.75

SETS = {
    "hflines": {
        "dir": ROOT / "data" / "hflines" / "val",
        "tag": "hflines_val",
        "weight": 1.0,
        "control": CONTROL_CER_HFLINES,
    },
    "khatt": {
        "dir": ROOT / "data" / "benchmark" / "khatt",
        "tag": "khatt_bench",
        "weight": 1.0,
        "control": CONTROL_CER_KHATT,
    },
    "isippt": {
        "dir": ROOT / "data" / "benchmark" / "isippt",
        "tag": "isippt_bench",
        "weight": 1.0,
        "control": CONTROL_CER_ISIPPT,
    },
    # Stage 0.5d: the REAL held-out splits (never in any teacher's training
    # data). control=None until the Stage 0 control itself is measured here.
    "muharaf_test": {
        "dir": ROOT / "data" / "hf" / "muharaf" / "test",
        "tag": "muharaf_test",
        "weight": 1.0,
        "control": None,
    },
    "khatt_test": {
        "dir": ROOT / "data" / "hf" / "khatt_teklia" / "test",
        "tag": "khatt_teklia_test",
        "weight": 1.0,
        "control": None,
    },
}

ALL_SETS = ["hflines", "khatt", "isippt"]


def _ensure_cv2():
    """Make cv2 importable.

    This repo's .venv has no opencv, but the sibling .venv-paddle (and
    .venv-paddle-gpu) already ship cv2 5.0.0 and its extension module loads
    under this interpreter. Nothing is installed: the existing package is only
    appended to sys.path. The author's unmodified preprocess.py needs cv2 for
    the CLAHE + Otsu binarization the checkpoint was trained with.
    """
    try:
        import cv2
        return "native (%s)" % cv2.__version__
    except ImportError:
        pass
    for rel in (".venv-paddle", ".venv-paddle-gpu"):
        cand = ROOT / rel / "Lib" / "site-packages"
        if cand.is_dir():
            sys.path.append(str(cand))
            try:
                import cv2
                return "borrowed from %s (%s)" % (rel, cv2.__version__)
            except ImportError:
                sys.path.remove(str(cand))
    raise RuntimeError("cv2/opencv is not importable - the author's binarization cannot run")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load module from %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- metrics
def edit_distance(a, b) -> int:
    """Levenshtein distance, sequence-agnostic (str or list)."""
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


_re_diac = re.compile(r"[\u064B-\u0652]")


def norm_ar(s: str) -> str:
    """Author's orthographic normalization (github .../arabicocr_khatt/metrics.py)."""
    s = ud.normalize("NFKC", s)
    s = s.replace("\u0640", "")
    s = _re_diac.sub("", s)
    s = (s.replace("\u0623", "\u0627")
           .replace("\u0625", "\u0627")
           .replace("\u0622", "\u0627")
           .replace("\u0649", "\u064A"))
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- preprocessing
def make_preprocessor(prep_mod):
    def prep(path: Path) -> np.ndarray:
        img = Image.open(path)
        img = prep_mod.to_grayscale(img)
        img = prep_mod.binarize(img)
        img = prep_mod.resize_keep_ratio_height(img, HEIGHT)
        if img.width > MAX_W:
            img = img.resize((MAX_W, HEIGHT), Image.LANCZOS)
        img = prep_mod.pad_width(img, MAX_W)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr[None, :, :]  # (1, H, W)
    return prep


def load_items(set_dir: Path):
    labels = json.loads((set_dir / "labels.json").read_text(encoding="utf-8"))
    items = []
    for r in labels:
        name = r.get("image") or r.get("img") or r.get("file")
        items.append((set_dir / name, r.get("text", r.get("label", ""))))
    return items


# ---------------------------------------------------------------- evaluation
def evaluate(name: str, model, id2char, prep, device, batch_size: int,
             limit: int = 0, want_examples: int = 0):
    spec = SETS[name]
    items = load_items(spec["dir"])
    if limit:
        items = items[:limit]

    preds_flip, preds_raw, refs, paths = [], [], [], []
    cers = []
    t0 = time.perf_counter()
    n_done = 0

    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        arrs = np.stack([prep(p) for p, _ in chunk], axis=0)
        x = torch.from_numpy(arrs).to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(x)
            hyps_rev = _ctc_greedy(logits, id2char)
        for (p, ref), h in zip(chunk, hyps_rev):
            hf = h[::-1]  # author protocol: model emits reversed order
            preds_flip.append(hf)
            preds_raw.append(h)
            refs.append(ref)
            paths.append(str(p.relative_to(ROOT)))
            cers.append(100.0 * edit_distance(hf, ref) / max(1, len(ref)))
        n_done += len(chunk)
        if n_done % (batch_size * 20) < batch_size:
            el = time.perf_counter() - t0
            print("    %d/%d  %.2f img/s" % (n_done, len(items), n_done / max(el, 1e-9)),
                  flush=True)

    elapsed = time.perf_counter() - t0

    perline = [{"image": paths[i], "ref": refs[i], "hyp": preds_flip[i],
                "cer": round(cers[i], 2)} for i in range(len(refs))]

    def agg(hyp_list):
        ed = sum(edit_distance(h, r) for h, r in zip(hyp_list, refs))
        rc = sum(len(r) for r in refs)
        ew = 0
        rw = 0
        exact = 0
        for h, r in zip(hyp_list, refs):
            hw, rww = h.split(), r.split()
            ew += edit_distance(hw, rww)
            rw += len(rww)
            if h == r:
                exact += 1
        n = max(1, len(hyp_list))
        return {
            "cer": 100.0 * ed / max(1, rc),
            "wer": 100.0 * ew / max(1, rw),
            "exact_pct": 100.0 * exact / n,
        }

    def agg_norm(hyp_list):
        ed = sum(edit_distance(norm_ar(h), norm_ar(r)) for h, r in zip(hyp_list, refs))
        rc = sum(len(norm_ar(r)) for r in refs)
        ew = 0
        rw = 0
        for h, r in zip(hyp_list, refs):
            hw, rww = norm_ar(h).split(), norm_ar(r).split()
            ew += edit_distance(hw, rww)
            rw += len(rww)
        return {"cer_norm": 100.0 * ed / max(1, rc), "wer_norm": 100.0 * ew / max(1, rw)}

    n = max(1, len(items))
    main = agg(preds_flip)
    norm = agg_norm(preds_flip)
    noflip = agg(preds_raw)

    result = {
        "tag": spec["tag"],
        "dataset": str(spec["dir"].relative_to(ROOT)).replace("\\", "/"),
        "n": len(items),
        "model_id": "FixFips/arabicocr-khatt",
        "checkpoint": str(CKPT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha_expected": "d1e394ced68290bf92e88ba5d34bb02d710f9fc1",
        "decoder": "greedy CTC, no language model",
        "flip": "author protocol: predictions reversed with h[::-1] (model trained on reversed labels)",
        "params": sum(p.numel() for p in model.parameters()),
        "cer": round(main["cer"], 2),
        "wer": round(main["wer"], 2),
        "exact_pct": round(main["exact_pct"], 2),
        "cer_norm": round(norm["cer_norm"], 2),
        "wer_norm": round(norm["wer_norm"], 2),
        "cer_no_flip": round(noflip["cer"], 2),
        "cer_median": round(sorted(cers)[len(cers) // 2], 2) if cers else None,
        "cer_lt10_pct": round(100.0 * sum(1 for c in cers if c < 10) / n, 1),
        "cer_lt20_pct": round(100.0 * sum(1 for c in cers if c < 20) / n, 1),
        "cer_lt50_pct": round(100.0 * sum(1 for c in cers if c < 50) / n, 1),
        "seconds": round(elapsed, 1),
        "img_per_s": round(len(items) / max(elapsed, 1e-9), 2),
        "control_cer": spec.get("control"),
        "control_model": "medyas/arabic_PP-OCRv6_small_rec (zero-shot, un-mirrored, dynamic width)",
        "_perline": perline,
    }
    if result["control_cer"] is not None:
        result["delta_vs_control_pp"] = round(result["control_cer"] - result["cer"], 2)

    if want_examples:
        idx = list(range(min(want_examples, len(items))))
        result["examples"] = [
            {
                "idx": i,
                "image": paths[i],
                "ref": refs[i],
                "hyp": preds_flip[i],
                "cer": round(cers[i], 2),
            }
            for i in idx
        ]
    return result


def _ctc_greedy(logits, id2char):
    """Greedy CTC decode (author's ctc_greedy_decode, inlined for batch use)."""
    pred = logits.argmax(-1).detach().cpu().numpy()  # [T, B]
    T, B = pred.shape
    texts = []
    for b in range(B):
        seq, last = [], -1
        for t in range(T):
            p = int(pred[t, b])
            if p != last and p != 0:
                seq.append(p)
            last = p
        out = [id2char[int(i)] for i in seq]
        out = [c for c in out if not (c.startswith("<") and c.endswith(">"))]
        texts.append(re.sub(r"\s+", " ", "".join(out)).strip())
    return texts


def write_examples_file(results, path: Path):
    lines = []
    for r in results:
        if not r.get("examples"):
            continue
        lines.append("=" * 100)
        ctl = "n/a" if r["control_cer"] is None else "%.2f%%" % r["control_cer"]
        lines.append("DATASET: %s  (n=%d, CER=%.2f%%, control=%s)" %
                     (r["dataset"], r["n"], r["cer"], ctl))
        lines.append("=" * 100)
        for ex in r["examples"]:
            lines.append("")
            lines.append("[%d] %s   line CER=%.2f%%" % (ex["idx"], ex["image"], ex["cer"]))
            lines.append("  REF: %s" % ex["ref"])
            lines.append("  HYP: %s" % ex["hyp"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hflines",
                    help="comma-separated set names, or 'all' (= %s)" % ",".join(ALL_SETS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--examples", type=int, default=20)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cv2_src = _ensure_cv2()
    print("cv2: %s" % cv2_src)
    author_model = _load_module("fixfips_author_model", AUTHOR_DIR / "model.py")
    prep_mod = _load_module("fixfips_author_preprocess", AUTHOR_DIR / "preprocess.py")
    prep = make_preprocessor(prep_mod)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device: %s" % device)

    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    vocab = state["vocab"]
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    print("checkpoint arch_version=%s vocab=%d use_attention=%s" %
          (state.get("arch_version"), len(vocab), state.get("use_attention")))
    model = author_model.CRNN(len(vocab), use_attention=bool(state.get("use_attention", False)))
    missing, unexpected = model.load_state_dict(state["model"], strict=True), None
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print("params: %d" % n_params)

    if args.set == "all":
        names = list(ALL_SETS)
    else:
        names = [s.strip() for s in args.set.split(",") if s.strip()]
    unknown = [s for s in names if s not in SETS]
    if unknown:
        ap.error("unknown set(s) %s; known: %s" % (", ".join(unknown), ", ".join(SETS)))
    results = []
    for name in names:
        print("== evaluating %s (%s)" % (name, SETS[name]["dir"]))
        r = evaluate(name, model, id2char, prep, device, args.batch,
                     limit=args.limit, want_examples=args.examples)
        r["params"] = n_params
        results.append(r)
        perline = r.pop("_perline", [])
        print(json.dumps({k: v for k, v in r.items() if k != "examples"},
                         ensure_ascii=True, indent=2))
        pl = OUT_DIR / ("perline_%s.jsonl" % r["tag"])
        with pl.open("w", encoding="utf-8") as fh:
            for row in perline:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("wrote %s (%d lines)" % (pl.relative_to(ROOT), len(perline)))

    suffix = "_limit%d" % args.limit if args.limit else ""
    for r in results:
        out = OUT_DIR / ("eval_fixfips_%s%s.json" % (r["tag"], suffix))
        out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote %s" % out.relative_to(ROOT))

    if args.set == "all" and not args.limit:
        summary = [{k: v for k, v in r.items() if k != "examples"} for r in results]
        (OUT_DIR / "eval_fixfips_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote %s" % (OUT_DIR / "eval_fixfips_summary.json").relative_to(ROOT))

    tags = [r["tag"] for r in results]
    if tags == ["hflines_val"]:
        ex_name = "examples_fixfips_hflines_val%s.txt" % suffix
    else:
        ex_name = "examples_fixfips_%s%s.txt" % ("_".join(tags), suffix)
    ex_path = OUT_DIR / ex_name
    write_examples_file(results, ex_path)
    print("wrote %s" % ex_path.relative_to(ROOT))

    # machine-readable direction diagnostics on stdout (console cannot show Arabic)
    for r in results:
        print("DIRECTION %s: cer_flip=%.2f  cer_no_flip=%.2f" %
              (r["tag"], r["cer"], r["cer_no_flip"]))


if __name__ == "__main__":
    main()
