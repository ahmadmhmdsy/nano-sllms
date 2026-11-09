"""Stage 0.5d (B1') - low-lr adaptation of the Muharaf CNN-Transformer-CTC backbone.

PRE-REGISTERED BEFORE THE RUN (TASKS 138). Inference-only baseline first
(TASKS 138 step 0, E-92): on the REAL held-out splits, our raw protocol,
  sdkv2/muharaf-arabic-ocr  4.25M  Muharaf test 22.11 (16.83 norm)  KHATT test 40.79
  FixFips/arabicocr-khatt  15.06M  Muharaf test 53.09               KHATT test 10.99
The teacher is already trained on Muharaf train, so fine-tuning it on Muharaf
train alone cannot beat 22.11 - the actionable gap is the CROSS-domain one.

Run: Muharaf train (22,091) + KHATT teklia train (4,672, oversampled) +
reallines train (1,568), warm-start from the teacher, low lr, no LM.

GATES (checked by hand against the scorecard, not by this script):
  G1  KHATT teklia test raw CER <= 20.00  (from 40.79, -20.8 pp)
  G2  Muharaf test  raw CER <= 23.11      (from 22.11, <= +1.00 pp regression)
  G3  printed ISIPPT / KHATT benches reported, not a gate yet
If G2 fails, the recipe is re-run at lower lr / fewer KHATT-oversampled steps.

Metric is imported from the verified TASKS 134 harness - no duplicated metric.

OPTION (D) EXTENSION (2026-10-07). The same script also supports a from-scratch, scaled
architecture at a taller input: `--scratch --arch-d-model 384 --arch-layers 8 --arch-ffn
1536 --target-height 128 --warmup-steps N`. Every one of those defaults to the pre-(D)
behaviour, so the B1'/B2 recipes above stay reproducible. E-102 proved that a height
change is only valid at TRAINING time (re-decoding the same weights at a taller input
destroys them); E-103 measured the 4.25M model to be underfit on its own training labels,
which is why (D) scales capacity.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "arabic_ocr" / "models" / "muharaf_ocr"
CKPT_PATH = MODEL_DIR / "models" / "ctc_backbone" / "model.pt"
DEFAULT_OUT = ROOT / "arabic_ocr" / "runs" / "stage05d"
TARGET_H = 96

sys.path.insert(0, str(MODEL_DIR))

from submission_code.ocr_helpers import (ImageConfig, normalize_arabic_text,   # noqa: E402
                                         preprocess_image)
from submission_code.torch_models import load_ctc_backbone, recognise_line     # noqa: E402


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

TRAIN_SETS = [
    ("muharaf_train", ROOT / "data" / "hf" / "muharaf" / "train", 1),
    ("khatt_teklia_train", ROOT / "data" / "hf" / "khatt_teklia" / "train", 0),   # repeat from args
    ("reallines_train", ROOT / "data" / "reallines" / "train", 0),   # repeat from args
]

# AMIDDA sub-sets (TASKS 155), discovered at import time so a partially-ingested
# corpus cannot be trained on by accident. Every split of every sub-set is used:
# AMIDDA's own train/validation/test split is internal to AMIDDA's benchmark, has no
# meaning for us, and we never report an AMIDDA number. The leakage audit
# (arabic_ocr/scripts/audit_amidda_leakage.py) is a HARD gate in main() - the run
# refuses to start without it, because these corpora are third-party copies of
# manuscript pages and may share pages with our own eval splits.
for _amidda_dir in sorted((ROOT / "data" / "hf").glob("amidda_*")):
    for _split in ("train", "validation", "test"):
        if (_amidda_dir / _split / "labels.json").exists():
            TRAIN_SETS.append(("%s_%s" % (_amidda_dir.name, _split),
                               _amidda_dir / _split, 0))
AMIDDA_AUDIT = ROOT / "data" / "hf" / "amidda_audit.json"

EVAL_SETS = {
    "muharaf_test": ROOT / "data" / "hf" / "muharaf" / "test",
    "khatt_teklia_test": ROOT / "data" / "hf" / "khatt_teklia" / "test",
    "isippt_bench": ROOT / "data" / "benchmark" / "isippt",
    "khatt_bench": ROOT / "data" / "benchmark" / "khatt",
}
PRIMARY = ["muharaf_test", "khatt_teklia_test"]


def estimate_width(rec: dict, folder: Path, target_h: int = TARGET_H) -> int:
    w, h = rec.get("w"), rec.get("h")
    if isinstance(w, (int, float)) and isinstance(h, (int, float)) and h:
        return max(1, int(round(float(w) * target_h / float(h))))
    with Image.open(folder / rec["image"]) as im:
        return max(1, int(round(im.width * target_h / max(1, im.height))))


def encode_label(charset, text: str):
    """Encode a label, repairing diacritics if the charset cannot represent them.

    The Muharaf CTC charset is 165 characters and contains NO diacritics
    (U+064B-U+065F, U+0670), so a diacritized label is charset-OOV. Dropping such
    lines discards 7,472 of the 22,091 Muharaf train lines (34%); instead the
    marks are stripped and the line is kept.
    Returns (ids, repaired) or (None, False) when the label cannot be encoded.
    """
    base = normalize_arabic_text(text)
    for index, candidate in enumerate((base, strip_diacritics(base))):
        if len(candidate) < 2:
            return None, False
        try:
            return charset.encode(candidate), index == 1
        except ValueError:
            continue
    return None, False


def strip_diacritics(text: str) -> str:
    kept = "".join(c for c in text
                   if not ("\u064b" <= c <= "\u065f" or c == "\u0670"))
    return normalize_arabic_text(kept)


def light_augment(im, rng):
    """Light geometric + photometric jitter, TRAINING images only.

    The published record contains only the NAME `augmentation_mode: "light"`; the
    authors' augmentation code is not in this repo. This is therefore a documented
    RECONSTRUCTION, not a copy: +-2.5 deg rotation, +-2.0 deg x-shear, 0.94-1.06
    scale, +-2% translation, and a 50% chance of 0.85-1.15 contrast. The white
    background is preserved (fillcolor=255), which is what the CTC frontend expects.
    """
    w, h = im.size
    angle = math.radians(rng.uniform(-2.5, 2.5))
    shear = math.radians(rng.uniform(-2.0, 2.0))
    scale = rng.uniform(0.94, 1.06)
    dx = rng.uniform(-0.02, 0.02) * w
    dy = rng.uniform(-0.02, 0.02) * h
    cx, cy = w / 2.0, h / 2.0
    # PIL wants the INVERSE map (output pixel -> input pixel). Forward, about the
    # centre: translate -> x-shear -> rotate -> scale; then invert.
    t = math.tan(shear)
    ca, sa = math.cos(angle) / scale, math.sin(angle) / scale
    a = ca
    b = sa - ca * t
    d = -sa
    e = ca + sa * t
    c = cx - a * (cx + dx) - b * (cy + dy)
    f = cy - d * (cx + dx) - e * (cy + dy)
    im = im.transform((w, h), Image.AFFINE, (a, b, c, d, e, f),
                      resample=Image.BILINEAR, fillcolor=255)
    if rng.random() < 0.5:
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    return im


def degrade_resolution(im, rng, band):
    """Simulate a lower-resolution SOURCE scan (E-100: ISIPPT lines are 27 px).

    The line is shrunk to a randomly chosen source height inside @@band@@ and
    preprocess_image then resamples it back up to target_h, so the network sees the
    real blur/aliasing of a low-res scan while the tensor geometry stays fixed. That
    is why this lever needs NO change to the batching code - unlike a target-height
    change, which moves every width and the whole memory profile with it.
    """
    lo, hi = band
    w, h = im.size
    top = min(int(hi), int(h))
    if int(lo) >= top:
        return im
    src_h = rng.randint(int(lo), top)
    src_w = max(1, int(round(w * src_h / float(h))))
    return im.resize((src_w, src_h), Image.BILINEAR)


def build_items(charset, folder: Path, repeat: int, limit: int, target_h: int = TARGET_H,
                skip=None):
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    if limit:
        labels = labels[:limit]
    items, oov, short, repaired, dropped = [], 0, 0, 0, 0
    for rec in labels:
        if skip and rec.get("image") in skip:
            dropped += 1
            continue
        ids, was_repaired = encode_label(charset, rec.get("text", ""))
        if ids is None:
            if len(normalize_arabic_text(rec.get("text", ""))) < 2:
                short += 1
            else:
                oov += 1
            continue
        repaired += 1 if was_repaired else 0
        item = {"path": str(folder / rec["image"]), "ids": ids,
                "est_w": estimate_width(rec, folder, target_h)}
        for _ in range(max(1, repeat)):
            items.append(item)
    return items, {"oov": oov, "too_short": short,
                   "diacritics_repaired": repaired, "kept": len(items),
                   "leak_dropped": dropped}


class LineDataset(torch.utils.data.Dataset):
    def __init__(self, items, target_h: int = TARGET_H, augment: str = "none",
                 seed: int = 1234, res_prob: float = 0.0,
                 res_band=(27, TARGET_H)):
        self.items = items
        self.target_h = target_h
        self.augment = augment
        self.seed = seed
        self.res_prob = res_prob
        self.res_band = res_band
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        # seeded per (seed, epoch, index) so a resumed run reproduces the exact same
        # augmentation instead of a fresh random draw
        rng = random.Random("%d-%d-%d" % (self.seed, self.epoch, index))
        with Image.open(item["path"]) as im:
            if self.augment == "light":
                im = light_augment(im.convert("L"), rng)
            if self.res_prob and rng.random() < self.res_prob:
                # order matters: geometric jitter at full resolution first, THEN the
                # low-res scan simulation, then preprocess_image upsamples to target_h
                im = degrade_resolution(im.convert("L"), rng, self.res_band)
            arr = preprocess_image(im, ImageConfig(target_height=self.target_h,
                                                   grayscale=True, normalize=True))
        return {"img": torch.from_numpy(arr[None, :, :].astype("float32")),
                "ids": torch.tensor(item["ids"], dtype=torch.long),
                "w": int(arr.shape[1])}


def collate(batch):
    widths = [b["img"].shape[-1] for b in batch]
    height = batch[0]["img"].shape[-2]
    max_w = max(widths)
    images = torch.ones(len(batch), 1, height, max_w)     # white background, as the author
    for i, b in enumerate(batch):
        images[i, :, :, :b["img"].shape[-1]] = b["img"]
    return {
        "images": images,
        "input_widths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.cat([b["ids"] for b in batch]),
        "target_lengths": torch.tensor([len(b["ids"]) for b in batch], dtype=torch.long),
    }


@torch.no_grad()
def evaluate(model, charset, device, sets, limit=0, target_h: int = TARGET_H):
    was_training = model.training
    model.eval()
    out = []
    for tag in sets:
        folder = EVAL_SETS[tag]
        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        if limit:
            labels = labels[:limit]
        edits = ref_chars = edits_norm = ref_chars_norm = 0
        cers = []
        for rec in labels:
            hyp = recognise_line(model, charset, str(folder / rec["image"]),
                                 image_height=target_h, device=device)
            ref = rec.get("text", "")
            e = edit_distance(hyp, ref)
            edits += e
            ref_chars += len(ref)
            cers.append(100.0 * e / max(1, len(ref)))
            nh, nr = norm_ar(hyp), norm_ar(ref)
            edits_norm += edit_distance(nh, nr)
            ref_chars_norm += len(nr)
        out.append({"tag": tag, "n": len(labels),
                    "cer": round(100.0 * edits / max(1, ref_chars), 2),
                    "cer_norm": round(100.0 * edits_norm / max(1, ref_chars_norm), 2),
                    "cer_median": round(sorted(cers)[len(cers) // 2], 2) if cers else None})
    if was_training:
        model.train()
    return out


def load_charset_and_config():
    """Charset + base model_config, both taken from the published checkpoint.

    The charset is the LABEL SPACE (165 characters + blank), not a learned parameter, so
    even a from-scratch model must inherit it from here.
    """
    from submission_code.ocr_helpers import Charset

    blob = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    return Charset(blob["charset"]), dict(blob.get("model_config", {}))


def build_model_from_config(cfg, charset, device):
    from submission_code.torch_models import CNNTransformerCTC

    model = CNNTransformerCTC(
        num_classes=charset.vocab_size,
        d_model=cfg["d_model"],
        num_encoder_layers=cfg["transformer_layers"],
        num_heads=cfg["transformer_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg.get("dropout", 0.1),
        height_pooling=cfg.get("height_pooling", "attention"),
    )
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--khatt-repeat", type=int, default=3)
    ap.add_argument("--reallines-repeat", type=int, default=0,
                    help="0 = excluded: reallines text is fully diacritized and the "
                         "Muharaf charset has no diacritics (charset OOV)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--eval-every", type=int, default=800)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--init-ckpt", default="",
                    help="warm-start from this run checkpoint instead of the teacher")
    ap.add_argument("--smoke", type=int, default=0, help="run N steps then exit 0")
    ap.add_argument("--eval-only", action="store_true",
                    help="score --ckpt on all eval sets and exit (no training)")
    ap.add_argument("--ckpt", default="", help="checkpoint for --eval-only")
    # --- option (D): all default to the pre-(D) behaviour ---
    ap.add_argument("--target-height", type=int, default=0,
                    help="0 = the 96 default. Option (D) trains at 128: E-102 showed a "
                         "height change is only valid at TRAINING time")
    ap.add_argument("--scratch", action="store_true",
                    help="random-init a new architecture instead of warm-starting")
    ap.add_argument("--arch-d-model", type=int, default=0)
    ap.add_argument("--arch-layers", type=int, default=0)
    ap.add_argument("--arch-heads", type=int, default=0)
    ap.add_argument("--arch-ffn", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="linear LR warmup before the cosine decay (0 = off, the "
                         "pre-(D) behaviour); a from-scratch run needs it")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/ckpt_last.pt (model + optimizer + step)")
    ap.add_argument("--stage", default="0.5d-b1prime")
    # --- step 1 (TASKS 151): multi-scale + resolution degradation + AMIDDA ---
    ap.add_argument("--amidda-repeat", type=int, default=1,
                    help="0 = exclude the discovered AMIDDA sub-sets from this run")
    ap.add_argument("--vram-cap-gib", type=float, default=0.0,
                    help="hard allocator ceiling in GiB; turns a WDDM shared-memory "
                         "spill into a loud CUDA OOM (AGENTS 20.3). 0 = off")
    ap.add_argument("--height-band", default="0",
                    help="multi-scale target height as LO,HI (e.g. '80,128'); sampled "
                         "per BATCH so the collated tensor stays uniform. 0 = fixed "
                         "--target-height. E-102: height is a TRAINING-time contract, "
                         "and --eval-only still scores at --target-height")
    ap.add_argument("--res-degrade-prob", type=float, default=0.0,
                    help="per-image probability of shrinking the line to a lower "
                         "'source' resolution before preprocess_image resamples it to "
                         "target_h - the E-100 failure mode (ISIPPT lines are 27 px) "
                         "turned into training data")
    ap.add_argument("--res-degrade-band", default="27,96",
                    help="source-height band for --res-degrade-prob")
    # --- plan (A): the authors' fine-tune recipe; every default preserves the
    # pre-(A) behaviour, so the B1'/B2 recipes stay reproducible ---
    ap.add_argument("--augment", choices=("none", "light"), default="none",
                    help="'light' = light affine+contrast jitter on TRAINING images only")
    ap.add_argument("--train-blank-penalty", type=float, default=0.0,
                    help="subtract this from the blank logit BEFORE the training loss "
                         "(the authors' train_blank_logit_penalty; anti-deletion)")
    ap.add_argument("--scheduler", choices=("cosine", "onecycle"), default="cosine",
                    help="'onecycle' = the authors' schedule, peaking at --lr")
    ap.add_argument("--pct-start", type=float, default=0.3)
    ap.add_argument("--keep-eval-ckpts", action="store_true",
                    help="also save ckpt_step<N>.pt at every periodic eval, so the best "
                         "step stays scorable later (ckpt_last.pt alone loses it)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    target_h = args.target_height or TARGET_H

    height_band = None
    if args.height_band and args.height_band != "0":
        lo_s, _, hi_s = args.height_band.partition(",")
        height_band = (int(lo_s), int(hi_s or lo_s))
        if height_band[0] > height_band[1]:
            raise SystemExit("--height-band LO,HI needs LO <= HI")

    lo_s, _, hi_s = args.res_degrade_band.partition(",")
    res_band = (int(lo_s), int(hi_s or lo_s))

    amidda_drop = {}
    amidda_sets = [t for t in TRAIN_SETS if t[0].startswith("amidda")]
    if amidda_sets and args.amidda_repeat > 0:
        if not AMIDDA_AUDIT.exists():
            raise SystemExit("AMIDDA sets %s are present but %s is missing - run "
                             "arabic_ocr/scripts/audit_amidda_leakage.py first"
                             % ([t[0] for t in amidda_sets], AMIDDA_AUDIT))
        _blob = json.loads(AMIDDA_AUDIT.read_text(encoding="utf-8"))
        _leaks = _blob.get("leaks", {})
        _n_leak = sum(len(v) for v in _leaks.values()) if isinstance(_leaks, dict) else 0
        for _d in _blob.get("drop", []):
            amidda_drop.setdefault((_d["name"], _d["split"]), set()).add(_d["image"])
        if _n_leak and not amidda_drop:
            raise SystemExit("AMIDDA audit reports %d leak(s) but carries no drop list - "
                             "re-run arabic_ocr/scripts/audit_amidda_leakage.py"
                             % _n_leak)
        print("amidda audit: rows=%s dup_text=%s dup_page=%s leaks=%d -> DROPPING %d row(s) "
              "from training" % (_blob.get("rows"), _blob.get("dup_text"),
                                 _blob.get("dup_page"), _n_leak, len(_blob.get("drop", []))),
              flush=True)

    device = torch.device(args.device) if args.device != "auto" else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.vram_cap_gib > 0:
        _tot = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        _frac = min(0.97, float(args.vram_cap_gib) / _tot)
        torch.cuda.set_per_process_memory_fraction(_frac, torch.cuda.current_device())
        print("vram cap: %.2f GiB = %.3f of %.2f GiB device (a spill now raises OOM)" %
              (_frac * _tot, _frac, _tot), flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not CKPT_PATH.exists():
        raise SystemExit("checkpoint missing: %s" % CKPT_PATH)

    if args.eval_only:
        ckpt_path = Path(args.ckpt) if args.ckpt else out_dir / "ckpt_last.pt"
        # the checkpoint carries its own dims - never trust --arch-* on this path
        model, charset, cfg = load_ctc_backbone(str(ckpt_path), device=device)
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        res = evaluate(model, charset, device, list(EVAL_SETS), limit=args.eval_limit,
                       target_h=target_h)
        by_tag = {r["tag"]: r for r in res}
        card = {
            "stage": "%s eval-only" % args.stage, "ckpt": str(ckpt_path),
            "checkpoint_step": blob.get("step"), "params": n_params,
            "target_height": target_h,
            "results": res,
            "gates": {
                "G1_khatt_test_raw_le_20.00": by_tag["khatt_teklia_test"]["cer"] <= 20.00,
                "G1_khatt_test_raw_le_21.00": by_tag["khatt_teklia_test"]["cer"] <= 21.00,
                "G2_muharaf_test_raw_le_23.11": by_tag["muharaf_test"]["cer"] <= 23.11,
                "G3_khatt_bench_raw_le_29.54": by_tag["khatt_bench"]["cer"] <= 29.54,
                "G3_isippt_raw_le_58.34": by_tag["isippt_bench"]["cer"] <= 58.34,
            },
            "baselines_zero_shot_raw": {"muharaf_test": 22.11, "khatt_teklia_test": 40.79},
            "baselines_adapted_raw": {"muharaf_test": 22.57, "khatt_teklia_test": 21.82,
                                      "isippt_bench": 53.63, "khatt_bench": 28.47},
        }
        (out_dir / "scorecard_eval.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        print("GATES " + json.dumps(card["gates"]), flush=True)
        return

    if args.scratch:
        charset, cfg = load_charset_and_config()
        for key, val, default in (("d_model", args.arch_d_model, 256),
                                  ("transformer_layers", args.arch_layers, 4),
                                  ("transformer_heads", args.arch_heads, 8),
                                  ("dim_feedforward", args.arch_ffn, 1024)):
            cfg[key] = val or default
        model = build_model_from_config(cfg, charset, device)
        init_label = "scratch d=%d L=%d H=%d F=%d" % (
            cfg["d_model"], cfg["transformer_layers"], cfg["transformer_heads"],
            cfg["dim_feedforward"])
    else:
        model, charset, cfg = load_ctc_backbone(str(args.init_ckpt or CKPT_PATH),
                                                device=device)
        init_label = CKPT_PATH.name
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print("init: %s  params=%d  charset=%d  height_pooling=%s  target_h=%d  device=%s" %
          (init_label, n_params, charset.vocab_size, cfg.get("height_pooling"),
           target_h, device), flush=True)
    print("step1 knobs: height_band=%s  res_degrade_prob=%g  res_degrade_band=%s  "
          "amidda_repeat=%d" % (height_band, args.res_degrade_prob, res_band,
                                args.amidda_repeat), flush=True)

    items = []
    audit = {}
    for name, folder, _default in TRAIN_SETS:
        if name.startswith("reallines"):
            rep = args.reallines_repeat
        elif name.startswith("khatt"):
            rep = args.khatt_repeat
        elif name.startswith("amidda"):
            rep = args.amidda_repeat
        else:
            rep = 1
        skip = None
        if name.startswith("amidda"):
            _base, _, _split = name.rpartition("_")
            skip = amidda_drop.get((_base, _split))
        got, stats = build_items(charset, folder, max(1, rep), args.limit, target_h, skip)
        stats["repeat"] = rep
        stats["included"] = rep > 0
        audit[name] = stats
        if rep > 0:
            items.extend(got)
        print("train %-18s kept=%-7d included=%s (repeat x%d, oov=%d, repaired=%d, "
              "too_short=%d)" % (name, stats["kept"], stats["included"], rep,
                                 stats["oov"], stats["diacritics_repaired"],
                                 stats["too_short"]), flush=True)
    if not items:
        raise SystemExit("no training items")
    print("total train items: %d  widest_est_w=%d (reference height)" %
          (len(items), max(i["est_w"] for i in items)), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(blank=charset.blank_id, zero_infinity=True)
    order = sorted(range(len(items)), key=lambda i: items[i]["est_w"])
    # drop size-1 tail batches: BatchNorm in train mode needs >1 value per channel
    batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
    batches = [b for b in batches if len(b) > 1]
    steps_per_epoch = len(batches)
    opt_steps_per_epoch = max(1, steps_per_epoch // max(1, args.accum))
    total_steps = max(1, opt_steps_per_epoch * args.epochs)
    print("steps/epoch=%d  opt_steps/epoch=%d  total_steps=%d  lr=%g  batch=%d  accum=%d" %
          (steps_per_epoch, opt_steps_per_epoch, total_steps, args.lr, args.batch,
           args.accum), flush=True)

    warmup = max(0, min(args.warmup_steps, max(0, total_steps - 1)))
    if args.scheduler == "onecycle":
        if warmup:
            raise SystemExit("--scheduler onecycle has its own warmup: drop --warmup-steps")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=args.pct_start, anneal_strategy="cos")
        sched_label = ("OneCycleLR(max_lr=%g, total_steps=%d, pct_start=%g, "
                       "div_factor=25, final_div_factor=1e4)" %
                       (args.lr, total_steps, args.pct_start))
    else:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup))
        if warmup:
            linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3,
                                                       total_iters=warmup)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[linear, cosine], milestones=[warmup])
            sched_label = "linear-warmup+cosine"
        else:
            scheduler = cosine
            sched_label = "cosine"

    ds = LineDataset(items, target_h, augment=args.augment, seed=args.seed,
                     res_prob=args.res_degrade_prob, res_band=res_band)
    log_path = out_dir / "train_log.jsonl"
    log = log_path.open("a", encoding="utf-8")

    step = 0
    start_epoch, start_batch = 0, 0
    if args.resume and (out_dir / "ckpt_last.pt").exists():
        blob = torch.load(out_dir / "ckpt_last.pt", map_location=device,
                          weights_only=False)
        model.load_state_dict(blob["model_state"])
        if "optimizer_state" in blob:
            optimizer.load_state_dict(blob["optimizer_state"])
        if "scheduler_state" in blob:
            scheduler.load_state_dict(blob["scheduler_state"])
        step = int(blob.get("step", 0))
        start_epoch = int(blob.get("epoch", 0))
        start_batch = int(blob.get("batch_idx", -1)) + 1
        print("RESUME %s: epoch %d batch %d step %d" % (out_dir / "ckpt_last.pt",
              start_epoch, start_batch, step), flush=True)

    t0 = time.perf_counter()
    running = 0.0
    running_n = 0
    scorecards = []
    stop = False
    for epoch in range(start_epoch, args.epochs):
        # deterministic per-epoch order, so a resumed run reproduces the same batches
        random.Random(args.seed + epoch).shuffle(batches)
        ds.epoch = epoch
        begin = start_batch if epoch == start_epoch else 0
        for batch_idx in range(begin, steps_per_epoch):
            idxs = batches[batch_idx]
            if height_band:
                # ONE height per batch (the collated tensor must stay uniform), derived
                # from (seed, epoch, batch_idx) so a resume reproduces the same batch
                ds.target_h = random.Random("%d-%d-%d" % (args.seed, epoch,
                                                          batch_idx)).randint(*height_band)
            else:
                ds.target_h = target_h
            pixel_batch = collate([ds[i] for i in idxs])
            images = pixel_batch["images"].to(device)
            widths = pixel_batch["input_widths"].to(device)
            targets = pixel_batch["targets"].to(device)
            target_lengths = pixel_batch["target_lengths"].to(device)
            logits = model(images, input_widths=widths)
            if args.train_blank_penalty:
                logits = logits.clone()
                logits[..., charset.blank_id] -= args.train_blank_penalty
            log_probs = logits.log_softmax(dim=-1)
            in_lens = model.output_lengths(widths)
            loss = criterion(log_probs, targets, in_lens, target_lengths) / args.accum
            loss.backward()
            if (batch_idx + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
            running += float(loss.detach()) * args.accum
            running_n += 1

            if running_n % 50 == 0:
                el = time.perf_counter() - t0
                print("  ep%d step%-6d loss=%.4f  lr=%.2e  %.2f steps/s" %
                      (epoch, step, running / running_n,
                       optimizer.param_groups[0]["lr"], step / max(el, 1e-9)),
                      flush=True)
            if args.smoke and step >= args.smoke:
                el = time.perf_counter() - t0
                print("SMOKE OK: %d optimizer steps in %.1fs (%.2fs/step, batch=%d, "
                      "target_h=%d)" % (step, el, el / max(step, 1), args.batch,
                                        target_h), flush=True)
                print("SMOKE last batch shape=%s  last_h=%d" %
                      (tuple(pixel_batch["images"].shape), ds.target_h), flush=True)
                if device.type == "cuda":
                    print("SMOKE peak_vram allocated=%.2f GiB reserved=%.2f GiB  "
                          "total=%.2f GiB" % (
                              torch.cuda.max_memory_allocated() / 2 ** 30,
                              torch.cuda.max_memory_reserved() / 2 ** 30,
                              torch.cuda.get_device_properties(0).total_memory / 2 ** 30),
                          flush=True)
                log.close()
                return
            if args.eval_every and step and step % args.eval_every == 0:
                res = evaluate(model, charset, device, PRIMARY, limit=args.eval_limit,
                               target_h=target_h)
                card = {"step": step, "epoch": epoch, "batch_idx": batch_idx,
                        "results": res}
                scorecards.append(card)
                print("  EVAL %s" % json.dumps(card), flush=True)
                log.write(json.dumps(card) + chr(10))
                log.flush()
                torch.save({"model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "charset": getattr(charset, "characters", None),
                            "model_config": cfg, "step": step, "epoch": epoch,
                            "batch_idx": batch_idx, "stage": args.stage,
                            "target_height": target_h},
                           out_dir / "ckpt_last.pt")
                if args.keep_eval_ckpts:
                    torch.save({"model_state": model.state_dict(),
                                "charset": getattr(charset, "characters", None),
                                "model_config": cfg, "step": step, "epoch": epoch,
                                "stage": args.stage, "target_height": target_h},
                               out_dir / ("ckpt_step%06d.pt" % step))
            if args.smoke and batch_idx + 1 >= args.smoke:
                stop = True
                break
        if stop:
            break

    elapsed = time.perf_counter() - t0
    final = evaluate(model, charset, device, list(EVAL_SETS), limit=args.eval_limit,
                     target_h=target_h)
    card = {"step": step, "epoch": args.epochs - 1, "final": True, "results": final}
    scorecards.append(card)
    log.write(json.dumps(card) + chr(10))
    log.close()
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "charset": getattr(charset, "characters", None),
                "model_config": cfg, "step": step, "stage": args.stage,
                "target_height": target_h},
               out_dir / "ckpt_final.pt")

    by_tag = {r["tag"]: r for r in final}
    summary = {
        "stage": args.stage,
        "init": init_label,
        "params": n_params,
        "recipe": {"epochs": args.epochs, "lr": args.lr, "batch": args.batch,
                   "accum": args.accum, "khatt_repeat": args.khatt_repeat,
                   "weight_decay": args.weight_decay, "clip": args.clip,
                   "scheduler": sched_label,
                   "warmup_steps": warmup, "seed": args.seed,
                   "target_height": target_h, "augmentation": args.augment,
                   "height_band": height_band, "res_degrade_prob": args.res_degrade_prob,
                   "res_degrade_band": list(res_band),
                   "amidda_repeat": args.amidda_repeat,
                   "train_blank_logit_penalty": args.train_blank_penalty,
                   "arch": {k: cfg[k] for k in ("d_model", "transformer_layers",
                                               "transformer_heads",
                                               "dim_feedforward")},
                   "loss": "CTC (blank=0, zero_infinity)", "precision": "fp32"},
        "train_audit": audit,
        "train_items": len(items),
        "total_steps": step,
        "seconds": round(elapsed, 1),
        "gates": {
            "G1_khatt_test_raw_le_21.00": by_tag["khatt_teklia_test"]["cer"] <= 21.00,
            "G2_muharaf_test_raw_le_23.11": by_tag["muharaf_test"]["cer"] <= 23.11,
            "G3_khatt_bench_raw_le_29.54": by_tag["khatt_bench"]["cer"] <= 29.54,
            "G3_isippt_raw_le_58.34": by_tag["isippt_bench"]["cer"] <= 58.34,
        },
        "baselines_zero_shot_raw": {"muharaf_test": 22.11, "khatt_teklia_test": 40.79},
        "baselines_adapted_raw": {"muharaf_test": 22.57, "khatt_teklia_test": 21.82,
                                  "isippt_bench": 53.63, "khatt_bench": 28.47},
        "notes": {
            "why_not_muharaf_only": ("the init checkpoint was itself trained on Muharaf "
                                     "train, so further fine-tuning on Muharaf train alone "
                                     "cannot beat its own 22.11 - the actionable gap is the "
                                     "cross-domain KHATT one"),
            "reallines": ("excluded on this rung: outside the Muharaf/KHATT pair under "
                          "test, and it would change two things at once"),
            "label_repair": ("the checkpoint charset is 165 chars and has NO diacritics "
                             "(verified), so diacritized labels are repaired by stripping "
                             "U+064B-U+065F/U+0670 instead of being dropped - dropping "
                             "would discard 7,472 of 22,091 Muharaf train lines (34%)"),
        },
        "scorecards": scorecards,
    }
    (out_dir / "scorecard.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps({"final": final, "steps": step, "seconds": round(elapsed, 1)}), flush=True)
    print("wrote %s" % (out_dir / "scorecard.json"), flush=True)


if __name__ == "__main__":
    main()
