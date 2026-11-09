"""mex/scripts/eval_mu2_g2.py — G2 gates (E-31): fill-in CE/accuracy + retention.

Model = runs/mex/mu2_g2/final (merged plain safetensors + G2 config).
  val_task.bin       -> CE + fill accuracy at '|' positions (label = true mark)
  holdout_retent.bin -> clean-stream CE (retention gate vs G1 best 0.7077)
Trivial baselines: either-guess mark unigram CE from the clean replay stream
and its top-class accuracy. GPU-only eval, runs after training.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mex.src.vocab import CharVocab
from src.model import build_model
from safetensors.torch import load_file

G1_BEST = 0.7077  # E-30 best eval (nats/char); retention gate = within +5%

import os as _os
FINAL = _os.environ.get("G2_FINAL", "runs/mex/mu2_g2/final")
CFG_PATH = _os.environ.get("G2_CFG", "configs/mu2_g2.yaml")
cfg = yaml.safe_load((ROOT / CFG_PATH).read_text(encoding="utf-8"))
voc = CharVocab()
V = len(voc.vocab)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(cfg, vocab_size=V).to(dev).eval()
import os as _os
sd = load_file(str(ROOT / FINAL / "model.safetensors"))
model.load_state_dict(sd, strict=False)

CTX = int(cfg["model"]["ctx"])
MID = voc.vocab["|"]
MARK_IDS = {voc.vocab[c] for c in "ًٌٍَُِّّْ"}


def ids_of(name: str) -> np.ndarray:
    p = ROOT / "data" / "mex" / "mu2" / "g2" / name
    if not p.is_file():
        p = ROOT / "data" / "mex" / "mu2" / "g2" / "tokens" / (name.replace(".bin", "") + "_0000.bin")
    if not p.is_file():
        p = ROOT / "data" / "mex" / "mu2" / "g2" / "tokens" / name
    return np.frombuffer(p.read_bytes(), dtype=np.uint32).astype(np.int64)


@torch.inference_mode()
def run(path_name: str, tag: str, lab_arr=None):
    arr = ids_of(path_name)
    lab_src = lab_arr
    n_block = len(arr) // CTX
    nll = 0.0
    n_mark = n_mark_ok = 0
    for b in range(n_block):
        chunk = arr[b * CTX:(b + 1) * CTX]
        ids = torch.as_tensor(chunk, dtype=torch.long, device=dev).unsqueeze(0)
        logits = model(input_ids=ids).logits[0].float()
        lab = (lab_src if lab_src is not None else arr)[b * CTX + 1:(b + 1) * CTX]
        nll += torch.nn.functional.cross_entropy(
            logits[:-1], torch.as_tensor(lab, device=dev), reduction="sum").item()
        pred = logits[:-1].argmax(-1).cpu().numpy()
        pos = np.array([i for i, t in enumerate(lab) if t in MARK_SET_IDS])
        if pos.size:
            n_mark_ok += int((pred[pos] == lab[pos]).sum())
            n_mark += pos.size
    ce = nll / (n_block * CTX)
    acc = n_mark_ok / max(n_mark, 1)
    print(f"{tag}: CE={ce:.4f} nats/char | fill acc={acc:.4f} ({n_mark_ok}/{n_mark})")
    return ce, acc


@torch.inference_mode()
def run_arr(arr, tag: str, lab_arr=None):
    n_block = len(arr) // CTX
    nll = 0.0; n_mark = n_mark_ok = 0
    for b in range(n_block):
        chunk = arr[b * CTX:(b + 1) * CTX]
        ids = torch.as_tensor(chunk, dtype=torch.long, device=dev).unsqueeze(0)
        logits = model(input_ids=ids).logits[0].float()
        lab = (lab_arr if lab_arr is not None else arr)[b * CTX + 1:(b + 1) * CTX]
        nll += torch.nn.functional.cross_entropy(
            logits[:-1], torch.as_tensor(np.ascontiguousarray(lab), device=dev), reduction="sum").item()
        pred = logits[:-1].argmax(-1).cpu().numpy()
        pos = np.array([i for i, t in enumerate(lab) if t in MARK_SET_IDS])
        if pos.size:
            n_mark_ok += int((pred[pos] == lab[pos]).sum()); n_mark += pos.size
    ce = nll / (n_block * CTX); acc = n_mark_ok / max(n_mark, 1)
    print(f"{tag}: CE={ce:.4f} nats/char | fill acc={acc:.4f} ({n_mark_ok}/{n_mark})"); return ce, acc


MARK_SET_IDS = MARK_IDS
rp = (ROOT / "data/mex/mu2/g2/train_replay.txt").read_text(encoding="utf-8", errors="ignore")
mcount = Counter(i for i in voc.encode(rp) if i in MARK_SET_IDS)
tot_m = sum(mcount.values())
H = -sum((c / tot_m) * math.log(c / tot_m) for c in mcount.values())
top, top_c = mcount.most_common(1)[0]
print(f"either-guess baseline: CE={H:.4f} acc={top_c / tot_m:.4f} "
      f"({voc.decode([top])} dominates)")

hold = ids_of("holdout_retent.bin")
corr = hold.copy()
mask_arr = np.asarray(sorted(MARK_SET_IDS), dtype=np.int64)
corr[np.isin(corr, mask_arr)] = MID
(ROOT / "data/mex/mu2/g2/tokens").mkdir(parents=True, exist_ok=True)
corr.tofile(ROOT / "data/mex/mu2/g2/tokens/val_corr_inbatch.bin")
corr = hold.copy(); corr[np.isin(corr, mask_arr)] = MID
ce_t, acc_t = run_arr(corr, "G2 val (corrupted-input fill-in)", lab_arr=hold)
ce_r, _ = run_arr(hold, "G2 retention (clean)")
print(f"GATE fill-in  : {'PASS' if acc_t > top_c / tot_m else 'FAIL'} "
      f"({acc_t:.4f} vs {top_c / tot_m:.4f} either-guess)")
print(f"GATE retention: {'PASS' if ce_r <= G1_BEST * 1.05 else 'FAIL'} "
      f"({ce_r:.4f} vs {G1_BEST * 1.05:.4f} guard)")
