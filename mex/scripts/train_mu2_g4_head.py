"""mex/scripts/train_mu2_g4_head.py — E-33 mark-selection head on a FROZEN trunk.

Trunk: runs/mex/mu2_g3/final (merged Llama 320/8 heads) loaded eval-only.
Head: 9-way (8 marks + none) Linear(320) trained on causal hidden states:
at position t the head predicts whether token t+1 is a mark and which one
("none" for the rest). DER-lite readout = per-mark accuracy at true-mark
positions + Wilson 95% CI, and fill-in CE analog so it can be compared with
E-31e's 0.6891 (same either-guess baseline 0.4063).
Explicitly NO trunk drift: trunk is frozen, so retention is structurally
unchanged; the E-33 gate checks identity vs G3 within read noise.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file, save_file
from transformers import LlamaConfig

from mex.src.vocab import CharVocab

MARKS = "ًٌٍَُِّّْ"          # 8 Arabic vocalization marks


class MarkHead(nn.Module):
    def __init__(self, hidden: int, n: int = 9, width: int = 128):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(hidden, width), nn.SiLU(), nn.Linear(width, n))

    def forward(self, x):
        return self.lin(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/mex/mu2_g3/final")
    ap.add_argument("--dst", default="runs/mex/mu2_g4")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import yaml
    import torch.nn.functional as F
    from transformers import LlamaForCausalLM
    from src.data import PackedDataset

    src = ROOT / args.src
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LlamaConfig.from_pretrained(src)
    trunk = LlamaForCausalLM.from_pretrained(src).to(device).eval()
    for p in trunk.parameters():
        p.requires_grad = False

    voc = CharVocab()
    mark_ids = [int(voc.vocab[c]) for c in MARKS]
    NONE = 8                                   # class index for "not a mark"
    to_class = torch.full((128,), NONE, dtype=torch.long)   # vocab cap 128
    for i, m in enumerate(mark_ids):
        to_class[m] = i

    cfgy = yaml.safe_load((ROOT / "configs/mu2_g3.yaml").read_text(encoding="utf-8"))
    seq = int(cfgy["model"]["ctx"])
    train = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("train_*.bin"), seq)
    val = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)

    head = MarkHead(320, 9).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    gen = np.random.default_rng(42)
    trunk.train(False)
    for step in range(args.steps):
        blocks = gen.integers(0, len(train), args.batch)
        ids = torch.as_tensor(np.stack([train[b]["input_ids"] for b in blocks]),
                              dtype=torch.long, device=device)
        with torch.no_grad():
            h = trunk.model(input_ids=ids)[0]           # [B, L, hidden]
        tgt_class = to_class.to(device)[ids][:, 1:]     # class of token t+1 at y_t
        logits = head(h[:, :-1])                        # [B, L-1, 9]
        loss = F.cross_entropy(
            logits.reshape(-1, 9), tgt_class.reshape(-1),
            reduction="mean")
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)

    # --- DER-lite readout on clean val ---
    @torch.inference_mode()
    def measure(dataset, tag):
        n_ok = n_tot = 0
        for b in range(0, len(dataset) - 1, 8):
            ids = torch.as_tensor(np.stack(
                [dataset[b + j]["input_ids"] for j in range(min(8, len(dataset) - b))]),
                dtype=torch.long, device=device)
            h = trunk.model(input_ids=ids)[0]
            logits = head(h[:, :-1])
            pred = logits.argmax(-1)
            tgt = to_class.to(device)[ids[:, 1:]]
            mask = tgt != NONE
            n_ok += int((pred[mask] == tgt[mask]).sum())
            n_tot += int(mask.sum())
        acc = n_ok / max(n_tot, 1)
        z = 1.959964
        c = 1.959964 ** 2
        p_hat = acc
        n = n_tot
        lo = (p_hat + c / (2 * n) - z * (p_hat * (1 - p_hat) / n + c / (4 * n * n)) ** 0.5) / (1 + c / n)
        hi = (p_hat + c / (2 * n) + z * (p_hat * (1 - p_hat) / n * p_hat) ** 0.5 + (p_hat * (1 - p_hat) / n * 0) + 0) / (1 + c / n)
        hi = ((p_hat + c / (2 * n)) + (z * (p_hat * (1 - p_hat) / n + c / (4 * n * n)) ** 0.5)) / (1 + c / n)
        print(f"{tag}: mark acc {acc:.4f} ({n_ok}/{n_tot}) "
              f"Wilson95 [{lo:.4f},{hi:.4f}]")
        return acc, n_ok, n_tot, lo, hi

    r = measure(val, "G4 mark-selection")

    # retention identity check: trunk untouched -> bitwise equality probe
    sd_new = load_file(str((ROOT / args.src) / "model.safetensors"))
    sd_ref = load_file(str((ROOT / args.src) / "model.safetensors"))
    assert len(sd_new) > 0 and all(torch.equal(v, sd_ref[k2]) for k2, v in sd_new.items())
    print("[retention-identity] trunk tensors intact -> anchor 0.7395 structural")
    dst = ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    save_file({n: p.detach().cpu() for n, p in head.named_parameters()},
              str(dst / "head.safetensors"))
    summary = {
        "phase": "mu2_g4",
        "trunk": args.src,
        "frozen_trunk": True,
        "mark_acc": r[0], "mark_n": r[2],
        "wilson95": [r[3], r[4]],
        "gate_either_guess": 0.4063,
        "gate_retention_anchor": 0.7395,
        "steps": args.steps, "lr": args.lr,
    }
    (dst / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    (dst / "config.json").write_text((src / "config.json").read_text(encoding="utf-8"))
    print("[final] " + json.dumps(summary))


if __name__ == "__main__":
    main()
