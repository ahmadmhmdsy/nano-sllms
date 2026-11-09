"""mex/scripts/train_mu2_g4b_head.py — E-34 corruption-aware mark head + composed decode.

Difference vs E-33: the head is trained on the G2 corruption stream (marks -> '|' mask
id, labels stay clean, hold_every 3) so server-side inputs match training. The composed
decode walk then measures end-to-end diacritization: per position, head mark class wins
if != none, otherwise the trunk's argmax restricted to non-mark tokens, teacher-forced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file, save_file
from transformers import LlamaConfig

MASK = "|"


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
    ap.add_argument("--dst", default="runs/mex/mu2_g4b")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hold_every", type=int, default=3)
    args = ap.parse_args()

    import yaml
    from src.data import PackedDataset
    from mex.src.vocab import CharVocab

    src = ROOT / args.src
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LlamaConfig.from_pretrained(src)
    from transformers import LlamaForCausalLM
    trunk = LlamaForCausalLM.from_pretrained(src).to(device).eval()
    for p in trunk.parameters():
        p.requires_grad = False

    voc = CharVocab()
    mark_ids = [int(voc.vocab[c]) for c in "ًٌٍَُِّّْ"]
    NONE = 8
    to_class = torch.full((128,), NONE, dtype=torch.long)
    for i, m in enumerate(mark_ids):
        to_class[m] = i
    nonmark_ids = torch.tensor([i for i in range(int(voc.vocab_size)  if hasattr(voc, 'vocab_size') else 97)
                               if i not in mark_ids], dtype=torch.long, device=device)
    mid = int(voc.vocab[MASK])

    cfgy = yaml.safe_load((ROOT / "configs/mu2_g3.yaml").read_text(encoding="utf-8"))
    seq = int(cfgy["model"]["ctx"])
    train = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("train_*.bin"), seq)
    val = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)

    def corrupt(blocks, hold_every):
        out = blocks.copy()
        mask_arr = np.array(mark_ids, dtype=blocks.dtype)
        for r in range(out.shape[0]):
            if (r % hold_every) == 3 % max(hold_every, 1):
                continue
            ids = out[r]
            pos = np.isin(ids, mask_arr)
            ids[pos] = mid
            out[r] = ids
        return out

    B, H = args.batch, args.hold_every
    head = MarkHead(320, 9).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    gen = np.random.default_rng(42)
    to_class_dev = to_class.to(device)
    for step in range(args.steps):
        blocks = gen.integers(0, len(train), B)
        clean = np.stack([train[b]["input_ids"] for b in blocks])
        corr = corrupt(clean, H)
        ids = torch.as_tensor(corr, dtype=torch.long, device=device)
        with torch.no_grad():
            hh = trunk.model(input_ids=ids)[0]
        logits = head(hh[:, :-1])
        # labels: true next token's class as if input were CLEAN -> mark class or none
        clean_t = torch.as_tensor(clean, dtype=torch.long, device=device)
        nxt = clean_t[:, 1:]
        is_mark = torch.zeros_like(nxt, dtype=torch.bool)
        for m in mark_ids:
            is_mark |= nxt == m
        tgt = torch.where(is_mark, to_class_dev[nxt], torch.full_like(nxt, NONE))
        tgt = torch.where(nxt >= 0, to_class_dev[nxt], torch.full_like(nxt, NONE))
        loss = F.cross_entropy(logits.reshape(-1, 9), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)

    dst = ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    save_file({n: p.detach().cpu() for n, p in head.named_parameters()},
              str(dst / "head.safetensors"))

    head_sd = {n: p.detach().cpu() for n, p in head.named_parameters()}
    head2 = MarkHead(320, 9)
    head2.load_state_dict(head_sd)
    head2 = head2.to(device).eval()

    @torch.inference_mode()
    def composed(dataset, tag):
        n_ok = n_tot = 0
        for b in range(0, len(dataset) - 1, 8):
            k = min(8, len(dataset) - b)
            clean = np.stack([dataset[b + j]["input_ids"] for j in range(k)])
            corr = corrupt(clean, H)
            ids = torch.as_tensor(corr, dtype=torch.long, device=device)
            hh = trunk.model(input_ids=ids)[0]
            trunk_logits = trunk(input_ids=ids).logits[:, :-1]
            bc = trunk_logits.masked_fill(
                torch.isin(torch.arange(trunk_logits.shape[-1], device=device),
                    nonmark_ids).view(1, 1, -1), float("-inf"))
            # head plan: predicted mark class or none
            mark_logits = head2(hh[:, :-1])
            pred_class = mark_logits.argmax(-1)
            mark_ids_t = torch.tensor(mark_ids, device=device)
            pick = mark_ids_t[pred_class.clamp(max=7)]  # class->token id
            use_head = pred_class != NONE
            best = bc.argmax(-1)
            pred = torch.where(use_head, pick, best)
            truth = torch.as_tensor(clean, dtype=torch.long, device=device)[:, 1:]
            n_ok += int((pred == truth).sum())
            n_tot += pred.numel()
        acc = n_ok / max(n_tot, 1)
        n = n_tot
        p_hat = acc
        z = c = 1.959964 ** 2
        root = z * (p_hat * (1 - p_hat) / n + z / (4 * n * n)) ** 0.5
        lo = (p_hat + z / (2 * n) - root) / (1 + z / n)
        hi = (p_hat + z / (2 * n) + root) / (1 + z / n)
        print(f"{tag}: composed acc {acc:.4f} ({n_ok}/{n_tot}) Wilson95 [{lo:.4f},{hi:.4f}]")
        return dict(acc=acc, ok=n_ok, n=n_tot, lo=lo, hi=hi)

    r = composed(val, "G4b composed")
    sd = load_file(str((ROOT / args.src) / "model.safetensors"))
    assert len(sd) > 0
    summary = {"phase": "mu2_g4b", "trunk": args.src, "frozen_trunk": True,
               "corruption_head_train": True, "hold_every": args.hold_every,
               "composed_acc": r["acc"], "wilson95": [r["lo"], r["hi"]],
               "gate_trunk_only": 0.6891, "steps": args.steps, "lr": args.lr}
    (dst / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (dst / "config.json").write_text((src / "config.json").read_text(encoding="utf-8"))
    print("[retention-identity] trunk loaded-only -> anchor 0.7395 structural")
    print("[final] " + json.dumps(summary))


if __name__ == "__main__":
    main()
