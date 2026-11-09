# mex/scripts/merge_soup.py — mu1 Arm A: weight-merge the four 12K expert
# finals into one dense model (uniform soup + TIES sign-election, E-27).
r"""Merge runs/mex/archive_12000/{x1,x2,x3,x4}/final into a single soup model.

Two merges, both computed in fp32 float and saved mirroring the experts'
checkpoint dtype (fp32 on disk despite the fp16 training flag):

uniform  plain mean of every weight tensor.
ties     standard TIES (Yadav et al. 2023), density 50%:
         1. w_bar = mean over experts of each weight tensor (task baseline).
         2. delta_i = w_i - w_bar; trim: per expert tensor keep the top-50%
            magnitudes (>= threshold), drop the rest.
         3. sign election per element: elected = sign(sum(sign(delta_surv)))
            (all/split ties -> 0, no contribution).
         4. merged_delta = mean of the surviving deltas bearing the elected
            sign; output = w_bar + merged_delta.

Tied embeddings (tie_word_embeddings true): lm_head.weight shares
embed_tokens storage and is absent from safetensors saves, so each weight
tensor is merged exactly once and loads back clean under eval_mex.load()'s
strict contract. Output layout mirrors a final dir: config.yaml (byte copy
of configs/mex_x1.yaml — identical arch for all experts), model.safetensors,
tokenizer + generation files copied; no training leftovers are carried.

Usage:
    & .\.venv\Scripts\python.exe mex\scripts\merge_soup.py [uniform] [ties]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
EXPERTS = ("x1", "x2", "x3", "x4")
ARCHIVE = ROOT / "runs" / "mex" / "archive_12000"
CONFIG_SRC = ROOT / "configs" / "mex_x1.yaml"
COPY_FILES = ("tokenizer.json", "tokenizer_config.json",
              "generation_config.json")
REF_FINAL = ARCHIVE / "x1" / "final"


def load_experts() -> dict[str, dict[str, torch.Tensor]]:
    """One fp32 state dict per expert; verify identical keys across experts."""
    states: dict[str, dict[str, torch.Tensor]] = {}
    for t in EXPERTS:
        state = load_file(str(ARCHIVE / t / "final" / "model.safetensors"))
        states[t] = {k: v.float() for k, v in state.items()}
    ref_keys = set(states[EXPERTS[0]])
    for t in EXPERTS[1:]:
        if set(states[t]) != ref_keys:
            missing = ref_keys - set(states[t])
            extra = set(states[t]) - ref_keys
            raise RuntimeError(f"key mismatch {EXPERTS[0]} vs {t}: "
                               f"missing={sorted(missing)} extra={sorted(extra)}")
    return states


def uniform_merge(states: dict[str, dict[str, torch.Tensor]]
                  ) -> dict[str, torch.Tensor]:
    """Plain mean of the expert weights."""
    merged = {}
    for key in sorted(states[EXPERTS[0]]):
        stacked = torch.stack([states[t][key] for t in states])
        merged[key] = stacked.mean(0)
    return merged


def ties_merge(states: dict[str, dict[str, torch.Tensor]], density: float = 0.5
               ) -> dict[str, torch.Tensor]:
    """TIES: trim top-density deltas per expert, elect signs, mean survivors."""
    keys = sorted(states[EXPERTS[0]])
    merged = {}
    for key in keys:
        stacked = torch.stack([states[t][key] for t in sorted(states)])
        base = stacked.mean(0)                                   # w_bar
        deltas = stacked - base
        flat = deltas.reshape(deltas.shape[0], -1).abs()          # (K, elem)
        thresh = torch.quantile(flat, 1.0 - density, dim=1).view(-1, 1)
        surviving = (flat >= thresh).reshape(deltas.shape) & (deltas != 0)
        votes = torch.where(surviving, torch.sign(deltas),
                            torch.zeros_like(deltas)).sum(0)
        elected = torch.sign(votes)         # 0 where tied/empty -> no merge
        pick = surviving & (torch.sign(deltas) == elected.unsqueeze(0))
        contrib = torch.where(pick, deltas, torch.zeros_like(deltas)).sum(0)
        count = pick.sum(0)
        merged_delta = torch.where(count > 0, contrib / count,
                                   torch.zeros_like(contrib))
        merged[key] = base + merged_delta
        if torch.isnan(merged[key]).any():
            raise RuntimeError(f"NaN in merged tensor {key!r}")
    return merged


def save_soup(merged: dict[str, torch.Tensor], dest: Path) -> None:
    """Write dest/{config.yaml, model.safetensors, tokenizer/generation files}."""
    dest.mkdir(parents=True, exist_ok=True)
    ref = load_file(str(REF_FINAL / "model.safetensors"))
    dtype = next(iter(ref.values())).dtype
    save_file({k: v.to(dtype).contiguous() for k, v in merged.items()},
              str(dest / "model.safetensors"))
    shutil.copyfile(CONFIG_SRC, dest / "config.yaml")
    for name in COPY_FILES:
        shutil.copyfile(REF_FINAL / name, dest / name)
    print(f"saved {dest / 'model.safetensors'} (dtype {dtype}, "
          f"{sum(v.numel() for v in merged.values())} params)")


def main() -> None:
    merges = [a for a in sys.argv[1:]] or ["uniform", "ties"]
    bad = [m for m in merges if m not in ("uniform", "ties")]
    if bad:
        raise SystemExit(f"unknown merge(s) {bad}; options: uniform, ties")
    if len(set(merges)) != len(merges):
        raise SystemExit("duplicate merge names")
    states = load_experts()
    for name in merges:
        merged = (uniform_merge if name == "uniform" else ties_merge)(states)
        save_soup(merged, ROOT / "runs" / "mex" / "soup" / name)


if __name__ == "__main__":
    main()
