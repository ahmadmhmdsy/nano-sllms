#!/usr/bin/env python3
"""Weight-space soup / wiSE-FT interpolation between sibling checkpoints.

Row 41 P0 (user-approved 2026-09-10): merge e1 (full SFT) with yarn_4096 at
several alphas and measure where the guard/instruct tradeoff lands. Also the
post-hoc knob for any later SFT model (LoRA-merged vs e1, etc.).

All ingredient checkpoints must share ONE architecture (same lineage). The
output dir ships the --config-from donor's config.json + tokenizer files so
from_pretrained applies THAT model's rope/ctx family (yarn here). CPU-only;
never touches the GPU. Weights are normalized over the given mix weights.

Usage:
  & .\.venv\Scripts\python.exe scripts\soup_merge.py \
      --mix runs/yarn_4096/final=0.75 --mix runs/sft_v2_e1/final=0.25 \
      --config-from runs/yarn_4096/final --out runs/soup_p0/alpha025
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "generation_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mix", action="append", required=True,
                    help="DIR=WEIGHT checkpoint ingredients (repeatable)")
    ap.add_argument("--config-from", required=True,
                    help="donor dir for config.json + tokenizer files")
    ap.add_argument("--out", required=True, help="output model dir")
    args = ap.parse_args()

    ingredients: list[tuple[Path, float]] = []
    for spec in args.mix:
        path_s, _, w_s = spec.rpartition("=")
        path, weight = Path(path_s), float(w_s)
        if not (path / "model.safetensors").exists():
            raise SystemExit(f"missing model.safetensors under {path}")
        if weight < 0:
            raise SystemExit(f"negative weight for {path}")
        ingredients.append((path, weight))
    total = sum(w for _, w in ingredients)
    if total <= 0:
        raise SystemExit("mix weights sum to zero")
    ingredients = [(p, w / total) for p, w in ingredients]

    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing {out_dir}")
    out_dir.mkdir(parents=True)

    acc: dict[str, torch.Tensor] | None = None
    for path, weight in ingredients:
        sd = load_file(str(path / "model.safetensors"))
        if acc is None:
            acc = {k: v.to(torch.float32) * weight for k, v in sd.items()}
        else:
            if set(sd) != set(acc):
                missing = set(acc) ^ set(sd)
                raise SystemExit(f"tensor-key mismatch with {path}: {sorted(missing)[:4]}")
            for k, v in sd.items():
                if acc[k].shape != v.shape:
                    raise SystemExit(f"shape mismatch {k}: {acc[k].shape} vs {v.shape}")
                acc[k] += v.to(torch.float32) * weight
        print(f"[soup] + {weight:.3f} x {path}")

    donor = Path(args.config_from)
    save_file(acc, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    shutil.copy2(donor / "config.json", out_dir / "config.json")
    for name in TOKENIZER_FILES:
        src = donor / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    n = len(acc)
    print(f"[soup] wrote {n} tensors -> {out_dir} "
          f"(dtype fp32, config/tokenizer from {donor})")


if __name__ == "__main__":
    main()
