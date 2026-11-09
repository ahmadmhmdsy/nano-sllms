"""C12 Tier 1 pre-launch gate - run BEFORE scripts/sft.py (research/c12_runbook.md).

    .venv/Scripts/python scripts/c12_preflight.py             # full-SFT mode (launch gate)
    .venv/Scripts/python scripts/c12_preflight.py --pilot     # pilot-mode disk bar
    .venv/Scripts/python scripts/c12_preflight.py --base-model runs/target/checkpoint-500
        # ^ tool dry-run BEFORE M3 completes: exercises every check against an
        # existing target-format checkpoint; the base-model and GPU checks will
        # (correctly) FAIL while M3 is still live.

Checks (PASS / FAIL / WARN, exit 1 on any hard FAIL):
  1. Base model present + loadable (default runs/target/final = the M3 gate):
     CPU load, param count ~226.5M, ctx >= 1024.
  2. GPU idle enough for SFT (>= 5.5 GB free) - FAIL is EXPECTED while M3 trains;
     never co-run SFT with a live training run (plan hard rule).
  3. Tokenized SFT dataset intact: data/sft/evol/ds train/val row counts match
     data/sft/evol/meta.json; rows 0-1 show prompt masking (-100 prefix) with a
     trained eos as the final label.
  4. Held-out instructions >= eval.n_instructions (50).
  5. Forgetting-guard inputs: data/target/tokens/val_*.bin exist (eval.py reads
     them via configs/sft_t1.yaml data.tokens_dir).
  6. Disk: >= 4.5 GB free (pilot) / >= 10.0 GB free (full). Full mode realistically
     also needs runs/target/checkpoint-* deleted after M3 final exists (user-approved,
     M2 precedent) because M3 leaves ~9 GB of checkpoints behind.
  7. WARN if runs/sft_t1/checkpoint-* exist: auto-resume would continue them;
     archive before a FRESH start (runbook section 5).

No new dependencies: torch / transformers / datasets / pyyaml only.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PARAMS_TARGET = 226.5e6   # M1-verified target config size
GPU_FREE_MIN_GB = 5.5
DISK_MIN_GB = {"pilot": 4.5, "full": 10.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="C12 Tier 1 launch gate")
    ap.add_argument("--pilot", action="store_true",
                    help="pilot mode: lower disk bar (the runs/sft_t1 --pilot run)")
    ap.add_argument("--base-model", default="runs/target/final",
                    help="base model dir to validate (default runs/target/final; "
                         "e.g. runs/target/checkpoint-500 for a pre-M3 dry run)")
    args = ap.parse_args()
    mode = "pilot" if args.pilot else "full"

    fails = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""),
              flush=True)
        if not ok:
            fails.append(name)

    def warn(name, detail):
        print(f"[WARN] {name} - {detail}", flush=True)

    # --- 1. base model (the M3 gate) ----------------------------------------
    base_dir = Path(args.base_model)
    if not base_dir.is_absolute():
        base_dir = ROOT / base_dir
    if (base_dir / "model.safetensors").is_file() and (base_dir / "config.json").is_file():
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(str(base_dir))
            n_params = sum(p.numel() for p in model.parameters())
            ctx = int(getattr(model.config, "max_position_embeddings", 0) or 0)
            del model
            ok = abs(n_params - PARAMS_TARGET) / PARAMS_TARGET < 0.02 and ctx >= 1024
            check("base model loads", ok,
                  f"{base_dir.name}: {n_params / 1e6:.1f}M params (target ~226.5M), ctx {ctx}")
        except Exception as exc:  # noqa: BLE001 - report, do not crash the gate
            check("base model loads", False, f"load failed: {exc}")
    else:
        check("base model present (M3 gate)", False,
              f"{base_dir} missing model.safetensors/config.json - M3 not finished yet "
              "(EXPECTED while M3 trains; hard blocker at launch)")

    # --- 2. GPU idle enough ---------------------------------------------------
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()[0]
        free_mb, total_mb = (int(x.strip()) for x in out.split(","))
        check("gpu free for SFT", free_mb / 1024 >= GPU_FREE_MIN_GB,
              f"{free_mb / 1024:.1f}/{total_mb / 1024:.1f} GB free "
              f"(need >= {GPU_FREE_MIN_GB}); FAIL is EXPECTED while M3 trains")
    except Exception as exc:  # noqa: BLE001
        try:
            import torch
            free_b, total_b = torch.cuda.mem_get_info()
            check("gpu free for SFT", free_b / 2**30 >= GPU_FREE_MIN_GB,
                  f"(torch fallback) {free_b / 2**30:.1f}/{total_b / 2**30:.1f} GB free")
        except Exception:
            check("gpu free for SFT", False, f"nvidia-smi and torch both failed: {exc}")

    # --- 3-6. data / config / disk -------------------------------------------
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "sft_t1.yaml").read_text(encoding="utf-8"))

    meta_path = ROOT / cfg["data"]["raw_dir"] / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    from datasets import load_from_disk
    ds_dir = ROOT / cfg["data"]["dataset_dir"]
    train = load_from_disk(str(ds_dir / "train"))
    val = load_from_disk(str(ds_dir / "val"))
    check("sft dataset rows match meta.json",
          len(train) == meta["train_pairs"] and len(val) == meta["val_pairs"],
          f"train={len(train)} val={len(val)} "
          f"(meta: {meta['train_pairs']}+{meta['val_pairs']})")

    masking_ok, mask_detail = True, []
    for i in range(2):
        lbl = list(train[i]["labels"])
        n_mask = sum(1 for x in lbl if x == -100)
        mask_detail.append(f"row{i}: -100 {n_mask}/{len(lbl)}, last label {lbl[-1]}")
        if lbl[0] != -100 or lbl[-1] == -100 or n_mask < 1:
            masking_ok = False
    check("prompt masking intact (-100 prefix + trained eos)", masking_ok,
          "; ".join(mask_detail) + f" (meta eos id {meta['eos_token_id']})")

    instr_path = ROOT / cfg["eval"]["instructions_file"]
    n_instr = sum(1 for line in instr_path.read_text(encoding="utf-8").splitlines()
                  if line.strip())
    check("held-out instructions available",
          n_instr >= int(cfg["eval"]["n_instructions"]),
          f"{n_instr} rows >= {cfg['eval']['n_instructions']} ({instr_path.name})")

    shards = sorted((ROOT / cfg["data"]["tokens_dir"]).glob("val_*.bin"))
    check("forgetting-guard CSN val shards", len(shards) > 0,
          f"{len(shards)} shard(s) in {cfg['data']['tokens_dir']} "
          "(eval.py packs them at ctx 1024)")

    free_gb = shutil.disk_usage(ROOT).free / 2**30
    need_gb = DISK_MIN_GB[mode]
    hint = ("3 rotating checkpoints (~2.7 GB each) + ~0.9 GB final; after M3 completes, "
            "user-approved deletion of runs/target/checkpoint-* frees ~8 GB (M2 precedent)")
    check(f"disk free for {mode}", free_gb >= need_gb,
          f"{free_gb:.1f} GB free, need >= {need_gb}: {hint}")

    # --- 7. stale SFT state ----------------------------------------------------
    sft_out = ROOT / cfg["train"]["output_dir"]
    stale = sorted(p.name for p in sft_out.glob("checkpoint-*")) if sft_out.is_dir() else []
    if stale:
        warn("stale SFT checkpoints",
             f"{', '.join(stale)} exist in {cfg['train']['output_dir']} - auto-resume "
             "will continue them; archive (Move-Item) before a FRESH start, runbook section 5")
    else:
        print("[PASS] runs/sft_t1 clean (no stale checkpoints)", flush=True)

    print("", flush=True)
    if fails:
        print(f"GATE: NOT READY - {len(fails)} hard FAIL(s): {', '.join(fails)}", flush=True)
        print("Do not launch scripts/sft.py until every FAIL above is resolved.", flush=True)
        sys.exit(1)
    print(f"GATE: READY ({mode} mode) - safe to launch scripts/sft.py "
          + ("--pilot" if args.pilot else ""), flush=True)


if __name__ == "__main__":
    main()
