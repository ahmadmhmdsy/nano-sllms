"""Train one phase (smoke/pilot/target): auto-checkpoint + AUTO-RESUME.

Auto-resume contract (PLAN.md §5.3): every start scans runs/<phase>/ for
checkpoint-* dirs. If any exist, trainer.train(resume_from_checkpoint=True)
continues from the newest -- no flags, no manual paths. Relaunching this
exact command after a crash/kill is all it takes.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.stop import CoopStopCallback, clear_stop_flag  # noqa: E402


def find_latest_checkpoint(output_dir: Path):
    if not output_dir.is_dir():
        return None
    best = None
    for p in output_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        # trainer_state.json is written LAST by the saver; its presence
        # proves the checkpoint is complete (a kill mid-write must not
        # crash auto-resume - unattended-safe requirement, PLAN 5.3).
        if m and p.is_dir() and (p / "trainer_state.json").is_file():
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
    return best[1] if best else None


def _env_fingerprint(cfg) -> dict:
    """Env fingerprint stamped into train_summary.json (Milestone B CPU part).

    Additive-only fields so the auto-resume contract (PLAN.md 5.3) is never
    affected; all lookups are guarded because a summary must never fail a run.
    """
    import platform
    import subprocess
    from datetime import datetime, timezone
    from importlib.metadata import version

    fp = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in ("torch", "transformers", "tokenizers", "datasets",
                "accelerate", "peft", "bitsandbytes"):
        try:
            fp[pkg] = version(pkg)
        except Exception:  # noqa: BLE001 - not installed is a valid state
            fp[pkg] = None
    try:
        fp["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001
        fp["git_commit"] = None
    try:
        fp["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        fp["gpu"] = None
    fp["resolved_config"] = cfg
    return fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to configs/<phase>.yaml")
    args = ap.parse_args()

    import torch
    import yaml
    from transformers import (AutoTokenizer, EarlyStoppingCallback, Trainer,
                              TrainingArguments, default_data_collator)

    from src.data import PackedDataset
    from src.model import (build_model, load_finetune_init, maybe_wrap_peft,
                           save_final)

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg, d, t = cfg["tokenizer"], cfg["data"], cfg["train"]
    seq_len = int(cfg["model"]["ctx"])

    # Hard VRAM ceiling (config key vram_cap_gib, same pattern as
    # diacritizer/scripts/train_lm.py). The torch allocator is pinned so the
    # process NEVER allocates past the dedicated VRAM - otherwise Windows WDDM
    # silently spills the tail into shared iGPU memory (~10x slowdown, the
    # 2026-09-14 WDDM incident). Set BEFORE the model/dataset touch CUDA.
    cap = t.get("vram_cap_gib")
    if cap and torch.cuda.is_available():
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        frac = min(0.999, float(cap) / total_gib)
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"[VRAM-CAP] {round(frac * total_gib, 2)} GiB of {round(total_gib, 2)} GiB",
              flush=True)

    tok = AutoTokenizer.from_pretrained(tcfg["name"])
    train_ds = PackedDataset((ROOT / d["tokens_dir"]).glob("train_*.bin"), seq_len)
    val_ds = PackedDataset((ROOT / d["tokens_dir"]).glob("val_*.bin"), seq_len)
    print(f"[train] blocks: train={len(train_ds)} val={len(val_ds)} seq_len={seq_len}",
          flush=True)

    vocab = max(int(tcfg["vocab_size"]), len(tok))
    model = build_model(cfg, vocab_size=vocab)
    # Track B (row 30): config-gated fine-tune init (defaults off). The base
    # apex is loaded into the NEW run's model; the base dir itself is never
    # touched in place (adoption_plan B base discipline).
    init_from = t.get("init_from")
    if init_from:
        diag = load_finetune_init(model, ROOT / init_from)
        print(f"[init] fine-tune init from {init_from}: "
              f"{diag['tensors']} tensors loaded", flush=True)
    # KT ladder Phase 1 (row 43): config-gated embedding transplant, DEFAULT
    # OFF. Applied AFTER init_from (embeddings override) and BEFORE
    # maybe_wrap_peft; the tied lm_head follows via _tied_weights_keys.
    # Auto-resume untouched: a resumed run overwrites every weight from its
    # checkpoint right after this, so the init only ever seeds step 0.
    init_embeddings = t.get("init_embeddings")
    if init_embeddings:
        blob = torch.load(ROOT / init_embeddings, map_location="cpu",
                          weights_only=True)
        emb = blob["embed_tokens"] if isinstance(blob, dict) else blob
        cur = model.get_input_embeddings().weight
        if tuple(emb.shape) != tuple(cur.shape):
            raise ValueError(
                f"[init] init_embeddings shape {tuple(emb.shape)} != model "
                f"{tuple(cur.shape)} - refusing (vocab/dim mismatch)")
        with torch.no_grad():
            cur.copy_(emb.to(cur.dtype))
        print(f"[init] embeddings transplanted from {init_embeddings} "
              f"({emb.shape[0]}x{emb.shape[1]} -> fp{8 * cur.element_size()}, "
              f"tied head follows)", flush=True)
    model = maybe_wrap_peft(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params / 1e6:.1f}M vocab={vocab} fp16={t['fp16']} "
          f"grad_ckpt={t['grad_ckpt']}", flush=True)

    print(f"[train] model summary:")
    print(f"  layers={model.config.num_hidden_layers} hidden={model.config.hidden_size}")
    print(f"  heads={model.config.num_attention_heads} kv_heads={model.config.num_key_value_heads}")
    print(f"  ffn={model.config.intermediate_size} ctx={model.config.max_position_embeddings}")
    print(f"  vocab={model.config.vocab_size} tie={model.config.tie_word_embeddings}")
    print(f"  params={n_params/1e6:.1f}M")

    import shutil
    free_gb = shutil.disk_usage(ROOT).free / (1024**3)
    estimated_need_gb = int(t["save_total_limit"]) * (n_params * 6 / 1e9) + 2.0
    if free_gb < estimated_need_gb:
        raise SystemExit(f"[train] LOW DISK: {free_gb:.1f} GB free, need ~{estimated_need_gb:.1f} GB for checkpoints")
    print(f"[train] disk check: {free_gb:.1f} GB free, need ~{estimated_need_gb:.1f} GB")

    output_dir = ROOT / t["output_dir"]
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(output_dir / "logs"))
    resume_from = find_latest_checkpoint(output_dir)
    if resume_from is not None:
        print(f"[resume] found {resume_from.name} -> auto-resume enabled", flush=True)
        ckpt_config = json.loads((resume_from / "config.json").read_text())
        current_config = model.config.to_dict()
        for key in ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"]:
            if ckpt_config.get(key) != current_config.get(key):
                raise SystemExit(f"[train] CONFIG MISMATCH on resume: {key} differs")
    if clear_stop_flag(output_dir):
        print("[stop] cleared stale STOP flag -> continuing", flush=True)

    targs = TrainingArguments(
        output_dir=str(output_dir),
        seed=int(t.get("seed", 42)),
        max_steps=int(t["max_steps"]),
        per_device_train_batch_size=int(t["batch"]),
        per_device_eval_batch_size=int(t.get("eval_batch", t["batch"])),
        gradient_accumulation_steps=int(t["accum"]),
        learning_rate=float(t["lr"]),
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type=t.get("scheduler", "cosine"),
        warmup_steps=int(t["warmup_steps"]),
        weight_decay=float(t.get("weight_decay", 0.1)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        logging_steps=int(t["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(t["eval_steps"]),
        save_strategy="steps",
        save_steps=int(t["save_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=bool(t.get("fp16", True)),
        gradient_checkpointing=bool(t.get("grad_ckpt", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t.get("optim", "adamw_torch"),
        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=['input_ids'],
        data_seed=int(t.get("seed", 42)),
    )

    trainer = Trainer(
        callbacks=[CoopStopCallback(), EarlyStoppingCallback(
            early_stopping_patience=int(t.get("early_stopping_patience", 3)))],
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        processing_class=tok,
    )

    # Pass the PATH, not True: HF's internal discovery ignores the
    # completeness guard and would pick a partial checkpoint (seen live
    # 2026-09-08: sleep killed a save mid-write -> FileNotFoundError).
    try:
        trainer.train(resume_from_checkpoint=str(resume_from)
                      if resume_from is not None else None)
    except torch.cuda.OutOfMemoryError:
        print("[train] OOM ERROR: Try reducing batch size, or set optim='adamw_bnb_8bit' in config")
        raise

    # §5.3.5 keep-final-forever: in-memory model is the BEST checkpoint
    # (load_best_model_at_end). PeftModel is merged to a full model first;
    # saved outside the save_total_limit rotation.
    final_dir = ROOT / t["final_dir"]
    final_dir.mkdir(parents=True, exist_ok=True)
    save_final(trainer, final_dir)
    from safetensors.torch import load_file
    final_ckpt = trainer.state.best_model_checkpoint
    if final_ckpt:
        shards = list(Path(final_ckpt).glob("*.safetensors"))
        for s in shards:
            _ = load_file(str(s))
        print(f"[train] checkpoint verification: {len(shards)} shards OK")
    tok.save_pretrained(str(final_dir))

    final_metrics = trainer.evaluate()
    metrics = trainer.state.log_history
    (final_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = {
        "phase": cfg["name"],
        "params_m": round(n_params / 1e6, 2),
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "final_eval": final_metrics,
        "resumed_from": resume_from.name if resume_from is not None else None,
        "env": _env_fingerprint(cfg),
    }
    (final_dir / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print("[final] summary: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
