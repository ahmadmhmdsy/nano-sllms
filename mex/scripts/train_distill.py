"""Train the committee-distilled student (MU1 arm D, EXPERIMENTS row E-29).

Identical mechanics to scripts/train.py (auto-resume with zero flags on
rerun, cosine schedule, warmup 20, fp16, adamw_bnb_8bit, eval/save rotation,
keep-final-forever). The ONLY difference is the training loss, supplied
through a KDTrainer.compute_loss hook (train.py itself uses the default
Trainer loss, so the hook lives here rather than forking train.py's loop):

    loss = alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))
         + (1 - alpha) * CE(student, labels)

Teacher logits come from the PRECOMPUTED cache written by
mex/scripts/distill_cache.py (memmap aligned to the PackedDataset blocks of
data/mex/control/tokens/train_*.bin, same sorted shard order). NO teacher
model is loaded during training. At a cache miss the block's teacher row
falls back to the block's own student logits, which makes the KL term
exactly 0 for that block (pure CE) - graceful degradation, never a crash.

CLI overrides (--max-steps / --output-dir / --final-dir / --logging-steps)
exist ONLY for the 20-step smoke; the production command is the plain
--config call exactly like train.py's.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Import scripts/train.py as a module (scripts/ is not a package): reuse its
# checkpoint finder and env fingerprint instead of forking them.
_spec = importlib.util.spec_from_file_location(
    "root_train", str(ROOT / "scripts" / "train.py"))
root_train = importlib.util.module_from_spec(_spec)
sys.modules["root_train"] = root_train
_spec.loader.exec_module(root_train)

find_latest_checkpoint = root_train.find_latest_checkpoint
clear_stop_flag = root_train.clear_stop_flag
_env_fingerprint = root_train._env_fingerprint

from src.data import PackedDataset  # noqa: E402  (same class train.py packs with)

import numpy as np  # noqa: E402


class IndexedPackedDataset(PackedDataset):
    """PackedDataset + the block index so the KD loss can look up the
    teacher-logit cache row. default_data_collator turns the int into a
    tensor column; compute_loss pops it before the forward."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["block_idx"] = int(idx)
        return item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="path to configs/mex_distill.yaml")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke override: 0 = config max_steps")
    ap.add_argument("--output-dir", default="",
                    help="smoke override for train.output_dir")
    ap.add_argument("--final-dir", default="",
                    help="smoke override for train.final_dir")
    ap.add_argument("--logging-steps", type=int, default=0,
                    help="smoke override: 0 = config logging_steps")
    args = ap.parse_args()

    import torch
    import yaml
    from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                              default_data_collator)

    from mex.src.kd import TeacherLogitCache, kd_loss
    from src.model import build_model, save_final

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg, d, t, k = cfg["tokenizer"], cfg["data"], cfg["train"], cfg["kd"]
    seq_len = int(cfg["model"]["ctx"])
    if args.max_steps > 0:
        t = dict(t, max_steps=args.max_steps)
    if args.logging_steps > 0:
        t = dict(t, logging_steps=args.logging_steps)
    if args.output_dir:
        t = dict(t, output_dir=args.output_dir)
    if args.final_dir:
        t = dict(t, final_dir=args.final_dir)

    alpha = float(k.get("alpha", 0.5))
    temperature = float(k.get("temperature", 1.0))

    tok = AutoTokenizer.from_pretrained(tcfg["name"])
    train_ds = IndexedPackedDataset(
        (ROOT / d["tokens_dir"]).glob("train_*.bin"), seq_len)
    val_ds = IndexedPackedDataset(
        (ROOT / d["tokens_dir"]).glob("val_*.bin"), seq_len)
    print(f"[train_distill] blocks: train={len(train_ds)} "
          f"val={len(val_ds)} seq_len={seq_len}", flush=True)

    # Teacher-logit cache ONLY - no teacher model is ever loaded here.
    # Alignment proof: distill_cache.py walks the same sorted shards with the
    # same PackedDataset class and seq_len, so row i == dataset block i.
    cache_path = ROOT / k["teacher_cache"]
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"{cache_path} missing - run mex/scripts/distill_cache.py first")
    cache = TeacherLogitCache(cache_path, seq_len=seq_len,
                              vocab=int(tcfg["vocab_size"]))
    if cache.n_blocks != len(train_ds):
        print(f"[kd] WARN cache rows {cache.n_blocks} != train blocks "
              f"{len(train_ds)}; uncovered blocks train as pure CE", flush=True)
    else:
        print(f"[kd] cache aligned: {cache.n_blocks} rows == train blocks "
              f"({cache.n_blocks * seq_len} tokens)", flush=True)

    vocab = max(int(tcfg["vocab_size"]), len(tok))
    model = build_model(cfg, vocab_size=vocab)  # Trainer moves it to CUDA
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_distill] params={n_params / 1e6:.1f}M vocab={vocab} "
          f"fp16={t['fp16']} grad_ckpt={t['grad_ckpt']} alpha={alpha} "
          f"T={temperature}", flush=True)

    output_dir = ROOT / t["output_dir"]
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(output_dir / "logs"))
    resume_from = find_latest_checkpoint(output_dir)
    if resume_from is not None:
        print(f"[resume] found {resume_from.name} -> auto-resume enabled",
              flush=True)
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
        remove_unused_columns=False,  # keep block_idx
    )

    class KDTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kwargs):
            block_idx = inputs.pop("block_idx")
            input_ids = inputs["input_ids"]
            labels = inputs["labels"]
            out = model(input_ids=input_ids)
            logits = out.logits
            rows = []
            for i in block_idx.tolist():
                i = int(i)
                if 0 <= i < cache.n_blocks:
                    rows.append(torch.as_tensor(
                        np.asarray(cache.block_logits(i)),
                        dtype=torch.float32, device=logits.device))
                else:  # miss -> teacher == student => KL==0 (pure CE block)
                    rows.append(logits[i].float().detach())
            teacher = torch.stack(rows)
            loss = kd_loss(logits, teacher, labels, alpha=alpha,
                           temperature=temperature)
            return (loss, out) if return_outputs else loss

    trainer = KDTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        callbacks=[root_train.CoopStopCallback()],
        processing_class=tok,
    )

    # Pass the PATH, not True: HF's internal discovery ignores the
    # completeness guard and would pick a partial checkpoint (train.py
    # comment; same unattended-safe contract).
    trainer.train(resume_from_checkpoint=str(resume_from)
                  if resume_from is not None else None)

    final_dir = ROOT / t["final_dir"]
    final_dir.mkdir(parents=True, exist_ok=True)
    save_final(trainer, final_dir)
    tok.save_pretrained(str(final_dir))

    final_metrics = trainer.evaluate()
    summary = {
        "phase": cfg["name"],
        "params_m": round(n_params / 1e6, 2),
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "final_eval": final_metrics,
        "resumed_from": resume_from.name if resume_from is not None else None,
        "kd": {"alpha": alpha, "temperature": temperature,
               "teacher_cache": k["teacher_cache"],
               "cache_rows": cache.n_blocks},
        "env": _env_fingerprint(cfg),
    }
    (final_dir / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print("[final] summary: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
