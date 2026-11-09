"""C12 Tier 1: teacher-trace SFT of a base model on tokenized Evol-Instruct.

Loads runs/target/final (the M3 base model) and fine-tunes it on the dataset
written by scripts/sft_data.py, using HF Trainer + a hand-rolled padding
collator with prompt-masked labels (DataCollatorForSeq2Seq-style, no new
dependencies).

Auto-resume contract (PLAN.md §5.3) - identical to scripts/train.py: every
start scans runs/sft_t1/ for checkpoint-* dirs and continues from the newest
with no flags. Relaunching this exact command after ANY interruption is all
it takes.

GATE (HANDOFF §8.2/§8.3): only run after M3 completes (runs/target/final
exists) and never while another training run is live on the GPU.

Run: .venv/Scripts/python scripts/sft.py --config configs/sft_t1.yaml [--pilot]
--pilot isolates the plumbing-proof run: it writes checkpoints + final to
<output_dir>_pilot (e.g. runs/sft_t1_pilot) so the FULL run always starts
clean in output_dir (auto-resume would otherwise continue pilot checkpoints
with a mismatched dataset/LR schedule).
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

import torch  # noqa: E402  (module-level: SftCollator builds tensors)


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


def apply_vram_cap(gib: float) -> dict:
    """Hard allocator ceiling (AGENTS 20.3 / the 2026-09-14 WDDM spill incident).

    Mirrors glm-5.3-flash-from-scratch/scripts/train_pretrain.py +
    glm53_flash.runtime.apply_vram_cap: fraction = cap / total device memory,
    clamped to 1.0. A no-op on CPU or when the cap is unset (0), so the default
    path is byte-for-byte unchanged. Must run BEFORE the model is moved to the
    GPU so the ceiling is in place before the first CUDA allocation.
    """
    if not gib or gib <= 0 or not torch.cuda.is_available():
        return {}
    device = torch.device("cuda", torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, (gib * 1024**3) / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return {"vram_cap_gib": gib, "vram_cap_fraction": round(fraction, 4)}


class SftCollator:
    """Pad input_ids/labels/attention_mask to the longest row in the batch.

    Pad positions get label -100 so they never contribute to the loss. The
    trained eos sits INSIDE the sequence and is unaffected even when
    pad_token == eos (masking is positional, not by token id).
    """

    def __init__(self, pad_id: int):
        self.pad_id = int(pad_id)

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lbl = list(f["labels"])
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lbl + [-100] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot", action="store_true",
                    help="plumbing-proof run: first sft.pilot_rows pairs, 1 epoch")
    ap.add_argument("--vram-cap-gib", type=float, default=0.0,
                    help="hard per-process CUDA allocator ceiling in GiB "
                         "(0 = no cap, the default; AGENTS 20.3). Applied before "
                         "the model is moved to the GPU.")
    args = ap.parse_args()

    import yaml
    from datasets import load_from_disk
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    from src.model import maybe_wrap_peft, save_final

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg, d, t = cfg["tokenizer"], cfg["data"], cfg["train"]
    sft = cfg["sft"]
    ctx = int(sft["ctx"])

    base_dir = ROOT / cfg["base_model"]
    if not base_dir.is_dir():
        raise SystemExit(
            f"[sft] base model not found: {base_dir}\n"
            "[sft] GATE: Tier 1 starts only after M3 completes (runs/target/final,\n"
            "HANDOFF §8.2) and never while another training run is live on the GPU.")

    data_dir = ROOT / d["dataset_dir"]
    if not (data_dir / "train").is_dir():
        raise SystemExit(f"[sft] tokenized SFT dataset not found: {data_dir}\n"
                         "[sft] run scripts/sft_data.py first")
    train_ds = load_from_disk(str(data_dir / "train"))
    val_ds = load_from_disk(str(data_dir / "val"))
    if args.pilot:
        pilot_rows = int(sft.get("pilot_rows", 5000))
        train_ds = train_ds.select(range(min(pilot_rows, len(train_ds))))
        epochs = 1
        print(f"[sft] PILOT: {len(train_ds)} pairs, 1 epoch", flush=True)
    else:
        epochs = int(t.get("epochs", 2))
    print(f"[sft] pairs: train={len(train_ds)} val={len(val_ds)} ctx={ctx} "
          f"epochs={epochs}", flush=True)

    tok = AutoTokenizer.from_pretrained(tcfg["name"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        print("[sft] tokenizer has no pad token -> pad = eos", flush=True)

    cap = apply_vram_cap(args.vram_cap_gib)
    if cap:
        print(f"[sft] VRAM cap: {cap}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(str(base_dir),
                                                 attn_implementation="sdpa")
    # Turing (sm_75) has NO bf16. SmolLM2's config declares torch_dtype=bfloat16 and
    # from_pretrained honors it, overriding our fp16 setting. Two failures followed:
    #   (1) model left in bf16  -> NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda for 'BFloat16'
    #   (2) model forced to fp16 -> ValueError: Attempting to unscale FP16 gradients.
    # The correct AMP contract is fp32 MASTER WEIGHTS + autocast: Trainer(fp16=True) supplies the
    # autocast and GradScaler, and the scaler requires unscaleable fp32 gradients. So keep the
    # model in fp32; never cast it to a half precision here.
    if torch.cuda.is_available():
        model = model.to(torch.float32)
    model.config.use_cache = False
    model = maybe_wrap_peft(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sft] base={base_dir} params={n_params / 1e6:.1f}M fp16={t['fp16']} "
          f"grad_ckpt={t['grad_ckpt']}", flush=True)

    output_dir = ROOT / t["output_dir"]
    final_dir = ROOT / t["final_dir"]
    if args.pilot:
        # Isolate the pilot: auto-resume scans output_dir for checkpoint-*, so
        # pilot scratch must live OUTSIDE the full-run tree - otherwise the
        # full run would resume into pilot checkpoints with a mismatched
        # dataset and LR schedule. No manual Move-Item needed.
        output_dir = output_dir.with_name(output_dir.name + "_pilot")
        final_dir = output_dir / "final"
        print(f"[sft] pilot output: {output_dir}", flush=True)
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(output_dir / "logs"))
    resume_from = find_latest_checkpoint(output_dir)
    if resume_from is not None:
        print(f"[resume] found {resume_from.name} -> auto-resume enabled", flush=True)
    if clear_stop_flag(output_dir):
        print("[stop] cleared stale STOP flag -> continuing", flush=True)

    targs = TrainingArguments(
        output_dir=str(output_dir),
        seed=int(t.get("seed", 42)),
        num_train_epochs=float(epochs),
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
    )

    trainer = Trainer(
        callbacks=[CoopStopCallback()],
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=SftCollator(tok.pad_token_id),
        processing_class=tok,
    )

    # Pass the PATH, not True: HF's internal discovery ignores the
    # completeness guard and would pick a partial checkpoint (seen live
    # 2026-09-08: sleep killed a save mid-write -> FileNotFoundError).
    trainer.train(resume_from_checkpoint=str(resume_from)
                  if resume_from is not None else None)

    # §5.3.5 keep-final-forever: in-memory model is the BEST checkpoint
    # (load_best_model_at_end). PeftModel is merged to a full model first;
    # saved outside the save_total_limit rotation.
    final_dir.mkdir(parents=True, exist_ok=True)
    save_final(trainer, final_dir)
    tok.save_pretrained(str(final_dir))

    final_metrics = trainer.evaluate()
    summary = {
        "phase": cfg["name"],
        "base_model": cfg["base_model"],
        "pilot": bool(args.pilot),
        "pairs": len(train_ds),
        "params_m": round(n_params / 1e6, 2),
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "final_eval": final_metrics,
        "resumed_from": resume_from.name if resume_from is not None else None,
        **cap,
    }
    (final_dir / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print("[final] summary: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
