"""C12 Tier 3 / Track C: intra-ladder logit KD - frozen teacher teaches a
smaller student on the same packed pretraining shards.

Loss = alpha * KL(teacher_probs || student_probs, tau) + (1-alpha) * CE,
standard distillation direction (match the teacher distribution; plan's
"KL(student || teacher)" notation, tau=1 default, alpha=0.5 default). The
fair A/B baseline (from-scratch student, identical steps/data/seed) is a
plain train.py run with a student-dims config - no KD code involved.

kd.skew_lambda (Track C2, default 0.0 = the exact plain path above): the
KD term becomes DistiLLM alpha-SKL (Ko et al. 2024, sec. 3.1) -
KL(t || m), m = skew*t + (1-skew)*s - same forward-KL direction, but the
mixture floor removes the -log q explosion that makes plain KLD unstable
in fp16. Paper ablation optimum: skew = 0.1. CPU-validated exact vs a
brute-force mixture-KL at skew in {0.1, 0.3, 0.9} (diff <= 2.4e-7).

Auto-resume contract (PLAN.md §5.3) - identical to scripts/train.py: scans
runs/<name>/ for complete checkpoint-* dirs and continues with no flags.
GATE: never while another training run is live on the GPU.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_latest_checkpoint(output_dir: Path):
    if not output_dir.is_dir():
        return None
    best = None
    for p in output_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        # trainer_state.json is written LAST by the saver; its presence
        # proves the checkpoint is complete (kill mid-write must not crash
        # auto-resume - unattended-safe requirement, PLAN 5.3).
        if m and p.is_dir() and (p / "trainer_state.json").is_file():
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
    return best[1] if best else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to configs/kd_*.yaml")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import yaml
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, default_data_collator)

    from src.data import PackedDataset
    from src.model import build_model, save_final

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg, d, t = cfg["tokenizer"], cfg["data"], cfg["train"]
    kd_cfg = cfg["kd"]
    seq_len = int(cfg["model"]["ctx"])
    tau = float(kd_cfg.get("tau", 1.0))
    alpha = float(kd_cfg.get("alpha", 0.5))
    skew = float(kd_cfg.get("skew_lambda", 0.0))
    if not (0.0 <= skew < 1.0):
        raise ValueError(f"kd.skew_lambda must be in [0, 1); got {skew}")

    tok = AutoTokenizer.from_pretrained(tcfg["name"])
    train_ds = PackedDataset((ROOT / d["tokens_dir"]).glob("train_*.bin"), seq_len)
    val_ds = PackedDataset((ROOT / d["tokens_dir"]).glob("val_*.bin"), seq_len)
    print(f"[kd] blocks: train={len(train_ds)} val={len(val_ds)} seq_len={seq_len}",
          flush=True)

    vocab = max(int(tcfg["vocab_size"]), len(tok))
    student = build_model(cfg, vocab_size=vocab)

    teacher_dir = ROOT / kd_cfg["teacher"]
    teacher = AutoModelForCausalLM.from_pretrained(str(teacher_dir),
                                                   attn_implementation="sdpa")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher.to(device)

    class KDTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            # Eval stays PURE CE: it is the A/B metric against the baseline
            # run and drives load_best_model_at_end - the KD blend must not
            # leak into it.
            if not self.model.training:
                return super().compute_loss(model, inputs,
                                            return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            ids = inputs["input_ids"]
            with torch.no_grad():
                t_logits = teacher(input_ids=ids).logits.float()
            out = model(input_ids=ids, labels=ids)
            if skew > 0.0:
                # DistiLLM alpha-SKL: KL(t || m), m = skew*t + (1-skew)*s.
                # Exact log-mixture via logaddexp; CHUNKED over the batch dim
                # so the [T, V] fp32 intermediates stay one-sequence small
                # (the full-batch form pushed the b8 probe to 7897/8192 MiB).
                V = out.logits.shape[-1]
                t_logp = F.log_softmax(t_logits / tau, dim=-1)
                t_logp = t_logp.view(ids.shape[0], -1, V)
                terms = []
                for b in range(ids.shape[0]):
                    t = t_logp[b]
                    s = F.log_softmax(out.logits[b].float() / tau, dim=-1)
                    log_m = torch.logaddexp(t + math.log(skew),
                                            s + math.log1p(-skew))
                    # F.kl_div(input=log m, target=log t, log_target=True)
                    # = sum t * (log t - log m) = KL(t || m)
                    terms.append(F.kl_div(log_m, t, reduction="batchmean",
                                          log_target=True))
                kd = torch.stack(terms).mean() * (tau * tau)
            else:
                n_tok = ids.numel()
                t_soft = F.softmax(t_logits / tau, dim=-1).view(n_tok, -1)
                s_logp = F.log_softmax(out.logits.float() / tau, dim=-1).view(n_tok, -1)
                kd = F.kl_div(s_logp, t_soft, reduction="batchmean") * (tau * tau)
            loss = alpha * kd + (1.0 - alpha) * out.loss
            return (loss, out) if return_outputs else loss

    n_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"[kd] student={n_params / 1e6:.1f}M teacher={t_params / 1e6:.1f}M "
          f"tau={tau} alpha={alpha} skew={skew} device={device}", flush=True)

    output_dir = ROOT / t["output_dir"]
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(output_dir / "logs"))
    resume_from = find_latest_checkpoint(output_dir)
    if resume_from is not None:
        print(f"[resume] found {resume_from.name} -> auto-resume enabled", flush=True)

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
    )

    trainer = KDTrainer(
        model=student,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        processing_class=tok,
    )

    # Pass the PATH, not True: HF's internal discovery ignores the
    # completeness guard and would pick a partial checkpoint (seen live
    # 2026-09-08: sleep killed a save mid-write -> FileNotFoundError).
    trainer.train(resume_from_checkpoint=str(resume_from)
                  if resume_from is not None else None)

    final_dir = ROOT / t["final_dir"]
    final_dir.mkdir(parents=True, exist_ok=True)
    save_final(trainer, final_dir)
    tok.save_pretrained(str(final_dir))

    final_metrics = trainer.evaluate()
    summary = {
        "phase": cfg["name"],
        "kd_teacher": kd_cfg["teacher"],
        "kd_tau": tau,
        "kd_alpha": alpha,
        "kd_skew": skew,
        "student_params_m": round(n_params / 1e6, 2),
        "teacher_params_m": round(t_params / 1e6, 2),
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "final_eval": final_metrics,
        "resumed_from": resume_from.name if resume_from is not None else None,
    }
    (final_dir / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print("[final] summary: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
