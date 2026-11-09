"""mex/scripts/train_mu2_lora.py — mu2 rung LoRA trainer (E-31 onward).

Mechanics identical to mex/scripts/train_distill.py (root_train reuse,
auto-resume zero flags, cosine, fp16, load_best_at_end, save_final) with
mu2-ladder differences ONLY:
  1. trunk = cfg['lora']['base_run'] bot final weights loaded from safetensors
     into build_model(cfg) (same config template -> keys match exactly).
  2. peft LoRA from cfg['lora'] (r/alpha/dropout/targets); ONLY deltas train.
  3. E-31 fix: the mark-drop corruption happens IN BATCH (Mu2CorruptDataset):
     input marks -> '|', labels stay CLEAN (the shifted CE then supervises
     the fill), every 7th train block stays clean (14% retention replay);
     eval blocks never corrupt so eval CE IS the clean retention reading.
  4. final save merges the adapter back (plain safetensors for G3).
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

_spec = importlib.util.spec_from_file_location(
    "root_train", str(ROOT / "scripts" / "train.py"))
root_train = importlib.util.module_from_spec(_spec)
sys.modules["root_train"] = root_train
_spec.loader.exec_module(root_train)

find_latest_checkpoint = root_train.find_latest_checkpoint
clear_stop_flag = root_train.clear_stop_flag
_env_fingerprint = root_train._env_fingerprint

from src.data import PackedDataset  # noqa: E402
from mex.src.vocab import CharVocab  # noqa: E402

MARKS = "ًٌٍَُِّّْ"


class Mu2CorruptDataset(PackedDataset):
    """E-31 fix: labels must carry the TRUE mark where the INPUT is masked.
    Rewriting streams breaks 1:1 alignment under a shifted CE, so corruption
    is IN BATCH: input marks -> mask_char, labels stay clean. corrupt=True
    masks train blocks except every 7th (clean replay); eval never corrupts,
    so eval CE doubles as the clean-stream retention reading.
    """

    def __init__(self, shards, seq_len: int, *, corrupt: bool,
                 voc: CharVocab, mask_char: str = "|", hold_every: int = 7):
        super().__init__(shards, seq_len)
        import torch
        self._corrupt_enabled = corrupt
        self._mask_id = int(voc.vocab[mask_char])
        self._marks = [int(voc.vocab[c]) for c in MARKS]
        import numpy as np
        self._mark_arr = np.asarray(self._marks, dtype=np.int64)
        self._np = np
        self._hold_every = int(hold_every)

    def _corrupt_inplace(self, ids):
        np = self._np
        m = np.isin(ids, self._mark_arr)
        if m.any():
            ids = ids.copy()
            ids[m] = self._mask_id
        return ids

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        if self._corrupt_enabled and (idx % self._hold_every) == 3 % max(self._hold_every, 1):
            return item                      # clean replay block (unmasked)
        if self._corrupt_enabled:
            item["input_ids"] = self._corrupt_inplace(item["input_ids"])
        return item



import torch
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0.0, momentum=0.95, ns_iters=5):
        super().__init__(list(params), {"lr": lr, "weight_decay": weight_decay, "momentum": momentum})
        self.momentum = momentum
        self.ns_iters = ns_iters

    @staticmethod
    def _newton_schulz(G, iters=5):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G / (G.norm() + 1e-12)
        transposed = G.shape[0] > G.shape[1]
        if transposed:
            X = X.T
        for _ in range(iters):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return X.T if transposed else X

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]; wd = group["weight_decay"]; mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state.setdefault(p, {})
                buf = st.setdefault("m", torch.zeros_like(g, dtype=torch.float32))
                buf.mul_(mom).add_(g.to(torch.float32), alpha=1 - mom)
                u = buf
                if p.dim() >= 2:
                    u = self._newton_schulz(u)
                p.add_(u.to(p.dtype), alpha=-lr)
                if wd:
                    p.mul_(1 - lr * wd)
        return None
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--logging-steps", type=int, default=0)
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--final-dir", default="")
    args = ap.parse_args()

    import yaml
    from safetensors.torch import load_file
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                              default_data_collator)

    from src.model import build_model, save_final

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg, d, t, l = cfg["tokenizer"], cfg["data"], cfg["train"], cfg["lora"]
    if args.max_steps > 0:
        t = dict(t, max_steps=args.max_steps)
    if args.logging_steps > 0:
        t = dict(t, logging_steps=args.logging_steps)
    if args.output_dir:
        t = dict(t, output_dir=args.output_dir)
    if args.final_dir:
        t = dict(t, final_dir=args.final_dir)
    seq_len = int(cfg["model"]["ctx"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tcfg["name"])
    voc = CharVocab()
    cr = l.get("corrupt") or {}
    train_ds = Mu2CorruptDataset(
        (ROOT / d["tokens_dir"]).glob("train_*.bin"), seq_len,
        corrupt=bool(cr.get("mask_marks")), voc=voc,
        mask_char=str(cr.get("mask_char", "|")),
        hold_every=int(cr.get("hold_every", 7)))
    val_ds = Mu2CorruptDataset(
        (ROOT / d["tokens_dir"]).glob("val_*.bin"), seq_len,
        corrupt=False, voc=voc, mask_char=str(cr.get("mask_char", "|")))
    print(f"[train_mu2_lora] blocks: train={len(train_ds)} val={len(val_ds)} "
          f"seq_len={seq_len} corrupt={bool(cr.get('mask_marks'))}", flush=True)

    model = build_model(cfg, vocab_size=tcfg["vocab_size"])
    base_sd = load_file(str(ROOT / l["base_run"] / "model.safetensors"))
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {list(unexpected)[:5]}")
    print(f"[lora] trunk from {l['base_run']} missing={len(missing)}",
          flush=True)
    if not l.get("enabled", True):
        pass                                   # G3 settle: full weights train
    else:
        model.enable_input_require_grads()
    if not l.get("enabled", True):
        print("[settle] full-weights fine-tune (Net2Net widen candidate)",
              flush=True)
    else:
        peft_cfg = LoraConfig(
            task_type="CAUSAL_LM", r=int(l["r"]), lora_alpha=int(l["alpha"]),
            lora_dropout=float(l["dropout"]),
            target_modules=list(l["targets"]))
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    output_dir = ROOT / t["output_dir"]
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(output_dir / "logs"))
    resume_from = find_latest_checkpoint(output_dir)
    if resume_from is not None:
        print(f"[resume] {resume_from.name} -> auto-resume", flush=True)
    if clear_stop_flag(output_dir):
        print("[stop] cleared stale STOP flag", flush=True)

    targs = TrainingArguments(
        output_dir=str(output_dir),
        seed=int(t["seed"]),
        max_steps=int(t["max_steps"]),
        per_device_train_batch_size=int(t["batch"]),
        per_device_eval_batch_size=int(t["eval_batch"]),
        gradient_accumulation_steps=int(t["accum"]),
        learning_rate=float(t["lr"]),
        adam_beta1=0.9, adam_beta2=0.95,
        lr_scheduler_type=t["scheduler"],
        warmup_steps=int(t["warmup_steps"]),
        weight_decay=float(t["weight_decay"]),
        max_grad_norm=float(t["max_grad_norm"]),
        logging_steps=int(t["logging_steps"]),
        eval_strategy="steps", eval_steps=int(t["eval_steps"]),
        save_strategy="steps", save_steps=int(t["save_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=bool(t["fp16"]), gradient_checkpointing=bool(t["grad_ckpt"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        report_to=["tensorboard"], remove_unused_columns=False,
    )

    _muon = os.environ.get("DIA2_BY_2_MUON", "0") == "1"
    optimizer_pair = None
    if _muon:
        trainable = [p for p in model.parameters() if p.requires_grad]
        twos = [p for p in trainable if p.dim() >= 2]
        optimizer_pair = (Muon(twos, lr=float(t["lr"])), None)
        print(f"[muon] override on {len(twos)} 2D trainable params", flush=True)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=default_data_collator,
                      optimizers=optimizer_pair,
                      callbacks=[root_train.CoopStopCallback()],
                      processing_class=tok)
    trainer.train(resume_from_checkpoint=str(resume_from)
                  if resume_from is not None else None)

    final_dir = ROOT / t["final_dir"]
    final_dir.mkdir(parents=True, exist_ok=True)
    save_final(trainer, final_dir)               # merge_and_unload inside
    tok.save_pretrained(str(final_dir))

    summary = {
        "phase": cfg["name"],
        "lora": {"r": l["r"], "alpha": l["alpha"], "targets": l["targets"],
                 "base_run": l["base_run"],
                 "corrupt": bool(cr.get("mask_marks"))},
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "resumed_from": resume_from.name if resume_from is not None else None,
        "env": _env_fingerprint(cfg),
    }
    (final_dir / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print("[final] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
