"""Mounting trainer (mounting design 2026-09-11, sections 5-6): clones
scripts/kd.py's auto-resume/TrainingArguments/save_final scaffold and trains
the student around FROZEN-TEACHER bridges (src/mount.py), with two ways of
using the teacher signal:

- bridge modes {gate, drop, hybrid}: MountBridge modules are attached to every
  student layer; each receives cross-attention KV from a frozen SmolLM2-135M
  teacher hidden state at the layer's teacher anchor. gate: 0->1 warm-in,
  hold, anneal to 0. drop: fixed strength 1.0 + progressive head severance.
  hybrid: both schedules at once. One teacher hidden-states forward per
  micro-batch (batch 1, accum 32) under no_grad; SKIPPED entirely when the
  bridge strength is 0 (exact plain CE without the teacher forward - the
  cost win). FULL severance note: the current keep-one-head rule never
  zeroes the bridge path at pct=cutoff cap, so only strength-0 skips.
- fill: no bridges at all; the student's OWN layer outputs are pulled toward
  teacher anchored hidden states by plain Linear(576, 768) probes (trained,
  discarded at save: the student must stay bridge-free).

Loss (train branch): CE blended with the mode's schedule; eval stays PURE CE -
it is the A/B metric against the plain control run and drives the cliff
schedule, so the mount blend must not leak into it.

Cliff book (Step 3.4): after each eval boundary, eval_loss - prev_eval_loss
> cliff_delta adds recovery_steps to paused_extra, persisted to
output_dir/mount_schedule.json (file survives crashes => resume-pure:
effective_step = global_step + paused_extra is re-read on every resume).

Independence save (Step 3.5): detach_mounts + probe/kv_proj removal BEFORE
the safetensors write, then a safetensors header check asserts no such
tensor survives in the artifact.

Auto-resume contract (PLAN.md 5.3) - identical to scripts/kd.py: scans
runs/<name>/ for complete checkpoint-* dirs and continues with no flags
(checkpoint found is passed BY PATH). GATE: never while another training run
is live on the GPU.
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


def mount_save_final(trainer, final_dir, mcfg) -> None:
    """Independence save (Step 3.5): BEFORE the safetensors write, strip
    every mount artifact from the model, save, then verify via the
    safetensors headers that no such tensor survived."""
    from safetensors import safe_open

    from src.mount import detach_mounts

    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    if str(mcfg.get("mode", "gate")) == "fill":
        if hasattr(model, "mount_probes"):
            del model.mount_probes            # probes were wiring, not weights
    else:
        detach_mounts(model)                  # bridges OFF first...
        if hasattr(model, "mount_kv_proj"):
            del model.mount_kv_proj           # ...then the KV wiring
    model.config.use_cache = True
    model.save_pretrained(str(final_dir))
    bad = []
    for f in sorted(Path(final_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                if "bridge" in k or "mount_kv_proj" in k or "mount_probes" in k:
                    bad.append((f.name, k))
    if bad:
        raise RuntimeError(
            "mount plumbing survived into the final artifact: " + repr(bad))


class MountCliffCallback:
    """Step 3.4: cliff book in output_dir/mount_schedule.json.

    File-persistent so paused_extra is resume-pure (re-read at every run
    start, fresh or resumed); an eval-loss cliff adds extra recovery steps
    to the effective schedule step."""
    def __init__(self, path: Path, cliff_delta: float, recovery_steps: int):
        self.path = Path(path)
        self.cliff_delta = float(cliff_delta)
        self.recovery_steps = int(recovery_steps)
        if self.path.is_file():
            book = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            book = {}
        self.book = {
            "paused_extra": int(book.get("paused_extra", 0)),
            "prev_eval_loss": book.get("prev_eval_loss"),
            "events": list(book.get("events", [])),
        }

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self.book, indent=2), encoding="utf-8")

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics is None or metrics.get("eval_loss") is None:
            return
        prev = self.book["prev_eval_loss"]
        eval_loss = float(metrics["eval_loss"])
        if prev is not None:
            cliff = eval_loss - float(prev)
            if cliff > self.cliff_delta:
                self.book["paused_extra"] += self.recovery_steps
                self.book["events"].append({
                    "step": int(state.global_step),
                    "eval_loss": eval_loss,
                    "prev_eval_loss": float(prev),
                    "cliff": cliff,
                    "paused_extra": self.book["paused_extra"],
                })
        self.book["prev_eval_loss"] = eval_loss
        self._persist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to configs/mount_*.yaml")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import yaml
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, default_data_collator)
    from transformers.trainer_callback import TrainerCallback

    from src.model import build_model
    from src.mount import attach_mounts, gate_strength, severance_pct
    from scripts.mount_dataset import MountDataset

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg, d, t = cfg["tokenizer"], cfg["data"], cfg["train"]
    mcfg = cfg["mount"]
    seq_len = int(cfg["model"]["ctx"])
    mode = str(mcfg["mode"])
    if mode not in ("gate", "drop", "hybrid", "fill"):
        raise ValueError(f"mount.mode must be gate/drop/hybrid/fill; got {mode}")
    teacher_path = ROOT / mcfg["teacher_path"]
    teacher_stream_dir = ROOT / mcfg["teacher_stream"]
    teacher_hidden = int(mcfg.get("teacher_hidden", 576))
    teacher_layers = int(mcfg.get("teacher_layers", 30))
    hidden_dim = int(cfg["model"]["hidden"])
    gate_kwargs = {k: int(v) for k, v in mcfg.get("gate", {}).items()}
    drop_kwargs = {k: int(v) for k, v in mcfg.get("drop", {}).items()}
    feat_alpha = float(mcfg.get("feat_alpha", 0.5))
    cliff_delta = float(mcfg.get("cliff_delta", 0.15))
    recovery_steps = int(mcfg.get("recovery_steps", 150))
    seed = int(t.get("seed", 42))
    if mode in ("gate", "hybrid") and not gate_kwargs:
        raise ValueError(f"mount.mode={mode} requires the mount.gate block")
    if mode in ("drop", "hybrid") and not drop_kwargs:
        raise ValueError(f"mount.mode={mode} requires the mount.drop block")

    tok = AutoTokenizer.from_pretrained(tcfg["name"])
    tok_dir = ROOT / d["tokens_dir"]
    train_ds = MountDataset(tok_dir, teacher_stream_dir, "train", seq_len)
    val_ds = MountDataset(tok_dir, teacher_stream_dir, "val", seq_len)
    print(f"[mount] blocks: train={len(train_ds)} val={len(val_ds)} "
          f"seq_len={seq_len} mode={mode}", flush=True)

    vocab = max(int(tcfg["vocab_size"]), len(tok))
    student = build_model(cfg, vocab_size=vocab)

    # Force the teacher fp32: SmolLM2's config carries torch_dtype bfloat16,
    # and a bf16 teacher forward feeds bf16 hidden states into the fp32
    # student probes (fill) / mount_kv_proj (gate/drop/hybrid) -> "mat1 and
    # mat2 must have the same dtype, but got BFloat16 and Float" at the
    # first micro-batch (GPU drill mount_fill_drill3). fp32 teacher keeps
    # every consumption site homogeneous with the fp32 student params.
    teacher = AutoModelForCausalLM.from_pretrained(str(teacher_path),
                                                   dtype=torch.float32,
                                                   attn_implementation="sdpa")
    teacher.eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher.to(device)
    n_t_layers = int(teacher.config.num_hidden_layers)
    if n_t_layers != teacher_layers:
        raise ValueError(
            f"mount.teacher_layers={teacher_layers} but the teacher model has "
            f"{n_t_layers} layers - anchors would be miscomputed")
    if int(teacher.config.hidden_size) != teacher_hidden:
        raise ValueError(
            f"mount.teacher_hidden={teacher_hidden} but the teacher model's "
            f"hidden_size is {int(teacher.config.hidden_size)}")

    if mode != "fill":
        # Step 3.3: bridges first; kv_proj (- one plain Linear per DISTINCT
        # anchor) then carries each teacher hidden state into the bridge
        # input space. Registered on the student so the optimizer/checkpoint
        # plumbing sees them; removed again at the independence save.
        bridges_list = attach_mounts(student, teacher_layers=teacher_layers)
        anchors = sorted({b.teacher_anchor for b in bridges_list})
        student.mount_kv_proj = nn.ModuleList(
            nn.Linear(teacher_hidden, hidden_dim) for _ in range(len(anchors)))
        print(f"[mount] bridges attached: {len(bridges_list)} "
              f"(anchors {anchors}, {len(anchors)} distinct kv_projs)", flush=True)
    else:
        # Step 3.2 fill wiring: probe per anchor (trained; discarded at save).
        layers_n = len(student.model.layers)
        fill_anchors = [round((l + 0.5) / layers_n * teacher_layers)
                        for l in range(layers_n)]
        probes = nn.ModuleList(
            nn.Linear(teacher_hidden, hidden_dim) for _ in range(layers_n))
        student.mount_probes = probes
        print(f"[mount] fill probes attached: {fill_anchors}", flush=True)

    # Effective-step cliff book (file-persistent). The file is re-read BEFORE
    # trainer construction on both fresh and resumed runs.
    output_dir = ROOT / t["output_dir"]
    book_path = output_dir / "mount_schedule.json"
    cliff = MountCliffCallback(book_path, cliff_delta, recovery_steps)

    class MountTrainer(Trainer):
        def __init__(self, *a, captures_flag=None, **kw):
            super().__init__(*a, **kw)
            self._fill_captures = None
            self._fill_flag = captures_flag if captures_flag is not None else [False]
            if mode == "fill":
                # F4 (Task 2 review): outputs.hidden_states[l] is recorded
                # PRE-bridge under transformers 5.16.1, so the post-layer
                # student stream is captured through a forward hook on each
                # original layer (fill mode has no bridges: post-orig IS the
                # stream the next layer consumes).
                for layer in student.model.layers:
                    layer.register_forward_hook(self._fill_hook)

        def _fill_hook(self, module, inp, output):
            if self._fill_flag[0]:
                core = output[0] if isinstance(output, tuple) else output
                self._fill_captures.append(core)

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            # Eval stays PURE CE: it is the A/B metric against the control
            # run and drives the cliff schedule - no mount term here.
            if not self.model.training:
                ids = inputs["input_ids"]
                labels = inputs.get("labels", ids)
                out = model(input_ids=ids, labels=labels)
                return (out.loss, out) if return_outputs else out.loss

            t_ids = inputs["teacher_ids"]
            t_pad = inputs["teacher_pad_mask"]
            eff_step = self.state.global_step + int(cliff.book["paused_extra"])
            if mode == "fill":
                anneal = 1.0 - gate_strength(eff_step, **gate_kwargs)
                self._fill_captures = []
                self._fill_flag[0] = True
                with torch.no_grad():
                    t_out = teacher(input_ids=t_ids, attention_mask=~t_pad,
                                    output_hidden_states=True, use_cache=False)
                t_hs = t_out.hidden_states
                causal_out = model(input_ids=inputs["input_ids"],
                                   labels=inputs["input_ids"])
                self._fill_flag[0] = False
                feat = 0.0
                for l, (p, h_s) in enumerate(zip(probes, self._fill_captures)):
                    feat = feat + F.mse_loss(p(t_hs[fill_anchors[l]]), h_s)
                self._fill_captures = None
                loss = anneal * feat_alpha * feat + (1.0 - anneal) * causal_out.loss
                return (loss, causal_out) if return_outputs else loss

            # bridge modes: gate / drop / hybrid. All attributes and the
            # severance schedule are computed FIRST and set on the bridges
            # BEFORE the student forward (the bridge runs inside the wrapped
            # layers); the teacher forward happens at most once per
            # micro-batch, under no_grad.
            s = 1.0 if mode == "drop" else gate_strength(eff_step, **gate_kwargs)
            pct = severance_pct(eff_step, **drop_kwargs) if drop_kwargs else 0.0
            skip_teacher = s <= 0.0
            for b in bridges_list:
                b._current_pad = t_pad
                b._strength = s
                if mode in ("drop", "hybrid"):
                    b.set_severance(pct, seed)
            if not skip_teacher:
                with torch.no_grad():
                    t_out = teacher(input_ids=t_ids, attention_mask=~t_pad,
                                    output_hidden_states=True, use_cache=False)
                t_hs = t_out.hidden_states
                kv_proj = student.mount_kv_proj
                for b in bridges_list:
                    idx = anchors.index(b.teacher_anchor)
                    b._current_kv = kv_proj[idx](t_hs[b.teacher_anchor])
            else:
                # teacher-SKIP: strength 0 => the bridge is an exact identity;
                # compute the EXACT plain CE without any teacher forward.
                for b in bridges_list:
                    b._current_kv = None
            out = model(input_ids=inputs["input_ids"],
                        labels=inputs["input_ids"])
            return out.loss

    class _CliffCallbackTrainerCB(TrainerCallback):
        # Bridge the module-level cliff book into the Trainer callback API.
        def on_evaluate(self, cb_args, cb_state, cb_control, metrics=None, **kw):
            cliff.on_evaluate(cb_args, cb_state, cb_control, metrics=metrics, **kw)

    n_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"[mount] student={n_params / 1e6:.1f}M teacher={t_params / 1e6:.1f}M "
          f"cliff_delta={cliff_delta} recovery_steps={recovery_steps} "
          f"device={device}", flush=True)

    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(output_dir / "logs"))
    resume_from = find_latest_checkpoint(output_dir)
    if resume_from is not None:
        print(f"[resume] found {resume_from.name} -> auto-resume enabled", flush=True)

    targs = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
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
        optim=t.get("optim", "adamw_bnb_8bit"),
        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        # MountDataset is a plain torch Dataset (not datasets.Dataset), so the
        # Trainer's _get_dataloader wraps OUR collator in RemoveColumnsCollator
        # (signature columns = the model's forward args) and strips
        # teacher_ids/teacher_pad_mask BEFORE default_data_collator runs ->
        # KeyError: 'teacher_ids' in compute_loss (GPU drill mount_fill_drill3).
        # Keep the paired keys: they are consumed in MountTrainer.compute_loss;
        # the model's forward only ever receives input_ids/labels explicitly.
        remove_unused_columns=False,
        # Eval batches carry no "labels" key: with the default
        # label_names=["labels"], prediction_step's has_labels is False and
        # eval NEVER routes through compute_loss's pure-CE branch
        # (model(**inputs) with labels=None -> loss=None -> no eval_loss at
        # all, cliff book + A/B metric inert). Route it there via input_ids.
        # Safe for training math: _get_num_items_in_batch only fires on a
        # literal "labels" key, so the /accum loss normalization is unchanged.
        label_names=["input_ids"],
        report_to=["tensorboard"],
    )

    trainer = MountTrainer(
        model=student,
        args=targs,
        callbacks=[_CliffCallbackTrainerCB()],
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        processing_class=tok,
    )

    # Pass the PATH, not True: HF's internal discovery ignores the
    # completeness guard and would pick a partial checkpoint (MEMORY 14;
    # seen live 2026-09-08: sleep killed a save mid-write).
    trainer.train(resume_from_checkpoint=str(resume_from)
                  if resume_from is not None else None)

    final_dir = ROOT / t["final_dir"]
    final_dir.mkdir(parents=True, exist_ok=True)
    mount_save_final(trainer, final_dir, mcfg)   # Step 3.5 independence save
    tok.save_pretrained(str(final_dir))

    final_metrics = trainer.evaluate()           # pure CE
    summary = {
        "phase": cfg["name"],
        "mount_mode": mode,
        "teacher_path": str(mcfg["teacher_path"]),
        "teacher_stream": str(mcfg["teacher_stream"]),
        "cliff_delta": cliff_delta,
        "recovery_steps": recovery_steps,
        "paused_extra": cliff.book["paused_extra"],
        "events": cliff.book["events"],
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