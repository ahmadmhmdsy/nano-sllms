# Mounting Experiment Implementation Plan (pilot dims, SmolLM2-135M)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and run the 5-arm mounting A/B (control / annealed fill-in / Mount A gate-anneal / Mount B stochastic wire-drop / Mount C hybrid) at pilot dims with a frozen SmolLM2-135M teacher, reporting wall-clock-to-parity and full-independence quality.

**Architecture:** The mount is a per-layer cross-attention bridge from the 12L/768 student to a frozen SmolLM2-135M (30L/576) run inference-only on a dual-tokenized shard stream (same raw text, the teacher own 49k Llama-3 BPE tokenizer, per-block alignment built at prep time). Bridge mechanics live in a new src/mount.py; training reuses the scripts/kd.py Trainer pattern in a new scripts/mount.py; every knob is config-gated and DEFAULT OFF elsewhere (repo contract: init_from / init_embeddings style). Teacher weights are never trained.

**Tech Stack:** Python 3.12 (.venv), torch 2.14.0+cu126 (fp16 only, Turing sm_75), transformers 5.16.1, HF Trainer, PyYAML, TensorBoard. SmolLM2-135M from HF hub (ungated). No pytest in this repo — tests are runnable assert-scripts under scripts/ run with the venv python.

**Spec:** docs/plans/2026-09-11_mounting_design.md - TASKS row 49 - All GPU work strictly sequential + zero-flag auto-resume (PLAN §5.3).

---

## File structure

| File | Responsibility |
|---|---|
| scripts/mount_teacher_stream.py (new) | Build the teacher-tokenized shard stream aligned PER-BLOCK to the student stream (prep-time decode then re-encode). |
| src/mount.py (new) | Bridge module (gated cross-attn), schedule functions (step-to-strength / step-to-severance), deterministic severance masks, schedule persistence file I/O, attach/detach helpers. |
| scripts/mount.py (new) | Trainer for arms 2-5 (annealed fill-in with trained probes; mount modes gate / drop / hybrid). Clone of the kd.py scaffold + mount forward. |
| scripts/mount_dataset.py (new) | Pair-aligned dataset: student uint32 blocks + teacher uint32 blocks + int16 length map -> collate dict. |
| configs/mount_control.yaml, mount_fill.yaml, mount_gate.yaml, mount_drop.yaml, mount_hybrid.yaml (new) | The five arms; identical except the mount block. |
| scripts/test_mount.py (new) | CPU assert-script: zero-init identity, schedule shapes, severance determinism. |
| scripts/train.py, scripts/sanity_check.py, src/model.py | UNTOUCHED. All existing configs stay byte-identical (mount is additive). |

---

## Task 0: Preflights (CPU-only, do first)

- [ ] **Step 0.1 Environment + idle GPU**

Run: venv python -c "import torch; print(torch.__version__)" - Expected 2.14.0+cu126.
Run: nvidia-smi --query-compute-apps=pid,process_name - Expected no LIVE python training process (stale WDDM entries are documented noise; run_custom.py filter precedent).

- [ ] **Step 0.2 SmolLM2-135M availability**

Probe the HF cache for models--HuggingFaceTB--SmolLM2-135M (AutoConfig load OK). If absent, download via the download_teacher.py pattern (~270 MB, ungated, network OK). Record the snapshot path - it is the config value for mount.teacher_path.

- [ ] **Step 0.3 Pilot corpus on disk**

Confirm data/pilot/tokens train+val shards exist (the KT-1 arms trained on them). If missing: STOP and report BLOCKED: data.
---

## Task 1: Teacher token stream - scripts/mount_teacher_stream.py

**Files:** Create scripts/mount_teacher_stream.py.

- [ ] **Step 1.1 Write the script**

```python
"""Build a SmolLM2-tokenized shard stream aligned PER-BLOCK to the student
stream (mounting design section 7). For each student uint32 block we decode
the exact text span with the student tokenizer, re-encode with the mounted
teacher's tokenizer, cap at ctx tokens, right-pad with id 0; a parallel
.len.bin (int16, one value per block) stores true lengths so cross-attention
masks pads out (pads are NEVER attended). Boundary-span docs decode across
blocks by construction (block b continues block b-1's tail); off-by-one
boundary tokens are accepted noise, documented in the spec.
CPU-only; co-run-safe beside a live GPU training run."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-tokens-dir", default="data/pilot/tokens")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-tok", default="codellama/CodeLlama-7b-hf")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    student_tok = AutoTokenizer.from_pretrained(args.student_tok)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model)

    shards = sorted(Path(args.student_tokens_dir).glob(f"{args.split}_*.bin"))
    assert shards, "no " + args.split + " shards"
    n_blocks = 0
    for shard in shards:
        ids = np.fromfile(shard, dtype=np.uint32)
        ids = ids[: (len(ids) // args.ctx) * args.ctx]
        blocks = ids.reshape(-1, args.ctx)
        lens, tblocks = [], []
        for b in blocks:
            text = student_tok.decode(b.astype(int).tolist())
            t = teacher_tok(text, add_special_tokens=False)["input_ids"][: args.ctx]
            lens.append(len(t) if t else 1)
            tblocks.append(t + [0] * (args.ctx - len(t)))
        np.stack(tblocks).astype(np.uint32).tofile(out / shard.name)
        np.asarray(lens, dtype=np.int16).tofile(out / (shard.name[:-4] + ".len.bin"))
        n_blocks += len(tblocks)
    (out / "meta.json").write_text(json.dumps(
        {"teacher": args.teacher_model, "ctx": args.ctx, "blocks": n_blocks,
         "split": args.split, "pad_id": 0}, indent=2), encoding="utf-8")
    print("[teacher-stream] wrote", n_blocks, "blocks to", out, flush=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2 Micro dry-run (CPU)**

Build a 2-block slice of train_0000 into a temp dir, run the script into data/pilot/tokens_teacher_smol135_dryrun, verify: identical block count, meta.json written, .len.bin first entry greater than 0. DELETE the dry-run artifacts after (report-before-delete rule: these are MY OWN scratch).

- [ ] **Step 1.3 Real run (CPU, co-run-safe)**

Run twice (split train, then val): venv python scripts/mount_teacher_stream.py --teacher-model SNAPSHOT_PATH --split SPLIT --out-dir data/pilot/tokens_teacher_smol135
Expected: teacher block counts equal the student stream block counts exactly.

## Task 2: Bridge core - src/mount.py

- [ ] **Step 2.1 Write the module (final form)**

```python
"""Mounting bridges (design 2026-09-11 sections 5-6): frozen-teacher
cross-attention wired into each student decoder layer, trained, then
progressively unlinked. Everything config-gated; DEFAULT OFF everywhere else
in the repo. NOT-SHIP SHORTCUT: never pass bridge state through kwargs into
the original decoder layer - the trainer sets b._current_kv / b._strength /
b._current_pad attributes per batch BEFORE the student forward (attribute
injection only)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn

def gate_strength(step: int, *, warmup: int, hold: int, anneal_end: int) -> float:
    """Design section 6: 0->1 warm-in, hold, linear anneal 1->0, then 0."""
    if step <= warmup:
        return step / max(warmup, 1)
    if step <= hold:
        return 1.0
    if step <= anneal_end:
        return max(0.0, 1.0 - (step - hold) / (anneal_end - hold))
    return 0.0

def severance_pct(step: int, *, unlink_start: int, stage_len: int,
                  init: float, cap: float) -> float:
    """Mount B ladder: init percent at unlink_start, +10 per stage_len steps, capped."""
    if step < unlink_start:
        return 0.0
    k = (step - unlink_start) // stage_len
    return min(cap, init + 10.0 * k)

class MountBridge(nn.Module):
    """y = layer_out + strength * tanh(a) * tanh(MHA(q=y, kv=t_kv)).
    a := learned scalar, zero-init, so tanh(a)=0 and step 0 is an exact
    identity (Flamingo gating). Mount B severance zeroes whole MHA heads;
    surviving heads rescale by 1/(1-p) (DARE rule) so head mass stays calibrated."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.a = nn.Parameter(torch.zeros(()))
        self.heads, self.dim = heads, dim
        self._strength = 0.0
        self._sev_keep = None
        self._sev_rescale = 1.0
        self._current_kv = None
        self._current_pad = None
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def set_severance(self, pct: float, seed: int):
        if pct <= 0:
            self._sev_keep, self._sev_rescale = None, 1.0
            return
        H = self.heads
        g = torch.Generator().manual_seed(int(seed) * 10007 + int(pct))
        r = torch.rand(H, generator=g)
        keep = torch.argsort(r) >= int(round((pct / 100.0) * H))
        if not keep.any():
            keep[torch.argmax(r)] = True
        self._sev_keep = keep
        self._sev_rescale = 1.0 / (1.0 - pct / 100.0) if pct < 100.0 else 1.0

    def apply_sev(self, h):
        if self._sev_keep is None:
            return h
        H, d = self.heads, self.dim // self.heads
        m = self._sev_keep.to(h.dtype).view(1, 1, H, 1)
        h = (h.view(*h.shape[:-1], H, d) * m).reshape(h.shape)
        return h * self._sev_rescale

    def forward(self, y, strength: float):
        if strength <= 0.0 or self._current_kv is None:
            return y
        h, _ = self.attn(y, self._current_kv, self._current_kv,
                         key_padding_mask=self._current_pad, need_weights=False)
        h = self.apply_sev(h)
        return y + strength * torch.tanh(self.a) * torch.tanh(h)

class WrappedLayer(nn.Module):
    """Student decoder layer wrapper: orig layer first, then the bridge."""
    def __init__(self, orig, bridge):
        super().__init__()
        self.orig = orig
        self.bridge = bridge
    def forward(self, *a, **kw):
        y = self.orig(*a, **kw)
        core = y[0] if isinstance(y, tuple) else y
        out = self.bridge(core, self.bridge._strength)
        return (out,) if not isinstance(y, tuple) else (out,) + tuple(y[1:])

def attach_mounts(student, teacher_layers: int):
    n_layers = len(student.model.layers)
    bridges = []
    for l in range(n_layers):
        b = MountBridge(student.config.hidden_size,
                        heads=max(1, student.config.hidden_size // 128))
        b.teacher_anchor = round((l + 0.5) / n_layers * teacher_layers)
        bridges.append(b)
        student.model.layers[l] = WrappedLayer(student.model.layers[l], b)
    student._mount_bridges = nn.ModuleList(bridges)
    return bridges

def detach_mounts(student):
    """Full independence: replace wrappers with the original layers."""
    if not getattr(student, "_mount_bridges", None):
        return student
    for l in range(len(student.model.layers)):
        w = student.model.layers[l]
        if isinstance(w, WrappedLayer):
            student.model.layers[l] = w.orig
    student._mount_bridges = None
    return student
```

Executor notes (binding):
1. WrappedLayer must keep working under HF gradient checkpointing (use_reentrant=False): verify with a 4-layer CPU forward/backward in scripts/test_mount.py.
2. Bridge reads ONLY its own per-batch attributes (trainer sets BEFORE model forward): b._strength float, b._current_kv [B, Tt, student_dim], b._current_pad [B, Tt] bool (True = pad).
3. Teacher hidden indexing: outputs.hidden_states[0] = embeddings, [l] = post-decoder-layer l; student layer l anchors teacher layer round((l+0.5)/12*30); stored as b.teacher_anchor at attach time.
---

## Task 3: Trainer - scripts/mount.py (clone of scripts/kd.py + deltas)

**Files:** Create scripts/mount.py and scripts/mount_dataset.py.

- [ ] **Step 3.1 Start from the kd.py scaffold**

Copy scripts/kd.py to scripts/mount.py and keep EVERYTHING that is not listed here byte-identical: find_latest_checkpoint (partial-ckpt guard), auto-resume path-pass, TrainingArguments block, save_final + train_summary. Rename the module docstring and defaults (optim default = adamw_bnb_8bit per Milestone-B).

- [ ] **Step 3.2 Paired dataset (scripts/mount_dataset.py)**

```python
"""Pair-aligned mount dataset: student uint32 blocks zip with teacher
per-block uint32/len blocks produced by scripts/mount_teacher_stream.py.
Asserts equal block counts at init; collate returns dicts with
input_ids (student, labels implicit), teacher_ids, teacher_pad_mask."""
from pathlib import Path
import numpy as np
import torch
from src.data import PackedDataset

class MountDataset(torch.utils.data.Dataset):
    def __init__(self, student_dir: Path, teacher_dir: Path, split: str, seq_len: int):
        self.student = PackedDataset(sorted(student_dir.glob(split + "_*.bin")), seq_len)
        self.t_files = sorted(teacher_dir.glob(split + "_*.bin"))
        self.len_files = sorted(teacher_dir.glob(split + "_*.len.bin"))
        n_s = len(self.student)
        blocks_t = [np.fromfile(f, dtype=np.uint32).reshape(-1, seq_len) for f in self.t_files]
        self.teacher = np.concatenate(blocks_t, axis=0)
        lens = np.concatenate([np.fromfile(f, dtype=np.int16) for f in self.len_files])
        assert len(self.teacher) == n_s == len(lens), "block count mismatch"
        self.teacher_lens = lens

    def __len__(self):
        return len(self.student)

    def __getitem__(self, i):
        s = self.student[i]
        if isinstance(s, dict):          # PackedDataset contract check at runtime
            s = s["input_ids"]
        t = self.teacher[i]
        m = np.arange(len(t)) >= int(self.teacher_lens[i])
        return {"input_ids": s.astype(np.int64),
                "teacher_ids": t.astype(np.int64),
                "teacher_pad_mask": torch.from_numpy(m)}
```

- [ ] **Step 3.3 Mount trainer deltas (in scripts/mount.py)**

1. Read the mount block: mode in {gate, drop, hybrid, fill}; teacher_path; teacher_stream; gate{warmup,hold,anneal_end}; drop{unlink_start,stage_len,init,cap}; cliff_delta; recovery_steps.
2. Build student via src/model.py build_model(cfg, vocab_size=32768) EXACTLY as kd.py does. For mode fill: NO bridges (no attach); probe list = ModuleList of nn.Linear(576, 768) for the 12 anchors (trained; discarded at save).
3. Build teacher: AutoModelForCausalLM.from_pretrained(mount.teacher_path, attn_implementation="sdpa"); eval; requires_grad_(False) for all params; to(device); use_cache=False.
4. Wrap student: for modes != fill: bridges = attach_mounts(student, teacher_layers=30); kv_proj = ModuleList of nn.Linear(576, 768) per distinct anchor (DepthwiseSeparable not needed; keep plain Linear, it is the wiring and it is discarded).
5. compute_loss (train branch only; eval stays PURE CE like kd.py):

```python
effective_step = self.state.global_step + int(book.get("paused_extra", 0))
if self.mode == "fill":
    t_out = teacher(input_ids=t_ids, attention_mask=~t_pad, output_hidden_states=True)
    anneal = 1.0 - gate_strength(effective_step, **self.gate_kwargs)
    ce = out.loss from model(...); feat = 0
    for l, (p, h_s) in enumerate(zip(probes, student_captures)):  # captures via a forward hook per layer
        feat = feat + F.mse_loss(p(t_out.hidden_states[anchors[l]]), h_s)
        return anneal * feat_alpha + (1.0 - anneal) * ce   # feat_alpha = launch alpha (0.5)
if self.mode in ("gate", "drop", "hybrid"):
    s = 1.0 if self.mode == "drop" else gate_strength(effective_step, **self.gate_kwargs)
    for b in bridges:
        b._current_pad = t_pad
        b._current_kv = kv_proj[b.teacher_anchor](hidden_state_for(b.teacher_anchor))
        b._strength = s
        if self.mode in ("drop", "hybrid"):
            b.set_severance(severance_pct(effective_step, **self.drop_kwargs), self.seed)
    if all bridges have strength 0 and full severance:
        for b in bridges: b._current_kv = None   # teacher forward SKIPPED; exact CE = the cost win
    out = model(input_ids=ids, labels=ids)
    return out.loss
```

Real-code ordering requirement: set bridge attributes BEFORE calling model(...) (bridge forward runs inside the wrapped layers). hidden_states fetched ONCE per optimizer step under no_grad: wrap the teacher call in the micro-batch (batch 1, accum 32 - one teacher forward per micro-batch, cached KV reused across nothing else).

- [ ] **Step 3.4 Cliff book (mount_schedule.json; auto-resume safe)**

A TrainerCallback: file at output_dir/mount_schedule.json with {"paused_extra": int, "prev_eval_loss": float, "events": []}. On each eval boundary: if eval_loss - prev_eval_loss exceeds cliff_delta (default 0.15), paused_extra += recovery_steps (default 150), event logged AND persisted (this file survives crashes; effective_step is therefore resume-pure).

- [ ] **Step 3.5 Independence save**

In save_final (mount script local copy): BEFORE writing safetensors call detach_mounts(model), then verify no tensor name contains the substring bridge (safetensors header check), then write. Bridge/kv_proj/probe params must not appear in the final artifact.
---

## Task 4: scripts/test_mount.py (CPU assert script)

- [ ] **Step 4.1 Write**

```python
"""CPU-only tests. Run: venv python scripts/test_mount.py -> ALL PASS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.mount import (gate_strength, severance_pct, MountBridge,
                       attach_mounts, detach_mounts)
from transformers import LlamaConfig, AutoModelForCausalLM

assert gate_strength(0, warmup=10, hold=20, anneal_end=40) == 0.0
assert gate_strength(10, warmup=10, hold=20, anneal_end=40) == 1.0
assert abs(gate_strength(30, warmup=10, hold=20, anneal_end=40) - 0.5) < 1e-9
assert gate_strength(40, warmup=10, hold=20, anneal_end=40) == 0.0
assert severance_pct(0, unlink_start=100, stage_len=10, init=10, cap=100) == 0.0
assert severance_pct(100, unlink_start=100, stage_len=10, init=10, cap=100) == 10.0
assert severance_pct(9999, unlink_start=100, stage_len=10, init=10, cap=100) == 100.0

# zero-init bridge == exact identity (step-0 contract, Flamingo gating)
b = MountBridge(768, heads=6)
x = torch.randn(2, 8, 768)
assert torch.equal(b(x, 0.0), x)
b._current_kv = torch.randn(2, 12, 768)
assert torch.allclose(b(x, 1.0), x, atol=1e-6)

# severance deterministic per (seed, pct)
b.set_severance(50.0, seed=42); k1 = b._sev_keep.clone()
b.set_severance(50.0, seed=42)
assert torch.equal(b._sev_keep, k1)
assert int(k1.sum()) == 3  # 6 heads at 50 percent

# wrap / attach / detach on a real tiny Llama model
cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=16)
m = AutoModelForCausalLM.from_config(cfg)
bridges = attach_mounts(m, teacher_layers=6)
ids = torch.randint(0, 64, (2, 16))
for br in bridges:
    br._strength = 0.5
    br._current_kv = torch.randn(2, 5, 64)
m(input_ids=ids, labels=ids).loss.backward()
n_before = sum(p.numel() for p in m.parameters())
detach_mounts(m)
n_after = sum(p.numel() for p in m.parameters())
assert n_after < n_before
n_ref = sum(p.numel() for p in AutoModelForCausalLM.from_config(cfg).parameters())
assert n_after == n_ref, "detach must restore the exact original param set"
print("test_mount: ALL PASS")
```

- [ ] **Step 4.2 Run** venv python scripts/test_mount.py - Expected: test_mount: ALL PASS. If gradient checkpointing breaks WrappedLayer, use the GradientCheckpointingLayer pattern (transformers 5.16.1, non-reentrant; the GDN wrapper precedent works through custom modules) and re-run until PASS.

- [ ] **Step 4.3 Commit** git add src/mount.py scripts/test_mount.py && git commit -m "mount: bridge core + CPU test PASS (TASKS row 49)"

## Task 5: Configs + CPU sanity gates

- [ ] **Step 5.1 configs/mount_gate.yaml (arm 3)**

```yaml
name: mount-gate
tokenizer: {name: codellama/CodeLlama-7b-hf, vocab_size: 32768}
model: {layers: 12, hidden: 768, heads: 12, kv_heads: 4, ffn: 2048, ctx: 512, dropout: 0.0, tie_embeddings: true}
data: {tokens_dir: data/pilot/tokens}
mount:
  teacher_path: SNAPSHOT_PATH
  teacher_stream: data/pilot/tokens_teacher_smol135
  mode: gate
  teacher_hidden: 576
  teacher_layers: 30
  heads_bridge: 6
  gate: {warmup: 150, hold: 300, anneal_end: 900}
  cliff_delta: 0.15
  recovery_steps: 150
train: {output_dir: runs/mount_gate, final_dir: runs/mount_gate/final, max_steps: 1000, batch: 1, accum: 32, lr: 4.0e-4, warmup_steps: 150, logging_steps: 25, eval_steps: 100, save_steps: 100, save_total_limit: 3, fp16: true, grad_ckpt: true, optim: adamw_bnb_8bit, seed: 42}
```

- [ ] **Step 5.2 Remaining four configs**

- configs/mount_drop.yaml: identical except mount.mode = drop and the gate block ABSENT (fixed strength 1.0 while bridges live); drop block {unlink_start: 300, stage_len: 100, init: 10, cap: 100}.
- configs/mount_hybrid.yaml: mode hybrid; BOTH the gate schedule and the drop ladder from the same blocks.
- configs/mount_fill.yaml: mode fill; same gate schedule (used inverted=anneal); probes trained, no bridge modules.
- configs/mount_control.yaml: NO mount block; runs/mount_control; plain scripts/train.py run at identical data/steps/seed (the kt_ab_control rig precedent).

- [ ] **Step 5.3 CPU sanity on all five**

venv python scripts/sanity_check.py --config configs/mount_X.yaml for each - Expected PASS 4/4 on all five.

- [ ] **Step 5.4 VRAM probe (GPU, minutes)**

vram_probe with the mount config (batch 1, ctx 512) - Expected well under 6 GB (LoRA precedent 0.72 GB; frozen teacher adds ~0.5-1.5 GB). Record the measurement.

- [ ] **Step 5.5 GPU kill/resume drill (~10 min)**

150-step override of mount_gate, kill near step 60, re-run the EXACT command zero flags - resume must continue the identical trajectory (path-pass + partial-ckpt guard precedents; cliff book is file-persistent; schedules are pure step functions). PASS REQUIRED before any full arm.

- [ ] **Step 5.6 Commit** git add configs/mount_*.yaml && git commit -m "mount: arm configs + sanity + drill PASS (TASKS row 49)"

## Task 6: GPU execution (user-opened window; strictly sequential)

- [x] **Step 6.1 Order: control + fill-in first (anchors), then A, then B, then C** (DONE 2026-09-11→09-12; order: control → fill → gate → drop → hybrid, sequential, exact command lines below executed; drop resumed zer-flag from ckpt-300 after the device-bug fix commit 7e1abf2)

1. venv python scripts/train.py --config configs/mount_control.yaml
2. venv python scripts/mount.py --config configs/mount_fill.yaml
3. venv python scripts/mount.py --config configs/mount_gate.yaml
4. venv python scripts/mount.py --config configs/mount_drop.yaml
5. venv python scripts/mount.py --config configs/mount_hybrid.yaml

Rules: zero-flag auto-resume after ANY interruption; TensorBoard is the mid-run truth (MEMORY 2); thermal pace swings are never a kill reason (MEMORY 6); single GPU - nothing co-runs.

- [x] **Step 6.2 Independence eval per mount arm** (DONE: 110 tensors / zero bridge names verified on all 5 finals; eval_report.json written per arm; mount-on == independent to 4 decimals)

save_final detached bridges already; verify no final tensor name contains bridge; run scripts/eval.py. Record the Stage-1 mount-on eval (hold-end eval from tfevents) vs the final independent eval = the user third gate.

- [x] **Step 6.3 Report: research/mounting_ab_report.md** (WRITTEN + committed 7e1abf2; TASKS row 49 = done; HANDOFF section finalized)

Per arm: wall-clock GPU-h to Arm-1 anchor eval points (tfevents readback per MEMORY 2); final student-only eval; pace profile per stage (does Mount B's post-severance teacher-skip translate to real tokens/s?); cliff count and pause cost; VRAM peak; fp16 grad-norm stability; generation samples; the pros/cons table of gate vs stochastic-drop vs hybrid mapped to later use cases and scale transfer. Update TASKS row 49 + HANDOFF; weights LOCAL per AGENTS section 5.

## Task 7: Honesty gates (hard)

- [x] Any arm NaN or eval_loss diverging beyond +20 percent above the control at matched steps: never triggered (drop peaked +8.6% @700; fill diverged but completed its scheduled anneal and is reported FAIL honestly).
- [x] Wall-clock is the headline metric (confirmed: teacher arms 4-5h+ vs control 1h23m; hybrid 2h05m best).
- [x] Report failures as FAIL/SKIPPED with reasons — fill = FAIL (post-cliff eval flat ~6.5, fp16 grad-norm runaway median 1454; reported in the AB report).


- [ ] **Step 3.6 Commit** git add scripts/mount.py scripts/mount_dataset.py && git commit -m "mount: trainer + paired dataset (TASKS row 49)"


