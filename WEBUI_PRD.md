# WEBUI_PRD.md — web UI for the nano_SLMs pipeline

Status: U1-U5 BUILT — full e2e PASS 2026-09-08. U6-U11 scope APPROVED by
the user 2026-09-08 (every proposed item except the Hub-backup button;
new requirement: HF-dataset training with local-fetch vs stream modes +
persistent keys). U12-U13 (Model tab: Architecture Explorer + real-run-replay
Training Simulator) designed in a brainstorm session and APPROVED by the user
2026-09-09 — Approach A: Gradio-native SVG, read-only + CPU-only, no new deps
(spec in §5). Owner: user + agent.
Builds on the brainstorm session of 2026-09-07 (Gradio chosen; GPU/CPU chat
policy decided; localhost-first).

## 1) Problem & goal

Training and chatting with the repo's models requires shell commands, YAML,
and terminal literacy. This UI makes the whole pipeline usable by a
non-technical user **on this machine** (LAN exposure is a later option):

1. Chat with any trained checkpoint (base completion or SFT instruct).
2. Configure data + pipeline from the UI and launch training.
3. Monitor a live run (loss, progress/ETA, GPU) without a terminal.

Hard rule inherited from CLAUDE.md/AGENTS.md: **the UI wraps, never
reimplements.** It launches the exact existing scripts as subprocesses and
reads state only from files that already exist (`runs/*/trainer_state.json`,
TensorBoard tfevents, `train_summary.json`, `eval_report.json`, `runs/` dirs).
The auto-resume contract (PLAN §5.3) is never touched.

## 2) Decisions locked in the brainstorm

| Decision | Choice |
|---|---|
| Stack | **Gradio** (one Python app, in-repo `webui/` dir, run via the venv python) |
| Chat during training | GPU by default; **warn + CPU fallback** while a training run is live; **reject GPU + fall back to CPU** if free VRAM < model need |
| Users | Only the user, this machine (localhost); LAN later via Gradio `auth=` |
| Job model | At most **one GPU job** (train/sft/eval) at a time, enforced in-app with visible status |
| Train launch | UI generates a config YAML + runs the existing chain (`run_custom.py` already implements sanity → prepare → tokenize → train → eval with stop-on-fail and a never-co-run guard — reuse it, do not rebuild the chain) |
| Kill policy | **No Stop/Kill button on a live run.** Crash-and-resume is legitimate (auto-resume, zero flags); killing for pace is forbidden. Deleting checkpoints requires typed confirmation + free-disk report |
| Presets | smoke / pilot / target / SFT configs ARE the presets; ~5 visible knobs (preset, dataset, rows, epochs/steps, LR preset), everything else behind an "advanced" YAML disclosure |
| Data modes (2026-09-08) | **Stream-pack is the default**: HF rows stream over the net and only the needed rows are written (this is what `prepare_data.py` already does, `streaming=True`) — disk stays bounded by `rows`, never by dataset size. Opt-in **full local cache** mode downloads the dataset under `data/<name>/raw/` first. Literal per-step net-feeding into the trainer is REJECTED: deterministic zero-flag auto-resume (PLAN §5.3) and the memmap shard contract require fixed local shards. Revisitable only by explicit user decision |
| Keys (2026-09-08) | Secrets live in the gitignored project-root `.env` (already auto-loaded by `prepare_data.py`), managed by a masked Settings tab: add / override / delete, path shown; never in configs, logs, or UI state |
| Cooperative stop (2026-09-08) | A checkpoint-aligned stop flag is APPROVED (U11) as the single exception to "no stop on a live run" — still never a process kill from the UI |

## 3) Functional spec

### 3.1 Chat tab
- Checkpoint picker: scan `runs/*/final` + `runs/*/checkpoint-*` (label name + step).
- Mode toggle: completion (base models) vs instruct (applies the config's
  `data.template`, matching `infer.py --sft`). Base models are labeled
  "code completion playground", SFT models "instruct chat" — expectation
  setting for non-technical users.
- Generation params: max_new_tokens, temperature/top-p/top-k (sample vs greedy).
- Model service: one model loaded at a time; `nvidia-smi` free-VRAM check at
  load; CPU fallback per §2; explicit Unload button (small models reload in
  seconds). Never allocate GPU VRAM while a training run is live.

### 3.2 Monitor tab
- Loss curve: train+eval loss parsed from tfevents (reuse the parsing logic
  already in `scripts/status.py`).
- Progress bar + ETA: global_step/total from `trainer_state.json`.
- GPU panel: memory, utilization, temp via `nvidia-smi`.
- Raw log tail in a collapsed "advanced" section.
- Eval report cards: `eval_report.json` completions + pass-rates rendered
  readably (deferred detail, see U5).

### 3.3 Data tab
- Dataset source: HF name (pre-wired with the repo's ungated fallback list)
  OR local file upload (`.txt/.jsonl` — `prepare_data.py`/`sft_data.py`
  already support local files).
- Knobs: rows, val_fraction, min_chars. Preview: first N rows + dedupe/
  filter drop counts before committing.
- Everything writes into the repo's existing `data/<phase>/` layout.

### 3.4 Train tab
- Preset picker (smoke/pilot/target/SFT-like) + the ~5 knobs → generates
  `configs/webui_<name>.yaml` (never overwrites the shipped configs).
- Preflight before Start is clickable: GPU idle (single-job lock),
  disk headroom check (repo has seen ~10 GB swings), data prepared.
- Start = subprocess `run_custom.py --config configs/webui_<name>.yaml`;
  live status surfaces in the Monitor tab.
- Resume = the same command, zero flags (that is the whole contract).

## 4) Non-goals (explicitly)

- No multi-user accounts/DB, no remote/cloud, no training logic in the web
  layer, no React/frontend build chain, no rewrite of train.py/sft.py resume
  behavior, no raw-YAML-first editing, no killing live runs from the UI (the U11
checkpoint-aligned stop flag is the single user-approved exception).

## 5) Milestones (U1–U5)

Each milestone is independently useful and ends with a PASS/FAIL check.
GPU-touching milestones only run when the GPU is free (TASKS single-job rule).

### U1 — Read-only skeleton + runs dashboard
- Gradio app in `webui/app.py`, launches via `& .\.venv\Scripts\python.exe webui\app.py`.
- Lists configs + runs (phases, checkpoints, final dirs, eval reports) by
  reading the filesystem; reuses `scripts/status.py` parsing where possible.
- **PASS:** app starts, shows every existing run correctly, zero GPU/VRAM touched.

### U2 — Chat with VRAM policy
- Model service (load/unload), checkpoint picker, completion + instruct modes,
  `nvidia-smi` VRAM gate with warn/reject → CPU fallback per §2.
- **PASS:** chat works on GPU when idle; with a training run live, chat loads
  on CPU with a visible warning and the training run's pace is unaffected.

### U3 — Live monitor
- Loss curves from tfevents, progress/ETA from `trainer_state.json`, GPU panel,
  log tail. Poll-based refresh (Gradio timer), no training-process coupling.
- **PASS:** while a real (or 200-step smoke) run is live, curves/progress/ETA
  update without touching the run; read-only next to a live run, like status.py.

### U4 — Configure + launch training
- Data tab (§3.3) + Train tab (§3.4), config generation, preflights,
  single-job lock, launch via `run_custom.py`.
- **PASS:** a non-technical flow end-to-end: pick dataset → preset → Start →
  smoke-phase run trains to completion → appears in Monitor → chat-able in U2;
  Start is refused with a clear reason while another GPU job is live or disk
  is low.

### U5 — Polish (each item user-gated)
- LAN exposure + Gradio `auth=`; done-notification (browser or webhook);
  eval report cards in Monitor; checkpoint delete-with-confirmation flow.

### U6 — Hardening & resume UX (approved 2026-09-08)
- One-click **Resume** in Monitor when the job pid is dead: re-launches
  the exact config, zero flags (the whole contract) instead of telling a
  non-technical user to open a terminal.
- Resume-collision guard in Train preflight: if `runs/<name>` exists and
  the regenerated config differs from the one on disk → refuse/warn (a
  mismatched arch must never auto-resume over an old checkpoint).
- Refuse `--lan` without `--auth` (delete + launch must not be
  LAN-exposed unauthenticated).
- Optimizer dropdown: `adamw_bnb_8bit` DEFAULT (Milestone B decision,
  TASKS rows 11/18) with fp32 fallback — replaces the `adamw_torch`
  hardcode in `build_config`.
- Chat input-length guard: truncate/warn at ctx − max_new.
- Chain-step indicator from `runs/<name>/custom_chain.json` +
  `chain_out.log` (sanity → prepare → tokenize → train → eval); today
  everything before the first checkpoint shows only "No checkpoint yet."
- Chat checkpoint-picker refresh (currently computed once at startup).
- Current-job banner (phase, pid, started, ETA) on Dashboard + Train.
- tok/s + MFU line in Monitor (`status.py` already computes it).
- **PASS:** UI e2e incl. a kill → Resume drill to completion; `--lan`
  alone exits with a clear error; a UI-launched smoke run trains with
  8-bit optim; an over-long paste degrades gracefully; a fresh run
  appears in the Chat picker without a page reload.

### U7 — Data & keys: HF datasets, two modes (approved 2026-09-08)
- Settings tab key manager over the project-root `.env`: known keys
  `HF_TOKEN`, `EXA_API_KEY`; masked display (last 4 chars only), add /
  override / delete, path displayed; never logged, never written into
  configs or UI state; verify `.env` stays gitignored.
- Train tab data-mode radio: **Stream (low disk — default)** vs
  **Download full local cache** (see §2 for the locked semantics).
- Preflight disk estimate for the chosen mode (raw ≈ rows × avg chars,
  tokens ≈ rows × tok/row) + a hard rows cap; gated-dataset 401/403
  failures surface as "gated dataset — add HF_TOKEN in Settings".
- **PASS:** gated dataset without key → clear prompt; with key in .env →
  prepare succeeds; both modes produce the same `tokens/*.bin` layout;
  key override + delete verified on disk; stream-mode e2e from the UI
  (prepare → tokenize → 100-step train).

### U8 — Transparency & monitor polish (approved 2026-09-08)
- Dataset preview (PRD §3.3, now actually built): first N rows +
  dedupe/filter drop counts before committing.
- Advanced YAML disclosure (PRD §2) + read-only preview of the generated
  config before Start.
- Per-checkpoint MB in the Danger zone; `runs/` footprint on Dashboard.
- Multi-run loss overlay in Monitor (compare phases).
- **PASS:** preview counts match prepare on a small dataset; YAML
  preview equals the written file; overlay renders ≥2 phases.

### U9 — Chat streaming + stop (approved 2026-09-08)
- `TextIteratorStreamer` progressive output + generation-level stop
  (never process-level).
- **PASS:** streaming on GPU and CPU-fallback; stop halts generation
  cleanly; the model stays usable after stop.

### U10 — Fine-tune launch path: SFT + LoRA (approved 2026-09-08)
- Train tab "SFT from checkpoint" preset → generates an SFT config (base
  = picked checkpoint, instruct template, Fine-tune LR preset,
  forgetting-guard eval kept); LoRA checkbox (peft hook, TASKS row 10).
- Chain = `sft_data.py` (CPU, co-run safe) → `sft.py` under the existing
  job_state / single-lock mechanism; `run_custom.py` stays pretrain-only.
- KD (`kd.py`) optional / deferred.
- **PASS:** UI-launched SFT pilot to completion → appears in Monitor →
  chat-able in instruct mode; LoRA checkbox emits the peft block; the
  single-job lock holds across both stages.

### U11 — Cooperative stop + polish (approved 2026-09-08; runs LAST)
- Stop-at-next-checkpoint flag file, checked by `train.py`/`sft.py` at
  save boundaries: clean exit, valid checkpoint, resume = exact same
  command. Amends the §2 no-kill rule (user-approved): checkpoint-
  aligned cooperative stop only, NEVER a process kill from the UI.
- Browser notifications (Web Notifications API) beside toast + webhook.
- Optional HTTPS for LAN (`--ssl-certfile/--ssl-keyfile`).
- **PASS:** flag set mid-run → clean stop at the next save, zero-flag
  relaunch resumes with no loss bump; no other trainer behavior change.

### U12 — Model tab: Architecture Explorer (designed + user-APPROVED 2026-09-09)
- New top-level **"Model"** tab with a nested switcher: Architecture Explorer |
  Training Simulator (U13). Shared model/run picker; nav grows to 6 tabs.
- Graph built LIVE from real artifacts, read-only + CPU-only (the wraps-never-
  reimplements and zero-GPU rules stay intact): `configs/<phase>.yaml`,
  `runs/<phase>/final/config.json` (+ `adapter_config.json` on LoRA runs), and
  the **safetensors HEADER only** (tensor names/shapes/dtypes — weights are
  never loaded). Configs without a checkpoint render the same diagram labeled
  "projected — no checkpoint yet"; totals cross-check the known ladder numbers
  (S ~12M / P 100.7M / T 226.5M).
- Diagram = SVG stack: Token IDs → Embedding (tied?) → N × [RMSNorm → GQA
  Attention (RoPE, causal) → +residual → RMSNorm → SwiGLU FFN → +residual] →
  RMSNorm → lm_head → logits. Layer tiles expand to their sub-blocks,
  color-coded; click any block (SVG via a tiny JS shim; robust fallback =
  clickable gr.Dataset layer list) → detail panel.
- Detail panel, TWO LEVELS (locked in brainstorm): plain-English what/why by
  default; a "technical" toggle adds real tensor names/shapes/dtypes from the
  header, per-block param math, and the config values feeding the block
  (rope_theta, rms_eps, head_dim…). Computed bonus: KV-cache bytes/token.
  All numbers computed live — nothing hardcoded.
- **Trace a prompt**: the real CodeLlama tokenizer on CPU → tokens + IDs →
  step-through of shapes flowing through every block. No weights involved.
- v1 cut line: attention heatmaps + next-token probability bars are OUT
  (they need a CPU forward pass — approved-as-future option); guided tour
  optional.
- Files: `webui/model_tab.py` (thin tab + wiring; app.py passes its existing
  `_ckpts`/`_full_curve` helpers in — no duplication, no circular import),
  `webui/explorer.py` (graph builder + shape math + info text),
  `webui/artifacts.py` (shared read-only readers). No new dependencies;
  app.py grows by ~5 lines.
- **PASS:** real shapes render for smoke/pilot/target finals and totals match
  12M/100.7M/226.5M; every configs/*.yaml loadable; token trace correct on a
  sample prompt; ZERO GPU/VRAM touched (file reads only); usable while a
  training run is live.

### U13 — Model tab: Training Simulator — real-run REPLAY (designed + user-APPROVED 2026-09-09)
- Educational replay, NOT training (satisfies §4 "no training logic in the web
  layer"): a technique radio (Pretrain / SFT / LoRA / KD) filters a run
  dropdown to real runs on disk (pretrain: smoke/pilot/target; SFT: sft_t1,
  sft_v2_e1; LoRA: h2_copy_lora, h2p2_mixed_lora; KD pairs:
  kd-t2p-kd+kd-t2p-baseline, kd-s-t1+kd-s-baseline). Curves/params/eval numbers
  are REAL (tfevents + train_summary.json + eval_report.json); only TIME is
  compressed. Permanent banner: "SIMULATION — real data, compressed time;
  reads runs/ only, writes nothing, zero GPU/VRAM".
- Stage cards light up in virtual time per technique: pretrain (stream →
  filter+dedupe → tokenize+pack → train loop → eval → final), SFT (pairs →
  template → filters → tokenize ctx512 → train → forgetting-guard + AST eval →
  merged final), LoRA (+ frozen base / adapter-only ckpts / merge_and_unload),
  KD (frozen teacher fwd → student fwd → 0.5·KL(τ=1)+0.5·CE → student step).
  Cards carry real numbers where available (rows, drop counts, chain timings,
  tok/s).
- Replay engine: curve parsed once (existing tfevents parsing), animated by a
  Gradio generator on a virtual clock — play/pause/step, speed 60×–3600×, and
  a scrubber (drag to any step → cards + gauges jump). Checkpoint toasts fire
  at the run's REAL save_steps; the 3-slot disk rotation animates the real
  behavior; VRAM gauge anchored to real probe numbers (4.24/4.4 GB full-FT,
  1.36 GB 8-bit, 0.72 GB LoRA); KD pairs draw two synchronized curves + a live
  delta readout; end card = the run's real summary + eval-report numbers.
- Curve source is a small pluggable interface: replay = v1; synthetic
  what-if + real-CPU-toy are approved-as-FUTURE, not v1 scope.
- Files: `webui/simulator.py` (+ `webui/artifacts.py` shared with U12).
- **PASS:** smoke replay e2e with curve values matching status.py's; an SFT
  replay and a KD pair replay correct; play/pause/step/speed/scrub work;
  rotation events match the real save steps; ZERO writes anywhere; banner
  always visible; usable while a training run is live.

## 6) Risks / honest notes

- Gradio's form ergonomics for the train tab may hit a ceiling → fallback is
  FastAPI + htmx, decided after U4, not before (do not pre-pay).
- Windows specifics: subprocess spawn + `nvidia-smi` parsing must handle the
  WDDM desktop-app quirk (see TASKS row 17 — filter to python compute apps).
- The webui server itself is cheap (CPU-only by default) and must never
  compete with a run for VRAM — the model service is the only component
  allowed to allocate GPU memory.
