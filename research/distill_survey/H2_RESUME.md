# H2 RESUME — Track H2 continuation handoff (fresh-context entry point)

> START HERE if TASKS.md row 37 sent you. Everything needed to resume Track H2
> (fix the recall wall) from zero conversation context. Written 2026-09-08
> after the Track H baseline finished and the subagent corpus authoring failed
> on infrastructure.

## 0) State in five lines

- Track H (zero-training agent-memory eval) EXECUTED, 3 GPU iterations,
  baseline evidence in runs/agent_memory_h/ (report.json + transcripts;
  iter2 snapshot in iter2_qa_cue/). Mechanism PASS, recall gate FAIL 1/6
  (required >= 2). Root cause = MEMORY lesson 18: the 226M Python-only student
  answers QA with import boilerplate and cannot copy injected facts out.
- User decision: fix it via distillation/SFT. Phase 1 = copy-behavior SFT
  (APPROVED, started). Phase 2 = general-knowledge teacher distillation
  (QUEUED; teacher NOT Qwen3.5-0.8B - user veto 2026-09-08; teacher TBD).
- Phase 1 blocker: the user wanted the corpus authored by SUBAGENTS; subagent
  spawning is broken in that session (see section 4). Corpus not yet written.
- Everything downstream of the corpus is READY: validator, LoRA config, eval.
- GPU free at last check; 43+ GB disk; no train runs active.

## 1) Decision log (user-owned - do not silently overturn)

1. 2026-09-08: "go with the best outcome track first" -> Track H executed.
2. Track H result shown -> user diagnosis: models are Python-only, so recall
   fails on distribution, not intelligence; "we should expand its knowledge to
   fix the fail and confirm, can we use the distill?" -> yes, text-level
   distillation (logit-KL is impossible cross-tokenizer, TASKS row 5/7).
3. Phase plan approved: "option 1" = Phase 1 now, Phase 2 queued - with TWO
   amendments: (a) corpus authored by SUBAGENTS for quality, not templates;
   (b) do NOT use Qwen 0.8B (applies to Phase 2 teacher choice).

## 2) File inventory (what already exists - do not rebuild)

| File | Role |
|---|---|
| research/h2_corpus_spec.md | Authoritative corpus contract (schema, rules, gates) |
| scripts/h2_validate_corpus.py | Corpus gate: schema + no-code + copy-fidelity + dedupe; < 200 pairs = FAIL |
| configs/h2_copy_lora.yaml | LoRA r=16 SFT config on runs/sft_v2_e1/final (ast_filter FALSE, CSN guard kept, LR 1e-4, 2 epochs ~75 steps) |
| scripts/agent_memory_eval.py | The Track H recall eval (reuse unchanged with --ckpt and a NEW --out) |
| runs/agent_memory_h/ | BASELINE evidence: report.json, session1/2_transcript.jsonl, iter2_qa_cue/ snapshot - NEVER overwrite |
| research/distill_survey/adoption_plan.md | Track H spec + RESULT note |
| research/distill_survey/H2_RESUME.md | This file |
| TASKS.md rows 36-37, MEMORY.md lesson 18, HANDOFF 2026-09-08 entries | Board/lessons/narrative |

Corpus targets: data/sft/h2_copy/raw/slice_{1..4}.jsonl (raw, gitignored via
data/*/raw/) -> data/sft/h2_copy/pairs.jsonl (validated, tracked).

## 3) Baseline numbers to beat / guard (from runs/agent_memory_h/report.json)

- Recall gate: store arm 1/6 (f3 only, deterministic across 2 runs), control
  0/6; probe style 0/6. REQUIRED: store arm >= 2/6.
- Mechanism (should stay green): store 6/6, retrieval 6/6, model-mode summary
  7/7 folds (61 tok), regression ast 4/4 preamble vs 2/4 plain, overhead ~7.4% ctx.
- Code guards: e1 ast greedy 0.98 / sampled 0.96; e1 CSN val_loss 2.0466
  (+9.8% vs target/final 1.8641). H2 gates: CSN <= e1 +10% (hard +15%),
  ast greedy >= 0.85 (else report honestly and consider data mix).

## 4) Blocker record: subagent infrastructure (do not burn time)

Three attempts in the losing session, all before any corpus existed:
1. 4x background spawn -> returned continuable ids, children registry STAYED
   EMPTY; later runtime notices: all 4 failed, no closing message.
2. Re-spawn 4x background (user-requested) -> identical empty registry + 4
   failure notices.
3. 1x foreground spawn test -> ToolCallError "subagent run failed".

Retry recipe for the next session (infra may be transient): spawn ONE
foreground author (run_in_background: false, slice_1, 150 pairs), verify
data/sft/h2_copy/raw/slice_1.jsonl exists with 150 valid lines. If it fails
again -> STOP retrying and use the FALLBACK:

FALLBACK (acceptable, record it in TASKS): author the 4 slices YOURSELF
following research/h2_corpus_spec.md exactly (150 pairs x 4, same diversity
axes, hand-written JSONL). The validator enforces copy fidelity either way,
so quality is gated by scripts/h2_validate_corpus.py, not by who wrote it.
Tell the user the spawn infra died and self-authoring was used.

## 5) Step-by-step todos (execute in order)

[ ] 1. Preflight: read TASKS rows 36-37 + this file + research/h2_corpus_spec.md.
       Confirm no live train (see 5.0 command). Confirm runs/agent_memory_h/
       untouched. GPU + disk check (>= 5 GB free rule trivially met).
[ ] 2. Get the corpus: subagent retry recipe (section 4) or fallback
       self-authoring. Target: 4 x 150 pairs in data/sft/h2_copy/raw/.
[ ] 3. Validate: & .\.venv\Scripts\python.exe scripts\h2_validate_corpus.py
       PASS = >= 200 valid pairs (expect ~500-600). Inspect drop reasons.
       If < 200 -> re-author missing volume (do not lower the gate).
[ ] 4. sanity_check: & .\.venv\Scripts\python.exe scripts\sanity_check.py
       --config configs\h2_copy_lora.yaml
       Expect PASS incl. peft block (trainable ~2.5% of 226M per row 10).
[ ] 5. sft_data (CPU, co-run safe): & .\.venv\Scripts\python.exe
       scripts\sft_data.py --config configs\h2_copy_lora.yaml
       Check kept-rows math (val_fraction 0.05, dedupe, ast_filter false).
[ ] 6. GPU window check (command in 5.0), then PILOT:
       & .\.venv\Scripts\python.exe scripts\sft.py --config
       configs\h2_copy_lora.yaml --pilot
       (~300 pairs, 1 epoch, ~20 steps; output runs/h2_copy_lora_pilot).
       Proof-of-plumbing: loss decreases, checkpoint + final written.
[ ] 7. FULL SFT (same window or after re-check):
       & .\.venv\Scripts\python.exe scripts\sft.py --config configs\h2_copy_lora.yaml
       (all pairs x 2 epochs, ~75 steps, ~5-8 min). runs/h2_copy_lora/final.
[ ] 8. Code guards: & .\.venv\Scripts\python.exe scripts\eval.py --config
       configs\h2_copy_lora.yaml
       Gates: CSN vs e1 (section 3), ast greedy >= 0.85. FAIL -> mix ~20-30%
       minimax3 pairs into the corpus and redo (record the deviation).
[ ] 9. THE DECISIVE TEST - rerun the SAME Track H eval on the patched model:
       & .\.venv\Scripts\python.exe scripts\agent_memory_eval.py --config
       configs\sft_v2_e1.yaml --ckpt runs\h2_copy_lora\final --out
       runs\agent_memory_h2_rerun
       (NEW --out: never overwrite the baseline evidence!)
       PASS = gate_recall store arm >= 2/6 AND regression gate PASS.
[ ] 10. Report honestly to the user (PASS or FAIL with mechanism-vs-capability
        split), then: TASKS rows 36-37, HANDOFF entry, MEMORY lesson if new
        gotchas, adoption_plan.md Track H RESULT addendum. If PASS -> Phase 1
        done; ask user about Phase 2 (teacher choice; NOT Qwen 0.8B).

## 6) Hard rules for the next agent (repo OS applies - CLAUDE.md first)

- Venv only: & .\.venv\Scripts\python.exe ... - never bare python/pip.
- NEVER co-run GPU jobs (AGENTS.md section 4). Check before every GPU step:
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  | Select-String python  (empty = free; desktop apps in the list are the
  WDDM quirk, ignore them).
- Never bare-kill anything; never overwrite runs/agent_memory_h/ or any
  existing evidence dir; new runs get new --out dirs.
- Honest labels PASS/FAIL/SKIPPED/BLOCKED; a failed gate is a RESULT, not a
  failure to report.
- Do not restart Phase 2 or pick a teacher without the user (Qwen 0.8B veto).
- If the recall rerun still fails with a green mechanism: the wall is the
  student; options = Phase 2 knowledge breadth, a stronger student, or accept.
  Present to the user - do not silently loop more SFT variants.

## 7) Phase 2 sketch (queued - user-gated)

Goal: widen the student beyond Python with REAL distilled knowledge (text-level).
Teacher: TBD, NOT Qwen3.5-0.8B (user veto). Candidates to discuss: a bigger
local instruct model (VRAM check needed on 6 GB), or an open pre-existing
general-instruct dataset (no teacher inference at all). Same pipeline shape:
same three gates + the Track H recall rerun.

## 8) Machine-move package (fresh clone + ONE zip = everything)

> **2026-09-09 UPDATE:** a full resume pack now exists —
> `checkpoint_backup/resume_pack_2026-09-09.zip` (5.8 GiB, sha256
> ceda94ad…705bd4e, 10 finals, extract AT THE REPO ROOT, in-zip RESUME.md
> + BUNDLE_MANIFEST.json). It SUPERSEDES the e1-only zip below (kept) and
> adds every other weights-bearing final. Steps 1-10 below need nothing
> else from git either way.

- Clone: git clone https://github.com/alshahia/nano_SLMs (branch main).
  The clone brings ALL code, configs, tracked data (data/target/tokens CSN
  shards for the forgetting guard, data/sft/minimax3 val instructions),
  every research note + raw Exa payload, and the runs/agent_memory_h/
  baseline evidence - section 5 steps 1-10 need nothing else from git.
- Weights zip (the ONLY artifact git cannot carry):
  checkpoint_backup/checkpoint-sft_v2_e1-final.zip
  size 909,759,554 B; sha256
  148a4af85cfc738edf59cc4248e38be65d100059decd7c3816d11dd3d41c35c8
  (verify: certutil -hashfile <zip> SHA256; built + CRC-verified by
  scripts/bundle_checkpoint.py, 8 entries, zip root is "final/").
- Extract INTO the repo (zip root "final/" lands exactly right):
  Expand-Archive .\checkpoint-sft_v2_e1-final.zip -DestinationPath .\runs\sft_v2_e1
  -> runs/sft_v2_e1/final/{model.safetensors,config.json,tokenizer*,...}
- Rebuild the venv per HANDOFF section 4, then validate with
  python -c "import torch; print(torch.cuda.is_available())" and
  scripts\sanity_check.py --config configs\smoke.yaml (4 PASS gates).
- .env is NOT in the zip (secrets, never leave the machine): add
  EXA_API_KEY / HF_TOKEN manually IF web research is needed; the H2 steps
  1-10 need neither.
- Then resume at section 5, step 1.
