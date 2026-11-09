# C12 Tier 2 readiness brief - on-policy distillation, decision-ready

Status: BRIEF ONLY (2026-09-06, prepared in the M3 training window per the user's
milestone decision). The Tier 2 RUN stays user-gated (TASKS.md row 5) and starts
only after Tier 1 passes (plan §6). Nothing here downloads or trains anything.
Supersedes the 6 GB VRAM assumptions in c12_distillation_report.md §5 (we are on
the RTX 4000 8 GB now).

## 1. What Tier 2 is (recap: report §3/§6, plan §4)

Qwen's second distillation half: the student generates its own continuations, and
a stronger frozen teacher corrects them - fixing exposure bias (the student trains
on its own distribution). The plan sketched: student samples -> frozen teacher
logits score them -> KL loss.

## 2. Verified today (2026-09-06)

- **bitsandbytes 0.50.2: SUCCESS on compute capability (7,5)** - the 8-bit teacher
  option the plan flagged for verification is VIABLE on this Turing card
  (python -m bitsandbytes diagnostic; also: peft/triton/trl not installed).
- VRAM reality on 8 GB: student full fine-tune at ctx 512 measured regime
  ~3.5-4.2 GB (M3 measured 4.24 GB at ctx 1024; ctx 512 is lighter) - see §4.

## 3. THE BLOCKER THE PLAN MISSED - tokenizer/vocab mismatch

Position-wise logit KL requires teacher and student to share tokenization.
Student = CodeLlama 32k. Off-the-shelf code teachers (Qwen2.5-Coder-*) use
~151k-BPE vocabularies -> the plan's literal "teacher logits score them (KL)"
is **mathematically undefined across tokenizers**. Same-vocab teachers at
<=1.5B with the CodeLlama tokenizer effectively do not exist on the Hub.

Consequence - two honest variants:

- **V1 (recommended): on-policy, teacher-as-judge reranking SFT.** The student
  samples N continuations per instruction; the teacher SCORES each candidate as
  TEXT (mean token logprob under its own tokenizer, plus the ast.parse gate);
  the best candidate per instruction becomes an SFT target; the student trains
  with ordinary CE on its own best samples. Cross-tokenizer safe, exposure bias
  addressed at the data level, and it reuses the ENTIRE existing pipeline:
  the new local-file support (prepare/sft_data) means the judge only has to
  write a .jsonl pairs file. Also VRAM-friendly: judge phase = teacher
  inference only (no student optimizer resident); train phase = plain sft.py
  (already measured).
- **V2 (true logit KL): only realistic intra-ladder** - teacher P (100.68M,
  CodeLlama 32k vocab) -> student S (12.3M): that is exactly plan Tier 3, not
  Tier 2 for T. Defer V2-for-T; Tier 3 already owns it.

## 4. Teacher candidates + costs (V1 needs teacher INFERENCE ONLY)

| Teacher | Download (~230 KB/s net) | VRAM inference | Notes |
|---|---|---|---|
| Qwen/Qwen2.5-Coder-1.5B | ~3.1 GB -> ~4 h | fp16 ~3.1 GB; 8-bit ~1.6 GB | strongest judge that fits easily |
| Qwen/Qwen2.5-Coder-0.5B | ~1.0 GB -> ~1.2 h | fp16 ~1.0 GB | fallback; weaker judge |
| (none) - self-judge (ast + heuristics only) | 0 | 0 | baseline denoising, weaker signal |

Phased VRAM on the 8 GB card (V1): judge phase teacher fp16 1.5B = ~3.1 GB
(+ short student generation bursts ~1 GB) -> ~4.1 GB peak, comfortable; train
phase = existing sft.py profile (~3.5-4 GB). No 8-bit needed for V1; 8-bit
matters only for a V2-style co-resident design (Tier 3 option later).

## 5. Implementation sketch (V1) - only AFTER Tier 1 gates pass

1. scripts/distill_onpolicy.py (~250-350 lines): load Tier-1 student
   (runs/sft_t1/final) -> sample N=4-8 continuations per instruction
   (temp 0.8, ctx 512) from a pool of instructions (Evol val + train tail) ->
   teacher scores each candidate (mean logprob/token via its own tokenizer,
   fenced-code extraction + ast.parse gate) -> write the winning
   (instruction, response) pairs to data/sft/onpolicy/pairs.jsonl.
2. configs/sft_t2.yaml: base = runs/sft_t1/final (continue from Tier 1),
   dataset = the local pairs file (custom-dataset path), same SFT hyperparams
   as Tier 1 (lr 3e-5 cosine, ctx 512, accum 16), auto-resume contract.
3. Run: sft_data.py -> sft.py (pilot first, then full) - unchanged machinery.
4. Est.: implementation 4-6 h + CPU tests; GPU pilot ~1 h; judge GPU cost
   ~1-2 s/candidate at 1.5B fp16 (N x instructions x ~2 forward passes).

## 6. Success criteria (Tier 2 gate)

- ast.parse pass-rate on the 50 held-out instructions STRICTLY above the
  Tier-1 report (research/c12_runbook.md §7 artifact).
- CSN val regression gate still <= ~10-15% vs base T (forgetting guard).
- M2/M3 training signature: grad_norm settles <= ~0.6 x clip; zero NaNs.
- Side-by-side eyeball: Tier-2 generations beat Tier-1 on the same prompts.

## 7. User decision checklist (raise at Tier-1 pass, NOT before)

1. Approve teacher download (which: 1.5B vs 0.5B vs none) - network + ~1-3 GB
   disk (E: had 16.3 GB free at the 2026-09-06 probe; fine either way).
2. Approve the GPU window (judge + pilot after Tier 1 completes).
3. Approve scripts/distill_onpolicy.py implementation (4-6 h window work).

## 8. What NOT to do

- No teacher download without explicit user approval (slow network, MEMORY
  standing rule).
- No Tier 2 implementation or run while M3 trains; nothing touches runs/ -
  this brief is documentation only.
- Do not attempt cross-tokenizer logit KL (§3) - it is undefined; V1 or Tier 3.
