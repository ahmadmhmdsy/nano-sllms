# Design — Frozen Core + External Knowledge Memory (SmolLM2-135M)

**Status:** DRAFT for user review (brainstorming skill output; not yet approved; no code touched)
**Date:** 2026-09-12
**Decisions locked in by user:** base = SmolLM2-135M (frozen), first pillar = external knowledge memory.
**Success gate (adopted by default after user "proceed", user may revise anytime):**
recall-probe battery passes (store / retrieve / **copy-out**) AND eval ppl improves at least 2-3% on memory-relevant held-out eval vs the no-memory baseline.

## 1) Motivation & evidence

- User idea: freeze a small model's knowledge permanently; attach trainable/external modules to enhance it.
- Literature precedent: kNN-LM (Khandelwal et al. 2020) - frozen LM + non-parametric key-value store, interpolating frozen logits with retrieved next-token distribution; RETRO (Borgeaud et al. 2022); Memorizing Transformers (Wu et al. 2022) succeeded with memory on a FROZEN pretrained model; Ladder Side-Tuning (NeurIPS 2022) for the later "add layers" pillar.
- Our own evidence REUSED:
  - Track H (runs/agent_memory_h*): zero-training memory store+retrieval 6/6+6/6, but copy-out 1/6 FAIL -> this design's trained bridge exists specifically to fix copy-out.
  - LoRA-on-frozen forgetting +0.13% vs full-SFT +9.8%/+19.7% -> frozen-core + tiny trained module is the knowledge-preserving mode this repo already trusts.
  - Streaming-sink+abs [E-17] stays the eval-only long-ctx lever; NOT part of this pillar.
  - 8-bit Adam (adamw_bnb_8bit) batch b1/accum a32 = recorded default optimizer lever [E-12].
  - Standing instruments: VRAM ladder probes, wall-clock to matched-step anchor, kill/resume drill, eval_report.json authoritative; CSN forgetting guard trivially zero here (weights never change) so the REAL gate is copy-out + ppl A/B.

## 2) Architecture

FROZEN (never trained, never modified, sha256 recorded at download):
- HuggingFaceTB/SmolLM2-135M (~135M params, Llama-arch, ~0.27 GB fp16, Llama3-family tokenizer vocab 49152, tied embeddings). Weights + tokenizer cached locally; local-only weights policy (gitignored).

MEMORY STORE (built ONCE offline; read-only at inference; manifest with sha256 + build params):
- Corpus: our measured Milestone-D mixture rows (stack-smol / starcoder / CSN / Evol at 30/55/10/5 row shares, STREAMED SUBSETS ONLY - slow network) + optional fresh text rows (USER-GATED choice).
- Keys: hidden states of the FROZEN model over the corpus, computed in small GPU batches inside serialized GPU windows (one-off pass, resumable via manifest).
- Values: next token id (+ optionally the token hidden state for richer copy).
- Index: FAISS CPU if install is clean; FALLBACK = plain NumPy fp16 matrix + top-k dot product (pure-torch/numpy path is first-class, not a hack).
- SIZE: storing every token is too big (~115 GB at 100M tokens x 576 dims fp16) -> STRIDE SAMPLING (default every 8th token) + optional n-gram dedupe -> target 300-500 MB on-disk store; exact stride USER-GATED after a size probe on a sample shard.

TRAINED BRIDGE (the ONLY trainable thing in this pillar; exists to fix Track-H copy-out):
- One tiny gate head: input = frozen-model hidden state at the decode step (+ top-1 retrieved key), output = low-dim lambda controlling the kNN-LM interpolation P = (1-lambda)*P_frozen + lambda*P_store.
- ZERO-INIT: lambda head outputs 0 at init -> day-one behavior is EXACTLY the frozen model -> knowledge preservation is structural (byte-identical day-one), not hoped-for.
- Trainable params ~1-2M max (small LoRA-r + gate head). adamw_bnb_8bit b1/a32; fp16 only; micro-batch via VRAM ladder probe.

## 3) Data flow (inference)

text -> SmolLM2 tokenizer -> frozen forward pass -> per step:
(a) frozen logits P_frozen
(b) query current hidden state vs store -> top-k neighbors -> aggregate next-token distribution P_store
(c) gate head computes lambda from (frozen hidden, top keys)
(d) P = (1-lambda) P_frozen + lambda P_store -> sample

Training: same path, teacher-forced on held-out text; loss = CE on blended P; stop-grad on P_frozen so the gate only learns WHEN to trust the store, never edits the core.

## 4) Evaluation plan (pre-registered gates)

1. PLUMBING gate: weights fingerprint (sha256) recorded at download and re-hashed before+after every run - frozen must be VERIFIED byte-identical, never assumed.
2. RECALL PROBE battery (Track-H style, upgraded): store / retrieve / copy-out folds; PASS = copy-out >= 5/6 (Track H baseline 1/6 is the concrete failure to beat), junk-filter report included.
3. EVAL A/B: same held-out val, frozen-only vs +memory; PASS = >=2-3% ppl improvement on memory-relevant splits; base split may not regress beyond +1%; deltas reported per split.
4. OPS: VRAM ladder probe before the train arm; exact-zero-flag kill/resume drill before launch; wall-clock to matched-step anchor; tok/s with and without store.
5. Single-source numbers (eval_report.json / tfevents); end-state re-eval after any restore (MEMORY 31).

## 5) Error handling / known risks

- Store too large -> stride + dedupe + size-probe gate; never delete user data to make room.
- kNN latency -> batched queries per step; measure inference tok/s with vs without store.
- Gate offloads everything (lambda -> 1) -> watch lambda distribution, cap with a prior if needed.
- 135M core ceiling: comprehension does NOT improve; report as capability gain, not intelligence gain.
- Single GPU: store-build pass, train arm, eval arms strictly sequential, never co-run with a live train job.

## 6) Out of scope (later pillars, recorded not started)

- Pillar 2: memory window (gated streaming-sink + small trained memory head) - builds directly on E-17.
- Pillar 3: agentic side-layers / tool use (Ladder Side-Tuning style + KT-2 judge corpus; biggest cost, last).
- Attention swap (GDN hybrid etc.) = retrain-from-scratch track, separate plan, never a frozen graft.
- KT-1 embedding transplant irrelevant here - the teacher arrives fully pretrained.
