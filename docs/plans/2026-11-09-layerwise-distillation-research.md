# Layer-wise / block-wise distillation — research report

**Date:** 2026-11-09 · **Status:** research / brainstorm (no experiment run, no GPU used)
**Question (user):** distill a big model into a tiny one layer by layer, so poor compute/VRAM can still do it. Find prior work, gaps, and anything missed.

## 1. Prior art — the idea exists under the name *block-wise (layer-wise) local distillation*

Each student block receives the teacher's input at that depth and learns to reproduce the teacher block's output; only one block pair needs to be in memory.

| Work | What it does | What we take from it |
|---|---|---|
| Puzzle (NVIDIA, 2024) | Each child block trained independently to mimic its parent block (normalized MSE) | Closest match; blocks are independent and parallelizable |
| LLM-Streamline (ICLR 2025) | Remove a run of unimportant layers, train one lightweight replacement network | N teacher layers → 1 small block; removal without replacement hurts a lot |
| Iterative Layer-wise Distillation (2025) | Iterative layer removal + KL(logits) + MSE(hidden) healing | Qwen2.5-3B 36→28 layers ≈ 9.7% quality loss, →24 ≈ 18% |
| BERT-of-Theseus (2020) | Randomly swap teacher modules for smaller substitutes, rising probability | Smooth progressive replacement |
| TinyBERT | Layer mapping, attention + hidden-state matching | Attention matching matters more than hidden alone (secondary source) |
| MOHAWK / Llamba / Mamba-in-the-Llama | Staged cross-architecture distillation (mixer matrices → hidden states → end-to-end) | Student block may be a different architecture |
| Relaxed Recursive Transformers (ICLR 2025) | Shared looped block + per-depth LoRA, init from pretrained | Very few unique parameters |
| ReplaceMe | Training-free linear replacement of a block run | Free baseline to beat |
| Minitron / Sheared-LLaMA | Structured pruning, then distill / continued pretraining | Initialize the student from teacher weights |
| Dasgupta & Cohn (ICLR 2025) | CKA hidden-state matching across different widths | Width mismatch handled without a fixed projection |

## 2. Weaknesses and mitigations

1. **Error accumulation.** Local losses do not guarantee end-to-end fidelity (SAES-SVD: similarity 0.97→0.79 across layers; LaCo). Mitigate: student-forced refinement + mandatory end-to-end KL healing (CURing: PPL 77→17.6, mostly within ~100 steps).
2. **Residual width is the hard limit.** Depth compression is easy; width compression needs projections or CKA. Expect ~2–4× from matching; beyond that, logit-level distillation dominates.
3. **Massive activations / outlier dimensions** dominate plain MSE → per-dim normalization, cosine, or normalized MSE.
4. **Activation-cache disk cost:** tokens × d_model × 2 bytes per boundary (7B, 20M tokens ≈ 164 GB). Mitigate: stream one boundary at a time, quantized cache, top-k logits.
5. **Tokenizer must match for hidden/logit KD** — repo precedent: E-008 KD collapsed twice (research/experiments.yaml:398-405; teacher too weak / different tokenization).
6. **Embeddings dominate tiny models** (150k vocab × 512 ≈ 77M params).
7. **Capacity gap** — consider a teacher-assistant intermediate.

## 3. Things easy to miss

- Free wins first: drop near-identity layers (ShortGPT Block Influence), ReplaceMe linear patch.
- Initialize from teacher weights (slice/SVD), not random.
- Treat attention and MLP separately (MLP → low-rank/MoE, e.g. LaDiMo; attention → linear/SSM).
- Block independence suits interrupted, auto-resume compute.
- Quantization (4-bit teacher) is the honest competitor.
- Judge by perplexity/downstream vs a from-scratch model at equal compute, not per-layer error.

## 4. Pilot shape (first draft; superseded by section 5 once written)

Arms at equal token budget: A from-scratch tiny; B layer drop + linear patch; C block-wise local distillation (teacher-forced then student-forced); D = C + end-to-end KL healing. Success: D beats A.

## 5. Arabic char-level student plan (turn 2; supersedes section 4)

**User constraints:** no M3 models. The teacher should be an Arabic model of 100–300M parameters, or about 500M if none exists. The student should be 25–30M parameters and able to hold a conversation in Arabic. Its embedding is char-level. Hardware is 6 GB VRAM and about 10 GB free disk (both stated by the user, not measured).

### 5.1 Teacher candidates (web-checked 2026-11-09)

| Model | Params | License | Arabic evidence | Verdict |
|---|---|---|---|---|
| **inceptionai/jais-family-590m-chat** | 590M (the safetensors metadata says 0.8B) | Apache 2.0 | ArabicMMLU 39.1, HellaSwag-ar 57.8, EXAMS 29.5 (model card, zero-shot); Arabic and English; ctx 2048 | **Pick.** Arabic-first, chat-tuned, permissive license, best measured Arabic numbers in range |
| RightNow-Arabic-0.5B-Turbo (2605.28827) | 518M | not stated | ArabicMMLU 23.2 (below chance 25), HellaSwag-ar 26.0 | Fallback only; weak and license unclear |
| Qwen2.5-0.5B-Instruct / Qwen3-0.6B | 0.5–0.6B | Apache 2.0 | multilingual; no Arabic-specific benchmark found | Fallback |
| Gemma 3 270M | 270M | Gemma terms | 140+ languages; no Arabic number found | Only open option in the 100–300M band; check its Arabic quality before use |
| Kuwain-1.5B, Jais 2 | — | — | weights unreleased / 8B+ | Out |

No open, Arabic-specialised model in the 100–300M band was found. Jais-590m needs `trust_remote_code`, so read its modelling code before running it. VRAM: about 1.2–1.6 GB in bf16, so it runs online beside the student on 6 GB.

### 5.2 Is a char-level student feasible? Yes, with two costs

- **The embedding gets cheaper.** About 300 symbols (Arabic letters and diacritics, Latin, digits, punctuation, and dialogue control tokens) × d=384 is about 0.1M parameters. A BPE vocabulary of 85k–180k at d=384 would take 33–69M, which is more than the whole student budget. So almost all 25–30M parameters go to the transformer blocks.
- **Cost 1: sequence length.** Arabic averages about 3.5–5 characters per teacher token, so a 512-token teacher window becomes roughly 2,000–2,500 characters. Mitigation: an **hourglass** student. Two char-level blocks feed a ×4 downsample, then about 8 blocks run at the pooled rate, then a ×4 upsample and two char-level blocks (Hourglass/BLT/H-Net family). Attention cost then falls about 16× in the middle layers.
- **Cost 2: hierarchical char models lose some character-exact skill.** This matters less for conversation than for diacritization.
- **Repo asset:** the diacritizer D-line is already a char-level Arabic model with a 171-symbol vocabulary. E-067 showed that a shape-matched char-LM warm start is a positive lever at 64-hidden. Reuse its char inventory and normalisation code.

### 5.3 How the char student learns from a subword teacher

The problem is that the two models don't share an output space or positions. Three signals solve it, from strongest to most optional:

1. **Logit KD through char marginalisation (BLD/ALM family).** Take the teacher's top-k next-token distribution (k = 64–256) at each token boundary. It gives the next-char distribution P(c) ∝ Σ P(t) over tokens t that start with c. Inside a token whose prefix is p, P(c | p) ∝ Σ P(t) over tokens starting with p+c. This gives a dense KL target at every character position, at no disk cost. The approximation ignores continuations that cross a token boundary, which is fine for a first version. ALM's chunk-likelihood matching is the more principled variant if this one plateaus.
2. **Layer-wise hidden matching (the user's original idea, adapted).** Align positions causally: the student's hidden state at the **last character of each teacher token's span** has seen exactly the text that the teacher's hidden state at that token has seen. In the hourglass design, the pooled middle layers are already close to token rate. The loss is a learned linear projection (d_s → d_t, thrown away after training) with normalised MSE or cosine, or CKA, which needs no projection. Map student pooled layer i to teacher layer ≈ i·L_t/L_s, and take Jais's layer count from its config. Normalise each dimension to handle massive activations (§2.3).
3. **Sequence-level KD for conversation.** Generate multi-turn Arabic dialogues with the teacher and mix in existing Arabic SFT data. Filter the dialogues, because Jais-590m's own quality is modest.

**What "layer by layer" buys here:** the original motive for layer-wise training was VRAM, so that only one block pair has to sit in memory. That motive is moot, because the whole teacher fits on 6 GB. The value that remains is **denser supervision per token** for a tiny student. Schedule: stage 1, train lower blocks against teacher layer ⅓ with a short warm-up; stage 2, match all mapped layers; stage 3, end-to-end KL plus CE (healing, §2.1). Pure Puzzle-style block replacement does not apply, because tokenisation and width both differ.

### 5.4 Budget (6 GB VRAM / ~10 GB disk)

- **VRAM:** the teacher takes about 1.6 GB in bf16, with no grad. The student is about 27M parameters, and in fp32 with AdamW that is about 0.45 GB. Activations for a batch of 8 × 2,048 chars with the hourglass fit easily. Use a 4-bit teacher only if needed, because it degrades the targets.
- **Disk:** the teacher download is about 1.2–3.2 GB depending on dtype, plus the corpus. **No hidden-state cache**: 100M tokens × 1,536 × 2 B is about 300 GB. A top-k logit cache is also too big: 100M tokens × k=32 × 6 B is about 19 GB. So the teacher runs **online**.
- **Throughput (estimate, not measured):** a 0.6B forward pass costs about 1.2 GFLOP/token, so on a 6 GB consumer GPU expect thousands of teacher tokens per second. 100M teacher tokens then takes a few hours. Measure this in the smoke run.
- Rules: one GPU job at a time against repository runs (rules/gpu.md), a 60% free-VRAM cap at launch, and checkpoints every 100 steps or fewer.

### 5.5 Pilot arms (equal student-char budget)

- A: char hourglass student from scratch, CE only.
- B: A + char-marginalised logit KD.
- C: B + layer-wise hidden matching.
- D: C + teacher dialogue data (sequence KD).

Metrics:
- Held-out Arabic bits per char.
- A fixed 100-prompt Arabic chat set, judged blind by a human, or by a stronger external judge if the user approves one.
- Distinct-n and repetition rate.

Success: B/C beat A at equal budget. Smoke first, then pilot on a fixed slice, per repo protocol.

### 5.6 Risks and honest expectations

- **Capacity.** A 25–30M model will chat fluently in a narrow register (TinyStories-scale behaviour) but will not carry world knowledge. ArabicMMLU will stay near chance. Narrow the domain (for example everyday MSA chit-chat or a single topic) for usable quality.
- **Teacher ceiling.** Jais-590m is modest; the student can't exceed it.
- **Repo precedent E-008.** KD collapsed when the teacher was weak and the tokenization didn't match. Here the mismatch is handled explicitly (§5.3.1), but compare against arm A, because collapse shows up as B ≤ A.
- **Dialect vs MSA.** The teacher is MSA-centric.
- **Diacritics.** Teacher text is mostly undiacritized. Keep diacritics in the vocabulary, but expect them to be rare.

## Sources

Puzzle https://arxiv.org/pdf/2411.19146 · LLM-Streamline https://arxiv.org/pdf/2403.19135 · Iterative LWD https://arxiv.org/pdf/2511.05085 · BERT-of-Theseus https://aclanthology.org/2020.emnlp-main.633 · TinyBERT https://www.arxiv-vanity.com/papers/1909.10351/ · Mamba in the Llama https://arxiv.org/html/2408.15237v3 · MOHAWK https://arxiv.org/html/2408.10189v1 · RRT https://arxiv.org/abs/2410.20672v2 · ReplaceMe https://arxiv.org/pdf/2505.02819v2 · CKA matching https://proceedings.iclr.cc/paper_files/paper/2025/hash/2fb462e23667ad5e6471a4e9af8e4774-Abstract-Conference.html · Sheared-LLaMA https://arxiv.org/pdf/2310.06694 · SAES-SVD https://arxiv.org/html/2602.03051v1 · LaCo https://aclanthology.org/2026.acl-long.1342.pdf · CURing https://arxiv.org/pdf/2501.04211 · LaDiMo https://arxiv.org/pdf/2408.04278 · DeltaLLM https://arxiv.org/pdf/2501.18596v2
