# Distilling from a model you can ONLY call via API (no weights, no logits)

This directly answers: "can we link this or make this model provide certain data so [our] model can learn and copy this model with the least weight we can have it."

Sources:
- arXiv:2511.10643 — "Black-Box On-Policy Distillation of Large Language Models" (GAD / Generative Adversarial Distillation)
- arXiv:2604.00626 — "A Survey of On-Policy Distillation for Large Language Models" (section 5.1.2, Black-Box and API-Constrained Distillation)
- Hsieh et al. 2023 — "Distilling Step-by-Step"
- emergentmind.com/topics/black-box-on-policy-distillation
- dev.to explainer on distillation mechanics and the DeepSeek allegations context
- researchgate.net summary of GAD

## The core constraint
When your "teacher" is a closed API (GPT-5, Claude, Gemini, etc.) you get **only the text it outputs** — no logits (token probability distribution), no hidden states, no gradients, no weights. This rules out the classic, richest distillation signal (forward/reverse KL-divergence between teacher and student token distributions), because that requires the full probability distribution over the vocabulary at every position, which closed APIs don't expose (at best you might get top-k logprobs from some providers, not the full distribution).

## Method 1 (baseline, works today): SeqKD — Sequence-level Knowledge Distillation
1. Query the teacher API with a large, diverse set of prompts covering your target domain.
2. Collect the teacher's full text responses.
3. Supervised-fine-tune (SFT) your student model directly on these (prompt, teacher-response) pairs, as if they were ordinary labeled training data.
This is "off-policy" — the student is trained on fixed, pre-generated teacher outputs, never on its own generations. Simple, cheap, works, but caps out lower than richer methods because of exposure bias (train-time text the student sees ≠ what it actually produces at inference).

## Method 2 (higher ceiling, still black-box): Distilling Step-by-Step
Instead of extracting only the teacher's *final answer*, extract its **intermediate reasoning steps / rationale** as additional supervision. Concretely: prompt the teacher to show its work (chain-of-thought), and train the student to produce both the rationale and the final answer, not just the answer. Reported result (Hsieh et al.): a much smaller student can **outperform a larger model trained on final answers alone, using less data** — because the rationale gives the student a denser training signal per example (it learns *how* to get there, not just the *destination*).

## Method 3 (state of the art, Nov 2025): GAD — Generative Adversarial Distillation
This is the most relevant new technique for "trick a closed model into teaching mine cheaply and well."

**Setup**: frame it as a minimax game, like a GAN:
- **Generator = your student LLM.**
- **Discriminator = a small classifier/reward model** you train alongside the student, whose job is to distinguish "is this text from the teacher or from my student?" — trained using a **sequence-level scalar score derived from the last token's hidden state** via an added prediction head, and a **Bradley-Terry pairwise-preference loss** (rank teacher responses above student responses).
- The discriminator acts as an **on-policy reward model** — critically, it co-evolves with the student (unlike a fixed/static reward model), so it keeps giving useful gradient signal even as the student improves and starts producing harder-to-distinguish outputs.
- The student is optimized via **RL** using this learned discriminator as the reward signal, training on the student's **own generated rollouts** (on-policy) rather than only teacher-authored text (off-policy) — this directly fixes the exposure-bias weakness of SeqKD, because the student learns to correct its own actual outputs, not just imitate static examples it might never naturally reach.
- RL-style optimization here promotes **mode-seeking** behavior (student commits confidently to one good answer style) rather than **mode-covering** (naively averaging over all the teacher's possible response styles), which better aligns the student to *reachable* teacher behaviors given its own smaller capacity.

**Reported results**: Qwen2.5-14B-Instruct trained as a student with GAD, using GPT-5-Chat as the (API-only, black-box) teacher, became **comparable to the teacher itself** on LMSYS-Chat automatic evaluation — and GAD "consistently surpasses the commonly used sequence-level knowledge distillation" (i.e. beats plain SeqKD).

**Practical costs reported**: on the order of ~30 hours on 16×H100 GPUs for a 14B student — i.e. this is not free, it's a real training run, but it's within reach of a serious small lab, not just frontier labs.

**Caveats it explicitly flags**:
- Removing generator or discriminator "warmup" phases degrades performance and balance — you need to pretrain/stabilize each side a bit before starting the adversarial game.
- Adds computational overhead vs plain SFT: co-evolving discriminator, longer training time, larger batch/group sizes needed.
- Risk of **reward hacking** — the student can learn to exploit the discriminator's blind spots (e.g. produce excessively long outputs if the discriminator has a length bias) rather than genuinely matching quality — needs monitoring.
- **Respect the API Terms of Service** of whatever closed model you're calling — most providers explicitly restrict using their outputs to train competing models. This is a real legal/policy constraint, not just a technical one — the DeepSeek/"distilled our model" controversy referenced in these sources exists precisely because of this gray area.

## Practical recipe if you want to try this at your scale
1. Pick a strong, accessible-via-API teacher and a domain of prompts representative of what you actually need your student to do well at.
2. Start with **Method 2 (Distilling Step-by-Step)** — it's dramatically cheaper to implement than GAD (just prompt engineering + SFT) and already captures a lot of the available gain.
3. If you have budget for a real training run and need to close the gap further, implement **GAD**: you need (a) your student model, (b) a small discriminator head (can literally be a linear head on top of a frozen or lightly-tuned encoder / on top of the student's own last hidden state), (c) an RL loop (e.g. GRPO/PPO-style) that samples student rollouts, scores them with the discriminator, and updates both.
4. Mix hard labels (verified-correct final answers, where you can check correctness — e.g. code that passes tests, math with known answers) with soft/teacher-imitation signal — literature consistently finds mixing beats either alone.
5. Always keep a held-out eval set scored by a *different* strong model (or by your own domain-specific automatic checks) to catch reward-hacking early.
