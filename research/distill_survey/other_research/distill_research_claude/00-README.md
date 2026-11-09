# LLM Training / Distillation / Context Research

Structure:
- 01-distillation/qwen-strong-to-weak-distillation.md — how Qwen ("the Queen") distills flagship → small models, why it beats RL/full pretrain-from-scratch for small models
- 02-frontier-training-techniques/deepseek-v4.md — DeepSeek-V4 Pro/Flash architecture + training recipe
- 02-frontier-training-techniques/qwen3.8-flash-next.md — Qwen3.8/Next-Flash architecture + training recipe
- 03-blackbox-api-distillation/README.md — how to distill from a model you can only call via API (no weights/logits)
- 04-context-window-boosting/README.md — practical ways to fake/extend a small context cap (your 1000-token VRAM limit), plus how frontier labs do it at scale (compression + persistent memory)

Bottom line up front (read this first, details in each file):

1. **"Distillation beats pretraining" for small models** because you're not asking the small model to discover knowledge from raw internet text — you're asking it to imitate a much stronger model's *already-organized* outputs (answers, reasoning traces, or logit distributions). That's a far easier learning signal per token, so a student gets more capability per training step/per parameter than raw next-token pretraining on generic web text would give it. It doesn't let the student exceed the teacher's ceiling.

2. There are two totally different distillation regimes:
   - **White-box** (you have the teacher's weights or at least logits): match token-level probability distributions (KL divergence), or hidden states — richest signal.
   - **Black-box / API-only** (you can only call the teacher, e.g. GPT-5/Claude/Gemini via API): you only get text out. Standard method = SeqKD (supervised fine-tune on teacher's text outputs / reasoning traces). Newer/better method = on-policy adversarial distillation (GAD) which gets close to white-box quality without needing logits.

3. **Context-window "boosting" on a tiny VRAM budget** is really three separable problems that the frontier labs solve at large scale and that you can imitate cheaply at small scale:
   - a) reduce the *storage cost per token* (attention/KV-cache compression) — DeepSeek-V4's CSA/HCA lineage, GQA/MQA, quantized KV cache.
   - b) reduce the *number of tokens the model has to actually attend to* densely — sliding window + summarization + retrieval (RAG) — this is the trick you can implement TODAY with a 1000-token cap.
   - c) give the model a persistent side-memory that survives even when you throw away the context — DeepSeek's Engram (parametric N-gram memory baked into weights) vs. agent-style external memory (Mem0/MemGPT-style paged memory, vector DB retrieval) — the practical version of "the model remembers even after you wipe context."

See each subfolder for sources, quotes-in-your-own-words, and concrete recipes.
