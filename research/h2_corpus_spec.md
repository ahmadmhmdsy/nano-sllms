# H2 corpus spec — copy-behavior SFT (Track H2 Phase 1)

Purpose: teach the 226M student ONE behavior it never learned (its pretrain/SFT
diet was Python-only): given a short context containing user facts plus a
question about one of them, reply with a short plain-English sentence that
COPIES the fact from the context. This is the behavior the Track H recall gate
needs; it adds no world knowledge (that is Phase 2, teacher TBD — Qwen3.5-0.8B
vetoed by the user 2026-09-08).

Authoring: 4 parallel subagent writers, 150 pairs each, slices 1-4 with
different domain emphasis (personal/prefs, technical ops, everyday logistics,
note-completion form). Raw slices land in data/sft/h2_copy/raw/ (gitignored via
the data/*/raw/ pattern); the validated, deduped set is data/sft/h2_copy/pairs.jsonl
(tracked). Validation: scripts/h2_validate_corpus.py.

Per-pair schema (one JSON object per line):
  {"instruction": "<context + one question>", "response": "<one short sentence>"}

Hard rules (enforced by the validator):
  - response = 1 short sentence, 1-30 words, no code, no markdown, no preamble
  - every content word of the response must appear in the instruction
    (copy fidelity; <= 1 out-of-context word tolerated)
  - fact values copied verbatim (names, numbers, ports, dates)
  - forbidden substrings: "import ", "def ", "class ", ">>>", code fences
  - instruction = context block (2-6 bullet facts or a 2-4 sentence summary)
    + exactly one question; 30-1200 chars

Context header forms used by the Track H eval (about half of each slice should
use them; the rest are natural variants):
  "Known facts about the user:" / "Conversation summary so far:" / "Notes:"
Note-completion form (slice 4 emphasis; instruction ends mid-note):
  "... Note about the user: My sister's name is" -> response completes it.

Pipeline: validate/dedupe -> configs/h2_copy_lora.yaml (LoRA r=16 on
runs/sft_v2_e1/final, LR 1e-4, ast_filter FALSE — English responses) ->
sanity_check -> sft.py --pilot -> full SFT (2 epochs, tiny corpus deviation
from the 1-epoch standing gate, argued by ~75 total steps) -> eval.py (ast +
CSN forgetting guard) -> agent_memory_eval.py --ckpt runs/h2_copy_lora/final.
Gates: recall >= 2/6 (store arm), CSN <= +10-15%, ast not collapsing.
