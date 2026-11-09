# Tashkeel Compare Space - design and implementation plan

- **Date:** 2026-10-14
- **Owner:** agent
- **Status:** approved by user ("approve them all , Go-ahead on the recommended bundle")
- **Target:** a public HuggingFace Space under `ahmadsy/`
- **Registry:** to be recorded as `diacritizer/E-105` on completion

## Goal

A public, reproducible demo where anyone can diacritize Arabic text with our published
nano models **and** with published third-party models, side by side, including large
pasted/uploaded inputs that the Space splits and reassembles automatically.

The point is not decoration. It is to make the claim we just put on the model cards
*checkable by a stranger in a browser*: a 1.08M-parameter model competing with models
orders of magnitude larger, at a fraction of the latency.

## Non-goals

- Not a training tool, not a leaderboard authority. It re-displays published numbers and
  measures its own; it never invents a ranking.
- Not a replacement for `webui/` (the local Gradio shell on 127.0.0.1:7860). Separate
  directory, separate repo, no shared process. The Space never touches 127.0.0.1:3080.

## Roster

Every compared model is a downloadable HF checkpoint. No unpublished model appears
anywhere in the Space - that is the E-104 rule, enforced here at the source.

| key | repo | params | family | notes |
|---|---|---|---|---|
| `nano1p1m` | `ahmadsy/tashkeel-nano-1p1m` | 1.08 M | our `tashkeel_nano` | headline model |
| `nano4p5m` | `ahmadsy/tashkeel-nano-4p5m` | 4.49 M | our `tashkeel_nano` | predecessor; legacy mark order |
| `mishkala` | `flokymind/mishkala` | 12.5 M | custom + CRF | needs `pytorch-crf` |
| `tashkeel50m` | `ahmedsamirtarjama/Tashkeel-50M` | ~50 M | Llama causal | prompt-wrapped generation |
| `flan` | `Abdou/arabic-tashkeel-flan-t5-small` | ~77 M | T5 seq2seq | 256-token cap, needs chunking |
| `byt5` | `basharalrfooh/Fine-Tashkeel` | ~300 M | ByT5 seq2seq | byte-level, slowest on CPU |

Roster membership is **result-based**: a model ships only if it loads and produces sane
output in the local probe (`scratch/probe_one.py`). A model that fails is dropped from
the registry, not hidden behind a broken button.

The published SadeedDiac-25 leaderboard (Claude-3.7-Sonnet, GPT-4, Gemini-Flash-2.0,
Sadeed, Aya-23-8B, ALLaM-7B, Yehia-7B, Jais-13B, Gemma-2-9B, SILMA-9B) is shown as a
**static, clearly-labelled reference table** - those models cannot run on free CPU.

## File structure

New top-level `space/` directory. One responsibility per file:

| file | responsibility |
|---|---|
| `app.py` | Gradio UI: 4 tabs, wiring, progress, downloads. Entry point. |
| `registry.py` | model specs, lazy load, LRU eviction, per-model predict dispatch, timings |
| `adapters.py` | one predict function per model family; knows each model's quirks |
| `splitter.py` | split text into chunks on paragraph/sentence boundaries + reassemble exactly |
| `scoring.py` | order-insensitive canonical DER/WER + the published leaderboard table |
| `samples.py` | the sample gallery texts |
| `theme.css` | RTL rendering, model-card styling |
| `README.md` | Space card (`sdk: gradio`, `app_file: app.py`) |
| `requirements.txt` | pinned deps |

## UX

### Tab 1 - Compare
Input textarea + file upload; model checkboxes (our two default-on, third-party opt-in
with a CPU-latency warning). Output: one column per model showing the diacritized text
with a stats strip - params, download size, latency, chars/sec, and a **mark-order
badge** (`NFC` vs `legacy`).

### Tab 2 - Bulk
Paste or upload `.txt` / `.csv` / `.md`. The splitter runs with a configurable char
budget and prints a **splitting report** (N paragraphs to M chunks, longest chunk,
anything that hit the cap). Progress bar during the run. Reassembly preserves the
original line structure byte-for-byte outside the diacritized spans. Downloads:
per-model `.txt`, plus a combined side-by-side `.csv` and `.json`.

### Tab 3 - Score (reference mode)
Paste/upload a diacritized reference; get per-model `DER (CE)` and `DER (w/o CE)` - the
same columns as the published leaderboard, so numbers are directly comparable.
**Scoring canonicalizes mark order on both sides by default**, with a visible toggle and
an explanation, because the E-104 finding is that a naive literal comparison of
shadda-first gold against NFC output marks every geminate wrong.

### Tab 4 - Leaderboard
The published table plus a **cost-vs-quality scatter** (x = params, log scale; y = DER),
our two models highlighted. Each row is badged `measured here` or `quoted`.

## Splitting algorithm

1. Split on blank lines into paragraphs, then on sentence terminators
   (`. ! ? ، ؛ ؟` and newline), keeping the delimiter attached to the preceding span.
2. Greedily pack spans into chunks under `max_chars` (default 400, per-model override
   where a model has a hard cap - e.g. T5's 256 tokens).
3. A single span longer than `max_chars` is hard-split at a word boundary; if a single
   word exceeds the budget it is emitted whole and **flagged in the report**, never
   silently truncated.
4. Record the exact separator span between chunks so reassembly is lossless.

Reassembly guarantee: `reassemble(split(x)) == x` for the *passthrough* text, and for a
diacritized result the separators are re-inserted at the recorded offsets. This is a
unit-tested invariant, not a hope.

## Honesty requirements (non-negotiable)

1. No unpublished model is named or compared anywhere.
2. Every third-party number is badged `quoted` with its source, never presented as
   measured here.
3. The mark-order caveat is visible in the Score tab and explained in the README.
4. The caveats panel states what the demo does **not** prove: sadeed25 overlap,
   wikinews2024 contamination, and that SadeedDiac-25's gold is not NFC.
5. Each model's pinned revision hash is shown, so a screenshot is reproducible.

## Testing

- `space/tests/test_splitter.py` - round-trip invariant, boundary cases, oversized span.
- `space/tests/test_scoring.py` - canonical metric agrees across mark orders; a known
  perfect match scores 0; an off-by-one mark scores as expected.
- `space/tests/test_registry.py` - registry loads, specs are well-formed, unknown key
  raises.
- `space/tests/test_adapters.py` - our two nano models run end-to-end on a sample and
  produce NFC output (skipped if weights unavailable).
- Headless boot: launch `app.py` on a free port with `--no-browser`, assert clean boot,
  then close it. Interaction claims additionally need a real functional call of each
  tab's underlying function (Gradio errors surface only at interaction time).

## Rollout

1. Build and test locally.
2. Create the Space repo, upload `space/` contents.
3. Verify the live Space boots and that the deployed app matches the local tree.
4. Record as `diacritizer/E-105`; update `state/tasks.yaml` and `MEMORY.md`; regenerate
   `CURRENT_STATE.md`.

## Risks

| risk | mitigation |
|---|---|
| Free CPU too slow for ByT5/350M models | third-party opt-in; per-model latency shown; hard char cap |
| RAM exhaustion loading several models | lazy load + LRU eviction to 2 resident models; `concurrency_limit=1` |
| A third-party model produces garbage | it is badged as its own output, never blended; the probe decides roster membership |
| Causal LM truncates or hallucinates | generation capped and the output length is compared to the input; mismatch is flagged |
| Gradio build/dependency drift on the Hub | pin versions in `requirements.txt` |
