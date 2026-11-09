# Special tokens for LLMs: a sourced survey, filtered for our 25.7M byte model

**Date:** 2026-10-15  
**Author:** delegated research subagent (CPU + network only; no GPU work, no training launched)  
**Status:** DONE (survey + judgment). Written incrementally; this is the landed version.  
**Companion docs:** [research/2026-10-15_eos_rebaseline.md](2026-10-15_eos_rebaseline.md), [research/2026-10-15_fim_corpus.md](2026-10-15_fim_corpus.md) (FIM corpus landed 2026-10-15), [MEMORY.md](../MEMORY.md) 2026-10-14/15 entries.

> **Evidence labels:** **[S]** = sourced fact with a link; **[REPO]** = measured in this repository; **[I]** = my inference, not a sourced finding.
> **Rules followed:** every URL below was actually retrieved (Exa search / contents), no paper or finding is invented. A token is never recommended without the data that would teach it.

---

## TL;DR: ranked short list for OUR model

| Rank | Token(s) | Verdict | Why |
|---|---|---|---|
| 1 | **EOS (id 2)** | **KEEP, make it mandatory in every packed corpus** | Already proven here: 3/71 -> 59/71 strict when example-level EOS entered the data. This is the only special we have causal evidence for. |
| 2 | **FIM sentinels** (prefix / middle / suffix) | **ADD, as the single next token experiment** | It is the one new token set whose training data already exists: FIM is a programmatic transform of the current corpus, i.e. free data augmentation. |
| 3 | SEP (id 3) | **DROP** | EOS already delimits examples; a second separator buys nothing. sep_id has no writer anywhere in the repo. |
| 4 | BOS (id 1) | **FREEZE, do not invest** | Optional in modern models (Qwen ships none); our runtime blocks it at generation, so any corpus that writes BOS is training a token it can never emit. |
| 5 | PAD (id 0) | **KEEP, structural only** | Needed to rectangularise batches; never a content target. Can be aliased to EOS if an id is ever needed. |
| 6 | UNK | **DROP** | A byte tokenizer has no out-of-vocabulary input by construction. |
| 7 | MASK | **DROP** | Encoder/MLM token (BERT); the decoder-only equivalent is FIM, ranked above. |
| 8 | Chat / role / turn tokens (`<\|im_start\|>`, `<\|im_end\|>`, roles) | **DROP** | Our task is single-turn generation; chat tokens need multi-turn data to mean anything. |
| 9 | Reasoning / thinking tokens (` thinking`) | **DROP** | Needs large CoT data plus RL-scale compute; no evidence at 25.7M params. |
| 10 | Tool / execution tokens (`<\|python_tag\|>`, `<\|eom_id\|>`) | **DROP** | No tool loop in this task; the sandbox evaluates, it does not call back. |

**The filter that decides nearly all of this:** a special token that no training example exercises is worse than no token at all. Our EOS incident is the local proof: eos_id == 2 sits below byte_offset == 4, so every packed corpus contained only ids >= 4 and the model had literally never seen its own stop token; every completion ran to the generation cap until example-level EOS was added to the data ([REPO] [research/2026-10-15_eos_rebaseline.md](2026-10-15_eos_rebaseline.md), [MEMORY.md](../MEMORY.md) 2026-10-14 and 2026-10-15). A broader statement of the same failure mode appears in Sebastian Raschka's tokenization FAQ: *using the wrong end token or chat template can cause generation and stopping errors even when the regular text tokenization is correct* ([S] [raschka](https://sebastianraschka.com/faq/docs/tokenization-bpe.html)).

---

## 0. Our model's constraints (the filter applied to everything below)

[REPO] From [glm-5.3-flash-from-scratch/glm53_flash/tokenizer.py](../glm-5.3-flash-from-scratch/glm53_flash/tokenizer.py):

- `pad_id = 0`, `bos_id = 1`, `eos_id = 2`, `sep_id = 3`, `byte_offset = 4`, `vocab_size = 260`.
- ids 0-3 are the only specials; 4-259 are the 256 byte values (`4 + utf8_byte`). There are **two genuinely free ids** by the user's count; measured usage in this repo is: eos is now written (example-level EOS corpora), bos is written by a few experiment/SFT corpora, pad is used as a batching pad value, **sep has no writer at all**.
- [REPO] [glm53_flash/runtime.py](../glm-5.3-flash-from-scratch/glm53_flash/runtime.py) line ~357 masks logits with `next_logits[:, [pad_id, bos_id, sep_id]] = -inf` at generation. **EOS is the only special the model is allowed to emit at inference.** Any new *generative* sentinel must be removed from that mask list, or it can never be produced.
- [REPO] The model is 25.7M total / 9.8M active params, trained on an 88-rung playground ladder of short Python functions, single-turn, no chat, no tool loop ([TASKS.md](../TASKS.md) row 162; [research/2026-10-15_eos_rebaseline.md](2026-10-15_eos_rebaseline.md)).

**Architectural cost of a new token [I]:** `vocab_size` is exactly `4 + 256`. Adding *k* tokens means `vocab_size = 260 + k`, resizing the token embedding and the LM head, and load-time surgery on every existing checkpoint. The parameter cost is negligible (k rows of d_model), but the checkpoint-compatibility cost is real. Two ways out are contrasted in section 6: real ids (vocab resize) versus byte-string sentinels that reuse the existing byte vocabulary.

---

## 1. The standard set: BOS, EOS, PAD, UNK, MASK

### EOS (id 2 for us) - essential for a decoder-only model

**[S]** The end-of-sequence token is what lets an autoregressive model terminate. GPT-2's byte-level BPE vocabulary carries `<|endoftext|>` for this role and it is also the document separator in packed data ([S] [GPT-2 paper PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf); a study of the token's effect on generation finds it changes sample quality: [S] [Semantics of the Unwritten, arXiv:2004.02251](https://arxiv.org/abs/2004.02251)).
**[S]** Qwen uses `<|endoftext|>` as an end-of-document control token between packed documents and aliases `eos` to its end-of-turn token for inference stopping ([S] [Qwen concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)).
**[REPO]** In our model the role of EOS is not theoretical. The corpus without it produced 71/71 completions at exactly the 160-byte budget (it never stopped even once); after example-level EOS entered the corpus, min/median/max completion length became 21/57/141 bytes with 0/71 at budget, and strict score went 3/71 -> 59/71 ([REPO] [MEMORY.md](../MEMORY.md) 2026-10-15; [research/2026-10-15_eos_rebaseline.md](2026-10-15_eos_rebaseline.md)).

**Verdict:** the one non-negotiable special. Not a candidate to add - a candidate to enforce everywhere.

### BOS (id 1 for us) - optional, model-specific

**[S]** Llama 2 uses the SentencePiece BOS (`<s>`) and places it between every user and assistant message in multi-turn chat ([S] [Meta Llama 2 model cards](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/meta-llama-2)).
**[S]** Qwen deliberately ships **no BOS**: *Qwen does not prepend a fixed token to each packed training sequence* ([S] [Qwen concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)). The same page warns not to set `bos` to `<|im_start|>` because it produces double start tokens.
**[I]** So BOS is an interface convention, not a capability. It can help a model locate the start of a sequence, but modern practice is split, and there is no small-model evidence that it earns its keep. For us it is worse than neutral: `runtime.py` masks bos at generation, so any corpus that includes bos trains a token the decoder is forbidden to emit - a train/inference mismatch, and exactly the class of bug that EOS was.

### PAD (id 0 for us) - structural, not semantic

**[S]** The Llama tokenizer docstring notes that the default padding token is unset *as there is no padding token in the original model*, and Transformers lets you add one when batching requires it ([S] [transformers Llama tokenizer](https://github.com/pytorch-tpu/transformers/blob/llama2-google-next-training/src/transformers/models/llama/tokenization_llama.py)).
**[S]** Qwen: *no pad token ... Qwen does not make use of padded sequences in training. One could use any special token together with the attention masks ... It is commonly set the same as eod* ([S] [Qwen concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)).
**[I]** PAD earns its id only if you pad. Our packed-shard data path is fixed-width windows, and the places that do pad ([train_aspect_lora.py](../glm-5.3-flash-from-scratch/scripts/train_aspect_lora.py) `make_batch`) mask the pad labels to `-100`. Keep id 0 for that, never as a modelling target; it could be aliased to EOS to reclaim the id if a future design needs one.

### UNK - obsolete for byte/subword models

**[S]** BERT relied on `[UNK]` because it used a fixed wordpiece vocabulary ([S] [BERT, arXiv:1810.04805](https://arxiv.org/abs/1810.04805)).
**[S]** Qwen reports *no unk token - BBPE ensures no unknown tokens* ([S] [Qwen concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)).
**[I]** Our ByteTokenizer is stronger than BBPE here: every UTF-8 byte maps to an id, so there is no input that can be out of vocabulary. UNK is unrepresentable and unnecessary; spending a reserved id on it would be waste.

### MASK - an encoder/MLM token, not a decoder one

**[S]** `[MASK]` exists because BERT is pretrained with masked-language modelling and must hide tokens from a bidirectional encoder ([S] [BERT, arXiv:1810.04805](https://arxiv.org/abs/1810.04805)); RoBERTa keeps a mask token while changing the masking schedule ([S] [RoBERTa, arXiv:1907.11692](https://arxiv.org/html/1907.11692v1)).
**[I]** A causal decoder never needs a MASK id: the causal mask is architectural, and the useful analogue of infilling is done by reordering the sequence with FIM sentinels (section 2). Adding MASK to a decoder-only model would be a token with no role.

---

## 2. Fill-in-the-middle (FIM): fim_prefix / fim_middle / fim_suffix / fim_pad

FIM is the one special-token family whose *data* is a transform of data we already have, which is why it dominates our short list.

**How the transform works.** Bavarian et al. split each document at random into prefix / middle / suffix and move the middle to the end, prefixed by sentinel tokens ([S] [Efficient Training of Language Models to Fill in the Middle, arXiv:2207.14255](https://arxiv.org/abs/2207.14255); [S] [OpenAI page](https://openai.com/index/efficient-training-of-language-models-to-fill-in-the-middle/)). Two orderings:

- **PSM:** `fim_prefix` prefix `fim_suffix` suffix `fim_middle` middle  
- **SPM:** the same three pieces with prefix and suffix swapped, so the prefix and middle become one contiguous continuation; the paper's motivation is KV-cache friendliness (appending to the prefix no longer invalidates the suffix's keys) ([S] arXiv:2207.14255 section 3.1).

Key sourced findings from that paper: FIM *does not harm the original left-to-right generative capability* (the *FIM-for-free* result); a FIM rate anywhere from 50% to 90% is reasonable; joint PSM+SPM training transfers between formats and is the most efficient; splits should be **character-level random spans**; **context-level** FIM beats document-level FIM; SPM is slightly stronger than PSM in their benchmarks but was suboptimal for mid-token robustness ([S] arXiv:2207.14255 sections 3-4).

**How the major code models instantiate it:**

- **Code Llama** extends the Llama 2 tokenizer with four special tokens, splits at the character level, applies the transform with probability 0.9, formats half PSM / half SPM, and reports an explicit cost/benefit: the 7B model loses 0.6 percentage points and the 13B loses 1.1 points across HumanEval+MBPP pass@1/@10/@100, which they judged worth the new capability ([S] [Code Llama, arXiv:2308.12950](https://arxiv.org/html/2308.12950v3)).
- **StarCoder** applies FIM at the character level with FIM-rate 0.5, PSM with probability 0.5 and SPMv2 with probability 0.5, and uses `<fim_prefix>`, `<fim_middle>`, `<fim_suffix>`, `<fim_pad>` ([S] [StarCoder, arXiv:2305.06161](https://arxiv.org/pdf/2305.06161); template shown in [S] [santacoder_fim_benchmark](https://github.com/arjunguha/santacoder_fim_benchmark/blob/master/generation.py)).
- **SantaCoder** (1.1B) ran FIM ablations as one of its de-risking experiments ([S] [SantaCoder, arXiv:2301.03988](https://arxiv.org/abs/2301.03988); [S] [HF model card](https://huggingface.co/bigcode/santacoder)).
- **DeepSeek-Coder** lists fill-in-the-middle as its second pretraining objective and notes that next-token prediction alone is insufficient to learn it; the released template is `<｜fim▁begin｜>prefix<｜fim▁hole｜>suffix<｜fim▁end｜>` (the exact Unicode full-width bars and `▁` matter, per the discussion thread) ([S] [DeepSeek-Coder, arXiv:2401.14196](https://arxiv.org/abs/2401.14196); [S] [HF discussion with template](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/discussions/20)).

**Does it work at small scale?** I found no peer-reviewed FIM result below ~1.1B params. Every model above is 1.1B-15.5B ([S] the Code Llama cost table, SantaCoder/StarCoder/DeepSeek-Coder papers). The mechanisms FIM relies on - reordering context and learning a sentinel-conditional infill distribution - are not obviously scale-gated, but **that is an unverified assumption for a 25.7M model [I]**, and it must be treated as the experiment's hypothesis, not its premise. There are newer FIM papers about *improving* infilling (context/curriculum, horizon-length prediction) but they too are at conventional code-model scale ([S] [arXiv:2412.16589](https://arxiv.org/abs/2412.16589), [S] [arXiv:2410.03103](https://arxiv.org/html/2410.03103v3)).

---

## 3. Chat / role / turn tokens (`<|im_start|>`, `<|user|>`, `<|assistant|>`, `<|endoftext|>`)

**[S]** The ChatML format used by Qwen is `<|im_start|>{role}\n{content}<|im_end|>`, with roles `system`/`user`/`assistant`; the same page documents `eod` (end of document) as the pretraining separator and states that frameworks stop on `eos`, which Qwen sets to the end-of-turn token ([S] [Qwen concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)).
**[S]** Llama 2 uses `[INST]`/`[/INST]` and `<<SYS>>` textual markers plus `<s>`/`</s>` ([S] [Meta Llama 2 model cards](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/meta-llama-2)); Llama 3.x uses header/turn specials such as `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, and for tools `<|python_tag|>` and `<|eom_id|>` ([S] [Meta Llama 3.1 model cards](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama3_1/)).
**[S]** Small models do ship these: SmolLM2-135M's `special_tokens_map.json` includes `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<repo_name>`, `<file_sep>`, `<gh_stars>`, `<issue_*>` and `<jupyter_*>` ([S] [SmolLM2-135M special_tokens_map.json](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/special_tokens_map.json)). That proves small models *can* carry many specials; it is **not** evidence that they *help**, since no ablation is reported in the model card [I].

**When they matter:** multi-turn dialogue, instruct/RLHF data, tool loops, and any setting where the role or turn boundary is a real decision the model must make. **When they are dead weight:** a single-turn, fixed-format task where the role never varies. Our task is the latter, so role/turn tokens would be pure vocabulary overhead with no data that varies the role.

---

## 4. Reasoning / thinking tokens, tool / execution tokens, verifier tokens

**Reasoning / thinking tokens.** DeepSeek-R1 introduced a reasoning mode whose chain-of-thought is delimited by thinking boundary tokens; vLLM's default DeepSeek-R1 reasoner uses `<think>` / `</think>` as the reasoning start/end strings, and DeepSeek's API exposes the CoT as a separate `reasoning_content` field ([S] [vLLM reasoning outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/); [S] [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/); [S] [DeepSeek-R1 tokenizer_config.json](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json)). The capability that makes the tokens meaningful - long CoT traces during RL - is described in the R1 paper ([S] [DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948)).
**[I]** A thinking token is a *container for a lot of generated text*, and the container is only useful if the model is trained on that text with a reward signal. We have neither CoT data nor an RL reasoning loop, and at 25.7M params most of the value of explicit reasoning is out of reach. Adding one would create a token that is essentially never sampled - the exact EOS failure mode in a new costume.

**Tool / execution tokens.** Llama 3.1 emits `<|python_tag|>` to start a tool call and `<|eom_id|>` (as opposed to `<|eot_id|>`) to say *do not stop, I expect the tool result back* ([S] [Meta Llama 3.1 model cards](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama3_1/)). DeepSeek's chat template contains a whole tool vocabulary: `<|tool calls begin|>`, `<|tool call begin|>`, `<|tool sep|>`, `<|tool call end|>`, `<|tool outputs begin|>`, `<|tool outputs end|>` ([S] [DeepSeek-R1 tokenizer_config.json](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json)).
**[S]** The orchestration cost is visible in DeepSeek's API rules: once tools are in play, the intermediate `reasoning_content` must be passed back on every subsequent turn or the API returns a 400 ([S] [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)).
**[I]** These tokens encode a control protocol between model and executor. Our playground has a sandbox *evaluator*, not an incremental executor: it runs the finished function and scores it. There is no mid-generation result to splice back in, so tool tokens have no producer and no consumer.

**A place to put intermediate state / call a verifier.** This is the most interesting *conceptual* slot, but I found no documented small-model special token that implements it. The closest real mechanism is FIM's middle sentinel: it gives the model an explicit slot whose content is conditioned on both sides of the context, which is a form of intermediate-state channel that is trained by data we can generate. I therefore fold this desire into the FIM recommendation rather than proposing a new `verifier` token with no data [I].

---

## 5. Small-model-specific evidence (below ~100M params)

**What I could source:**

- **[S]** TinyStories shows models far below 100M can produce coherent English when the data is appropriately simple; the comparison models named are GPT-Neo/GPT-2 *small* (~125M) ([S] [TinyStories, arXiv:2305.07759](https://arxiv.org/abs/2305.07759); [S] [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/tinystories-how-small-can-language-models-be-and-still-speak-coherent-english/)). It is evidence about data, not about special tokens.
- **[S]** SmolLM2-135M ships the full SmolLM special-token set (ChatML plus code/issue/jupyter markers) ([S] [special_tokens_map.json](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/special_tokens_map.json)); the SmolLM2 paper is data-centric and does not ablate special tokens ([S] [SmolLM2, arXiv:2502.02737](https://arxiv.org/html/2502.02737)).
- **[S]** BabyLM is the community benchmark for data-limited small models and explicitly studies tokenization (BPE vs morphology-aware vs syllable segmentation) at that scale ([S] [BabyLM guidelines](https://babylm.github.io/guidelines.html); [S] [Linguistic Units as Tokens, ACL 2025](https://aclanthology.org/2025.babylm-main.35/)) - but the comparison is segmentation strategy, not special-token presence.
- **[S]** Vocabulary scaling is measured at small scale: models from **33M** to 3B params, and the finding is that compute-optimal vocabulary size grows with model size ([S] [Scaling Laws with Vocabulary, arXiv:2407.13623](https://arxiv.org/abs/2407.13623)). A related result scales vocabularies generally and shows gains ([S] [Over-Tokenized Transformer, ICML 2025](https://proceedings.mlr.press/v267/huang25bb.html)). This is about vocabulary *size*, not special-token *inventory* - relevant context for a 260-id byte vocab, but not an answer here.
- **[S]** Post-hoc vocabulary expansion is risky: In-Place Tokenizer Expansion for Pre-trained LLMs studies exactly the *tokenizer fixed at the start of pre-training* problem and finds that the data mixture during recovery training decides whether prior ability survives ([S] [arXiv:2607.15232](https://arxiv.org/abs/2607.15232)). Anecdotally, a widely-cited bug report describes a DeepSeek-Coder model *completely losing performance* after `tokenizer.add_tokens(...)` because the new embeddings were never trained ([S] [huggingface/tokenizers #1490](https://github.com/huggingface/tokenizers/issues/1490) - a bug report, not a controlled study).

**The gap, stated honestly:** I found **no controlled ablation of special-token presence in a model below ~100M parameters**. Every statement in section 5 is adjacent evidence. The judgment in section 6 is therefore explicitly an inference from (a) the FIM/code-model results at >=1.1B, (b) the vocabulary-scaling results at >=33M, and (c) our own measured EOS result - not a direct small-model special-token finding. [I]

---

## 6. Judgment for OUR model, ranked by expected value

### #1 - EOS (id 2): keep and enforce. Already validated.

Do not touch the mechanism; extend its coverage. The rule that follows from the incident: **every** corpus that trains this model must carry an example-level terminator, and the terminator must be the id the decoder stops on. The packer already has `--append-eos` and the one example-level EOS corpus is `data/reference/bytes_pg_replay_eos` (2,876,493 ids / 16,800 examples) ([REPO] [research/2026-10-15_eos_rebaseline.md](2026-10-15_eos_rebaseline.md)). Before any new token work, re-pack `bytes_playground` with `--append-eos` so the largest task corpus is termination-complete.

### #2 - FIM sentinels: the only new tokens worth an experiment

**Why it wins:**
1. **The data already exists.** FIM is a deterministic transform of our reference functions: split a function at a random byte boundary, move the middle to the end, wrap with sentinels. No new corpus, no new downloads, no teacher model. This directly satisfies the user's point that FIM is attractive because it is free data augmentation.
2. **It matches the model's grain.** Our vocabulary is bytes, so a *character-level* split - the setting Bavarian et al. recommend - is simply a byte-level split. The mid-token robustness problem that complicates BPE models (Code Llama notes its SPM format never sees split subtokens while PSM does) does not exist for us [I].
3. **It composes with the one thing we have proven.** The middle is generated last, so an example-level EOS after the middle is exactly the termination it already learned.

**Why it is still an experiment, not a certainty:** the capability FIM buys is *infilling*, and our current task is left-to-right function generation. If we never evaluate infilling, FIM is a regularizer whose benefit at 25.7M params is unmeasured [I]. It must be run as a controlled A/B against the identical corpus with no FIM transform, with a new held-out infilling rung set, and the funnel must include EOS-present in both arms (per the falsification rule in [research/2026-10-15_eos_rebaseline.md](2026-10-15_eos_rebaseline.md) section 4).

**Two implementation routes [I]:**
- **(a) Real ids (recommended if we commit):** `vocab_size = 260 + 3 = 263` (prefix, middle, suffix; optionally +1 for `fim_pad` following StarCoder, though we do not left-pad in training, so `fim_pad` is droppable), resize embedding + LM head, remap old checkpoints on load. Cost is small in params, moderate in plumbing. **Also remove the new ids from the `runtime.py` generation mask**, or they can never be emitted.
- **(b) Byte-string sentinels (zero tokenizer change):** emit literal ASCII markers (for example `<|fim_prefix|>`) using existing byte ids. This is possible *only* because the model is byte-level, and it avoids checkpoint surgery. Costs: each marker is ~15 positions of context, sentinel recognition at eval is a string match, and the model could in principle emit marker bytes inside a real body (mitigated by choosing a marker that never occurs in Python source).

**Concrete teaching data:** transform `data/reference/bytes_pg_replay_eos` (or its larger EOS re-pack) at FIM rate 0.5, with PSM and SPM each taking half, character/byte-level spans, example-level EOS kept. With 16,800 examples that is ~8,400 FIM examples and ~25k sentinel positions - the same order as the 16,800 EOS occurrences that were already sufficient to teach termination [I]. New eval: hold out dev rungs, delete the reference body (or one line), feed prefix+suffix, and assert that the completion equals the reference and that EOS fires. Report score *and* completion length, as E-122g did.

**Landed (2026-10-15, CPU only):** the byte-string-sentinel FIM corpus this paragraph calls for is built and independently verified (9/9 checks) in [research/2026-10-15_fim_corpus.md](2026-10-15_fim_corpus.md) — `data/reference/bytes_pg_replay_fim/`, rate 0.5, PSM/SPM 50/50, byte-level spans, exactly one EOS per example, sentinels `b'<|fim_prefix|>'` / `b'<|fim_suffix|>'` / `b'<|fim_middle|>'` (`fim_pad` reserved, not emitted). The held-out infilling metric (exact middle match rate + EOS compliance + AR regression guard) is specified there; no model was trained.

### #3 - SEP (id 3): drop
EOS already marks the end of an example; a packed sequence does not need a second boundary id unless we want to distinguish *within-document* from *between-document* joins, which this task does not. `sep_id` has no writer anywhere in the repo [REPO]. It is a free id if FIM needs one, but giving it a second meaning while it is unexercised is the EOS trap again.

### #4 - BOS (id 1): freeze; do not add data for it
No sourced finding shows BOS improves a small code model; Qwen omits it ([S] Qwen concepts), Llama 2 uses it ([S] Meta Llama 2 cards). Our runtime masks it at generation [REPO], so a corpus that writes BOS is teaching an unusable token. Either remove BOS from corpora that write it, or stop masking it and make it a genuine per-example start - but that is a consistency bug-fix, not a new-token investment.

### #5 - PAD (id 0): keep, structural only
Used by batching paths and label-masked [REPO]. Never a target. If an id is ever needed, aliasing pad to eos is the standard move ([S] Qwen concepts: commonly set the same as eod).

### #6-#10 - UNK, MASK, chat, reasoning, tool tokens: drop
Each fails the filter for a different, sourced reason: UNK cannot occur in a byte vocabulary ([S] Qwen/BBPE); MASK belongs to MLM encoders ([S] BERT/RoBERTa); chat tokens need multi-turn role data we do not have ([S] ChatML/Qwen); reasoning tokens need long CoT data plus RL scale ([S] DeepSeek-R1); tool tokens need an executor that returns results mid-generation ([S] Llama 3.1). None would receive data in this project, so each would be a random embedding the model must learn to avoid rather than use.

---

## 7. The data rule, restated for whoever acts on this

For every token recommended or kept above, the matching data is named:

| Token | Data that teaches it | Status |
|---|---|---|
| EOS (2) | example-level EOS appended by the packer (`--append-eos`) | EXISTS (one corpus); extend to all |
| FIM sentinels | programmatic FIM transform (PSM/SPM, byte-level, rate ~0.5) of the existing reference corpus | TO BUILD (CPU only) |
| SEP (3), BOS (1), PAD (0), UNK, MASK, chat, reasoning, tool | none | DO NOT ADD |

**A token without data is harmful.** Our incident is the proof: appending an EOS the corpus never contained made termination impossible; the same token, once present in the data at the right semantic unit (one per function), took strict score from 3/71 to 59/71 ([REPO] [MEMORY.md](../MEMORY.md) 2026-10-14/15). Any future special-token proposal must arrive with the corpus change and the eval that exercises it, or it is a regression waiting to be measured.

---

## 8. Sources (all retrieved 2026-10-15 via the Exa helper)

**FIM**
- https://arxiv.org/abs/2207.14255 - Efficient Training of Language Models to Fill in the Middle (Bavarian et al.)
- https://openai.com/index/efficient-training-of-language-models-to-fill-in-the-middle/ - same, OpenAI summary
- https://arxiv.org/html/2308.12950v3 - Code Llama (FIM rate 0.9, PSM/SPM, 0.6/1.1 pt cost)
- https://arxiv.org/pdf/2305.06161 - StarCoder (FIM tokens, SPMv2)
- https://github.com/arjunguha/santacoder_fim_benchmark/blob/master/generation.py - FIM templates
- https://arxiv.org/abs/2301.03988 - SantaCoder (1.1B, FIM ablations)
- https://huggingface.co/bigcode/santacoder - SantaCoder model card
- https://arxiv.org/abs/2401.14196 - DeepSeek-Coder (FIM objective)
- https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/discussions/20 - DeepSeek-Coder FIM template
- https://arxiv.org/abs/2412.16589 - Improving FIM Code Completions via Context & Curriculum Based Learning
- https://arxiv.org/html/2410.03103v3 - Planning-Aware Code Infilling via Horizon-Length Prediction

**Standard special tokens**
- https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf - GPT-2
- https://arxiv.org/abs/2004.02251 - Semantics of the Unwritten (EOS effect on GPT-2)
- https://arxiv.org/abs/1810.04805 - BERT ([MASK], [UNK])
- https://arxiv.org/html/1907.11692v1 - RoBERTa (mask token)
- https://github.com/pytorch-tpu/transformers/blob/llama2-google-next-training/src/transformers/models/llama/tokenization_llama.py - Llama tokenizer docstring on padding
- https://qwen.readthedocs.io/en/latest/getting_started/concepts.html - Qwen control tokens (no unk, no pad, no bos; eod/eot)
- https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md - same, source form
- https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/meta-llama-2 - Meta Llama 2 special tokens
- https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama3_1/ - Meta Llama 3.1 special tokens, python_tag/eom_id
- https://sebastianraschka.com/faq/docs/tokenization-bpe.html - special tokens and stopping errors

**Reasoning / tool tokens**
- https://docs.vllm.ai/en/stable/features/reasoning_outputs/ - reasoning start/end tokens, thinking budgets
- https://api-docs.deepseek.com/guides/thinking_mode/ - reasoning_content handling
- https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json - R1 tokenizer/chat template
- https://arxiv.org/abs/2501.12948 - DeepSeek-R1

**Small-model / vocabulary evidence**
- https://arxiv.org/abs/2305.07759 - TinyStories
- https://www.microsoft.com/en-us/research/publication/tinystories-how-small-can-language-models-be-and-still-speak-coherent-english/ - TinyStories (MSR)
- https://arxiv.org/html/2502.02737 - SmolLM2
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/special_tokens_map.json - SmolLM2-135M special tokens
- https://babylm.github.io/guidelines.html - BabyLM guidelines
- https://aclanthology.org/2025.babylm-main.35/ - Linguistic Units as Tokens (BabyLM)
- https://arxiv.org/abs/2407.13623 - Scaling Laws with Vocabulary (33M-3B)
- https://proceedings.mlr.press/v267/huang25bb.html - Over-Tokenized Transformer
- https://arxiv.org/abs/2607.15232 - In-Place Tokenizer Expansion for Pre-trained LLMs
- https://github.com/huggingface/tokenizers/issues/1490 - add_tokens destroys performance (bug report)

**Repository evidence**
- glm-5.3-flash-from-scratch/glm53_flash/tokenizer.py - reserved ids and byte offset
- glm-5.3-flash-from-scratch/glm53_flash/runtime.py (generation mask of pad/bos/sep)
- research/2026-10-15_eos_rebaseline.md - corpus inventory by EOS class, falsification rule
- MEMORY.md 2026-10-14 (never taught to stop) and 2026-10-15 (example-level EOS breakthrough)
- research/2026-10-14_per_part_playground_design.md and TASKS.md row 162 - task shape and model size

---

## 9. Limitations of this survey

- **No controlled sub-100M special-token ablation was found.** The rank order in section 6 is an inference from adjacent evidence plus our own EOS result, and is labelled as such.
- **FIM at 25.7M is untested.** The recommendation is to run it as a bounded A/B with EOS present in both arms, not to assume the >=1.1B results transfer.
- **Cost figures for FIM are from 7B/13B models**, so no small-scale wall-clock or quality estimate is given here.
- This document reports no training result of its own; no GPU work was run.
