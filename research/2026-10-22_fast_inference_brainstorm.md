# Faster inference for the SmolLM2-135M agent: brainstorm (2026-10-22)

Status: brainstorm only. Nothing measured or built. GPU was busy with SA-3 when written.

## Facts about our model and stack

- SmolLM2-135M = Llama: 30 layers, hidden 576, 9 query / 3 KV heads (GQA), head_dim 64, vocab 49,152, tied embeddings.
- KV cache per token = 30 layers x 2 (K,V) x 3 heads x 64 x 2 bytes (fp16) = **22.5 KiB**. A 1k-token prompt is ~23 MB.
  Storing the cache for thousands of prompts in RAM or on disk is cheap.
- Weights are ~270 MB in fp16. The LM head (49,152 x 576 = 28M multiply-adds per token) is **~21 % of the
  per-token FLOPs**. The 30 blocks are the other ~106M.
- Every `generate()` call in the repo (eval.py, bench_passk.py, rl_smollm2.py, ...) already uses the default
  DynamicCache. Inside one call, the prompt is computed once and each new token costs one position. **Ordinary KV caching is already on.**
  Nothing is reused *across* calls.
- Installed transformers 5.16.1 has `prompt_lookup_num_tokens`, `assistant_model` and `assistant_early_exit`.
  All three are batch-size-1 only, with `num_return_sequences` = 1 (generation/utils.py:3669, 1575).
- Not installed: triton (so no torch.compile inductor on Windows), vLLM, SGLang, llama.cpp, onnxruntime.
- At batch 1, a 135M model on this GPU is likely **kernel-launch-bound**: ~300+ small kernels per token, with the GPU
  mostly idle between them. This is unmeasured; it is the first thing to measure.

## Options, ranked for us

| # | Method | Training? | Lossless? | Fits our stack | Expected win | Verdict |
|---|---|---|---|---|---|---|
| 1 | Prompt-lookup / n-gram drafting (copy spans from the prompt, verify in one forward) | no | yes (greedy); yes with rejection sampling | built in, batch 1 | large on fix-from-traceback / edit tasks, where the output repeats the input code; ~none on fresh code | **do first** |
| 2 | Prefix KV cache (system prompt + tool schema computed once and reused) + multi-turn cache (append each tool result instead of re-prefilling the whole conversation) | no | yes (verify vs no-cache baseline, forum reports small numeric drift) | `past_key_values=` + copy | big for long agent loops; small for short single prompts | **do in SA-7 runtime** |
| 3 | Jump-forward decoding under the constrained JSON grammar (tokens the grammar forces are appended without a model call) | no | yes | own logits processor, SA-7 already plans constrained JSON | moderate on tool-call turns | **do with SA-7** |
| 4 | CUDA-graph decode step (static cache, capture one decode forward, replay) | no | yes | manual `torch.cuda.CUDAGraph` works **without triton** | potentially the largest batch-1 win if launch-bound | measure first |
| 5 | Suffix decoding (n-gram tree over prompt **and previous outputs**, for retry loops) | no | yes | own `custom_generate` | good for retry-with-traceback loops | after #1 proves out |
| 6 | Shared-prefix sampling for pass@k / RL groups (prefill once, expand the cache k times) | no | yes | own loop | small (prefill is one parallel pass) | low priority |
| 7 | Vocab trimming of the LM head to the tokens our Python corpus uses | light re-tune | yes if dropped tokens never occur | tokenizer remap | cuts the ~21 % LM-head cost and the sampling cost | idea; needs a token-usage scan |
| 8 | Medusa / MTP heads (extra heads predict t+2..t+k, tree verify) | yes (heads only, base frozen) | yes with verify | custom tree attention | 1.5-2.5x in papers on 7B+; unknown at 135M | later research |
| 9 | LayerSkip self-speculation (early layers draft, full model verifies) | yes (early-exit loss + layer dropout in SFT) | yes | `assistant_early_exit` built in | 1.3-2x in paper; a 30-layer thin model may lose quality at early exits | later research |
| 10 | Separate draft model | — | yes | — | a draft must be ~10x smaller (~13M); not worth it | **no** |
| 11 | Jacobi / CLLM consistency training | heavy | yes | — | 2.4-3.4x in paper, on 7B | **no** for now |
| 12 | Reusing activations for non-prefix text (CacheBlend-style) | — | **no**, approximate | — | — | **no**: a token's activations depend on everything before it |

## Caveats

- Speculative methods help **latency at small batch**. At batch 32 (our eval) or with group sampling (RL), the GPU is already
  busy, and drafting can make things slower. "Speculative Decoding: Performance or Illusion?" (arXiv 2601.11580) is a warning on this.
- Sampling with temperature lowers draft acceptance compared with greedy decoding.
- Evidence at the 135M scale is thin. Every claim here must be benchmarked on our own HE/MBPP/fix prompts.

## Measurement plan (after SA-3 frees the GPU)

1. Tokens/s at batch 1 and batch 32 for the incumbent, with prefill and decode split out. Profile one decode step
   (kernel count, GPU busy %, LM-head share).
2. The same run with `prompt_lookup_num_tokens=10` on HE, MBPP and fix rows, greedy. Report the speedup and
   check that outputs are identical.
3. Decide on #4 from the profile.
