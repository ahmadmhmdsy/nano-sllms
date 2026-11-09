# LoRA/PEFT efficient fine-tune - design & cost (TASKS row 9)

STATUS: DESIGN COMPLETE (2026-09-06). User-approved scope: install peft and
write the design; the train.py/sft.py implementation is TASKS row 10 and stays
user-gated. No pipeline code changed - only this doc and a config template.

## 1. Verified on this machine (2026-09-06)

- peft 0.20.0 installed via uv AFTER a dry-run gate: 12 NEW packages, zero
  upgrades to existing ones - install was safe beside the live M3 run.
- Import check PASS: peft 0.20.0 + transformers 5.16.1 + torch 2.14.0+cu126.
- CPU LoRA smoke PASS: LoraConfig(r=8, lora_alpha=16, target_modules=['0','2'])
  on a 2-layer MLP -> trainable 3,072 of 19,648 params, exactly r*(in+out)*2
  per targeted linear; forward output shape correct.
- Ladder models are LlamaForCausalLM (src/model.py builds transformers' Llama
  arch with SDPA), so peft's STANDARD Llama target names apply as-is:
  q_proj k_proj v_proj o_proj gate_proj up_proj down_proj.
- bitsandbytes 0.50.2 verified on sm_75 earlier the same day (Tier 2 brief):
  8-bit base/optimizer options remain available later if needed.

## 2. Cost math (design estimates; measured numbers come from vram_probe in row 10)

Dims from the configs, verified by exact param-count match:
- T target 16L / h 1024 / kv 4 (kvd 256) / ffn 3072 / vocab 32768 tied:
  226,492,416 params = the known ~226.5M.
- custom 12L / h 768 / kv 4 / ffn 2048 / vocab 32768 tied:
  100,663,296 params = the known ~100.7M.

LoRA trainable at r=16 over all 7 projections (per layer r*(in+out) summed):
- T: 303,104 per layer x 16 = 4,849,664 trainable = 2.14% of the model.
- custom: 217,088 per layer x 12 = 2,605,056 trainable = 2.59%.

Optimizer state dominates (AdamW fp32 moments = 12 B/param):
- T full FT: ~2.7 GB optimizer state; LoRA r=16: ~58 MB.
- T training VRAM: 4.24 GB measured (M1 probe) -> est ~2.0-2.5 GB with LoRA
  (fp16 base ~0.45 GB + tiny opt/grad + roughly the same activation footprint).
  The freed headroom enables ctx 2048 or batch 4 x accum 8 inside 8 GB.
- custom full FT ~2.4 GB -> LoRA est ~1.0-1.5 GB.
- Adapter on disk: T r=16 ~19 MB safetensors - FITS the LFS quota (unlike the
  ~450 MB fp16 full model), so adapters could even be committed; default stays
  local like all weights (decide finally at row 10).

## 3. Pipeline integration sketch (row 10 scope)

- Configs get an optional peft block (shape below; the copy in
  configs/lora_example.yaml is commented until the hook exists):

      peft:
        r: 16
        lora_alpha: 32
        lora_dropout: 0.05
        target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
        bias: none

- ONE hook in train.py/sft.py: after model build (or base_model load), if the
  block is present, get_peft_model(model, LoraConfig(...)) then Trainer as
  usual - only the LoRA params train.
- Auto-resume contract (PLAN 5.3) must be re-proven: HF checkpoints save
  adapter state for a PeftModel; the kill/resume drill on smoke is part of
  row 10 acceptance.
- Save: final_dir receives adapter_model.safetensors + adapter_config.json;
  optional merge_and_unload() then writes the full safetensors so eval.py and
  infer.py stay unchanged (default: merge; an infer --adapter flag can come
  later if asked).
- fp16/sm_75 note: peft keeps adapter params in fp32 (autocast_adapter_dtype
  default) even on an fp16 base - stability-friendly; verify at row 10.
- tie_word_embeddings=true: lm_head is not in the default target list - no
  conflict with the tied embedding.
- User custom models: anything built via src/model.py (Llama arch) takes the
  same target names - no per-model lists needed.

## 4. Risks / open questions

- peft 0.20.0 vs transformers 5.16.1 save/resume drift - acceptance-tested at
  row 10 before any real run.
- vram_probe has no LoRA mode yet; row 10 adds --lora to measure real numbers
  and replace the section-2 estimates.
- Trainer grad-ckpt with a frozen base: expected fine on a single GPU; verify.

## 5. Acceptance criteria for row 10

1. sanity_check PASS on configs/lora_example.yaml (blocks uncommented).
2. CPU e2e tiny run with ONE kill/resume cycle; the adapter-only checkpoint
   resumes to the same loss trajectory.
3. vram_probe --lora measured on GPU post-M3; numbers pasted back here.
4. The merged final loads in eval.py and infer.py unchanged.
