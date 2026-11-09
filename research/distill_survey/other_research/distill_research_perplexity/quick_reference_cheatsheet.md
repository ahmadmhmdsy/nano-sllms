# Quick Reference Cheatsheet: Distillation & Context Compression

---

## Distillation Techniques

| Technique | Teacher Access | Data Needed | Performance | Complexity |
|-----------|---------------|-------------|-------------|------------|
| **Hard-Label SFT** | API only | 80K-100K examples | Good | Easy |
| **Soft-Label KD** | Logits | 20K-50K examples | Better | Medium |
| **Feature Distillation** | Hidden states | 10K-30K examples | Best | Hard |
| **On-Policy Distillation** | API + rollouts | 50K-200K rollouts | Excellent | Medium |
| **Multi-Teacher OPD** | Multiple APIs | 100K-500K rollouts | SOTA | Hard |
| **GAD (Adversarial)** | API only | 50K-100K examples | Very Good | Hard |

---

## Context Compression

| Technique | Memory Savings | Accuracy Loss | Implementation | Best For |
|-----------|---------------|---------------|----------------|----------|
| **FP8 KV Cache** | 50% | <0.1% | 1 flag | Everyone |
| **INT8 KV Cache** | 50% | <0.5% | 1 flag | Most GPUs |
| **Prefix Caching** | 2-4x (repeated) | 0% | 1 flag | RAG, few-shot |
| **CompLLM** | 2-4x | <1% | Train compressor | Long docs |
| **KV Eviction** | 2-8x | 1-5% | Config | Streaming |
| **External Memory** | Unlimited* | 2-10% | Custom module | Agents |

*Limited by external storage, not VRAM

---

## Quick Commands

### Serve with FP8
```bash
vllm serve ./your-model --kv-cache-dtype fp8_e5m2 --max-model-len 8192 --enable-prefix-caching
```

### Generate Teacher Data
```python
import openai
response = openai.ChatCompletion.create(model="gpt-4-turbo", messages=[{"role": "user", "content": "Your prompt"}])
```

### Fine-Tune with LoRA
```python
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)
```

---

## Data Requirements

| Task | Seed Prompts | Training Time |
|------|-------------|---------------|
| Narrow task | 1K-5K | 1-2 hours |
| Domain QA | 5K-20K | 2-6 hours |
| Instruction-following | 20K-100K | 6-24 hours |
| Multi-domain | 50K-200K | 1-3 days |

---

## VRAM Requirements

| Model | LoRA FT | Inference (BF16) | Inference (FP8) |
|-------|---------|------------------|-----------------|
| 1B | 4GB | 2GB | 1GB |
| 3B | 8GB | 6GB | 3GB |
| 7B | 12GB | 14GB | 7GB |
| 14B | 24GB | 28GB | 14GB |

*Assumes 1K context

---

## Performance Benchmarks

| Student | Teacher | Examples | Performance |
|---------|---------|----------|-------------|
| Qwen2.5-1.5B | GPT-4 | 10K | 75-85% |
| Qwen2.5-3B | GPT-4 | 50K | 85-92% |
| Qwen2.5-7B | GPT-4 | 100K | 92-97% |

---

## API Costs

| Model | Cost/1K tokens | 10K Examples | 100K Examples |
|-------|---------------|--------------|---------------|
| GPT-4-turbo | $0.01 | ~$100 | ~$1,000 |
| GPT-4o | $0.005 | ~$50 | ~$500 |
| Claude-3-Haiku | $0.0008 | ~$8 | ~$80 |

---

## Quality Gates

Before rollout:
- [ ] ROUGE-L > 0.75 vs teacher
- [ ] Hallucination rate < 5%
- [ ] Format compliance > 95%
- [ ] Latency < 2x teacher
- [ ] Cost savings > 50%

---

## Common Pitfalls

❌ Low-quality seed prompts → garbage in, garbage out  
✅ Use diverse, representative prompts

❌ No deduplication → overfitting  
✅ Deduplicate (MinHash + semantic)

❌ No test set → can't measure performance  
✅ Hold out 20% for evaluation

❌ Skip progressive rollout → production failures  
✅ 10% → 25% → 50% → 75% → 90% with gates

---

**Last Updated:** September 8, 2026
