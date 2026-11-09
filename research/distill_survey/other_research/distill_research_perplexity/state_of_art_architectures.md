# State-of-the-Art Model Architectures & Training Methods

**Focus:** DeepSeek-V4, MiniMax-M1, Qwen 3.8 Next-Flash, and other 2025-2026 frontier models

---

## 1. DeepSeek-V4-Flash Architecture

### 1.1 Model Specifications [web:16][web:18][web:22]

| Component | Specification |
|-----------|--------------|
| **Total Parameters** | 284 billion (MoE) |
| **Active Parameters** | 13 billion per forward pass |
| **Context Window** | 1 million tokens (native) |
| **Attention** | Hybrid CSA + HCA |
| **License** | MIT (open weights) |
| **Inference Cost** | 27% of V3.2 |
| **Pricing** | $0.14/$0.28 per 1M tokens (off-peak/peak) |

### 1.2 Hybrid Attention: CSA + HCA [web:18][web:22][web:49]

**CSA (Compressed Sparse Attention):**
- Reduces stored history length
- Attends only to most relevant tokens
- Sparse attention pattern learned during training

**HCA (Heavily Compressed Attention):**
- Compresses older context into summary representations
- Similar to Compressive Transformer but more aggressive
- Enables 1M token context without quadratic memory growth

**Combined Effect:**
- Recent tokens: Full attention (high resolution)
- Mid-range tokens: Sparse attention (medium resolution)
- Old tokens: Heavily compressed attention (low resolution)

### 1.3 Five-Phase Training Pipeline [web:22]

**Phase 1: Pre-training (32T+ tokens)**
- Optimizer: Muon
- Data mix:
  - 30-60% curated/synthetic high-quality
  - 30-50% filtered web (FineWeb-Edu, DCLM)
  - 5-15% code, math, science
  - 1-5% multilingual
  - 1-5% reasoning traces

**Phase 2: Long-Context Continual Training**
- Extend to 1M tokens
- Specialized positional encoding (RoPE variants)
- Hybrid attention training

**Phase 3: Independent Domain Expert Cultivation** [web:19][web:23][web:29]

Train 10+ specialist models independently:

| Expert Domain | Training Data | RL Reward |
|--------------|---------------|-----------|
| Math | MATH, GSM8K, AIME | Correctness, step validity |
| Code | HumanEval, MBPP, LeetCode | Execution success, test pass |
| Agents | Agent trajectories | Task completion, tool use |
| Instruction-following | UltraFeedback, HelpSteer | Helpfulness, harmlessness |
| Reasoning | Custom reasoning datasets | Logical coherence |

Each expert: SFT → GRPO (Group Relative Policy Optimization) RL

**Why separate?** Mixed RL causes capability interference. Separate training preserves domain expertise.

**Phase 4: Unified On-Policy Distillation** [web:24][web:25][web:27]

Student: Single V4-Flash model
Teachers: All 10+ domain specialists

Process:
1. Student generates trajectory from prompt
2. Each teacher provides full-vocabulary distribution for every token
3. Student loss = weighted sum of reverse KL divergences from all teachers
4. Update student weights
5. Repeat with new student-generated trajectories

Loss function:
```
L = Σ_i w_i * KL_divergence(student_output, teacher_i_output)
```

where w_i is weight for teacher i (can be domain-specific)

**Key Innovation:** Replaces "mixed RL" with "multi-teacher OPD"
- Mixed RL: One model trained on multiple domains → capability conflicts
- Multi-teacher OPD: Distill all experts into one → preserves all capabilities

**Phase 5: Quantization & Finalization** [web:22]
- QAT (Quantization-Aware Training): Prepare for FP8/INT8 inference
- MTP (Multi-Token Prediction) draft head: Enable speculative decoding
- Result: 70%+ inference cost reduction

---

## 2. MiniMax-M1 Architecture

### 2.1 Model Specifications [web:37][web:40]

| Component | Specification |
|-----------|--------------|
| **Architecture** | Lightning Attention (linear attention variant) |
| **Training** | Continual pretraining + SFT + large-scale RL |
| **Tokens** | 7.5T tokens |
| **Context** | Extended context (exact size varies by version) |
| **Focus** | Test-time compute scaling |

### 2.2 Lightning Attention

**Key Idea:** Linear attention instead of quadratic softmax attention.

**Mechanism:**
- Approximates softmax attention with linear kernel
- O(N) complexity instead of O(N^2)
- Enables longer contexts with less memory

**Training:**
1. Continual pretraining on 7.5T tokens
2. SFT for chain-of-thought reasoning patterns
3. Large-scale RL on diverse problems (sandbox-based, real-world software engineering)

---

## 3. Qwen 3.8 Next-Flash / 3.8 Flash

### 3.1 Known Specifications

Based on Qwen family patterns and "Flash" naming:

| Component | Expected Specification |
|-----------|----------------------|
| **Parameters** | ~3.8B (efficient variant) |
| **Attention** | Likely Flash Attention 2 or 3 |
| **Context** | 32K-128K tokens (estimated) |
| **Quantization** | FP8/INT8 native support |
| **Distillation** | Likely distilled from larger Qwen models |

### 3.2 Flash Attention

**Flash Attention 2/3 Features:**
- Tiling to reduce memory reads
- Recomputation instead of storage (trade compute for memory)
- 2-4x speedup over standard attention
- Enables longer contexts on same hardware

---

## 4. Other Notable 2025-2026 Architectures

### 4.1 Google TurboQuant [web:2]

**Innovation:** KV cache compression up to 6x.

**Technique:**
- 3.5-bit quantization for KV cache
- Near-zero accuracy loss
- No retraining needed

**Impact:**
- Llama 70B @ 1M context: 328GB → ~55GB VRAM for KV cache

### 4.2 CompLLM [web:3][web:51][web:53][web:54]

**Architecture:**
- Segmented compression (20-token segments)
- LoRA layer + linear compression layer
- Concept Embeddings (CEs) in same space as token embeddings

**Results:**
- 4x TTFT speedup
- 2x decoding speedup
- 50% KV cache reduction
- Matches or exceeds baseline on long-context QA

### 4.3 AdmTree [web:56]

**Architecture:**
- Adaptive hierarchical compression
- Semantic tree structure
- Dynamic gist token allocation

**Results:**
- 10%+ improvement on LongBench
- Up to 20 points in QA tasks
- Balances global and local semantics

### 4.4 AttnComp [web:58]

**Architecture:**
- Attention-guided compression for RAG
- Cross-attention layer for relevance scoring
- Top-P adaptive compression

**Results:**
- 17x compression rate
- 1.9 point accuracy improvement
- 49% latency reduction

---

## 5. Common Architectural Patterns

### 5.1 Attention Mechanisms

| Model | Attention Type | Context | Memory |
|-------|---------------|---------|--------|
| DeepSeek-V4 | Hybrid CSA+HCA | 1M tokens | O(N) with compression |
| MiniMax-M1 | Lightning (linear) | Extended | O(N) |
| CompLLM | Segmented full | 100K+ | O(N/C^2) |
| AdmTree | Tree-based | 100K+ | O(N log N) |
| TurboQuant | Standard + quant | 1M+ | O(N) with 6x compression |

### 5.2 Training Strategies

| Model | Pre-training | Post-training | Distillation |
|-------|-------------|---------------|--------------|
| DeepSeek-V4 | 32T tokens, Muon | Independent experts + OPD | Multi-teacher OPD |
| MiniMax-M1 | 7.5T tokens | SFT + large-scale RL | Not specified |
| Qwen-Flash | Unknown | Likely SFT + RL | Likely from larger Qwen |

### 5.3 Context Extension Techniques

1. **Attention Compression**
   - Sparse attention (DeepSeek CSA)
   - Linear attention (MiniMax Lightning)
   - Segmented attention (CompLLM)

2. **Memory Compression**
   - KV cache quantization (TurboQuant)
   - Compressive memory (DeepSeek HCA)
   - Hierarchical summaries (AdmTree)

3. **External Memory**
   - Associative matrices (ARMT)
   - Fixed-size memory (Infini-Attention)
   - Vector retrieval (RAG systems)

---

## 6. Lessons for Your Training

### 6.1 From DeepSeek-V4

**Do:**
- Train domain specialists independently before merging
- Use on-policy distillation (student generates, teachers grade)
- Multi-teacher OPD instead of mixed RL
- Hybrid attention for long contexts

**Avoid:**
- Mixed RL on generalist model (causes capability conflicts)
- Naive weight merging (loses domain expertise)
- Full attention for 100K+ tokens (quadratic memory)

### 6.2 From MiniMax-M1

**Do:**
- Consider linear attention for very long contexts
- Large-scale RL on diverse, real-world problems
- Chain-of-thought SFT before RL

**Avoid:**
- Standard attention for 100K+ contexts
- RL without domain-specific reward models

### 6.3 From CompLLM/AdmTree

**Do:**
- Segment-wise compression (not holistic)
- Train lightweight compressor, freeze base model
- Use Concept Embeddings in same space as tokens

**Avoid:**
- Compressing entire context at once (quadratic complexity)
- Compressors requiring base model retraining

---

## 7. Implementation Priorities

### Immediate (Day 1-7)
1. Enable FP8 KV cache (50% savings, 1 flag)
2. Add prefix caching (2-4x for repeated prompts)
3. Start black-box distillation pipeline

### Short-term (Week 2-4)
1. Train segmented compressor (CompLLM-style)
2. Add external memory module
3. Implement multi-teacher distillation (if multiple APIs available)

### Medium-term (Month 2-3)
1. Experiment with linear attention variants
2. Train on-policy distillation with your rollouts
3. Implement hybrid attention (sparse + compressed)

### Long-term (Month 4-6)
1. Full DeepSeek-V4-style pipeline (independent experts + OPD)
2. Custom architecture optimizations for your use case
3. Production deployment with monitoring and iterative improvement

---

## 8. Key Papers & Resources

### DeepSeek-V4
- Technical deep dive: [web:22]
- Training methodology: [web:24][web:27]
- Architecture details: [web:18][web:49]

### CompLLM
- Paper: arXiv:2509.xxxxx (search "CompLLM compression")
- Implementation: TheMoonlight.io review [web:51]

### AdmTree
- Paper: arXiv:2512.xxxxx (search "AdmTree adaptive")
- Review: TheMoonlight.io [web:56]

### AttnComp
- Paper: arXiv:2602.07778
- Review: TheMoonlight.io [web:58]

### Survey Papers
- Distillation in 2026: Hugging Face Blog [web:21]
- On-Policy Distillation Survey: arXiv:2604.00626 [web:34]
- Knowledge & Dataset Distillation: Springer 2025 [web:61]

---

**Last Updated:** September 8, 2026  
**Version:** 1.0
