# 2026-10-13 - Two-pass / critic literature scan (prep for the base -> discriminator -> base idea)

Scanned before designing any D-line two-pass experiment. Treat as research input,
not as a result: nothing here has been measured on THIS line.

## 1. base -> critic -> base: published gains come from added capacity, not critic semantics

- Deliberation Networks (Xia et al., NeurIPS 2017): the second pass is a FULL second
  decoder trained multi-stage. The gain is capacity + training, not "checking".
- Adaptive Multi-pass (Geng et al., D18-1048): +1.55 BLEU; the polisher is driven by an
  RL policy and is itself a model.
- Two-pass ASR (arXiv:1908.10992): 17-22 pct relative WER drop, but pass 2 is a large
  non-streaming LAS, i.e. much more compute.
- Mask-Predict (D19-1633) / Levenshtein transformer (arXiv:1905.11006): quality-neutral
  against strong baselines; the win is SPEED.
- Reranking needs candidate DIVERSITY: Niehues (arXiv:1708.00563) shows the better
  hypothesis is already in a 50-best list.
- Vassertail & Levy (arXiv:2212.08926): a reranker with NO extra data/params is only
  slightly better than raw log-prob, and the gain comes from interpolation.

CONSEQUENCE FOR THIS LINE: this line decodes with plain per-character argmax, so there
is NO candidate set to re-rank. A critic would have to GENERATE alternates, which means
it is a second model with real capacity - not a cheap "checker" bolted on.

SPECULATION, flagged as such: no published work tests a critic loop on Arabic
diacritization at all.

## 2. What is measured to work, and what is not

MEASURED TO WORK - constrained / validity decoding. arXiv:2607.18946 restricts CTC
hypotheses to valid diacritized realizations and reports ArVoice test DER 19.21 -> 12.04
and 11.90 -> 7.73. This is the one mechanism with a DER number behind it.

MEASURED NOT TO WORK in a matched-resource regime:
- learned reranker ~ties the baseline (arXiv:2212.08926);
- intrinsic self-correction without external feedback can DEGRADE (arXiv:2310.01798);
- no source measures an adversarial/GAN gain in DER or char accuracy for diacritization.
  SLGAN (ACL 2024, emnlp-main.593) has a discriminator, but it predicts per-tag
  correctness and is JOINT training, not a post-hoc critic.

## 3. Cheapest test design (< 1 h GPU), as an experiment plan - NOT a literature result

Because vote/centering/ensemble are all measured no-ops on this line, do not start with
a critic. Start with the mechanism that has DER evidence:

(a) LEGAL-CLASS MASK AT DECODE. Per-character constraints: shadda takes exactly one
    partner mark; no two non-shadda marks stack; tanween is word-final only; no sukoon on
    the first letter; some letters accept no mark. Near-zero GPU.
    CHEAP PRE-SCREEN, CPU-ONLY: count how many ILLEGAL configurations already appear in
    the existing prediction files. If the count is ~0 the mask cannot help and the axis
    closes for free, before spending any GPU.

(b) TINY SINGLE-PASS REPAIR HEAD. Same 15 classes, ~4 conv / 1 layer, < 0.2M params,
    trained on (base argmax, gold) pairs, teacher-forced, ctx 256, ~10-20 min. This is
    base -> checker -> base as a single-pass POST-EDITOR, not an iterative loop.

DO NOT expect a same-size bidirectional critic to discover errors the base encoder
already saw (Huang, arXiv:2310.01798).

## Sources

- https://proceedings.neurips.cc/paper_files/paper/2017/hash/c6036a69be21cb660499b75718a3ef24-Abstract.html
- https://aclanthology.org/D18-1048/
- https://ar5iv.labs.arxiv.org/html/1908.10992
- https://aclanthology.org/D19-1633/
- https://arxiv.org/abs/1905.11006
- https://ar5iv.labs.arxiv.org/html/1708.00563
- https://ar5iv.labs.arxiv.org/html/2212.08926
- https://arxiv.org/html/2310.01798v2
- https://arxiv.org/html/2607.18946v1
- https://aclanthology.org/2024.emnlp-main.593/

## 4. CPU-only pre-screen result, 2026-10-13: the validity-constraint axis has NO headroom

Run BEFORE any GPU was spent, on E-097 arm B's existing predictions, by
scratch/e99_validity_prescreen.py. Counted the three positional patterns a
legal-class mask could in principle fix (sukun on a word's first letter, shadda on a
word's first letter, tanwin on a non-final letter), over full corpora:

    fadel_test  GOLD         words=107291  total=2998  (2.7943 pct of words)
    fadel_test  armB single  words=107291  total=2968  (2.7663 pct of words)
    abdou_test  GOLD         words=38217   total=1232  (3.2237 pct of words)
    abdou_test  armB single  words=38184   total=1218  (3.1898 pct of words)

READ THIS CAREFULLY. The GOLD references contain these patterns at the SAME rate as
the model, so the patterns are NOT illegal forms - they are how the corpus is actually
written, and my word-splitting is coarser than the corpus's own orthography. The
correct conclusion is therefore NOT "the constraints are never violated". It is: THE
MODEL'S OUTPUT ALREADY MATCHES THE REFERENCE'S POSITIONAL STATISTICS, so a mask can
only ever move the roughly 1 pct of these patterns that differ - about 0.03 pct of
words.

CONSEQUENCE: mechanism (a) from section 3 is CLOSED for this line, with zero GPU spent.
The remaining option-4 candidate is (b), the tiny single-pass repair head - and the
same pre-screen logic applies to it: it can only win if the base model's errors are
SYSTEMATIC rather than the same noise the reference already contains. That must be
measured before it is trained, not after.
