# Pre-registration (markdown anchor of the registry entry)


# Pre-registration 2026-09-26 - pipeline_v2 post-remediation promotion (pilot + target)
question: >
  Does the remediated pipeline_v2 (declared schedule, accumulation, policy
  selection, held-out lane) settle the pilot (P) and target (T) promotion
  sequences the Release 1 runbook declared, moving them from
  NEEDS_USER_DECISION to recorded facts?
arms:
  - id: p-pilot-post-remediation
    config: pipeline_v2/configs/experiments/pilot.yaml
    stage: P
    gates: [evaluation.thresholds from the resolved config (val_loss, perplexity, generation_smoke)]
    metric_source: runs/v2/core/pilot/internal_evaluation/evaluations/COMPLETE.json + train/checkpoints/COMPLETE.json
  - id: t-target-post-remediation
    config: pipeline_v2/configs/experiments/target.yaml
    stage: T
    gates: [same declared thresholds]
    metric_source: runs/v2/core/target/... (same layout)
gpu_window_estimate: >
  Quadro RTX 3000 (sm_75, fp16, 6 GB); base r1 profile nano-200m;
  2000 optimizer steps each at batch 1 x accum 8, ctx 1024.
  Measured right before launch per launch gate.
data: data/target/raw/train.jsonl (pinned role train; unpackaged pilot reuses the same declared source per its config), tokenizer data/target/tokenizer/tokenizer.json
overlap_check: >
  Both arms consume the SAME declared source at DIFFERENT declared stages of
  the registered S->P->T ladder; the sequence itself is the registration
  (each run is its own generation-derived artifact under runs/v2/core/);
  no tacit overlap with a foreign anchor.
verdict_source: runs/v2/core/<run>/status.json + train/checkpoints/COMPLETE.json + external_gates
