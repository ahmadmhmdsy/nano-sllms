param([switch]$Run)
#
# 2026-10-15 EOS re-baseline - READY-TO-RUN GPU command list.
# Companion evidence + caveats: research/2026-10-15_rebaseline_verdicts.md
#
# DO NOT RUN without reading that doc. The commands below launch CUDA training and
# greedy scoring on the single shared Turing GPU. They were written but NOT executed.
#
# Usage:
#   . ./research/2026-10-15_rebaseline_gpu_commands.ps1            # dry list only (no GPU)
#   . ./research/2026-10-15_rebaseline_gpu_commands.ps1 -Run       # actually run, stage by stage
#
$ErrorActionPreference = "Stop"

if (-not $Run) {
    Write-Host ""
    Write-Host "DRY LIST ONLY - nothing executed."
    Write-Host "This file holds the C1 and C2 EOS re-baseline commands."
    Write-Host "Read research/2026-10-15_rebaseline_verdicts.md first."
    Write-Host "Re-run with -Run to execute."
    exit 0
}

$py = "./.venv/Scripts/python.exe"
$R  = "glm-5.3-flash-from-scratch/runs"

# =====================================================================
# STAGE 0 - preflight (Test-Path gate, forward slashes)
# =====================================================================
$required = @(
    "glm-5.3-flash-from-scratch/scripts/train_pretrain.py",
    "scratch/bench_playground_batched.py",
    "scratch/bench_aspect_pool.py",
    "scratch/bench_aspects.py",
    "data/reference/bytes_playground_eos/train.bin",
    "data/reference/bytes_playground_eos/val.bin",
    "data/reference/bytes_pg_replay_eos/train.bin",
    "data/reference/bytes_pg_replay_eos/val.bin",
    "glm-5.3-flash-from-scratch/runs/e122_ref600/checkpoint-0330",
    "glm-5.3-flash-from-scratch/runs/e122d_ref_pg/checkpoint-0510",
    "glm-5.3-flash-from-scratch/runs/e122d_ref_pg/checkpoint-0600"
)
foreach ($p in $required) {
    if (-not (Test-Path $p)) { throw "PREFLIGHT MISSING: $p" }
}
Write-Host "PREFLIGHT PASS: all source/corpus/checkpoint paths exist."

# Single-GPU guard: refuse to start if a REPO TRAIN JOB is resident.
# Foreign desktop/WebView processes (msedgewebview2, WidgetBoard, LockApp, ...) can hold a
# stale CUDA context and are NOT compute jobs - MEMORY lessons 9/16, CLAUDE.md 20 rule 9.
# Block only on python, which is what this repo's own train.py/sft.py runs appear as.
# 2026-10-15: the previous "throw on ANY app" version blocked a legitimate launch.
$busy = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
$repo = $busy | Where-Object { $_ -match "python" }
if ($repo) {
    Write-Host $repo
    throw "GPU has an active python compute app (possible repo train job). Do not launch (single-GPU rule)."
}
if ($busy) {
    Write-Host "FOREIGN non-python compute apps present - not a repo train job; proceeding per CLAUDE.md 20 rule 9:"
    Write-Host $busy
}
Write-Host "GPU IDLE (no python compute app): proceeding."

# NOTE (AGENTS.md 3): the config-based laya preflight gate does not directly cover
# train_pretrain.py. Cheaper gate first: read its --help and run a 2-step smoke before
# the full 600-step launch.

# =====================================================================
# STAGE C1 - EOS re-run of the E-122d / Phase 1.3 recipe
# =====================================================================
# Fixes the non-termination confound: same recipe as e122d_ref_pg but the corpus that
# has example-level EOS. Underlying variable: append_eos only (bytes_playground_eos),
# BUT note it is a 108-source superset of the original 47-source bytes_playground.
$c1train = @(
    "glm-5.3-flash-from-scratch/scripts/train_pretrain.py",
    "--corpus",       "data/reference/bytes_playground_eos/train.bin",
    "--val-corpus",   "data/reference/bytes_playground_eos/val.bin",
    "--init-from",    "glm-5.3-flash-from-scratch/runs/e122_ref600/checkpoint-0330",
    "--output",       "glm-5.3-flash-from-scratch/runs/e122d_ref_pg_eos",
    "--steps",        "600",
    "--batch-size",   "4",
    "--grad-accum",   "8",
    "--sequence-length", "1024",
    "--learning-rate",   "1e-4",
    "--lr-schedule",  "cosine",
    "--warmup-steps", "10",
    "--val-windows",  "32",
    "--checkpoint-fraction", "0.05",
    "--vram-cap-gib", "5.0",
    "--amp-dtype",    "fp16",
    "--device",       "cuda",
    "--seed",         "42"
)
& $py @c1train 2>&1 | Tee-Object -FilePath scratch/e122d_ref_pg_eos_train.log

# Scoring: rung battery (108 tasks) at the ORIGINAL instrument (max_tokens 64) for
# direct comparability with e122d_rung_cuda.json ...
$armsC1 = @(
    "--arm", "base=glm:glm-5.3-flash-from-scratch/runs/e122_ref600/checkpoint-0330",
    "--arm", "pg510=glm:glm-5.3-flash-from-scratch/runs/e122d_ref_pg/checkpoint-0510",
    "--arm", "pg510eos=glm:glm-5.3-flash-from-scratch/runs/e122d_ref_pg_eos/checkpoint-0510",
    "--arm", "pg600eos=glm:glm-5.3-flash-from-scratch/runs/e122d_ref_pg_eos/checkpoint-0600"
)
$rungOrig = @("scratch/bench_playground_batched.py") + $armsC1 + @(
    "--split","dev","--tier","all","--per-rung","4","--budget-bytes","160",
    "--max-tokens","64","--batch-size","8","--device","cuda",
    "--out","research/raw/e122d_rebaseline_rung_orig_cuda.json"
)
& $py @rungOrig 2>&1 | Tee-Object -FilePath scratch/e122d_rebaseline_rung_orig.log

# ... and at the REPAIRED instrument (budget 160, max_tokens 1000) which also removes
# the 48-byte-budget confound on the aspect and legacy batteries.
$rungFixed = @("scratch/bench_playground_batched.py") + $armsC1 + @(
    "--split","dev","--tier","all","--per-rung","4","--budget-bytes","160",
    "--max-tokens","1000","--batch-size","8","--device","cuda",
    "--out","research/raw/e122d_rebaseline_rung_fixed_cuda.json"
)
& $py @rungFixed 2>&1 | Tee-Object -FilePath scratch/e122d_rebaseline_rung_fixed.log

# Aspect-pool battery (48 tasks), repaired instrument.
$aspect = @("scratch/bench_aspect_pool.py") + $armsC1 + @(
    "--split","dev","--tier","all","--per-aspect","4","--budget-bytes","160",
    "--max-tokens","1000","--device","cuda",
    "--out","research/raw/e122d_rebaseline_aspectpool_fixed_cuda.json"
)
& $py @aspect 2>&1 | Tee-Object -FilePath scratch/e122d_rebaseline_aspectpool_fixed.log

# Legacy battery (32 tasks). CAVEAT: bench_aspects.py hardcodes BUDGET_BYTES and takes
# no --budget-bytes / --max-tokens; verify the constant before treating this as the
# repaired instrument, or add the args.
$legacy = @("scratch/bench_aspects.py") + $armsC1 + @(
    "--split","dev","--per-family","4","--device","cuda",
    "--out","research/raw/e122d_rebaseline_legacy_cuda.json"
)
& $py @legacy 2>&1 | Tee-Object -FilePath scratch/e122d_rebaseline_legacy.log

# =====================================================================
# STAGE C2 - replay A/B with EOS (both learning rates)
# =====================================================================
$c2base = @(
    "glm-5.3-flash-from-scratch/scripts/train_pretrain.py",
    "--corpus",       "data/reference/bytes_pg_replay_eos/train.bin",
    "--val-corpus",   "data/reference/bytes_pg_replay_eos/val.bin",
    "--init-from",    "glm-5.3-flash-from-scratch/runs/e122_ref600/checkpoint-0330",
    "--steps",        "600",
    "--batch-size",   "4",
    "--grad-accum",   "8",
    "--sequence-length", "1024",
    "--lr-schedule",  "cosine",
    "--warmup-steps", "10",
    "--val-windows",  "32",
    "--checkpoint-fraction", "0.05",
    "--vram-cap-gib", "5.0",
    "--amp-dtype",    "fp16",
    "--device",       "cuda",
    "--seed",         "42"
)
$lr1e4 = $c2base + @("--learning-rate","1e-4","--output","glm-5.3-flash-from-scratch/runs/e122e_replay_lr1e4_eos")
$lr3e5 = $c2base + @("--learning-rate","3e-5","--output","glm-5.3-flash-from-scratch/runs/e122e_replay_lr3e5_eos")
& $py @lr1e4 2>&1 | Tee-Object -FilePath scratch/e122e_replay_lr1e4_eos_train.log
& $py @lr3e5 2>&1 | Tee-Object -FilePath scratch/e122e_replay_lr3e5_eos_train.log

# C2 scoring: core playground, per-rung 1, budget 160, max_tokens 256 (same instrument
# as e122e_ab_256.json). Pairs each old NO-EOS arm with its new EOS twin.
$c2score = @(
    "scratch/bench_playground_batched.py",
    "--arm", "base=glm:glm-5.3-flash-from-scratch/runs/e122_ref600/checkpoint-0330",
    "--arm", "lr1e4=glm:glm-5.3-flash-from-scratch/runs/e122e_replay_lr1e4/checkpoint-0600",
    "--arm", "lr1e4eos=glm:glm-5.3-flash-from-scratch/runs/e122e_replay_lr1e4_eos/checkpoint-0600",
    "--arm", "lr3e5=glm:glm-5.3-flash-from-scratch/runs/e122e_replay_lr3e5/checkpoint-0600",
    "--arm", "lr3e5eos=glm:glm-5.3-flash-from-scratch/runs/e122e_replay_lr3e5_eos/checkpoint-0600",
    "--split","dev","--tier","core","--pool","playground","--per-rung","1",
    "--budget-bytes","160","--max-tokens","256","--batch-size","8","--device","cuda",
    "--out","research/raw/e122e_rebaseline_ab_256.json"
)
& $py @c2score 2>&1 | Tee-Object -FilePath scratch/e122e_rebaseline_ab_256.log

Write-Host "STAGES COMPLETE. Verify the JSON artifacts exist before reporting."
