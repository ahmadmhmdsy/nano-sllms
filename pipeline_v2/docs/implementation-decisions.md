schema_version: 1
oom:
  same_run: BLOCKED
  last_valid_checkpoint: preserve
  altered_batch_context_optimizer_precision_command: forbidden
  reduced_attempt: new_user_approved_versioned_launch_profile_and_new_run
  status: PASS
preflight:
  v2_command: pipeline_v2/run.py preflight --gpu
  laya_import: forbidden
  external_laya_gate: forbidden
  status: PASS

environment:
  root: .venv
  nested_venv: pipeline_v2/.venv_forbidden
  one_time_shared_venv_drift: user_authorized_and_accepted
  freeze_further_sync_or_environment_changes: pending_new_user_decision
  evidence: .superpowers/sdd/pipeline-v2-task-1-report.md
dependency_command:
  lock: uv lock --project pipeline_v2
  install: uv sync --project pipeline_v2 --active --inexact --locked --extra test
runtime_roots:
  prepared: data/v2/prepared
  tokens: data/v2/tokens
  attempts: runs/v2/_attempts
  final_runs: runs/v2/<namespace>/<run_id>
  cpu_test_artifacts: pipeline_v2/test-artifacts
retention:
  checkpoint_and_log_deletion: manual_under_data_policy
