# Acceptance Criteria

This document defines the acceptance criteria for the nano_SLMs pipeline.

## Core Pipeline (S/P/T Ladder)

### Smoke (S)
- Model builds and runs forward/backward on GPU
- Loss is finite and decreases
- Checkpoint is published

### Pilot (P)
- All 13 R1 stages complete
- Training loss is finite and decreases
- Held-out validation loss is recorded
- Checkpoint retention follows policy

### Target (T)
- All 13 R1 stages complete
- Model meets declared evaluation thresholds
- External gates pass
- Final model is published

## Evidence Requirements

- Every claim must cite an artifact (file path)
- Every metric must be measured, never estimated
- Every gate must have a pass/fail criterion
- Unknown values must be labeled UNKNOWN

## Validation Labels

- PASS: The criterion is met
- FAIL: The criterion is not met
- SKIPPED: The criterion was not evaluated
- BLOCKED: The criterion cannot be evaluated due to a dependency
- NEEDS_USER_DECISION: The criterion requires user input
