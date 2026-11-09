"""Train event bus (plan docs/plans/2026-09-23-pipeline-feature-registry-plan.md)."""
CALLBACKS = {k: [] for k in (
    "on_start", "on_step", "on_eval", "on_checkpoint", "on_end", "on_anomaly",
)}


def emit(event, **payload):
    for cb in CALLBACKS.get(event, []):
        try:
            cb(**payload)
        except Exception as e:  # monitoring must never kill a run
            print("[PIPELINE-CALLBACK-FAIL]", event, repr(e), flush=True)


def subscribe(event, cb):
    CALLBACKS[event].append(cb)
