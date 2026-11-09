"""Feature registry (plan S5): experimental vs validated features.

- experimental: applied ONLY when the config lists the name in
  pipeline.features (one-use experiments never leak into other runs).
- validated: auto-applied to every train run; explicit opt-out only.
Promotion path: experimental -> closed experiment verdict + user confirmation
-> move to VALIDATED (this file is the audit trail).
"""
EXPERIMENTAL = []
AT_BIRTH = [
    "warm_init_loader", "scaler_skip_on_overflow", "autosave_rotation",
    "auto_resume_zero_flag", "monitor", "smoke_preflight",
    "internal_fullval_eval", "external_gates_profile", "early_stop",
]
VALIDATED = list(AT_BIRTH)

REGISTRY = {name: "validated" for name in VALIDATED}
REGISTRY.update({name: "experimental" for name in EXPERIMENTAL})


def features_for_config(cfg):
    want = (cfg.get("pipeline") or {}).get("features", []) or []
    for name in want:
        if name not in REGISTRY:
            raise ValueError("pipeline feature not registered: " + name)
        if REGISTRY[name] != "experimental":
            raise ValueError(f"{name} is validated: it is already default; "
                             "remove it from pipeline.features")
    return {"validated": list(VALIDATED), "opt_in": list(want)}
