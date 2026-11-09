"""runtime-profile registry package (see store.py / registry.json).
Quick use inside any script:
    from pipeline.profiles import ProfileStore
    bs = ProfileStore().batch_for("gate_bench", cfg)
"""
from .store import ProfileStore, model_bucket, REGISTRY  # noqa: F401
