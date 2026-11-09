"""V2 data lane: pinned sources, provenance, deterministic preparation.

Preparation dedupes before splitting, splits deterministically from pinned
identities plus a salt, and publishes generations atomically under
data/v2/prepared/<dataset_id>/<generation_id>.
"""
from slm_train.data.generation import PreparedGeneration, generation_key_for
from slm_train.data.prepare import PreparationError, prepare_generation

__all__ = [
    "PreparationError",
    "PreparedGeneration",
    "generation_key_for",
    "prepare_generation",
]
