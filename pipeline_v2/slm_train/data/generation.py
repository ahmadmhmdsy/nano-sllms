"""Generation identity: a changed pinned identity is a NEW generation.

A generation identity is derived from the dataset id and the recorded
preparation semantics; a changed source content or changed pinned identity
necessarily yields a new generation id (never an implicit refresh).
"""
from slm_train.contracts.hash import canonical_sha256


def generation_key_for(dataset_id: str, semantic_hash: str) -> str:
    """Stable generation key derived from dataset id and the semantic hash."""
    value = canonical_sha256({
        "dataset_id": str(dataset_id),
        "semantic_hash": str(semantic_hash),
    })
    return f"g-{value[:16]}"


class PreparedGeneration:
    """Result object returned by the deterministic preparation lane.

    It is intentionally not a pydantic contract: the immutable contract is
    the published manifest; this object only carries the published facts.
    """

    def __init__(
        self,
        semantic_hash: str,
        counts: dict,
        published_path,
        generation_key: str,
        split_assignments: dict,
        instance_label: str = "PASS",
        provenance: object = None,
    ) -> None:
        self.semantic_hash = semantic_hash
        self.counts = dict(counts)
        self.published_path = published_path
        self.generation_key = generation_key
        self._split_assignments = dict(split_assignments)
        self.label = instance_label
        self.provenance = dict(provenance) if provenance else {}

    @property
    def split_assignments(self) -> dict:
        return dict(self._split_assignments)

    @property
    def doc_split_counts(self) -> dict:
        return dict(self._count_by_split())

    def _count_by_split(self) -> dict:
        counts: dict = {}
        for split in self._split_assignments.values():
            counts[split] = counts.get(split, 0) + 1
        return counts
