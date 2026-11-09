"""Read-only V1 configuration migration.

V1 is frozen. This module reads a V1 config that a user named explicitly,
records its hash before and after, and writes one new V2 file. It does not
import V1 code, call V1 lifecycle code, touch the GPU, or start training. The
before-and-after hashes are not decoration: they are the evidence that the
migration left its input byte-identical, and a mismatch refuses the migration.

Migration is translation, not adoption. A migrated config describes an intent;
it carries no evidence, so it cannot by itself make a later run eligible to
promote a predecessor.

The translation parses the V1 file with PyYAML (the same loader family the V2
loader uses), maps only fields with a real V2 meaning, and reports everything
else as not_migrated. The emitted file carries schema_version: 2 and only keys
the V2 schema knows, so a migrated file can be validated instead of silently
rejected.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from slm_train.contracts.hash import sha256_bytes

__all__ = [
    "MigrationError",
    "MigrationResult",
    "migrate_v1_config",
    "render_v2_config",
]


class MigrationError(RuntimeError):
    """The migration refused. The V1 input is untouched either way."""


class MigrationResult:
    """What was written, and the evidence that the input did not change."""

    __slots__ = ("source", "target", "source_sha256_before",
                 "source_sha256_after", "migrated_fields", "notes")

    def __init__(self, source, target, before, after, migrated_fields, notes):
        self.source = source
        self.target = target
        self.source_sha256_before = before
        self.source_sha256_after = after
        self.migrated_fields = tuple(migrated_fields)
        self.notes = tuple(notes)


def _read_mapping(path) -> dict:
    """Parse the V1 file as YAML data. The file is data, never instructions.

    The V1 code is never imported or executed; a V1 document that requires V1
    code to be understood is honestly reported as unmigrated instead. Nested
    blocks are flattened to dotted paths (train.lr) so a nested V1 key can map
    to a real V2 field rather than dragging its whole section.
    """
    text = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise MigrationError(
            "a V1 config is a mapping of keys to values, not a "
            + type(parsed).__name__)
    flat: dict = {}

    def flatten(prefix, value):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                child = str(child_key)
                flatten(child if not prefix else prefix + "." + child, child_value)
        else:
            flat[prefix] = value

    flatten("", parsed)
    return flat


def _int(value) -> int:
    if isinstance(value, bool):
        raise MigrationError("an integer field received a boolean")
    return int(value)


def _float(value) -> float:
    if isinstance(value, bool):
        raise MigrationError("a float field received a boolean")
    return float(value)


def _optimizer(value) -> str:
    text = str(value).lower().replace("-", "").replace("_", "")
    if text in ("adamw", "adamwtorch"):
        return "adamw"
    raise MigrationError("the V1 optimizer has no R1 equivalence: " + str(value))


def _dedupe(value) -> str:
    return "sha-normalized-first" if bool(value) else "none"


#: Every mapped field with its exact V2 meaning. Each entry returns a
#: list of (target_key, value) pairs so one V1 field can map to several V2
#: fields (e.g. model.ctx must set model.context, tokenizer.context, and
#: packing.sequence_length to satisfy the cross-field context rule).
#: Anything outside this map is reported as unmigrated rather than guessed at.
FIELD_MAP = {
    "name": lambda doc: [("run.name", str(doc["name"]))],
    "seed": lambda doc: [("run.seed", _int(doc["seed"]))],
    "train.lr": lambda doc: [("train.learning_rate", _float(doc["train.lr"]))],
    "train.batch": lambda doc: [("train.batch_size", _int(doc["train.batch"]))],
    "train.accum": lambda doc: [("train.gradient_accumulation", _int(doc["train.accum"]))],
    "train.max_steps": lambda doc: [("train.max_steps", _int(doc["train.max_steps"]))],
    "train.max_grad_norm": lambda doc: [("train.max_grad_norm", _float(doc["train.max_grad_norm"]))],
    "train.scheduler": lambda doc: [("train.scheduler", str(doc["train.scheduler"]))],
    "train.warmup_steps": lambda doc: [("train.warmup_steps", _int(doc["train.warmup_steps"]))],
    "train.weight_decay": lambda doc: [("train.weight_decay", _float(doc["train.weight_decay"]))],
    "train.optim": lambda doc: [("train.optimizer", _optimizer(doc["train.optim"]))],
    "train.seed": lambda doc: [("run.seed", _int(doc["train.seed"]))],
    "model.layers": lambda doc: [("model.layers", _int(doc["model.layers"]))],
    "model.hidden": lambda doc: [("model.hidden", _int(doc["model.hidden"]))],
    "model.heads": lambda doc: [("model.heads", _int(doc["model.heads"]))],
    "model.kv_heads": lambda doc: [("model.kv_heads", _int(doc["model.kv_heads"]))],
    "model.ffn": lambda doc: [("model.intermediate", _int(doc["model.ffn"]))],
    # V2 requires model.context == tokenizer.context == packing.sequence_length,
    # so a single V1 ctx value must set all three.
    "model.ctx": lambda doc: [
        ("model.context", _int(doc["model.ctx"])),
        ("tokenizer.context", _int(doc["model.ctx"])),
        ("packing.sequence_length", _int(doc["model.ctx"])),
    ],
    "model.tie_embeddings": lambda doc: [("model.tied_embeddings", bool(doc["model.tie_embeddings"]))],
    # data.min_chars is deliberately NOT mapped to data.filters: the V2
    # inheritance merge rejects list overrides (ambiguous list merge), so
    # emitting filters would produce a config that fails schema validation.
    # The field is reported as not_migrated instead.
    "data.val_fraction": lambda doc: [("data.validation_fraction", _float(doc["data.val_fraction"]))],
    "data.dedupe": lambda doc: [("data.dedupe", _dedupe(doc["data.dedupe"]))],
}

_MAPPED_KEYS = frozenset(FIELD_MAP)

#: Checkpoint and hub references are never carried across: adoption is refused
#: by design, not by accident.
UNMAPPED_KEYS = (
    "pretrained_checkpoint", "resume_from", "huggingface_repo",
)


def _nest_dotted(mapping) -> dict:
    """Fold dotted V2 paths (run.name) into a nested override document."""
    nested: dict = {}
    for dotted, value in mapping.items():
        keys = [part for part in str(dotted).split(".") if part]
        cursor = nested
        for key in keys[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                child = {}
                cursor[key] = child
            cursor = child
        cursor[keys[-1]] = value
    return nested


def render_v2_config(migrated, unmigrated, source_name) -> str:
    """A V2 override document that says what it came from and what was left.

    schema_version and the R1 discriminators are always present, and only keys
    the V2 schema knows are emitted, so a migrated file can be loaded and
    validated instead of being structurally un-validatable.
    """
    document = {
        "extends": ["../base/r1.yaml"],
        "release": "r1",
        "capabilities": ["pretrain"],
        "schema_version": 2,
    }
    document.update(_nest_dotted(migrated))
    header = (
        "# Generated by migrate-v1. This is a translation, not an adoption.\n"
        "# It carries no evidence and makes no run promotion eligible.\n"
    )
    footer = ""
    if unmigrated:
        footer = "\n# NOT migrated, and deliberately so:\n"
        for key in unmigrated:
            footer += "#   " + str(key) + "\n"
    return header + yaml.safe_dump(document, sort_keys=True) + footer


def migrate_v1_config(source, target):
    """Write a new V2 config from a V1 config, changing nothing in the source."""
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_file():
        raise MigrationError("V1 config does not exist: " + str(source_path))
    if not isinstance(target, (str, Path)) or str(target).strip() == "":
        raise MigrationError("migration needs an explicit output path")
    if source_path.resolve() == target_path.resolve():
        raise MigrationError("the V2 output may not be the V1 input")
    if target_path.exists():
        raise MigrationError("refusing to overwrite an existing output: " + str(target_path))

    before = sha256_bytes(source_path.read_bytes())
    document = _read_mapping(source_path)

    migrated: dict = {}
    fields = []
    for legacy, follow in FIELD_MAP.items():
        if legacy in document:
            try:
                mappings = follow(document)
            except (MigrationError, TypeError, ValueError) as error:
                raise MigrationError(
                    "the V1 field " + legacy + " has no mapping: " + str(error)
                ) from None
            for target_key, value in mappings:
                migrated[target_key] = value
                fields.append(legacy + " -> " + target_key)
    if "run.name" not in migrated:
        raise MigrationError(
            "a V1 config requires a name; there is nothing to migrate in "
            + str(source_path))

    noted = list(key for key in UNMAPPED_KEYS if key in document)
    for key in document:
        if key not in _MAPPED_KEYS and key not in UNMAPPED_KEYS:
            noted.append(key)

    rendered = render_v2_config(migrated, noted, source_path.name)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Create-if-absent: the refusal above is a race guard, not the mechanism.
    with open(target_path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)

    after = sha256_bytes(source_path.read_bytes())
    if after != before:
        raise MigrationError(
            "the V1 input changed during migration: " + str(source_path))
    return MigrationResult(source_path, target_path, before, after, fields, noted)
