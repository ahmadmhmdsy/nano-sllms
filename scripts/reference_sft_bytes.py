"""Pack the playground reference solutions into ByteTokenizer id shards (E-121 arm A).

Every task carries its ground-truth body (Family.body / Aspect.body). This builder renders those
bodies as prompt + reference_completion - the exact string the evaluator scores - for many task
instances drawn from the held-out 'pretrain' split (seed 9319, disjoint from dev/final/confirm/rl).
That is the one training stage that does not require pass@k > 0, and it is what shows the model the
forms it never emits: the ternary in absolute/nonnegative, % in even, max in clamp, comprehensions
in count_even, statement-form branching in grade.

Phase 1.1 (E-122): with --include-playground the per-part playground ladders
(glm53_flash.playground, held-out split 'pretrain', seed 8951) are added as extra sources, one
sub-skill at a time. That changes the corpus, so it is gated OFF by default and must be written to a
NEW --out-dir; the existing data/reference/bytes E-122 corpus is never touched.

With --append-eos (default off) every rendered example is terminated with
glm53_flash.tokenizer.ByteTokenizer.eos_id, so the model can learn to stop; without it the packed
corpus carries no end-of-sequence token at all. The flag changes the bytes and meta.json, so write
to a new --out-dir.

Train instances use indices [0, --instances); val uses the disjoint tail so the diagnostic is honest.
CPU only. Regenerable: safe to delete.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEST = ROOT / "glm-5.3-flash-from-scratch"
sys.path.insert(0, str(NEST))
BYTE_OFFSET = 4
PRETRAIN_SEED = 9319  # glm53_flash.aspects.ASPECT_SEEDS["pretrain"]
DEFAULT_OUT = ROOT / "data" / "reference" / "bytes"
PLAYGROUND_TIERS = ("core", "loops", "attributes", "all")


def tokenizer_eos_id() -> int:
    """The end-of-sequence id, read from ByteTokenizer (never hardcoded)."""
    from glm53_flash.tokenizer import ByteTokenizer
    return ByteTokenizer.eos_id


def pack_texts(texts: list[str], append_eos: bool = False,
               eos_id: int | None = None) -> tuple[bytes, int]:
    """Encode texts as a uint16 shard. With append_eos, every example ends with EOS.

    The no-flag branch is byte-for-byte the original packer (one joined UTF-8 blob);
    the flag branch encodes each example separately and terminates it.
    """
    if not append_eos:
        joined = "".join(texts)
        values = np.frombuffer(joined.encode("utf-8"), dtype=np.uint8).astype(np.uint16) + BYTE_OFFSET
        return values.tobytes(), int(values.size)
    if eos_id is None:
        eos_id = tokenizer_eos_id()
    eos = np.asarray([eos_id], dtype=np.uint16)
    pieces = []
    for text in texts:
        values = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16) + BYTE_OFFSET
        pieces.append(np.concatenate([values, eos]))
    values = np.concatenate(pieces) if pieces else np.asarray([], dtype=np.uint16)
    return values.tobytes(), int(values.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)  # legacy alias
    parser.add_argument("--instances", type=int, default=400, help="task instances per source")
    parser.add_argument("--val-instances", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-playground", action="store_true",
                        help="add the per-part playground ladder sources (held-out 'pretrain' split)")
    parser.add_argument("--playground-tier", choices=PLAYGROUND_TIERS, default="all")
    parser.add_argument("--playground-instances", type=int, default=None,
                        help="task instances per playground rung; defaults to --instances. "
                             "Set lower than --instances for a REPLAY-DOMINANT mixture (E-122d showed "
                             "a playground-dominated corpus catastrophically forgets)")
    parser.add_argument("--append-eos", action="store_true",
                        help="terminate every rendered example with ByteTokenizer.eos_id (default: off)")
    args = parser.parse_args()

    out_dir = args.out if args.out is not None else args.out_dir
    eos_id = tokenizer_eos_id() if args.append_eos else None
    if args.include_playground and out_dir.resolve() == DEFAULT_OUT.resolve():
        raise SystemExit(
            "refusing to write the playground corpus into %s (that is the E-122 corpus); "
            "pass --out-dir data/reference/bytes_playground" % DEFAULT_OUT)

    from glm53_flash.aspects import ASPECTS, make_aspect_task  # noqa: E402
    from glm53_flash.tasks import FAMILIES, make_task  # noqa: E402

    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit("refusing to overwrite %s (pass --force)" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def legacy(index: int) -> list[str]:
        return [make_task(f, split="pretrain", index=index, seed=PRETRAIN_SEED).reference_source
                for f in FAMILIES]

    def aspects(index: int) -> list[str]:
        return [make_aspect_task(a, split="pretrain", index=index).reference_source for a in ASPECTS]

    pool = []
    make_playground_task = None
    if args.include_playground:
        from glm53_flash.playground import make_playground_task, playground_pool  # noqa: E402
        pool = playground_pool(args.playground_tier)

    def playground(index: int) -> list[str]:
        return [make_playground_task(s, split="pretrain", index=index).reference_source for s in pool]

    pg_instances = args.instances if args.playground_instances is None else args.playground_instances
    if args.include_playground and not 0 <= pg_instances <= args.instances:
        raise SystemExit("--playground-instances must be within [0, --instances] so the val tail stays disjoint")
    pg_val = max(1, round(args.val_instances * pg_instances / args.instances)) if args.instances else 0

    train_texts: list[str] = []
    for index in range(args.instances):
        train_texts.extend(legacy(index))
        train_texts.extend(aspects(index))
    if args.include_playground:
        for index in range(pg_instances):
            train_texts.extend(playground(index))
    val_texts: list[str] = []
    for offset in range(args.val_instances):
        index = args.instances + offset
        val_texts.extend(legacy(index))
        val_texts.extend(aspects(index))
    if args.include_playground:
        for offset in range(pg_val):
            val_texts.extend(playground(args.instances + offset))

    rng = random.Random(PRETRAIN_SEED)
    rng.shuffle(train_texts)  # deterministic: no family forms a long run

    train_blob, train_bytes = pack_texts(train_texts, args.append_eos, eos_id)
    val_blob, val_bytes = pack_texts(val_texts, args.append_eos, eos_id)
    (out_dir / "train.bin").write_bytes(train_blob)
    (out_dir / "val.bin").write_bytes(val_blob)

    sources = [f.name for f in FAMILIES] + [a.name for a in ASPECTS]
    if args.include_playground:
        sources += [s.key for s in pool]
    meta = {
        "shape": "reference-solutions", "byte_offset": BYTE_OFFSET, "split": "pretrain",
        "seed": PRETRAIN_SEED, "sources": sources, "instances_per_source": args.instances,
        "append_eos": bool(args.append_eos), "eos_id": eos_id,
        "train": {"bytes": train_bytes, "examples": len(train_texts), "path": "train.bin"},
        "val": {"bytes": val_bytes, "examples": len(val_texts), "path": "val.bin"},
        "epochs_at_5000_steps": {"byte_arm_positions": 5000 * 32 * 1024,
                                 "epochs": (5000 * 32 * 1024) / max(train_bytes, 1)},
    }
    if args.include_playground:
        meta["playground"] = {
            "tier": args.playground_tier,
            "sub_skills": len(pool),
            "train_examples": pg_instances * len(pool),
            "val_examples": pg_val * len(pool),
        }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"event": "done", **meta}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
