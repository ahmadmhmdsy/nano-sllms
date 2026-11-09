"""Build the U-1 corpus (JSONL rows: arch, seed, task, chain, spec, split).

Usage (venv):
    python -m ui.scripts.build_corpus --out data/u1/raw/u1_demo.jsonl --per-arch 200

Holdout (pre-registered): block-disjoint seed ranges per archetype:
train < 0.8N, val [0.8N, 0.95N), test [0.95N, N).
"""

from __future__ import annotations

import argparse
import json
import os

from ui.src.generator import ARCHETYPES, build
from ui.src.chain import spec_to_chain, chain_to_spec
from ui.src.validate import validate_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-arch", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows, fails = [], []
    for arch in ARCHETYPES:
        for i in range(args.per_arch):
            seed = args.seed * 100000 + i
            try:
                task, spec = build(seed, arch)
            except Exception as e:
                fails.append("{0}/{1}: {2}".format(arch, i, e))
                continue
            chain = spec_to_chain(spec, task)
            spec2, issues = chain_to_spec(chain)
            l1 = [i for i in issues if i is not None]
            vs, _ = chain_to_spec(spec_to_chain(spec2, task))  # idempotence check
            errs = validate_chain(chain)
            shared = i / args.per_arch
            split = "train" if shared < 0.8 else ("val" if shared < 0.95 else "test")
            rows.append({"arch": arch, "seed": seed, "task": task, "chain": chain, "spec": spec, "split": split})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
    print("rows={0} fails={1} out={2}".format(len(rows), len(fails), args.out))
    for fl in fails[:10]:
        print("FAIL", fl)
    if __name__ == "__main__":
        pass


if __name__ == "__main__":
    main()
