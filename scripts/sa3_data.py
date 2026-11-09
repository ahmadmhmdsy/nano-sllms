"""SA-3 data: tokenize the verified SA-2 rows (data/sa2/{train,val}.jsonl) into datasets for scripts/sft.py.

Each row becomes prompt + completion + EOS; labels are -100 over the prompt, so only the completion
(and its EOS) is learned. Rows longer than --ctx are dropped, not truncated (a cut body would teach a
broken function). --shapes picks which SA-2 shapes go in: SA-3 is instruction -> code (he, mbpp);
fix and tool turns are SA-4 (research/2026-10-22_sa_plan_v2.md §1).

  .venv/Scripts/python.exe -I scripts/sa3_data.py
  .venv/Scripts/python.exe -I scripts/sa3_data.py --shapes fix,tool --out data/sa4/ds
"""
import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def encode(rows, tok, ctx):
    prompts = tok([r["prompt"] for r in rows], add_special_tokens=False)["input_ids"]
    completions = tok([r["completion"] for r in rows], add_special_tokens=False)["input_ids"]
    ids, labels, kept, dropped = [], [], collections.Counter(), collections.Counter()
    for row, p, c in zip(rows, prompts, completions):
        c = c + [tok.eos_token_id]
        if len(p) + len(c) > ctx:
            dropped[row["shape"]] += 1
            continue
        ids.append(p + c)
        labels.append([-100] * len(p) + c)
        kept[row["shape"]] += 1
    return ids, labels, kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Tokenize SA-2 rows for sft.py.")
    ap.add_argument("--src", default="data/sa2")
    ap.add_argument("--out", default="data/sa3/ds")
    ap.add_argument("--shapes", default="he,mbpp")
    ap.add_argument("--tokenizer", default="models/hf/SmolLM2-135M")
    ap.add_argument("--ctx", type=int, default=1024)
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer

    shapes = {s.strip() for s in args.shapes.split(",") if s.strip()}
    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer))
    if tok.eos_token_id is None:
        raise SystemExit("[sa3_data] tokenizer has no EOS token")
    out = ROOT / args.out
    manifest = {"src": args.src, "shapes": sorted(shapes), "ctx": args.ctx, "eos_id": tok.eos_token_id,
                "splits": {}}
    for split in ("train", "val"):
        path = ROOT / args.src / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [r for r in rows if r.get("shape") in shapes]
        if not rows:
            raise SystemExit(f"[sa3_data] no rows with shapes {sorted(shapes)} in {path}")
        ids, labels, kept, dropped = encode(rows, tok, args.ctx)
        Dataset.from_dict({"input_ids": ids, "labels": labels}).save_to_disk(str(out / split))
        lengths = sorted(len(x) for x in ids)
        target = sum(sum(1 for v in lab if v != -100) for lab in labels)
        manifest["splits"][split] = {"rows": len(ids), "kept": dict(kept), "dropped_over_ctx": dict(dropped),
                                     "tokens": sum(lengths), "target_tokens": target,
                                     "len_p50": lengths[len(lengths) // 2],
                                     "len_p99": lengths[int(len(lengths) * 0.99)]}
        print(json.dumps({"split": split, **manifest["splits"][split]}), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("DONE", out)


if __name__ == "__main__":
    main()
