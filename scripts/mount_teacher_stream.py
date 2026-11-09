"""Build a SmolLM2-tokenized shard stream aligned PER-BLOCK to the student
stream (mounting design section 7). For each student uint32 block we decode
the exact text span with the student tokenizer, re-encode with the mounted
teacher's tokenizer, cap at ctx tokens, right-pad with id 0; a parallel
.len.bin (int16, one value per block) stores true lengths so cross-attention
masks pads out (pads are NEVER attended). Boundary-span docs decode across
blocks by construction (block b continues block b-1's tail); off-by-one
boundary tokens are accepted noise, documented in the spec.
CPU-only; co-run-safe beside a live GPU training run."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-tokens-dir", default="data/pilot/tokens")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-tok", default="codellama/CodeLlama-7b-hf")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    student_tok = AutoTokenizer.from_pretrained(args.student_tok)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model)

    shards = sorted(Path(args.student_tokens_dir).glob(f"{args.split}_*.bin"))
    assert shards, "no " + args.split + " shards"
    n_blocks = 0
    for shard in shards:
        ids = np.fromfile(shard, dtype=np.uint32)
        ids = ids[: (len(ids) // args.ctx) * args.ctx]
        blocks = ids.reshape(-1, args.ctx)
        lens, tblocks = [], []
        for b in blocks:
            text = student_tok.decode(b.astype(int).tolist())
            t = teacher_tok(text, add_special_tokens=False)["input_ids"][: args.ctx]
            lens.append(len(t) if t else 1)
            tblocks.append(t + [0] * (args.ctx - len(t)))
        np.stack(tblocks).astype(np.uint32).tofile(out / shard.name)
        np.asarray(lens, dtype=np.int16).tofile(out / (shard.name[:-4] + ".len.bin"))
        n_blocks += len(tblocks)
    (out / "meta.json").write_text(json.dumps(
        {"teacher": args.teacher_model, "ctx": args.ctx, "blocks": n_blocks,
         "split": args.split, "pad_id": 0}, indent=2), encoding="utf-8")
    print("[teacher-stream] wrote", n_blocks, "blocks to", out, flush=True)

if __name__ == "__main__":
    main()
