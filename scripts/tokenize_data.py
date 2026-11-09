"""Tokenize raw JSONL into packed fixed-length uint32 .bin shards.

Run: .venv/Scripts/python scripts/tokenize_data.py --config configs/smoke.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    import yaml
    from transformers import AutoTokenizer

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg, d = cfg["tokenizer"], cfg["data"]
    seq_len = int(cfg["model"]["ctx"])
    shard_tokens = int(d.get("shard_tokens", 8_000_000))
    shuffle = bool(d.get("shuffle_before_pack", False))

    raw_dir = ROOT / d["raw_dir"]
    if not (raw_dir / "source.txt").exists():
        raise SystemExit(f"raw data not prepared yet (missing {raw_dir / 'source.txt'}); "
                         f"run scripts/prepare_data.py first")
    tok_dir = ROOT / d["tokens_dir"]
    tok_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(tcfg["name"])
    print(f"[tokenize] tokenizer={tcfg['name']} vocab={len(tok)}", flush=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id

    def encode(src: Path, prefix: str) -> int:
        state = {"shard": 0, "buf": [], "total": 0}

        def flush():
            if not state["buf"]:
                return
            arr = np.asarray(state["buf"], dtype=np.uint32)
            arr.tofile(str(tok_dir / f"{prefix}_{state['shard']:03d}.bin"))
            state["total"] += len(state["buf"])
            print(f"[tokenize] {prefix}_{state['shard']:03d}.bin: "
                  f"{len(state['buf']):,} tokens", flush=True)
            state["shard"] += 1
            state["buf"] = []

        with src.open(encoding="utf-8") as f:
            from tqdm import tqdm
            for line in tqdm(f, desc=f"[tokenize] {prefix}"):
                text = json.loads(line)["text"]
                ids = tok(text, add_special_tokens=False)["input_ids"]
                state["buf"].append(bos)
                state["buf"].extend(ids)
                state["buf"].append(eos)
                if len(state["buf"]) >= shard_tokens:
                    flush()
        if shuffle:
            import random
            random.seed(int(d.get("seed", 42)))
            random.shuffle(state["buf"])
        flush()
        return state["total"]

    n_train = encode(raw_dir / "train.jsonl", "train")
    n_val = encode(raw_dir / "val.jsonl", "val")
    meta = {
        "tokenizer": tcfg["name"],
        "tokenizer_vocab": len(tok),
        "model_vocab": max(int(tcfg["vocab_size"]), len(tok)),
        "seq_len": seq_len,
        "shard_tokens": shard_tokens,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "train_blocks": n_train // seq_len,
        "val_blocks": n_val // seq_len,
    }
    (tok_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
