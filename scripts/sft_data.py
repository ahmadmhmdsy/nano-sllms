"""C12 Tier 1 data prep: teacher traces -> tokenized SFT dataset.

Source = data.dataset: an HF dataset name (streamed; default
nickrosh/Evol-Instruct-Code-80k-v1, strong-model-generated instruction/
response pairs - research/c12_distillation_report.md) OR a local
.jsonl/.json pairs file (repo-root path; custom-dataset support), filters
(min_chars, dedupe, optional ast.parse quality pre-filter), splits 98/2, renders
the fixed plain template (CodeLlama ships no chat template; plan §3.1),
tokenizes with PROMPT MASKING (labels = -100 through the "### Response:" line;
the model trains only on response + eos), drops rows longer than the SFT ctx,
and saves the HF dataset to disk plus the held-out instruction list for the
generation eval.

Run: .venv/Scripts/python scripts/sft_data.py --config configs/sft_t1.yaml
Safe while M3 trains: CPU + network only; touches nothing under runs/.
"""
import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESPONSE_KEYS = ("output", "response", "answer", "completion")
INSTRUCTION_KEYS = ("instruction", "prompt", "question", "input")
FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def render_prompt(template: str, instruction: str) -> str:
    return template.format(instruction=instruction)


def extract_code(response: str) -> str:
    """Best-effort Python extraction: longest fenced block if fenced, else raw."""
    blocks = FENCE_RE.findall(response)
    if blocks:
        return max(blocks, key=len)
    return response


def parses_as_python(text: str) -> bool:
    try:
        # responses often contain regex-ish strings ("\d", "\s") that emit
        # SyntaxWarnings while compiling; they are noise, not filter signals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(extract_code(text))
        return True
    except SyntaxError:
        return False


def encode_pair(tok, template: str, instruction: str, response: str,
                ctx: int, eos_id: int):
    """Tokenize one (instruction, response) pair with prompt masking.

    Returns (input_ids, labels) or None if the row exceeds ctx. The mask
    boundary uses offset ENDS, not prefix equality: BPE can merge the template's
    trailing newline with the response's first characters, and a token that
    straddles the boundary is kept trained so the response is fully covered.
    """
    prompt_text = render_prompt(template, instruction)
    full_text = prompt_text + response
    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) + 1 > ctx:  # +1: the trained eos
        return None
    labels = [t if off[1] > len(prompt_text) else -100
              for t, off in zip(ids, enc["offset_mapping"])]
    return ids + [eos_id], labels + [eos_id]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    import yaml
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg, d, t = cfg["tokenizer"], cfg["data"], cfg["train"]
    sft = cfg["sft"]
    ctx = int(sft["ctx"])
    template = d["template"]

    raw_dir = ROOT / d["raw_dir"]
    out_dir = ROOT / d["dataset_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(tcfg["name"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token  # plan §2 prereq; recorded in the config
    eos_id = tok.eos_token_id
    print(f"[sft_data] tokenizer={tcfg['name']} vocab={len(tok)} "
          f"pad_id={tok.pad_token_id} eos_id={eos_id} ctx={ctx}", flush=True)

    n_rows = int(d["rows"])
    n_val = max(1, round(n_rows * float(d.get("val_fraction", 0.02))))
    n_target = n_rows + n_val
    min_chars = int(d.get("min_chars", 200))
    min_instr = int(d.get("min_instruction_chars", 20))
    dedupe = bool(d.get("dedupe", True))
    ast_filter = bool(d.get("ast_filter", True))

    ds_name = d["dataset"]
    if ds_name.lower().endswith((".jsonl", ".json")):
        # local pairs file (custom-dataset path): .jsonl with one JSON object
        # per line, or .json holding a list / {"rows": [...]}
        loc = Path(ds_name)
        if not loc.is_absolute():
            loc = ROOT / loc

        def local_rows():
            if loc.suffix.lower() == ".json":
                data = json.loads(loc.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data = data.get("rows", [data])
                for row in data:
                    yield row
            else:
                with loc.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)

        source = local_rows()
        ds_name = f"local:{loc.name}"
    else:
        source = load_dataset(ds_name, split="train", streaming=True)
    print(f"[sft_data] source={ds_name} (target {n_target} accepted pairs, "
          f"ast_filter={ast_filter})", flush=True)
    stats = {"seen": 0, "kept": 0, "drop_short": 0, "drop_instr": 0,
             "drop_dup": 0, "drop_ast": 0, "drop_long": 0}
    rows = []
    seen = set()
    for ex in source:
        stats["seen"] += 1
        instruction = ""
        for key in INSTRUCTION_KEYS:
            v = ex.get(key)
            if isinstance(v, str) and v.strip():
                instruction = v.strip()
                break
        response = ""
        for key in RESPONSE_KEYS:
            v = ex.get(key)
            if isinstance(v, str) and v.strip():
                response = v.strip()
                break
        if not instruction or not response or len(response) < min_chars:
            stats["drop_short"] += 1
            continue
        if len(instruction) < min_instr:
            stats["drop_instr"] += 1
            continue
        if dedupe:
            h = hashlib.sha1((instruction + "\x00" + response).encode("utf-8")).hexdigest()
            if h in seen:
                stats["drop_dup"] += 1
                continue
            seen.add(h)
        if ast_filter and not parses_as_python(response):
            stats["drop_ast"] += 1
            continue
        enc = encode_pair(tok, template, instruction, response, ctx, eos_id)
        if enc is None:
            stats["drop_long"] += 1
            continue
        ids, labels = enc
        rows.append({"input_ids": ids, "labels": labels, "instruction": instruction})
        stats["kept"] += 1
        if stats["kept"] % 500 == 0:
            print(f"[sft_data] {stats['kept']}/{n_target} pairs, stats={stats}",
                  flush=True)
        if stats["kept"] >= n_target:
            break

    if stats["kept"] == 0:
        raise SystemExit("[sft_data] no rows survived filtering")
    if stats["kept"] < n_target:
        print(f"[sft_data] WARNING: stream ended early with {stats['kept']}/{n_target} pairs",
              flush=True)

    rng = random.Random(int(t.get("seed", 42)))
    rng.shuffle(rows)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    n_train = len(train_rows)

    train_ds = Dataset.from_list(
        [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in train_rows])
    val_ds = Dataset.from_list(
        [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in val_rows])
    train_ds.save_to_disk(str(out_dir / "train"))
    val_ds.save_to_disk(str(out_dir / "val"))

    # held-out instructions for the generation eval (eval.py reads this file)
    with (raw_dir / "val_instructions.jsonl").open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps({"instruction": r["instruction"]},
                               ensure_ascii=False) + "\n")

    (raw_dir / "source.txt").write_text(ds_name, encoding="utf-8")
    meta = {
        "dataset": ds_name,
        "tokenizer": tcfg["name"],
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": eos_id,
        "ctx": ctx,
        "template": template,
        "ast_filter": ast_filter,
        "min_chars": min_chars,
        "seen": stats["seen"],
        "kept": stats["kept"],
        "train_pairs": n_train,
        "val_pairs": len(val_rows),
        "stats": stats,
    }
    (raw_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[sft_data] meta: " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
