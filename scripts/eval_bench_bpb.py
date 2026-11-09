"""E-119: bits-per-byte of Python benchmark text across model families.

EVAL ONLY - nothing under data/bench/ is ever trained on.

    BPB = sum(-log p(t_i | t_<i)) / ln2 / total_utf8_bytes

is the only tokenizer-agnostic scalar, so it is the one axis on which the byte-level
glm arm, the CodeLlama-BPE target and external HF baselines (SmolLM2, Qwen) compare.

Arms (repeatable):
    --arm name=hf:PATH        any transformers causal-LM directory (target, SmolLM2, ...)
    --arm name=glm-byte:DIR   glm53_flash byte arm (nested repo), vocab 260, 1 token = 1 byte

Sets come from data/bench/raw (manifest.json is the receipt); each benchmark row is
tokenized on its own.  A row longer than the arm's native context is split into
non-overlapping windows; the first token of every window has no context and is excluded
from the NLL sum (transformers' shifted loss already excludes it).  The denominator is
the row's UTF-8 byte count, so a byte arm and a BPE arm share one scale.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

LN2 = math.log(2)
ROOT = Path(__file__).resolve().parents[1]
NEST = ROOT / "glm-5.3-flash-from-scratch"
RAW = ROOT / "data" / "bench" / "raw"

SETS = ("humaneval", "mbpp-sanitized", "mbpp", "ds1000", "bigcodebench")


def _write(path: Path, report: dict) -> None:
    """Write the report incrementally - a crash must not lose completed rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def load_texts(name: str, limit: int | None = None) -> list[str]:
    """Raw code text per benchmark row (columns per data/bench/manifest.json)."""
    if name == "humaneval":
        rows = [json.loads(line) for line in gzip.open(RAW / "HumanEval.jsonl.gz", "rt", encoding="utf-8")]
        texts = [r["prompt"] + "\n" + r["canonical_solution"] for r in rows]
    elif name == "mbpp-sanitized":
        rows = json.loads((RAW / "sanitized-mbpp.json").read_text(encoding="utf-8"))
        texts = [r["prompt"] + "\n" + r["code"] for r in rows]
    elif name == "mbpp":
        texts = [r["text"] + "\n" + r["code"] for r in _jsonl(RAW / "mbpp.jsonl")]
    elif name == "ds1000":
        texts = [r["prompt"] + "\n" + r["reference_code"] for r in _jsonl(RAW / "ds1000.jsonl")]
    elif name == "bigcodebench":
        texts = [r["complete_prompt"] + "\n" + r["canonical_solution"]
                 for r in _parquet(RAW / "bigcodebench_v0.1.4.parquet")]
    else:
        raise SystemExit(f"unknown set {name!r}; known: {SETS}")
    return texts[:limit] if limit else texts


class HfArm:
    """Any transformers causal LM (target, SmolLM2, Qwen)."""

    def __init__(self, name: str, path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.model.to("cpu").eval()
        self.ctx = int(getattr(self.model.config, "max_position_embeddings", 2048) or 2048)
        self.params = sum(p.numel() for p in self.model.parameters())
        self.bos = self.tok.bos_token_id

    def tokenize(self, text: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return [self.bos] + ids if self.bos is not None else ids

    def nll_sum(self, ids: list[int]) -> float:
        return _forward_nll(self.torch, self.model, ids)


class GlmByteArm:
    """glm53_flash byte arm: ByteTokenizer gives 1 token per UTF-8 byte."""

    def __init__(self, name: str, ckpt: str) -> None:
        import torch

        if str(NEST) not in sys.path:
            sys.path.insert(0, str(NEST))
        from glm53_flash import ByteTokenizer
        from glm53_flash.runtime import load_checkpoint

        self.name = name
        self.torch = torch
        self.tok = ByteTokenizer()
        self.model = load_checkpoint(Path(ckpt), torch.device("cpu"))
        self.model.eval()
        self.ctx = int(self.model.config.max_sequence_length)
        self.params = sum(p.numel() for p in self.model.parameters())

    def tokenize(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def nll_sum(self, ids: list[int]) -> float:
        return _forward_nll(self.torch, self.model, ids)


def _forward_nll(torch, model, ids: list[int]) -> float:
    """Sum of -log p over the len(ids)-1 predicted positions of one window."""
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        try:
            out = model(x, labels=x)
        except TypeError:
            try:
                out = model(x, targets=x)
            except TypeError:
                out = model(x)  # glm53_flash: forward(input_ids) -> (logits, usages)
    loss = logits = None
    if isinstance(out, dict):
        loss, logits = out.get("loss"), out.get("logits")
    elif isinstance(out, tuple):
        # (loss, ...) if the first element is a scalar, else (logits, ...): glm53_flash
        # returns (logits, usages), so a positional first element is NOT automatically a loss.
        first = out[0]
        if getattr(first, "ndim", 0) == 0:
            loss = first
        else:
            logits = first
    else:
        loss, logits = getattr(out, "loss", None), getattr(out, "logits", None)
        if loss is None and logits is None and hasattr(out, "ndim"):
            loss, logits = (out, None) if out.ndim == 0 else (None, out)
    if loss is None and logits is None:
        raise RuntimeError(f"no loss/logits in {type(out).__name__}")
    n_pred = len(ids) - 1
    if loss is not None:
        return float(loss) * n_pred
    import torch.nn.functional as F

    return float(F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                                 x[:, 1:].reshape(-1), reduction="sum"))


def evaluate(arm, texts: list[str]) -> dict:
    total_nll = 0.0
    total_bytes = 0
    windows = 0
    started = time.perf_counter()
    for index, text in enumerate(texts, start=1):
        ids = arm.tokenize(text)
        total_bytes += len(text.encode("utf-8"))
        for start in range(0, len(ids), arm.ctx):
            chunk = ids[start:start + arm.ctx]
            if len(chunk) < 2:
                continue
            total_nll += arm.nll_sum(chunk)
            windows += 1
        if index % 25 == 0 or index == len(texts):
            elapsed = time.perf_counter() - started
            print(f"  [{arm.name}] rows {index}/{len(texts)} bytes={total_bytes} "
                  f"sec={elapsed:.0f} bpb={total_nll / LN2 / total_bytes:.5f}", flush=True)
    return {
        "nll_nats": total_nll,
        "bytes": total_bytes,
        "windows": windows,
        "bpb": total_nll / LN2 / total_bytes if total_bytes else None,
        "seconds": round(time.perf_counter() - started, 2),
    }


def build_arm(spec: str):
    name, rest = spec.split("=", 1)
    kind, path = rest.split(":", 1)
    if kind == "hf":
        return HfArm(name, path)
    if kind == "glm-byte":
        return GlmByteArm(name, path)
    raise SystemExit(f"unknown arm kind {kind!r} (hf | glm-byte)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=hf:PATH | name=glm-byte:CKPT_DIR")
    ap.add_argument("--sets", default="humaneval", help=f"comma list from {SETS}")
    ap.add_argument("--limit", type=int, default=None, help="first N rows per set (smoke)")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (leave the GPU trainer headroom)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(max(1, args.threads))
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    report: dict = {"label": args.label, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metric": "bits per UTF-8 byte", "threads": args.threads,
                    "limit": args.limit, "arms": {}, "sets": {}}
    print(f"{'arm':<22} {'set':<15} {'bytes':>9} {'windows':>8} {'bpb':>8} {'sec':>7}", flush=True)
    out = Path(args.out) if args.out else ROOT / "research" / "raw" / f"e119_bpb_{args.label}.json"
    all_texts = {name: load_texts(name) for name in sets}
    report["sets"] = {name: {"rows_total": len(all_texts[name])} for name in sets}
    for spec in args.arm:
        arm = build_arm(spec)
        report["arms"][arm.name] = {"spec": spec, "params": arm.params, "ctx": arm.ctx}
        print(f"[arm] {arm.name} params={arm.params} ctx={arm.ctx} spec={spec}", flush=True)
        for set_name in sets:
            texts = all_texts[set_name][:args.limit] if args.limit else all_texts[set_name]
            res = evaluate(arm, texts)
            res["rows"] = len(texts)
            report["sets"][set_name][arm.name] = res
            print(f"{arm.name:<22} {set_name:<15} {res['bytes']:>9} {res['windows']:>8} "
                  f"{res['bpb']:>8.5f} {res['seconds']:>7.1f}", flush=True)
            _write(out, report)  # incremental
        del arm

    _write(out, report)
    print(f"[written] {out}", flush=True)


if __name__ == "__main__":
    main()
