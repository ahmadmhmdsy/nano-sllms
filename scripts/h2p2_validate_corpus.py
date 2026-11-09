"""H2 Phase 2 corpus validator (general-QA slice + mix, TASKS row 37 phase 2).

Validates the downloaded QA slices in data/sft/h2p2_mixed/raw/ (built by
scripts/h2p2_build_corpus.py), then mixes them with the Phase 1 copy pairs.
General-QA rules - NO copy-fidelity check here (QA pairs deliberately teach
knowledge answers, not copying):

  - schema: {"instruction": str, "response": str}
  - instruction 20-400 chars
  - response 20-700 chars, 1-5 sentences, <= 130 words
  - no code markers / AI-meta boilerplate in the response
  - dedupe by sha1(instruction + \0 + response) inside QA and against
    the Phase 1 copy pairs (disjoint corpora by construction)
  - mix: copy pairs appended VERBATIM (they were validated by
    scripts/h2_validate_corpus.py in Phase 1)

FAIL if fewer than 200 valid QA pairs (mix would be dominated by copy pairs).

Run (CPU, co-run safe):
  & .\.venv\Scripts\python.exe scripts\h2p2_validate_corpus.py
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FENCE = chr(96) * 3
CODE_MARKERS = ("import ", "def ", "class ", ">>>", FENCE, "print(")
META_BOILER = ("as an ai", "language model", "openai", "chatgpt", "i cannot")


def pair_ok(ins, res):
    if not isinstance(ins, str) or not isinstance(res, str):
        return "missing instruction/response"
    if not (20 <= len(ins) <= 400):
        return f"instruction len {len(ins)}"
    if not (20 <= len(res) <= 700):
        return f"response len {len(res)}"
    words = len(res.split())
    if not (1 <= words <= 130):
        return f"response words {words}"
    n_sent = len([s for s in re.split(r"[.!?]+", res) if s.strip()])
    if n_sent > 5:
        return "response > 5 sentences"
    bad = [x for x in CODE_MARKERS if x in res]
    if bad:
        return f"code marker {bad[0]!r}"
    boiler = [x for x in META_BOILER if x in res.lower()]
    if boiler:
        return f"ai-boilerplate {boiler[0]!r}"
    return None


def reason_key(r):
    if r.startswith("instruction len"):
        return "instruction_len"
    if r.startswith("response len"):
        return "response_len"
    if r.startswith("response words"):
        return "response_words"
    if r.startswith("response >"):
        return "response_sentences"
    if r.startswith("code marker"):
        return "code_marker"
    if r.startswith("ai-boilerplate"):
        return "ai_boilerplate"
    if r.startswith("duplicate instruction"):
        return "dup_instruction_vs_copy"
    if r.startswith("duplicate"):
        return "duplicate"
    if r.startswith("json"):
        return "json"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/sft/h2p2_mixed/raw")
    ap.add_argument("--copy", default="data/sft/h2_copy/pairs.jsonl")
    ap.add_argument("--out", default="data/sft/h2p2_mixed/pairs.jsonl")
    args = ap.parse_args()

    raw_dir = ROOT / args.raw_dir
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no QA slices found in {raw_dir} (run the builder first)")

    copy_path = ROOT / args.copy
    copy_pairs = [json.loads(line) for line in
                  copy_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {hashlib.sha1((p["instruction"] + "\x00" + p["response"])
                         .encode("utf-8")).hexdigest() for p in copy_pairs}
    seen_ins = {hashlib.sha1(p["instruction"].encode("utf-8")).hexdigest()
                for p in copy_pairs}

    kept, drops = [], []
    for f in files:
        n_in = n_ok = 0
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                drops.append((f.name, ln, f"json: {e}"))
                continue
            reason = pair_ok(d.get("instruction"), d.get("response"))
            if reason:
                drops.append((f.name, ln, reason))
                continue
            key = hashlib.sha1((d["instruction"] + "\x00" + d["response"])
                               .encode("utf-8")).hexdigest()
            if key in seen:
                drops.append((f.name, ln, "duplicate"))
                continue
            ins_key = hashlib.sha1(d["instruction"].encode("utf-8")).hexdigest()
            if ins_key in seen_ins:
                drops.append((f.name, ln, "duplicate instruction vs copy corpus"))
                continue
            seen.add(key)
            seen_ins.add(ins_key)
            kept.append({"instruction": d["instruction"], "response": d["response"]})
            n_ok += 1
        print(f"[h2p2] {f.name}: {n_ok}/{n_in} kept", flush=True)

    mixed = copy_pairs + kept
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in mixed) + "\n",
                   encoding="utf-8")
    vmeta = {
        "copy_pairs": len(copy_pairs),
        "qa_seen": sum(1 for f in files for line in
                       f.read_text(encoding="utf-8").splitlines() if line.strip()),
        "qa_kept": len(kept),
        "qa_dropped": len(drops),
        "drop_reasons": {k: sum(1 for d in drops if reason_key(d[2]) == k)
                          for k in sorted({reason_key(d[2]) for d in drops})},
        "mixed_total": len(mixed),
        "sources": sorted(f.name for f in files),
    }
    (out.parent / "validate_meta.json").write_text(json.dumps(vmeta, indent=2),
                                                   encoding="utf-8")
    print(f"[h2p2] mixed corpus: {len(copy_pairs)} copy + {len(kept)} QA "
          f"= {len(mixed)} pairs -> {args.out}", flush=True)
    print(f"[h2p2] QA drops: {len(drops)}; reasons: "
          + ", ".join(f"{k}={v}" for k, v in sorted(vmeta['drop_reasons'].items(),
                                                     key=lambda t: -t[1])), flush=True)
    if len(kept) < 200:
        print("[h2p2] FAIL: fewer than 200 valid QA pairs", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
