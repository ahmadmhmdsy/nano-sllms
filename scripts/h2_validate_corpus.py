"""H2 corpus validator (Track H2 Phase 1 - copy-behavior corpus).

Validates + dedupes the subagent-authored slices in data/sft/h2_copy/raw/
into one training file. Enforces the research/h2_corpus_spec.md rules:

  - schema: {"instruction": str, "response": str}
  - instruction 30-1200 chars, contains a question OR ends mid-note
  - response 1-30 words, <= 2 sentences, no code markers
  - copy fidelity: every content word of the response must appear in the
    instruction (<= --allowed-oov tolerated) - the corpus must teach
    copying, never inventing
  - dedupe by sha1(instruction + response)

Run (CPU, co-run safe):
  & .\.venv\Scripts\python.exe scripts\h2_validate_corpus.py
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FENCE = chr(96) * 3
FORBIDDEN = ("import ", "def ", "class ", ">>>", FENCE)
STOP = {"the", "a", "an", "my", "me", "i", "is", "are", "was", "and", "or",
        "of", "in", "on", "for", "to", "with", "at", "by", "it", "its",
        "our", "their", "his", "her", "your", "you", "we", "us", "that",
        "this", "these", "those", "she", "he", "they", "them", "be", "been"}


def norm_word(w):
    return re.sub(r"[^a-z0-9]+", "", w.lower())


def content_words(text):
    out = []
    for w in text.split():
        n = norm_word(w)
        if n and n not in STOP:
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/sft/h2_copy/raw")
    ap.add_argument("--out", default="data/sft/h2_copy/pairs.jsonl")
    ap.add_argument("--allowed-oov", type=int, default=1)
    args = ap.parse_args()

    raw_dir = ROOT / args.raw_dir
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no slices found in {raw_dir}")

    kept, drops = [], []
    seen = set()
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
            ins, res = d.get("instruction"), d.get("response")
            if not isinstance(ins, str) or not isinstance(res, str):
                drops.append((f.name, ln, "missing instruction/response"))
                continue
            if not (30 <= len(ins) <= 1200):
                drops.append((f.name, ln, f"instruction len {len(ins)}"))
                continue
            if len(res.split()) < 1 or len(res.split()) > 30:
                drops.append((f.name, ln, f"response words {len(res.split())}"))
                continue
            if len([s for s in re.split(r"[.!?]+", res) if s.strip()]) > 2:
                drops.append((f.name, ln, "response > 2 sentences"))
                continue
            bad = [x for x in FORBIDDEN if x in res]
            if bad:
                drops.append((f.name, ln, f"code marker {bad[0]!r}"))
                continue
            # copy fidelity: response content words must come from instruction
            ctx_words = set(content_words(ins))
            oov = [w for w in content_words(res) if w not in ctx_words]
            if len(oov) > args.allowed_oov:
                drops.append((f.name, ln, f"not a copy (oov: {oov[:4]})"))
                continue
            key = hashlib.sha1((ins + "\x00" + res).encode("utf-8")).hexdigest()
            if key in seen:
                drops.append((f.name, ln, "duplicate"))
                continue
            seen.add(key)
            kept.append({"instruction": ins, "response": res})
            n_ok += 1
        print(f"[h2] {f.name}: {n_ok}/{n_in} kept", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in kept) + "\n",
                   encoding="utf-8")
    print(f"[h2] kept {len(kept)} pairs -> {args.out}; dropped {len(drops)}", flush=True)
    reasons = {}
    for _, _, r in drops:
        reasons[r.split(":")[0].split(" ")[0]] = reasons.get(r.split(":")[0].split(" ")[0], 0) + 1
    top = sorted(reasons.items(), key=lambda t: -t[1])[:8]
    print("[h2] drop reasons: " + ", ".join(f"{k}={v}" for k, v in top), flush=True)
    for d in drops[:15]:
        print(f"[h2]   drop {d[0]}:{d[1]} {d[2]}", flush=True)
    if len(kept) < 200:
        print("[h2] FAIL: fewer than 200 valid pairs - re-spawn authors", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
