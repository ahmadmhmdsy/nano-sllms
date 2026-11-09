"""H2 Phase 2 general-QA corpus builder (TASKS row 37 phase 2).

User decision 2026-09-09: NO teacher inference - sample human-written
general-QA pairs from two ungated datasets and mix them with the Phase 1
copy corpus (585 pairs) so QA-style prompts stop collapsing into
copy-tautologies (Phase 1 honest delta) and the student widens beyond
Python with real text knowledge.

Sources (VERIFIED via HF metadata probe 2026-09-09):
  - databricks/databricks-dolly-15k  (CC-BY-SA-3.0, 15k crowdworker pairs, ungated)
  - HuggingFaceH4/no_robots          (CC-BY-NC-4.0, 10k human-written pairs, ungated)
  - teknium/OpenHermes-2.5           (~1M GPT-4-distilled pairs, ungated) - added
    2026-09-09 on the user's "more powerful datasets" sanction; .env keys are
    loaded (names only are printed) but the HF_TOKEN is not needed for these
    ungated repos

Slices (deterministic, seed 42; v2 quotas after the first build measured
  attrition ~30-45% and showed no_robots Chat is multi-turn by construction -
  795/796 skipped - so Chat contributes nothing under the single-turn rule):
  - dolly open_qa                              -> up to 400 pairs
  - no_robots OpenQA/Brainstorm/Generation     -> up to 120/120/160
    (single-turn only; Chat + Coding/Classify/Extract/Rewrite/Summarize/
    ClosedQA/multi-turn excluded - we want prose QA, not code)

Filters: instruction 20-400 chars, response 20-700 chars, no code markers,
no AI-meta boilerplate, sha1 dedupe inside each slice. The official
validator is scripts/h2p2_validate_corpus.py (run after this).

Run (CPU + network, co-run safe):
  & .\.venv\Scripts\python.exe scripts\h2p2_build_corpus.py
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env_keys():
    """Load user keys from .env into os.environ (only key NAMES get printed)."""
    env = ROOT / ".env"
    loaded = []
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
                loaded.append(k)
    return loaded


FENCE = chr(96) * 3
CODE_MARKERS = ("import ", "def ", "class ", ">>>", FENCE, "print(")
META_BOILER = ("as an ai", "language model", "openai", "chatgpt", "i cannot")

DOLLY_QUOTA = 400
NR_QUOTAS = {"Open QA": 120, "Brainstorm": 120, "Generation": 160}
OPENHERMES_QUOTA = 500


def pair_ok(ins, res):
    """Return None if the pair passes the builder filters, else a reason."""
    if not isinstance(ins, str) or not isinstance(res, str):
        return "missing field"
    if not (20 <= len(ins) <= 400):
        return "instruction len %d" % len(ins)
    if not (20 <= len(res) <= 700):
        return "response len %d" % len(res)
    bad = [x for x in CODE_MARKERS if x in res]
    if bad:
        return "code marker %r" % bad[0]
    low = res.lower()
    boiler = [x for x in META_BOILER if x in low]
    if boiler:
        return "ai-boilerplate %r" % boiler[0]
    return None


def take(pool, quota, seen, tag):
    rng = random.Random(42)
    rng.shuffle(pool)
    out = []
    for row in pool:
        if len(out) >= quota:
            break
        reason = pair_ok(row["instruction"], row["response"])
        if reason:
            continue
        key = hashlib.sha1(
            (row["instruction"] + "\x00" + row["response"]).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    print(f"[h2p2] {tag}: {len(out)}/{quota} sampled (pool {len(pool)})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft/h2p2_mixed")
    args = ap.parse_args()

    keys = load_env_keys()
    print(f"[h2p2] .env keys loaded: {keys or 'none'} (values never printed)",
          flush=True)

    from datasets import load_dataset

    out = ROOT / args.out_dir
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    seen = set()

    # --- dolly: open_qa only (self-contained question + prose answer) -----
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    pool = []
    n_cat = {}
    for ex in dolly:
        cat = ex.get("category")
        n_cat[cat] = n_cat.get(cat, 0) + 1
        if cat == "open_qa":
            pool.append({"instruction": ex["instruction"].strip(),
                         "response": ex["response"].strip()})
    print(f"[h2p2] dolly categories: {json.dumps(n_cat, sort_keys=True)}", flush=True)
    dolly_rows = take(pool, DOLLY_QUOTA, seen, "dolly open_qa")

    # --- no_robots: 4 prose categories, single-turn only -------------------
    nr = load_dataset("HuggingFaceH4/no_robots", split="train")
    pools = {cat: [] for cat in NR_QUOTAS}
    n_cat = {}
    n_multi = 0
    for ex in nr:
        cat = ex.get("category")
        n_cat[cat] = n_cat.get(cat, 0) + 1
        msgs = ex.get("messages") or []
        if len(msgs) != 2 or msgs[0].get("role") != "user":
            n_multi += 1
            continue
        ins = (msgs[0].get("content") or "").strip()
        res = (msgs[1].get("content") or "").strip()
        if cat in pools:
            pools[cat].append({"instruction": ins, "response": res})
    print(f"[h2p2] no_robots categories: {json.dumps(n_cat, sort_keys=True)} "
          f"(multi-turn/system skipped: {n_multi})", flush=True)
    nr_rows = []
    for cat, quota in NR_QUOTAS.items():
        nr_rows += take(pools[cat], quota, seen, f"no_robots {cat}")

    # --- OpenHermes-2.5: GPT-4-distilled knowledge QA (user upgrade) -------
    # Schema probed 2026-09-09: ShareGPT format - conversations [{from, value}];
    # no top-level instruction/output keys (the v3 run sampled 0/500 for this
    # reason and was fixed). Self-contained 2-turn human->gpt pairs only.
    oh = load_dataset("teknium/OpenHermes-2.5", split="train")
    pool = []
    n_skip_ctx = 0
    for ex in oh:
        conv = ex.get("conversations") or []
        if ex.get("system_prompt") or len(conv) != 2 or conv[0].get("from") != "human":
            n_skip_ctx += 1  # system-prompted, multi-turn or wrong roles
            continue
        ins = (conv[0].get("value") or "").strip()
        res = (conv[1].get("value") or "").strip()
        if any(m in ins or m in res for m in ("Human:", "Assistant:")):
            n_skip_ctx += 1  # embedded multi-turn formatting
            continue
        pool.append({"instruction": ins, "response": res})
    print(f"[h2p2] openhermes self-contained pool: {len(pool)} "
          f"(system/multi-turn skipped: {n_skip_ctx})", flush=True)
    oh_rows = take(pool, OPENHERMES_QUOTA, seen, "openhermes")

    (raw / "qa_dolly.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in dolly_rows) + "\n",
        encoding="utf-8")
    (raw / "qa_no_robots.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in nr_rows) + "\n",
        encoding="utf-8")
    (raw / "qa_openhermes.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in oh_rows) + "\n",
        encoding="utf-8")
    build_meta = {
        "sources": {
            "dolly": {"repo": "databricks/databricks-dolly-15k",
                      "license": "CC-BY-SA-3.0", "category": "open_qa",
                      "quota": DOLLY_QUOTA, "sampled": len(dolly_rows)},
            "no_robots": {"repo": "HuggingFaceH4/no_robots",
                          "license": "CC-BY-NC-4.0",
                          "categories": sorted(NR_QUOTAS), "quota_per_cat": NR_QUOTAS,
                          "sampled": len(nr_rows)},
            "openhermes": {"repo": "teknium/OpenHermes-2.5",
                           "license": "unlisted (community GPT-4 distillate)",
                           "quota": OPENHERMES_QUOTA, "sampled": len(oh_rows)},
        },
        "seed": 42,
        "filters": {"instruction_chars": [20, 400], "response_chars": [20, 700],
                    "code_markers": list(CODE_MARKERS), "ai_boilerplate": META_BOILER,
                    "single_turn_only": True},
        "qa_pairs_total": len(dolly_rows) + len(nr_rows) + len(oh_rows),
    }
    (raw / "build_meta.json").write_text(json.dumps(build_meta, indent=2),
                                         encoding="utf-8")
    print(f"[h2p2] wrote {build_meta['qa_pairs_total']} QA pairs -> {raw} "
          f"(dolly {len(dolly_rows)}, no_robots {len(nr_rows)}, "
          f"openhermes {len(oh_rows)})", flush=True)
    if build_meta["qa_pairs_total"] < 200:
        raise SystemExit("[h2p2] FAIL: fewer than 200 QA pairs sampled")


if __name__ == "__main__":
    main()
