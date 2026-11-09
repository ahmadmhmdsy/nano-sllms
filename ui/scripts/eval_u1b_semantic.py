"""U-1b semantic gate: does the model FOLLOW the ask, or echo a memorized row?

Why this gate exists (research/EXPERIMENTS.md, U-1 verdict + "U-1b PRE-REGISTER"):
the U-1 validity gate reported greedy_valid 1.00 / pass@4 1.00, but its held-out set
was 500 rows holding only 36 DISTINCT prompts (top prompt repeated 27x). That score
cannot separate "learned request -> UI" from "reproduced a training row". This gate
scores semantics on distinct prompts, and measures memorization directly.

M1/M5 decide the verdict against the PRE-REGISTERED bars. M2/M3/M4/M6 are diagnostics
and do not move the bars.

Usage (venv only):
    .venv/Scripts/python.exe -m ui.scripts.eval_u1b_semantic --n-prod 300
    .venv/Scripts/python.exe -m ui.scripts.eval_u1b_semantic --smoke
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ui.src.chain import chain_to_spec, spec_to_chain
from ui.src.validate import validate_chain

MAIN_CORPUS = Path("data/u1/raw/u1_main_50k.jsonl")
PROD_CORPUS = Path("ui/data/space_bunny/production_50k/corpus.jsonl")
DEFAULT_MODEL = "runs/u1_full/final"

STOP = set(
    "a an the for with and or of to in on at is are be been my our me i we you want need "
    "make build show create add new simple quick demo widget page screen app data team "
    "that this it as of use using get give put let".split()
)


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def words(text):
    t = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())
    return {w for w in t.split() if w not in STOP and len(w) > 1}


def types_of(spec):
    els = (spec or {}).get("elements") or {}
    return [e.get("type") for e in els.values() if isinstance(e, dict) and e.get("type")]


def string_props(spec):
    out = []
    for e in ((spec or {}).get("elements") or {}).values():
        for _, v in ((e.get("props") or {}) if isinstance(e, dict) else {}).items():
            if isinstance(v, str):
                out.append(v)
    return out


def norm_chain(c):
    return "\n".join(ln.strip() for ln in str(c).splitlines() if ln.strip())


def task_line_of(chain):
    for ln in str(chain).splitlines():
        s = ln.strip()
        if s.startswith("TASK "):
            return s[5:].strip()
    return None


def f1(gen, ref):
    g, r = set(gen), set(ref)
    if not r:
        return None
    if not g:
        return 0.0
    inter = len(g & r)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(g), inter / len(r)
    return 2 * prec * rec / (prec + rec)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-main", type=int, default=36)
    ap.add_argument("--n-prod", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=780)
    ap.add_argument("--out", default="runs/u1_full/u1b_gate.json")
    ap.add_argument("--resume", action="store_true", help="reuse records already in --out and score only what is missing")
    args = ap.parse_args()

    if args.smoke:
        args.n_main, args.n_prod = 4, 4

    t_load = time.time()
    main_rows = load_jsonl(MAIN_CORPUS)
    prod_rows = load_jsonl(PROD_CORPUS)

    # ---- eval sets: DISTINCT prompts only ------------------------------------
    seen, set_a = set(), []
    for r in main_rows:
        if r.get("split") != "test":
            continue
        t = r["task"].strip()
        if t in seen:
            continue
        seen.add(t)
        set_a.append({"task": t, "spec": r.get("spec"), "arch": r.get("arch"), "set": "A_main_test"})
    set_a = set_a[: args.n_main]

    seen, set_b = set(), []
    for r in prod_rows:
        if r.get("split") != "test":
            continue
        t = r["task"].strip()
        if t in seen:
            continue
        seen.add(t)
        set_b.append({"task": t, "spec": r.get("spec"), "arch": r.get("arch_hint"), "set": "B_prod_test"})
        if len(set_b) >= args.n_prod:
            break
    rows = set_a + set_b

    # ---- k-NN reference pool (replaces the degenerate centroid classifier) ----
    pool = []
    for src_rows, key in ((main_rows, "arch"), (prod_rows, "arch_hint")):
        tr = [r for r in src_rows if r.get("split") == "train" and r.get("spec")]
        step = max(1, len(tr) // 2000)
        for r in tr[::step]:
            pool.append((set(types_of(r["spec"])), r.get(key)))
    pool_t = [p[0] for p in pool]
    pool_y = [p[1] for p in pool]

    def predict_arch(gen_types, k=5):
        g = set(gen_types)
        if not g:
            return None
        scored = sorted(((jaccard(g, s), y) for s, y in zip(pool_t, pool_y)), key=lambda t: -t[0])[:k]
        return collections.Counter(y for _, y in scored).most_common(1)[0][0]

    # ---- memorization index over the TRAIN chains ----------------------------
    train_prompts = set()
    train_chains = set()
    for src_rows in (main_rows, prod_rows):
        for r in src_rows:
            if r.get("split") != "train":
                continue
            train_prompts.add(r["task"].strip())
            c = r.get("chain")
            if not c and r.get("spec"):
                c = spec_to_chain(r["spec"], r["task"])
            if c:
                train_chains.add(norm_chain(c))

    print(
        json.dumps(
            {
                "eval_rows": len(rows),
                "A_main_test": len(set_a),
                "B_prod_test": len(set_b),
                "knn_pool": len(pool_t),
                "train_chains": len(train_chains),
                "train_prompts": len(train_prompts),
                "load_s": round(time.time() - t_load, 1),
            }
        ),
        flush=True,
    )

    out_path = Path(args.out)
    records = []
    if args.resume and out_path.exists():
        try:
            records = json.loads(out_path.read_text(encoding="utf-8")).get("records", [])
            print(json.dumps({"resumed_records": len(records)}), flush=True)
        except Exception as exc:
            print(json.dumps({"resume_failed": str(exc)}), flush=True)
            records = []
    done_keys = {(r["set"], r["task"]) for r in records}
    todo = [r for r in rows if (r["set"], r["task"]) not in done_keys]
    rows = todo

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).cuda().eval()

    def flush_partial():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"partial": True, "records": records}, indent=1), encoding="utf-8")

    def gen(task):
        ids = tok(task.strip() + "\n", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        txt = tok.batch_decode(out[:, ids.input_ids.shape[1]:], skip_special_tokens=True)
        return txt[0].split("<|endoftext|>")[0].strip()

    t0 = time.time()
    per_set = collections.defaultdict(list)
    sig_counter = collections.Counter()

    def _score(r):
        chain = gen(r["task"])
        issues = [x for x in validate_chain(chain) if x is not None]
        spec = None
        try:
            spec, cissues = chain_to_spec(chain)
            spec_issues = [x for x in (cissues or []) if x is not None]
        except Exception as exc:
            spec_issues = [str(exc)]
        valid = (not issues) and (not spec_issues)
        g_types, r_types = types_of(spec), types_of(r["spec"])
        sc = f1(g_types, r_types)
        sig = tuple(sorted(g_types))
        sig_counter[sig] += 1
        tl = task_line_of(chain)
        return {
            "set": r["set"],
            "task": r["task"][:90],
            "valid": valid,
            "type_f1": None if sc is None else round(sc, 3),
            "arch_ref": r["arch"],
            "arch_pred": predict_arch(g_types),
            "content_jaccard": round(jaccard(words(r["task"]), words(" ".join(string_props(spec)))), 3),
            "n_gen_elements": len(g_types),
            "n_ref_elements": len(r_types),
            "sig": list(sig),
            "chain_copies_train": norm_chain(chain) in train_chains,
            "emitted_task": (tl or "")[:70],
            "task_replaced": bool(tl) and tl.strip() != r["task"].strip(),
            "emitted_task_is_known_prompt": bool(tl) and tl.strip() in train_prompts,
        }

    for i, r in enumerate(rows):
        # A model generation can produce a spec that no validator can survive. Scoring must
        # degrade to an explicit failure row, never kill a 40+ minute run (seen 2026-09-26:
        # a dangling child id crashed validate_chain with KeyError).
        try:
            rec = _score(r)
        except Exception as exc:
            rec = {
                "set": r["set"],
                "task": r["task"][:90],
                "valid": False,
                "type_f1": None,
                "arch_ref": r["arch"],
                "arch_pred": None,
                "content_jaccard": 0.0,
                "n_gen_elements": 0,
                "n_ref_elements": len(types_of(r["spec"])),
                "sig": [],
                "chain_copies_train": False,
                "emitted_task": "",
                "task_replaced": False,
                "emitted_task_is_known_prompt": False,
                "error": (type(exc).__name__ + ": " + str(exc))[:160],
            }
        records.append(rec)
        per_set[r["set"]].append(rec)
        if (i + 1) % 25 == 0:
            flush_partial()
            print(json.dumps({"done": i + 1, "of": len(rows), "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    def agg(rs):
        f1s = [r["type_f1"] for r in rs if r["type_f1"] is not None]
        cj = [r["content_jaccard"] for r in rs if r["content_jaccard"] is not None]
        arch = [r for r in rs if r["arch_ref"]]
        n = max(len(rs), 1)
        return {
            "n": len(rs),
            "valid_rate": round(sum(r["valid"] for r in rs) / n, 3),
            "mean_type_f1": round(sum(f1s) / max(len(f1s), 1), 3),
            "median_type_f1": round(sorted(f1s)[len(f1s) // 2], 3) if f1s else None,
            "f1_ge_0.5_rate": round(sum(1 for x in f1s if x >= 0.5) / max(len(f1s), 1), 3),
            "arch_accuracy": round(sum(r["arch_ref"] == r["arch_pred"] for r in arch) / max(len(arch), 1), 3),
            "mean_content_jaccard": round(sum(cj) / max(len(cj), 1), 3),
            "copy_rate": round(sum(r["chain_copies_train"] for r in rs) / n, 3),
            "task_replaced_rate": round(sum(r["task_replaced"] for r in rs) / n, 3),
            "known_prompt_rate": round(sum(r["emitted_task_is_known_prompt"] for r in rs) / n, 3),
        }

    per_set = collections.defaultdict(list)
    sig_counter = collections.Counter()
    for r in records:
        per_set[r["set"]].append(r)
        if r.get("sig"):
            sig_counter[tuple(r["sig"])] += 1
    modal_sig, modal_n = sig_counter.most_common(1)[0] if sig_counter else ((), 0)
    out = {
        "gate": "U-1b semantic follow",
        "model": args.model,
        "n_main_test": len(set_a),
        "n_prod_test": len(set_b),
        "results": {k: agg(v) for k, v in sorted(per_set.items())},
        "echo": {
            "distinct_signatures": len(sig_counter),
            "modal_signature_share": round(modal_n / max(sum(sig_counter.values()), 1), 3),
            "modal_signature_types": list(modal_sig)[:12],
        },
        "wall_s": round(time.time() - t0, 1),
        "worst": sorted([r for r in records if r["type_f1"] is not None], key=lambda r: r["type_f1"])[:8],
    }
    print("RESULT " + json.dumps(out, indent=1), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": out, "records": records}, indent=1), encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
