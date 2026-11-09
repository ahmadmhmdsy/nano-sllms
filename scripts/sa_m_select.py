"""SA-M step 2: pick the answer among k saved samples by behaviour, not only by AST shape (CPU only, no model).

sa7_runtime keeps the samples that pass the task's one visible test and votes among them by AST shape. Here every
runnable sample (visible kind pass / wrong / error) is also run on up to PROBES extra inputs made by mutating the
visible call's literal arguments (empty, reversed, +-1, one element changed, ...). Their expected outputs are
unknown and the hidden tests are never read: the probes only tell samples apart.
Methods, each choosing among the visible passers (no passer: all fall back to the sa7 vote over all samples):
  ast      sa7_runtime.vote, the baseline
  exec     largest cluster of identical probe outputs, then the AST vote inside it
  mbr      most probe agreement with the other passers (MBR-exec), ties by the AST vote
  mbr_all  most probe agreement with every runnable sample, failers included
Scores use first-round samples in draw order at N = 16/32/64 prefixes, as scratch/sa_m_curve.py does. Subsets:
"scoped" = the 99 SA-B scoped ids, "rest" = the other gate ids (the dev side; scoped is the check).
Per-task confidence features for the step 3 abstain rule go to --features.
  .venv/Scripts/python.exe -I scripts/sa_m_select.py --out research/raw/sa_m_select.json \
      --features research/raw/sa_m_select_features.jsonl full_sa4=research/raw/sa_m_cov_sa4_full.samples.jsonl \
      scoped_sa4=research/raw/sa_m_cov_sa4_scoped.samples.jsonl scoped_sa_b=research/raw/sa_m_cov_sa_b_scoped.samples.jsonl
"""
import argparse
import ast
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from sa2_build import GUARD, parallel, run_guarded  # noqa: E402
from sa5_rft import load_task_ids  # noqa: E402
from sa7_runtime import load_gate, safe_example, vote  # noqa: E402

VERSION = "probe-v1"
PROBES = 16
NS = (16, 32, 64)
METHODS = ("ast", "exec", "mbr", "mbr_all")
RUNNABLE = ("pass", "wrong", "error")
PROBE_TAG = "SAM_PROBE "
CACHE = ROOT / "scratch" / "sa_m_probe_cache.jsonl"
CANON = '''
def _sam_canon(x, depth=0):
    if depth > 30:
        return "..."
    if isinstance(x, float):
        return round(x, 6)
    if type(x) in (list, tuple):
        return type(x)(_sam_canon(y, depth + 1) for y in x)
    if isinstance(x, (set, frozenset)):
        return ("set", sorted(repr(_sam_canon(y, depth + 1)) for y in x))
    if isinstance(x, dict):
        return ("dict", sorted((repr(_sam_canon(k, depth + 1)), repr(_sam_canon(v, depth + 1))) for k, v in x.items()))
    if hasattr(x, "__next__"):
        return ("iter", [_sam_canon(y, depth + 1) for y in _sam_itertools.islice(x, 1000)])
    return x
'''


# ---------------------------------------------------------------- probes

def mutations(v, rng: random.Random) -> list:
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, int):
        return [v + 1, v - 1, 0, -v, v + rng.randint(2, 5)]
    if isinstance(v, float):
        return [v + 1.0, -v, 0.0, v / 2]
    if isinstance(v, str):
        chars = list(v)
        rng.shuffle(chars)
        return ["", v[::-1], v[:len(v) // 2], v + v[:1], v.swapcase(), "".join(chars)]
    if isinstance(v, (list, tuple)):
        items = list(v)
        shuffled = items[:]
        rng.shuffle(shuffled)
        out = [[], items[::-1], shuffled, items[:-1], items + items[:1]]
        try:
            out.append(sorted(items))
        except TypeError:
            pass
        if items:
            i = rng.randrange(len(items))
            sub = mutations(items[i], rng)
            if sub:
                out.append(items[:i] + [rng.choice(sub)] + items[i + 1:])
        return [type(v)(x) for x in out]
    if isinstance(v, dict):
        out = [{}]
        if v:
            k = rng.choice(list(v))
            out.append({x: y for x, y in v.items() if x != k})
            sub = mutations(v[k], rng)
            if sub:
                out.append({**v, k: rng.choice(sub)})
        return out
    return []


def call_text(entry: str, args: list, kwargs: dict) -> str:
    return f"{entry}(" + ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]) + ")"


def probe_calls(task: dict) -> list[str]:
    """Mutations of the visible call's literal arguments, one argument at a time; [] if the call is not literal."""
    if not task["visible"]:
        return []
    example = safe_example(task["visible"][0], task["entry_point"])
    if example is None:
        return []
    call = ast.parse(example[0], mode="eval").body
    try:
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return []
    if None in kwargs:
        return []  # **mapping
    rng = random.Random(f"{VERSION}:{task['id']}")
    entry, original = task["entry_point"], call_text(task["entry_point"], args, kwargs)
    found = []
    for slot in [*range(len(args)), *kwargs]:
        value = args[slot] if isinstance(slot, int) else kwargs[slot]
        for m in mutations(value, rng):
            a, kw = list(args), dict(kwargs)
            if isinstance(slot, int):
                a[slot] = m
            else:
                kw[slot] = m
            found.append(call_text(entry, a, kw))
    unique = [c for c in dict.fromkeys(found) if c != original]
    return unique if len(unique) <= PROBES else rng.sample(unique, PROBES)


def probe_program(code: str, calls: list[str]) -> str:
    parts = [GUARD, "import itertools as _sam_itertools\nimport json as _sam_json\nimport re as _sam_re", code, CANON]
    for i, call in enumerate(calls):
        parts.append(f"try:\n    _sam_out = 'V' + _sam_re.sub(r'0x[0-9a-fA-F]+', '0x', repr(_sam_canon({call})))[:300]\n"
                     f"except Exception:\n    _sam_out = 'E'\n"
                     f"print({PROBE_TAG!r} + _sam_json.dumps([{i}, _sam_out]), flush=True)")
    return "\n".join(parts) + "\n"


def run_probes(item: tuple[str, str, list[str]]) -> list[str]:
    """Probe outputs in call order: 'V<repr>', 'E' (raised; the type is dropped) or 'K' (killed or never reached)."""
    _, code, calls = item
    _, _, stdout = run_guarded(probe_program(code, calls))
    got = {}
    for line in stdout.splitlines():
        if line.startswith(PROBE_TAG):
            try:
                i, out = json.loads(line[len(PROBE_TAG):])
                got[i] = out
            except (ValueError, TypeError):
                pass
    return [got.get(i, "K") for i in range(len(calls))]


def cache_key(task_id: str, code: str, calls: list[str]) -> str:
    return hashlib.sha256(json.dumps([VERSION, task_id, code, calls]).encode("utf-8")).hexdigest()


def probe_all(dumps: dict, calls: dict, workers: int) -> dict:
    done = {}
    if CACHE.exists():
        with CACHE.open(encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    done[e["k"]] = e["o"]
                except (ValueError, KeyError):
                    continue  # a torn last line
    todo = {}
    for samples in dumps.values():
        for tid, rows in samples.items():
            if not calls.get(tid):
                continue
            for s in rows:
                key = cache_key(tid, s["code"], calls[tid])
                if s["kind"] in RUNNABLE and key not in done:
                    todo[key] = (tid, s["code"], calls[tid])
    print(f"[probe] {len(done)} cached, {len(todo)} to run", flush=True)
    with CACHE.open("a", encoding="utf-8", newline="\n") as f:
        for (key, _), out in parallel(list(todo.items()), lambda kv: run_probes(kv[1]), workers, "probe"):
            done[key] = out
            if "K" not in out:  # kills depend on host load, as sa2_build.VerifyCache
                f.write(json.dumps({"k": key, "o": out}) + "\n")
    return done


# ---------------------------------------------------------------- selection

def load_dump(path: Path) -> dict[str, list[dict]]:
    by_id = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["origin"] == "sample":
                by_id[s["id"]].append({"i": s["i"], "code": s["code"], "kind": s["visible"]["kind"],
                                       "passed": s["passed"]})
    return {tid: sorted(rows, key=lambda s: s["i"]) for tid, rows in by_id.items()}


def agreement(sigs: list[list[str] | None]) -> np.ndarray:
    """sim[i, j] = share of probes on which samples i and j return the same output; 0 where either is unprobed."""
    vocab: dict[str, int] = {}
    width = max(len(s) for s in sigs if s is not None)
    arr = np.full((len(sigs), width), -1, dtype=np.int64)
    for i, s in enumerate(sigs):
        if s is not None:
            arr[i] = [vocab.setdefault(x, len(vocab)) for x in s]
    sim = (arr[:, None, :] == arr[None, :, :]).mean(-1)
    sim[arr[:, 0] < 0] = 0
    sim[:, arr[:, 0] < 0] = 0
    return sim


def task_scores(pre: list[dict], sigs: list | None) -> tuple[dict, dict]:
    """(method -> picked index, confidence features) for one task's first-N samples."""
    n = len(pre)
    passers = [i for i, s in enumerate(pre) if s["kind"] == "pass"]
    feats = {"n_pass": len(passers), "pass_share": len(passers) / n, "probed": sigs is not None}
    if not passers:
        pick = vote(pre, list(range(n)))
        return {m: pick for m in METHODS}, feats
    picks = {"ast": vote(pre, passers)}
    if sigs is None:
        return {m: picks["ast"] for m in METHODS}, feats
    groups = defaultdict(list)
    for i in passers:
        groups[tuple(sigs[i])].append(i)
    top = max(len(g) for g in groups.values())
    picks["exec"] = vote(pre, [i for g in groups.values() if len(g) == top for i in g])
    sim = agreement(sigs)
    runnable = [j for j, s in enumerate(pre) if s["kind"] in RUNNABLE]
    for method, pool in (("mbr", passers), ("mbr_all", runnable)):
        score = {i: sim[i, pool].sum() - (sim[i, i] if i in pool else 0) for i in passers}
        best = max(score.values())
        picks[method] = vote(pre, [i for i in passers if score[i] >= best - 1e-9])
    others_all = max(1, len(runnable) - 1)
    feats.update({"exec_clusters": len(groups), "exec_top": top, "exec_top_share": top / len(passers),
                  "exec_top_of_n": top / n,
                  "mbr_agree": float((sim[picks["mbr"], passers].sum() - 1) / max(1, len(passers) - 1)),
                  "mbr_all_agree": float((sim[picks["mbr_all"], runnable].sum() - 1) / others_all)})
    return picks, feats


def ci(diffs: list[float], n: int = 10000) -> list[float]:
    d = np.asarray(diffs, dtype=float)
    means = np.sort(d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1))
    return [round(100 * float(means[int(0.025 * n)]), 1), round(100 * float(means[int(0.975 * n) - 1]), 1)]


def summarize(rows: list[dict]) -> dict:
    out = {}
    for n in NS:
        col = [r["by_n"][n] for r in rows if n in r["by_n"]]
        if len(col) != len(rows):
            continue
        answered = [c for c in col if c["feats"]["n_pass"]]
        base = [float(c["picked"]["ast"]) for c in col]
        oracle = 100 * sum(c["oracle"] for c in col) / len(col)
        entry = {"tasks": len(col), "oracle": round(oracle, 1), "answered": round(100 * len(answered) / len(col), 1)}
        for m in METHODS:
            runtime = 100 * sum(c["picked"][m] for c in col) / len(col)
            entry[m] = {"runtime": round(runtime, 1),
                        "acc_on_answered": round(100 * sum(c["picked"][m] for c in answered) / max(1, len(answered)), 1)}
            if m != "ast":
                gap = oracle - 100 * sum(base) / len(col)
                entry[m]["vs_ast_ci"] = ci([float(c["picked"][m]) - b for c, b in zip(col, base)])
                entry[m]["gap_closed"] = round((runtime - 100 * sum(base) / len(col)) / gap, 2) if gap else None
        out[n] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="research/raw/sa_v_clean_gate.json")
    ap.add_argument("--scoped", default="research/raw/sa_b_scoped_gate.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("dumps", nargs="+", help="label=samples.jsonl")
    args = ap.parse_args()
    tasks = {t["id"]: t for t in load_gate(args.ids)}
    scoped = set(load_task_ids(args.scoped))
    calls = {tid: probe_calls(t) for tid, t in tasks.items()}
    dumps = {}
    for arg in args.dumps:
        label, path = arg.split("=", 1)
        dumps[label] = {tid: rows for tid, rows in load_dump(ROOT / path).items() if tid in tasks}
    probed = probe_all(dumps, calls, args.workers)

    report = {"ids": args.ids, "probes_max": PROBES, "version": VERSION,
              "tasks_with_probes": sum(bool(c) for c in calls.values()), "models": {}}
    with (ROOT / args.features).open("w", encoding="utf-8", newline="\n") as feat_f:
        for label, samples in dumps.items():
            rows = []
            for tid, all_rows in samples.items():
                row = {"id": tid, "scoped": tid in scoped, "by_n": {}}
                for n in [n for n in NS if n <= len(all_rows)]:
                    pre = all_rows[:n]
                    sigs = None
                    if calls[tid]:
                        sigs = [probed.get(cache_key(tid, s["code"], calls[tid])) if s["kind"] in RUNNABLE else None
                                for s in pre]
                        if all(s is None for s in sigs):
                            sigs = None
                    picks, feats = task_scores(pre, sigs)
                    cell = {"oracle": any(s["passed"] for s in pre), "feats": feats,
                            "picked": {m: pre[i]["passed"] for m, i in picks.items()}}
                    row["by_n"][n] = cell
                    feat_f.write(json.dumps({"label": label, "id": tid, "scoped": row["scoped"], "n": n,
                                             "oracle": cell["oracle"], "picked": cell["picked"], **feats}) + "\n")
                rows.append(row)
            subsets = {"all": rows, "scoped": [r for r in rows if r["scoped"]], "rest": [r for r in rows if not r["scoped"]]}
            report["models"][label] = {name: summarize(sub) for name, sub in subsets.items()
                                       if sub and not (name == "all" and len(sub) == len(subsets["scoped"]))}
    (ROOT / args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label, subs in report["models"].items():
        for name, by_n in subs.items():
            for n, e in by_n.items():
                cells = "  ".join(f"{m} {e[m]['runtime']:5.1f}/{e[m]['acc_on_answered']:5.1f}"
                                  + (f" {e[m]['vs_ast_ci']}" if m != "ast" else "") for m in METHODS)
                print(f"{label:12s} {name:6s} N{n:<3d} n={e['tasks']:3d} oracle {e['oracle']:5.1f}  {cells}")


if __name__ == "__main__":
    main()
