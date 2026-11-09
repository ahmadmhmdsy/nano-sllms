"""SA-M step 3 (2026-10-28): CPU router + abstain on top of the step-2 mbr selector. No generation, no GPU.

1. Router: a prompt-only logistic regression (numpy Newton, no sklearn) trained on data/sa2/train.jsonl
   (he/mbpp shapes, read-only) to predict "in scope" = the scoped-gate rule on the reference solution
   (<= 48 AST nodes, no imports, one function; scratch/sa_b_scoped_gate.in_scope). Evaluated on the clean
   370 (val prompts, labels from the val references; the scoped flag in the step-2 features must match).
2. Abstain: a small correctness model on runtime-only signals (router logit, visible-pass share, mbr
   agreement, exec clusters, size/imports of the passing samples) predicting "the mbr pick is correct".
   Fit on full_sa4 N64 with 5-fold CV grouped by task; the confidence threshold is chosen inside each
   training fold (inner 4-fold, lowest threshold reaching --target accuracy) and applied to the held-out
   fold. scoped_sa4 / scoped_sa_b rows are scored with the fold model that never saw that task.
   Rule: answer iff the task has a visible passer and p(correct) >= threshold; otherwise abstain.

  .venv/Scripts/python.exe -I scripts/sa_m_route.py --out research/raw/sa_m_route.json
"""
import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scratch")]  # -I leaves the script directory off sys.path
from sa5_rft import load_task_ids  # noqa: E402
from sa_b_families import TOPICS, features  # noqa: E402
from sa_b_scoped_gate import in_scope  # noqa: E402
from sa_m_select import load_dump  # noqa: E402

DUMPS = {"full_sa4": "research/raw/sa_m_cov_sa4_full.samples.jsonl",
         "scoped_sa4": "research/raw/sa_m_cov_sa4_scoped.samples.jsonl",
         "scoped_sa_b": "research/raw/sa_m_cov_sa_b_scoped.samples.jsonl"}
FOLDS = 5
MODULE_RE = re.compile(r"\b(import|module|library|math|regex|regular expression|heapq|heap|collections|counter|"
                       r"deque|itertools|datetime|date|time|bisect|random|cmath|lambda|re)\b")
HARD_RE = re.compile(r"\b(efficient\w*|dynamic programming|recurs\w*|graph|tree|path\w*|permutation\w*|"
                     r"combination\w*|subsequence\w*|bracket\w*|parenthes\w*|encod\w*|decod\w*|pattern\w*|"
                     r"class|object|cost|ways|optimal|interval\w*|schedule\w*)\b")
ROUTER_NAMES = (["shape_he", "log_chars", "log_desc_words", "desc_lines", "examples", "argc", "typed",
                 "module_hits", "hard_hits", "digits_share"] + [f"topic_{name}" for name, _, _ in TOPICS])
ABSTAIN_NAMES = ["router_logit", "pass_share", "mbr_agree", "probed", "exec_top_share", "log_clusters",
                 "log_pass_nodes", "pass_imports", "pass_helpers"]
SCOPE_NAMES = ["log_med_nodes", "log_p25_nodes", "log_min_pass_nodes", "imports", "helpers", "nested_or_rec",
               "parsed", "visible_error"]  # over all N samples; option B (sample-based scope router)


def arg_count(prompt: str, entry: str) -> int:
    try:
        for n in ast.walk(ast.parse(prompt + "\n    pass")):
            if isinstance(n, ast.FunctionDef) and n.name == entry:
                return len(n.args.args) + len(n.args.kwonlyargs)
    except SyntaxError:
        pass
    for line in prompt.splitlines():
        line = line.strip()
        if entry + "(" not in line:
            continue
        try:
            tree = ast.parse(re.sub(r"^(assert|>>>)\s*", "", line))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == entry:
                return len(n.args) + len(n.keywords)
    return 0


def prompt_features(prompt: str, shape: str, entry: str) -> list[float]:
    low = prompt.lower()
    lines = [s.strip() for s in prompt.splitlines() if s.strip()]
    examples = sum(s.startswith((">>>", "assert")) for s in lines)
    desc = [s for s in lines if not s.startswith((">>>", "assert", "def ", '"""', "'''", "from ", "import "))]
    words = sum(len(s.split()) for s in desc)
    sig = next((s for s in lines if s.startswith("def ")), "")
    return [float(shape == "he"), math.log1p(len(prompt)), math.log1p(words), min(len(desc), 20),
            min(examples, 10), min(arg_count(prompt, entry), 6), float(":" in sig[sig.find("("):] if sig else 0),
            len(MODULE_RE.findall(low)), len(HARD_RE.findall(low)), sum(c.isdigit() for c in prompt) / max(1, len(prompt))
            ] + [float(bool(re.search(t_re, low))) for _, t_re, _ in TOPICS]


def label(row: dict) -> int:
    ref = row["completion"] if row["shape"] == "mbpp" else row["prompt"] + row["completion"]
    return int(in_scope(features(ref, row["prompt"])))


def load_rows(path: Path, ids: set | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, xs, ys = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["shape"] not in ("he", "mbpp") or (ids is not None and r["id"] not in ids):
                continue
            names.append(r["id"])
            xs.append(prompt_features(r["prompt"], r["shape"], r["entry_point"]))
            ys.append(label(r))
    return names, np.array(xs), np.array(ys, dtype=float)


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float) -> tuple:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + np.diag(reg) * len(y) / 1000 + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, Z.T @ (p - y) + reg * w * len(y) / 1000)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def logit(model: tuple, X: np.ndarray) -> np.ndarray:
    mu, sd, w = model
    return ((X - mu) / sd) @ w[:-1] + w[-1]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def sample_stats(samples: list[dict]) -> tuple[float, float, float]:
    sizes, imports, helpers = [], 0, 0
    for s in samples:
        if s["kind"] != "pass":
            continue
        try:
            tree = ast.parse(s["code"])
        except SyntaxError:
            continue
        sizes.append(sum(1 for _ in ast.walk(tree)))
        imports += any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
        helpers += sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree)) > 1
    if not sizes:
        return 0.0, 0.0, 0.0
    return math.log1p(float(np.median(sizes))), imports / len(sizes), helpers / len(sizes)


def scope_stats(samples: list[dict]) -> list[float]:
    sizes, pass_sizes, imports, helpers, deep = [], [], 0, 0, 0
    for s in samples:
        try:
            tree = ast.parse(s["code"])
        except SyntaxError:
            continue
        nodes = list(ast.walk(tree))
        sizes.append(len(nodes))
        if s["kind"] == "pass":
            pass_sizes.append(len(nodes))
        imports += any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes)
        funcs = [n for n in nodes if isinstance(n, ast.FunctionDef)]
        helpers += len(funcs) > 1
        loops = [n for n in nodes if isinstance(n, (ast.For, ast.While))]
        deep += (any(isinstance(m, (ast.For, ast.While)) for n in loops for m in ast.walk(n) if m is not n)
                 or any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == f.name
                        for f in funcs for c in ast.walk(f)))
    if not sizes:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sum(s["kind"] == "error" for s in samples) / max(1, len(samples))]
    k = len(sizes)
    return [math.log1p(float(np.median(sizes))), math.log1p(float(np.percentile(sizes, 25))),
            math.log1p(min(pass_sizes or sizes)), imports / k, helpers / k, deep / k, k / len(samples),
            sum(s["kind"] == "error" for s in samples) / len(samples)]


def abstain_x(f: dict, router_z: float, stats: tuple) -> list[float]:
    probed = float(bool(f.get("probed")) and "mbr_agree" in f)
    return [router_z, f["pass_share"], f["mbr_agree"] if probed else f["pass_share"], probed,
            f.get("exec_top_share", 1.0), math.log1p(f.get("exec_clusters", 1)), *stats]


def pick_threshold(p: np.ndarray, y: np.ndarray, target: float) -> float:
    order = np.argsort(-p)
    hits = np.cumsum(y[order])
    for k in range(len(order), 0, -1):  # largest answered set whose accuracy reaches the target
        if hits[k - 1] / k >= target:
            return float(p[order[k - 1]])
    return float("inf")


def inner_threshold(X: np.ndarray, y: np.ndarray, groups: np.ndarray, l2: float, target: float) -> float:
    oof = np.zeros(len(y))
    for k in range(FOLDS - 1):
        test = groups % (FOLDS - 1) == k
        oof[test] = sigmoid(logit(fit_logreg(X[~test], y[~test], l2), X[test]))
    return pick_threshold(oof, y, target)


def wilson(k: int, n: int) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def point(rows: list[dict], answered_key: str) -> dict:
    n = len(rows)
    ans = [r for r in rows if r[answered_key]]
    k = sum(r["correct"] for r in ans)
    return {"tasks": n, "answered_pct": round(100 * len(ans) / max(1, n), 1),
            "acc_answered_pct": round(100 * k / max(1, len(ans)), 1), "acc_ci95": wilson(k, len(ans)),
            "correct_pct": round(100 * k / max(1, n), 1), "wrong_pct": round(100 * (len(ans) - k) / max(1, n), 1)}


def curve(rows: list[dict], key: str) -> list[dict]:
    pool = sorted((r for r in rows if r["n_pass"]), key=lambda r: -r[key])
    out = []
    for cov in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        k = max(1, round(cov * len(pool)))
        out.append({"top_share_of_passers": cov, "answered_pct": round(100 * k / len(rows), 1),
                    "acc_answered_pct": round(100 * sum(r["correct"] for r in pool[:k]) / k, 1)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="research/raw/sa_v_clean_gate.json")
    ap.add_argument("--features", default="research/raw/sa_m_select_features.jsonl")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--target", type=float, default=0.78, help="inner-fold accuracy target for the threshold")
    ap.add_argument("--gate", type=float, default=0.75)
    ap.add_argument("--l2", type=float, default=1.0, help="abstain-model L2 (router uses 1e-3)")
    ap.add_argument("--abstain-features", choices=("base", "samples", "scope_router"), default="base",
                    help="samples = also feed the SCOPE_NAMES sample statistics to the abstain model; scope_router ="
                         " replace router_logit with the out-of-fold sample-based scope logit (option B)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print("router: featurising train split ...", flush=True)
    _, Xtr, ytr = load_rows(ROOT / "data/sa2/train.jsonl")
    router = fit_logreg(Xtr, ytr, 1e-3)
    gate_ids = load_task_ids(args.ids)
    vids, Xv, yv = load_rows(ROOT / "data/sa2/val.jsonl", gate_ids)
    zv = logit(router, Xv)
    router_z = dict(zip(vids, zv))
    in_scope_ref = dict(zip(vids, yv))
    train_auc = auc(logit(router, Xtr), ytr)
    print(f"router: train rows {len(ytr)} pos {ytr.mean():.3f} auc {train_auc:.3f}; "
          f"gate rows {len(yv)} pos {yv.mean():.3f} auc {auc(zv, yv):.3f}", flush=True)

    feats = [json.loads(s) for s in (ROOT / args.features).open(encoding="utf-8")]
    feats = [f for f in feats if f["n"] == args.n]
    mismatch = [f["id"] for f in feats if f["label"] == "full_sa4" and bool(in_scope_ref[f["id"]]) != f["scoped"]]
    if mismatch:
        raise SystemExit(f"ABORT: scoped label mismatch on {len(mismatch)} tasks, e.g. {mismatch[:3]}")
    rows = []
    for name, path in DUMPS.items():
        dump = load_dump(ROOT / path)
        for f in (f for f in feats if f["label"] == name):
            pre = dump[f["id"]][:args.n]
            scope = scope_stats(pre)
            x = abstain_x(f, float(router_z[f["id"]]), sample_stats(pre))
            rows.append({"label": name, "id": f["id"], "scoped": f["scoped"], "n_pass": f["n_pass"],
                         "correct": int(f["picked"]["mbr"]), "router_z": float(router_z[f["id"]]),
                         "mbr_agree": f.get("mbr_agree", f["pass_share"]) if f["n_pass"] else 0.0,
                         "scope_x": scope, "x": x + scope if args.abstain_features == "samples" else x})

    task_ids = sorted({r["id"] for r in rows if r["label"] == "full_sa4"})
    perm = np.random.default_rng(0).permutation(len(task_ids))
    fold = {tid: int(perm[i]) % FOLDS for i, tid in enumerate(task_ids)}

    # option B: scope router from the samples (labels = reference scope rule), task-grouped out-of-fold AUC
    full = [r for r in rows if r["label"] == "full_sa4"]
    y_scope = np.array([float(r["scoped"]) for r in full])
    scope_auc = {}
    for kind, getx in (("prompt", lambda r: [r["router_z"]]), ("samples", lambda r: r["scope_x"]),
                       ("prompt+samples", lambda r: [r["router_z"], *r["scope_x"]])):
        X = np.array([getx(r) for r in full])
        oof = np.zeros(len(full))
        for k in range(FOLDS):
            test = np.array([fold[r["id"]] == k for r in full])
            model = fit_logreg(X[~test], y_scope[~test], args.l2)
            oof[test] = logit(model, X[test])
            for r in rows:  # every dump row is scored by the fold model that held out its task
                if fold[r["id"]] == k:
                    r[f"scope_oof_{kind}"] = float(logit(model, np.array([getx(r)]))[0])
        scope_auc[kind] = round(auc(oof, y_scope), 3)
    print("scope router out-of-fold AUC on clean 370:", scope_auc, flush=True)
    if args.abstain_features == "scope_router":
        for r in rows:
            r["x"][0] = r["scope_oof_samples"]
    base = [r for r in rows if r["label"] == "full_sa4" and r["n_pass"]]
    thresholds, coefs = [], []
    for k in range(FOLDS):
        tr = [r for r in base if fold[r["id"]] != k]
        X = np.array([r["x"] for r in tr])
        y = np.array([r["correct"] for r in tr], dtype=float)
        model = fit_logreg(X, y, args.l2)
        groups = np.array([fold[r["id"]] for r in tr])
        groups = np.array([g if g < k else g - 1 for g in groups])
        t = inner_threshold(X, y, groups, args.l2, args.target)
        thresholds.append(t)
        coefs.append(model[2][:-1].tolist())
        for r in rows:
            if fold[r["id"]] == k and r["n_pass"]:
                r["p"] = float(sigmoid(logit(model, np.array([r["x"]])))[0])
                r["confident"] = r["p"] >= t
    for r in rows:
        r.setdefault("p", 0.0)
        r.setdefault("confident", False)
        r["answer_all"] = True
        r["passer"] = bool(r["n_pass"])

    names = ABSTAIN_NAMES + (SCOPE_NAMES if args.abstain_features == "samples" else [])
    report = {"version": "route-v1", "n": args.n, "target_inner": args.target, "gate": args.gate,
              "abstain_features": args.abstain_features, "scope_router_oof_auc": scope_auc,
              "router": {"train_rows": len(ytr), "train_pos": round(float(ytr.mean()), 3),
                         "train_auc": round(train_auc, 3), "gate_rows": len(yv),
                         "gate_pos": round(float(yv.mean()), 3), "gate_auc": round(auc(zv, yv), 3),
                         "coef_std": dict(zip(ROUTER_NAMES, np.round(router[2][:-1], 3).tolist()))},
              "abstain": {"thresholds": [round(t, 3) for t in thresholds],
                          "coef_std_mean": dict(zip(names, np.round(np.mean(coefs, 0), 3).tolist()))},
              "sets": {}}
    for name in DUMPS:
        for sub in ("all", "scoped", "rest"):
            sel = [r for r in rows if r["label"] == name and (sub == "all" or r["scoped"] == (sub == "scoped"))]
            if not sel:
                continue
            ys = np.array([r["correct"] for r in sel if r["n_pass"]], dtype=float)
            ps = np.array([r["p"] for r in sel if r["n_pass"]])
            report["sets"][f"{name}/{sub}"] = {
                "answer_all": point(sel, "answer_all"), "passer_only": point(sel, "passer"),
                "confident": point(sel, "confident"), "abstain_auc": round(auc(ps, ys), 3),
                "router_auc_scope": round(auc(np.array([r["router_z"] for r in sel]),
                                              np.array([float(r["scoped"]) for r in sel])), 3),
                "curve_p": curve(sel, "p"), "curve_mbr_agree": curve(sel, "mbr_agree"),
                "curve_router": curve(sel, "router_z")}
    conf = report["sets"]["full_sa4/all"]["confident"]
    report["gate_verdict"] = {"population": "full_sa4/all (clean 370, out-of-fold)",
                              "acc_answered_pct": conf["acc_answered_pct"], "ci95": conf["acc_ci95"],
                              "pass": conf["acc_answered_pct"] >= 100 * args.gate}
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with args.out.with_name(args.out.stem + "_tasks.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("x", "scope_x")}) + "\n")
    for key, s in report["sets"].items():
        print(f"{key:20s} all {s['answer_all']['correct_pct']:5.1f} | passer {s['passer_only']['answered_pct']:5.1f}% "
              f"ans acc {s['passer_only']['acc_answered_pct']:5.1f} | confident {s['confident']['answered_pct']:5.1f}% "
              f"ans acc {s['confident']['acc_answered_pct']:5.1f} {s['confident']['acc_ci95']} "
              f"wrong {s['confident']['wrong_pct']:4.1f} | abst auc {s['abstain_auc']} router auc {s['router_auc_scope']}")
    print("thresholds", report["abstain"]["thresholds"], "gate", report["gate_verdict"])


if __name__ == "__main__":
    main()
