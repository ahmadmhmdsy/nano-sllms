"""SA-M step 4 (2026-10-28): the frozen product config and its held-out scoring. CPU only, no model.

Frozen config (user-approved "Approve N64"): runs/sa4_scale075, k = 64 samples, no retry, mbr selector
(sa_m_select), B2 abstain (sa_m_route --abstain-features scope_router). Fit once on the clean 370 (full_sa4 N64):
the sample-based scope router on all 370 tasks; the abstain model on the 370 passers with their out-of-fold scope
logits as feature 0 (as in step 3); threshold = mean of the five step-3 fold thresholds (re-derived here and
checked against research/raw/sa_m_route_b2.json). Saved to --config.
Each held-out arm (an sa7_runtime --set he|mbpp --k 64 --retries 0 dump) is then scored:
  answer-all  the mbr pick on every task
  confident   answer iff the task has visible tests, a visible passer and p(correct) >= threshold
Lane = the scoped rule on the reference solution (sa_b_scoped_gate.in_scope), as for the 99 scoped gate tasks.
Step 4 gate: on the pooled scoped lane of the held-out arms, answer-all >= 45 % and confident accuracy >= 75 %.
An arm named gate (the full_sa4 dump) is an in-sample replay: a pipeline check, not evidence.
  .venv/Scripts/python.exe -I scripts/sa_m_product.py --out research/raw/sa_m_product.json \
      mbpp=research/raw/sa_m_prod_mbpp.samples.jsonl he=research/raw/sa_m_prod_he.samples.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scratch")]  # -I leaves the script directory off sys.path
from sa_b_families import features  # noqa: E402
from sa_b_scoped_gate import in_scope  # noqa: E402
from sa_m_route import (ABSTAIN_NAMES, DUMPS, FOLDS, SCOPE_NAMES, abstain_x, fit_logreg, inner_threshold,  # noqa: E402
                        logit, point, sample_stats, scope_stats, sigmoid)
from sa_m_select import RUNNABLE, cache_key, load_dump, probe_all, probe_calls, task_scores  # noqa: E402
from sa7_runtime import load_gate, load_he, load_mbpp, parallel, run_visible  # noqa: E402

N = 64
L2 = 1.0
TARGET = 0.78
GATE_ANSWER_ALL = 45.0
GATE_CONFIDENT = 75.0


def pack(model: tuple, names: list[str]) -> dict:
    mu, sd, w = model
    return {"names": names, "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def fit_frozen(features_path: Path) -> tuple[dict, tuple, tuple, dict]:
    feats = [json.loads(s) for s in features_path.open(encoding="utf-8")]
    feats = {f["id"]: f for f in feats if f["label"] == "full_sa4" and f["n"] == N}
    dump = load_dump(ROOT / DUMPS["full_sa4"])
    rows = []
    for tid, f in sorted(feats.items()):
        pre = dump[tid][:N]
        rows.append({"id": tid, "scoped": f["scoped"], "n_pass": f["n_pass"], "correct": float(f["picked"]["mbr"]),
                     "scope_x": scope_stats(pre), "x": abstain_x(f, 0.0, sample_stats(pre))})
    perm = np.random.default_rng(0).permutation(len(rows))  # the step-3 task folds (sa_m_route)
    folds = np.array([int(p) % FOLDS for p in perm])
    S = np.array([r["scope_x"] for r in rows])
    ys = np.array([float(r["scoped"]) for r in rows])
    for k in range(FOLDS):
        model = fit_logreg(S[folds != k], ys[folds != k], L2)
        for i in np.where(folds == k)[0]:
            rows[i]["x"][0] = float(logit(model, S[i:i + 1])[0])
    keep = [i for i, r in enumerate(rows) if r["n_pass"]]
    X = np.array([rows[i]["x"] for i in keep])
    y = np.array([rows[i]["correct"] for i in keep])
    g = folds[keep]
    thresholds = []
    for k in range(FOLDS):
        groups = g[g != k]
        thresholds.append(inner_threshold(X[g != k], y[g != k], np.where(groups < k, groups, groups - 1), L2, TARGET))
    step3 = json.loads((ROOT / "research/raw/sa_m_route_b2.json").read_text(encoding="utf-8"))["abstain"]["thresholds"]
    if not np.allclose(np.round(thresholds, 3), step3, atol=1e-3):
        raise SystemExit(f"ABORT: fold thresholds {np.round(thresholds, 3).tolist()} != step 3 {step3}")
    scope_model, abstain_model = fit_logreg(S, ys, L2), fit_logreg(X, y, L2)
    config = {"version": "sa-m-product-v1", "model": "runs/sa4_scale075", "k": N, "retries": 0, "selector": "mbr",
              "abstain": "B2 (scope_router)", "fit_on": "clean 370 full_sa4 N64", "l2": L2, "target_inner": TARGET,
              "threshold": round(float(np.mean(thresholds)), 4), "fold_thresholds": [round(t, 4) for t in thresholds],
              "abstain_rows": len(keep), "scope_router": pack(scope_model, SCOPE_NAMES),
              "abstain_model": pack(abstain_model, ["scope_logit"] + ABSTAIN_NAMES[1:]),
              "confident_rule": "visible tests and n_pass > 0 and p >= threshold"}
    return config, scope_model, abstain_model, feats


def score_arm(tasks: list[dict], dump_path: Path, workers: int, scope_model: tuple, abstain_model: tuple,
              threshold: float) -> list[dict]:
    dump = load_dump(dump_path)
    missing = [t["id"] for t in tasks if len(dump.get(t["id"], [])) < N]
    if missing:
        raise SystemExit(f"ABORT: {dump_path.name}: {len(missing)} tasks with < {N} samples, e.g. {missing[:3]}")
    # Step 4b: tasks whose visible tests came from the wide prose parser were sampled without them; re-run the
    # new visible tests on the saved samples (nostop keeps its kind, as in sa7_runtime.check).
    todo = [(t, s) for t in tasks if t.get("wide") and t["visible"] for s in dump[t["id"]][:N] if s["kind"] != "nostop"]
    for (_, s), result in parallel(todo, lambda ts: run_visible(ts[0], ts[1]["code"]), workers, "wide visible"):
        s["kind"] = result["kind"]
    calls = {t["id"]: probe_calls(t) for t in tasks}
    probed = probe_all({"arm": {t["id"]: dump[t["id"]] for t in tasks}}, calls, workers)
    rows = []
    for t in tasks:
        pre = dump[t["id"]][:N]
        sigs = None
        if calls[t["id"]]:
            sigs = [probed.get(cache_key(t["id"], s["code"], calls[t["id"]])) if s["kind"] in RUNNABLE else None
                    for s in pre]
            if all(s is None for s in sigs):
                sigs = None
        picks, f = task_scores(pre, sigs)
        x = abstain_x(f, 0.0, sample_stats(pre))
        x[0] = float(logit(scope_model, np.array([scope_stats(pre)]))[0])
        p = float(sigmoid(logit(abstain_model, np.array([x])))[0]) if f["n_pass"] else 0.0
        ref = t["reference_raw"] if t["kind"] == "full" else t["prompt"] + t["reference_raw"]
        rows.append({"id": t["id"], "lane": in_scope(features(ref, t["prompt"])), "visible": bool(t["visible"]),
                     "n_pass": f["n_pass"], "correct": int(pre[picks["mbr"]]["passed"]),
                     "oracle": any(s["passed"] for s in pre), "p": round(p, 4), "answer_all": True,
                     "confident": bool(t["visible"]) and f["n_pass"] > 0 and p >= threshold})
    return rows


def summary(rows: list[dict]) -> dict:
    return {"answer_all": point(rows, "answer_all"), "confident": point(rows, "confident"),
            "oracle_pct": round(100 * sum(r["oracle"] for r in rows) / max(1, len(rows)), 1),
            "no_visible_tasks": sum(not r["visible"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="research/raw/sa_m_select_features.jsonl")
    ap.add_argument("--ids", default="research/raw/sa_v_clean_gate.json")
    ap.add_argument("--config", type=Path, default=Path("research/raw/sa_m_product_config.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("arms", nargs="+", help="he|he_wide|mbpp|gate=samples.jsonl (gate = in-sample replay; "
                                            "he_wide = step 4b, + prose docstring examples on the same HE samples)")
    args = ap.parse_args()

    config, scope_model, abstain_model, feats = fit_frozen(ROOT / args.features)
    (ROOT / args.config).write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    print(f"frozen config -> {args.config}: threshold {config['threshold']} (folds {config['fold_thresholds']})",
          flush=True)
    report = {"config": str(args.config), "threshold": config["threshold"], "sets": {}}
    pooled = []
    with (ROOT / args.out).with_name(args.out.stem + "_tasks.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for arm in args.arms:
            name, path = arm.split("=", 1)
            if name == "gate":
                tasks = load_gate(args.ids)
            elif name in ("he", "he_wide"):
                tasks, he_stats = load_he(args.workers, wide=name == "he_wide")
                report[f"{name}_examples"] = he_stats
                print(name, "examples", he_stats, flush=True)
            elif name == "mbpp":
                tasks = load_mbpp()
            else:
                raise SystemExit(f"unknown arm {name}")
            rows = score_arm(tasks, ROOT / path, args.workers, scope_model, abstain_model, config["threshold"])
            if name == "gate":
                report["gate_replay_check"] = {
                    "lane_mismatch": sum(r["lane"] != feats[r["id"]]["scoped"] for r in rows),
                    "mbr_mismatch": sum(r["correct"] != int(feats[r["id"]]["picked"]["mbr"]) for r in rows)}
            else:
                pooled += rows
            for r in rows:
                f.write(json.dumps({"arm": name, **r}) + "\n")
            for sub in ("all", "scoped", "rest"):
                sel = [r for r in rows if sub == "all" or r["lane"] == (sub == "scoped")]
                report["sets"][f"{name}/{sub}"] = summary(sel)
    for sub in ("all", "scoped", "rest"):
        sel = [r for r in pooled if sub == "all" or r["lane"] == (sub == "scoped")]
        if sel:
            report["sets"][f"heldout/{sub}"] = summary(sel)
    if "heldout/scoped" in report["sets"]:
        s = report["sets"]["heldout/scoped"]
        report["gate_verdict"] = {
            "population": "held-out scoped lane (pooled arms)", "tasks": s["answer_all"]["tasks"],
            "answer_all_pct": s["answer_all"]["correct_pct"], "answer_all_ci95": s["answer_all"]["acc_ci95"],
            "confident_answered_pct": s["confident"]["answered_pct"],
            "confident_acc_pct": s["confident"]["acc_answered_pct"], "confident_ci95": s["confident"]["acc_ci95"],
            "pass": s["answer_all"]["correct_pct"] >= GATE_ANSWER_ALL
                    and s["confident"]["acc_answered_pct"] >= GATE_CONFIDENT}
    (ROOT / args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for key, s in report["sets"].items():
        a, c = s["answer_all"], s["confident"]
        print(f"{key:16s} n={a['tasks']:3d} oracle {s['oracle_pct']:5.1f} | answer-all {a['correct_pct']:5.1f} "
              f"{a['acc_ci95']} | confident {c['answered_pct']:5.1f}% acc {c['acc_answered_pct']:5.1f} {c['acc_ci95']} "
              f"wrong {c['wrong_pct']:4.1f} | no-visible {s['no_visible_tasks']}")
    for key in ("gate_replay_check", "gate_verdict"):
        if key in report:
            print(key, report[key])


if __name__ == "__main__":
    main()
