# mex/scripts/eval_mex.py — μ0 exact-match eval + μ1 mixed/binned CI scoring.
r"""μ0 eval: per-task exact-match on held-out prompts (no training-data reuse).

Loads runs/mex/<task>/final (config.yaml if present, else the committed
configs/mex_<task>.yaml; model.safetensors or any *.safetensors shard),
greedy-decodes after the task prompt prefix, and compares the continuation
up to the first newline with the held-out target. Decoding uses the
mex.src.vocab.CharVocab id mapping (our own per-char encoding) — no
AutoTokenizer needed. Every exact_match in a result payload carries its
Wilson 95% CI (mex.src.metrics.wilson_ci) under key "ci95" — the mu1 plan's
binomial-CI gate on every rate claim (TASKS row 80 follow-up).

Mode mixed (mu1 plan "Mixed eval set"): scores data/mex/mixed/val.jsonl
(200 held-out items = 50/task; the true task id never enters the text).
Without --route, the dense control (runs/mex/control/final) decodes every
prompt. With --route PATH, the arm-C router (mex/src/router.py) picks ONE
expert per prompt under argmax routing; that expert greedy-decodes it.
Experts load ONE AT A TIME on CPU from runs/mex/archive_12000/<task>/final
(the mu1 expert material). Mixed payloads report:
    routed_accuracy + ci95     decode quality end-to-end (headline)
    routing_accuracy + ci95    routing-only accuracy vs true labels
                               (0.25 random floor over 4 tasks)
    expert_<t> payloads        per-expert n/exact_match/ci95

Usage:
    & .\.venv\Scripts\python.exe mex\scripts\eval_mex.py x1|x2|x3|x4|control|all|mixed
    [--limit N]      debug subset (default: full 500/2000 per task / 200 mixed)
    [--route PATH]   mixed mode only: router.pt path for arm-C dispatch
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mex.src.metrics import wilson_ci  # plan-mandated binomial 95% CI
from mex.src.tasks import arith, structure, strops  # same seeds/formats as packing
from mex.src.vocab import CharVocab

RUNS = {"x1": "x1", "x2": "x2", "x3": "x3", "x4": "x4", "control": "control"}
CONFIGS = {"x1": "mex_x1", "x2": "mex_x2", "x3": "mex_x3",
           "x4": "mex_x4", "control": "mex_control"}
MIXED_DIR = ROOT / "data" / "mex" / "mixed"


def load(task: str, run_dir: Path | None = None):
    """Build the model for runs/mex/<task>/final and load its weights.

    run_dir overrides the run location (mixed mode points it at
    runs/mex/archive_12000/<task>/final). Config resolution: run/config.yaml
    first (train.py saves it via save_pretrained), else the committed
    configs/mex_<task>.yaml. Weights: model.safetensors, else any
    *.safetensors in the dir; a missing set is a hard error — the eval runs
    after Task 6 training, never before.
    """
    import yaml
    from safetensors.torch import load_file

    from src.model import build_model

    run = (Path(run_dir) if run_dir is not None
           else ROOT / "runs" / "mex" / task / "final")
    cfg_path = run / "config.yaml"
    cfg_json = run / "config.json"
    if not cfg_path.exists() and not cfg_json.is_file():
        cfg_path = ROOT / "configs" / f"{CONFIGS[task]}.yaml"
    if cfg_path and cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    elif cfg_json.is_file():
        # Arbitrary run dirs (mu1 --eval-run, e.g. the distill student) save
        # HF-style config.json instead — map it back to our yaml shape.
        cj = json.loads(cfg_json.read_text(encoding="utf-8"))
        cfg = {"model": {
            "layers": cj["num_hidden_layers"],
            "hidden": cj["hidden_size"],
            "heads": cj["num_attention_heads"],
            "kv_heads": cj["num_key_value_heads"],
            "ffn": cj["intermediate_size"],
            "ctx": cj["max_position_embeddings"],
            "dropout": cj.get("attention_dropout", 0.0),
            "tie_embeddings": cj.get("tie_word_embeddings", True),
        }, "tokenizer": {"vocab_size": cj["vocab_size"]}}
    else:
        raise FileNotFoundError(f"no config for task {task!r}: tried "
                                f"{run / 'config.yaml'}, run/config.json, "
                                f"and {ROOT / 'configs'}")

    weights = run / "model.safetensors"
    if not weights.exists():
        shards = sorted(run.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(
                f"no .safetensors under {run} — Task 6 (training) has not run "
                f"yet; eval_mex.py needs the trained final dir")
        weights = shards[0]
    state = load_file(str(weights))

    model = build_model(cfg, vocab_size=cfg["tokenizer"]["vocab_size"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    # lm_head.weight is tied to embed_tokens (same storage) and omitted by
    # safetensors saves of tied models (src/model.py load_finetune_init rule).
    bad_missing = [k for k in missing if k != "lm_head.weight"]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"weight mismatch for {task}: missing={bad_missing} "
            f"unexpected={sorted(unexpected)}")
    model.eval()
    # Training builds the model with use_cache=False; generation needs it on.
    model.config.use_cache = True
    return cfg, model


@torch.no_grad()
def exact_match(model, prompts: list[str], targets: list[str]) -> float:
    """Greedy-decode each prompt; hit iff the first decoded line == target."""
    voc = CharVocab()
    nl = voc.vocab["\n"]
    ctx = int(model.config.max_position_embeddings)
    hits = 0
    for p, t in zip(prompts, targets):
        ids = voc.encode(p)[-ctx:]
        # Generate past the target length so an unterminated line can never
        # fake a match by truncation; eos (newline) stops early anyway.
        max_new = min(ctx, len(voc.encode(t)) + 2)
        out = model.generate(torch.tensor([ids]), max_new_tokens=max_new,
                             do_sample=False, pad_token_id=voc.vocab["<pad>"],
                             eos_token_id=nl)
        pred = voc.decode(out[0][len(ids):]).split("\n", 1)[0]
        hits += int(pred.strip() == t.strip())
    return hits / len(prompts)


def _payload(exact_match: float, n: int,
             trivial: float | None = None) -> dict:
    """Exact-match payload plus the Wilson 95% CI on the rate (key ci95)."""
    k = round(exact_match * n)
    out = {"exact_match": exact_match, "ci95": wilson_ci(k, n)}
    if trivial is not None:
        out["trivial_baseline"] = trivial
    return out


def _mode_frac(targets: list[str], pool: list[str]) -> float:
    """Fraction of targets equal to the mode (most frequent) of pool.

    Counter.most_common breaks count ties by first-encounter order —
    deterministic across processes (a max(set(...)) tie-break would flip
    with PYTHONHASHSEED; measured ties exist, e.g. x2 '62'/'76' at 77).
    """
    mode = Counter(pool).most_common(1)[0][0]
    return sum(1 for x in targets if x == mode) / len(targets)


def samples(t: str, limit: int | None = None) -> tuple[list[str], list[str], float]:
    """Rebuild the SAME held-out prompts the generators produced (seed=42),
    plus this task's pre-registered trivial baseline, measured over the
    actual (possibly --limit-truncated) test mix."""
    if t == "x1":
        lines = (ROOT / "data" / "mex" / "x1" / "test.txt").read_text(
            encoding="utf-8").splitlines()
        if limit:
            lines = lines[:limit]
        prompts = [ln.split("|", 1)[0] + "|" for ln in lines]
        targets = [ln.split("|", 1)[1] for ln in lines]
        # Trivial = echo the bare word. build_x1_words.py's filter requires
        # voc strictly longer than bare, so a bare echo never matches the
        # held-out vocalization — ~0 by construction, measured here anyway.
        trivial = sum(1 for tar, ln in zip(targets, lines)
                      if tar == ln.split("|", 1)[0]) / len(targets)
        return prompts, targets, trivial
    if t == "x2":
        d = arith(seed=42, n_val=200, n_test=500)
        test = d["test"][:limit] if limit else d["test"]
        prompts = [ln.split("|", 1)[0] + "|" for ln in test]
        targets = [ln.split("|", 1)[1] for ln in test]
        # Trivial = always answer the mode of the TRAIN answers.
        train_targets = [ln.split("|", 1)[1] for ln in d["train"]]
        return prompts, targets, _mode_frac(targets, train_targets)
    if t == "x3":
        d = structure(seed=42, n_val=200, n_test=500)
        test = d["test"][:limit] if limit else d["test"]
        prompts, targets = [], []
        for ln in test:
            seq, label = ln.rstrip("\n").split("\n")
            prompts.append(seq + "\n")
            targets.append(label)
        # Trivial = always answer the mode of the TRAIN labels.
        train_labels = [ln.rstrip("\n").split("\n")[1] for ln in d["train"]]
        return prompts, targets, _mode_frac(targets, train_labels)
    if t == "x4":
        d = strops(seed=42, n_val=200, n_test=500)
        test = d["test"][:limit] if limit else d["test"]
        prompts = [ln.split("|", 1)[0] + "|" for ln in test]
        targets = [ln.split("|", 1)[1] for ln in test]
        # Trivial = identity: echo the source back (prompt form `mode:src|`).
        # Measured over the ACTUAL test mix: it matches exactly the lines
        # whose target equals the echoed source (copy always, rev/sort only
        # when the transform is a fixed point). Targets keep the line's
        # trailing newline; strip it before comparing.
        trivial = sum(1 for p, x in zip(prompts, targets)
                      if p.split(":", 1)[1][:-1] == x.strip()) / len(targets)
        return prompts, targets, trivial
    raise ValueError(f"unknown task {t!r}")


def mixed_eval(route_path: str | None, limit: int | None,
               model_dir: str | None = None) -> dict:
    """μ1 mixed-mode eval on data/mex/mixed/val.jsonl (200 held-out).

    route_path None: dense control decodes every mixed prompt. Otherwise the
    arm-C router picks ONE expert per prompt; experts load one at a time on
    CPU from runs/mex/archive_12000/<task>/final.

    model_dir (mu1 arm A --eval-run): replaces the dense control with an
    arbitrary run dir's model when --route is not given (read-only ref for
    the report; argmax-router mode is untouched).
    """
    from mex.src.router import load_router, route_tasks  # torch-weight import

    path = MIXED_DIR / "val.jsonl"
    items = [json.loads(line) for line
             in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        items = items[:limit]
    prompts = [i["prompt"] for i in items]
    targets = [i["target"] for i in items]
    true = [i["task"] for i in items]
    n = len(items)
    report: dict = {"mode": "mixed", "n": n,
                    "source": str(path.relative_to(ROOT))}

    if route_path is None:
        if model_dir:
            d = (Path(model_dir) if Path(model_dir).is_absolute()
                 else ROOT / model_dir)
            _, model = load("x1", d)
            model_name = str(d)
        else:
            _, model = load("control")
            model_name = "runs/mex/control/final"
        acc = exact_match(model, prompts, targets)
        report.update(routed=False, model=model_name,
                      **_payload(acc, n))
        return report

    router_model, rmeta = load_router(route_path)
    preds = route_tasks(router_model, prompts, CharVocab())
    routing_acc = sum(1 for p, t in zip(preds, true) if p == t) / n
    report.update(routed=True, model=str(route_path),
                  routing_accuracy=routing_acc,
                  routing_ci95=wilson_ci(round(routing_acc * n), n),
                  router_meta=rmeta)
    hits = 0
    for t in ("x1", "x2", "x3", "x4"):
        idxs = [k for k, p in enumerate(preds) if p == t]
        if not idxs:
            report[f"expert_{t}"] = {"n": 0}
            continue
        _, model = load(t, ROOT / "runs" / "mex" / "archive_12000" / t / "final")
        acc = exact_match(model, [prompts[k] for k in idxs],
                          [targets[k] for k in idxs])
        report[f"expert_{t}"] = {"n": len(idxs), **_payload(acc, len(idxs))}
        hits += round(acc * len(idxs))
    report["routed_accuracy"] = hits / n
    report["routed_ci95"] = wilson_ci(hits, n)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="mex exact-match eval")
    ap.add_argument("task", choices=[*RUNS, "all", "mixed"], default="all",
                    nargs="?")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: evaluate only the first N prompts")
    ap.add_argument("--route", default=None,
                    help="mixed mode: router.pt path (arm-C dispatch)")
    ap.add_argument("--eval-run", default=None, metavar="DIR",
                    help="(mu1 arm A) evaluate an arbitrary run dir (e.g. "
                         "runs/mex/soup/uniform) across x1..x4 like the "
                         "control branch; mex_eval.json is written into DIR. "
                         "In mixed mode: DIR's model replaces the dense "
                         "control when --route is not given")
    args = ap.parse_args()
    if args.eval_run and args.task != "mixed":
        run = Path(args.eval_run)
        if not run.is_absolute():
            run = ROOT / run
        _, model = load("x1", run)  # config.yaml must sit in the run dir
        report = {}
        for sub in ("x1", "x2", "x3", "x4"):
            prompts, targets, triv = samples(sub, args.limit)
            em = exact_match(model, prompts, targets)
            report[sub] = _payload(em, len(prompts), triv)
        (run / "mex_eval.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False),
            encoding="utf-8")
        print(str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else run,
              json.dumps(report, ensure_ascii=False))
        return
    if args.task == "mixed":
        if args.route and args.eval_run:
            raise SystemExit("--eval-run on mixed mode is for unrouted soup "
                             "decoding; drop --route or --eval-run")
        report = mixed_eval(args.route, args.limit, args.eval_run)
        out = ROOT / "runs" / "mex" / "mex_eval_mixed.json"
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        printed = dict(report)
        printed.pop("router_meta", None)  # keep the console line short
        print("mixed", json.dumps(printed, ensure_ascii=False))
        return
    todo = list(RUNS) if args.task == "all" else [args.task]
    for t in todo:
        run = ROOT / "runs" / "mex" / RUNS[t] / "final"
        if t == "control":  # one report, per-task breakdown
            _, model = load(t)
            report = {}
            for sub in ("x1", "x2", "x3", "x4"):
                prompts, targets, triv = samples(sub, args.limit)
                em = exact_match(model, prompts, targets)
                report[sub] = _payload(em, len(prompts), triv)
        else:
            _, model = load(t)
            prompts, targets, triv = samples(t, args.limit)
            em = exact_match(model, prompts, targets)
            report = _payload(em, len(prompts), triv)
        run.mkdir(parents=True, exist_ok=True)
        (run / "mex_eval.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(RUNS[t], json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
