"""E2 steps 2-3 (2026-10-29): a verifier head on a frozen encoder, TinyGSM-style. The encoder never trains.

embed (GPU): one forward pass of a frozen causal LM (default: the untouched base models/hf/SmolLM2-135M) over
  prompt + completion for the chosen samples of an sa7_runtime-format dump. Features per sample: mean final-layer
  hidden state over the completion tokens, mean mid-layer hidden state over them, the final hidden state of the
  last token, mean and min token log-prob of the completion, log completion length. Saved as an .npz.
    --set train   (e2_verifier_data dump) every non-nostop sample of the mixed tasks (some pass, some fail)
    --set gate|he|mbpp  the visible passers among the first --n samples: the only candidates a selector ranks
fit (CPU): numpy logistic regression (sa_m_route.fit_logreg) on train labels = passed the full tests. The l2 and the
  consensus weight lam are picked by 5-fold task-grouped CV on train, by top-1 accuracy among visible passers.
  Selection on an eval dump: among the first N samples' visible passers, verifier = argmax logit,
  combo = argmax logit + lam * consensus (consensus = the sample's mean probe agreement with the other passers,
  the mbr signal of sa_m_select; ast cluster share where the task has no probe calls); no visible passer = the
  sa_m_select vote, as mbr. Compared with mbr per task, paired bootstrap CI (sa_m_select.ci).
  E2 gate (user-approved plan): combo on the clean 370 at N64 >= mbr + 3 points with the CI above 0, and no loss
  on held-out HE + MBPP pooled at N64.

  .venv/Scripts/python.exe -I scripts/e2_verifier.py embed --set train --tasks research/raw/e2_train_sa4.tasks.jsonl \
      --dump research/raw/e2_train_sa4.samples.jsonl --vram-cap-gib 6.4 --out research/raw/e2_emb_base_train.npz
  .venv/Scripts/python.exe -I scripts/e2_verifier.py fit --train research/raw/e2_emb_base_train.npz \
      --train-dump research/raw/e2_train_sa4.samples.jsonl --train-tasks research/raw/e2_train_sa4.tasks.jsonl \
      --eval gate=research/raw/e2_emb_base_gate.npz:research/raw/sa_m_cov_sa4_full.samples.jsonl \
      --out research/raw/e2_verifier_base.json
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from sa_m_route import auc, fit_logreg, logit  # noqa: E402
from sa_m_select import NS, RUNNABLE, agreement, cache_key, ci, load_dump, probe_all, probe_calls, task_scores  # noqa: E402
from sa7_runtime import load_gate, load_he, load_mbpp, shape  # noqa: E402

L2S = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LAMS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
FOLDS = 5
GATE_POINTS = 3.0


def arm_tasks(name: str, tasks_path: str, ids: str, workers: int) -> list[dict]:
    if name == "gate":
        return load_gate(ids)
    if name == "he":
        return load_he(workers)[0]
    if name == "mbpp":
        return load_mbpp()
    with (ROOT / tasks_path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def raw_samples(path: Path) -> dict[str, list[dict]]:
    by_id = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["origin"] == "sample":
                by_id[s["id"]].append(s)
    return {tid: sorted(rows, key=lambda s: s["i"]) for tid, rows in by_id.items()}


# ---------------------------------------------------------------- embed

def chosen(dump: dict[str, list[dict]], mode: str, n: int) -> list[tuple[str, dict]]:
    out = []
    for tid, rows in dump.items():
        if mode == "train":
            outcomes = {s["passed"] for s in rows}
            if outcomes == {True, False}:
                out += [(tid, s) for s in rows if not s["nostop"]]
        else:
            out += [(tid, s) for s in rows[:n] if s["visible"]["kind"] == "pass"]
    return out


def embed(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda":
        if args.vram_cap_gib <= 0:
            raise SystemExit("--vram-cap-gib is required on cuda (fresh free VRAM - 0.75 GiB)")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gib * (1 << 30) / total), 0)
    t0 = time.perf_counter()
    prompts = {t["id"]: t["prompt"] for t in arm_tasks(args.set, args.tasks, args.ids, args.workers)}
    items = chosen(raw_samples(ROOT / args.dump), "train" if args.set == "train" else "eval", args.n)
    missing = {tid for tid, _ in items if tid not in prompts}
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} dump ids without a task, e.g. {sorted(missing)[:3]}")
    tok = AutoTokenizer.from_pretrained(ROOT / args.encoder)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(ROOT / args.encoder, torch_dtype=dtype).to(args.device).eval()
    mid = model.config.num_hidden_layers // 2
    seqs = []
    for tid, s in items:  # the generation context: prompt (as HFModel tokenized it) then the completion
        c = tok(s["completion"], add_special_tokens=False)["input_ids"][:args.max_tokens // 2] or [tok.eos_token_id]
        p = tok(prompts[tid])["input_ids"][-(args.max_tokens - len(c)):]
        seqs.append((p + c, len(p)))
    print(f"[embed] {args.set}: {len(items)} samples over {len({t for t, _ in items})} tasks, "
          f"encoder {args.encoder}, mid layer {mid}", flush=True)
    order = sorted(range(len(seqs)), key=lambda j: len(seqs[j][0]))
    X = np.zeros((len(seqs), 3 * model.config.hidden_size + 3), dtype=np.float32)
    start, done = 0, 0
    while start < len(order):
        width = len(seqs[order[start]][0])
        end = start + 1
        while end < len(order) and (end - start + 1) * len(seqs[order[end]][0]) <= args.token_budget:
            end += 1
        idx = order[start:end]
        width = len(seqs[idx[-1]][0])
        ids = torch.full((len(idx), width), tok.eos_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for r, j in enumerate(idx):
            ids[r, :len(seqs[j][0])] = torch.tensor(seqs[j][0])
            mask[r, :len(seqs[j][0])] = 1
        with torch.no_grad():
            out = model(input_ids=ids.to(args.device), attention_mask=mask.to(args.device), output_hidden_states=True)
        final, middle = out.hidden_states[-1].float(), out.hidden_states[mid].float()
        for r, j in enumerate(idx):
            seq, s0 = seqs[j]
            n = len(seq)
            lp = torch.log_softmax(out.logits[r, s0 - 1:n - 1].float(), -1)
            tok_lp = lp.gather(1, ids[r, s0:n].to(args.device)[:, None])[:, 0]
            X[j] = np.concatenate([final[r, s0:n].mean(0).cpu().numpy(), middle[r, s0:n].mean(0).cpu().numpy(),
                                   final[r, n - 1].cpu().numpy(),
                                   [tok_lp.mean().item(), tok_lp.min().item(), math.log(n - s0)]])
        del out, final, middle
        start = end
        done += len(idx)
        if done // 2000 != (done - len(idx)) // 2000 or start == len(order):
            print(f"[embed] {done}/{len(order)} {time.perf_counter() - t0:.0f}s", flush=True)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, id=np.array([tid for tid, _ in items]), i=np.array([s["i"] for _, s in items]),
                        passed=np.array([s["passed"] for _, s in items]), encoder=args.encoder, set=args.set,
                        dump=args.dump)
    print(f"[embed] -> {args.out} {X.shape} {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- fit / score

def load_emb(path: str) -> tuple[np.ndarray, dict, str]:
    z = np.load(ROOT / path)
    index = {(str(tid), int(i)): r for r, (tid, i) in enumerate(zip(z["id"], z["i"]))}
    return z["X"], index, str(z["encoder"])


def probe_sigs(tasks: list[dict], dump: dict, workers: int) -> dict[str, list | None]:
    calls = {t["id"]: probe_calls(t) for t in tasks}
    probed = probe_all({"arm": {t["id"]: dump[t["id"]] for t in tasks}}, calls, workers)
    out = {}
    for t in tasks:
        sigs = None
        if calls[t["id"]]:
            sigs = [probed.get(cache_key(t["id"], s["code"], calls[t["id"]])) if s["kind"] in RUNNABLE else None
                    for s in dump[t["id"]]]
            if all(s is None for s in sigs):
                sigs = None
        out[t["id"]] = sigs
    return out


def consensus(pre: list[dict], sigs: list | None, passers: list[int]) -> np.ndarray:
    """Per passer: mean agreement with the other passers (probe outputs; ast shape where unprobed)."""
    if len(passers) < 2:
        return np.zeros(len(passers))
    if sigs is not None and any(sigs[i] is not None for i in passers):
        sim = agreement(sigs)[np.ix_(passers, passers)]
    else:
        shapes = [shape(pre[i]["code"]) for i in passers]
        sim = np.array([[float(a == b) for b in shapes] for a in shapes])
    return (sim.sum(1) - np.diag(sim)) / (len(passers) - 1)


def select(pre: list[dict], sigs: list | None, z: np.ndarray, lam: float) -> int:
    """z = verifier logits of the visible passers, in index order."""
    passers = [i for i, s in enumerate(pre) if s["kind"] == "pass"]
    score = z + lam * consensus(pre, sigs, passers) if lam else z
    return passers[int(np.argmax(score))]


def train_cv(X: np.ndarray, index: dict, dump: dict, sigs: dict, folds: int) -> dict:
    keys = sorted(index, key=index.get)
    y = np.array([float(next(s for s in dump[tid] if s["i"] == i)["passed"]) for tid, i in keys])
    tids = sorted({tid for tid, _ in keys})
    fold_of = {tid: int(f) % folds for tid, f in zip(tids, np.random.default_rng(0).permutation(len(tids)))}
    g = np.array([fold_of[tid] for tid, _ in keys])
    rows_of = {tid: {i: index[(tid, i)] for t2, i in keys if t2 == tid} for tid in tids} if len(tids) < 50 else None
    if rows_of is None:
        rows_of = defaultdict(dict)
        for (tid, i), r in index.items():
            rows_of[tid][i] = r
    sel_tasks = [tid for tid in tids if any(s["kind"] == "pass" for s in dump[tid])]
    base = {"mbr": [], "ast": []}
    for tid in sel_tasks:
        picks, _ = task_scores(dump[tid], sigs[tid])
        base["mbr"].append(float(dump[tid][picks["mbr"]]["passed"]))
        base["ast"].append(float(dump[tid][picks["ast"]]["passed"]))
    report = {"samples": len(y), "pos_share": round(float(y.mean()), 4), "tasks": len(tids),
              "selection_tasks": len(sel_tasks), "mbr_pct": round(100 * float(np.mean(base["mbr"])), 1),
              "ast_pct": round(100 * float(np.mean(base["ast"])), 1), "l2": {}, "lam": {}}
    oof_by_l2 = {}
    for l2 in L2S:
        oof = np.zeros(len(y))
        for k in range(folds):
            oof[g == k] = logit(fit_logreg(X[g != k], y[g != k], l2), X[g == k])
        oof_by_l2[l2] = oof
        top1, pz, py = [], [], []
        for tid in sel_tasks:
            pre = dump[tid]
            passers = [i for i, s in enumerate(pre) if s["kind"] == "pass"]
            z = np.array([oof[rows_of[tid][pre[i]["i"]]] for i in passers])
            top1.append(float(pre[select(pre, sigs[tid], z, 0.0)]["passed"]))
            pz += z.tolist()
            py += [float(pre[i]["passed"]) for i in passers]
        report["l2"][l2] = {"top1_pct": round(100 * float(np.mean(top1)), 1),
                            "auc_all": round(auc(oof, y), 4), "auc_passers": round(auc(np.array(pz), np.array(py)), 4)}
    best_l2 = max(L2S, key=lambda l2: (report["l2"][l2]["top1_pct"], report["l2"][l2]["auc_passers"]))
    oof = oof_by_l2[best_l2]
    for lam in LAMS:
        top1 = []
        for tid in sel_tasks:
            pre = dump[tid]
            passers = [i for i, s in enumerate(pre) if s["kind"] == "pass"]
            z = np.array([oof[rows_of[tid][pre[i]["i"]]] for i in passers])
            top1.append(float(pre[select(pre, sigs[tid], z, lam)]["passed"]))
        report["lam"][lam] = round(100 * float(np.mean(top1)), 1)
    best_lam = max(LAMS, key=lambda lam: (report["lam"][lam], -lam))
    report.update({"best_l2": best_l2, "best_lam": best_lam})
    report["model"] = fit_logreg(X, y, best_l2)
    return report


def score_arm(name: str, tasks: list[dict], dump: dict, sigs: dict, X: np.ndarray, index: dict, model: tuple,
              lam: float) -> tuple[dict, list[dict]]:
    rows = []
    for t in tasks:
        full = dump[t["id"]]
        row = {"id": t["id"], "by_n": {}}
        for n in NS:
            if len(full) < n:
                continue
            pre = full[:n]
            s_n = sigs[t["id"]][:n] if sigs[t["id"]] is not None else None
            if s_n is not None and all(s is None for s in s_n):
                s_n = None
            picks, f = task_scores(pre, s_n)
            res = {"oracle": int(any(s["passed"] for s in pre)), "mbr": int(pre[picks["mbr"]]["passed"]),
                   "n_pass": f["n_pass"]}
            if f["n_pass"]:
                passers = [i for i, s in enumerate(pre) if s["kind"] == "pass"]
                try:
                    z = logit(model, X[[index[(t["id"], pre[i]["i"])] for i in passers]])
                except KeyError as e:
                    raise SystemExit(f"ABORT: {name}: no embedding for passer {e}") from None
                res["verifier"] = int(pre[select(pre, s_n, z, 0.0)]["passed"])
                res["combo"] = int(pre[select(pre, s_n, z, lam)]["passed"])
                if n == max(NS):
                    res["z"] = [round(float(v), 3) for v in z]
                    res["y"] = [int(pre[i]["passed"]) for i in passers]
            else:
                res["verifier"] = res["combo"] = res["mbr"]
            row["by_n"][n] = res
        rows.append(row)
    summary = {}
    for n in NS:
        col = [r["by_n"][n] for r in rows if n in r["by_n"]]
        if not col:
            continue
        pct = {m: round(100 * sum(c[m] for c in col) / len(col), 1) for m in ("oracle", "mbr", "verifier", "combo")}
        summary[n] = {"tasks": len(col), **pct,
                      "verifier_vs_mbr_ci": ci([c["verifier"] - c["mbr"] for c in col]),
                      "combo_vs_mbr_ci": ci([c["combo"] - c["mbr"] for c in col])}
    zs = [(v, y) for r in rows for v, y in zip(r["by_n"].get(max(NS), {}).get("z", []),
                                                 r["by_n"].get(max(NS), {}).get("y", []))]
    if zs:
        summary["auc_passers"] = round(auc(np.array([v for v, _ in zs]), np.array([y for _, y in zs])), 4)
    return summary, rows


def fit(args) -> None:
    t0 = time.perf_counter()
    X, index, encoder = load_emb(args.train)
    train_tasks = arm_tasks("train", args.train_tasks, args.ids, args.workers)
    dump = load_dump(ROOT / args.train_dump)
    used = {tid for tid, _ in index}
    train_tasks = [t for t in train_tasks if t["id"] in used]
    sigs = probe_sigs(train_tasks, dump, args.workers)
    cv = train_cv(X, index, {tid: dump[tid] for tid in used}, sigs, FOLDS)
    model = cv.pop("model")
    print(f"[cv] encoder {encoder}: {json.dumps(cv)}", flush=True)
    report = {"encoder": encoder, "train": args.train, "cv": cv, "arms": {}}
    pooled = defaultdict(list)
    with (ROOT / args.out).with_name(Path(args.out).stem + "_tasks.jsonl").open("w", encoding="utf-8",
                                                                                 newline="\n") as f:
        for spec in args.eval:
            name, paths = spec.split("=", 1)
            emb_path, dump_path = paths.split(":", 1)
            Xe, index_e, enc_e = load_emb(emb_path)
            if enc_e != encoder:
                raise SystemExit(f"ABORT: {name} embedded with {enc_e}, train with {encoder}")
            tasks = arm_tasks(name, "", args.ids, args.workers)
            dump_e = load_dump(ROOT / dump_path)
            missing = [t["id"] for t in tasks if t["id"] not in dump_e]
            if missing:
                raise SystemExit(f"ABORT: {name}: {len(missing)} tasks missing from {dump_path}")
            summary, rows = score_arm(name, tasks, dump_e, probe_sigs(tasks, dump_e, args.workers), Xe, index_e,
                                      model, cv["best_lam"])
            report["arms"][name] = summary
            for r in rows:
                f.write(json.dumps({"arm": name, **r}) + "\n")
                if name != "gate" and max(NS) in r["by_n"]:
                    pooled["heldout"].append(r["by_n"][max(NS)])
            print(f"[{name}] {json.dumps(summary)}", flush=True)
    n = max(NS)
    verdict = {}
    if n in report["arms"].get("gate", {}):
        g = report["arms"]["gate"][n]
        verdict.update({"gate_combo": g["combo"], "gate_mbr": g["mbr"], "gate_ci": g["combo_vs_mbr_ci"],
                        "gate_mbr_matches_29_2": abs(g["mbr"] - 29.2) < 0.05,
                        "gate_pass": g["combo"] - g["mbr"] >= GATE_POINTS and g["combo_vs_mbr_ci"][0] > 0})
    if pooled["heldout"]:
        col = pooled["heldout"]
        diff = [c["combo"] - c["mbr"] for c in col]
        verdict.update({"heldout_tasks": len(col), "heldout_mbr": round(100 * np.mean([c["mbr"] for c in col]), 1),
                        "heldout_combo": round(100 * np.mean([c["combo"] for c in col]), 1),
                        "heldout_ci": ci(diff), "heldout_no_loss": float(np.mean(diff)) >= 0})
    if "gate_pass" in verdict and "heldout_no_loss" in verdict:
        verdict["pass"] = verdict["gate_pass"] and verdict["heldout_no_loss"]
    report["verdict"] = verdict
    report["seconds"] = round(time.perf_counter() - t0, 1)
    (ROOT / args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"[verdict] {json.dumps(verdict)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--set", choices=("train", "gate", "he", "mbpp"), required=True)
    e.add_argument("--dump", required=True, help="sa7_runtime-format samples.jsonl")
    e.add_argument("--tasks", default="", help="--set train: the e2_verifier_data tasks.jsonl")
    e.add_argument("--n", type=int, default=max(NS), help="eval sets: candidates among the first n samples")
    e.add_argument("--encoder", default="models/hf/SmolLM2-135M")
    e.add_argument("--max-tokens", type=int, default=1024)
    e.add_argument("--token-budget", type=int, default=8192, help="padded tokens per forward pass")
    e.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    e.add_argument("--vram-cap-gib", type=float, default=0.0)
    f = sub.add_parser("fit")
    f.add_argument("--train", required=True, help="embed --set train .npz")
    f.add_argument("--train-dump", required=True)
    f.add_argument("--train-tasks", required=True)
    f.add_argument("--eval", nargs="*", default=[], help="gate|he|mbpp=<embed npz>:<samples.jsonl>")
    f.add_argument("--out", required=True)
    for p in (e, f):
        p.add_argument("--ids", default="research/raw/sa_v_clean_gate.json")
        p.add_argument("--workers", type=int, default=8)
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    embed(args) if args.cmd == "embed" else fit(args)


if __name__ == "__main__":
    main()
