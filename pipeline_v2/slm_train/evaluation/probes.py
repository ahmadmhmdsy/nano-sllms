"""Held-out generation probes used by the native train monitor (E-005).

Stays inside pipeline_v2 (never imports V1): a self-contained syntax check
for the UI1 chain contract plus a light semantic signal (archetype match and
required-section coverage read off the TASK line vs the emitted sections).
Richer validators (ui/src) still own the FINAL external gates; a probe here
exists to observe under/overfit during training, read-only on reserved test
tasks, and never drives selection or stopping.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable


ARCH_HINTS = {
    "dashboard_metrics": ("dash", "kpi", "metric", "tile", "overview", "chart", "numbers", "spend", "count track"),
    "settings_form": ("settings", "preferences", "preferences page", "form with", "role choice", "options page"),
    "todo_list": ("list", "checklist", "add to", "tick", "chores", "task list", "queue"),
    "profile_card": ("profile", "card for", "bio", "who runs", "member page", "skills and badges"),
    "notifications_center": ("alerts", "notices", "notification", "caution", "toggle a", "warn when"),
    "pricing_cards": ("pricing", "plans", "tiers", "rate cards", "price cards", "monthly cost"),
    "search_results": ("search", "results page", "find", "lookup", "query", "hit list"),
    "orders_table": ("orders", "invoices", "purchases", "ledger", "bills", "lines with reference"),
}


def _arch_of(text: str) -> str | None:
    low = " " + text.lower() + " "
    scores = {}
    for arch, words in ARCH_HINTS.items():
        scores[arch] = sum(low.count(" " + w) for w in words)
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else None


def syntax_issues(chain: str) -> list[str]:
    """L1 structural check mirroring ui/src/chain.py chain_to_spec.

    The full validator IGNORES UI1/TASK header lines (they carry a random
    catalog id the model cannot predict); so does this probe. The contract
    that matters is ROOT/STAX/EL + PROP/BIND/COND/REPT/EVNT structure.
    """
    issues: list[str] = []
    lines = [l for l in chain.strip().splitlines() if l.strip()]
    if not lines:
        return ["empty"]
    root = None
    has_stax = False
    has_el = False
    keys: set[str] = set()
    for ln in lines:
        T = ln.split()
        op = T[0]
        if op in ("UI1", "TASK"):
            continue  # optional header lines (full validator ignores them)
        if op == "ROOT":
            if len(T) != 2:
                issues.append("bad-root-line")
            else:
                root = T[1]
        elif op == "STAX":
            try:
                parsed = json.loads(ln[5:])
                if not isinstance(parsed, dict):
                    issues.append("STAX-not-object")
                else:
                    has_stax = True
            except Exception:
                issues.append("STAX-not-json")
        elif op == "EL":
            if len(T) != 4:
                issues.append("bad-el-line")
            else:
                if T[1] in keys:
                    issues.append("duplicate-key")
                keys.add(T[1])
                has_el = True
        elif op in ("PROP", "BIND"):
            if len(T) < 2 or T[1] not in keys:
                issues.append(op.lower() + "-unknown-key")
        elif op == "COND":
            parts = ln.split(None, 3)
            if len(parts) != 4 or parts[2] != "visible" or parts[1] not in keys:
                issues.append("bad-cond-line")
        elif op == "REPT":
            parts = ln.split(None, 3)
            if len(parts) != 4 or parts[1] not in keys:
                issues.append("bad-rept-line")
        elif op == "EVNT":
            parts = ln.split(None, 3)
            if len(parts) != 4 or parts[1] not in keys:
                issues.append("bad-evnt-line")
        else:
            issues.append("bad-op")
    if root is None:
        issues.append("no-root")
    if not has_stax:
        issues.append("no-stax")
    if not has_el:
        issues.append("no-el")
    return issues


def probe_result(gens: "Iterable[dict[str, Any]]") -> dict[str, Any]:
    """Aggregate one probe batch: rows of {task, arch, chain, eos_stopped}."""
    rows = list(gens)
    n = len(rows)
    if not n:
        return {"valid": 0.0, "n": 0}
    valid = 0
    eos_ok = 0
    arch_hit = 0
    lens: list[int] = []
    for r in rows:
        chain = str(r.get("chain", ""))
        if not syntax_issues(chain):
            valid += 1
        if r.get("eos_stopped") is True or "</s>" in chain or chain.strip().endswith("</s>"):
            eos_ok += 1
        lens.append(len(chain))
        got = _arch_of(str(r.get("chain", ""))[:400] if r.get("chain") else str(r.get("task", "")))
        want = r.get("arch")
        if got and want and got == want:
            arch_hit += 1
    return {
        "valid": round(valid / n, 4),
        "eos_rate": round(eos_ok / n, 4),
        "arch_match": round(arch_hit / n, 4),
        "mean_len": round(sum(lens) / n, 1),
        "n": n,
    }


def pass_k_result(rows: "Iterable[dict[str, Any]]", sampled_chains: "list[list[str]]") -> dict[str, Any]:
    """pass@k: fraction of tasks where the greedy chain OR any of k sampled
    chains is structurally valid. Complements probe_result (greedy metrics).

    rows: probe rows (each with a "chain" = greedy). sampled_chains: per task,
    the list of sampled chains (length k). The greedy chain counts as one of
    the k candidates, so pass@k >= greedy valid by construction.
    """
    rows = list(rows)
    n = len(rows)
    if not n:
        return {"pass_k": 0.0}
    hits = 0
    for row, chains in zip(rows, sampled_chains):
        greedy_ok = not syntax_issues(str(row.get("chain", "")))
        sampled_ok = any(not syntax_issues(str(c)) for c in chains)
        hits += bool(greedy_ok or sampled_ok)
    return {"pass_k": round(hits / n, 4)}


def load_probe_tasks(path: str, count: int, seed: int = 42) -> "list[dict[str, Any]]":
    """Deterministic read-only slice of a reserved TASKS jsonl file."""
    import random

    tasks = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks[: int(count)] if count else tasks


def make_task_probe(generate_fn: Callable[[str], str], tasks: "list[dict[str, Any]]") -> Callable[[], dict[str, Any]]:
    """Bind a generate-one-chain callable to probe rows. Reusable by arms."""

    def _probe() -> dict[str, Any]:
        return probe_result({"task": t.get("task", ""), "arch": t.get("arch"), "chain": generate(t)} for t in tasks for generate in (generate_fn,))

    return _probe
