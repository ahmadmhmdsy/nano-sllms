#!/usr/bin/env python3
r"""Batch web-research driver on top of exa_search.py (project root).

Runs a list of Exa tasks (search / answer / contents), saves every raw JSON
payload to research/raw/<slug>.json, and prints a compact digest. The default
TASKS cover the Qwen3.8-Flash-Next ("queen3.8 flash next") research; pass a
different task list with --tasks <file.json>:

    [{"kind": "search", "slug": "x", "query": "...", "num": 8, "type": "auto"},
     {"kind": "answer", "slug": "y", "query": "..."},
     {"kind": "contents", "slug": "z", "urls": ["https://..."], "max_characters": 8000}]

Usage:
    & .\.venv\Scripts\python.exe scripts\exa_research.py
    & .\.venv\Scripts\python.exe scripts\exa_research.py --tasks research\round2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exa_search import exa_answer, exa_contents, exa_search  # noqa: E402

OUT_DIR = ROOT / "research" / "raw"

DEFAULT_TASKS: List[Dict[str, Any]] = [
    {
        "kind": "search",
        "slug": "s1_qwen38_flash_next",
        "query": "Qwen3.8 Flash Next model Alibaba release",
        "num": 8,
        "type": "auto",
    },
    {
        "kind": "search",
        "slug": "s2_qwen38_term",
        "query": "Qwen3.8",
        "num": 8,
        "type": "auto",
    },
    {
        "kind": "search",
        "slug": "s3_qwen3_next_architecture",
        "query": "Qwen3-Next hybrid Gated DeltaNet gated attention architecture",
        "num": 8,
        "type": "auto",
    },
    {
        "kind": "search",
        "slug": "s4_qwen_training_pipeline",
        "query": "Qwen model training pipeline pretraining SFT reinforcement learning GRPO",
        "num": 8,
        "type": "auto",
    },
    {
        "kind": "answer",
        "slug": "a1_what_is_qwen38_flash_next",
        "query": (
            "What is Qwen3.8 Flash-Next (also written 'queen model 3.8')? How does it "
            "relate to Qwen3-Next's hybrid Gated DeltaNet + gated attention architecture, "
            "and how was it trained?"
        ),
    },
    {
        "kind": "answer",
        "slug": "a2_qwen_small_model_training",
        "query": (
            "How does Alibaba's Qwen team train their small language models (0.5B-4B)? "
            "Do they use distillation from larger Qwen flagships, and what does the full "
            "training recipe look like (pretraining, SFT, reinforcement learning)?"
        ),
    },
]


def _clean(value: Any, limit: int = 220) -> str:
    return " ".join(str(value or "").split())[:limit]


def _digest_search(payload: Dict[str, Any]) -> None:
    print(f"-> {payload.get('num_results', 0)} result(s)")
    for i, r in enumerate(payload.get("results") or [], 1):
        print(f"  {i}. {_clean(r.get('title'), 100)}")
        print(f"     {_clean(r.get('url'), 120)}")
        if r.get("published_date"):
            print(f"     published: {r['published_date']}")
        for h in (r.get("highlights") or [])[:2]:
            print(f"     >> {_clean(h, 260)}")


def _digest_answer(payload: Dict[str, Any]) -> None:
    print("-> answer:")
    print(_clean(payload.get("answer"), 3500))
    for i, c in enumerate(payload.get("citations") or [], 1):
        print(f"  [{i}] {_clean(c.get('title'), 90)} - {_clean(c.get('url'), 120)}")


def _digest_contents(payload: Dict[str, Any]) -> None:
    print(f"-> {payload.get('num_results', 0)} page(s)")
    for r in payload.get("results") or []:
        print(f"  - {_clean(r.get('title'), 100)}")
        print(f"    {_clean(r.get('url'), 120)}")
        print(f"    text: {_clean(r.get('text'), 700)}")


def _run_task(task: Dict[str, Any]) -> None:
    kind, slug = task["kind"], task["slug"]
    print(f"\n=== [{kind}] {slug} ===")
    if kind == "search":
        payload = exa_search(
            task["query"],
            num_results=int(task.get("num", 8)),
            search_type=task.get("type", "auto"),
            start_published_date=task.get("start_published_date"),
            include_domains=task.get("domains"),
            summary=bool(task.get("summary", False)),
        )
        _digest_search(payload)
    elif kind == "answer":
        payload = exa_answer(task["query"], model=task.get("model", "exa"))
        _digest_answer(payload)
    elif kind == "contents":
        payload = exa_contents(
            task["urls"],
            text=True,
            max_characters=int(task.get("max_characters", 8000)),
        )
        _digest_contents(payload)
    else:
        raise ValueError(f"unknown task kind: {kind}")
    payload["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    out_path = OUT_DIR / f"{slug}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   saved: research/raw/{slug}.json")


def _extract_texts() -> None:
    """Dump the text of every saved contents payload to research/raw/txt/<slug>.txt."""
    txt_dir = OUT_DIR / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(OUT_DIR.glob("c*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for i, result in enumerate(payload.get("results") or []):
            text = result.get("text") or ""
            suffix = "" if i == 0 else f"_{i + 1}"
            out = txt_dir / f"{path.stem}{suffix}.txt"
            out.write_text(text + f"\n\nSOURCE_URL: {result.get('url')}\n", encoding="utf-8")
            print(f"extracted: research/raw/txt/{out.name} ({len(text)} chars)")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(description="Batch Exa research driver (saves raw JSON to research/raw/).")
    parser.add_argument("--tasks", metavar="FILE", help="JSON task list; default: built-in Qwen3.8-Flash-Next round")
    parser.add_argument("--extract", action="store_true", help="dump saved contents payloads to research/raw/txt/")
    args = parser.parse_args()
    tasks = DEFAULT_TASKS
    if args.tasks:
        tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("tasks file must contain a JSON list")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for task in tasks:
        try:
            _run_task(task)
        except Exception as exc:  # keep going; record per-task failure
            failed += 1
            print(f"[FAIL] {task.get('slug')}: {type(exc).__name__}: {exc}", file=sys.stderr)
            (OUT_DIR / f"{task.get('slug')}.error.txt").write_text(
                f"{type(exc).__name__}: {exc}", encoding="utf-8"
            )
    if args.extract:
        _extract_texts()
    print(f"\nDone. {len(tasks) - failed}/{len(tasks)} tasks succeeded; raw payloads in research/raw/.")
    return 1 if failed and not (len(tasks) - failed) else 0


if __name__ == "__main__":
    sys.exit(main())
