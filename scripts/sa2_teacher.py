"""SA-2 teacher tasks (research/2026-10-22_sa_plan_v2.md §2 E): multi-line docstring functions from a free model.

Each request is seeded by the concept list of one self-oss-instruct row and asks for one self-contained
task as JSON. Replies are only parsed here; sa2_build.py cleans and execution-verifies them like every
other source (load_teacher), so a wrong teacher solution is dropped, never trained on.
Resumable: ids already in data/sa2/teacher_candidates.jsonl are skipped. Free models only (teacher_client).
Models are used in --models order; when one's daily free quota is spent the next takes over, and the
run stops when all are spent (re-run after the quota resets at midnight Pacific).

  .venv/Scripts/python.exe -I scripts/sa2_teacher.py --n 5          # probe
  .venv/Scripts/python.exe -I scripts/sa2_teacher.py --n 3000
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # -I leaves the script directory off sys.path
from teacher_client import ALL_FREE_MODELS, GEMINI_FREE_MODELS, ModelExhausted, Teacher, TeacherAbort  # noqa: E402

SELFOSS = ROOT / "data" / "hf" / "sa2" / "bigcode__self-oss-instruct-sc2-exec-filter-50k" / "data" / \
    "train-00000-of-00001.parquet"
OUT = ROOT / "data" / "sa2" / "teacher_candidates.jsonl"
JSON_RE = re.compile(r"\{.*\}", re.S)
ITEM_RE = re.compile(r'^(\s*)\\"(.*)\\"(,?)[ \t]*$', re.M)
PROMPT = """Invent one self-contained Python programming task that exercises: {concepts}.

Return ONLY a JSON object with these keys:
- "instruction": 1-3 sentences describing what the function must do (no code, no examples).
- "entry_point": the function name.
- "solution": the complete function, starting with "def", with a short docstring.
- "tests": 4 to 6 strings, each a single line of the form "assert entry_point(...) == expected".

Rules: exactly one top-level function; standard library only (imports go inside the function);
no input(), print(), files, randomness or time; the body is 3 to 15 lines; tests are deterministic
and cover an edge case."""


def concepts(limit: int, seed: int) -> list[str]:
    values = pq.read_table(SELFOSS, columns=["concepts"]).column("concepts").to_pylist()
    out = []
    for value in values:
        items = value if isinstance(value, list) else str(value or "").split(",")
        text = ", ".join(" ".join(str(x).split()) for x in items if str(x).strip())[:300]
        if text:
            out.append(text)
    random.Random(seed).shuffle(out)
    return out[:limit]


def loads_lenient(text: str):
    try:
        return json.loads(text)
    except ValueError:
        # Gemini sometimes writes each tests item as \"...\" (one escape level too many); fix only the
        # outer quotes of those lines. The build execution-checks every test, so a bad repair is dropped.
        cut = text.find('"tests"')
        if cut < 0:
            raise
        return json.loads(text[:cut] + ITEM_RE.sub(r'\1"\2"\3', text[cut:]))


def parse(reply: str) -> dict | None:
    match = JSON_RE.search(reply)
    try:
        data = loads_lenient(match.group(0)) if match else None
        if not isinstance(data, dict):
            return None
        tests = [t for t in data["tests"] if isinstance(t, str)]
        return {"instruction": str(data["instruction"]), "code": str(data["solution"]),
                "entry_point": str(data["entry_point"]), "tests": tests}
    except (ValueError, KeyError, TypeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SA-2 teacher task candidates.")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--models", default=",".join(GEMINI_FREE_MODELS),
                    help="comma-separated free model ids, used in order")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ALL_FREE_MODELS]
    if bad or not models:
        ap.error(f"not on the free allowlist: {bad}")
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["id"] for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()}
    teachers = [Teacher(m) for m in models]
    current = 0
    stats = {"asked": 0, "parsed": 0, "unparsed": 0, "skipped": 0}
    with OUT.open("a", encoding="utf-8") as handle:
        for i, concept in enumerate(concepts(args.n, args.seed)):
            row_id = f"teacher/{args.seed}/{i}"
            if row_id in done:
                stats["skipped"] += 1
                continue
            reply = None
            try:
                while reply is None:
                    teacher = teachers[current]
                    try:
                        reply = teacher.chat([{"role": "user", "content": PROMPT.format(concepts=concept)}],
                                             max_tokens=4000, temperature=0.7,  # reasoning tokens count here
                                             json_mode=True)
                    except ModelExhausted as error:
                        print(f"NEXT MODEL: {error}", flush=True)
                        current += 1
                        if current == len(teachers):
                            raise TeacherAbort("every free model is exhausted for today") from None
            except TeacherAbort as error:
                print(f"STOP: {error}", flush=True)
                break
            stats["asked"] += 1
            row = parse(reply)
            if row is None:
                stats["unparsed"] += 1
                continue
            stats["parsed"] += 1
            handle.write(json.dumps({"id": row_id, **row, "concepts": concept, "model": teacher.model},
                                    ensure_ascii=False) + "\n")
            handle.flush()
            if stats["asked"] % 50 == 0:
                print(json.dumps({**stats, "model": teacher.model}), flush=True)
    calls = {t.model: t.calls for t in teachers if t.calls}
    print(json.dumps({**stats, "requests": calls}), flush=True)


if __name__ == "__main__":
    main()
