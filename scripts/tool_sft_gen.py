"""Tool-call SFT data in the NS bench's Python format (stubs + a handle_request() whose docstring is the user request),
never from the bench.

Two sources, both sandbox-checked by the bench builder's own recorder (bench_ns_gen.build_tool):
  gen      Gemini free tier, tool-use tasks on topics disjoint from bench_ns_gen.DOMAINS; own cache and raw file.
  convert  sa_b's xlam rows (JSON tool specs + JSON call list) rewritten as Python stubs + a handler body.
  build    check both, drop anything that shares a tool name or a request 8-gram with data/bench/ns/B_tool_call,
           mix with replay rows from data/sa_b_scoped/sa4, write data/tool_sft/{train,val}.jsonl (sa3_data schema).

  .venv/Scripts/python.exe -I scripts/tool_sft_gen.py --phase gen
  .venv/Scripts/python.exe -I scripts/tool_sft_gen.py --phase convert
  .venv/Scripts/python.exe -I scripts/tool_sft_gen.py --phase build
  .venv/Scripts/python.exe -I scripts/sa3_data.py --src data/tool_sft --out data/tool_sft/ds \
      --shapes tool_py,fix,repair,he,mbpp,replay
"""
import argparse
import ast
import json
import keyword
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from bench_ns_gen import ASKS, COMMON, DOMAINS, as_source, build_tool  # noqa: E402
from sa2_build import CHILD_RSS_CAP, MIN_FREE_RAM, parallel, ram_guard  # noqa: E402
from teacher_client import GEMINI_FREE_MODELS, ModelExhausted, Teacher, TeacherAbort  # noqa: E402

OUT_DIR = ROOT / "data" / "tool_sft"
GEN_RAW = OUT_DIR / "gen_raw.jsonl"
XLAM_RAW = OUT_DIR / "xlam_raw.jsonl"
CACHE = ROOT / "data" / "teacher" / "tool_sft_cache.jsonl"
BENCH = ROOT / "data" / "bench" / "ns" / "B_tool_call.jsonl"
SA_B = ROOT / "data" / "sa_b_scoped" / "sa4"
PER_CALL = 8
TOOL_DOMAINS = ("hotel bookings", "flight search", "smart home lights", "podcast apps", "email sending",
                "restaurant reviews", "ride hailing", "fitness tracking", "currency conversion", "stock quotes",
                "translation services", "map directions", "movie tickets", "job listings", "pet adoption",
                "plant care", "car rentals", "news feeds", "social media posts", "meeting rooms", "food delivery",
                "online courses", "support tickets", "package tracking", "museum visits", "gym classes",
                "photo albums", "password resets", "electricity usage", "dentist appointments", "book search",
                "concert tickets", "home thermostats", "crypto prices", "wine pairing", "bike sharing",
                "language flashcards", "insurance quotes", "garden watering", "conference talks")
assert not set(TOOL_DOMAINS) & set(DOMAINS)
REPLAY = {"fix": 1500, "repair": 1500, "he": 1500, "mbpp": 1500, "replay": 1000}  # per-shape train rows
TYPES = {"str": "str", "int": "int", "float": "float", "bool": "bool", "list": "list", "dict": "dict"}


# ---------------------------------------------------------------- gen: Gemini, disjoint topics

def ask(teachers: list, state: dict, rng: random.Random) -> list[dict] | None:
    topics = ", ".join(rng.sample(TOOL_DOMAINS, 3))
    prompt = (ASKS["tool_call"] % PER_CALL).format(topics=topics) + "\n" + COMMON.replace("exactly 5", f"exactly {PER_CALL}")
    while True:
        teacher = teachers[state["current"]]
        try:
            reply = teacher.chat([{"role": "user", "content": prompt}], max_tokens=6000, temperature=0.9,
                                 json_mode=True)
            break
        except ModelExhausted as error:
            print(f"NEXT MODEL: {error}", flush=True)
            state["current"] += 1
            if state["current"] == len(teachers):
                raise TeacherAbort("every free model is exhausted for today") from None
    try:
        items = json.loads(reply)["items"]
    except (ValueError, KeyError, TypeError):
        return None
    return [{"source": "gemini", "topics": topics, "model": teacher.model, **it} for it in items
            if isinstance(it, dict)]


def gen(args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    have = sum(1 for line in GEN_RAW.open(encoding="utf-8") if line.strip()) if GEN_RAW.exists() else 0
    teachers = [Teacher(m, cache_path=CACHE) for m in GEMINI_FREE_MODELS]
    state, fails = {"current": 0}, 0
    rng = random.Random(f"{args.seed}/tool_sft/{have}")
    with GEN_RAW.open("a", encoding="utf-8", newline="\n") as out:
        try:
            while have < args.gen_target and fails < 10:
                items = ask(teachers, state, rng)
                if not items:
                    fails += 1
                    continue
                for it in items:
                    out.write(json.dumps(it, ensure_ascii=False) + "\n")
                out.flush()
                have += len(items)
                print(f"[gen] {have}/{args.gen_target}", flush=True)
        except TeacherAbort as error:
            print(f"STOP: {error}", flush=True)
    print(json.dumps({"raw": have, "requests": {t.model: t.calls for t in teachers if t.calls}}), flush=True)


# ---------------------------------------------------------------- convert: xlam JSON rows -> Python stubs

def annotation(xtype: str) -> str:
    base = xtype.split(",")[0].strip()
    m = re.fullmatch(r"(?:List|list)\[(\w+)\]", base)
    if m and m.group(1).lower() in TYPES:
        return f"list[{TYPES[m.group(1).lower()]}]"
    return TYPES.get(base.split("[")[0].lower(), "")


def stub(spec: dict) -> tuple[str, list[str]] | None:
    """Python stub for one xlam tool spec; required params first (a Python rule xlam does not follow)."""
    name, params = spec.get("name", ""), spec.get("parameters") or {}
    if not name.isidentifier() or keyword.iskeyword(name) or not isinstance(params, dict):
        return None
    required, optional = [], []
    for pname, p in params.items():
        if not pname.isidentifier() or keyword.iskeyword(pname) or not isinstance(p, dict):
            return None
        ann = annotation(str(p.get("type", "")))
        head = f"{pname}: {ann}" if ann else pname
        if "optional" in str(p.get("type", "")):
            default = p.get("default", None)
            default = None if default in ("", None) else default
            optional.append(f"{head} = {default!r}")
        else:
            required.append((pname, head))
    doc = re.split(r"(?<=\.)\s", str(spec.get("description", "")).strip())[0][:110]
    doc = doc.replace("\\", "").replace('"', "'") or "Call the tool."
    sig = ", ".join([h for _, h in required] + optional)
    return f'def {name}({sig}):\n    """{doc}"""\n    ...\n', [p for p, _ in required]


def convert(args) -> None:
    rows = [json.loads(line) for line in (SA_B / "train.jsonl").open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r["shape"] == "tool" and r["source"] == "xlam"]
    random.Random(f"{args.seed}/xlam").shuffle(rows)
    why, n = Counter(), 0
    with XLAM_RAW.open("w", encoding="utf-8", newline="\n") as out:
        for r in rows:
            lines = r["prompt"].split("\n")
            try:
                specs = [json.loads(line[2:]) for line in lines if line.startswith("# {")]
                calls = json.loads(r["completion"])
            except ValueError:
                why["json"] += 1
                continue
            request = next((line[len("# Request: "):] for line in lines if line.startswith("# Request: ")), "")
            if not request or '"""' in request or "\\" in request or not 1 <= len(calls) <= 4 or len(specs) > 4:
                why["shape"] += 1
                continue
            stubs = {}
            for spec in specs:
                s = stub(spec) if isinstance(spec, dict) else None
                if s is None:
                    break
                stubs[spec["name"]] = s
            else:
                body = []
                for c in calls:
                    if not isinstance(c, dict) or c.get("name") not in stubs or not isinstance(c.get("arguments"), dict):
                        break
                    req, kw = stubs[c["name"]][1], dict(c["arguments"])
                    pos = []
                    for p in req:  # positional for required params, keywords for the rest (bench style)
                        if p not in kw:
                            break
                        pos.append(repr(kw.pop(p)))
                    body.append(f"    {c['name']}({', '.join(pos + [f'{k}={v!r}' for k, v in kw.items()])})")
                else:
                    tools = "\n\n".join(s for s, _ in stubs.values())
                    out.write(json.dumps({"source": "xlam", "src_id": r["id"], "tools": tools, "request": request,
                                          "solution": "\n".join(body) + "\n"}, ensure_ascii=False) + "\n")
                    n += 1
                    if n >= args.xlam_max:
                        break
                    continue
                why["call"] += 1
                continue
            why["stub"] += 1
    print(json.dumps({"xlam_rows": len(rows), "converted": n, "skipped": dict(why)}), flush=True)


# ---------------------------------------------------------------- build: check, decontaminate, mix

def check(pair) -> tuple[dict | None, str]:
    idx, raw = pair
    it, why = build_tool({**raw, "category": "tool_call"}, idx)
    return (it, "ok") if it is not None else (None, why)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def grams(text: str, n: int = 8) -> set:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def build(args) -> None:
    ram_guard(MIN_FREE_RAM + args.workers * CHILD_RSS_CAP)
    bench = [json.loads(line) for line in BENCH.open(encoding="utf-8") if line.strip()]
    bench_names = {n.name for b in bench for n in ast.parse(b["prompt"]).body if isinstance(n, ast.FunctionDef)}
    bench_names.discard("handle_request")
    bench_grams = set().union(*(grams(b["prompt"].split('"""User: ')[-1]) for b in bench))
    raws = []
    for path in (GEN_RAW, XLAM_RAW):
        if path.exists():
            raws += [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    why, rows, seen = Counter(), [], set()
    for (_, raw), (it, verdict) in parallel(list(enumerate(raws)), check, args.workers, "check tool"):
        src = raw.get("source", "?")
        if it is not None:
            names = {n.name for n in ast.parse(it["prompt"]).body if isinstance(n, ast.FunctionDef)}
            request = it["prompt"].split('"""User: ')[-1]
            if names & bench_names or grams(request) & bench_grams:
                verdict, it = "bench-overlap", None
            elif it["prompt"] in seen:
                verdict, it = "duplicate", None
        why[f"{src}:{verdict}"] += 1
        if it is None:
            continue
        seen.add(it["prompt"])
        rows.append({"id": f"tool_sft/{src}/{len(rows)}", "source": f"tool_sft_{src}", "shape": "tool_py",
                     "entry_point": "handle_request", "tests": it["hidden"], "support": it["support"],
                     "prompt": it["prompt"].rstrip(), "completion": "\n" + it["reference"].rstrip("\n") + "\n"})
    rng = random.Random(f"{args.seed}/split")
    rng.shuffle(rows)
    n_val = max(50, len(rows) // 33)
    split = {"train": rows[n_val:], "val": rows[:n_val]}
    for name in ("train", "val"):
        old = [json.loads(line) for line in (SA_B / f"{name}.jsonl").open(encoding="utf-8") if line.strip()]
        for shape, k in REPLAY.items():
            pool = [r for r in old if r["shape"] == shape]
            rng.shuffle(pool)
            split[name] += pool[:k if name == "train" else max(20, k // 30)]
        rng.shuffle(split[name])
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in split[name]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"verdicts": dict(why), "bench_tool_names": len(bench_names),
              "splits": {k: dict(Counter(r["shape"] for r in v)) for k, v in split.items()}}
    (OUT_DIR / "build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("gen", "convert", "build"), required=True)
    ap.add_argument("--gen-target", type=int, default=1600, help="raw Gemini items to collect (gen)")
    ap.add_argument("--xlam-max", type=int, default=7000, help="xlam rows to convert (convert)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    {"gen": gen, "convert": convert, "build": build}[args.phase](args)


if __name__ == "__main__":
    main()
