r"""Track H - agent-side memory experiment (zero training).

adoption_plan.md Track H: rolling summary + external fact store around the
fixed 1024-token context, evaluated with the two-session recall eval
(note 04 section 7 item 5) plus a single-session regression probe.

Pattern sources (VERIFIED 2026-09-08, research/raw/ver7_memgpt_mem0.json):
MemGPT (arXiv 2310.08560) virtual-context paging; Mem0 (arXiv 2504.19413)
extract / update / search memory modules.

What it does
  session 1: scripted multi-turn chat. The raw transcript window is capped
    at --raw-budget tokens inside the model ctx; overflow turns are folded
    into a rolling summary (model-summarized, deterministic tail-truncation
    fallback) capped at --summary-budget tokens. Durable facts are extracted
    from every user turn (prompted extraction validated against the turn;
    deterministic fallback) into a JSON store.
  context swap: the raw transcript is wiped (simulated app restart); only
    the rolling summary survives.
  session 2: each recall question is answered twice - control arm (summary
    only) and store arm (summary + top-k retrieved facts injected).
  regression: 4 mini_eval-style code prompts, plain vs memory preamble;
    gate: AST-parseable rate must not drop.

Gates (adoption_plan.md Track H)
  - two-session recall: >= 2 facts recalled in the store arm across the swap
  - no regression on single-session prompts (AST rate preamble >= plain)
  - token overhead of summary + injected facts reported honestly

Run (GPU window; NEVER co-run with a live train - AGENTS.md section 4):
  & .\.venv\Scripts\python.exe scripts\agent_memory_eval.py --config configs\sft_v2_e1.yaml
CPU mechanics preflight (no torch/model load, generation stubbed empty;
use small budgets so fold/compression paths actually trigger):
  & .\.venv\Scripts\python.exe scripts\agent_memory_eval.py --no-model --raw-budget 120 --summary-budget 60
"""

import argparse
import ast
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- scripted session ----------------------------------------------------

FACTS = [
    {"id": "f1", "text": "My favorite programming language is Rust."},
    {"id": "f2", "text": "Our deploy server is called falcon and it runs on port 8081."},
    {"id": "f3", "text": "My sister's name is Layla and she lives in Oslo."},
    {"id": "f4", "text": "The project deadline is March 15."},
    {"id": "f5", "text": "I use the editor Neovim with the Dracula theme."},
    {"id": "f6", "text": "The wifi password for the office network is Pineapple42."},
]

FILLERS = [
    "Write a Python function that reverses a string.",
    "Write a Python function that checks whether a number is prime.",
    "Write a Python function that reads a text file line by line.",
    "Write a Python function that sorts a list of dictionaries by a given key.",
    "Write a Python class that implements a stack with push and pop.",
    "Write a Python function that prints the FizzBuzz sequence up to n.",
]

# interleaved so the early facts leave the raw window first (turn 1-4 are the
# ones that get folded into the summary / store; late facts stay raw)
TURNS = [
    ("filler", 0), ("fact", 0), ("filler", 1), ("fact", 1), ("fact", 2),
    ("filler", 2), ("fact", 3), ("fact", 4), ("filler", 3), ("fact", 5),
    ("filler", 4), ("filler", 5),
]

RECALL = [
    {"fact": "f1", "question": "What is my favorite programming language?",
     "probe": "My favorite programming language is",
     "expect": ["rust"]},
    {"fact": "f2", "question": "What is our deploy server called and which port does it run on?",
     "probe": "Our deploy server is called",
     "expect": ["falcon", "8081"]},
    {"fact": "f3", "question": "What is my sister's name?",
     "probe": "My sister's name is",
     "expect": ["layla"]},
    {"fact": "f4", "question": "When is the project deadline?",
     "probe": "The project deadline is",
     "expect": ["march 15", "15 march"]},
    {"fact": "f5", "question": "Which code editor do I use?",
     "probe": "I use the editor",
     "expect": ["neovim"]},
    {"fact": "f6", "question": "What is the wifi password for the office network?",
     "probe": "The wifi password is",
     "expect": ["pineapple42"]},
]

PROBE_PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Write a Python function that counts the vowels in a string.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Write a Python function that merges two sorted lists into one sorted list.",
]

STOP = {"the", "a", "an", "my", "me", "i", "is", "are", "what", "which", "when",
        "and", "of", "in", "on", "for", "to", "do", "does", "you", "your", "it",
        "our", "we", "user", "says", "asks"}

# imperative openers that mark a code request, not a durable fact
IMPERATIVE = {"write", "create", "implement", "explain", "make", "build",
              "generate", "refactor", "fix", "debug", "optimize", "show"}

EXTRACT_INSTR = (
    "Extract durable facts (names, numbers, preferences, settings) from the "
    "user message below. Reply with one short line per fact. If there are no "
    "durable facts, reply NONE.\n\nUser message: {text}"
)

SUMMARY_INSTR = (
    "Summarize the important facts and events of the conversation below in at "
    "most {words} words. Keep every name, number and setting exactly as "
    "written.\n\nExisting summary:\n{old}\n\nConversation part to fold in:\n{dropped}"
)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def wordset(s):
    return set(norm(s).split())


def content_words(s):
    return {w for w in wordset(s) if w not in STOP}


def overlap(a, b):
    A, B = wordset(a), wordset(b)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def det_extract(turn_text):
    out = []
    for sent in sentences(turn_text):
        words = sent.split()
        if not (4 <= len(words) <= 30):
            continue
        if words[0].strip(".,!?").lower() in IMPERATIVE:
            continue
        has_cap = any(w[:1].isalpha() and w[0].isupper() for w in words[1:])
        if has_cap or any(ch.isdigit() for ch in sent):
            out.append(sent)
    return out


def retrieve(store, question, k):
    q = content_words(question)
    scored = []
    for f in store["facts"]:
        shared = len(q & content_words(f["text"]))
        if shared:
            scored.append((-shared, f["id"], f))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [f for _, _, f in scored[:k]]


def preamble(summary, facts=None):
    lines = ["You are continuing a conversation with the user."]
    if summary.strip():
        lines += ["", "Conversation summary so far:", summary.strip()]
    if facts:
        lines += ["", "Known facts about the user:"]
        lines += ["- " + f["text"] for f in facts]
    return "\n".join(lines)


def answer_of(text):
    text = text.split("###")[0] if "###" in text else text
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s[:200]
    return ""


def score(answer, expects):
    a = norm(answer)
    return [e for e in expects if norm(e) in a]


def ast_ok(text):
    idxs = [x for x in (text.find("def "), text.find("class ")) if x >= 0]
    if not idxs:
        return False
    code = text[min(idxs):]
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        try:
            ast.parse(code[:code.rfind("\n")])
            return True
        except (SyntaxError, ValueError):
            return False


def main():
    ap = argparse.ArgumentParser(description="Track H agent-memory recall eval")
    ap.add_argument("--config", default="configs/sft_v2_e1.yaml")
    ap.add_argument("--ckpt", default=None, help="model dir; default config final_dir")
    ap.add_argument("--store", default="data/agent_memory/store.json")
    ap.add_argument("--out", default="runs/agent_memory_h")
    ap.add_argument("--raw-budget", type=int, default=400)
    ap.add_argument("--summary-budget", type=int, default=220)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-new-reply", type=int, default=48)
    ap.add_argument("--max-new-recall", type=int, default=96)
    ap.add_argument("--probe-max-new", type=int, default=160)
    ap.add_argument("--no-model", action="store_true",
                    help="CPU mechanics preflight: no torch load, generations stubbed empty")
    ap.add_argument("--extractor", choices=("auto", "deterministic"), default="auto",
                    help="auto = prompted-first with deterministic fallback (the "
                         "Phase 1 harness, unchanged); deterministic = store facts "
                         "verbatim from the transcript, never prompted (Phase 2 "
                         "mechanism arm - the prompted extractor corrupted 2 of 3 "
                         "stored facts in the Phase 1 rerun)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    ctx = int(cfg["model"]["ctx"])
    tmpl = (cfg.get("data") or {}).get("template")
    if not tmpl:
        raise SystemExit("config needs data.template (instruct-style SFT model)")
    real = not args.no_model

    ckpt = Path(args.ckpt) if args.ckpt else ROOT / cfg["train"]["final_dir"]
    if real and not ckpt.is_dir():
        raise SystemExit(f"model dir not found: {ckpt} (train first)")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = ROOT / args.store
    store_path.parent.mkdir(parents=True, exist_ok=True)

    tok = model = None
    device = "none"
    if real:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForCausalLM.from_pretrained(str(ckpt))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        print(f"[track-h] model={ckpt} device={device} ctx={ctx}", flush=True)
    else:
        print("[track-h] --no-model mechanics preflight (generation stubbed)", flush=True)

    def toklen(s):
        if tok is not None:
            return len(tok.encode(s, add_special_tokens=False))
        return len(s.split())  # preflight approximation

    def gen(instruction, max_new):
        if not real:
            return ""
        import torch
        prompt = tmpl.format(instruction=instruction)
        room = ctx - max_new
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
        n = ids["input_ids"].shape[1]
        if n > room:
            raise SystemExit(f"[track-h] prompt {n} tokens > room {room}; budget bug")
        ids = {k: v.to(model.device) for k, v in ids.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new,
                                 pad_token_id=tok.eos_token_id, do_sample=False)
        new = out[0][ids["input_ids"].shape[1]:]
        dt = time.perf_counter() - t0
        text = tok.decode(new, skip_special_tokens=True).strip()
        last_gen[0] = {"prompt_tokens": n, "gen_seconds": round(dt, 2)}
        return text

    last_gen = [None]  # token/time bookkeeping of the most recent generation

    def extract_facts(turn_text):
        if args.extractor == "deterministic":
            # Phase 2 arm: facts stored verbatim from the turn - zero prompt
            # means zero rewrite risk (Phase 1 rerun evidence: the prompted
            # extractor corrupted "March 15" -> "ISO 15" and echoed the wifi
            # value into "the wifi password", both while passing the overlap
            # validation). The overlap check still applies to the det facts
            # downstream via the scripted match, so reporting stays honest.
            return [("deterministic", s) for s in det_extract(turn_text)]
        # prompted extraction first, validated against the turn; deterministic
        # fallback keeps the store populated when the 226M extractor fails
        raw = gen(EXTRACT_INSTR.format(text=turn_text), 40)
        cands = []
        for line in raw.splitlines():
            s = line.strip().lstrip("-*0123456789. ")
            if not s or s.upper() == "NONE" or not (3 <= len(s.split()) <= 30):
                continue
            low = s.lower()
            if '"""' in s or s.startswith(">>>") or low.startswith(("import ", "def ", "from ", "class ")):
                continue  # code-fragment echo, not a fact
            best = max((overlap(s, sent) for sent in sentences(turn_text)), default=0.0)
            if best >= 0.5:
                cands.append(("prompted", s))
        if not cands:
            for sent in det_extract(turn_text):
                cands.append(("deterministic", sent))
        return cands

    def fold_summary(summary, dropped):
        if real:
            text = gen(SUMMARY_INSTR.format(
                words=max(20, args.summary_budget // 2),
                old=(summary + "\n" if summary else ""), dropped=dropped), args.summary_budget)
            if text and toklen(text) <= args.summary_budget and overlap(text, dropped) >= 0.15:
                return text, "model"
        joined = (summary + "\n" if summary else "") + dropped
        if tok is not None:
            ids = tok.encode(joined, add_special_tokens=False)
            if len(ids) > args.summary_budget:
                ids = ids[-args.summary_budget:]  # keep the recent tail
            return tok.decode(ids, skip_special_tokens=True), "fallback-tail"
        words = joined.split()
        return " ".join(words[-args.summary_budget:]), "fallback-tail"

    # --- session 1 --------------------------------------------------------
    store = {"meta": {"created": datetime.now(timezone.utc).isoformat(),
                      "model": str(ckpt) if real else "none",
                      "config": args.config, "pattern": "rolling-summary+fact-store"},
             "facts": []}
    summary = ""
    transcript = []
    fold_events = []
    extract_stats = {"prompted": 0, "deterministic": 0}
    s1_log = []

    t_s1 = time.perf_counter()
    for idx, (kind, fi) in enumerate(TURNS, 1):
        turn = FACTS[fi]["text"] if kind == "fact" else FILLERS[fi]
        body_lines = ["You are in an ongoing conversation with the user."]
        if summary.strip():
            body_lines += ["", "Conversation summary so far:", summary.strip()]
        if transcript:
            body_lines += ["", "Recent conversation:"]
            body_lines += transcript
        body_lines += ["", "User says: " + turn]
        body = "\n".join(body_lines)
        reply = gen(body, args.max_new_reply)
        meta = last_gen[0] or {"prompt_tokens": toklen(tmpl.format(instruction=body)),
                               "gen_seconds": 0.0}
        entry = "User: " + turn + "\nAssistant: " + (reply or "(no reply)")
        transcript.append(entry)

        got = extract_facts(turn)
        stored = []
        for via, fact_text in got:
            if any(norm(fact_text) == norm(f["text"]) for f in store["facts"]):
                continue
            best_id, best_o = None, 0.0
            for fa in FACTS:
                o = overlap(fact_text, fa["text"])
                if o > best_o:
                    best_o, best_id = o, fa["id"]
            rec = {"id": "m%d" % (len(store["facts"]) + 1), "text": fact_text,
                   "turn": idx, "via": via,
                   "scripted": best_id if best_o >= 0.5 else None}
            store["facts"].append(rec)
            extract_stats[via] += 1
            stored.append(rec["id"])

        folds_this_turn = 0
        while toklen("\n".join(transcript)) > args.raw_budget and len(transcript) > 2:
            dropped = transcript.pop(0)
            summary, mode = fold_summary(summary, dropped)
            fold_events.append({"after_turn": idx, "mode": mode,
                                "dropped_tokens": toklen(dropped)})
            folds_this_turn += 1

        s1_log.append({"turn": idx, "kind": kind, "user": turn, "reply": reply,
                       "prompt_tokens": meta["prompt_tokens"],
                       "gen_seconds": meta["gen_seconds"],
                       "stored": stored, "folds_after": folds_this_turn})
        print(f"[s1] turn {idx:2d} ({kind:6s}) stored={stored or '-'} folds={folds_this_turn}", flush=True)
    s1_seconds = round(time.perf_counter() - t_s1, 1)

    raw_at_swap = toklen("\n".join(transcript))
    transcript = []  # context swap: app restart; only the summary survives
    print(f"[s1] done in {s1_seconds}s; facts={len(store['facts'])} folds={len(fold_events)} summary={toklen(summary)}tok; raw window WIPED (was {raw_at_swap}tok)", flush=True)

    # --- session 2 --------------------------------------------------------
    s2_log = []
    recall_results = {}
    t_s2 = time.perf_counter()
    for r in RECALL:
        fa = next(f for f in FACTS if f["id"] == r["fact"])
        retrieved = retrieve(store, r["question"], args.top_k)
        rec = {"question": r["question"], "expected": r["expect"],
               "fact_text": fa["text"], "arms": {}}
        for style, ask in (("qa", "The user asks: " + r["question"]),
                           ("probe", "Note about the user: " + r["probe"])):
            for arm in ("control", "store"):
                facts = retrieved if arm == "store" else None
                body = preamble(summary, facts) + "\n\n" + ask
                out = gen(body, args.max_new_recall)
                ans = answer_of(out)
                matched = score(out, r["expect"])
                inj = "- " + "\n- ".join(f["text"] for f in facts) if facts else ""
                key = style + "_" + arm
                rec["arms"][key] = {"answer": ans, "matched": matched,
                                    "pass": bool(matched),
                                    "retrieved": [f["id"] for f in (facts or [])],
                                    "preamble_tokens": toklen(tmpl.format(instruction=body)),
                                    "injected_tokens": toklen(inj) if inj else 0}
                s2_log.append({"fact": r["fact"], "style": style, "arm": arm, **rec["arms"][key]})
        recall_results[r["fact"]] = rec
        qs = rec["arms"]["qa_store"]
        ps = rec["arms"]["probe_store"]
        print(f"[s2] {r['fact']}: qa_store={'PASS' if qs['pass'] else 'fail'} {qs['matched']} | probe_store={'PASS' if ps['pass'] else 'fail'} {ps['matched']}", flush=True)
    s2_seconds = round(time.perf_counter() - t_s2, 1)

    def style_passes(style):
        s = sum(1 for r in RECALL if recall_results[r["fact"]]["arms"][style + "_store"]["pass"])
        c = sum(1 for r in RECALL if recall_results[r["fact"]]["arms"][style + "_control"]["pass"])
        return s, c
    qa_s, qa_c = style_passes("qa")
    pr_s, pr_c = style_passes("probe")
    best_style = "qa" if qa_s >= pr_s else "probe"
    best_s, best_c = (qa_s, qa_c) if qa_s >= pr_s else (pr_s, pr_c)
    gate_recall = {"required": 2, "qa_store": qa_s, "qa_control": qa_c,
                   "probe_store": pr_s, "probe_control": pr_c,
                   "style_used": best_style,
                   "store_arm_passes": best_s, "control_arm_passes": best_c,
                   "result": "PASS" if best_s >= 2 else "FAIL"}

    # --- single-session regression probe ----------------------------------
    reg = []
    plain_ast = pre_ast = 0
    t_reg = time.perf_counter()
    for p in PROBE_PROMPTS:
        row = {"prompt": p}
        for cond in ("plain", "preamble"):
            if cond == "plain":
                body = p
            else:
                body = preamble(summary, store["facts"]) + "\n\nTask: " + p
            out = gen(body, args.probe_max_new)
            ok = ast_ok(out)
            row[cond] = {"ast": ok, "chars": len(out)}
            if ok:
                if cond == "plain":
                    plain_ast += 1
                else:
                    pre_ast += 1
        reg.append(row)
    reg_seconds = round(time.perf_counter() - t_reg, 1)
    gate_reg = {"plain_ast": "%d/%d" % (plain_ast, len(PROBE_PROMPTS)),
                "preamble_ast": "%d/%d" % (pre_ast, len(PROBE_PROMPTS)),
                "result": "PASS" if pre_ast >= plain_ast else "FAIL"}

    # --- report -----------------------------------------------------------
    injected_total = sum(e["injected_tokens"] for e in s2_log)
    overhead = {
        "summary_tokens": toklen(summary),
        "facts_stored_tokens": sum(toklen("- " + f["text"]) for f in store["facts"]),
        "avg_injected_tokens_per_question": round(injected_total / max(1, 2 * len(RECALL)), 1),
        "avg_preamble_tokens_per_question": round(
            sum(e["preamble_tokens"] for e in s2_log) / max(1, len(s2_log)), 1),
        "budget_share_of_ctx_pct": round(100 * (toklen(summary) + injected_total / max(1, 2 * len(RECALL))) / ctx, 1),
    }
    scripted_covered = sorted({f["scripted"] for f in store["facts"] if f["scripted"]})

    report = {
        "track": "H", "date": datetime.now(timezone.utc).isoformat(),
        "model": str(ckpt) if real else "none", "config": args.config,
        "no_model": not real, "device": device, "ctx": ctx,
        "extractor_mode": args.extractor,
        "budgets": {"raw": args.raw_budget, "summary": args.summary_budget,
                    "top_k": args.top_k},
        "token_overhead": overhead,
        "session1": {"turns": len(TURNS), "fold_events": fold_events,
                     "summarizer_modes": sorted({e["mode"] for e in fold_events}),
                     "extractor": {**extract_stats,
                                   "scripted_facts_covered": scripted_covered,
                                   "scripted_coverage": "%d/6" % len(scripted_covered)},
                     "raw_tokens_at_swap": raw_at_swap, "seconds": s1_seconds},
        "store_path": args.store, "facts_stored": store["facts"],
        "recall": recall_results,
        "gate_recall": gate_recall,
        "regression": reg, "gate_regression": gate_reg,
        "timings": {"session1_s": s1_seconds, "session2_s": s2_seconds,
                    "regression_s": reg_seconds},
    }

    (store_path).write_text(json.dumps(store, indent=2), encoding="utf-8")
    (out_dir / "session1_transcript.jsonl").write_text(
        "\n".join(json.dumps(e) for e in s1_log) + "\n", encoding="utf-8")
    (out_dir / "session2_transcript.jsonl").write_text(
        "\n".join(json.dumps(e) for e in s2_log) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Track H result ===", flush=True)
    print(f"gate_recall     : {gate_recall['result']} (best style: {best_style}, store {best_s}/6 vs control {best_c}/6; qa {qa_s}/6, probe {pr_s}/6)", flush=True)
    print(f"gate_regression : {gate_reg['result']} (ast {gate_reg['preamble_ast']} preamble vs {gate_reg['plain_ast']} plain)", flush=True)
    print(f"overhead        : summary {overhead['summary_tokens']}tok, injected ~{overhead['avg_injected_tokens_per_question']}tok/question ({overhead['budget_share_of_ctx_pct']}% of ctx)", flush=True)
    print(f"extractor       : [{args.extractor}] {extract_stats['prompted']} prompted / {extract_stats['deterministic']} deterministic; scripted coverage {len(scripted_covered)}/6", flush=True)
    print(f"summarizer      : {sorted({e['mode'] for e in fold_events}) or 'no folds'}", flush=True)
    print(f"store           : {args.store} ({len(store['facts'])} facts)", flush=True)
    print(f"report          : {args.out}/report.json", flush=True)

    if not real:
        mech = {"facts_stored": len(store["facts"]), "scripted_coverage": len(scripted_covered),
                "fold_events": len(fold_events), "recall_pairs": len(s2_log), "probe_rows": len(reg)}
        ok = (mech["facts_stored"] >= 6 and mech["scripted_coverage"] == 6
              and mech["fold_events"] >= 1 and mech["recall_pairs"] == 12 and mech["probe_rows"] == 4)
        print(f"mechanics       : {'PASS' if ok else 'FAIL'} {mech}", flush=True)
        raise SystemExit(0 if ok else 1)

    raise SystemExit(0 if gate_recall["result"] == "PASS" and gate_reg["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
