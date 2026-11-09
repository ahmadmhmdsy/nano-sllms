"""KT-2 (ladder Phase 2, TASKS row 44): SmolLM2-360M as STANDALONE on-policy judge.

Track D V1 judge-rerank contract with a code-capable teacher. The STUDENT
(row-41 winner runs/yarn_lora_sft_v1/final_a60) samples K candidates per
instruction; the TEACHER (SmolLM2-360M, local HF-cache snapshot) scores each
candidate as TEXT - mean token logprob under its own tokenizer (1 forward
pass) plus the FROZEN row-14 JSON-verdict rubric (temp 0.2 / top_p 0.9 /
max 120, 5-step parse ladder ending in ast.literal_eval, unparseable ->
ast baseline). Winners must pass the AST gate and become SFT pairs for
sft_data.py. Knowledge flows through SCORES -> cross-tokenizer is safe.

Stages (run in order; single GPU - never co-run; each GPU stage standalone):
  prompts (CPU) -> sample (GPU student) -> score (GPU teacher) -> select (CPU)

Contract references: TASKS rows 14 (frozen judge protocol) / 32 (Track D V1) /
44 (360M-first sizing); docs/plans/2026-09-10_kt_ladder_plan.md section 5.
"""
import argparse
import ast
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # train.py pattern: script runs from scripts/
FENCE = chr(96) * 3  # code-fence marker (kept out of source literals)

RUBRIC = (
    "Instruction:\n{instruction}\n\n"
    "Candidate answer:\n{response}\n\n"
    "Check whether the answer actually satisfies the instruction and the "
    "code would run. Respond with ONLY valid JSON - double-quoted keys, "
    "no other text:\n"
    '{{"verdict": "pass", "score": 2, "reason": "one short sentence"}}\n'
    "verdict: pass (correct and complete) | partial (some issues) | "
    "fail (wrong or broken). score: 0-2.\n"
    "JSON:"
)


def jread(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def jappend(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def gpu_idle(min_free_mib: int = 4500) -> bool:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15, check=True,
    ).stdout.strip().splitlines()[0]
    used = int(re.sub(r"[^0-9]", "", out) or "0")
    ok = used <= 6000 - min_free_mib
    print(f"[gpu] used={used} MiB -> {'IDLE-OK' if ok else 'BUSY - abort'}",
          flush=True)
    return ok


def extract_code(text: str):
    """Fenced-code extraction first, whole-text fallback; ast gate."""
    blocks = re.findall(FENCE + r"(?:python|py)?\s*\n(.*?)" + FENCE,
                        text, re.DOTALL)
    cand = blocks[0].strip() if blocks else text.strip()
    if cand:
        try:
            ast.parse(cand)
            return cand, True
        except SyntaxError:
            pass
    try:
        ast.parse(text.strip())
        return text.strip(), True
    except SyntaxError:
        return cand, False


def parse_verdict(raw: str, ast_pass: bool) -> dict:
    """Row-14 frozen 5-step parse ladder; unparseable -> ast baseline."""

    def _ok(d):
        if not isinstance(d, dict):
            return None
        v = str(d.get("verdict", "")).strip().lower()
        s = d.get("score", None)
        try:
            s = int(round(float(s)))
        except (TypeError, ValueError):
            s = None
        if v not in ("pass", "partial", "fail"):
            v = {2: "pass", 1: "partial", 0: "fail"}.get(s)
        if v is None or s is None or s not in (0, 1, 2):
            return None
        return {"verdict": v, "score": s,
                "reason": str(d.get("reason", ""))[:300]}

    def _quote_bare_keys(s: str) -> str:
        s = re.sub(r"([{,]\s*)([A-Za-z_]\w*)\s*:", r'\1"\2":', s)
        return re.sub(r",\s*([}]])", r"\1", s)  # trailing commas

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    block = m.group(0) if m else None
    attempts = (
        ("json_direct", lambda: json.loads(raw.strip())),
        ("json_defence", lambda: json.loads(
            re.sub(FENCE + r"(?:json)?", "", raw).strip().strip(FENCE[:1])
            .strip())),
        ("json_block", lambda: json.loads(block)),
        ("json_repair", lambda: json.loads(_quote_bare_keys(block))),
        ("literal_block", lambda: ast.literal_eval(
            _quote_bare_keys(block) if block else block)),
    )
    for name, fn in attempts:
        if name in ("json_block", "literal_block") and block is None:
            continue
        try:
            d = _ok(fn())
            if d:
                d["parse_path"] = name
                return d
        except Exception:
            continue
    # step 5: loose key extraction
    mv = re.search(r"['\"]?verdict['\"]?\s*[:=]\s*['\"]?(\w+)", raw,
                   re.IGNORECASE)
    ms = re.search(r"['\"]?score['\"]?\s*[:=]\s*([0-2])", raw)
    if mv and ms:
        return {"verdict": mv.group(1).lower(), "score": int(ms.group(1)),
                "reason": "", "parse_path": "regex_keys"}
    # baseline: unparseable -> ast baseline (never crash a run)
    return {"verdict": "pass" if ast_pass else "fail",
            "score": 1 if ast_pass else 0,
            "reason": "ast baseline (unparseable judge)",
            "parse_path": "baseline"}


def resolve(cfg: dict, pilot: bool) -> dict:
    if not pilot:
        return cfg
    out = dict(cfg)
    out["prompts"] = dict(cfg["prompts"], n_prompts=8, n_probe=4,
                          k_candidates=4)
    out["output"] = dict(cfg["output"], dir="runs/kt2_judge_pilot",
                         pairs_out="data/sft/kt2_judge_pilot/pairs.jsonl")
    return out


def odir(out: dict) -> Path:
    p = ROOT / out["dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def synth_completion(text: str) -> str | None:
    """Text-only corpus record -> completion-prompt instruction."""
    lines = text.strip().splitlines()
    if len(lines) < 3 or len(text.strip()) < 120:
        return None
    cut = max(1, int(len(lines) * 0.6))
    while cut < len(lines) - 1 and not lines[cut].strip():
        cut += 1
    prefix = "\n".join(lines[:cut]).strip()
    if not (60 <= len(prefix) <= 900):
        return None
    return ("Complete the following Python code (continue it naturally, "
            "do not repeat the given part):\n\n" + prefix)


def stage_prompts(cfg: dict) -> None:
    p = cfg["prompts"]
    seen_d, seen_s = set(), set()
    direct, synth = [], []
    for src in p["sources"]:
        for rec in jread(ROOT / src):
            ins = str(rec.get("instruction", "")).strip()
            if ins and p["min_instruction_chars"] <= len(ins) \
                    <= p["max_instruction_chars"]:
                h = sha1(ins)
                if h not in seen_d:
                    seen_d.add(h)
                    direct.append({"id": h[:12], "instruction": ins,
                                   "source": src, "kind": "direct"})
                continue
            if "instruction" not in rec:
                s = synth_completion(str(rec.get("text", "")))
                if s:
                    h = sha1(s)
                    if h not in seen_s:
                        seen_s.add(h)
                        synth.append({"id": h[:12], "instruction": s,
                                      "source": src, "kind": "synth"})
    rng = random.Random(p["seed"])
    rng.shuffle(direct)
    rng.shuffle(synth)
    n_total = p["n_prompts"] + p["n_probe"]
    probe_from = p.get("probe_from")
    if probe_from and (ROOT / probe_from).is_file():
        # scale-round path: reuse the SAME probe set (round-over-round
        # comparability) and exclude every id consumed by earlier rounds.
        probe = list(jread(ROOT / probe_from))
        excl = {r["id"] for r in probe}
        for f in p.get("exclude_ids_files", []):
            fp = ROOT / f
            if fp.is_file():
                excl.update(r["id"] for r in jread(fp))
        n_synth = min(len(synth), int(p["n_prompts"] * p["synth_share"]))
        work = ([r for r in direct if r["id"] not in excl]
                [: p["n_prompts"] - n_synth]
                + [r for r in synth if r["id"] not in excl][:n_synth])
        rng.shuffle(work)
    else:
        n_synth = min(len(synth), int(n_total * p["synth_share"]))
        take = direct[: n_total - n_synth] + synth[:n_synth]
        rng.shuffle(take)
        probe, work = take[: p["n_probe"]], take[p["n_probe"]:]
    d = odir(cfg["output"])
    for name, rows in (("prompts.jsonl", work),
                       ("probe_instructions.jsonl", probe)):
        (d / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")
    kinds = {}
    for r in work:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[prompts] direct={len(direct)} synth={len(synth)} unique "
          f"-> work={len(work)} {kinds} probe={len(probe)} -> {d}", flush=True)


def stage_sample(cfg: dict) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import src.model  # noqa: F401 - registers our arch for native loading
    if not gpu_idle():
        sys.exit(2)
    s = cfg["student"]
    d = odir(cfg["output"])
    tok = AutoTokenizer.from_pretrained(str(ROOT / s["base"]))
    model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / s["base"]), dtype=torch.float16).cuda().eval()
    k = cfg["prompts"]["k_candidates"]
    # Resume (2026-09-11, r2 killed mid-sample): candidates.jsonl is
    # append-only; skip prompts already fully sampled by an earlier run.
    # prompts.jsonl is seed-deterministic, so ids/cand_ids line up 1:1.
    seen = set()
    cand_path = d / "candidates.jsonl"
    if cand_path.is_file():
        with open(cand_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    seen.add((r["prompt_id"], r["cand_id"]))
                except Exception:
                    print("[sample] skipping torn line", flush=True)
    resumed = 0
    t0 = time.time()
    n_cands = 0
    for i, rec in enumerate(jread(d / "prompts.jsonl"), 1):
        if all((rec["id"], c) in seen for c in range(k)):
            resumed += 1
            continue
        text = s["template"].format(instruction=rec["instruction"])
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=s["gen_ctx"] - s["max_new_tokens"]).input_ids.cuda()
        with torch.inference_mode():
            gen = model.generate(
                ids, do_sample=True, temperature=s["temperature"],
                top_p=s["top_p"], max_new_tokens=s["max_new_tokens"],
                eos_token_id=s["eos_token_id"], pad_token_id=s["pad_token_id"],
                num_return_sequences=k)
        for c in range(k):
            if (rec["id"], c) in seen:
                continue  # resume: candidate already on disk
            resp = tok.decode(gen[c][ids.shape[1]:], skip_special_tokens=True)
            code, ast_ok = extract_code(resp)
            jappend(d / "candidates.jsonl", {
                "prompt_id": rec["id"], "instruction": rec["instruction"],
                "cand_id": c, "response": resp, "code": code,
                "ast_pass": ast_ok})
            n_cands += 1
        if i % 10 == 0 or i == 1:
            print(f"[sample] {i} prompts, {n_cands} candidates, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"[sample] DONE {n_cands} candidates, resumed {resumed} prompts "
          f"in {time.time() - t0:.0f}s", flush=True)


def stage_score(cfg: dict) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not gpu_idle():
        sys.exit(2)
    t = cfg["teacher"]
    d = odir(cfg["output"])
    s = cfg["student"]
    tok = AutoTokenizer.from_pretrained(t["snapshot"])
    tok.padding_side = "left"  # decoder-only batched generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        t["snapshot"], dtype=torch.float16).cuda().eval()

    items = list(jread(d / "candidates.jsonl"))
    ptexts = [s["template"].format(instruction=r["instruction"])
              for r in items]
    fulls = tok([p + r["response"] for p, r in zip(ptexts, items)],
                add_special_tokens=True).input_ids
    fulls = [ids[:3000] for ids in fulls]
    plens = [len(tok(p).input_ids) for p in ptexts]

    def chunks(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    # pass 1: teacher LL scores, packed by padded-token budget (fp16 logits)
    ll_means, ll_totals = [0.0] * len(items), [0.0] * len(items)
    budget = 6144
    t0 = time.time()
    packed, cur, cur_max = [], [], 0
    for i, ids in enumerate(fulls):
        L = len(ids)
        new_max = max(cur_max, L)
        if cur and new_max * (len(cur) + 1) > budget:
            packed.append(cur)
            cur, cur_max = [], 0
            new_max = L
        cur.append(i)
        cur_max = new_max
    if cur:
        packed.append(cur)
    pad_id = tok.pad_token_id
    for bi, idxs in enumerate(packed, 1):
        maxlen = max(len(fulls[i]) for i in idxs)
        b_ids = torch.full((len(idxs), maxlen), pad_id, dtype=torch.long)
        att = torch.zeros((len(idxs), maxlen), dtype=torch.long)
        for j, i in enumerate(idxs):
            L = len(fulls[i])
            b_ids[j, :L] = torch.tensor(fulls[i], dtype=torch.long)
            att[j, :L] = 1
        with torch.inference_mode():
            logits = model(b_ids.cuda(), attention_mask=att.cuda()).logits
        for j, i in enumerate(idxs):
            L = len(fulls[i])
            pl = plens[i]
            n_resp = L - pl
            if n_resp > 0:
                rows = logits[j, pl - 1:L - 1].float()
                lsm = torch.log_softmax(rows, dim=-1)
                tgt = b_ids[j, pl:L].cuda()
                lp = lsm.gather(1, tgt.unsqueeze(1)).squeeze(1)
                ll_means[i] = round(lp.mean().item(), 4)
                ll_totals[i] = round(lp.sum().item(), 2)
        if bi % 10 == 0 or bi == 1:
            print(f"[score] LL {bi}/{len(packed)} batches, "
                  f"{time.time() - t0:.0f}s", flush=True)
        del logits

    # pass 2: FROZEN row-14 JSON verdicts, batch-8 left-padded generation
    rubrics = [RUBRIC.format(instruction=r["instruction"][:1500],
                             response=r["response"][:1500]) for r in items]
    verdicts = [None] * len(items)
    unparse = parsed = 0
    for gi, gidx in enumerate(chunks(list(range(len(items))), 8), 1):
        jids = tok([rubrics[i] for i in gidx], return_tensors="pt",
                   padding=True, truncation=True, max_length=1600)
        with torch.inference_mode():
            out = model.generate(
                jids.input_ids.cuda(), attention_mask=jids.attention_mask.cuda(),
                do_sample=True, temperature=t["temperature"], top_p=t["top_p"],
                max_new_tokens=t["max_new_tokens"],
                pad_token_id=tok.pad_token_id)
        gen = out[:, jids.input_ids.shape[1]:]
        for j, i in enumerate(gidx):
            v = parse_verdict(tok.decode(gen[j], skip_special_tokens=True),
                              items[i]["ast_pass"])
            verdicts[i] = v
            if v["parse_path"] == "baseline":
                unparse += 1
            else:
                parsed += 1
        n_done = gi * len(gidx)
        print(f"[score] verdicts {n_done}/{len(items)}, "
              f"{time.time() - t0:.0f}s, unparseable {unparse}/{n_done}",
              flush=True)

    for i, rec in enumerate(items):
        jappend(d / "verdicts.jsonl", {
            "prompt_id": rec["prompt_id"], "cand_id": rec["cand_id"],
            "ll_mean": ll_means[i], "ll_total": ll_totals[i],
            "ast_pass": rec["ast_pass"], **(verdicts[i] or {
                "verdict": "fail", "score": 0, "reason": "missing",
                "parse_path": "baseline"})})
    n = len(items)
    rate = unparse / max(1, n)
    gate = "PASS" if rate < cfg["gates"]["unparseable_max"] else "FAIL"
    print(f"[score] DONE {n} in {time.time() - t0:.0f}s | unparseable-rate "
          f"{rate:.1%} (gate <10%: {gate})", flush=True)
    (d / "score_report.json").write_text(json.dumps(
        {"candidates": n, "unparseable": unparse, "parsed": parsed,
         "unparseable_rate": rate, "gate_10pct": gate}, indent=2),
        encoding="utf-8")


def stage_select(cfg: dict) -> None:
    d = odir(cfg["output"])
    cands = {(r["prompt_id"], r["cand_id"]): r
             for r in jread(d / "candidates.jsonl")}
    by_prompt = {}
    for v in jread(d / "verdicts.jsonl"):
        by_prompt.setdefault(v["prompt_id"], []).append(v)
    pairs, drops = [], {"no_ast": 0, "all_fail": 0, "dup": 0, "short": 0}
    agree_t = agree_n = 0
    for pid, vs in by_prompt.items():
        pool = [v for v in vs
                if cands.get((pid, v["cand_id"]), {}).get("ast_pass")]
        if not pool:
            drops["no_ast"] += 1
            continue
        best_v = max(pool, key=lambda r: (r["score"], r["ll_mean"]))
        best_l = max(pool, key=lambda r: r["ll_mean"])
        if len(pool) >= 2 and best_v["parse_path"] != "baseline":
            agree_n += 1
            agree_t += int(best_v["cand_id"] == best_l["cand_id"])
        if best_v["score"] <= 0:
            drops["all_fail"] += 1
            continue
        rec = cands[(pid, best_v["cand_id"])]
        if len(rec["response"].strip()) < 30:
            drops["short"] += 1
            continue
        h = sha1(rec["instruction"])
        if any(p["_h"] == h for p in pairs):
            drops["dup"] += 1
            continue
        pairs.append({"_h": h, "instruction": rec["instruction"],
                      "response": rec["response"]})
    out = ROOT / cfg["output"]["pairs_out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(
        {k: p[k] for k in ("instruction", "response")},
        ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    agree = (agree_t / agree_n) if agree_n else None
    print(f"[select] winners={len(pairs)} drops={drops} | judge-vs-LL "
          f"agreement: "
          f"{'n/a' if agree is None else format(agree, '.1%')}"
          f" ({agree_t}/{agree_n})", flush=True)
    (d / "select_report.json").write_text(json.dumps(
        {"winners": len(pairs), "drops": drops,
         "verdict_ll_agreement": agree}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prompts", "sample", "score", "select"])
    ap.add_argument("--config", default="configs/kt2_judge.yaml")
    ap.add_argument("--pilot", action="store_true",
                    help="tiny pool -> runs/kt2_judge_pilot (e2e pipeline test)")
    args = ap.parse_args()
    cfg = resolve(yaml.safe_load((ROOT / args.config).read_text(
        encoding="utf-8")), args.pilot)
    {"prompts": stage_prompts, "sample": stage_sample,
     "score": stage_score, "select": stage_select}[args.stage](cfg)


if __name__ == "__main__":
    main()
