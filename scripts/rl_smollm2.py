"""SA-6 (PLAN.md section 11.3): RLOO reinforcement learning for SmolLM2-135M + LoRA on the playground.

Port of glm-5.3-flash-from-scratch/scripts/train_rl.py to a HF causal LM. The reward
(`reward_for`: sandboxed evaluator, case-fraction or binary, invalid penalty) and the advantage
(`leave_one_out`) are imported from that script, not copied, so the two lines score identically.
What changes: rollouts come from `model.generate`, log-probabilities from HF logits, and only a
fresh LoRA adapter (default r16 on q/k/v/o, same as the SFT recipe) is trained on top of a merged
`final` dir.

Tasks use the playground `rl` split (seed 6733), which is disjoint from `dev` (ladder, during
runs) and `final` (gates only). RL needs pass@k > 0 on a task to give it any signal, so start from
the best SFT/RFT checkpoint, not the base.

Safety: --vram-cap-gib is a hard allocator ceiling (no shared-memory spill); the free-RAM guard
aborts before host RAM gets tight. Auto-resume: re-run the same command and it continues from
<output>/state.pt (adapter + optimizer + scaler + step index + task cursor).

Anti-collapse (SA-6 run 2, after run 1 collapsed to one-line `return` bodies):
  --kl-beta B          adds B * KL(policy || init) to the loss. The reference is the init itself
                       (the LoRA adapter disabled), and KL uses the k3 per-token estimator on the
                       sampled tokens.
  --checkpoint-every N writes a merged <output>/checkpoint-G every N steps and, with
                       --probe-limit K > 0, runs a greedy HumanEval/MBPP pass@1 probe on the first K
                       tasks of each set (scripts/bench_passk.py subprocess, training paused, its own
                       VRAM cap = fresh free - 0.75 GiB). Results go to <output>/probe_log.jsonl.

Task distribution (SA-6 run 3: KL alone only traded MBPP for HumanEval; the bodies stayed one-line):
  --docstring-tasks N  adds N extra rl-split instances of every multi-line `write` sub-skill,
                       rendered HumanEval-style: `def` header, then a docstring holding the
                       description and one `>>>` example, and the completion is the indented body.
                       The example is an extra case that is not graded, so hardcoding it earns
                       nothing. With 0 (the default), the pool is exactly what run 2 used.
  --mbpp-tasks N       (run 4: ~190/427 MBPP misses never define the function, mostly a
                       `def test_...` instead) adds N rl-split instances of every `write`
                       sub-skill, rendered as bench_passk's MBPP prompt: a module docstring with
                       the description and one `assert` example (not graded). The completion
                       must write the whole `def`; a test-only answer is invalid.

SA-6 run 5 (research/2026-10-26_sa6_rl_proposal.md). Every flag defaults to the run-4 behaviour:
  --source sa5         train on the SA-5 task pool (scripts/sa5_rft.py `pools`, train split) with its
                       scorer (`score`): code = SA-2 he/mbpp, fix = SA-4 fix/repair, tool = SA-4 xLAM.
                       Reward 1 if the sample passes, --invalid-penalty if it never stops, else 0.
                       Each step draws shapes by --mix; each shape reads its pool in order from slice
                       --slice-start (sizes --slice-code/fix/tool, so 1 = the SA-5 round-1 tasks) and
                       wraps at the end. Code/fix samples generate --max-new-tokens, tool samples
                       --max-new-tool. When the scorer cut a code sample, only the tokens of the graded
                       prefix are trained on.
  --prompts-per-step P generates P prompts x --group-size in one batch per max-new length and takes
                       one optimizer step over all non-flat groups (--groups counts steps).
  --loss-norm const    Dr. GRPO: a sample's log-prob sum is divided by its max-new length, not by its
                       own length (per-length division favours short answers). KL stays a token mean.
  --dynamic-sampling   DAPO: flat groups (all rewards equal) are dropped and refilled with fresh
                       prompts, up to 3 refill rounds per step.
  --stop-entropy F / --stop-length F
                       stop cleanly (state, final, stopped.json) once the mean of the last 50 steps'
                       sampled-token NLL (entropy estimate) or completion length falls below F x the
                       mean of the first 50 (needs >= 100 steps; 0 = off).

  & .\\.venv\\Scripts\\python.exe scripts\\rl_smollm2.py --init runs/sft_smollm2_cpt_pg/final `
      --output runs/rl_smollm2 --groups 2000 --vram-cap-gib 4.5
  & .\\.venv\\Scripts\\python.exe scripts\\rl_smollm2.py --init ... --output scratch/rl_dry --dry-run
  & .\\.venv\\Scripts\\python.exe scripts\\rl_smollm2.py --init runs/sa4_scale075 --output runs/rl_sa6r5 `
      --source sa5 --max-new-tokens 384 --reward-mode binary --kl-beta 0.05 --prompts-per-step 8 `
      --loss-norm const --dynamic-sampling --vram-cap-gib <free - 0.75>
"""
from __future__ import annotations

import argparse
import builtins
import copy
import itertools
import json
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
NEST = ROOT / "glm-5.3-flash-from-scratch"
sys.path.insert(0, str(NEST))
sys.path.insert(0, str(NEST / "scripts"))
from glm53_flash.evaluator import OUTCOME_TIER  # noqa: E402
from glm53_flash.playground import POOL_TIERS, make_playground_task, playground_pool  # noqa: E402
from glm53_flash.sandbox import SandboxRunner  # noqa: E402
from train_rl import leave_one_out, reward_for  # noqa: E402

MIN_FREE_RAM_GIB = 8.0
SA5_SHAPES = ("code", "fix", "tool")
MAX_REFILLS = 3
STOP_WINDOW = 50


DOCSTRING_INDEX_OFFSET = 500  # docstring instances never reuse a comment-format instance's cases


def docstring_task(sub_skill, index: int):
    """A HumanEval-shaped instance: case 0 becomes the docstring example; cases 1-3 are graded."""
    base = make_playground_task(sub_skill, split="rl", index=DOCSTRING_INDEX_OFFSET + index, cases=4)
    (example_args, example_value), graded = base.cases[0], base.cases[1:]
    description = base.prompt.split("\n")[1].removeprefix("# ")
    call = f"{base.entry_point}({', '.join(repr(arg) for arg in example_args)})"
    prompt = (f"def {base.entry_point}({sub_skill.arguments}):\n"
              f'    """{description}\n\n    >>> {call}\n    {example_value!r}\n    """')
    # No trailing "\n": the SmolLM2 tokenizer merges "\n" with the next indent, so a prompt that
    # ends in a bare newline makes the model emit EOS at once (same rule as bench_passk.model_prompt).
    return replace(base, task_id=base.task_id.replace("rl-", "rl-doc-", 1), prompt=prompt,
                   reference_completion=sub_skill.body, cases=graded)


MBPP_INDEX_OFFSET = 900  # nor a docstring instance's


def mbpp_task(sub_skill, index: int):
    """An MBPP-shaped instance: a module docstring holding the description and case 0 as an
    `assert`; the completion must write the whole `def`. Cases 1-3 are graded."""
    base = make_playground_task(sub_skill, split="rl", index=MBPP_INDEX_OFFSET + index, cases=4)
    # MBPP names read like `multiply_int`, not `construct_name_suffix` (natural names doubled the
    # def rate in CPU samples). Keep the suffixed name when the bare one is a builtin or is used
    # inside the body (e.g. `math.factorial`).
    name = sub_skill.name
    if name in dir(builtins) or re.search(rf"\b{name}\b", sub_skill.body):
        name = base.entry_point
    base = replace(base, entry_point=name)
    (example_args, example_value), graded = base.cases[0], base.cases[1:]
    description = base.prompt.split("\n")[1].removeprefix("# ")
    call = f"{base.entry_point}({', '.join(repr(arg) for arg in example_args)})"
    # Same text bench_passk feeds the model for MBPP (its prompt, rstripped): ends at `"""`.
    # MBPP descriptions open "Write a function to ..."; without that cue run 3 final defined the
    # function in 1/64 CPU samples.
    prompt = (f'"""\nWrite a function to {description[0].lower()}{description[1:]}\n'
              f'assert {call} == {example_value!r}\n"""')
    body = sub_skill.body if sub_skill.body.startswith("\n") else "\n" + sub_skill.body
    return replace(base, task_id=base.task_id.replace("rl-", "rl-mbpp-", 1), prompt=prompt,
                   reference_completion=f"\ndef {base.entry_point}({sub_skill.arguments}):{body}",
                   cases=graded)


def build_tasks(tier: str, per_sub_skill: int, docstring_tasks: int = 0, mbpp_tasks: int = 0) -> list[tuple]:
    entries = []
    for sub_skill in playground_pool(tier):
        for index in range(per_sub_skill):
            entries.append((make_playground_task(sub_skill, split="rl", index=index),
                            sub_skill.capabilities()))
        if sub_skill.kind == "write" and sub_skill.body.strip("\n").count("\n") >= 1:
            for index in range(docstring_tasks):
                entries.append((docstring_task(sub_skill, index), sub_skill.capabilities()))
        if sub_skill.kind == "write":
            for index in range(mbpp_tasks):
                entries.append((mbpp_task(sub_skill, index), sub_skill.capabilities()))
    return entries


def schedule(count: int, groups: int, seed: int) -> list[int]:
    order, epoch = [], 0
    while len(order) < groups:
        block = list(range(count))
        random.Random(seed + epoch * 1009).shuffle(block)
        order.extend(block)
        epoch += 1
    return order[:groups]


def playground_stream(tasks: list[tuple], seed: int):
    """Endless `schedule` order: epoch e is the same shuffled block schedule() would give it."""
    for epoch in itertools.count():
        for index in schedule(len(tasks), len(tasks), seed + epoch * 1009):
            yield tasks[index]


def load_sa5(args):
    """sa5_rft (imported, not copied) and its train pools."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import sa5_rft
    ns = SimpleNamespace(sa2=args.sa2, sa4=args.sa4, split="train", min_asserts=args.min_asserts,
                         seed=args.pool_seed)
    return sa5_rft, sa5_rft.pools(ns)


def sa5_stream(pool: dict[str, list[dict]], args):
    """(shape, row) draws: the shape by --mix weights, each shape's rows in pool order from its slice."""
    rng = random.Random(f"{args.pool_seed}:mix")
    used = dict.fromkeys(SA5_SHAPES, 0)
    starts = {name: (args.slice_start - 1) * getattr(args, f"slice_{name}") for name in SA5_SHAPES}
    while True:
        name = rng.choices(SA5_SHAPES, weights=args.mix_weights)[0]
        rows = pool[name]
        yield name, rows[(starts[name] + used[name]) % len(rows)]
        used[name] += 1


def sa5_reward(result: dict, invalid_penalty: float) -> float:
    return 1.0 if result["passed"] else invalid_penalty if result["why"] == "nostop" else 0.0


def encode(tokenizer, text: str) -> list[int]:
    """Same encoding as the battery's HfArm (scratch/bench_playground_batched.py)."""
    ids = list(tokenizer.encode(text))
    if tokenizer.bos_token_id is not None and (not ids or ids[0] != tokenizer.bos_token_id):
        ids = [tokenizer.bos_token_id] + ids
    return ids


def cut_at_eos(ids: list[int], eos_id: int) -> list[int]:
    """Keep the first eos (the model must learn to stop); drop everything after it."""
    for position, token in enumerate(ids):
        if token == eos_id:
            return ids[:position + 1]
    return ids


def token_prefix(tokenizer, ids: list[int], chars: int) -> list[int]:
    """Shortest prefix of `ids` whose decoded text reaches `chars` characters (a cut sample's graded part)."""
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=True)) >= chars:
            hi = mid
        else:
            lo = mid + 1
    return ids[:lo]


def completion_log_probs(model, prompt: list[int], completions: list[list[int]], pad_id: int,
                         device, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities of each completion and their mask, one forward pass per group.
    `temperature` divides the logits so the scored policy is the one the rollouts were sampled from."""
    width = max(map(len, completions))
    rows = [prompt + c + [pad_id] * (width - len(c)) for c in completions]
    sequence = torch.tensor(rows, dtype=torch.long, device=device)
    lengths = torch.tensor([len(c) for c in completions], device=device)
    mask = torch.arange(width, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    attention = torch.cat([torch.ones(len(rows), len(prompt), dtype=torch.long, device=device),
                           mask.long()], dim=1)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        # Only the last `width` positions predict completion tokens; skip the prompt's vocab logits.
        logits = model(input_ids=sequence[:, :-1], attention_mask=attention[:, :-1],
                       logits_to_keep=width).logits
    selected = logits.float()
    if temperature != 1.0:
        selected.div_(temperature)  # in place: a full-vocab copy is ~0.6 GiB at 8 x 384
    targets = sequence[:, len(prompt):]
    token_lp = torch.log_softmax(selected, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_lp, mask


def group_objective(model, group: dict, rows: int, loss_norm: str, kl_beta: float, pad_id: int,
                    device, temperature: float = 1.0) -> tuple[torch.Tensor, float | None, float, int]:
    """One group's share of the step loss: its rows' terms summed, divided by `rows` (all rows trained
    this step), so one group alone gives the run-4 mean. Returns (objective, kl, nll sum, tokens)."""
    ref_lp = None
    if kl_beta > 0:  # reference first, so its logits are freed before the policy graph exists
        with torch.no_grad(), model.disable_adapter():
            ref_lp, _ = completion_log_probs(model, group["prompt"], group["train"], pad_id, device, temperature)
    token_lp, mask = completion_log_probs(model, group["prompt"], group["train"], pad_id, device, temperature)
    lengths = mask.sum(dim=1)
    norm = lengths if loss_norm == "seq" else torch.full_like(lengths, group["max_new"])
    adv = torch.tensor(group["advantages"], device=device, dtype=torch.float32)
    objective = -(adv * (token_lp * mask).sum(dim=1) / norm).sum() / rows
    kl_value = None
    if ref_lp is not None:
        delta = ref_lp - token_lp  # k3: exp(d) - d - 1 >= 0, unbiased for KL(pi || ref)
        kl = (((torch.exp(delta) - delta - 1) * mask).sum(dim=1) / lengths).sum() / rows
        objective = objective + kl_beta * kl
        kl_value = float(kl.detach())
    nll = float(-(token_lp.detach() * mask).sum())
    return objective, kl_value, nll, int(mask.sum())


def micro_batches(group: dict, size: int):
    """Split a group's rows into chunks of `size` (0 = whole group). Every row term is divided by the
    step's total rows, so the chunks' objectives sum to the whole group's: same gradient, lower peak VRAM."""
    if size <= 0 or size >= len(group["train"]):
        yield group
        return
    for start in range(0, len(group["train"]), size):
        yield group | {"train": group["train"][start:start + size],
                       "advantages": group["advantages"][start:start + size]}


def probe(checkpoint: Path, limit: int, device) -> dict:
    """Greedy pass@1 on the first `limit` HumanEval / MBPP-sanitized tasks (training is paused)."""
    free = torch.cuda.mem_get_info(device)[0] / 2 ** 30 if device.type == "cuda" else 0.0
    cap = round(free - 0.75, 2)
    if device.type == "cuda" and cap < 1.0:
        return {"skipped": f"free VRAM {free:.2f} GiB too low for a probe"}
    out = checkpoint / "probe_passk.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "bench_passk.py"), "--arm", f"ckpt={checkpoint}",
           "--limit", str(limit), "--device", device.type, "--batch", "16", "--workers", "4",
           "--out", str(out)] + (["--vram-cap-gib", str(cap)] if device.type == "cuda" else [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0 or not out.is_file():
        return {"error": (result.stderr or result.stdout)[-400:]}
    per_set = json.loads(out.read_text(encoding="utf-8"))["arms"]["ckpt"]["per_set"]
    return {name: {"passed": row["passed"], "tasks": row["tasks"], "eos_rate": row["eos_rate"]}
            for name, row in per_set.items()} | {"vram_cap_gib": cap}


def ram_guard(floor: float = MIN_FREE_RAM_GIB) -> None:
    free = psutil.virtual_memory().available / 2 ** 30
    if free < floor:
        raise SystemExit(f"free RAM {free:.1f} GiB < {floor} GiB guard; aborting")


def collapsed(history: list[float], factor: float) -> bool:
    if factor <= 0 or len(history) < 2 * STOP_WINDOW:
        return False
    first, last = history[:STOP_WINDOW], history[-STOP_WINDOW:]
    return sum(last) / STOP_WINDOW < factor * sum(first) / STOP_WINDOW


def sa5_dry_run(args) -> int:
    """References of the start slice must all pass; half references must score as the scorer says."""
    sa5, pool = load_sa5(args)
    tasks = sa5.round_slice(pool, SimpleNamespace(code=args.slice_code, fix=args.slice_fix,
                                                  tool=args.slice_tool, round=args.slice_start))
    with ThreadPoolExecutor(args.workers) as executor:
        full = list(executor.map(lambda r: sa5.score(r, r["completion"], True), tasks))
        half = list(executor.map(lambda r: sa5.score(r, r["completion"][:len(r["completion"]) // 2], False),
                                 tasks))
    bad = [(row["id"], result["why"]) for row, result in zip(tasks, full) if not result["passed"]]
    rewards = {}
    for row, result in zip(tasks, half):
        key = (row["shape"], result["why"], sa5_reward(result, args.invalid_penalty))
        rewards[key] = rewards.get(key, 0) + 1
    tool_ok = all(result["why"] == "nostop" for row, result in zip(tasks, half) if row["shape"] == "tool")
    mix = dict.fromkeys(SA5_SHAPES, 0)
    for name, _ in itertools.islice(sa5_stream(pool, args), len(tasks)):
        mix[name] += 1
    print(f"[dry-run] sa5 pools {({k: len(v) for k, v in pool.items()})}; slice {args.slice_start}: "
          f"{len(tasks)} tasks; reference fails: {len(bad)} {bad[:5]}")
    print(f"[dry-run] half references, eos=False -> (shape, why, reward): {dict(sorted(rewards.items()))}")
    print(f"[dry-run] tool halves all nostop: {tool_ok}; first {len(tasks)} mix draws: {mix}")
    return 1 if bad or not tool_ok else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="merged HF final dir to start from")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=2000, help="optimizer steps")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--tier", choices=POOL_TIERS, default="all")  # SA-B: + breadth/wide/fix/tool
    ap.add_argument("--per-sub-skill", type=int, default=8, help="rl-split instances per sub-skill")
    ap.add_argument("--docstring-tasks", type=int, default=0,
                    help="HumanEval-style instances per multi-line write sub-skill (0 = off)")
    ap.add_argument("--mbpp-tasks", type=int, default=0,
                    help="MBPP-style instances per write sub-skill (0 = off)")
    ap.add_argument("--source", choices=("playground", "sa5"), default="playground")
    ap.add_argument("--sa2", default="data/sa2")
    ap.add_argument("--sa4", default="data/sa4")
    ap.add_argument("--min-asserts", type=int, default=3)
    ap.add_argument("--pool-seed", type=int, default=5, help="sa5_rft pool shuffle seed (its --seed)")
    ap.add_argument("--slice-code", type=int, default=300)
    ap.add_argument("--slice-fix", type=int, default=500)
    ap.add_argument("--slice-tool", type=int, default=300)
    ap.add_argument("--slice-start", type=int, default=1, help="1-based sa5 slice to start each shape at")
    ap.add_argument("--mix", default="0.4,0.35,0.25", help="sa5 code,fix,tool draw weights")
    ap.add_argument("--max-new-tool", type=int, default=256, help="sa5 tool samples")
    ap.add_argument("--workers", type=int, default=4, help="sa5 scorer threads (sandboxed subprocesses)")
    ap.add_argument("--prompts-per-step", type=int, default=1)
    ap.add_argument("--micro-rows", type=int, default=0,
                    help="rows per forward/backward chunk of a group (0 = whole group); same gradient")
    ap.add_argument("--loss-norm", choices=("seq", "const"), default="seq")
    ap.add_argument("--dynamic-sampling", action="store_true")
    ap.add_argument("--max-refills", type=int, default=MAX_REFILLS,
                    help="dynamic-sampling refill rounds per step (0 = none)")
    ap.add_argument("--stop-entropy", type=float, default=0.0)
    ap.add_argument("--stop-length", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--logprob-temperature", choices=("auto", "on", "off"), default="auto",
                    help="score log-probs at logits/--temperature (on-policy); auto = on for sa5, off for "
                         "playground so runs 1-4 replay unchanged")
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--reward-mode", choices=("case-fraction", "binary"), default="case-fraction")
    ap.add_argument("--invalid-penalty", type=float, default=-0.1)
    ap.add_argument("--exact-bonus", type=float, default=0.0)
    ap.add_argument("--brevity-weight", type=float, default=0.0)
    ap.add_argument("--policy", choices=("rung", "outcome"), default="rung",
                    help="rung: each sub-skill's own capability tier; outcome: OUTCOME_TIER for every task, "
                         "so the reward grades only the result, not the solution's style")
    ap.add_argument("--sandbox", choices=("off", "on"), default="on")
    ap.add_argument("--sandbox-timeout", type=float, default=2.0)
    ap.add_argument("--save-every", type=int, default=100, help="steps between state saves")
    ap.add_argument("--min-free-ram-gib", type=float, default=MIN_FREE_RAM_GIB,
                    help="abort when available host RAM drops below this")
    ap.add_argument("--kl-beta", type=float, default=0.0, help="KL(policy || init) weight, 0 = off")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="steps between merged checkpoints")
    ap.add_argument("--probe-limit", type=int, default=0, help="HumanEval/MBPP tasks per set per probe")
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--vram-cap-gib", type=float, default=0.0)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="build tasks + score references, no model")
    args = ap.parse_args()
    if args.group_size < 2:
        raise SystemExit("--group-size must be >= 2 (leave-one-out baseline)")
    if args.prompts_per_step < 1:
        raise SystemExit("--prompts-per-step must be >= 1")
    args.mix_weights = [float(w) for w in args.mix.split(",")]
    if len(args.mix_weights) != 3 or min(args.mix_weights) < 0 or sum(args.mix_weights) <= 0:
        raise SystemExit("--mix needs three non-negative code,fix,tool weights")
    if args.slice_start < 1:
        raise SystemExit("--slice-start is 1-based")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    tempered = args.logprob_temperature == "on" or (args.logprob_temperature == "auto" and args.source == "sa5")
    args.logprob_temp = args.temperature if tempered else 1.0

    if args.source == "sa5":
        if args.dry_run:
            return sa5_dry_run(args)
        sa5, pool = load_sa5(args)
        stream, runner = sa5_stream(pool, args), None
    else:
        tasks = build_tasks(args.tier, args.per_sub_skill, args.docstring_tasks, args.mbpp_tasks)
        if args.policy == "outcome":
            tasks = [(task, OUTCOME_TIER) for task, _ in tasks]
        runner = SandboxRunner(timeout=args.sandbox_timeout) if args.sandbox == "on" else None
        if args.dry_run:  # the reward must give the reference full marks, or RL chases noise
            bad = []
            for task, caps in tasks:
                reward, ev = reward_for(task, task.reference_completion, args.reward_mode,
                                        invalid_penalty=args.invalid_penalty, exact_bonus=args.exact_bonus,
                                        capabilities=caps, runner=runner)
                if not ev["passed"]:
                    bad.append((task.task_id, ev["status"]))
            print(f"[dry-run] {len(tasks)} rl-split tasks; reference fails: {len(bad)} {bad[:5]}")
            return 1 if bad else 0
        stream = playground_stream(tasks, args.seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    if device.type == "cuda" and args.vram_cap_gib > 0:
        total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gib / total), device)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.init)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    base = AutoModelForCausalLM.from_pretrained(args.init, dtype=torch.float32,
                                                attn_implementation="sdpa").to(device)
    model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                            bias="none", task_type="CAUSAL_LM"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.output.mkdir(parents=True, exist_ok=True)
    state_path, log_path = args.output / "state.pt", args.output / "rl_log.jsonl"
    probe_path = args.output / "probe_log.jsonl"
    start, cursor, history = 0, 0, {"entropy": [], "length": []}
    if state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        saved_lr = optimizer.param_groups[0]["lr"]
        for group in optimizer.param_groups:  # load_state_dict restores the saved lr; --learning-rate wins
            group["lr"] = args.learning_rate
        if saved_lr != args.learning_rate:
            print(f"[resume] lr {saved_lr:g} -> {args.learning_rate:g}", flush=True)
        scaler.load_state_dict(state["scaler"])
        start = state["group"]
        cursor = state.get("cursor", start)  # run 1-4 states: one task per step
        history = state.get("history", history)
        print(f"[resume] from step {start}, task {cursor}", flush=True)
    for _ in range(cursor):
        next(stream)

    def make_task(item) -> dict:
        if args.source == "sa5":
            shape, row = item
            return {"id": row["id"], "bucket": shape, "row": row, "prompt": tokenizer(row["prompt"]).input_ids,
                    "max_new": args.max_new_tool if shape == "tool" else args.max_new_tokens}
        task, caps = item
        return {"id": task.task_id, "bucket": "playground", "task": task, "caps": caps,
                "prompt": encode(tokenizer, task.prompt), "max_new": args.max_new_tokens}

    def rollout(batch: list[dict]) -> None:
        """Sample group_size completions per task; tasks sharing a max-new length share one generate call."""
        by_length = {}
        for item in batch:
            by_length.setdefault(item["max_new"], []).append(item)
        model.eval()
        for max_new, chunk in by_length.items():
            width = max(len(item["prompt"]) for item in chunk)
            ids = [[pad_id] * (width - len(item["prompt"])) + item["prompt"] for item in chunk]
            attention = [[0] * (width - len(item["prompt"])) + [1] * len(item["prompt"]) for item in chunk]
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                 enabled=device.type == "cuda"):
                out = model.generate(torch.tensor(ids, device=device),
                                     attention_mask=torch.tensor(attention, dtype=torch.long, device=device),
                                     do_sample=True, temperature=args.temperature, top_k=0,
                                     max_new_tokens=max_new,
                                     num_return_sequences=args.group_size,
                                     eos_token_id=eos_id, pad_token_id=pad_id)
            for slot, item in enumerate(chunk):  # num_return_sequences keeps a prompt's samples adjacent
                rows = out[slot * args.group_size:(slot + 1) * args.group_size]
                item["completions"] = [cut_at_eos(row[width:].tolist(), eos_id) or [eos_id] for row in rows]

    def score_batch(batch: list[dict], executor) -> None:
        """Rewards, statuses, advantages, and the tokens each sample is trained on."""
        if args.source == "playground":
            for item in batch:
                item["rewards"], item["statuses"] = [], []
                for ids in item["completions"]:
                    text = tokenizer.decode([t for t in ids if t != eos_id])
                    reward, ev = reward_for(item["task"], text, args.reward_mode,
                                            invalid_penalty=args.invalid_penalty, exact_bonus=args.exact_bonus,
                                            brevity_weight=args.brevity_weight, capabilities=item["caps"],
                                            runner=runner)
                    item["rewards"].append(reward)
                    item["statuses"].append(ev["status"] if not ev["passed"] else "passed")
                item["train"] = item["completions"]
                item["advantages"] = leave_one_out(item["rewards"])
            return
        jobs = [(item["row"], tokenizer.decode([t for t in ids if t != eos_id], skip_special_tokens=True),
                 ids[-1] == eos_id) for item in batch for ids in item["completions"]]
        results = iter(list(executor.map(lambda job: sa5.score(*job), jobs)))
        raws = iter(jobs)
        for item in batch:
            item["rewards"], item["statuses"], item["train"] = [], [], []
            for ids in item["completions"]:
                result, (_, raw, _) = next(results), next(raws)
                item["statuses"].append("passed" if result["passed"] else result["why"])
                if result["why"] == sa5.NOT_EVALUATED:
                    continue  # a kill the retry overturned: no verdict, so out of the RLOO group
                item["rewards"].append(sa5_reward(result, args.invalid_penalty))
                graded = result["completion"].rstrip()
                cut = item["bucket"] != "tool" and graded != raw.rstrip() and graded
                item["train"].append(token_prefix(tokenizer, ids, len(graded)) if cut else ids)
            rewards = item["rewards"]
            item["advantages"] = leave_one_out(rewards) if len(rewards) > 1 else [0.0] * len(rewards)

    def save(step: int) -> None:
        adapter = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        tmp = state_path.with_suffix(".tmp")
        torch.save({"adapter": adapter, "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "group": step, "cursor": cursor, "history": history,
                    "args": vars(args) | {"init": str(args.init), "output": str(args.output)}}, tmp)
        tmp.replace(state_path)

    def checkpoint(step: int) -> None:
        target = args.output / f"checkpoint-{step}"
        merged = copy.deepcopy(model).merge_and_unload()  # the live adapter stays unmerged
        merged.config.use_cache = True
        merged.save_pretrained(str(target))
        tokenizer.save_pretrained(str(target))
        del merged
        if device.type == "cuda":
            torch.cuda.empty_cache()
        row = {"group": step, "checkpoint": str(target)}
        if args.probe_limit > 0:
            row |= probe(target, args.probe_limit, device)
        with open(probe_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"[checkpoint {step}] {json.dumps(row)}", flush=True)

    began = time.time()
    done, stopped = start, None
    with open(log_path, "a", encoding="utf-8") as log, ThreadPoolExecutor(args.workers) as executor:
        for step in range(start, args.groups):
            if args.max_seconds and time.time() - began > args.max_seconds:
                break
            if step % 25 == 0:
                ram_guard(args.min_free_ram_gib)
            drawn, kept, want, refills = [], [], args.prompts_per_step, 0
            while True:
                batch = [make_task(next(stream)) for _ in range(want)]
                cursor += want
                rollout(batch)
                score_batch(batch, executor)
                drawn += batch
                kept += [item for item in batch if any(abs(a) > 1e-8 for a in item["advantages"])]
                if not args.dynamic_sampling or len(kept) >= args.prompts_per_step or refills >= args.max_refills:
                    break
                want, refills = args.prompts_per_step - len(kept), refills + 1
            loss_value, kl_value, entropy = None, None, None
            if kept:  # a flat group carries no gradient
                model.train()
                optimizer.zero_grad(set_to_none=True)
                rows = sum(len(item["train"]) for item in kept)
                loss_value, kl_sum, nll, tokens = 0.0, 0.0, 0.0, 0
                for item in (part for group in kept for part in micro_batches(group, args.micro_rows)):
                    objective, kl, item_nll, item_tokens = group_objective(
                        model, item, rows, args.loss_norm, args.kl_beta, pad_id, device, args.logprob_temp)
                    scaler.scale(objective).backward()
                    loss_value += float(objective.detach())
                    kl_sum += kl or 0.0
                    nll, tokens = nll + item_nll, tokens + item_tokens
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                kl_value = kl_sum if args.kl_beta > 0 else None
                entropy = nll / tokens
                history["entropy"].append(entropy)
            lengths = sorted(len(ids) for item in drawn for ids in item["completions"])
            history["length"].append(sum(lengths) / len(lengths))
            record = {"group": step}
            if len(drawn) == 1:
                item = drawn[0]
                record |= {"task_id": item["id"], "rewards": item["rewards"],
                           "passed": item["statuses"].count("passed"), "statuses": item["statuses"]}
            else:
                record["groups"] = [{"task_id": item["id"], "bucket": item["bucket"], "rewards": item["rewards"],
                                     "passed": item["statuses"].count("passed"), "statuses": item["statuses"]}
                                    for item in drawn]
            buckets = {}
            for item in drawn:
                tally = buckets.setdefault(item["bucket"], [0, 0])
                tally[0] += item["statuses"].count("passed")
                tally[1] += len(item["statuses"]) - item["statuses"].count("not_evaluated")
            record |= {"policy_loss": loss_value, "kl": kl_value, "seconds": round(time.time() - began, 1),
                       "entropy": entropy, "len_mean": history["length"][-1],
                       "len_p90": lengths[int(0.9 * (len(lengths) - 1))],
                       "flat_share": 1 - len(kept) / len(drawn), "refills": refills, "pass": buckets}
            log.write(json.dumps(record) + "\n")
            log.flush()
            done = step + 1
            if done % args.save_every == 0:
                save(done)
                print(f"[step {done}] saved; pass {buckets}", flush=True)
            if args.checkpoint_every and done % args.checkpoint_every == 0 \
                    and not (args.output / f"checkpoint-{done}").is_dir():
                save(done)
                checkpoint(done)
            for key, factor in (("entropy", args.stop_entropy), ("length", args.stop_length)):
                if collapsed(history[key], factor):
                    stopped = {"step": done, "rule": key, "factor": factor,
                               "first_mean": sum(history[key][:STOP_WINDOW]) / STOP_WINDOW,
                               "last_mean": sum(history[key][-STOP_WINDOW:]) / STOP_WINDOW}
            if stopped:
                print(f"[stop] {json.dumps(stopped)}", flush=True)
                break
    save(done)
    if done >= args.groups or stopped:
        final = args.output / "final"
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(str(final))
        tokenizer.save_pretrained(str(final))
        print(f"[final] merged -> {final}", flush=True)
        if stopped:
            (args.output / "stopped.json").write_text(json.dumps(stopped, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
