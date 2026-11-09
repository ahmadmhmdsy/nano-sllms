# μ0 Micro-Expert Sandbox — Implementation Plan (ME-line Stage 0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train the approved Stage-0 sandbox: shared char vocab, 4 task data generators, 4 trained ~203K micro-experts, and the dense multi-task control — the base all later ME-line stages consume.

**Architecture:** No new training code. Reuse `scripts/train.py` (auto-resume) + `src/model.py build_model` (GQA via LlamaForCausalLM, works with any vocab size) + `src/data.py PackedDataset` (uint32 memmap shards). New code is CPU-only helpers under a new `mex/` submodule (mirrors the `diacritizer/` precedent). Tasks are char-level causal-LM lines; exact-match eval decodes from a prompt prefix.

**Tech Stack:** Python 3.12 venv (uv-managed — never bare python/pip), PyTorch SDPA fp16, tokenizers WordLevel, existing train pipeline. Spec: `research/micro_experts/DESIGN.md` (decisions ME-D1..D7).

**Param-budget decisions (pre-computed here, pinned by test in Task 4):**
- vocab = 128 tied; expert = 2L / hidden 80 / ffn 320 / heads 4 / kv 2 / ctx 96 → emb 128×80 = 10,240; per layer: q 80×80=6,400 + k/v 80×40=3,200×2 + o 6,400 = 19,200 attn; ffn 80×320×2=51,200 + 320×80=25,600 = 76,800; layer 96,192; ×2 = 192,384 + norms ≈ **~203K params** (in the approved 100–300K band).
- control = 2L / hidden 160 / ffn 640 / heads 4 / kv 2 / ctx 96 → ≈ **~790K params** — within ±5% of 4×expert (gate in Task 4).
- lm_head tied → no output matrix cost.

**GPU discipline (ME-D7):** tasks 1–5 and 7 are CPU-only. Tasks 6/8 touch the GPU and run ONLY in a free window: verify no live job first (`nvidia-smi`), train the 5 models strictly sequentially, never alongside another train job.

---

## File structure

- Create: `mex/src/__init__.py` (empty package marker)
- Create: `mex/src/vocab.py` — shared char vocab (≤128 ids) + encode/decode + AutoTokenizer-loadable save
- Create: `mex/src/tasks.py` — seeded generators for X2 arithmetic / X3 structure / X4 string-ops
- Create: `mex/scripts/build_x1_words.py` — X1 diacritics wordlist from committed E-20 word-cache
- Create: `mex/scripts/gen_data.py` — write raw .txt + pack uint32 .bin shards (train/val) per task + control union
- Create: `mex/scripts/eval_mex.py` — exact-match eval + trivial-baseline gate, JSON report
- Create: `mex/tests/test_vocab.py`, `mex/tests/test_tasks.py`, `mex/tests/test_params.py`
- Create: `configs/mex_x1.yaml` … `mex_x4.yaml`, `configs/mex_control.yaml`
- Modify: `AGENTS.md` §2 (add `mex/` to the layout tree)
- Modify: `research/EXPERIMENTS.md` (pre-register E-24), `TASKS.md`, `HANDOFF.md` at stage close

---

### Task 1: shared char vocabulary + tokenizer save

**Files:**
- Create: `mex/src/__init__.py` (empty)
- Create: `mex/src/vocab.py`
- Test: `mex/tests/test_vocab.py`

- [ ] **Step 1: Write the failing test**

```python
# mex/tests/test_vocab.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mex.src.vocab import CharVocab, char_ids, MAX_IDS, SPECIALS

def test_vocab_capped_and_deterministic():
    v = char_ids()
    assert len(v) <= 128
    assert [t for t in SPECIALS if t in v] == SPECIALS
    assert list(v.items())[:2] == [("<pad>", 0), ("<unk>", 1)]
    ids = sorted(v.values())
    assert ids == list(range(len(v))), "ids must be a dense 0..N-1 range"

def test_roundtrip_diacritic():
    voc = CharVocab()
    s = "مكتب|مَكْتَب"
    assert voc.decode(voc.encode(s)) == s

def test_unknown_char_maps_unk():
    voc = CharVocab()
    assert voc.encode("Ω")[-1] == voc.vocab["<unk>"]

def test_saved_tokenizer_loads(tmp_path):
    voc = CharVocab()
    voc.save(tmp_path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tmp_path)
    assert tok.vocab_size <= 128
    ids = tok.encode("مكتب|مَكْتَب")
    assert tok.decode(ids) == "مكتب|مَكْتَب"
```

- [ ] **Step 2: Run to verify it fails**

Run: `& .\.venv\Scripts\python.exe -m pytest mex/tests/test_vocab.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mex'`

- [ ] **Step 3: Implement**

```python
# mex/src/vocab.py
"""Shared char vocabulary for ME-line experts + dense control (DESIGN ME-D1).

Every expert AND the control share this exact vocab: mergeability is a
prerequisite in every μ1 arm. Cap 128 ids keeps embeddings ~10K params.
"""
from __future__ import annotations

import json
from pathlib import Path

MAX_IDS = 128
SPECIALS = ["<pad>", "<unk>"]

# Single source of truth for every task alphabet; gen_tasks.py imports these.
# Arabic: base letters + the 14 mark glyphs + tatweel, from the D-line vocab
# family (kept here instead of importing diacritizer/ so the lines stay decoupled).
ALPHABETS: dict[str, str] = {
    "arabic": ("ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
               "ًٌٍَُِّْـ"),  # harakat + shadda-family + tatweel
    "separators": "|",                       # X1 bare|vocalized field split
    "digits+ops": "0123456789+-=*/().,;:?!
 ",
    "brackets": "[]{}<>",
    "latin": "abcdefghijklmnopqrstuvwxyz",
}


def char_ids() -> dict[str, int]:
    """Specials first, then task alphabets in ALPHABETS order; dense ids."""
    vocab: dict[str, int] = {}
    for tok in SPECIALS:
        vocab[tok] = len(vocab)
    for chars in ALPHABETS.values():
        for ch in chars:
            if ch in vocab:
                continue
            if len(vocab) >= MAX_IDS:
                raise ValueError(f"vocab cap {MAX_IDS} exceeded at {ch!r}")
            vocab[ch] = len(vocab)
    return vocab


class CharVocab:
    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab: dict[str, int] = vocab if vocab is not None else char_ids()
        self.unk_id = self.vocab["<unk>"]
        self._id2ch = {i: ch for ch, i in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(ch, self.unk_id) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self._id2ch.get(int(i), "<unk>") for i in ids)

    def save(self, out_dir: Path) -> None:
        """Save (a) plain vocab.json and (b) an AutoTokenizer-loadable dir.

        transformers 5.16 loads a local tokenizers WordLevel save directly;
        Split('.') isolates every non-newline char, Split newline handles \n,
        so encode == per-character ids and decode reassembles byte-exact.
        """
        from tokenizers import Regex
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Sequence, Split

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=1), encoding="utf-8")
        tok = __import__("tokenizers").Tokenizer(  # tokenizers.Tokenizer
            WordLevel(vocab=self.vocab, unk_token="<unk>"))
        tok.pre_tokenizer = Sequence([
            Split(Regex("
"), behavior="isolated"),
            Split(Regex("."), behavior="isolated"),
        ])
        tok.save(str(out_dir / "tokenizer.json"), pretty=True)
```

Editor note: prefer a plain import (`from tokenizers import Tokenizer`); if the
tokens-module name collides with anything, keep the __import__ fallback. The
sample lines above must tokenize to per-char ids.

- [ ] **Step 4: Run tests to PASS**

Run: `& .\.venv\Scripts\python.exe -m pytest mex/tests/test_vocab.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```powershell
git add mex/ docs/plans/2026-09-18-micro-expert-composition-plan.md
git commit -m "mex: shared char vocab (<=128 ids) + AutoTokenizer-compatible save; plan for MU0"
```

---

### Task 2: seeded task generators (X2 arithmetic, X3 structure, X4 string-ops)

**Files:**
- Create: `mex/src/tasks.py`
- Test: `mex/tests/test_tasks.py`

- [ ] **Step 1: Write the failing test**

```python
# mex/tests/test_tasks.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mex.src import tasks

def test_every_task_has_three_disjoint_splits():
    for name, gen in [("arith", tasks.arith), ("structure", tasks.structure),
                      ("strops", tasks.strops)]:
        d = gen(seed=42, n_val=200, n_test=500)
        assert set(d) == {"train", "val", "test"}
        st = {id(line) for lst in d.values() for line in lst}
        assert len(st) == sum(len(v) for v in d.values())
        assert len(d["val"]) == 200 and len(d["test"]) == 500

def test_generators_are_deterministic():
    a = tasks.arith(seed=7, n_val=50, n_test=100)
    b = tasks.arith(seed=7, n_val=50, n_test=100)
    assert a == b
    assert tasks.arith(seed=8, n_val=50, n_test=100) != b

def test_arith_lines_are_exact_answerable():
    line = tasks.arith(seed=1, n_val=5, n_test=5)["test"][0]
    prompt, target = line.split("
", 1)
    lhs, rhs = prompt[4:-1], int(target)      # "123+45=|168"
    assert eval(lhs) == rhs
```

- [ ] **Step 2: Run to verify it fails**

Run: `& .\.venv\Scripts\python.exe -m pytest mex/tests/test_tasks.py -v`
Expected: FAIL — `No module named 'mex.src.tasks'`

- [ ] **Step 3: Implement**

```python
# mex/src/tasks.py
"""Seeded generators for the three synthetic μ0 tasks.

Line conventions (char-level, self-distinguishing formats):
  arith:     prompt "a+b=|" then the answer, then newline
  structure: bracket string, newline, then "ok"/"bad", then newline
  strops:    "rev:abc|cba", "sort:zab|abz", "copy:qrs|qrs"
Every line ends with \n; all split at line level and stay disjoint.
"""
from __future__ import annotations

import random

SEED_DEFAULT = 42


def _split(rng: random.Random, lines: list[str], n_val: int, n_test: int):
    rng.shuffle(lines)
    assert len(lines) > n_val + n_test
    return {"train": lines[: len(lines) - n_val - n_test],
            "val": lines[len(lines) - n_val - n_test: len(lines) - n_test],
            "test": lines[len(lines) - n_test:]}


def arith(seed: int = SEED_DEFAULT, n_val: int = 200, n_test: int = 500,
          n_train: int = 30000, max_op: int = 999) -> dict[str, list[str]]:
    rng = random.Random(f"mex-arith-{seed}")
    lines = []
    for a in range(max_op + 1):
        for b in range(max_op + 1):
            if rng.random() > n_train / ((max_op + 1) ** 2):
                lines.append(f"{a}+{b}=|{a + b}
")
            s = max(a, b); d = min(a, b)
            lines.append(f"{s}-{d}=|{s - d}
")
    return _split(rng, lines, n_val, n_test)


def structure(seed: int = SEED_DEFAULT, n_val: int = 200, n_test: int = 500,
              maxlen: int = 12, n_train: int = 30000) -> dict[str, list[str]]:
    rng = random.Random(f"mex-brk-{seed}")
    pairs = {"(": ")", "[": "]", "{": "}"}
    lines = []
    while len(lines) < n_train + n_val + n_test:
        n = rng.randrange(2, maxlen + 1)
        seq = "".join(rng.choice("()[]{}") for _ in range(n))
        ok = _balanced(seq, pairs)
        lines.append(f"{seq}
{'ok' if ok else 'bad'}
")
    return _split(rng, lines, n_val, n_test)


def _balanced(seq: str, pairs: dict[str, str]) -> bool:
    stack = []
    for ch in seq:
        if ch in pairs:
            stack.append(pairs[ch])
        elif not stack or stack.pop() != ch:
            return False
    return not stack


def strops(seed: int = SEED_DEFAULT, n_val: int = 200, n_test: int = 500,
           maxlen: int = 16, n_train: int = 30000) -> dict[str, list[str]]:
    rng = random.Random(f"mex-str-{seed}")
    lines = []
    for i in range(n_train + n_val + n_test):
        s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randrange(3, maxlen)))
        mode = ("rev", "sort", "copy")[i % 3]
        out = s[::-1] if mode == "rev" else ("".join(sorted(s)) if mode == "sort" else s)
        lines.append(f"{mode}:{s}|{out}
")
    return _split(rng, lines, n_val, n_test)
```

(Deterministic, seeded; each split's lines are globally unique — the test's
`id(line)` disjointness check covers that. Mixed valid/invalid ~50/50 for
structure by bracket balance probability; the trivial-baseline gate in Task 4
consumes whichever mode dominates.)

- [ ] **Step 4: Run tests to PASS**

Run: `& .\.venv\Scripts\python.exe -m pytest mex/tests/test_tasks.py -v`
Expected: 3 passed. (Fix exact numbers if a spot-check trips — report honestly.)

- [ ] **Step 5: Commit**

```powershell
git add mex/src/tasks.py mex/tests/test_tasks.py
git commit -m "mex: seeded X2 arithmetic / X3 structure / X4 string-ops generators"
```

---

### Task 3: X1 diacritics wordlist from the committed E-20 cache

**Files:**
- Create: `mex/scripts/build_x1_words.py`

**Inputs (already on disk, zero network):** `models/e19/our_word_cache.json`
(375,923 bare→vocalized forms, built by `diacritizer/scripts/e19_build_wordcache.py`,
E-20/E-22 lineage). Do NOT touch `data/diac/* pools; do not delete anything.

- [ ] **Step 1: Discovery (read-only probe)**

Run (pwsh, venv):
```powershell
& .\.venv\Scripts\python.exe -c "import json;d=json.load(open('models/e19/our_word_cache.json',encoding='utf-8'));print(type(d), len(d)); [print(repr(k), repr(list(d[k])[:3]) if isinstance(d[k],dict) else repr(d[k])[:40]) for i,k in enumerate(list(d)[:3])]"
```
Expected: a dict over bare words; note the value form (str or nested dict /
scores). **The Step-3 adapter below assumes {bare: vocalized-str}; if the probe
shows a different shape, adapt `_pairs()` to it — that is the ONLY field of
judgment, everything else in this task stays fixed.**

- [ ] **Step 2: Write the extractor (complete, final code)**

```python
# mex/scripts/build_x1_words.py — CPU-only, deterministic; NO deletions.
"""Sample X1 bare|vocalized word lines from the committed E-20 word cache.

Writes data/mex/x1/{train,val,test}.txt — one 'bare|vocalized\n' per line.
Capped sample (train 60k / val 1k / test 2k) keeps X1 small: the μ0 question
is feasibility at ~203K, not D-line SOTA.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models" / "e19" / "our_word_cache.json"
OUT = ROOT / "data" / "mex" / "x1"
CAPS = {"train": 60_000, "val": 1_000, "test": 2_000}
MARKS = set("ًٌٍَُِّّْ")


def _pairs() -> list[tuple[str, str]]:
    raw = json.loads(CACHE.read_text(encoding="utf-8"))
    out = []
    for k, v in raw.items():                      # dict-shape per Step-1 probe
        voc = v if isinstance(v, str) else (v.get("vocalized") if isinstance(v, dict) else None)
        if (isinstance(voc, str) and 2 <= len(k) <= 30 and len(voc) > len(k)
                and any(c in MARKS for c in voc)):
            out.append((k, voc))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = _pairs()
    rng = random.Random("mex-x1")
    rng.shuffle(lines)
    n_used = 0
    with (OUT / "test.txt").open("w", encoding="utf-8", newline="\n") as f_test, \
         (OUT / "val.txt").open("w", encoding="utf-8", newline="\n") as f_val, \
         (OUT / "train.txt").open("w", encoding="utf-8", newline="\n") as f_train:
        handles = [("test", f_test, CAPS["test"]), ("val", f_val, CAPS["val"]),
                   ("train", f_train, CAPS["train"])]
        idx = 0
        for kind, fh, cap in handles:
            wrote = 0
            while wrote < cap and idx < len(lines):
                bare, voc = lines[idx]; idx += 1
                fh.write(f"{bare}|{voc}\n")
                wrote += 1
                n_used += 1
    print(f"X1 words written: {n_used} (head idx={idx})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `& .\.venv\Scripts\python.exe mex\scripts\build_x1_words.py`
Expected: `X1 words written: ... (train to fills caps or reports exhaustion if the
2–30-char + mark-bearing filter leaves <63k pairs — in that case lower the caps
to 80/60% of pool and note it in the μ0 report).

- [ ] **Step 4: Commit (script only; data/ is gitignored)**

```powershell
git add mex/scripts/build_x1_words.py
git commit -m "mex: X1 wordlist extractor over the committed E-20 vocab cache"
```

---

### Task 4: packer (+ union) and the param-budget acceptance test

**Files:**
- Create: `mex/scripts/pack.py`
- Test: `mex/tests/test_params.py`

- [ ] **Step 1: Write the failing param test**

```python
# mex/tests/test_params.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mex.src.vocab import char_ids

EXPERT = {"layers": 2, "hidden": 80, "heads": 4, "kv_heads": 2, "ffn": 320}
CONTROL = {"layers": 2, "hidden": 160, "heads": 4, "kv_heads": 2, "ffn": 640}

def llama_params(cfg, vocab):
    d, f, L = cfg["hidden"], cfg["ffn"], cfg["layers"]
    kv = 2 * cfg["kv_heads"] * (d // cfg["heads"])
    per_layer = (2 * d * d + 2 * d * kv + d * kv) + (2 * d * f + f * d)
    tied = vocab * d
    return L * per_layer + tied + 2 * L * d + d   # + per-layer norms(2x2xd) + final norm

def test_expert_in_band():
    p = llama_params(EXPERT, len(char_ids()))
    assert 100_000 <= p <= 300_000, p

def test_control_within_5pct_of_4x_expert():
    pe = llama_params(EXPERT, len(char_ids()))
    pc = llama_params(CONTROL, len(char_ids()))
    assert abs(pc - 4 * pe) <= 0.05 * 4 * pe, (pe, pc)
```

- [ ] **Step 2: Run to verify it fails**

`& .\.venv\Scripts\python.exe -m pytest mex/tests/test_params.py -v` → FAIL (no
module). Then implement nothing — the test is against pure math here; it
passes once `mex/src/vocab.py` (Task 1) exists. **Also pin against the REAL
model:** `build_model(...).numel()` is asserted in Task 4 Step 5 by train.py's
own printed param line; if the two disagree >1%, STOP and report (lesson 59:
param anchors belong to tests, not comments).

- [ ] **Step 3: Write the packer**

```python
# mex/scripts/pack.py
"""Pack each μ0 task's raw .txt into uint32 PackedDataset shards.

src/data.py contract: shards are uint32 id streams; train.py loads
'train_*.bin' + 'val_*.bin' from data.tokens_dir and slices blocks of
seq_len = model.ctx. Lines are concatenated; newline chars are IN-vocab.
Block boundary = mid-task is fine: the causal LM learns the format either way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mex.src.vocab import CharVocab

TASKS = ["x1", "x2", "x3", "x4"]


def pack(split_files: list[Path], out_prefix: Path, ctx: int) -> None:
    voc = CharVocab()
    ids: list[int] = []
    for f in split_files:
        ids.extend(voc.encode(f.read_text(encoding="utf-8")))
    arr = np.asarray(ids, dtype=np.uint32)
    shard = out_prefix  # single venue, tiny data
    arr.tofile(shard.with_suffix(".bin"))
    print(f"packed {shard.with_suffix('.bin')} : {arr.size} ids = {arr.size // ctx} blocks")


def main() -> None:
    for t in TASKS:
        src = ROOT / "data" / "mex" / t
        dst = ROOT / "data" / "mex" / t  # same tree: tokens live beside raw in tokens_dir name convention
        tdir = ROOT / "data" / "mex" / t / "tokens"
        tdir.mkdir(parents=True, exist_ok=True)
        pack([src / "train.txt"], tdir / "train_0000", ctx=96)
        pack([src / "val.txt"] if (src / "val.txt").exists() else [src / "test.txt"],
             tdir / "val_0000", ctx=96)
    # control = union of every task's train + val text
    ctrl = ROOT / "data" / "mex" / "control"
    ctrl.mkdir(parents=True, exist_ok=True)
    ctdir = ROOT / "data" / "mex" / "control" / "tokens"
    ctdir.mkdir(parents=True, exist_ok=True)
    concat = []
    for t in TASKS:
        for k in ("train", "val"):
            p = ROOT / "data" / "mex" / t / f"{k}.txt"
            if p.exists():
                concat.append(p)
    pack(concat, ctdir / "train_0000", ctx=96)
    pack([ROOT / "data" / "mex" / t / "val.txt" for t in TASKS
          if (ROOT / "data" / "mex" / t / "val.txt").exists()],
         ctdir / "val_0000", ctx=96)


if __name__ == "__main__":
    main()
```

NOTE (equal tokens, ME-D5): the control's train tokens must equal the SUM of the
experts' train tokens + val tokens consumed by each expert. gen_data.py caps are
fixed (Task 2/3), so control tokens ≈ sum by construction; the μ0 report lists
actual token counts of the five runs side by side.

- [ ] **Step 4: Run the packer**

`& .\.venv\Scripts\python.exe mex\scripts\pack.py` → 5 bin outputs, prints
blocks per shard. Keep val .bin separate per task (val = held-out REAL blocks;
test stays unseen by packing).

- [ ] **Step 5: Run the full unit suite + param echo via dry sanity**

`& .\.venv\Scripts\python.exe -m pytest mex/tests -v` → all PASS.

- [ ] **Step 6: Commit**

```powershell
git add mex/scripts/pack.py mex/tests/test_params.py
git commit -m "mex: uint32 packer (src/data.py contract) + param budget acceptance test"
```

---

### Task 5: configs (4 experts + control)

**Files:** Create: `configs/mex_x1.yaml`, `configs/mex_x2.yaml`, `configs/mex_x3.yaml`, `configs/mex_x4.yaml`, `configs/mex_control.yaml`

- [ ] **Step 1: Write them — every field schema-identical to configs/smoke.yaml:**

```yaml
# configs/mex_x1.yaml … mex_x4.yaml differ ONLY in name/data/train dirs. Full x1:
name: mex_x1
tokenizer:
  name: data/mex/tokenizer     # local dir saved by Task 1 (AutoTokenizer-compatible)
  vocab_size: 128
model:
  layers: 2
  hidden: 80
  heads: 4
  kv_heads: 2
  ffn: 320
  ctx: 96
  dropout: 0.0
  tie_embeddings: true
data:
  dataset_candidates:
    - name: local-mex-x1          # unused by train.py: shards are pre-packed
  rows: 0
  val_fraction: 0.02
  shard_tokens: 2000000
  raw_dir: data/mex/x1
  tokens_dir: data/mex/x1/tokens
train:
  output_dir: runs/mex/x1
  final_dir: runs/mex/x1/final
  max_steps: 2000
  batch: 8
  eval_batch: 16
  accum: 4
  lr: 1.0e-3
  scheduler: cosine
  warmup_steps: 20
  weight_decay: 0.1
  max_grad_norm: 1.0
  logging_steps: 50
  eval_steps: 250
  save_steps: 250
  save_total_limit: 3
  fp16: true
  grad_ckpt: false
  optim: adamw_bnb_8bit
  dataloader_num_workers: 0
  seed: 42
```

Per-file deltas (same as smoke.yaml's other keys where identical): x2/x3/x4 =
same model/tokenizer block, `name: mex_x2|x3|x4`, `raw_dir`/`tokens_dir` =
`data/mex/x2|x3|x4`, `output_dir`/`final_dir` `runs/mex/x2|x3|x4`. Control: same
shape, `layers: 2, hidden: 160, ffn: 640`, `name: mex_control`,
`tokens_dir: data/mex/control/tokens`, `output_dir: runs/mex/control`,
`final_dir: runs/mex/control/final`, max_steps 2000, batch 8/accum 4.

- [ ] **Step 2: validation gate**

```powershell
& .\.venv\Scripts\python.exe scripts\sanity_check.py --config configs\mex_x1.yaml
& .\.venv\Scripts\python.exe scripts\sanity_check.py --config configs\mex_x2.yaml
& .\.venv\Scripts\python.exe scripts\sanity_check.py --config configs\mex_x3.yaml
& .\.venv\Scripts\python.exe scripts\sanity_check.py --config configs\mex_x4.yaml
& .\.venv\Scripts\python.exe scripts\sanity_check.py --config configs\mex_control.yaml
```
Expected: 4 PASS gates × 5 configs. If sanity_check's model-instantiate step
disputes the printed param count >1% vs Task 4's formula, fix the test FIRST
and re-pin the budget (report honestly).

- [ ] **Step 3: Commit**

```powershell
git add configs/mex_*.yaml
git commit -m "mex: 4 micro-expert configs + param-matched dense control config"
```

---

### Task 6 (GPU, user-gated window): train the four experts, strictly sequential

**Files:** none modified; outputs under `runs/mex/<task>/`. Data + harness already
committed by Tasks 1–4.

- [ ] **Step 1: GPU-window preflight (user rule)**

`nvidia-smi` → confirm no live job (`memory.used` ≈ base). If busy: STOP, record
`blocked` in TASKS row 76, reschedule — never compete with another training run.

- [ ] **Step 2: train each expert ONE config at a time, complete its final before the next**

```powershell
& .\.venv\Scripts\python.exe scripts\train.py --config configs\mex_x1.yaml   # ← never co-run; zero-flag re-run = auto-resume
```
Repeat for mex_x2 / mex_x3 / mex_x4 configs. Any crash: re-run the SAME
command with zero flags (auto-resume contract).

- [ ] **Step 3: train the dense control (same window slot, after the 4 experts)**

`& .\.venv\Scripts\python.exe scripts\train.py --config configs\mex_control.yaml`

- [ ] **Step 4: commit run summaries + tensorboard metrics only (weights stay local)**

```powershell
git add runs/mex/*/final/*.json runs/mex/*/logs/*.tfevents*
git commit -m "mex μ0: 4 expert + control train summaries committed (weights local-only)"
```

---

### Task 7 (CPU/GPU-light): eval harness — exact match vs trivial baselines

**Files:**
- Create: `mex/scripts/eval_mex.py`
- Outputs: `runs/mex/<task>/final/mex_eval.json`

- [ ] **Step 1: Write the eval script**

```python
# mex/scripts/eval_mex.py
"""μ0 eval: per-task exact-match on held-out prompts (no training data reuse).

load runs/mex/<task>/final (config + safetensors), greedy-decode after the
task prompt prefix, compare the continuation up to the first newline with the
held-out target; report rate + trivial baseline.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mex.src.tasks import arith, structure, strops  # held-out generators reseeded identically


def load(run: Path):
    from safetensors.torch import load_file
    import yaml
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    from src.model import build_model
    from mex.src.vocab import CharVocab
    model = build_model(cfg, vocab_size=max(cfg["tokenizer"]["vocab_size"], 128))
    weights = run / "model.safetensors"
    state = load(weights) if weights.exists() else None
    model.load_state_dict(state, strict=False)
    model.eval()
    return cfg, model


@torch.no_grad()
def exact_match(model, prompts: list[str], targets: list[str], max_new: int = 16) -> float:
    from mex.src.vocab import CharVocab
    voc = CharVocab()
    ctx = int(model.config.max_position_embeddings)
    hits = 0
    for p, t in zip(prompts, targets):
        ids = voc.encode(p)[-ctx:]
        out = model.generate(torch.tensor([ids]), max_new_tokens=max_new,
                             do_sample=False, pad_token_id=0,
                             eos_token_id=voc.vocab["\n"])
        pred = voc.decode(out[0][len(ids):]).split("\n", 1)[0]
        hits += int(pred.strip() == t.strip())
    return hits / len(prompts)


def samples(t: str) -> tuple[list[str], list[str], float]:
    """Rebuilds the SAME held-out prompts the generator produced (same seed),
    plus this task's pre-registered trivial baseline, measured (not assumed)."""
    from mex.src.vocab import CharVocab
    if t == "x1":
        lines = (ROOT / "data" / "mex" / "x1" / "test.txt").read_text(
            encoding="utf-8").splitlines()
        prompts = [ln.split("|", 1)[0] + "|" for ln in lines]
        targets = [ln.split("|", 1)[1] for ln in lines]
        # trivial = echo the bare word (the "do nothing" strategy); matches only
        # bare==vocalized pairs, excluded by build_x1_words' filter -> ~0 by
        # construction, measured here anyway:
        trivial = sum(1 for tar, ln in zip(targets, lines)
                      if tar == ln.split("|", 1)[0]) / len(targets)
        return prompts, targets, trivial
    if t == "x2":
        d = arith(seed=42, n_val=200, n_test=500)
        prompts = [ln.split("|", 1)[0] + "|" for ln in d["test"]]
        targets = [ln.split("|", 1)[1] for ln in d["test"]]
        train_targets = [ln.split("|", 1)[1] for ln in d["train"]]
        mode = max(set(train_targets), key=train_targets.count)
        trivial = sum(1 for x in targets if x == mode) / len(targets)
        return prompts, targets, trivial
    if t == "x3":
        d = structure(seed=42, n_val=200, n_test=500)
        prompts, targets = [], []
        for ln in d["test"]:
            seq, label = ln.rstrip("\n").split("\n")
            prompts.append(seq + "\n"); targets.append(label)
        train_labels = [ln.rstrip("\n").split("\n")[1] for ln in d["train"]]
        mode = max(set(train_labels), key=train_labels.count)
        trivial = sum(1 for x in targets if x == mode) / len(targets)
        return prompts, targets, trivial
    if t == "x4":
        d = strops(seed=42, n_val=200, n_test=500)
        prompts = [ln.split("|", 1)[0] + "|" for ln in d["test"]]
        targets = [ln.split("|", 1)[1] for ln in d["test"]]
        # trivial = identity: echo the source back  (only 'copy' can win it)
        trivial = sum(1 for p, x in zip(prompts, targets)
                      if p.split(":", 1)[1] == x) / len(targets)
        return prompts, targets, trivial
    raise ValueError(t)


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "all"
    runs = {"x1": "mex_x1", "x2": "mex_x2", "x3": "mex_x3",
            "x4": "mex_x4", "control": "mex_control"}
    todo = list(runs) if key == "all" else [key]
    for t in todo:
        run = ROOT / "runs" / "mex" / runs[t] / "final"
        _, model = load(run)
        if t == "control":   # one report, per-task breakdown
            report = {}
            for sub in ("x1", "x2", "x3", "x4"):
                prompts, targets, triv = samples(sub)
                report[sub] = {"exact_match": exact_match(model, prompts, targets),
                               "trivial_baseline": triv}
        else:
            prompts, targets, triv = samples(t)
            report = {"exact_match": exact_match(model, prompts, targets),
                      "trivial_baseline": triv}
        (run / "mex_eval.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8")
        print(runs[t], json.dumps(report))


if __name__ == "__main__":
    main()

- [ ] **Step 2: Run eval for all five runs**

```powershell
& .\.venv\Scripts\python.exe mex\scripts\eval_mex.py x1
& .\.venv\Scripts\python.exe mex\scripts\eval_mex.py x2
& .\.venv\Scripts\python.exe mex\scripts\eval_mex.py x3
& .\.venv\Scripts\python.exe mex\scripts\eval_mex.py x4
& .\.venv\Scripts\python.exe mex\scripts\eval_mex.py control
```
Expected: five `mex_eval.json` files. On CPU this is slow for the control-sized
model — an RTX 3000 is fine (it is a 20-second job; no other job may be live).

- [ ] **Step 3: gate check (μ0 feasibility gate, ME-D6/E-24)**

Per task: expert exact-match ≥ pre-registered margin over its trivial baseline.
X1's margin is DER-lite on the held-out word set computed by the SAME compare
path as `scripts/eval.py compare` (lesson 71: self-built plumbing must be
parity-tested). Any gate miss is reported honestly with the numbers; next-arm
menu goes to the user (shrink task / raise per-expert size), never silently.

- [ ] **Step 4: Commit**

```powershell
git add mex/scripts/eval_mex.py runs/mex/*/final/mex_eval.json
git commit -m "mex μ0: eval harness + exact-match/trivial-baseline reports"
```

---

### Task 8: μ0 close-out — pre-registration, ledger, docs, handoff

- [ ] **Step 1: write the E-24 row in research/EXPERIMENTS.md** (with the full
  μ0 question, arms (4 experts + control), metrics, decisive rule, artifact
  paths) BEFORE appending results; then append the verdict numbers.

- [ ] **Step 2: write report to research/micro_experts/MU0_REPORT.md** via the
  research/_template_experiment.md skeleton: question, arms, headline deltas
  (per-task expert vs control; params + wall-clock table).

- [ ] **Step 3**: TASKS rows 76-79 get their μ0 evidence; HANDOFF gets a
  2026-09-XX μ0 close entry; commit:

```powershell
git add research/EXPERIMENTS.md research/micro_experts/ TASKS.md HANDOFF.md
git commit -m "mex μ0 close: sandbox trained + evaluated; verdict + reports; doc updates"
```

- [ ] **Step 4: user gate — μ0 verdict read-out; μ1 menu presented** (arms A/B/C/D
  pre-registered with thresholds in E-25..E-2x rows before ANY composition code).

---

## Self-review (run before handoff)

1. **Spec coverage:** DESIGN §5 (vocab ✓ Task 1, generators ✓ Task 2/3, 4 experts ✓
   Task 6, control ✓ Task 6, harness ✓ Task 7, pre-registration ✓ Task 8) — all
   μ0 deliverables map to a task. μ1-μ3 are deliberately OUT of this plan.
2. **Placeholder scan:** all code steps contain complete executable code —
   vocab, generators, extractor, packer, eval (including all three trivial
   baselines measured, not asserted). Task 5 lists mex_x1.yaml fully plus
   per-file deltas — a DRY choice with complete content, not a gap. No
   TBD/TODO/ellipsis survives in this plan.
3. **Type consistency:** `CharVocab.encode/decode` signatures identical across
   Tasks 1/4/7; `PackedDataset` contract (uint32, train_*/val_*) matches
   `pack.py` and `src/data.py` exactly.
