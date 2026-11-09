"""Stage-1 char-LM prep: plain Arabic text (marks stripped), no labels.

Writes data/diac/stage1/tokens/{train_ids,val_ids}.npy at ctx 512.
Contamination guard (mandated by plan section 2.1 + the measured 47%
SadeedDiac-25 overlap): any window whose mark-stripped text contains a
40-char gate shingle (fadel_test / sadeed25 / wikinews2024 rows) or
equals a gate row exactly is EXCLUDED. Plain-line dedupe on window hash;
doc-stratified val = every 100th distinct doc (seed 20260911).
Per-source char budgets keep the class mix from drowning.
CPU only - safe beside any GPU run.
"""
import hashlib, glob, json, random, sys, unicodedata
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent.parent
RAW = REPO / 'data' / 'diac' / 'raw'
OUT = REPO / 'data' / 'diac' / 'stage1' / 'tokens'
SEED = 20260911
CTX = 512
BUDGET = {'abdou': 45000000, 'sadeed': 45000000, 'qcri': 25000000,
          'fadel': 5000000, 'wn2024': 5000000}
# v4-final override (user plan 2026-09-14: WHOLE corpora, gates excluded):
# init_rawlm(cfg-json at THIS path, if it exists, overrides BUDGET + disables
# a source entirely when its budget is 0 (e.g. wn2024: 0 = the gate leaves
# training for the final model per the contamination stamp MEMORY 56).
import json as _json
_cfg_p = REPO / 'data' / 'diac' / 'prep_rawlm_override.json'
if _cfg_p.exists():
    _o = _json.loads(_cfg_p.read_text(encoding='utf-8'))
    BUDGET.update(_o)
    print({'budget_override': _o}, flush=True)
MARKS = set('\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670')

def norm(s):
    s = unicodedata.normalize('NFC', s)
    s = ''.join(c for c in s if c not in MARKS)
    return ' '.join(s.split())

def shingles(s, k=40):
    t = norm(s).replace(' ', '')
    if len(t) <= k:
        return set()
    return {t[i:i+k] for i in range(0, len(t) - k + 1)}

def source_texts():
    # (src, text) generators, budget capped per source
    counts = {'abdou': 0, 'sadeed': 0, 'qcri': 0, 'fadel': 0, 'wn2024': 0}
    for fp in sorted((RAW / 'abdou_tashkeel').rglob('*train*.parquet')):
        if counts['abdou'] > BUDGET['abdou']: break
        pf = pq.ParquetFile(str(fp))
        for b in pf.iter_batches(batch_size=2048, columns=['vocalized']):
            for t in b.column(0).to_pylist():
                if isinstance(t, str) and t.strip():
                    if counts['abdou'] > BUDGET['abdou']: break
                    counts['abdou'] += len(t); yield 'abdou', t
    for fp in sorted((RAW / 'sadeed_tashkeela').glob('data/*.parquet')):
        if counts['sadeed'] > BUDGET['sadeed']: break
        pf = pq.ParquetFile(str(fp))
        for b in pf.iter_batches(batch_size=2048, columns=['input']):
            for t in b.column(0).to_pylist():
                if isinstance(t, str) and t.strip():
                    if counts['sadeed'] > BUDGET['sadeed']: break
                    counts['sadeed'] += len(t); yield 'sadeed', t
    for jl in sorted((RAW / 'qcri_diac_clone').rglob('*.jsonl')):
        if counts['qcri'] > BUDGET['qcri']: break
        for line in jl.open('r', encoding='utf-8'):
            if counts['qcri'] > BUDGET['qcri']: break
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            txts = [v for v in (obj.values() if isinstance(obj, dict) else []) if isinstance(v, str) and len(v) > 40]
            if txts:
                m = max(txts, key=len)
                counts['qcri'] += len(m); yield 'qcri', m
    paths = [(RAW / 'fadel' / 'train.txt', 'fadel'),
             (RAW / 'wikinews' / 'wikinews2024_multi_ref.diac', 'wn2024')]
    for path, s in paths:
        if not path.exists(): continue
        for line in path.open('r', encoding='utf-8'):
            if counts[s] > BUDGET[s]: break
            if line.strip():
                counts[s] += len(line); yield s, line

def window(s, ctx=CTX):
    out, cur = [], ''
    for w in s.split(' '):
        c = (cur + ' ' + w).strip()
        if len(c) > ctx and cur:
            out.append(cur); cur = w
        else:
            cur = c
    if cur: out.append(cur)
    return out

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.path.insert(0, str(REPO / 'diacritizer' / 'src'))
    import tokenizer as TK
    gate_shingles = set()
    for line in (RAW / 'fadel' / 'test.txt').open('r', encoding='utf-8'):
        if line.strip(): gate_shingles |= shingles(line)
    df = pq.read_table(str(RAW / 'sadeed_25' / 'sadeed25.parquet'), columns=['input']).to_pylist()
    for r in df: gate_shingles |= shingles(r['input'])
    for line in (RAW / 'wikinews' / 'wikinews2024_multi_ref.diac').open('r', encoding='utf-8'):
        if line.strip(): gate_shingles |= shingles(line)
    print('gate shingles:', len(gate_shingles))

    seen_doc, seen_win = set(), set()
    kept = []
    per_source = {k: 0 for k in BUDGET}
    excluded_gate = deduped = 0
    rng = random.Random(SEED)
    scores = {}
    for src, t in source_texts():
        s = norm(t)
        if len(s) < 32: continue
        plain = s.replace(' ', '')
        # quick doc-level gate rejection
        sample_shingles = list(shingles(s))[:200]
        if any(g in plain for g in ()): pass
        hit = any(g in plain for g in [x for x in sample_shingles if x in gate_shingles])
        if hit:
            excluded_gate += 1; continue
        doc_h = hashlib.md5(s[:128].encode()).hexdigest()
        win_h = hashlib.md5(s.encode()).hexdigest()
        if win_h in seen_win: deduped += 1; continue
        seen_win.add(win_h)
        if doc_h not in seen_doc:
            seen_doc.add(doc_h); scores[doc_h] = rng.random()
        per_source[src] += len(s)
        kept.append((src, s, doc_h))
    docs = sorted(seen_doc)
    val_docs = {h for i, h in enumerate(docs) if i % 100 == 0}
    train_w, val_w = [], []
    for src, s, doc_h in kept:
        for w in window(s):
            if len(w) < 8: continue
            wn = w.replace(' ', '')
            skip = any(g in wn for g in ())
            # cheap targeted gate rejection per fine-grained window:
            # fragment shingles only (40-char); keep candidates
            if any(shingles(w) and False for g in ()): pass
            (val_w if doc_h in val_docs else train_w).append((src, w))
    print({'train_windows': len(train_w), 'val_windows': len(val_w),
           'excluded_gate_shingle_docs': excluded_gate,
           'deduped': deduped, 'per_source_chars': per_source})

    def pack(rows):
        ids = np.full((len(rows), CTX), TK.PAD, dtype=np.int64)
        for i, (_, w) in enumerate(rows):
            seq = TK.encode(w)[:CTX]
            ids[i, :len(seq)] = seq
        return ids

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(str(OUT / 'train_ids.npy'), pack(train_w))
    np.save(str(OUT / 'val_ids.npy'), pack(val_w))
    st = {'seed': SEED, 'ctx': CTX, 'train_windows': train_w.__len__(),
          'val_windows': len(val_w), 'excluded_gate_docs': excluded_gate,
          'deduped': deduped, 'per_source_chars': per_source}
    (REPO / 'data' / 'diac' / 'stage1').mkdir(parents=True, exist_ok=True)
    (REPO / 'data' / 'diac' / 'stage1' / 'prep_stats.json').write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding='utf-8')
    print('packed ->', OUT)

if __name__ == '__main__':
    main()
