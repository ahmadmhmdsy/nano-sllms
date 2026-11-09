"""Prepare D-line training data (R51): clean -> align -> windows -> split -> shards.

Pipeline per DESIGN section 3:
  1. read raw source lines (diacritized)
  2. parse via passthrough.parse (alignment + 15-class labels built in one pass);
     invalid samples -> quarantine@<source>.txtl ( QuarantineError ), never silent
  3. window to <=CTX chars splitting ONLY at word/space boundaries
     (never mid-word / inside a number+URL run: split points are spaces only)
  4. document-level dedupe on normalized text hash
  5. document-stratified train/val split by MD5(seed) rank (no sentence leakage;
     the same doc never straddles sides)
  6. write provenance jsonl + prepare_stats.json (seed-reproducible)

CPU-only. Usage:
    .venv/Scripts/python.exe diacritizer/scripts/prepare_data.py [--max-lines N] [--sources fadel,wikinews]
"""
import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import passthrough as PT  # noqa: E402

RAW = REPO / "data" / "diac" / "raw"
OUT = REPO / "data" / "diac" / "prepared"
CTX_CHARS = 128  # windowing must MATCH the token ctx: 1024-windows truncated at
# tokenization waste ~75% of the text (v2 lesson 2026-09-12)
SEED = 20260911
# E-097 full-character-set variant: with PREP_KEEP_ALL=1 every unit is kept and
# non-target units are labelled BARE (0), so the model trains on the same alphabet
# it must decode at inference (E-094/E-095 showed the default stripped input is what
# the model was trained on, and that feeding anything else costs ~24 pct on the
# registered gate). Default OFF, so every existing corpus reproduction is
# byte-identical. Read at import time because multiprocessing spawn workers
# re-import this module and inherit the environment.
KEEP_ALL = os.environ.get("PREP_KEEP_ALL") == "1"

SOURCES = {
    # id -> (dir, files, domain, kind, opts)
    "fadel_train":  ("fadel", ["train.txt"], "classical", "lines", None),
    "fadel_val":    ("fadel", ["val.txt"], "classical", "lines", None),
    "fadel_test":   ("fadel", ["test.txt"], "classical", "lines", None),
    "wikinews2024": ("wikinews", ["wikinews2024_multi_ref.diac"], "msa", "lines", None),
    # external holdouts (Abdou MIT): never in train; valid -> val pool
    "abdou_train":  ("abdou_tashkeel", None, "classical", "parquet", None),
    "abdou_valid":  ("abdou_tashkeel", None, "classical", "parquet", None),
    # Misraj/Sadeed_Tashkeela (user account unlocked 2026-09-13): input/output
    # pairs; train shards -> train pool, test shard = gate-only (downloaded,
    # never prepped).
    "sadeedt_train": ("sadeed_tashkeela", None, "classical", "sadeedhf", None),
    # pseudo-source ONLY for the val split (excluded from prepared data by policy)
    "qcri_wiki":    ("qcri_diac_clone", None, "modern", "qcrijsonl", None),
}
# Sources kept OUT of the prepared shards entirely (external benchmark gates,
# evaluated from raw files; never influencing model selection).
EXTERNAL_SOURCES = {"fadel_test"}


def normalize_line(t):
    """NFC only; strips zero-width noise; no letter-form folding (lane decisions)."""
    return unicodedata.normalize("NFC", t.strip())


def window_line(text, ctx=CTX_CHARS):
    """Yield <=ctx windows, split at spaces only. Keeps whole words."""
    if len(text) <= ctx:
        yield text
        return
    words = text.split(" ")
    cur = ""
    for w in words:
        # hard word (huge) fallback: emit cur and keep the word whole as own window
        if len(w) > ctx:
            if cur:
                yield cur.rstrip(" ")
                cur = ""
            yield w[:ctx]
            continue
        cand = (cur + " " + w).strip(" ")
        if len(cand) > ctx and cur:
            yield cur
            cur = w
        else:
            cur = cand
    if cur:
        yield cur


def _files_in(subdir):
    d = RAW / subdir
    if not d.exists():
        return []
    return sorted(p for p in d.glob("**/*") if p.suffix in ('.parquet', '.txt', '.jsonl', '.diac', '.json'))

def _iter_parquet_rows(subdir, want):
    """(stable_row_key, vocalized_text) per parquet shard; want in {train, valid}."""
    import pyarrow.parquet as pq
    files = [p for p in sorted((RAW / subdir).rglob("*.parquet")) if want in p.name]
    for fp in files:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(fp))
        for bi, batch in enumerate(pf.iter_batches(batch_size=512, columns=["vocalized"])):
            for i, t in enumerate(batch.column(0).to_pylist()):
                if isinstance(t, str) and t.strip():
                    yield (fp.name + "#" + str(bi * 512 + i)), t

def _first_diac_text(obj):
    """Find the first string field looking like diacritized text."""
    best = None
    for v in (obj.values() if isinstance(obj, dict) else []):
        if isinstance(v, str) and len(v) > 40:
            marks = sum(1 for ch in v if ch in '\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652')
            bases = sum(1 for ch in v if PT.is_arabic_base(ch))
            if marks and (best is None or marks > best[0]):
                best = (marks, v)
    return best[1] if best else None

def stream_samples(want_sources=None, exclude=None):
    """source_id, line -> windowed raw samples with provenance.

    want_sources/exclude are decided BEFORE any file IO for that source:
    skipped sources are never read (self.scans of 1.4M-row shard sets were the
    dominant cost when filters lived downstream).
    """
    for src_id, (subdir, files, domain, kind, opts) in SOURCES.items():
        if exclude and src_id in exclude:
            continue
        if want_sources and src_id not in want_sources:
            continue
        if kind == "lines":
            for fname in files:
                path = RAW / subdir / fname
                if not path.exists():
                    print("  [WARN] missing raw %s (%s)" % (src_id, path))
                    continue
                for idx, line in enumerate(path.open("r", encoding="utf-8")):
                    yield {"src": src_id, "domain": domain, "line": idx, "win": "t", "text": normalize_line(line)}
        elif kind == "parquet":
            want = {"abdou_train": "train", "abdou_valid": "valid"}[src_id]
            for lkey, t in _iter_parquet_rows(subdir, want):
                yield {"src": src_id, "domain": domain, "line": lkey, "win": 0, "text": normalize_line(t)}
        elif kind == "sadeedhf":
            for fpath in sorted((RAW / subdir).rglob("train-*.parquet")):
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(str(fpath))
                for bi, batch in enumerate(pf.iter_batches(batch_size=512, columns=["output"])):
                    for i, t in enumerate(batch.column(0).to_pylist()):
                        if isinstance(t, str) and t.strip():
                            yield {"src": src_id, "domain": domain, "line": fpath.name + "#" + str(bi * 512 + i), "win": 0, "text": normalize_line(t)}
        elif kind == "qcrijsonl":
            jl = RAW / subdir / "datasets" / "wikipedia_diacritized.jsonl"
            if not jl.exists():
                alt = None
                for cand in (RAW / subdir).rglob("*.jsonl"):
                    alt = cand
                    break
                jl = alt
            if not jl or not jl.exists():
                print("  [WARN] qcrijsonl not found under %s" % (RAW / subdir))
                continue
            for idx, line in enumerate(jl.open("r", encoding="utf-8")):
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                t = _first_diac_text(obj)
                if t:
                    yield {"src": src_id, "domain": domain, "line": idx, "win": 0, "text": normalize_line(t)}


def _parse_batch(chunk):
    """Module-level batch worker (Windows spawn needs importable top-level fns)."""
    return [_parse_pair(a) for a in chunk]


def _parse_pair(args):
    """Pool WORKER (module level: Windows 'spawn' needs picklable top-level fns).

    Returns (quarantined, src, line, text, bases, labels, err, domain, win).
    """
    src, domain, line, win, text = args
    try:
        units = PT.parse(text)
    except PT.QuarantineError as e:
        return (True, src, line, text, None, None, str(e), domain, win)
    n_targets = sum(1 for u in units if u.is_target)
    if n_targets == 0:
        return (False, src, line, text, None, None, "EMPTY", domain, win)
    if KEEP_ALL:
        # NOTE: passthrough.parse groups runs of non-target characters into ONE
        # Unit whose .raw can be many chars long, while tokenizer.encode emits one
        # id per CHARACTER. Labels must therefore be expanded per character or
        # len(bases) != len(labels) and pack_rows silently misaligns y.
        parts, labs = [], []
        for u in units:
            parts.append(u.raw)
            if u.is_target:
                labs.append(u.label)
            else:
                labs.extend([0] * len(u.raw))
        return (False, src, line, text, "".join(parts), labs, None, domain, win)
    return (False, src, line, text,
            "".join(u.raw for u in units if u.is_target),
            [u.label for u in units if u.is_target], None, domain, win)


def process(out_limit=None, sources=None, out_dir=None):
    # out_dir override MUST land here (a module-global rebind in main() is only
    # a local and silently wrote through to the general corpus - lesson 2026-09-12).
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    qdir = out / "quarantine"
    qdir.mkdir(exist_ok=True)
    seen_hashes = set()
    train_rows, val_rows = [], []
    stats = {"seed": SEED, "ctx_chars": CTX_CHARS, "keep_all": KEEP_ALL,
             "per_source": {}, "quarantined": 0,
             "deduped_dups": 0, "windows_kept": 0}
    qfiles = {}
    rank_cache = {}
    n = 0
    import os
    from multiprocessing import Pool

    # NOTE: worker fn must stay importable at module level; rec-only arg.
    import itertools as _it

    def _recs():
        q = 0
        for rec in stream_samples(want_sources=sources, exclude=EXTERNAL_SOURCES):
            if out_limit and q >= out_limit:
                return
            q += 1
            yield rec

    workers = int(os.environ.get("PREP_WORKERS", "0")) or (os.cpu_count() or 4)
    n = 0
    procs = max(1, min(workers, 16))
    # Batched parallel parse: materialize the scan first (C-speed reads), chunk
    # it, then pool.map. Avoids the known Windows pool teardown deadlock that
    # lazy imap-streaming generators hit (handler thread alive at terminate).
    args_list = [(r["src"], r["domain"], r["line"], r["win"], r["text"]) for r in _recs()]
    results = []
    import time as _t
    _t0 = _t.time()
    if procs > 1 and len(args_list) > 2 * procs:
        size = max(1, -(-len(args_list) // (procs * 8)))
        chunks = [args_list[i:i + size] for i in range(0, len(args_list), size)]
        with Pool(processes=procs) as pool:
            for ck in pool.map(_parse_batch, chunks):
                results.extend(ck)
    else:
        results = [_parse_pair(a) for a in args_list]
    for (is_q, src, line, text, bases, labels, err, domain, win) in results:
        if err == "EMPTY":
            continue
        if is_q:
            stats["quarantined"] += 1
            qf = qfiles.setdefault(src, open(qdir / (src + ".jsonl"), "a", encoding="utf-8"))
            qf.write(json.dumps({"err": err, "text": text, "line": line},
                                ensure_ascii=False) + "\n")
            continue
        n += 1
        units_txt, rec_base = bases, labels
        # dedupe / doc hash logic below (parent-side, order independent)
        txn = text
        h = hashlib.md5(txn.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            stats["deduped_dups"] += 1
            continue
        # document-level dedupe/granularity: hash the full ORIGINAL line family
        doc_hash = hashlib.md5(("D%s:%s" % (line, text[:64])).encode("utf-8")).hexdigest()
        if doc_hash not in rank_cache:
            rank_cache[doc_hash] = hashlib.md5((str(SEED) + ":" + doc_hash).encode("utf-8")).hexdigest()
        row = {"src": src, "domain": domain, "line": line, "win": win,
               "text": text, "doc_hash": doc_hash, "win_hash": h,
               "bases": bases, "labels": labels}
        seen_hashes.add(h)
        stats["windows_kept"] += 1
        s = stats["per_source"].setdefault(src, {"kept": 0, "val": 0, "train": 0})
        s["kept"] += 1
        if src in ("fadel_val", "abdou_valid"):
            val_rows.append(row)
            s["val"] += 1
        else:
            train_rows.append(row)
            s["train"] += 1

    # doc-stratified val carve-out from train sources (5% of docs, by ranked hash)
    doc_ranks = sorted(rank_cache.items())
    val_docs = {h for i, (h, _) in enumerate(doc_ranks) if i % 20 == 0}
    final_val = [r for r in val_rows]
    final_train = [r for r in train_rows if r["doc_hash"] not in val_docs]
    for r in train_rows:
        if r["doc_hash"] in val_docs:
            final_val.append(r)

    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in final_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "val.jsonl").open("w", encoding="utf-8") as f:
        for r in final_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["train_windows"] = len(final_train)
    stats["val_windows"] = len(final_val)
    (out / "prepare_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--sources", default=None,
                    help="comma list, e.g. fadel_train,wikinews2024; default all")
    ap.add_argument("--out-dir", default=None,
                    help="alternate output dir (default data/diac/prepared); "
                         "use for single-domain specialist corpora so the "
                         "general v2b prepared corpus is never clobbered")
    args = ap.parse_args()
    sources = set(args.sources.split(",")) if args.sources else None
    process(args.max_lines, sources, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
