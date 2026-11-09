"""Stream a small raw Python-code corpus to JSONL (train/val) for one phase.

Streaming with a hard row cap keeps network usage bounded (slow-network
constraint, PLAN.md §4.1). Candidates are tried in order; the first that
yields enough rows wins and is recorded in raw/source.txt.
Local files: a candidate of the form {path: <file>} is read directly (no
network) - .jsonl (one JSON object per line), .json (list or {rows: [...]}),
.txt (one row per line) or .csv (header row + text-ish columns).

Mix mode (TASKS.md row 13, research/pretrain_mix_proposal.md §3.2/§4.2): if
ANY candidate in data.dataset_candidates carries target_rows, the fallback
chain is replaced: every candidate must carry target_rows, ALL of them are
fetched (config order, not first-hit), each capped at its own target_rows,
and ONE SHA1 exact-dup set is shared across sources (cross-source dedupe,
§5 checklist item 11). Raw rows are written exactly like legacy mode
(data.raw_dir/{train,val}.jsonl; per source the first n_train_s rows go to
train, the rest to val - the legacy split rule applied per source).
Afterwards raw/source_stats.json records per-source {name, config,
target_rows, rows_seen, rows_kept, dups_dropped} + totals (kept rows,
overall dedupe rate). Phase dir follows the existing convention: raw_dir
comes from data.raw_dir, so stats land in data/<phase>/raw/.

Backward compatibility is sacred: a config WITHOUT any target_rows keeps the
first-hit fallback chain, single data.rows cap, fresh per-candidate dedupe
set and identical console lines; the only behavioral change is the val
split, now a seeded random sample in both modes. source_stats.json is
additive metadata written in both modes; it changes no data output.

Run: .venv/Scripts/python scripts/prepare_data.py --config configs/smoke.yaml
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    """Merge project-root .env into os.environ (real env vars win).

    Gated HF sources (bigcode/the-stack-v2, the-stack-smol, starcoderdata -
    terms already accepted) need HF_TOKEN; hf_hub/datasets do NOT read .env
    themselves (MEMORY gotcha). Same pattern as exa_search.py.
    """
    env_path = ROOT / ".env"
    if env_path.is_file():
        from dotenv import dotenv_values
        for key, val in dict(dotenv_values(env_path)).items():
            if key and val is not None:
                os.environ.setdefault(key, str(val).strip())


_load_dotenv()

TEXT_KEYS = ("content", "code", "text", "completion", "output", "answer",
             "response", "func_code_string")


def text_of(example: dict) -> str:
    for key in TEXT_KEYS:
        v = example.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def iter_local(path: Path):
    """Yield dict rows from a local .jsonl/.json/.txt/.csv file (no network)."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("rows", [data])
        for row in data:
            yield row
    elif suffix == ".txt":
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield {"text": line.rstrip("\n")}
    elif suffix == ".csv":
        import csv
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield row
    else:
        raise ValueError(f"unsupported local file type {suffix!r} "
                         "(supported: .jsonl .json .txt .csv)")


def _open_source(cand: dict, loc, name: str, sub, mode: str, raw_dir: Path):
    """Open one candidate's row iterator (exact legacy resolution semantics).

    Optional cand.data_dir passes through to load_dataset (needed for
    the-stack-smol python since 2026-09: its 'python' builder config is
    gone - only 'default' exists, so the python subdir is addressed with
    data_dir="data/python"; verified by probe 2026-09-09).
    """
    if "path" in cand:
        return iter_local(loc)
    from datasets import load_dataset

    data_dir = cand.get("data_dir")
    if mode == "download":
        print(f"[prepare] mode=download: fetching full {name} into "
              f"{raw_dir / 'hf_cache'} first (disk-bounded, not "
              "row-bounded)", flush=True)
        return load_dataset(name, sub, data_dir=data_dir, split="train",
                            streaming=False,
                            cache_dir=str(raw_dir / "hf_cache"))
    return load_dataset(name, sub, data_dir=data_dir, split="train",
                        streaming=True)


def _new_rec(name: str, sub, target_rows, requested: int) -> dict:
    return {"name": name, "config": sub, "target_rows": target_rows,
            "requested": requested, "rows_seen": 0, "rows_kept": 0,
            "dups_dropped": 0, "too_short": 0, "train_rows": 0,
            "val_rows": 0, "error": None}


def _stream_rows(ds, cap: int, seen: set, f_train, f_val, n_train: int,
                 min_chars: int, dedupe: bool, label: str, rec: dict,
                 val_indices=None, d=None) -> None:
    """Stream up to 'cap' kept rows into the OPEN train/val handles.

    Updates rec in place as rows flow, so a mid-stream failure still leaves
    partial counters for source_stats.json. seen is the caller's SHA1 set:
    fresh per candidate in legacy mode, ONE set shared across sources in
    mix mode (cross-source exact-dup, proposal §4.2/§5.11).
    """
    written = 0
    max_chars = int(d.get("max_chars", 0)) if d else 0
    require_code_keywords = bool(d.get("require_code_keywords", False)) if d else False
    code_keywords = ("def ", "class ", "import ", "return ", "for ", "if ", "print(")
    from tqdm import tqdm
    pbar = tqdm(total=cap, desc=f"[prepare] {label}")
    for ex in ds:
        if written >= cap:
            break
        rec["rows_seen"] += 1
        text = text_of(ex)
        if len(text) < min_chars:
            rec["too_short"] += 1
            continue
        if max_chars and len(text) > max_chars:
            rec["too_long"] = rec.get("too_long", 0) + 1
            continue
        if require_code_keywords and not any(kw in text for kw in code_keywords):
            rec["no_code_keywords"] = rec.get("no_code_keywords", 0) + 1
            continue
        if dedupe:
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in seen:
                rec["dups_dropped"] += 1
                continue
            seen.add(h)
        is_val = written in val_indices if val_indices is not None else written >= n_train
        if is_val:
            f_val.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            rec["val_rows"] += 1
        else:
            f_train.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            rec["train_rows"] += 1
        written += 1
        rec["rows_kept"] = written
        pbar.update(1)
        if written % 250 == 0:
            print(f"[prepare] {label}: {written}/{cap} rows", flush=True)
    pbar.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    d = cfg["data"]
    raw_dir = ROOT / d["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    cands = d["dataset_candidates"]
    mix_mode = any("target_rows" in cand for cand in cands)
    if mix_mode:
        missing = [str(cand.get("name", cand.get("path", "<unnamed>")))
                   for cand in cands if "target_rows" not in cand]
        if missing:
            raise SystemExit(
                "[prepare] mix mode requires target_rows on EVERY "
                f"dataset_candidates entry; missing on: {', '.join(missing)}")

    if mix_mode:
        n_rows = int(d.get("rows", 0))   # ignored in mix mode (per-source caps)
    else:
        n_rows = int(d["rows"])          # legacy: single cap, unchanged
    val_fraction = float(d.get("val_fraction", 0.02))
    n_val = max(1, round(n_rows * val_fraction))
    n_train = n_rows - n_val
    import random
    random.seed(int(d.get("seed", 42)))
    val_indices = set(random.sample(range(n_rows), n_val))
    min_chars = int(d.get("min_chars", 60))
    dedupe = bool(d.get("dedupe", True))
    # U7 data modes (WEBUI_PRD.md 2): "stream" (default) bounds disk by
    # rows; "download" caches the full dataset under raw_dir/hf_cache first.
    mode = d.get("data_mode", "stream")
    if mode not in ("stream", "download"):
        raise SystemExit(f"[prepare] unsupported data_mode: {mode!r}")

    out_train = raw_dir / "train.jsonl"
    out_val = raw_dir / "val.jsonl"

    stats = {
        "mode": "mix" if mix_mode else "legacy",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": args.config,
        "data_mode": mode,
        "min_chars": min_chars,
        "dedupe": dedupe,
        "val_fraction": val_fraction,
        "sources": [],
    }

    def write_stats() -> None:
        srcs = stats["sources"]
        seen_total = sum(s["rows_seen"] for s in srcs)
        kept_total = sum(s["rows_kept"] for s in srcs)
        dups_total = sum(s["dups_dropped"] for s in srcs)
        stats["totals"] = {
            "rows_seen": seen_total,
            "rows_kept": kept_total,
            "dups_dropped": dups_total,
            "too_short": sum(s["too_short"] for s in srcs),
            "train_rows": sum(s["train_rows"] for s in srcs),
            "val_rows": sum(s["val_rows"] for s in srcs),
            "dedupe_rate": round(dups_total / seen_total, 6) if seen_total else 0.0,
        }
        (raw_dir / "source_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")

    if mix_mode:
        # Fetch EVERY candidate, each capped at its own target_rows, sharing
        # one SHA1 set across sources (proposal §4.2/§5 item 11). A failing
        # source is recorded in source_stats.json and the run continues;
        # only 0 total kept rows is fatal.
        seen = set()
        f_train = out_train.open("w", encoding="utf-8")
        f_val = out_val.open("w", encoding="utf-8")
        try:
            for cand in cands:
                if "path" in cand:
                    loc = Path(cand["path"])
                    if not loc.is_absolute():
                        loc = ROOT / loc
                    name, sub = f"local:{loc.name}", None
                else:
                    name, sub = cand["name"], cand.get("config")
                    loc = None
                target = int(cand["target_rows"])
                n_val_s = max(1, round(target * val_fraction))
                n_train_s = target - n_val_s
                import random
                random.seed(int(d.get("seed", 42)))
                val_indices = set(random.sample(range(target), n_val_s))
                print(f"[prepare] mix source: {name}"
                      + (f" ({sub})" if sub else "")
                      + f" target_rows={target}", flush=True)
                rec = _new_rec(name, sub, target, target)
                stats["sources"].append(rec)
                try:
                    ds = _open_source(cand, loc, name, sub, mode, raw_dir)
                    _stream_rows(ds, target, seen, f_train, f_val, n_train_s,
                                 min_chars, dedupe, name, rec,
                                 val_indices=val_indices, d=d)
                    kept = rec["rows_kept"]
                    if kept < target:
                        print(f"[prepare] mix WARNING {name}: stream ended "
                              f"early ({kept}/{target} rows)", flush=True)
                    print(f"[prepare] mix OK {name}: {rec['train_rows']} train "
                          f"+ {rec['val_rows']} val rows (target {target})",
                          flush=True)
                except Exception as e:  # noqa: BLE001 - record and continue
                    es = str(e)
                    hint = ""
                    if "401" in es or "403" in es or "gated" in es.lower():
                        hint = (" [gated dataset - add HF_TOKEN to the "
                                "project-root .env (web UI: Settings tab), "
                                "then retry]")
                    rec["error"] = f"{type(e).__name__}: {es}"
                    print(f"[prepare] mix {name} failed: {type(e).__name__}: "
                          f"{es}{hint}", flush=True)
        finally:
            f_train.close()
            f_val.close()

        (raw_dir / "source.txt").write_text(
            "mix: " + ", ".join(
                r["name"] + (f" ({r['config']})" if r["config"] else "")
                for r in stats["sources"]),
            encoding="utf-8")
        write_stats()
        t = stats["totals"]
        print(f"[prepare] mix totals: {t['rows_kept']} rows kept from "
              f"{t['rows_seen']} seen ({t['dups_dropped']} exact dups dropped, "
              f"dedupe_rate={t['dedupe_rate']:.4f}); "
              f"{t['train_rows']} train + {t['val_rows']} val", flush=True)
        print(f"[prepare] mix stats written: {raw_dir / 'source_stats.json'}",
              flush=True)
        if t["rows_kept"] == 0:
            raise SystemExit("[prepare] mix mode kept 0 rows (every source "
                             "failed or was empty); see source_stats.json")
        return

    last_err = None
    for cand in cands:
        if "path" in cand:
            loc = Path(cand["path"])
            if not loc.is_absolute():
                loc = ROOT / loc
            name, sub = f"local:{loc.name}", None
        else:
            name, sub = cand["name"], cand.get("config")
            loc = None
        print(f"[prepare] trying dataset: {name}" + (f" ({sub})" if sub else ""), flush=True)
        rec = _new_rec(name, sub, None, n_rows)
        try:
            ds = _open_source(cand, loc, name, sub, mode, raw_dir)
            seen = set()
            f_train = out_train.open("w", encoding="utf-8")
            f_val = out_val.open("w", encoding="utf-8")
            try:
                _stream_rows(ds, n_rows, seen, f_train, f_val, n_train,
                             min_chars, dedupe, name, rec,
                             val_indices=val_indices, d=d)
            finally:
                f_train.close()
                f_val.close()
            written = rec["rows_kept"]
            if written < n_rows:
                raise RuntimeError(f"stream ended after {written}/{n_rows} rows")
            (raw_dir / "source.txt").write_text(name, encoding="utf-8")
            print(f"[prepare] OK {name}: {n_train} train + {written - n_train} val rows", flush=True)
            stats["sources"].append(rec)
            write_stats()
            return
        except Exception as e:
            last_err = e
            es = str(e)
            hint = ""
            if "401" in es or "403" in es or "gated" in es.lower():
                hint = (" [gated dataset - add HF_TOKEN to the project-root "
                        ".env (web UI: Settings tab), then retry]")
            rec["error"] = f"{type(e).__name__}: {es}"
            stats["sources"].append(rec)
            print(f"[prepare] {name} failed: {type(e).__name__}: {es}{hint}",
                  flush=True)

    write_stats()
    raise SystemExit(f"[prepare] all dataset candidates failed; last error: "
                     f"{last_err}")


if __name__ == "__main__":
    main()
