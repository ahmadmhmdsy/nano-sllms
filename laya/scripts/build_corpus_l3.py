r"""E-65: build the domain-matched pretraining corpus for the L3 encoder.

Reads laya/scripts/l3_sources.json (a ranked manifest produced by the data
sweep): [{name, id, config?, split, n_items, render, fields?}...]. Renders
documents to plain text, normalizes, HARD-dedups (a) within-corpus and
(b) against every eval set used by E-62/E-63/E-64 (typed-decisions test,
AreLit/PhishNChips core, AG News test, dair-ai/emotion test) by normalized-text
hash - eval items are downloaded ONLY to build the exclusion hash set and are
never written to the corpus.

Usage: & .\.venv\Scripts\python.exe laya/scripts/build_corpus_l3.py --config configs/laya_l3.yaml
"""
import argparse, hashlib, json, os, sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_SOURCES = [
    {"name": "typed_decisions_test", "id": "LocalLLaMA/typed-decisions", "config": "all",
     "split": "test", "render": "typed_state"},
    {"name": "phishnchips_eval", "id": "AreLit/PhishNChips", "config": "emails",
     "split": "core", "render": "email_content"},
    {"name": "ag_news_test", "id": "fancyzhx/ag_news", "split": "test", "render": "field",
     "fields": ["text"]},
    {"name": "emotion_test", "id": "dair-ai/emotion", "split": "test", "render": "field",
     "fields": ["text"]},
]


def norm(t):
    return " ".join(str(t).lower().split())


def h(t):
    return hashlib.sha1(norm(t).encode("utf-8")).hexdigest()


def render_email_content(r):
    c = r["email_content"]
    try:
        d = json.loads(c)
        if isinstance(d, dict):
            return "\n".join("%s: %s" % (k, v) for k, v in d.items() if v)
    except Exception:
        pass
    return str(c)


def render_typed_state(r):
    parts = []
    for k in ("state", "input", "text", "context"):
        if k in r and r[k]:
            parts.append(str(r[k]))
    return "\n".join(parts) if parts else json.dumps(r, ensure_ascii=False)[:2000]


def render_field(r, fields):
    return "\n".join("%s" % (r[f],) for f in fields if r.get(f))


def doc_from_row(r, spec):
    mode = spec.get("render", "field")
    if mode == "email_content":
        return render_email_content(r)
    if mode == "typed_state":
        return render_typed_state(r)
    if mode == "tokens_join":
        for f in spec.get("fields", ["tokens"]):
            v = r.get(f)
            if isinstance(v, list) and v:
                return " ".join(str(x) for x in v if x)
        return ""
    if mode == "json_row":
        return json.dumps(dict(r), ensure_ascii=False, default=str)
    return render_field(r, spec.get("fields", ["text"]))


def probe_sources(manifest_path):
    """Pre-build gate (E-65 reflection N5): 1-row load probe of every manifest source.

    Catches script-dataset removal (datasets>=3) and bad configs BEFORE a long build.
    Falls back to the refs/convert/parquet mirror once per source. Exits 1 if any
    source fails - fix or remove the manifest entry first (never silently skip)."""
    import json as _json
    with open(manifest_path, "r", encoding="utf-8") as f:
        specs = _json.load(f)
    fails = []
    for spec in specs:
        name = spec.get("name", spec.get("id", "?"))
        base = {"split": spec.get("split", "train")}
        if spec.get("config"):
            base["name"] = spec["config"]
        ok = False
        err = ""
        for rev in (spec.get("revision"), "refs/convert/parquet"):
            try:
                kw = dict(base)
                if rev:
                    kw["revision"] = rev
                if rev == "refs/convert/parquet":
                    kw.pop("name", None)  # parquet mirrors flatten named configs
                ds = load_dataset(spec["id"], **kw)
                print("[probe] PASS %-46s %d rows%s" % (name, len(ds), (" rev=" + rev) if rev else ""), flush=True)
                ok = True
                break
            except Exception as e:
                err = str(e)[:110]
        if not ok:
            print("[probe] FAIL %-46s %s" % (name, err), flush=True)
            fails.append(name)
    print("[probe] %d/%d sources load -> exit %d" % (len(specs) - len(fails), len(specs), 1 if fails else 0), flush=True)
    raise SystemExit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l3.yaml")
    ap.add_argument("--probe_sources", default="",
                    help="path to manifest json: 1-item load probe per source, then exit (pre-build gate)")
    args = ap.parse_args()
    if args.probe_sources:
        probe_sources(args.probe_sources)
        return
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = os.path.join("data", "laya", "l3_corpus")
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    stats = {"eval_exclusion_hashes": 0, "sources": {}}

    # 1) eval exclusion hashes (never written anywhere else)
    ban = set()
    for spec in EVAL_SOURCES:
        try:
            kw = {"split": spec["split"]}
            if spec.get("config"):
                kw["name"] = spec["config"]
            if spec.get("revision"):
                kw["revision"] = spec["revision"]
            ds = load_dataset(spec["id"], **kw)
            n0 = len(ban)
            for r in ds:
                d = doc_from_row(r, spec)
                if d and len(d) > 20:
                    ban.add(h(d))
                    ban.add(h(d[:200]))
            stats["eval_exclusion_hashes"] = len(ban)
            print("[l3corpus] eval set %s: %d ban hashes (total %d)"
                  % (spec["name"], len(ban) - n0, len(ban)), flush=True)
        except Exception as e:
            stats["sources"]["EVALFAIL_" + spec["name"]] = str(e)[:200]
            print("[l3corpus] WARN eval set %s failed: %s" % (spec["name"], e), flush=True)

    # 2) corpus from manifest
    with open("laya/scripts/l3_sources.json", "r", encoding="utf-8") as f:
        sources = json.load(f)
    docs = []
    seen = set()
    for spec in sources:
        got, dup, overlap = 0, 0, 0
        try:
            kw = {"split": spec.get("split", "train")}
            if spec.get("config"):
                kw["name"] = spec["config"]
            if spec.get("revision"):
                kw["revision"] = spec["revision"]
            ds = load_dataset(spec["id"], **kw)
            n_rows = len(ds)
            stride = max(1, n_rows // max(1, int(spec["n_items"])))
            dpf = spec.get("drop_prefixes", [])
            dpc = spec.get("drop_contains", [])
            for ri, r in enumerate(ds):
                if got >= int(spec["n_items"]):
                    break
                if stride > 1 and ri % stride != 0:
                    continue
                d = doc_from_row(r, spec)
                if not d or len(d) < 40:
                    continue
                if any(d.startswith(p) for p in dpf) or any(c in d for c in dpc):
                    continue
                dh = h(d)
                if dh in ban or h(d[:200]) in ban:
                    overlap += 1
                    continue
                if dh in seen:
                    dup += 1
                    continue
                seen.add(dh)
                docs.append(d)
                got += 1
            stats["sources"][spec["name"]] = {"id": spec["id"], "docs": got,
                                              "dropped_dup": dup,
                                              "dropped_eval_overlap": overlap}
            if overlap:
                raise SystemExit("[l3corpus] FATAL: %d eval-overlap docs in source %s "
                                 "- refusing to build (eval data must never enter training)"
                                 % (overlap, spec["name"]))
            print("[l3corpus] %s: %d docs (dup %d, eval-overlap %d)"
                  % (spec["name"], got, dup, overlap), flush=True)
        except Exception as e:
            stats["sources"]["FAIL_" + spec["name"]] = str(e)[:200]
            print("[l3corpus] FAIL %s: %s" % (spec["name"], e), flush=True)

    # 3) tokenize + pack
    rng = np.random.default_rng(42)
    order = rng.permutation(len(docs))
    all_ids = []
    for i in range(0, len(order), 512):
        chunk = [docs[j] for j in order[i:i + 512]]
        enc = tok([d[:20000] for d in chunk], add_special_tokens=False)["input_ids"]
        for ids in enc:
            all_ids.extend(ids)
            all_ids.append(tok.sep_token_id)
    n_hold = int(len(all_ids) * 0.02)
    hold = np.array(all_ids[:n_hold], dtype=np.uint16)
    tr = np.array(all_ids[n_hold:], dtype=np.uint16)
    tr.tofile(os.path.join(out_dir, "train.bin"))
    hold.tofile(os.path.join(out_dir, "heldout.bin"))
    stats["_tokens_train_m"] = round(len(tr) / 1e6, 1)
    stats["_tokens_heldout_m"] = round(len(hold) / 1e6, 1)
    stats["_docs"] = len(docs)
    with open(os.path.join(out_dir, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("[l3corpus] total %.1fM train tokens / %.2fM heldout (%d docs)"
          % (len(tr) / 1e6, len(hold) / 1e6, len(docs)), flush=True)


if __name__ == "__main__":
    main()
