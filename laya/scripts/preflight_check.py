r"""Pre-flight gate: run BEFORE any unattended long training launch.

Catches the failure classes that cost the E-65 session six resume events:
  C1 config   duplicate YAML keys (PyYAML silently keeps the LAST) + config echo
  C2 static   pyflakes undefined-name scan of the trainer (argparse-class NameErrors)
  C3 entry    --help subprocess smoke of the trainer (exit 0 required)
  C4 data     every *.pt the config points at loads, with item counts
  C5 runtime  trainer --smoke 1-batch GPU run when supported (exit 0 required)

Usage:
  & .\.venv\Scripts\python.exe laya/scripts/preflight_check.py --script laya/scripts/train_l3.py --config configs/laya_l3_ladder.yaml
Exit 0 = all clear; exit 1 = at least one FAIL.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

_DUPES = []


def _no_dup_mapping(loader, node):
    mapping = {}
    for knode, vnode in node.value:
        key = loader.construct_object(knode, deep=True)
        if key in mapping:
            _DUPES.append(str(key))
        mapping[key] = loader.construct_object(vnode, deep=True)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="trainer .py to validate")
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_gpu", action="store_true")
    args = ap.parse_args()

    results = []

    def check(tag, status, detail):
        results.append(status)
        print("[%s] %-6s %s" % (status, tag, detail), flush=True)

    # C1 config: duplicate keys + echo
    import yaml
    loader = type("StrictLoader", (yaml.SafeLoader,), {})
    loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup_mapping)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=loader)
    if _DUPES:
        check("C1-config", "FAIL", "duplicate YAML keys (PyYAML would keep the last): %s" % sorted(set(_DUPES)))
    else:
        check("C1-config", "PASS", "no duplicate keys; %d keys" % len(cfg))
    print('       effective config: %s' % json.dumps({k: cfg[k] for k in sorted(cfg) if not str(k).startswith('_')}, default=str)[:600], flush=True)

    # C2 static: undefined names
    if importlib.util.find_spec("pyflakes") is None:
        check("C2-static", "SKIPPED", "pyflakes not installed (pip install pyflakes into .venv)")
    else:
        r = subprocess.run([sys.executable, "-m", "pyflakes", args.script],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            check("C2-static", "PASS", "pyflakes clean")
        elif "undefined name" in (r.stdout or ""):
            check("C2-static", "FAIL", (r.stdout or "").strip()[:400])
        else:
            check("C2-static", "PASS", "pyflakes notes (non-blocking): " + (r.stdout or "").strip()[:180])

    # C3 entrypoint smoke
    r = subprocess.run([sys.executable, args.script, "--help"], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        check("C3-entry", "PASS", "--help exit 0")
    else:
        check("C3-entry", "FAIL", (r.stderr or r.stdout).strip()[-400:])

    # C4 data pointers
    import torch
    pts = [v for v in cfg.values() if isinstance(v, str) and v.endswith(".pt") and os.path.exists(v)]
    bad = []
    for p in pts:
        try:
            obj = torch.load(p, weights_only=False, map_location="cpu")
            n = len(obj) if hasattr(obj, "__len__") else "?"
            print("       data %s -> %s items" % (os.path.basename(p), n), flush=True)
        except Exception as e:
            bad.append("%s: %s" % (os.path.basename(p), str(e)[:80]))
    if bad:
        check("C4-data", "FAIL", "; ".join(bad))
    elif not pts:
        check("C4-data", "SKIPPED", "no *.pt paths in config")
    else:
        check("C4-data", "PASS", "%d tensors load" % len(pts))

    # C5 runtime smoke (trainer must support --smoke)
    if args.skip_gpu:
        check("C5-runtime", "SKIPPED", "--skip_gpu")
    else:
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        try:
            r = subprocess.run([sys.executable, args.script, "--config", args.config, "--smoke"],
                               capture_output=True, text=True, timeout=1500, env=env)
            if r.returncode == 0:
                tail = r.stdout.strip().splitlines()[-6:]
                check("C5-runtime", "PASS", "smoke exit 0 | " + " || ".join(tail)[:300])
            elif "unrecognized arguments: --smoke" in (r.stderr or ""):
                check("C5-runtime", "SKIPPED", "trainer has no --smoke support")
            else:
                tail = (r.stderr or r.stdout).strip().splitlines()
                check("C5-runtime", "FAIL", " | ".join(tail[-4:])[:400])
        except subprocess.TimeoutExpired:
            check("C5-runtime", "FAIL", "smoke timed out (1500s) - trainer hung")

    fails = results.count("FAIL")
    print("PREFLIGHT: %d PASS / %d FAIL / %d SKIPPED -> exit %d" %
          (results.count("PASS"), fails, results.count("SKIPPED"), 1 if fails else 0), flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()