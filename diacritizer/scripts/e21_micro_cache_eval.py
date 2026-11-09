"""E-21 — micro-model cache experiment (the user's train-time-cache idea).

The word cache (models/e19/our_word_cache.json) is built from the SAME raw
pool the v3 tokens were made of, so it is part of the training process's
 artifacts. This script answers: does cache-merge at eval help a MICRO model
more than it helped the 30M gold (which gained only -0.3..-0.6 DER)?

Flow: for the given run dir, snapshot best.pt into gate_probe/, produce RAW
bench preds for the 4 gates (the exact historical pipeline), merge the cache
over each raw pred (word-aligned), and score BOTH against bench.py's own
.ref.txt files. Writes <run_dir>/cache_eval.json.

Usage: & .\.venv\Scripts\python.exe -X utf8 diacritizer/scripts/e21_micro_cache_eval.py runs/diac/micro_a
"""
import json, re, subprocess, sys
from pathlib import Path
import torch, yaml

REPO = Path(__file__).resolve().parent.parent.parent
GATES = ["fadel_test", "sadeed25", "wikinews2024", "wikinews2014"]
DIAC = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670"
STRIP = re.compile("[" + DIAC + "]")
KEYMAP = str.maketrans({"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627", "\u0649": "\u064a", "\u0640": ""})

def bare_key(w):
    return STRIP.sub("", w).translate(KEYMAP)

def eval_score(pred_path, ref_path):
    r = subprocess.run([sys.executable, "-X", "utf8", str(REPO / "diacritizer/scripts/eval.py"),
                        "compare", "--pred", str(pred_path), "--ref", str(ref_path)],
                       capture_output=True, text=True)
    j = json.loads(r.stdout)["aggregate"]
    return {"DER": j["DER"] * 100, "WER": j["WER"] * 100}

def main():
    run_dir = Path(sys.argv[1])
    probe = run_dir / "gate_probe"
    ck = run_dir / "best.pt"
    snap = torch.load(ck, map_location="cpu", weights_only=False)
    probe_c = snap.get("config")
    if isinstance(probe_c, str) and probe_c:
        cfgd = json.loads(probe_c)
    else:
        cfgd = cfg_of(run_dir)
    torch.save({"model": snap.get("model", snap), "step": snap.get("step", -1),
                "config": json.dumps(cfgd)}, probe / "model.pt")
    (probe / "probe_config.yaml").write_text(yaml.safe_dump(cfgd, allow_unicode=True), encoding="utf-8")
    py = sys.executable
    bench = str(REPO / "diacritizer" / "scripts" / "bench.py")
    for g in GATES:
        pred = probe / f"e21_{g}_pred.txt"
        r1 = subprocess.run([py, bench, "--ckpt", str(probe), "--src", g, "--out", str(pred),
                             "--config", str(probe / "probe_config.yaml")], capture_output=True, text=True)
        if r1.returncode != 0:
            print("[BENCH-FAIL]", g, (r1.stderr or "")[-300:], flush=True)
            continue
        raw = pred.read_text(encoding="utf-8").splitlines()
        cache = json.loads((REPO / "models/e19/our_word_cache.json").read_text(encoding="utf-8"))
        VI = cache["words"]
        ref_path = pred.with_suffix(".ref.txt")
        ref_lines = ref_path.read_text(encoding="utf-8").splitlines()
        merged = []
        for pr, rl in zip(raw, ref_lines):
            if len(pr.split()) != len(rl.split()):
                merged.append(pr)
                continue
            ws = []
            for pw, rw in zip(pr.split(), rl.split()):
                k = bare_key(rw)
                v = VI.get(k)
                dv = next((w for w in v if bare_key(w) == k), None) if v else None
                ws.append(dv if dv else pw)
            merged.append(" ".join(ws))
        mc = probe / f"e21_{g}_cache.pred.txt"
        mc.write_text("\n".join(merged) + "\n", encoding="utf-8")
        out = {"raw": eval_score(pred, ref_path), "cache": eval_score(mc, ref_path)}
        print(g, out, flush=True)

def cfg_of(run_dir):
    return yaml.safe_load((run_dir / "final" / "config.yaml").read_text(encoding="utf-8"))

main()
