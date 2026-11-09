
"""Evaluate all e63s models on WikiNews 2024 multi-ref."""
import sys, os, re, json, unicodedata, yaml, torch
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
from model import build_from_config
import tokenizer as TK
from labels import marks_for_label
from passthrough import is_arabic_base
import bench as B

# Strip diacritics
BAND = "[ًٌٍَُِّْٰٴۖ-ۭ]"
strip_re = re.compile(BAND)

def strip_marks(t):
    return strip_re.sub("", t)

# Read WikiNews 2024 multi-ref file
wn2024_path = REPO / "data" / "diac" / "raw" / "wikinews" / "wikinews2024_multi_ref.diac"
lines = []
for line in wn2024_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    bare = strip_marks(line)
    lines.append(bare)

print(f"Processing {len(lines)} lines...", flush=True)

# Models to evaluate
models = [
    ("e63s_s1_12l272", "runs/diac/e63s_s1_12l272/final/model.pt", "runs/diac/e63s_s1_12l272/final/config.yaml"),
    ("e63s_s2_10l272", "runs/diac/e63s_s2_10l272/final/model.pt", "runs/diac/e63s_s2_10l272/final/config.yaml"),
    ("e63s_s3_8l272", "runs/diac/e63s_s3_8l272/final/model.pt", "runs/diac/e63s_s3_8l272/final/config.yaml"),
    ("e63s_s4_12l160", "runs/diac/e63s_s4_12l160/final/model.pt", "runs/diac/e63s_s4_12l160/final/config.yaml"),
    ("e63s_s5_10l128", "runs/diac/e63s_s5_10l128/final/model.pt", "runs/diac/e63s_s5_10l128/final/config.yaml"),
    ("e63s_s6_8l96", "runs/diac/e63s_s6_8l96/final/model.pt", "runs/diac/e63s_s6_8l96/final/config.yaml"),
]

device = "cuda"
results = {}

for name, ckpt_path, config_path in models:
    print(f"\n=== Evaluating {name} ===", flush=True)
    
    # Load model
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model = build_from_config(cfg["model"], TK.VOCAB_SIZE).to(device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    ctx = cfg["model"]["ctx"]
    n_params = sum(p.numel() for p in model.parameters())
    
    # Generate predictions using predict_bares (plural)
    outs = []
    for i, line in enumerate(lines):
        try:
            pred = B.predict_bares(model, [(line, [])], ctx, device, batch=64)[0]
            outs.append(pred)
        except Exception as e:
            outs.append(f"ERROR: {str(e)[:100]}")
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(lines)}]", flush=True)
    
    # Write predictions
    out_path = REPO / "models" / "e19" / f"wn2024.{name}.pred.txt"
    out_path.write_text("\n".join(outs) + "\n", encoding="utf-8")
    
    results[name] = {
        "params": n_params,
        "params_m": round(n_params / 1e6, 2),
        "layers": cfg["model"]["layers"],
        "hidden": cfg["model"]["hidden"],
        "ffn": cfg["model"]["ffn"],
        "ctx": cfg["model"]["ctx"],
    }
    print(f"  Params: {n_params:,} ({round(n_params/1e6, 2)}M)", flush=True)
    print(f"  Written to {out_path}", flush=True)

# Save results summary
results_path = REPO / "models" / "e19" / "e63s_model_info.json"
results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nResults saved to {results_path}", flush=True)
