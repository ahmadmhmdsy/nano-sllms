
"""Count parameters for different model configurations using the actual model code."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))

from model import build_from_config
import tokenizer as TK

VOCAB_SIZE = TK.VOCAB_SIZE

def count_params(layers, hidden, ffn, n_heads=8, n_kv=2, ctx=128, vocab_size=VOCAB_SIZE):
    """Build a model and count its parameters."""
    cfg = {
        "layers": layers,
        "hidden": hidden,
        "ffn": ffn,
        "n_heads": n_heads,
        "n_kv": n_kv,
        "ctx": ctx,
    }
    model = build_from_config(cfg, vocab_size)
    return sum(p.numel() for p in model.parameters())

# Existing e63s models
existing = [
    ("e63s_s1_12l272", 12, 272, 1088),
    ("e63s_s2_10l272", 10, 272, 1088),
    ("e63s_s3_8l272", 8, 272, 1088),
    ("e63s_s4_12l160", 12, 160, 640),
    ("e63s_s5_10l128", 10, 128, 512),
    ("e63s_s6_8l96", 8, 96, 384),
]

print("=== Existing e63s models (verified) ===")
for name, layers, hidden, ffn in existing:
    params = count_params(layers, hidden, ffn)
    print(f"{name}: {layers}L, {hidden}h, {ffn}ffn -> {params:,} params ({round(params/1e6, 2)}M)")

# Candidate ~1M param configurations (hidden must be divisible by 8)
print("\n=== Candidate ~1M param configurations ===")
candidates = [
    ("cand_10l88h352", 10, 88, 352),
    ("cand_12l80h320", 12, 80, 320),
    ("cand_14l72h288", 14, 72, 288),
    ("cand_16l64h256", 16, 64, 256),
    ("cand_18l56h224", 18, 56, 224),
    ("cand_20l48h192", 20, 48, 192),
    ("cand_22l40h160", 22, 40, 160),
    ("cand_24l32h128", 24, 32, 128),
    ("cand_8l100h400", 8, 100, 400),
    ("cand_6l108h432", 6, 108, 432),
    ("cand_5l120h480", 5, 120, 480),
    ("cand_4l140h560", 4, 140, 560),
    ("cand_6l112h448", 6, 112, 448),
    ("cand_4l128h512", 4, 128, 512),
]

for name, layers, hidden, ffn in candidates:
    try:
        params = count_params(layers, hidden, ffn)
        print(f"{name}: {layers}L, {hidden}h, {ffn}ffn -> {params:,} params ({round(params/1e6, 2)}M)")
    except Exception as e:
        print(f"{name}: {layers}L, {hidden}h, {ffn}ffn -> ERROR: {e}")

# Find configs closest to 1M
print("\n=== Configs closest to 1M params ===")
all_configs = []
for name, layers, hidden, ffn in candidates:
    try:
        params = count_params(layers, hidden, ffn)
        all_configs.append((name, layers, hidden, ffn, params))
    except:
        pass
all_configs.sort(key=lambda x: abs(x[4] - 1_000_000))
for name, layers, hidden, ffn, params in all_configs[:8]:
    print(f"{name}: {layers}L, {hidden}h, {ffn}ffn -> {params:,} params ({round(params/1e6, 2)}M)")
