
"""Find valid ~1M param configurations with even head_dim for RoPE."""
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

# Valid configurations: hidden must be divisible by 16 (n_heads=8, even head_dim)
# FFN ratio = 4.0
print("=== Valid ~1M param configurations (hidden divisible by 16) ===")
candidates = [
    ("cand_14l80h320", 14, 80, 320),
    ("cand_12l80h320", 12, 80, 320),
    ("cand_10l80h320", 10, 80, 320),
    ("cand_16l64h256", 16, 64, 256),
    ("cand_18l64h256", 18, 64, 256),
    ("cand_20l48h192", 20, 48, 192),
    ("cand_22l48h192", 22, 48, 192),
    ("cand_24l32h128", 24, 32, 128),
    ("cand_8l96h384", 8, 96, 384),  # Same as e63s_s6
    ("cand_6l112h448", 6, 112, 448),
    ("cand_4l128h512", 4, 128, 512),
    ("cand_5l120h480", 5, 120, 480),
]

for name, layers, hidden, ffn in candidates:
    try:
        params = count_params(layers, hidden, ffn)
        head_dim = hidden // 8
        print(f"{name}: {layers}L, {hidden}h, {ffn}ffn, head_dim={head_dim} -> {params:,} params ({round(params/1e6, 2)}M)")
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
    head_dim = hidden // 8
    print(f"{name}: {layers}L, {hidden}h, {ffn}ffn, head_dim={head_dim} -> {params:,} params ({round(params/1e6, 2)}M)")
