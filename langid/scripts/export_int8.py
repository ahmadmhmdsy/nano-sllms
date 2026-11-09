"""Export fp32 model -> per-language int8 npz artifact + latency benchmark."""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src.data import LANGS
from langid.src.infer import Int8LangID, quantize_columns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/langid_da1/model_fp32.pt")
    ap.add_argument("--out", default="runs/langid_da1/langid_int8.npz")
    args = ap.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    W = state["emb"].numpy().astype(np.float64)
    bias = state["bias"].numpy().astype(np.float64)
    Wq, scale = quantize_columns(W)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, W=Wq, scale=scale, bias=bias,
                        langs=np.array(LANGS))
    size = out.stat().st_size
    print(f"[export] {out} {size} bytes ({size / 1048576:.2f} MiB)")

    model = Int8LangID(str(out))
    words = ["hello", "bonjour", "guten", "mundo", "strana", "merhaba"] * 50
    t0 = time.perf_counter()
    for w in words:
        model.predict(w)
    dt = (time.perf_counter() - t0) / len(words)
    print(f"[bench] {dt * 1000:.3f} ms/word over {len(words)} words (CPU numpy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
