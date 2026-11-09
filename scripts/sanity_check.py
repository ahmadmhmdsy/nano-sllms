"""Fast no-dataset sanity check: tokenizer load, model build, fwd/bwd on GPU.
Run: .venv/Scripts/python scripts/sanity_check.py --config configs/smoke.yaml
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    args = ap.parse_args()
    import torch
    import yaml
    from transformers import AutoTokenizer

    from src.model import build_model, maybe_wrap_peft

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"]["name"])
    vocab = max(int(cfg["tokenizer"]["vocab_size"]), len(tok))
    model = build_model(cfg, vocab_size=vocab)
    model = maybe_wrap_peft(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ids = torch.randint(0, vocab, (2, int(cfg["model"]["ctx"])), device=device)
    with torch.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
        out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    gnorm = sum(p.grad.norm() ** 2 for p in model.parameters()
                if p.grad is not None) ** 0.5
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    enc = tok("def hello():\n    return 42")["input_ids"]
    dec = tok.decode(enc)
    ok = torch.isfinite(out.loss).item()

    print(f"{'PASS' if ok else 'FAIL'} model_build  params={n_params / 1e6:.1f}M device={device}")
    print(f"{'PASS' if ok else 'FAIL'} fwd_bwd      loss={float(out.loss):.4f} grad_norm={float(gnorm):.2f}")
    print(f"{'PASS' if device == 'cuda' else 'FAIL'} gpu          peak_vram_gb={peak:.2f}")
    print(f"{'PASS' if enc and dec else 'FAIL'} tokenizer    vocab={len(tok)} roundtrip={dec!r}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
