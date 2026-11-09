"""CPU pre-launch config check: build the model a training config declares.

Catches architecture errors (head/width divisibility, bad key names) BEFORE a GPU
window is spent. Also reports the parameter count so a matched-parameter claim is
computed rather than asserted.

CPU only. Builds nothing on the GPU and writes no files.

Usage:
  & .venv/Scripts/python.exe research/verdict_audit/tools/check_config.py \
      runs/diac/e79_gen/a_26l64.yaml runs/diac/e79_gen/c_18l64.yaml
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402

KNOWN_KEYS = {
    "accum", "batch_size", "early_stop_patience", "eval_every",
    "gate_eval_every", "gate_probe_batch", "log_every", "lr", "model",
    "optimizer", "phase", "pretrained_init", "save_every", "seed",
    "tokens_phase", "total_steps", "warmup_steps", "weight_decay",
}
MODEL_KEYS = {"ctx", "ffn", "hidden", "layers", "n_heads", "n_kv"}


def main():
    bad = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        unknown = sorted(set(cfg) - KNOWN_KEYS)
        m = cfg.get("model", {})
        unknown_m = sorted(set(m) - MODEL_KEYS)
        try:
            model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE)
            n = sum(q.numel() for q in model.parameters())
            built = "OK"
        except Exception as exc:  # noqa: BLE001
            n = 0
            built = "FAIL %s: %s" % (type(exc).__name__, exc)
            bad += 1
        print("%s" % p)
        print("  phase=%s tokens=%s steps=%s batch=%s"
              % (cfg.get("phase"), cfg.get("tokens_phase"),
                 cfg.get("total_steps"), cfg.get("batch_size")))
        print("  model=%s" % {k: m.get(k) for k in sorted(MODEL_KEYS)})
        print("  pretrained_init=%s" % cfg.get("pretrained_init", "<none: random init>"))
        print("  params=%d  build=%s" % (n, built))
        if unknown:
            print("  UNKNOWN top-level keys: %s" % unknown)
        if unknown_m:
            print("  UNKNOWN model keys: %s" % unknown_m)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
