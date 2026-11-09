"""AR decoding-latency probe: the STRUCTURAL speed cost of an autoregressive
decoder on this model.

WHY. The AR/causal arm has an ACCURACY reading (E-084 cal_causal warm-start:
KILL, selector +61.4 pct relative) but the user asked for SPEED as well. Any
left-to-right AR decoder pays a cost a bidirectional encoder does not: T
sequential steps instead of ONE parallel pass. This probe times that cost on the
DEPLOYED checkpoint. It trains nothing and claims no trained-AR accuracy.

Three timings, same model, same position budget T:
  A parallel  : ONE forward on [1, T]                    -> 1 sequential step
  B ar_step   : T forwards on [1, 1] (KV-cache shape)    -> T sequential steps
  C ar_prefix : forwards on [1, t] for t = 1..T (no KV cache, O(T^2) work)
Reported per position: seconds, and the slowdown of B and C against A.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402


def sync():
    torch.cuda.synchronize()


def causal_mask(t, device):
    return torch.ones(t, t, dtype=torch.bool, device=device).tril()


def timed(step_fn, reps, warmup=1):
    for _ in range(warmup):
        step_fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        step_fn()
    sync()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/diac/e73_soup/c4b_uniform")
    ap.add_argument("--config", default="runs/diac/e67_gen/c4b_ctx256.yaml")
    ap.add_argument("--seqs", default="128,256,512")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="runs/diac/_screen_out/ar_speed_probe.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    device = "cuda"
    ck = Path(args.ckpt)
    sp = ck / "state.pt" if (ck / "state.pt").exists() else ck / "model.pt"
    state = torch.load(sp, map_location=device, weights_only=False)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    res = {"status": "EXACT-TIMING", "ckpt": str(ck), "reps": args.reps,
           "device": torch.cuda.get_device_name(0), "seqs": {}}
    for T in [int(s) for s in args.seqs.split(",")]:
        ids = torch.randint(3, TK.VOCAB_SIZE, (1, T), device=device)
        full = torch.ones(1, 1, T, T, dtype=torch.bool, device=device)
        full = full & causal_mask(T, device)[None, None]

        def par():
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                model(ids, attn_mask=full)

        one = torch.randint(3, TK.VOCAB_SIZE, (1, 1), device=device)
        m1 = torch.ones(1, 1, 1, 1, dtype=torch.bool, device=device)

        def step():
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                model(one, attn_mask=m1)

        def prefix():
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                for t in range(1, T + 1):
                    x = ids[:, :t]
                    m = torch.ones(1, 1, t, t, dtype=torch.bool, device=device)
                    m = m & causal_mask(t, device)[None, None]
                    model(x, attn_mask=m)

        tA = timed(par, args.reps)
        tB = timed(lambda: [step() for _ in range(T)], args.reps)
        tC = timed(prefix, args.reps)
        rec = {"T": T, "parallel_s": tA, "ar_step_T_s": tB, "ar_prefix_T_s": tC,
               "parallel_us_per_pos": 1e6 * tA / T,
               "ar_step_us_per_pos": 1e6 * tB / T,
               "slowdown_ar_step": tB / tA,
               "slowdown_ar_prefix": tC / tA}
        res["seqs"][str(T)] = rec
        print(json.dumps({str(T): {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in rec.items()}}, ensure_ascii=False),
              flush=True)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote " + str(out), flush=True)


if __name__ == "__main__":
    main()
