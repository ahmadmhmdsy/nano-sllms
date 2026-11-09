"""D-line training script (R53/R54). SAME discipline contract as the M3 line:
- fp16 only (Turing sm_75: no bf16, no flash-attn -> SDPA used in model.py)
- AUTO-RESUME ZERO-FLAG CONTRACT: after any crash/kill, re-run the EXACT
  command with zero extra flags; it picks up the newest checkpoint under
  runs/diac/<phase>/checkpoint-* (model + optimizer + step).
- single-GPU exclusivity: never launch while another train.py runs.
- checkpoint rotation save_total_limit=3; weights stay local (gitignore).

Usage:
    .venv/Scripts/python.exe diacritizer/scripts/train.py --config configs/diac_smoke.yaml
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
from model import build_from_config  # noqa: E402
import tokenizer as TK  # noqa: E402
try:
    from pipeline import TrainMonitor  # noqa: E402
    from pipeline.evals import evaluate_unseen  # noqa: E402
    from pipeline.smoke import preflight_smoke  # noqa: E402
except ImportError as _pi:  # pipeline optional: trainer stays runnable without it
    TrainMonitor = None
    preflight_smoke = None
    print("[PIPELINE-MISSING]", _pi, flush=True)

RUNS = REPO / "runs" / "diac"
DATA = REPO / "data" / "diac"
GATES = ("fadel_test", "sadeed25", "wikinews2024", "wikinews2014")
GATE_EXTERNAL = ("fadel_test", "sadeed25", "wikinews2014")  # wn2024 = contaminated stamp (MEMORY 56); arms see the probe CSV for it either way



def _newton_schulz5(G, steps=5):
    """Muon's Newton-Schulz orthogonalization (fp32).

    E-63 NaN root cause: the raw polynomial amplifies sigma ~1.46x per step in
    fp32 and blows up super-exponentially for sigma>~1.3 (norm -> 1e9+ on
    random square grads). Fix: normalize X every iteration (keeps the map in
    its contraction basin) and rescale the result to Frobenius norm
    sqrt(min(m, n)) so the update scale matches the canonical unit-singular-
    value expectation.
    """
    a, b, c = 3.4445, -4.7750, 2.7885
    X = G.float() / (G.float().norm() + 1e-7)
    if X.shape[0] > X.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        X = X / (X.norm() + 1e-7)
    X = X * (min(X.shape[0], X.shape[1]) ** 0.5)
    return X if G.shape[0] <= G.shape[1] else X.T


class Muon(torch.optim.Optimizer):
    """Minimal Muon: momentum + orthogonalized update on 2D dense weights."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        super().__init__(list(params), {"lr": lr, "momentum": momentum,
                                        "nesterov": nesterov})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, mom, nest = group["lr"], group["momentum"], group["nesterov"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(g)
                m = st["mom"]
                m.mul_(mom).add_(g)
                eff = g.add(m, alpha=mom) if nest else m
                p.add_(_newton_schulz5(eff), alpha=-lr)
        return loss


def latest_checkpoint(run_dir):
    ckpts = sorted(run_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    return ckpts[-1] if ckpts else None


def load_np(names, tokens_dir):
    return (np.load(str(tokens_dir / (names[0] + ".npy"))),
            np.load(str(tokens_dir / (names[1] + ".npy"))))


def evaluate_batch(model, val_ids, val_y, device, batch):
    """Delegates to pipeline.evals (single eval entry, per-class included)."""
    vl, va, info = evaluate_unseen(model, val_ids, val_y, device, 128)
    evaluate_batch.last_info = info
    return vl, va

evaluate_batch.last_info = None




def disk_free_gb():
    import shutil as _sh
    return _sh.disk_usage(str(REPO)).free / (1024 ** 3)


def disk_warn(tag, warn_gb=3.0, prune=False):
    """User-requested disk guard: shout [DISK-WARN] when E: gets tight (the next
    save may fail mid-run); with prune=True, auto-delete per-probe
    weights_step{N}.pt EXCEPT the two newest + best_gate_weights.pt. Real
    checkpoint rotation + the gate-best artifact are never touched."""
    import shutil as _sh
    free = _sh.disk_usage(str(REPO)).free / (1024 ** 3)
    if free >= warn_gb:
        return free
    print(f"[DISK-WARN] {tag}: only {free:.2f} GB free on E: - sweeping per-probe weight snapshots ...", flush=True)
    if prune:
        cands = []
        for rd in RUNS.iterdir():
            gp = rd / "gate_probe"
            if gp.is_dir():
                cands += list(gp.glob("weights_step*.pt"))
        cands.sort(key=lambda f: int(f.stem.replace("weights_step", "")))
        for f in cands[:-2]:
            try:
                f.unlink()
                print("[DISK-PRUNE]", f, flush=True)
            except Exception:
                pass
    return free

def gate_probe(cfg, model, step, run_dir):
    """Bench the CURRENT live weights on the 4 registered gates + append CSV.

    User-requested overfit watchdog: rows append runs/<phase>/gate_eval.csv,
    every gate_eval_every steps + at the final step. Durability (arm-B incident:
    rotation deleted the live best checkpoint while the probe marked it gold):
    every probe writes a permanent weights_step{N}.pt and the mean-external-DER
    tracker maintains best_gate_weights.pt (superseded only by a better probe).
    Subprocesses: bench.py + eval.py = the EXACT historical gate pipeline.
    """
    import csv, shutil
    probe_dir = run_dir / "gate_probe"
    probe_dir.mkdir(exist_ok=True)
    snap = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "step": step, "config": json.dumps(cfg)}
    if disk_free_gb() < 3.0:
        disk_warn(f"gate probe @ step {step}", warn_gb=3.0, prune=True)

    torch.save(snap, probe_dir / "model.pt")                       # bench input (reused)
    torch.save(snap, probe_dir / f"weights_step{step}.pt")          # permanent per-probe
    cfg_path = probe_dir / "probe_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    py = sys.executable
    bench = str(REPO / "diacritizer" / "scripts" / "bench.py")
    ev = str(REPO / "diacritizer" / "scripts" / "eval.py")
    csv_path = run_dir / "gate_eval.csv"
    fields = ["step", "gate", "DER", "WER", "DER_nocase",
              "text_preservation", "lines", "pred"]
    new_file = not csv_path.exists()
    ders = []
    for src in GATES:
        pred = probe_dir / f"step{step}_{src}_pred.txt"
        r1 = subprocess.run([py, bench, "--ckpt", str(probe_dir), "--src", src,
                             "--out", str(pred), "--config", str(cfg_path),
                             "--batch", str(int(cfg.get("gate_probe_batch", 64)))],
                            capture_output=True, text=True)
        if r1.returncode != 0:
            print("[GATE-PROBE-FAIL]", src, (r1.stderr or "")[-300:], flush=True)
            continue
        ref = pred.with_suffix(".ref.txt")
        r2 = subprocess.run([py, ev, "compare", "--pred", str(pred), "--ref", str(ref)],
                            capture_output=True, text=True)
        try:
            agg = json.loads(r2.stdout)["aggregate"]
        except Exception:
            print("[GATE-EVAL-FAIL]", src, (r2.stderr or r2.stdout)[-400:], flush=True)
            continue
        row = {"step": step, "gate": src, "DER": agg["DER"], "WER": agg["WER"],
               "DER_nocase": agg.get("DER_nocase"),
               "text_preservation": agg["text_preservation"],
               "lines": agg.get("lines", ""), "pred": str(pred)}
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(row)
        print(f"[GATE] step {step} {src} DER {agg['DER']*100:.2f} "
              f"pres {agg['text_preservation']:.4f}", flush=True)
        ders.append((src, agg["DER"]))
    # external-gate best tracker: mean DER over GATE_EXTERNAL
    ext = [d for s, d in ders if s in GATE_EXTERNAL]
    if ext:
        mean_der = sum(ext) / len(ext)
        best_file = probe_dir / "best_gate.json"
        prev = None
        if best_file.exists():
            try:
                prev = json.loads(best_file.read_text(encoding="utf-8"))
            except Exception:
                prev = None
        if not prev or mean_der < float(prev.get("mean_der", 9.0)) - 1e-6:
            best_file.write_text(json.dumps({"step": step, "mean_der": mean_der}), encoding="utf-8")
            shutil.copyfile(probe_dir / f"weights_step{step}.pt",
                            probe_dir / "best_gate_weights.pt")
            print(f"[GATE-BEST] step {step} mean_external_DER {mean_der*100:.2f} "
                  "-> best_gate_weights.pt", flush=True)



def _build_optimizer(model, cfg):
    """adamw (default, byte-identical contract) | muon (2D hidden weights via
    Muon + AdamW on embeddings/head/1D params)."""
    opt_name = str(cfg.get("optimizer", "adamw")).lower()
    if opt_name == "muon":
        muon_ps, adam_ps = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (muon_ps if (p.ndim >= 2 and "embed" not in n and "head" not in n)
             else adam_ps).append(p)
        opt = Muon(muon_ps, lr=float(cfg.get("muon_lr", 0.025)))
        opt2 = torch.optim.AdamW(adam_ps, lr=cfg.get("lr", 1e-4),
                                 weight_decay=cfg.get("weight_decay", 0.1))
        print({"optimizer": "muon", "muon_params": len(muon_ps),
               "adam_params": len(adam_ps),
               "muon_lr": cfg.get("muon_lr", 0.025)}, flush=True)
        return opt, opt2
    assert opt_name == "adamw", "unknown optimizer " + repr(opt_name)
    return torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4),
                             weight_decay=cfg.get("weight_decay", 0.1),
                             fused=False), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto",
                    help="cpu = never touch the GPU (safe beside any active run)")
    ap.add_argument("--skip-smoke", metavar="REASON", default=None,
                    help="bypass the automated 60-step preflight smoke; a reason "
                         "is mandatory and gets recorded (MEMORY 2026-09-23 R4)")
    ap.add_argument("--seed", type=int, default=None,
                    help="deterministic SCREEN seed: seeds weight init, dropout and "
                         "the data order together. Unset = zero seeding calls = "
                         "legacy behaviour byte-identical. A paired screen needs the "
                         "arm and its control to share one seed (E-084 triage bed).")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    phase = cfg["phase"]
    # SCREEN SEEDING (E-084 triage bed). Weight init is UNSEEDED here today, which
    # is exactly the 0.82 percent init draw C0-R measured; the data order was
    # already fixed at default_rng(777). A few-minute screen is only viable as a
    # PAIRED comparison - arm and control start from the SAME weights and see the
    # SAME data order, so init drift cancels and the only thing left is the arm's
    # own mechanism. seed unset -> no torch seeding call at all, so every
    # historical run stays reproducible as-is.
    seed = args.seed if args.seed is not None else cfg.get("seed")
    if seed is not None:
        seed = int(seed)
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Seeding the RNGs is NOT enough for a paired screen: the CUDA kernels
        # are nondeterministic on their own. E-084 calibration measured a NULL
        # replay diverging from its own control by 0.138 percent relative on the
        # selector with EVERY RNG shared, because SDPA backward and cuDNN select
        # and reduce differently run to run. Forcing the deterministic paths is
        # what turns "same seed" into "same numbers". Guarded on seed is not
        # None, so every unseeded historical run stays byte-identical.
        import os as _os
        det = []
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            det.append("cudnn")
        except Exception as exc:
            det.append("cudnn-unavailable:%s" % exc)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            det.append("algorithms")
        except Exception as exc:
            det.append("algorithms-unavailable:%s" % exc)
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            det.append("sdp-math")
        except Exception as exc:
            det.append("sdp-unavailable:%s" % exc)
        print({"screen_seed": seed, "screen_determinism": det}, flush=True)
    # pipeline validation floor: reject unknown experimental feature names early
    try:
        from pipeline.registry import features_for_config
        eff = features_for_config(cfg)
        print({"pipeline_features": eff}, flush=True)
    except ImportError:
        pass
    run_dir = RUNS / phase
    run_dir.mkdir(parents=True, exist_ok=True)
    # SMOKE PREFLIGHT (MEMORY 2026-09-23 R4): end-to-end 60-step smoke with the
    # REAL config before any real train; auto-skipped for smoke phases / resume.
    if preflight_smoke is not None:
        verdict = preflight_smoke(args, cfg, run_dir, REPO)
        print({"smoke_preflight": verdict}, flush=True)
        if verdict.startswith("pass"):
            run_dir.mkdir(parents=True, exist_ok=True)
    # tokens_phase: reuse another phase's packed tokens (e.g. b65 trains on the
    # v2b matched-128-window corpus) while keeping a separate run dir.
    tokens_dir = DATA / cfg.get("tokens_phase", phase) / "tokens"
    train_ids, train_y = load_np(("train_ids", "train_y"), tokens_dir)
    val_ids, val_y = load_np(("val_ids", "val_y"), tokens_dir)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else "cpu"
    model = build_from_config(cfg, vocab_size=TK.VOCAB_SIZE).to(device)
    # pretrained_init: Stage-1 char-LM warm-start (plan 2026-09-13 sec 2.2).
    # strict=False: the LM state has no 15-class label-head counterpart -
    # embeddings + encoder stack load, heads proceed fresh. Skipped on a
    # checkpoint resume (the resumed state overrides this init anyway).
    if not latest_checkpoint(run_dir) and cfg.get("pretrained_init"):
        pi = REPO / cfg["pretrained_init"]
        st = torch.load(pi, map_location="cpu")
        sd = st.get("model", st) if isinstance(st, dict) else st
        # shape-filtered warm transfer (E-63s lesson 2026-09-23): strict=False
        # tolerates MISSING keys, not SHAPE mismatches - a smaller hidden arm
        # crashed the whole init. Keep every tensor whose shape fits, skip the
        # rest (they train from scratch), so partial warm start never hard-fails.
        msd = model.state_dict()
        fit = {k: v for k, v in sd.items()
               if k in msd and tuple(msd[k].shape) == tuple(v.shape)}
        dropped = len(sd) - len(fit)
        # D-12 GUARD (2026-10-06): a warm start that loads ZERO tensors is not a
        # partial warm start, it is a silent NO-OP - the run proceeds from random
        # init while the config, the report and the registry all claim a warm
        # start. That is how E-065 F1/F2 (the project best, 0.3974) and E-066 R1
        # were actually trained (272-hidden stage1lm vs the 64-hidden s7). A
        # NONZERO-but-partial load stays allowed on purpose: the E-63s sub-272
        # arms depend on shape-filtered partial transfer. Only a total wipe-out
        # is fatal, and it can be waived explicitly.
        if len(sd) and not fit and not cfg.get("allow_empty_warm_start", False):
            raise SystemExit(
                f"[FATAL] pretrained_init={pi} matched 0 of {len(sd)} tensors "
                f"(shape_skipped={dropped}). This is a silent warm-start NO-OP "
                f"(defect D-12), not a warm start. Fix pretrained_init, or set "
                f"allow_empty_warm_start: true to start from random init on purpose.")
        if dropped:
            print({"pretrained_init_partial": str(pi), "loaded": len(fit),
                   "shape_skipped": dropped}, flush=True)
        missing, unexpected = model.load_state_dict(fit, strict=False)
        # verify the transfer is REALLY the encoder stack, not a silent skip
        print({"pretrained_init": str(pi), "tensors_in": len(sd),
               "missing": len(missing), "unexpected": len(unexpected),
               "shape_skipped": dropped}, flush=True)
        # reset-last-N (CATT counterfactual arm B, user-activated 2026-09-14):
        # re-initialize the LAST N encoder blocks from a FRESH Block of the same
        # shape (their default-init params) - embeddings + earlier stack stay
        # warm, the top of the stack re-learns from the bidirectional domain.
        n_reset = int(cfg.get("reset_last_n_layers", 0))
        if n_reset and len(getattr(model, "_mod_uses", [])) > len(model.layers):
            print({"reset_last_n_layers": "SKIPPED: shared-layer lease (blocks "
                   "cycled per-use; per-index reset not defined)"}, flush=True)
        elif n_reset:
            from model import Block
            mm = cfg["model"]
            for bi in range(len(model.layers) - n_reset, len(model.layers)):
                fb = Block(mm["hidden"], mm["n_heads"], mm["n_kv"],
                           mm.get("ffn") or 4 * mm["hidden"], causal=False,
                           kv_mode=model.layers[bi].attn.kv_mode)
                model.layers[bi].load_state_dict(fb.state_dict())
                fb = None
            print({"reset_last_n_layers": n_reset, "blocks": len(model.layers)}, flush=True)
    # E-63 loss variants (config: loss.label_smoothing / loss.logit_adjust_tau)
    lc = cfg.get("loss") or {}
    model.loss_label_smooth = float(lc.get("label_smoothing", 0.0))
    model.loss_logit_adj = None
    tau = float(lc.get("logit_adjust_tau", 0.0))
    if tau:
        yy = np.asarray(train_y[train_y >= 0])[::17].ravel().astype(np.int64)
        freq = np.bincount(yy, minlength=15).astype(np.float64)
        prior = np.maximum(freq, 1.0) / max(freq.sum(), 1.0)
        model.loss_logit_adj = torch.tensor(
            -tau * np.log(prior), dtype=torch.float32, device=device)
    print({"loss_variant": {"label_smoothing": model.loss_label_smooth,
                            "logit_adjusted": tau}}, flush=True)
    # fp16 COMPUTE on Turing via autocast (pure-fp16 master weights break
    # GradScaler: it requires fp32 grads); NaN-ban = scaled loss + skip-on-overflow
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    n_params = sum(p.numel() for p in model.parameters())
    print({"phase": phase, "device": device, "params": n_params,
           "train_windows": len(train_ids), "val_windows": len(val_ids)})

    opt, opt2 = _build_optimizer(model, cfg)
    bs = cfg.get("batch_size", 16)
    accum = cfg.get("accum", 1)
    total_steps = cfg.get("total_steps", 1000)
    patience_log = cfg.get("log_every", 20)
    rotation = 3
    mon = TrainMonitor(run_dir, cfg, total_steps) if TrainMonitor else None

    # AUTO-RESUME zero-flag contract
    step = 0
    ck = latest_checkpoint(run_dir)
    if ck:
        state = torch.load(ck / "state.pt", map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if opt2 is not None and state.get("opt2"):
            opt2.load_state_dict(state["opt2"])
        step = state["step"]
        print(f"[RESUME] from {ck.name} step {step}")
    else:
        print("[START] fresh (zero flags, checkpoint rotates every save_every)")

    save_every = cfg.get("save_every", 100)
    patience = cfg.get("early_stop_patience", 12)   # evals without improvement
    best_vl = float("inf")
    bad_evals = 0
    best_path = run_dir / "best.pt"
    if best_path.exists():
        st = torch.load(best_path, map_location=device, weights_only=False)
        best_vl = float(st.get("best_val_loss", best_vl))  # survives resume
    model.train()
    t0 = time.time()
    rng = np.random.default_rng(777 if seed is None else int(seed))
    use_amp = device == "cuda"
    while step < total_steps:
        idx = rng.integers(0, len(train_ids), bs)
        x = torch.from_numpy(train_ids[idx]).to(device)
        y = torch.from_numpy(train_y[idx]).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x, y)
        loss = out["loss"] / accum
        scaler.scale(loss).backward()          # SCALED backward (AMP contract)
        scaler.unscale_(opt)
        if opt2 is not None:
            scaler.unscale_(opt2)
        loss_sum = out["loss"]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        if opt2 is not None:
            scaler.step(opt2)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if opt2 is not None:
            opt2.zero_grad(set_to_none=True)
        if scaler.get_scale() < scale_before:
            if mon: mon.on_step(step, skip=True)  # overflowomics: scale will drop
            continue  # overflowed batch -> skip logging/step counting
        step += 1
        if not (mon is None or mon.check_step_loss(step, float(loss_sum.item()))):
            print("[DIVERGED] non-finite train loss 3 consecutive -> stopping", flush=True)
            break
        if mon and step % patience_log == 0:
            mon.on_step(step, train_loss=round(float(loss_sum.item()), 4),
                        scale=round(float(scaler.get_scale()), 1))
        if step % cfg.get("eval_every", patience_log) == 0:
            vl, va = evaluate_batch(model, val_ids, val_y, device, None)
            info = evaluate_batch.last_info or {}
            if mon:
                mon.on_eval(step, vl, va, extra={"per_class_acc": info.get("per_class_acc")})
                mon.vram(step)
            el = time.time() - t0
            print(f"step {step} loss {float(loss_sum):.4f} val_loss {vl:.4f} "
                  f"val_acc {va:.4f} elapsed {el:.0f}s", flush=True)
            # EARLY STOP: patience evals without a new best val_loss.
            if vl < best_vl - 1e-4:
                best_vl = vl
                bad_evals = 0
                torch.save({"model": model.state_dict(), "step": step,
                            "best_val_loss": best_vl, "config": json.dumps(cfg)},
                           best_path)
                print(f"[BEST] val_loss {best_vl:.4f} saved -> {best_path.name}", flush=True)
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    print({"early_stop": step, "best_val_loss": best_vl,
                           "best_pt": str(best_path), "reason": "no val_loss improvement "
                           f"for {bad_evals} evals"}, flush=True)
                    if mon: mon.on_end(step, reason="early_stop")
                    break
        if step % save_every == 0 or step == total_steps:
            if disk_free_gb() < 3.0:
                disk_warn(f"checkpoint save @ step {step}", warn_gb=3.0, prune=True)

            sd = run_dir / f"checkpoint-{step}"
            sd.mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "opt2": (opt2.state_dict() if opt2 is not None else None),
                        "step": step, "config": json.dumps(cfg)},
                       sd / "state.pt")
            allc = sorted(run_dir.glob("checkpoint-*"),
                          key=lambda p: int(p.name.split("-")[-1]))
            while len(allc) > rotation:
                oldest = allc.pop(0)
                if oldest.exists():
                    import shutil
                    shutil.rmtree(oldest)
            print(f"[SAVE] {sd}")
            if mon: mon.on_save(step, sd)
        # IN-TRAINING REAL-GATE PROBE (overfit watchdog): every gate_eval_every
        # steps AND at the final step, score the 4 external gates with the live
        # weights; rows append to runs/<phase>/gate_eval.csv (user request).
        if cfg.get("gate_eval_every", 0) and (step % cfg["gate_eval_every"] == 0
                                              or step == total_steps):
            gate_probe(cfg, model, step, run_dir)
    model_dir = run_dir / "final"
    model_dir.mkdir(exist_ok=True)
    # Final export = the BEST val_loss weights when early stopping decided so.
    if best_path.exists() and best_vl < float("inf"):
        st = torch.load(best_path, map_location=device, weights_only=False)
        torch.save(st["model"], model_dir / "model.pt")
    else:
        torch.save(model.state_dict(), model_dir / "model.pt")
    # cfg yaml copy beside the bare state_dict so bench.py can rebuild any arch
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    if mon: mon.on_end(step, reason="completed")
    print({"done": step, "final": str(model_dir), "best_val_loss": best_vl})


if __name__ == "__main__":
    main()
