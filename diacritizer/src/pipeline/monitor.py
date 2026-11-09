"""ALWAYS-ON train monitor (MEMORY 2026-09-23 rule R3, plan S1).

Writes runs/diac/<phase>/monitor/log.jsonl — one record per logged step,
eval, save, VRAM probe, or anomaly — and monitor/summary.json at train end.
Watchdogs (report-only for the run, authoritative in the log):
- non-finite train loss 3 consecutive steps -> DIVERGED flag
- non-finite val metrics                      -> DIVERGED flag
- VRAM residency + reserved tracking on eval steps (WDDM spill guard)
Never raises into the train loop.
"""
import json
import math
import time
from pathlib import Path


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


class TrainMonitor:
    def __init__(self, run_dir, cfg, total_steps):
        self.dir = Path(run_dir) / "monitor"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "log.jsonl"
        self.cfg = cfg
        self.total_steps = total_steps
        self.t0 = time.time()
        self.best_vl = float("inf")
        self.best_step = -1
        self.diverged = False
        self._bad_finite = 0
        self._rec({"event": "on_start", "phase": cfg.get("phase"),
                   "total_steps": total_steps,
                   "config_keys": sorted(str(k) for k in cfg.keys())})

    def _rec(self, d):
        try:
            d = dict(d, t=round(time.time() - self.t0, 1))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        except Exception as e:
            print("[MONITOR-WRITE-FAIL]", repr(e), flush=True)

    # --- per-step finite watchdog; returns False when the loop should stop ---
    def check_step_loss(self, step, loss):
        """Train-side NaN/inf watchdog: 3 consecutive non-finite = DIVERGED."""
        if _finite(loss):
            self._bad_finite = 0
        else:
            self._bad_finite += 1
            self._rec({"event": "anomaly", "step": step,
                       "kind": "train_loss_nonfinite", "value": loss,
                       "consecutive": self._bad_finite})
            if self._bad_finite >= 3:
                self.diverged = True
                return False
        return True

    def on_step(self, step, train_loss=None, scale=None, grad_norm=None, skip=False):
        self._rec({"event": "on_step", "step": step, "train_loss": train_loss,
                   "scale": scale, "grad_norm": grad_norm, "skip": skip})

    def on_eval(self, step, vl, va, extra=None):
        d = {"event": "on_eval", "step": step, "val_loss": vl, "val_acc": va}
        if extra:
            d.update(extra)
        self._rec(d)
        if _finite(vl) and vl < self.best_vl - 1e-4:
            self.best_vl = vl
            self.best_step = step
        if not (_finite(vl) and _finite(va)):
            self.diverged = True
            self._rec({"event": "anomaly", "step": step,
                       "kind": "eval_nonfinite", "val_loss": vl, "val_acc": va})

    def on_anomaly(self, step, kind, value=None):
        self._rec({"event": "anomaly", "step": step, "kind": kind, "value": value})

    def vram(self, step=None):
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            free, total = torch.cuda.mem_get_info()
            d = {"event": "vram",
                 "used_gib": round((total - free) / 2**30, 2),
                 "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2)}
            if step is not None:
                d["step"] = step
            self._rec(d)
            return d
        except Exception as e:
            self._rec({"event": "vram_probe_failed", "err": repr(e)})
            return None

    def on_save(self, step, path):
        self._rec({"event": "on_save", "step": step, "path": str(path)})

    def on_end(self, done_step, reason="completed"):
        el = time.time() - self.t0
        self._rec({"event": "on_end", "step": done_step, "reason": reason,
                   "best_val_loss": self.best_vl, "best_step": self.best_step,
                   "diverged": self.diverged, "elapsed_s": round(el, 1)})
        summary = {"phase": self.cfg.get("phase"), "done_steps": done_step,
                   "reason": reason, "best_val_loss": self.best_vl,
                   "best_step": self.best_step, "diverged": self.diverged,
                   "elapsed_s": round(el, 1),
                   "total_steps_planned": self.total_steps}
        with open(self.dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("[MONITOR-END]", json.dumps(summary), flush=True)
