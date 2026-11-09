"""Native multi-metric train monitor: reusable, config-first, every arm.

Design (E-005 pre-registration, user mandate 2026-09-27):
- One authoritative instance per train stage, built from the parsed config
  (train.monitor / train.early_stop). No scrub code anywhere else.
- Cadence: evaluation-fraction cadence - max(1, round(max_steps * fraction))
  steps between probes (default fraction 0.1 = every 1/10 of the budget).
- VALIDATION, not test, drives every decision: held-out val_loss (already
  measured at the checkpoint cadence) feeds early stop and selection, so the
  reserved TEST split is measured read-only for observation and stays unseen
  by any decision.
- Multi-metric, never one metric. Each probe records: checkpoint val_loss +
  windowed train-loss mean (underfit signal), val direction (overfit signal:
  val_loss above its running best for >= overfit_patience probes), greedy
  chain validity on a held-out probe TASKS file (reserved rewritten
  phrasings), EOS-stop rate, mean generated length, archetype-match when the
  tasks carry an "arch" field.
- Early stop: only after min_steps of the budget AND patience consecutive
  probes without val improvement AND validity no longer improving (a validity
  that still rises outvotes a mild val rise and keeps training).
- Everything lands twice: one JSON line per probe under
  train/logs/monitor_report.jsonl and a final summary.json - artifacts,
  never console numbers (rules/evidence.md).
- The class is importable and inheritable: a future arm overrides probe() or
  keep_training() with no loop changes.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable


def _as_mapping(section: Any) -> dict[str, Any]:
    if isinstance(section, dict):
        return section
    return dict(getattr(section, "model_dump", lambda: {})() or {})


class TrainMonitorConfig:
    """Pure-logic snapshot of train.monitor / train.early_stop (GPU-free testable)."""

    def __init__(self, section: Any):
        top = _as_mapping(section)
        mon = top.get("monitor", {}) if isinstance(top.get("monitor", {}), dict) else {}
        mon = dict(_as_mapping(mon))
        self.enabled = bool(mon.get("enabled", False))
        self.eval_fraction = float(mon.get("eval_fraction", 0.1))
        self.probe_tasks_file = str(mon.get("probe_tasks_file", "") or "")
        self.probe_tasks = int(mon.get("probe_tasks", 0))
        self.probe_k = int(mon.get("probe_k", 1))
        self.probe_pass_k = int(mon.get("probe_pass_k", 1))
        self.probe_temperature = float(mon.get("probe_temperature", 0.8))
        self.probe_top_p = float(mon.get("probe_top_p", 0.95))
        self.overfit_patience = int(mon.get("overfit_patience", 2))
        self.probe_max_new_tokens = int(mon.get("probe_max_new_tokens", 512))

        es = top.get("early_stop", {}) if isinstance(top.get("early_stop", {}), dict) else {}
        es = dict(_as_mapping(es))
        self.early_stop_enabled = bool(es.get("enabled", False))
        self.patience_evals = int(es.get("patience_evals", 2))
        self.min_steps_fraction = float(es.get("min_steps_fraction", 0.5))

    def cadence_steps(self, max_steps: int) -> int:
        return max(1, min(max_steps, max(1, int(math.ceil(max_steps * self.eval_fraction)))))

    def minimum_steps_for_stop(self, max_steps: int) -> int:
        return int(max_steps * self.min_steps_fraction)


class TrainMonitor:
    """Probe + early-stop bookkeeping for the train loop (loop stays thin).

    Subclass contract: override probe() (extra metrics) or keep_training()
    (extra veto conditions) with no loop changes required.
    """

    def __init__(
        self,
        cfg: TrainMonitorConfig,
        max_steps: int,
        report_path: Path,
        summary_path: Path,
        task_probe: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.max_steps = int(max_steps)
        self.cadence = max(1, min(self.max_steps, int(math.ceil(self.max_steps * cfg.eval_fraction)) or 1))
        self.min_steps = cfg.minimum_steps_for_stop(self.max_steps)
        self.report_path = report_path
        self.summary_path = summary_path
        self.task_probe = task_probe
        self.clock = clock
        self.records: list[dict[str, Any]] = []
        self.best_val: float | None = None
        self.best_val_step = 0
        self.probes_since_best = 0
        self.validity_seen = 0.0
        self.validity_best = 0.0
        self.stop_reason: str | None = None
        self.started = self.clock()

    def should_probe(self, step: int) -> bool:
        return self.cfg.enabled and step > 0 and step % self.cadence == 0

    def record(self, step: int, val_loss: float, train_window: list[float] | None = None,
               extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        # Deduplicate: the training loop can reach this code multiple times
        # per optimizer step (gradient accumulation); only record once per step.
        if self.records and self.records[-1].get("step") == step:
            return self.records[-1]
        rec: dict[str, Any] = {
            "step": step, "val_loss": val_loss,
            "wall_s": round(self.clock() - self.started, 1),
        }
        if train_window:
            rec["train_loss_last20_mean"] = round(sum(train_window[-20:]) / min(20, len(train_window)), 5)
        if self.best_val is None or val_loss < self.best_val - 1e-6:
            self.best_val = val_loss
            self.best_val_step = step
            self.probes_since_best = 0
        else:
            self.probes_since_best += 1
        rec["val_best"] = self.best_val
        rec["probes_since_val_best"] = self.probes_since_best
        rec.update(extra or {})
        rec["overfit_signal"] = bool(
            self.best_val is not None and val_loss > self.best_val * 1.05
            and self.probes_since_best >= max(1, self.cfg.overfit_patience)
        )
        if self.task_probe is not None:
            try:
                probe = dict(self.task_probe())
                rec.update({("probe_" + k): v for k, v in probe.items()})
                validity = float(probe.get("valid", 0.0))
                self.validity_seen = validity
                was_best = self.validity_best
                self.validity_best = max(self.validity_best, validity)
                rec["validity_still_improving"] = validity > was_best + 1e-9
            except Exception as error:  # probe failure is OBSERVED, never fatal
                rec["probe_error"] = str(error)[:200]
        self._stop_decision(rec)
        self.records.append(rec)
        with self.report_path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        return rec

    def _stop_decision(self, rec: dict[str, Any]) -> None:
        if self.stop_reason:
            rec["early_stop"] = self.stop_reason
            return
        past_min = rec["step"] >= self.min_steps and self.max_steps >= 400
        no_progress = (
            self.cfg.early_stop_enabled and self.cfg.patience_evals > 0
            and self.probes_since_best >= self.cfg.patience_evals
        )
        still_improving_validity = rec.get("validity_still_improving", False)
        if no_progress and still_improving_validity:
            return  # validity keeps rising - keep training even if val stalls
        if past_min and no_progress:
            verdict = "plateau"
            self.stop_reason = (
                "early_stop plateau at step %d (val best %.5f @ step %d, "
                "probes_since_best %d)" % (rec["step"], self.best_val, self.best_val_step, self.probes_since_best)
            )
            rec["early_stop"] = self.stop_reason
            rec["early_stop_class"] = verdict

    def keep_training(self, rec: dict[str, Any]) -> bool:
        """Loop-level veto hook (override for extra keep-conditions)."""
        return self.stop_reason is None

    def summarize(self) -> dict[str, Any]:
        if not self.records:
            out = {"monitor": "no probes recorded", "cadence_steps": self.cadence}
            self.summary_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
            return out
        first, last = self.records[0], self.records[-1]
        out: dict[str, Any] = {
            "cadence_steps": self.cadence,
            "probes": len(self.records),
            "val_best": self.best_val,
            "val_best_step": self.best_val_step,
            "val_first": first.get("val_loss"),
            "val_last": last.get("val_loss"),
            "early_stop": self.stop_reason,
            "overfit_signal_any": any(bool(r.get("overfit_signal")) for r in self.records),
        }
        if all("probe_valid" in r for r in self.records):
            out["validity_best"] = self.validity_best
            out["validity_last"] = last.get("probe_valid")
            out["validity_first"] = first.get("probe_valid")
        if any("probe_pass_k" in r for r in self.records):
            out["pass_k_best"] = max(float(r.get("probe_pass_k", 0.0)) for r in self.records)
            out["pass_k_last"] = last.get("probe_pass_k")
            still_falling = (out["val_last"] is not None and out["val_best"] is not None
                             and out["val_best"] < out["val_last"] - 1e-6)
            out["underfit_signal"] = bool(self.validity_best <= 0.4 and still_falling)
            out["fit_verdict"] = (
                "underfit" if self.validity_best <= 0.4
                else ("overfit-risk" if out["overfit_signal_any"] and self.stop_reason else "healthy")
            )
        self.summary_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out
