"""Reusable runtime-profile store (batch/throughput knobs + SOP knobs).

Wrap the experience from running smoke/bench sweeps into a persistent
structured registry so later processes pick the right profile with the
least steps and least mistakes:

- same process + same model shape + same data scope -> auto reuse the
  saved config; the agent may also select explicitly (profile_ctl.py).
- every entry carries its evidence (sweep source, wall time, VRAM,
  parity result) and notes explaining WHY it holds.
- CLI: profile_ctl.py list|show|recommend|record|promote|retire.

Files: registry.json (data) + store.py (logic). No third-party deps.
"""
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.json"

STATUS_ORDER = {"validated": 0, "fallback": 1, "experimental": 2, "retired": 3}


def model_bucket(cfg):
    """cfg -> canonical shape bucket, e.g. '16Lx272' (from model cfg dict)."""
    m = (cfg or {}).get("model", cfg or {})
    return "%sLx%s" % (m.get("layers", 0), m.get("hidden", 0))


class ProfileStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else REGISTRY
        self._data = None

    def _load(self):
        if self._data is None:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        return self._data

    def _save(self):
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        self._data = None

    # CRUD --------------------------------------------------------------
    def list(self, status=None):
        return [e for e in self._load()["entries"]
                if status is None or e.get("status") == status]

    def show(self, eid):
        for e in self._load()["entries"]:
            if e["id"] == eid:
                return e
        raise KeyError("no profile id %r" % eid)

    def record(self, process, params, match, evidence, notes="",
               status="experimental", eid=None):
        ent = {"id": eid or ("%s.%s" % (process, time.strftime("%Y%md%H%M"))),
               "process": process, "status": status, "match": match,
               "params": params, "evidence": evidence, "notes": notes}
        self._load()["entries"].append(ent)
        self._save()
        return ent

    def promote(self, eid, status="validated"):
        e = self.show(eid)
        e["status"] = status
        self._save()
        return e

    def retire(self, eid, reason=""):
        e = self.show(eid)
        e["status"] = "retired"
        if reason:
            e["notes"] = (e.get("notes") or "") + " [retired %s: %s]" % (
                time.strftime("%Y-%m-%d"), reason)
        self._save()
        return e

    # the workhorse ------------------------------------------------------
    def recommend(self, process, cfg=None, device="sm_75_6gib"):
        """Best runnable profile for a process.

        Matching order: (1) exact model-bucket validated entry, else
        (2) any_model validated, else (3) explicit fallback. Each entry
        carries its own evidence/notes for instant re-audit.
        Returns (entry, why) or (None, None).
        """
        bucket = model_bucket(cfg)
        cands = [e for e in self.list() if e["process"] == process]
        def probe(e):
            m = e.get("match") or {}
            if m.get("device") and m["device"] != device:
                return None
            if STATUS_ORDER.get(e.get("status"), 9) > 1:
                return None            # retired / experimental not auto-run
            if m.get("model_bucket") == bucket:
                return "exact"
            if m.get("any_model"):
                return "any"
            return None
        exact = [e for e in cands if probe(e) == "exact"]
        anys = [e for e in cands if probe(e) == "any"]
        for pool, why in ((exact, "exact-model-bucket"), (anys, "any-model")):
            if pool:
                pool.sort(key=lambda e: (STATUS_ORDER[e["status"]], e["id"]))
                return pool[0], why
        return None, None

    def batch_for(self, process, cfg=None, device="sm_75_6gib"):
        """Shortcut used by bench/train daemons: returns an int batch or None."""
        ent, _ = self.recommend(process, cfg, device)
        if ent is None:
            return None
        return (ent.get("params") or {}).get("batch")