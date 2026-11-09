"""profile_ctl.py - least-step CLI over pipeline.profiles.ProfileStore.

Common uses:
  list:       profile_ctl.py list [--status validated]
  show:       profile_ctl.py show <id>
  recommend:  profile_ctl.py recommend gate_bench --config runs/diac/<run>/final/config.yaml [--json]
  record:     profile_ctl.py record gate_bench \
                  --match-json /tmp/m.json --params-json /tmp/p.json \
                  --evidence-json /tmp/e.json [--as-status experimental]
  promote:    profile_ctl.py promote <id> [--as-status validated]
  retire:     profile_ctl.py retire <id> --reason "..."

JSON-blob args read from FILES to avoid shell quoting traps.
Exit codes: 0 ok | 1 no match | 2 bad usage.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from pipeline.profiles import ProfileStore  # noqa: E402


def _read_json_arg(path_or_json):
    p = Path(path_or_json)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(path_or_json)


def main():
    ap = argparse.ArgumentParser(prog="profile_ctl")
    ap.add_argument("cmd")
    ap.add_argument("pos", nargs="*")
    ap.add_argument("--status", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--match-json", default=None)
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--evidence-json", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--reason", default="")
    ap.add_argument("--as-status", default="validated")
    a = ap.parse_args()
    st = ProfileStore()
    if a.cmd == "list":
        for e in st.list(a.status):
            print("%-52s %-11s %s" % (e["id"], e["status"],
                                      (e.get("params") or {}).get("batch", "-")))
        return 0
    if a.cmd == "show":
        print(json.dumps(st.show(a.pos[0]), indent=2))
        return 0
    if a.cmd == "recommend":
        cfg = None
        if a.config:
            import yaml
            cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        ent, why = st.recommend(a.pos[0], cfg)
        if ent is None:
            print("NO-PROFILE: nothing runnable for %r - use `record` first" % a.pos[0])
            return 1
        if a.json:
            print(json.dumps({"entry": ent, "why": why}))
        else:
            print("%s [chosen: %s] params=%s" % (ent["id"], why, ent.get("params")))
        return 0
    if a.cmd == "record":
        if not (a.match_json and a.params_json and a.evidence_json):
            print("record needs --match-json --params-json --evidence-json (files or inline)")
            return 2
        ent = st.record(a.pos[0], _read_json_arg(a.params_json),
                        _read_json_arg(a.match_json),
                        _read_json_arg(a.evidence_json),
                        notes=a.notes, status=a.as_status)
        print(ent["id"])
        return 0
    if a.cmd in ("promote", "retire"):
        if a.cmd == "promote":
            e = st.promote(a.pos[0], a.as_status)
        else:
            e = st.retire(a.pos[0], a.reason)
        print(e["id"])
        return 0
    print("unknown cmd %r" % a.cmd)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
