"""mu0c loss tracker: parse TensorBoard event files for every runs/mex/<task>/logs
directory and print a live summary; optionally write a flat CSV per task
(train/eval loss vs step) for later overfit/underfit analysis. CPU-only and
read-only on the log dirs — safe to run while training."""
import argparse
import csv
from pathlib import Path


def read_events(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ev = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ev.Reload()
    out = {}
    for tag in ev.Tags()["scalars"]:
        out[tag] = sorted({s.step: s.value for s in ev.Scalars(tag)}.items())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/mex")
    ap.add_argument("--csv", default="runs/mex/loss_track.csv")
    a = ap.parse_args()
    runs = Path(a.runs)
    rows = []
    for d in sorted(runs.iterdir()):
        logdir = d / "logs"
        if not (logdir.is_dir() and any(logdir.iterdir())):
            continue
        try:
            tags = read_events(logdir)
        except Exception as e:
            print(f"{d.name}: SKIP ({e})")
            continue
        train_tag = next((t for t in tags if t == "train/loss" or t == "loss"), None)
        eval_tags = [t for t in tags if "eval" in t.lower() and "loss" in t.lower()]
        for step, v in tags.get(train_tag, []):
            rows.append([d.name, "train", step, v])
        for t in eval_tags:
            for step, v in tags[t]:
                rows.append([d.name, "eval", step, v])
        if train_tag:
            last_tr = tags[train_tag][-1]
            msg = f"{d.name}: train@{last_tr[0]}={last_tr[1]:.4f}"
            if eval_tags:
                last_ev = tags[eval_tags[0]][-1]
                msg += f"  eval@{last_ev[0]}={last_ev[1]:.4f}"
            print(msg)
        else:
            print(f"{d.name}: no train loss tag ({list(tags)[:4]})")
    if rows:
        out = Path(a.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task", "split", "step", "loss"])
            w.writerows(rows)
        print(f"wrote {out} ({len(rows)} rows)")
    else:
        print("no loss points found")


if __name__ == "__main__":
    main()
