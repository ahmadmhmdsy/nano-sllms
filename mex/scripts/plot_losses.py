"""mu0c overfit analysis: read the tfevents of all runs/mex/<task>/logs dirs and
render train vs eval loss curves per task (PNG), plus an overfit-onset summary
(largest eval-loss jump window + min-eval step). Read-only on logs; CPU-only."""
import argparse
import json
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/mex")
    ap.add_argument("--out", default="runs/mex/loss_curves.png")
    a = ap.parse_args()
    tasks = []
    for d in sorted(Path(a.runs).iterdir()):
        logdir = d / "logs"
        if logdir.is_dir() and any(logdir.iterdir()):
            tasks.append((d.name, read_events(logdir)))
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10, 3 * len(tasks)), sharex=True)
    if len(tasks) == 1:
        axes = [axes]
    summary = {}
    for ax, (name, tags) in zip(axes, tasks):
        tr = tags.get("train/loss", [])
        ev = tags.get("eval/loss", [])
        if tr:
            ax.plot(*zip(*tr), label=f"{name} train", linewidth=0.8)
        if ev:
            ax.plot(*zip(*ev), label=f"{name} eval", linewidth=0.8)
        ax.set_title(name)
        ax.legend(loc="best")
        if ev:
            steps = [s for s, _ in ev]
            vals = [v for _, v in ev]
            best_i = vals.index(min(vals))
            worst_i = vals.index(max(vals))
            summary[name] = {
                "min_eval_loss": min(vals), "min_at_step": steps[best_i],
                "max_eval_loss": max(vals), "max_at_step": steps[worst_i],
                "final_eval_loss": vals[-1], "final_step": steps[-1],
                "overfit_gap_final": vals[-1] - min(vals),
            }
        if tr:
            summary.setdefault(name, {})["final_train_loss"] = tr[-1][1]
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
