"""tests for webui/simulator.py — replay logic against REAL run data."""
import _common  # noqa: F401

import artifacts as A
import simulator as S
from _common import run


def tb_curve(logs_dir, tag):
    """Real tfevents reader (same semantics as app.py _full_curve)."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator)
    ea = EventAccumulator(str(logs_dir), size_guidance={"scalars": 2000})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []
    evs = ea.Scalars(tag)
    return [s.step for s in evs], [s.value for s in evs]


def _rd(name, tech="Pretrain"):
    return S.load_run(name, tech, tb_curve)


def t_load_run_smoke_real():
    rd = _rd("smoke")
    assert rd.steps and rd.train, "no train curve parsed"
    assert rd.max_step >= 190, f"smoke replay ends at {rd.max_step}"
    assert rd.eval_s and rd.eval_v, "no eval curve"
    assert rd.save_steps == 50, rd.save_steps
    assert rd.summary and rd.summary.get("best_eval_loss"), rd.summary


def t_rotation_real_schedule():
    rd = _rd("smoke")
    ev = S.rotation(rd)
    assert [e["step"] for e in ev][:1] == [50], ev
    assert ev[0]["dropped"] is None
    assert ev[-1]["dropped"] == 50, "3-slot rotation did not drop the oldest"


def t_frame_stages_and_gauges():
    rd = _rd("smoke")
    f0 = S.frame_at(rd, 0, stage_override=0)
    assert "Stream" in f0["stages_html"] or "Pairs" in f0["stages_html"]
    fmid = S.frame_at(rd, 120)
    assert fmid["gauge_md"] and "120" in fmid["gauge_md"]
    assert "SIMULATION" in fmid["stages_html"], "honesty banner missing"
    fend = S.frame_at(rd, rd.max_step)
    assert ("checkpoint-200" in fend["events_md"]
            or "checkpoint-150" in fend["events_md"]), fend["events_md"]


def t_end_card_real_numbers():
    rd = _rd("smoke")
    card = S.end_card(rd)
    assert card and str(rd.summary.get("best_eval_loss"))[:4] in card, card


def t_kd_pair_delta_positive():
    kd = _rd("kd-t2p-kd", "KD")
    base = _rd("kd-t2p-baseline", "KD")
    deltas = S.kd_delta(kd, base)
    assert deltas, "no matched eval steps"
    last = deltas[-1][1]
    assert last > 0, f"expected KD ahead (baseline-eval - kd-eval > 0), got {last}"


def t_vram_estimates_anchored():
    pilot = _rd("pilot")
    full = S.vram_est_gb(pilot, "Pretrain")
    assert full is not None and abs(full - 1.91) < 0.25, full  # measured 1.91 GB
    rd_t = S.RunData("target", "Pretrain", [0], [0.0], [], [],
                     A.run_config("target"), {"params_m": 226.5}, None)
    assert abs(S.vram_est_gb(rd_t, "Pretrain") - 4.24) < 0.3   # measured 4.24
    kd_est = S.vram_est_gb(_rd("kd-t2p-kd", "KD"), "KD")
    assert kd_est is not None and abs(kd_est - 7.0) < 1.0  # measured peak 7.01
    kd_s_est = S.vram_est_gb(_rd("kd-s-t1", "KD"), "KD")
    assert kd_s_est is not None and kd_s_est < 3.5, kd_s_est \
        # pilot teacher (100.7M), NOT the target-teacher offset


def t_all_technique_runs_load():
    for tech, names in S.TECHNIQUES.items():
        for n in names:
            rd = _rd(n, tech)
            assert rd.max_step > 0, (n, "empty curve")


if __name__ == "__main__":
    run({k[2:]: v for k, v in sorted(globals().items()) if k.startswith("t_")})
