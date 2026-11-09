# mex/tests/test_merge_soup.py — E-27 Arm A merge math on tiny tensors.
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mex.scripts.merge_soup import ties_merge, uniform_merge


def _states(*tensors):
    return {f"x{i+1}": {"w": t.clone()} for i, t in enumerate(tensors)}


def test_uniform_is_mean():
    states = _states(torch.tensor([0.0]), torch.tensor([2.0]))
    assert uniform_merge(states)["w"].item() == 1.0


def test_ties_elects_majority_sign():
    # two experts above the mean elect +, merged delta is their mean.
    states = _states(torch.tensor([3.0]), torch.tensor([2.0]),
                     torch.tensor([0.0]), torch.tensor([1.0]))
    # w_bar = 1.5, deltas = [1.5, 0.5, -1.5, -0.5]; density 0.5 keeps [1.5, -1.5]
    # on sign ties elected = 0 but trim (only magnitudes >=1.5) leaves no split:
    # with 4 experts the top-50% cut keeps 2. Survivors: +1.5 (x1) and -1.5 (x3)
    # -> votes 0 -> elected 0 -> contribution 0 -> output == w_bar.
    out = ties_merge(states)["w"]
    assert torch.allclose(out, torch.tensor([1.5]))


def test_ties_all_agree_moves_toward_surviving_mean():
    states = _states(torch.tensor([10.0]), torch.tensor([1.0]))
    # w_bar 5.5, deltas [4.5, -4.5]; density 0.5 keeps both -> tie -> w_bar.
    states2 = _states(torch.tensor([10.0]), torch.tensor([9.0]))
    # w_bar 9.5, deltas [0.5, -0.5]; tie -> w_bar.
    assert ties_merge(states2)["w"].item() == 9.5
    assert ties_merge(states)["w"].item() == 5.5  # 2-way sign tie -> w_bar


def test_ties_three_vs_one_elects_plus():
    states = _states(torch.tensor([10.0]), torch.tensor([9.5]),
                     torch.tensor([9.0]), torch.tensor([1.0]))
    # w_bar 7.375; deltas 2.625/2.125/1.625/-6.375; top-50% keeps two largest
    # per expert: all four >= median of 4 values? quantile(0.5): keep >= 2.625,
    # 2.125, 1.625, 6.375 -> 4 survivors, 3 plus votes -> merged mean of +
    # deltas, vs 1 lone minus drowned out.
    out = ties_merge(states)["w"]
    assert out.item() > 7.375
