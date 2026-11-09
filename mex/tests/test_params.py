# mex/tests/test_params.py
"""Param-budget acceptance test for the ME expert/control pair (ME-D2/D5).

Pins the budgets against the REAL model: llama_params() is the exact
decomposition of src/model.py's LlamaForCausalLM (tied embeddings, no
biases, RoPE parameter-free) and is asserted equal to the autograd sum
sum(p.numel()) of build_model() so the formula can never drift from the
shipped architecture (lesson 59: param anchors belong to tests).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mex.src.vocab import char_ids

EXPERT = {"layers": 2, "hidden": 80, "heads": 4, "kv_heads": 2, "ffn": 320}
CONTROL = {"layers": 2, "hidden": 160, "heads": 4, "kv_heads": 2, "ffn": 640}
VOCAB = len(char_ids())  # 97


def llama_params(cfg, vocab):
    """Exact parameter count of the real LlamaForCausalLM (src/model.py).

    Per layer: q d*d, k kvd*d, v kvd*d, o d*kvd (kvd = kv_heads * head_dim;
    GQA shares kv across head groups) -> 2*d*d + 2*d*kvd; SwiGLU MLP
    gate/up/down = 3*d*f; two RMSNorms = 2*d. Top: tied lm_head/embed
    vocab*d + final norm d. No biases anywhere; RoPE holds no parameters.
    """
    d, f, L = cfg["hidden"], cfg["ffn"], cfg["layers"]
    kvd = cfg["kv_heads"] * (d // cfg["heads"])
    attn = 2 * d * d + 2 * (d * kvd)
    mlp = 3 * d * f
    per_layer = attn + mlp + 2 * d
    return L * per_layer + vocab * d + d


def test_expert_in_band():
    p = llama_params(EXPERT, VOCAB)
    assert 100_000 <= p <= 300_000, p


def test_control_within_5pct_of_4x_expert():
    pe = llama_params(EXPERT, VOCAB)
    pc = llama_params(CONTROL, VOCAB)
    assert abs(pc - 4 * pe) <= 0.05 * 4 * pe, (pe, pc)


def test_formula_matches_real_model():
    """Pin the closed form to the shipped architecture (Task 4 Step 2)."""
    from src.model import build_model

    def cfg(hidden, ffn):
        return {"model": {"layers": 2, "hidden": hidden, "heads": 4,
                          "kv_heads": 2, "ffn": ffn, "ctx": 96,
                          "tie_embeddings": True}}

    for spec in (EXPERT, CONTROL):
        model = build_model(cfg(spec["hidden"], spec["ffn"]), vocab_size=VOCAB)
        real = sum(p.numel() for p in model.parameters())
        assert real == llama_params(spec, VOCAB), (spec, real)


import unittest  # noqa: E402


class TestParams(unittest.TestCase):
    """unittest harness over the module-level pytest-style functions."""

    def test_expert_in_band(self):
        test_expert_in_band()

    def test_control_within_5pct_of_4x_expert(self):
        test_control_within_5pct_of_4x_expert()

    def test_formula_matches_real_model(self):
        test_formula_matches_real_model()
