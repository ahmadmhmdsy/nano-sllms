"""CPU-only tests for src/mount.py. Run via the project venv python; expect: test_mount: ALL PASS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn
from src.mount import (gate_strength, severance_pct, MountBridge,
                       WrappedLayer, attach_mounts, detach_mounts)
from transformers import LlamaConfig, AutoModelForCausalLM

# --- gate / severance math ---
assert gate_strength(0, warmup=10, hold=20, anneal_end=40) == 0.0
assert gate_strength(10, warmup=10, hold=20, anneal_end=40) == 1.0
assert abs(gate_strength(30, warmup=10, hold=20, anneal_end=40) - 0.5) < 1e-9
assert gate_strength(40, warmup=10, hold=20, anneal_end=40) == 0.0
assert severance_pct(0, unlink_start=100, stage_len=10, init=10, cap=100) == 0.0
assert severance_pct(100, unlink_start=100, stage_len=10, init=10, cap=100) == 10.0
assert severance_pct(9999, unlink_start=100, stage_len=10, init=10, cap=100) == 100.0

# --- zero-init bridge == exact identity (step-0 contract, Flamingo gating;
#     F1: out_proj keeps nn.MultiheadAttention default init, a := 0) ---
b = MountBridge(768, heads=6)
x = torch.randn(2, 8, 768)
assert torch.equal(b(x, 0.0), x)  # strength 0 -> early-return, bitwise identity
b._current_kv = torch.randn(2, 12, 768)
assert torch.allclose(b(x, 1.0), x, atol=1e-6)  # strength 1, a=0 -> tanh(a)=0
assert torch.equal(b.a, torch.zeros_like(b.a))  # zero-init a untouched

# --- severance deterministic per (seed, pct); 3-of-6 heads at 50% ---
b.set_severance(50.0, seed=42); k1 = b._sev_keep.clone()
b.set_severance(50.0, seed=42)
assert torch.equal(b._sev_keep, k1)
b.set_severance(50.0, seed=43)
assert not torch.equal(b._sev_keep, k1)  # different seed -> different mask
b.set_severance(50.0, seed=42)
assert int(k1.sum()) == 3  # 6 heads at 50 percent
assert abs(b._sev_rescale - 2.0) < 1e-12  # 1/(1-0.5) DARE rescale

# --- WrappedLayer preserves the orig layer's container type (transformers
#     5.16.1 LlamaDecoderLayer.forward returns a plain tensor, not a tuple) ---
class PlainLayer(nn.Module):
    def forward(self, h):
        return h * 2.0

plain = PlainLayer()
br = MountBridge(8, heads=2)
br._strength = 0.0  # bridge early-returns -> out == core == h*2
wrapped = WrappedLayer(plain, br)
h = torch.randn(2, 3, 8)
out = wrapped(h)
assert isinstance(out, torch.Tensor) and not isinstance(out, tuple), \
    "plain-tensor layer must dispatch a plain tensor, not (out,)"
assert torch.equal(out, h * 2.0)

# tuple-returning orig layer: (out,)+rest preserved
class TupleLayer(nn.Module):
    def forward(self, h):
        return h * 2.0, torch.ones(1)

br._strength = 0.0
tout = WrappedLayer(TupleLayer(), br)(h)
assert isinstance(tout, tuple) and len(tout) == 2 and torch.equal(tout[0], h * 2.0)

# zero-init: bridge contributes nothing even at strength 1 (tanh(a)=0)
br._strength = 1.0
br._current_kv = torch.randn(2, 3, 8)
out2 = wrapped(h)
assert torch.is_tensor(out2) and torch.allclose(out2, h * 2.0)
# after training a away from zero, the bridge mixes kv into the output
with torch.no_grad():
    br.a.fill_(0.1)
out3 = wrapped(h)
assert torch.is_tensor(out3) and not torch.allclose(out3, h * 2.0)

# --- wrap / attach / detach on a real tiny Llama model (4L, hidden 64,
#     anchors teacher_layers=6) ---
cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=16)
m = AutoModelForCausalLM.from_config(cfg)
bridges = attach_mounts(m, teacher_layers=6)
assert len(bridges) == 4
assert [br.teacher_anchor for br in bridges] == [1, 2, 4, 5]  # round((l+0.5)*6/4)
assert isinstance(m.model.layers[0], WrappedLayer)
assert isinstance(m._mount_bridges, list)   # bridges registered ONCE, in their WrappedLayer (mount.py) — plain-list registry, no ModuleList dup

ids = torch.randint(0, 64, (2, 16))
for br in bridges:
    br._strength = 0.5
    br._current_kv = torch.randn(2, 5, 64)
m(input_ids=ids, labels=ids).loss.backward()
# a got a gradient through the tanh(a) gate with strength > 0
assert any(br.a.grad is not None and br.a.grad.abs() > 0 for br in bridges), \
    "zero-init a must remain trainable (nonzero grad)"

n_before = sum(p.numel() for p in m.parameters())
detach_mounts(m)
n_after = sum(p.numel() for p in m.parameters())
assert n_after < n_before
n_ref = sum(p.numel() for p in AutoModelForCausalLM.from_config(cfg).parameters())
assert n_after == n_ref, "detach must restore the exact original param set"
assert m._mount_bridges is None
assert not any(isinstance(l, WrappedLayer) for l in m.model.layers)
# model still runs after detach
m(input_ids=ids, labels=ids)

print("test_mount: ALL PASS")
