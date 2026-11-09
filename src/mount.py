"""Mounting bridges (design 2026-09-11 sections 5-6): frozen-teacher
cross-attention wired into each student decoder layer, trained, then
progressively unlinked. Everything config-gated; DEFAULT OFF everywhere else
in the repo. NOT-SHIP SHORTCUT: never pass bridge state through kwargs into
the original decoder layer - the trainer sets b._current_kv / b._strength /
b._current_pad attributes per batch BEFORE the student forward (attribute
injection only)."""
from __future__ import annotations
import math

import torch
import torch.nn as nn


def gate_strength(step: int, *, warmup: int, hold: int, anneal_end: int) -> float:
    """Design section 6: 0->1 warm-in, hold, linear anneal 1->0, then 0."""
    if step <= warmup:
        return step / max(warmup, 1)
    if step <= hold:
        return 1.0
    if step <= anneal_end:
        return max(0.0, 1.0 - (step - hold) / (anneal_end - hold))
    return 0.0


def severance_pct(step: int, *, unlink_start: int, stage_len: int,
                  init: float, cap: float) -> float:
    """Mount B ladder: init percent at unlink_start, +10 per stage_len steps, capped."""
    if step < unlink_start:
        return 0.0
    k = (step - unlink_start) // stage_len
    return min(cap, init + 10.0 * k)


class MountBridge(nn.Module):
    """y = layer_out + strength * tanh(a) * tanh(MHA(q=y, kv=t_kv)).
    a := learned scalar, zero-init, so tanh(a)=0 and step 0 is an exact
    identity (Flamingo gating). Mount B severance zeroes whole MHA heads;
    surviving heads rescale by 1/(1-p) (DARE rule) so head mass stays calibrated."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.a = nn.Parameter(torch.zeros(()))
        self.heads, self.dim = heads, dim
        self._strength = 0.0
        self._sev_keep = None
        self._sev_rescale = 1.0
        self._current_kv = None
        self._current_pad = None
        # out_proj keeps nn.MultiheadAttention default init (F1 fix, user
        # decision): with a := 0 zero-init, tanh(a)=0 makes step 0 a bitwise
        # identity regardless of out_proj, and the bridge stays trainable
        # (a.grad != 0; zero out_proj also zeroed the attention weight
        # gradient path).

    def set_severance(self, pct: float, seed: int):
        if pct <= 0:
            self._sev_keep, self._sev_rescale = None, 1.0
            return
        H = self.heads
        g = torch.Generator().manual_seed(int(seed) * 10007 + int(pct))
        r = torch.rand(H, generator=g)
        keep = torch.argsort(r) >= int(round((pct / 100.0) * H))
        if not keep.any():
            keep[torch.argmax(r)] = True
        self._sev_keep = keep
        self._sev_rescale = 1.0 / (1.0 - pct / 100.0) if pct < 100.0 else 1.0

    def apply_sev(self, h):
        if self._sev_keep is None:
            return h
        H, d = self.heads, self.dim // self.heads
        m = self._sev_keep.to(device=h.device, dtype=h.dtype).view(1, 1, H, 1)
        h = (h.view(*h.shape[:-1], H, d) * m).reshape(h.shape)
        return h * self._sev_rescale

    def forward(self, y, strength: float):
        if strength <= 0.0 or self._current_kv is None:
            return y
        h, _ = self.attn(y, self._current_kv, self._current_kv,
                         key_padding_mask=self._current_pad, need_weights=False)
        h = self.apply_sev(h)
        return y + strength * torch.tanh(self.a) * torch.tanh(h)


class WrappedLayer(nn.Module):
    """Student decoder layer wrapper: orig layer first, then the bridge."""
    def __init__(self, orig, bridge):
        super().__init__()
        self.orig = orig
        self.bridge = bridge
    def forward(self, *a, **kw):
        y = self.orig(*a, **kw)
        core = y[0] if isinstance(y, tuple) else y
        out = self.bridge(core, self.bridge._strength)
        # transformers 5.16.1 LlamaDecoderLayer.forward returns a plain tensor,
        # not a tuple - return the same container type the orig layer returned
        # (deviation from brief literal, required by Executor note 1).
        if isinstance(y, tuple):
            return (out,) + tuple(y[1:])
        return out


def attach_mounts(student, teacher_layers: int):
    n_layers = len(student.model.layers)
    bridges = []
    for l in range(n_layers):
        b = MountBridge(student.config.hidden_size,
                        heads=max(1, student.config.hidden_size // 128))
        b.teacher_anchor = round((l + 0.5) / n_layers * teacher_layers)
        bridges.append(b)
        student.model.layers[l] = WrappedLayer(student.model.layers[l], b)
    # PLAIN list, NOT nn.ModuleList: each bridge is ALREADY registered
    # inside its WrappedLayer (model.layers.<l>.bridge.*). A ModuleList
    # would register every bridge a SECOND time under _mount_bridges.<l>.*
    # -> shared tensors in the state_dict -> safetensors duplicate-name
    # RuntimeError on EVERY checkpoint save (bridge-mode drill crash).
    # detach_mounts only checks truthiness here and the trainer iterates
    # the attach_mounts() return value, so a plain list is sufficient.
    student._mount_bridges = bridges
    return bridges


def detach_mounts(student):
    """Full independence: replace wrappers with the original layers."""
    if not getattr(student, "_mount_bridges", None):
        return student
    for l in range(len(student.model.layers)):
        w = student.model.layers[l]
        if isinstance(w, WrappedLayer):
            student.model.layers[l] = w.orig
    student._mount_bridges = None
    return student
