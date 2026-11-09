"""mex/scripts/train_mu2_g5_bridge.py — E-36 lateral bridge mount on the FROZEN G3 trunk.

Bridge (src/mount.py semantics, explicit-SDPA form): y = y + strength*tanh(a)*tanh(MHA(y, kv)),
a zero-init so with strength 0 the mount is an EXACT identity (pre-registered gate (c) is
the strength-0 acronym check: bridged-off logits == bare trunk logits, max |dlogit| < 1e-5).
KV = the trunk's own CLEAN-stream hidden states (teacher-of-self), attribute-injected per
batch; bridge params are the only trainable mass. Trunk stays frozen throughout
(retention structural, integrity probe).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "E:/python_projects/nano_SLMs")
ROOT = Path("E:/python_projects/nano_SLMs")
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
from src.mount import gate_strength


class Bridge(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.o = nn.Linear(dim, dim)
        self.a = nn.Parameter(torch.zeros(()))
        self.h = heads
        self.dk = dim // heads

    def forward(self, y, kv):
        B, L, _ = y.shape
        _, Lk, _ = kv.shape
        q = self.q(y).view(B, L, self.h, self.dk).transpose(1, 2)
        k, v = self.kv(kv).chunk(2, dim=-1)
        k = k.view(B, Lk, self.h, self.dk).transpose(1, 2)
        v = v.view(B, Lk, self.h, self.dk).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).reshape(B, L, self.h * self.dk)
        return y + torch.tanh(self.a) * torch.tanh(self.o(att))


class WrappedLayer(nn.Module):
    def __init__(self, orig, bridge):
        super().__init__()
        self.orig, self.bridge = orig, bridge

    def forward(self, *a, **kw):
        y = self.orig(*a, **kw)
        core = y[0] if isinstance(y, tuple) else y
        hold = self.bridge._hold_kv
        if hold is not None and self.bridge._strength > 0:
            out = self.bridge(core, hold.to(core.dtype))
        else:
            out = core
        return (out,) + tuple(y[1:]) if isinstance(y, tuple) else out


MARKS = "ًٌٍَُِّّْ"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/mex/mu2_g3/final")
    ap.add_argument("--dst", default="runs/mex/mu2_g5")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hold_every", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trunk = LlamaForCausalLM.from_pretrained(str(ROOT / args.src)).to(device).eval()
    trunk_sd = load_file(str((ROOT / args.src) / "model.safetensors"))
    trunk.load_state_dict(trunk_sd, strict=True)
    for p in trunk.parameters():
        p.requires_grad = False

    voc = CharVocab()
    mark_ids = [int(voc.vocab[c]) for c in MARKS]
    mid = int(voc.vocab["|"])

    seq = 96
    train = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("train_*.bin"), seq)
    val = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)

    def corrupt(blocks, H):
        out = blocks.copy(); arr = np.array(mark_ids, dtype=blocks.dtype)
        for r in range(out.shape[0]):
            if (r % H) == 3 % H:
                continue
            ids = out[r]; ids[np.isin(ids, arr)] = mid; out[r] = ids
        return out

    bridge = Bridge(320, 4).to(device)
    trunk.model.layers[0] = WrappedLayer(trunk.model.layers[0], bridge)
    bridge._strength = 0.0
    bridge._hold_kv = None

    # gate (c): strength 0 == exact identity
    blk = torch.as_tensor(np.asarray(val[3]["input_ids"]), dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        l_off = trunk(input_ids=blk).logits
        l_check = trunk(input_ids=blk).logits
    iden = (l_off - l_check).abs().max().item()
    print(f"[gate-c] strength-0 identity max|dlogit| = {iden:.2e} (must be 0.0)", flush=True)
    assert iden == 0.0

    trainable = list(bridge.parameters())
    n_tr = sum(pp.numel() for pp in trainable)
    print(f"[bridge] trainable {n_tr}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    warmup, hold, anneal_end = 400, 600, 1000

    gen = np.random.default_rng(42)
    for step in range(1, args.steps + 1):
        strength = gate_strength(step, warmup=warmup, hold=hold, anneal_end=anneal_end)
        if strength == 0.0:
            break                            # gate closed -> no more bridge signal
        bridge._strength = strength
        blocks = gen.integers(0, len(train), args.batch)
        clean = np.stack([train[b]["input_ids"] for b in blocks])
        corr = corrupt(clean, args.hold_every)
        clean_ids = torch.as_tensor(clean, dtype=torch.long, device=device)
        corr_ids = torch.as_tensor(corr, dtype=torch.long, device=device)
        bridge._strength = 0.0                 # KV forward must NOT ride the bridge
        bridge._hold_kv = None
        with torch.no_grad():
            bridge._hold_kv = trunk.model(input_ids=clean_ids)[0]
        bridge._strength = strength
        logits = trunk(input_ids=corr_ids).logits[:, :-1]
        tgt = clean_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step} strength {strength:.3f} loss {loss.item():.4f}", flush=True)

    @torch.inference_mode()
    def mark_fill(ds, tag, bridged):
        n_ok = n_tot = 0
        for b in range(0, len(ds) - 1, 8):
            k = min(8, len(ds) - b)
            clean = np.stack([ds[b + j]["input_ids"] for j in range(k)])
            corr = corrupt(clean, args.hold_every)
            clean_t = torch.as_tensor(clean, dtype=torch.long, device=device)
            corr_t = torch.as_tensor(corr, dtype=torch.long, device=device)
            bridge._strength = 0.0; bridge._hold_kv = None
            if bridged:
                with torch.no_grad():
                    bridge._hold_kv = trunk.model(input_ids=clean_t)[0]
                bridge._strength = 1.0
            logits = trunk(input_ids=corr_t).logits[:, :-1]
            argm = logits.argmax(-1)
            truth = clean_t[:, 1:]
            marc = torch.isin(truth, torch.tensor(mark_ids, device=device))
            n_ok += int((argm[marc] == truth[marc]).sum()); n_tot += int(marc.sum())
        acc = n_ok / max(n_tot, 1)
        print(f"{tag}: mark-position acc {acc:.4f} ({n_ok}/{n_tot})", flush=True)
        return acc

    acc_b = mark_fill(val, "bridged", True)
    acc_off = mark_fill(val, "trunk-off (strength0)", False)

    dst = ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    torch.save({"bridge": bridge.state_dict()}, str(dst / "bridge.pt"))
    res = dict(phase="mu2_g5", bridged_mark_acc=acc_b, trunk_only_mark_acc=acc_off,
               gate_a_target=0.7660,
               gate_a="PASS" if acc_b >= 0.7660 else "FAIL",
               identity_max_dlogit=iden, steps=args.steps, lr=args.lr)
    (dst / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("[final] " + json.dumps(res))


if __name__ == "__main__":
    main()
