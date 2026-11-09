"""mex/scripts/net2net_widen.py — function-preserving width doubling (G3, E-32).

Net2Net duplication algebra on the mu2 Llama trunk:
  hidden 160->320, heads 4->8, kv 2->4, ffn 640->1280; head_dim stays 40 so
  RoPE/SDPA math is untouched.
  - producer emitting a doubled axis concatenates WITHOUT a factor;
  - consumer reading the duplicated axis scales by 0.5 (0.5*(a+a) = a).
Every 2D weight gets both ops when needed (out-axis cat, in-axis cat *0.5).
Emb / lm_head tie conflict resolved by UNTYING: embed = producer form,
lm_head = E-dup * 0.5 consumer form.

Verifies max |dlogit| < 1e-2 on real val blocks in fp32, then saves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/mex/mu2_g2/final")
    ap.add_argument("--dst", default="runs/mex/mu2_g3_init")
    args = ap.parse_args()

    import yaml
    from transformers import LlamaConfig, LlamaForCausalLM

    src = ROOT / args.src
    cfg = LlamaConfig.from_pretrained(src)
    old_sd = load_file(str(src / "model.safetensors"))

    H2 = cfg.hidden_size * 2
    new_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=H2,
        intermediate_size=cfg.intermediate_size * 2,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads * 2,
        num_key_value_heads=cfg.num_key_value_heads * 2,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": getattr(cfg, "rope_type", "default"),
                          "rope_theta": 10000.0},
    )
    new_cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    torch.manual_seed(0)
    new_model = LlamaForCausalLM(new_cfg)
    new_sd = dict(new_model.state_dict())
    E = old_sd["model.embed_tokens.weight"]
    # head weight: duplicate whichever head the OLD trunk actually carries. For a tied
    # trunk that is the embedding (E-32 G2 case); for an already-untied trunk (G3+) it
    # is lm_head.weight = embed * 0.5, and duplicating the EMBED would double the logits.
    W_head = old_sd.get("lm_head.weight", E)
    if torch.equal(W_head, E):
        W_head = torch.cat([E] * 2, dim=1) * 0.5   # tied old trunk -> consumer form
    unmatched = []
    for name, w_new in list(new_sd.items()):
        if name == "lm_head.weight":
            new_sd[name] = torch.cat([W_head] * 2, dim=1) * 0.5
            continue
        if name == "model.embed_tokens.weight":
            new_sd[name] = torch.cat([E] * 2, dim=1)
            continue
        w_old = old_sd.get(name)
        if w_old is None:
            unmatched.append(f"{name} no-src"); continue
        s_new = tuple(w_new.shape); s_old = tuple(w_old.shape)
        if len(s_old) == 1:
            if s_new[0] == 2 * s_old[0]:
                new_sd[name] = torch.cat([w_old] * 2, dim=0)
            elif s_new[0] == s_old[0]:
                new_sd[name] = w_old.clone()
            else:
                unmatched.append(f"{name} {s_old}->{s_new}")
            continue
        w0 = torch.cat([w_old] * 2, dim=0) if s_new[0] == 2 * s_old[0] and s_new[1] != s_old[1] else w_old
        if w0.shape[0] != s_new[0]:
            unmatched.append(f"{name} {s_old}->{s_new}"); continue
        new_sd[name] = (torch.cat([w0] * 2, dim=1) * 0.5
                        if s_new[1] == 2 * w0.shape[1] else w0)
    if unmatched:
        print("UNMATCHED:", unmatched[:10]); raise SystemExit(1)
    missing, unexpected = new_model.load_state_dict(new_sd, strict=True)
    new_model = new_model.float().eval()

    # --- exact-preservation check on a real val block (fp32) ---
    cfgy = yaml.safe_load((ROOT / "configs/mu2_g2.yaml").read_text(encoding="utf-8"))
    seq = int(cfgy["model"]["ctx"])
    from src.data import PackedDataset
    val = PackedDataset((ROOT / "data/mex/mu2/g1/tokens").glob("val_*.bin"), seq)
    chunk = torch.as_tensor(np.asarray(val[7]["input_ids"]), dtype=torch.long)
    with torch.no_grad():
        old_model = LlamaForCausalLM.from_pretrained(src).eval()
        lo = old_model(input_ids=chunk.unsqueeze(0)).logits[0].float()
        ln = new_model(input_ids=chunk.unsqueeze(0)).logits[0].float()
    dmax = (lo - ln).abs().max().item()
    print(f"net2net max|dlogit| = {dmax:.6f}")
    assert dmax < 1e-2, "function NOT preserved"

    dst = ROOT / args.dst
    new_model.save_pretrained(str(dst))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        p = src / f
        if p.is_file():
            (dst / f).write_text(p.read_text(encoding="utf-8"))
    print(f"[net2net] saved {dst} (hidden {cfg.hidden_size}->{H2}, "
          f"ffn {cfg.intermediate_size}->{cfg.intermediate_size * 2})")


if __name__ == "__main__":
    main()
