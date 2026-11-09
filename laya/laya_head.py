"""Laya-line shared model, packing, and reward code.

Port of convaiinnovations/laya's decision head (architecture verified against
rl_common.py / laya/common.py as documented in
research/laya-decision-model-report.md) onto plain BERT-class encoders.
L1 uses soft-CE only; the RLCD reward for L2 lives here too.

Honest deviations from the original head (recorded in the E-62 report):
- Option marker = the encoder's pretrained [MASK] token. Laya adds a dedicated
  <opt> token; the pretrained [MASK] embedding serves the same purpose (one
  marker per option, scored at its own position) with zero vocab surgery.
- State text is truncated on the right (keep the beginning). Laya truncates
  left for multi-turn prefixes; this line packs single records only.
- act/escalate head exists for architectural parity but is NOT trained (the
  published typed-decisions notebook zeroes its loss as well).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QTYPES = ["choice", "score", "noul"]


def qtype_id(t):
    return QTYPES.index(t)


def _head_layer(d, dropout):
    return nn.TransformerEncoderLayer(
        d_model=d, nhead=max(1, d // 64), dim_feedforward=4 * d,
        dropout=dropout, norm_first=True, batch_first=True)


class LayaDecisionModel(nn.Module):
    """Encoder + Laya decision head.

    Every option carries its own [MASK] marker; the option is scored at its
    own marker position via the shared scorer. A per-question-type embedding
    is added to every position before the head layers.
    """

    def __init__(self, encoder_name, head_layers=2, dropout=0.1):
        super().__init__()
        if isinstance(encoder_name, nn.Module):   # share an already-built encoder
            self.encoder = encoder_name
        else:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(encoder_name)
        d = self.encoder.config.hidden_size
        self.d = d
        self.type_emb = nn.Embedding(3, d)
        self.head = nn.ModuleList([_head_layer(d, dropout) for _ in range(head_layers)])
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, input_ids, attention_mask, marker_pos, marker_batch, qtype):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        pad = attention_mask == 0
        for layer in self.head:
            h = layer(h, src_key_padding_mask=pad)
        marker_h = h[marker_batch, marker_pos]      # [sum_opts, d]
        return self.scorer(marker_h).squeeze(-1)    # [sum_opts]


def option_log_probs(logits, n_options):
    """Softmax within each sequence's own options (no cross-question leakage).

    logits: flat [sum_opts]. Returns (flat log-probs, list of per-seq tensors).
    """
    logp, blocks, ofs = [], [], 0
    for n in n_options:
        lp = F.log_softmax(logits[ofs:ofs + n], dim=-1)
        logp.append(lp)
        blocks.append(lp)
        ofs += n
    return torch.cat(logp, 0), blocks


def soft_ce_loss(logp_flat, blocks, targets, n_options):
    """Soft-CE: per sequence -sum(target * logp) over its options; mean over
    sequences. targets: flat float [sum_opts] aligned with the blocks."""
    losses, ofs = [], 0
    for n in n_options:
        lp = logp_flat[ofs:ofs + n]
        tgt = targets[ofs:ofs + n]
        losses.append(-(tgt * lp).sum())
        ofs += n
    return torch.stack(losses).mean()


def proper_reward(probs, targets, n_options, w_sph=0.75, w_rps=1.0):
    """Strictly proper per-decision reward = log + w_sph*spherical - w_rps*RPS.

    probs/targets: flat float tensors aligned with n_options. Higher is
    better; the only maximum of the expectation is the true distribution.
    """
    rewards, ofs = [], 0
    for n in n_options:
        p = probs[ofs:ofs + n].clamp_min(1e-8)
        t = targets[ofs:ofs + n]
        p = p / p.sum()
        log_r = (t * p.log()).sum()
        sph = (t * p).sum() / p.pow(2).sum().sqrt().clamp_min(1e-8)
        rps = (torch.cumsum(p, 0) - torch.cumsum(t, 0)).pow(2).sum()
        rewards.append(log_r + w_sph * sph - w_rps * rps)
        ofs += n
    return torch.stack(rewards)


def pack_sequence(tok, qtype, instructions, option_texts, state_text,
                  max_len=512, head_max_len=192, opt_max=48):
    """[CLS] <type> instructions [SEP] [MASK] opt1 [MASK] opt2 ... [SEP] state [SEP]

    Returns (input_ids list[int], marker_positions list[int])."""
    cls, sep, mask = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id
    opt_ids = []
    for text in option_texts:
        if not isinstance(text, str):
            raise ValueError("option text must be str, got %r" % (text,))
        ids = tok.encode(text, add_special_tokens=False)[:opt_max]
        opt_ids.append([mask] + ids)
    opt_len = sum(len(o) for o in opt_ids)
    instr_budget = max(8, head_max_len - 6 - opt_len)
    if not isinstance(instructions, str):
        instructions = str(instructions)
    type_ids = tok.encode(QTYPES[qtype], add_special_tokens=False)
    instr_ids = tok.encode(instructions, add_special_tokens=False)[:instr_budget]
    head = [cls] + type_ids + instr_ids + [sep]
    for o in opt_ids:
        head += o
    head += [sep]
    state_budget = max_len - len(head) - 1
    state_ids = tok.encode(str(state_text), add_special_tokens=False)[:max(0, state_budget)]
    ids = head + state_ids + [sep]
    marker_pos, pos = [], len([cls] + type_ids + instr_ids + [sep])
    for o in opt_ids:
        marker_pos.append(pos)
        pos += len(o)
    return ids, marker_pos


def collate(items, pad_id):
    """Pad a list of packed items into batch tensors."""
    B = len(items)
    L = max(len(it["input_ids"]) for it in items)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    marker_pos, marker_batch, n_options, qtype, targets = [], [], [], [], []
    for b, it in enumerate(items):
        ids = it["input_ids"]
        input_ids[b, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[b, :len(ids)] = 1
        for mp in it["marker_pos"]:
            marker_pos.append(mp)
            marker_batch.append(b)
        n_options.append(it["n_options"])
        qtype.append(it["qtype"])
        targets.extend(it["target"])
    assert len(marker_pos) == len(targets)
    return {
        "input_ids": input_ids, "attention_mask": attn,
        "marker_pos": torch.tensor(marker_pos, dtype=torch.long),
        "marker_batch": torch.tensor(marker_batch, dtype=torch.long),
        "n_options": n_options,
        "qtype": torch.tensor(qtype, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.float),
    }
