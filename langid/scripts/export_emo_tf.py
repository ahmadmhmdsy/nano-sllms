"""DA-2b (b): int8 export + numpy inference proxy for TinyTransformerEmo.

Per-column symmetric int8 quantization of every 2-D weight matrix.
Saves runs/langid_da2b/emo_tf_int8.npz + reports:
  - compressed size (D4 <= 3 MiB)
  - batch-1 CPU latency (D4 < 2 ms/text) via the pure-numpy forward
  - int8-vs-fp32 top-1 agreement per language
"""
import argparse
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src import emo_model

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data", "langid", "emo")


def qi(w: np.ndarray):
    w = w.astype(np.float32)
    scale = np.abs(w).max(axis=0) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def read_tsv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[0], int(parts[1]), "\t".join(parts[2:])))
    return rows


# ---------------- pure-numpy forward (matches TinyTransformerEmo) ----------
def enc_layer(x, pad, P):
    # self-attention, 4 heads on d=32 -> head_dim 8
    T, d = x.shape
    qkv = x @ P["in_proj_weight"].T + P["in_proj_bias"]
    q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
    hh = d // 4
    outs = []
    for i in range(4):
        qi = q[:, i * hh:(i + 1) * hh]
        ki = k[:, i * hh:(i + 1) * hh]
        vi = v[:, i * hh:(i + 1) * hh]
        sc = np.where(pad[:, None], -1e9, qi @ ki.T * (hh ** -0.5))
        e2 = np.exp(sc - sc.max(axis=1, keepdims=True))
        e2 /= e2.sum(axis=1, keepdims=True)
        outs.append(e2 @ vi)
    att = np.concatenate(outs, axis=1) @ P["out_proj_weight"].T + P["out_proj_bias"]
    x = x + att
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5) \
        * P["norm1_weight"] + P["norm1_bias"]
    y = np.maximum(x @ P["li1w"].T + P["li1b"], 0) @ P["li2w"].T + P["li2b"]
    x = x + y
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5) * P["norm2_weight"] + P["norm2_bias"]
    return x


def numpy_emodecode(ck, text):
    P = ck["np"]
    labels = ck["labels"]
    K = len(labels)
    ids, offs = emo_model.encode_batch([text])
    ids = ids.numpy().astype(np.int64)[: emo_model.MAX_LEN]
    L = max(len(ids), 1)
    ids = np.concatenate([ids, np.zeros((L - len(ids)), dtype=np.int64)])
    E = P["emb"][ids].astype(np.float32) + P["pos"][np.arange(L)]
    ids = ids.astype(np.int64)
    pad = ids == 0
    h = enc_layer(E, pad, P)
    m = (~pad).astype(np.float32)
    pooled = (h * m[:, None]).sum(0) / max(m.sum(), 1.0)
    logits = pooled @ P["headw"].T + P["headb"]
    return int(np.argmax(logits))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(BASE, "runs", "langid_da2b", "emo_tf_int8_src.pt"))
    args = ap.parse_args()
    src = args.model
    # the best transformer ckpt: pick whichever exists from the trainer default
    cand = [src, os.path.join(BASE, "runs", "langid_da2b", "emo_transformer.pt"),
            os.path.join(BASE, "runs", "langid_da2b", "emo_transformer_w_ls.pt")]
    for c in cand:
        if os.path.exists(c):
            src = c
            break
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k: v.float().numpy() for k, v in ck["state"].items()}
    ordered = collections.OrderedDict([
        ("emb", sd["emb.weight"]), ("pos", sd["pos.weight"]),
        ("in_proj_weight", sd["enc.layers.0.self_attn.in_proj_weight"]),
        ("in_proj_bias", sd["enc.layers.0.self_attn.in_proj_bias"]),
        ("out_proj_weight", sd["enc.layers.0.self_attn.out_proj.weight"]),
        ("out_proj_bias", sd["enc.layers.0.self_attn.out_proj.bias"]),
        ("norm1_weight", sd["enc.layers.0.norm1.weight"]),
        ("norm1_bias", sd["enc.layers.0.norm1.bias"]),
        ("li1w", sd["enc.layers.0.linear1.weight"]),
        ("li1b", sd["enc.layers.0.linear1.bias"]),
        ("li2w", sd["enc.layers.0.linear2.weight"]),
        ("li2b", sd["enc.layers.0.linear2.bias"]),
        ("norm2_weight", sd["enc.layers.0.norm2.weight"]),
        ("norm2_bias", sd["enc.layers.0.norm2.bias"]),
        ("headw", sd["head.weight"]), ("headb", sd["head.bias"]),
    ])
    ints, scales = {}, {}
    for k, w in ordered.items():
        if w.ndim == 2:
            ints[k], scales[k] = qi(w)
        else:
            ints[k] = w.astype(np.float32)  # biases/norms stay fp32 (tiny)
    outp = os.path.join(BASE, "runs", "langid_da2b", "emo_tf_int8.npz")
    np.savez_compressed(outp, **{k: v for k, v in ints.items()},
                        **{k + "_scale": v for k, v in scales.items()})
    size = os.path.getsize(outp) / (1 << 20)

    labels = ck["labels"]
    K = len(labels)
    emb = ordered["emb"]
    for k in ("emb", "in_proj_weight", "out_proj_weight", "li1w", "li2w", "headw"):
        if k in scales:
            ints[k] = ints[k].astype(np.float32) * scales[k]
    nck = {"labels": labels, "np": ints}
    nck["np"]["pos"] = ordered["pos"]

    # agreement + latency on test
    test = read_tsv(os.path.join(DATA, "test.tsv"))
    model = emo_model.TinyTransformerEmo(K)
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    model.eval()
    agree = {}
    lat0 = time.time()
    for lang in ["ar", "en"]:
        rows = [r for r in test if r[0] == lang][:1000]
        rows = [r for r in test if r[0] == lang][:1000]  # first 1000 rows of that lang
        ag = 0
        for i, (_, g, text) in enumerate(rows):
            with torch.no_grad():
                ids, offs = emo_model.encode_batch([text])
                fp = int(model(ids, offs).argmax())
            npd = numpy_emodecode(nck, text)
            ag += int(fp == npd)
        agree[lang] = (ag, len(rows))
    lat = (time.time() - lat0) / 2000.0
    report = {"src": src, "out": outp, "size_mib": round(size, 3),
              "latency_ms_per_text": round(lat * 1000, 3),
              "agreement": {k: v[0] for k, v in agree.items()},
              "agreement_n": {k: v[1] for k, v in agree.items()}}
    with open(outp + ".report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
