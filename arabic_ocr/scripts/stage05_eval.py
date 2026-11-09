"""Stage 0.5 pilot - free-running (never teacher-forced) evaluation.

Builds the recognizer from the PaddleOCR training config plus a training
checkpoint, then decodes greedily and un-mirrors exactly as
stage0_bakeoff.py does for the exported models, so the pilot number is
directly comparable to the Stage 0 zero-shot baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{", "<": ">", ">": "<"}
LTR_RUN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.,:/+%\-]*")


def edit_distance(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[len(prev) - 1]


def unmirror(s):
    r = "".join(reversed(s))
    out = []
    i = 0
    while i < len(r):
        mt = LTR_RUN.match(r, i)
        if mt:
            out.append("".join(reversed(mt.group(0))))
            i = mt.end()
        else:
            ch = r[i]
            out.append(MIRROR.get(ch, ch))
            i += 1
    return "".join(out)


def preprocess(path, img_h=48, max_w=3200, fixed_w=0):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if fixed_w:
        resized_w = fixed_w
    else:
        resized_w = int(round(img_h * w / h))
        resized_w = max(8, min(max_w, resized_w))
    im = im.resize((resized_w, img_h), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    arr = np.flip(arr, axis=2)
    arr = arr.transpose(2, 0, 1) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.ascontiguousarray(arr[None, ...])


def ctc_greedy(logits, chars):
    idx = logits.argmax(axis=2)
    out = []
    for row in idx:
        prev = -1
        cs = []
        for k in row:
            k = int(k)
            if k != prev and k != 0 and 0 <= k < len(chars):
                cs.append(chars[k])
            prev = k
        out.append("".join(cs))
    return out


def load_chars(dict_path):
    text = Path(dict_path).read_text(encoding="utf-8")
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and lines[len(lines) - 1] == "":
        lines = lines[0:len(lines) - 1]
    return ["blank"] + lines + [" "]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fixed-width", type=int, default=0, help="force this width (0 = dynamic)")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import paddle

    if args.device == "auto":
        dev = "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
    else:
        dev = args.device
    paddle.set_device(dev)
    print("eval device: %s | cuda_compiled: %s" % (dev, paddle.device.is_compiled_with_cuda()))

    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    from ppocr.utils.save_load import load_pretrained_params

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pp = build_post_process(cfg["PostProcess"], cfg["Global"])
    char_num = len(pp.character)
    arch = cfg["Architecture"]
    arch["Head"]["out_channels_list"] = {
        "CTCLabelDecode": char_num,
        "NRTRLabelDecode": char_num + 3,
    }
    model = build_model(arch)
    ckpt_path = args.checkpoint
    if not ckpt_path.endswith(".pdparams"):
        ckpt_path = ckpt_path + ".pdparams"
    ckpt = paddle.load(ckpt_path)
    sd = model.state_dict()
    matched = 0
    for k in ckpt:
        if k in sd and list(sd[k].shape) == list(ckpt[k].shape):
            matched += 1
        elif k in sd:
            print("  SHAPE MISMATCH %s model=%s ckpt=%s" % (k, list(sd[k].shape), list(ckpt[k].shape)))
        else:
            print("  NOT IN MODEL %s" % k)
    print("checkpoint tensors=%d  matched=%d  model tensors=%d" % (len(ckpt), matched, len(sd)))
    load_pretrained_params(model, ckpt_path)
    model.eval()

    chars = load_chars(args.dict)
    print("class count: postprocess=%d  alphabet=%d" % (char_num, len(chars)))

    val_dir = Path(args.val)
    data = json.loads((val_dir / "labels.json").read_text(encoding="utf-8"))
    items = []
    for r in data:
        img = r.get("image") or r.get("img") or r.get("file")
        items.append((val_dir / img, r.get("text", r.get("label", ""))))
    if args.limit:
        items = items[0:args.limit]

    preds, refs, paths, cers = [], [], [], []
    t0 = time.perf_counter()
    with paddle.no_grad():
        for i, (path, ref) in enumerate(items):
            arr = preprocess(path, fixed_w=args.fixed_width)
            out = model(paddle.to_tensor(arr))
            if isinstance(out, (list, tuple)):
                out = out[0]
            if hasattr(out, "numpy"):
                out = out.numpy()
            out = np.asarray(out)
            if out.ndim == 2:
                out = out[None, ...]
            p = unmirror(ctc_greedy(out, chars)[0])
            preds.append(p)
            refs.append(ref)
            cers.append(100.0 * edit_distance(p, ref) / max(1, len(ref)))
            if (i + 1) % 200 == 0:
                el = time.perf_counter() - t0
                print("  %d/%d  %.2f img/s" % (i + 1, len(items), (i + 1) / el))

    el = time.perf_counter() - t0
    ed = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    rc = sum(len(r) for r in refs)
    ew = 0
    rw = 0
    exact = 0
    for p, r in zip(preds, refs):
        pw = p.split()
        rww = r.split()
        ew += edit_distance(pw, rww)
        rw += len(rww)
        if p == r:
            exact += 1
    n = max(1, len(items))
    m = {
        "tag": args.tag,
        "n": len(items),
        "checkpoint": ckpt_path,
        "class_count": char_num,
        "alphabet_size": len(chars),
        "cer": round(100.0 * ed / max(1, rc), 2),
        "wer": round(100.0 * ew / max(1, rw), 2),
        "exact": round(100.0 * exact / n, 2),
        "cer_median": round(sorted(cers)[len(cers) // 2], 2) if cers else None,
        "cer_lt10_pct": round(100.0 * sum(1 for c in cers if c < 10) / n, 1),
        "cer_lt20_pct": round(100.0 * sum(1 for c in cers if c < 20) / n, 1),
        "cer_lt50_pct": round(100.0 * sum(1 for c in cers if c < 50) / n, 1),
        "seconds": round(el, 1),
        "img_per_s": round(len(items) / el, 2),
    }
    Path(args.out).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=2))
    rows = []
    for p, r, pa, ce in zip(preds, refs, paths, cers):
        rows.append({"path": pa, "cer": round(ce, 2), "pred": p, "ref": r})
    Path(args.out).with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
