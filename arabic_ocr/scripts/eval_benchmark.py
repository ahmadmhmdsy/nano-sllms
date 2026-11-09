"""Honest benchmark evaluation: autoregressive generation on ISIPPT and KHATT.

Unlike the trainer's monitoring metric, this never feeds ground-truth tokens
to the decoder. It reports CER, WER, model size and inference throughput,
which are the three numbers the project targets.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ArabicOCRDataset, collate_fn
from src.models import create_model
from src.training.eval import compute_cer, decode_token_ids


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
    return prev[-1]


def corpus_wer(preds, refs):
    num = den = 0
    for p, r in zip(preds, refs):
        num += edit_distance(p.split(), r.split())
        den += max(1, len(r.split()))
    return num / den


def build_dataset(root, data_cfg):
    return ArabicOCRDataset(
        root,
        image_height=data_cfg["image_height"],
        image_width=data_cfg["image_width"],
        augment=False,
        vocab_path=data_cfg.get("vocab_path"),
        max_len=data_cfg.get("max_len", 256),
        min_text_len=0,
        max_text_len=None,
        max_image_height=None,
        keep_aspect=data_cfg.get("keep_aspect", True),
    )


@torch.no_grad()
def evaluate(model, ds, device, batch_size, max_new_tokens, amp_dtype):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)
    preds, refs, n_imgs, elapsed = [], [], 0, 0.0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            tokens = model.generate(images, max_len=max_new_tokens)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        preds.extend(decode_token_ids(tokens.float().cpu(), ds.idx_to_char))
        refs.extend(batch["raw_texts"])
        n_imgs += images.size(0)
    return preds, refs, n_imgs, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bench-root", default="data/benchmark")
    ap.add_argument("--benchmarks", default="isippt,khatt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    data_cfg = config["data"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    max_seq_len = state["pos_embed_dec"].shape[1]

    # Vocab must come from the checkpoint's own vocab, not the config default.
    vocab_size = state["token_embed.weight"].shape[0]
    model_config = dict(config["model"])
    model_config["vocab_size"] = vocab_size
    model_config["image_height"] = data_cfg["image_height"]
    model_config["image_width"] = data_cfg["image_width"]
    model_config["max_seq_len"] = max_seq_len
    model = create_model(model_config)
    model.load_state_dict(state)
    model.to(device).eval()

    params = model.count_params()
    print("checkpoint     : %s" % args.checkpoint)
    print("epoch          : %s" % ckpt.get("epoch"))
    print("params         : %.2fM" % (params / 1e6))
    print("size (fp16)    : %.1f MB" % (params * 2 / 1e6))
    print("max_seq_len    : %d" % max_seq_len)
    print("input          : %dx%d" % (data_cfg["image_height"], data_cfg["image_width"]))
    print("device         : %s" % device)
    print()

    amp_dtype = torch.float16
    max_new = args.max_new_tokens or max_seq_len
    summary = {}
    for name in args.benchmarks.split(","):
        root = Path(args.bench_root) / name
        if not (root / "labels.json").exists():
            print("[%s] SKIPPED: no labels.json at %s" % (name, root))
            continue
        ds = build_dataset(str(root), data_cfg)
        if args.limit:
            ds.labels = ds.labels[: args.limit]
        preds, refs, n_imgs, elapsed = evaluate(
            model, ds, device, args.batch_size, max_new, amp_dtype)
        cer = compute_cer(preds, refs)
        wer = corpus_wer(preds, refs)
        ips = n_imgs / elapsed if elapsed > 0 else float("nan")
        summary[name] = {"n": n_imgs, "cer": cer, "wer": wer, "img_per_s": ips}
        print("[%s] n=%d  CER=%.4f (%.2f%%)  WER=%.4f  %.1f img/s"
              % (name, n_imgs, cer, cer * 100, wer, ips))
        for i in range(min(3, len(preds))):
            print("    REF: %s" % refs[i][:70])
            print("    HYP: %s" % preds[i][:70])

    if summary:
        print()
        print("=== SUMMARY ===")
        print("params=%.2fM  size_fp16=%.1fMB" % (params / 1e6, params * 2 / 1e6))
        for k, v in summary.items():
            print("  %-8s CER=%6.2f%%  WER=%6.2f%%  %5.1f img/s"
                  % (k, v["cer"] * 100, v["wer"] * 100, v["img_per_s"]))
        out = Path("runs/benchmark_results.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        prev = {}
        if out.exists():
            try:
                prev = json.loads(out.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
        prev[args.checkpoint] = {"params": params, "results": summary}
        out.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
        print("saved -> %s" % out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
