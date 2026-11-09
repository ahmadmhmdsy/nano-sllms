r"""E-65 ladder: mixture pretrain (adaptive) -> typed fine-tune with replay + aug.

Stage A: 3-5 epochs on the E-63 mixture (fresh encoder + fresh head); stops at
plateau (macro gain < plateau_min_gain after epoch 3) or at epochs_stageA_max.
Stage B: 4 epochs typed fine-tune with (a) 15% mixture replay interleaved per
batch (anti-forgetting, G1 lever), (b) option-ORDER shuffle augmentation
(re-pack with permuted options, p=order_aug_prob; G3 lever), (c) label-neutral
rename augmentation (option labels -> "Option A/B/...", p=rename_aug_prob).
Test data is never augmented - eval protocol identical to E-62/E-63.

Auto-resume per stage from out_dir/{A,B}_last.pt (zero-flag re-run).

Usage: & .\.venv\Scripts\python.exe laya/scripts/train_l3.py --config configs/laya_l3.yaml
"""
import argparse, json, math, os, random, sys, time
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya_head import LayaDecisionModel, pack_sequence, collate, soft_ce_loss, qtype_id
import train_l1 as t1
from bench_probe import decision_bench


def eval_A(model, items, pad_id, device, micro=32):
    model.eval()
    groups = {}
    with torch.no_grad():
        for i in range(0, len(items), micro):
            chunk = items[i:i + micro]
            _, logp, _ = t1.forward_batch(model, collate(chunk, pad_id), device)
            for j, it in enumerate(chunk):
                n = it["n_options"]
                p = logp.view(-1)[0:0]  # placeholder, replaced below
            lp = logp.tolist()
            ofs = 0
            for j, it in enumerate(chunk):
                n = it["n_options"]
                seg = lp[ofs:ofs + n]; ofs += n
                pred = max(range(n), key=lambda k: seg[k])
                ok = pred == it["gold_idx"]
                groups.setdefault(it.get("workflow", "?"), []).append(ok)
    per = {k: round(sum(v) / len(v), 4) for k, v in sorted(groups.items())}
    macro = round(sum(per.values()) / max(1, len(per)), 4)
    return {"macro": macro, "per_source": per}


def aug_item(it, tok, cfg, rng):
    """Re-pack with option-order shuffle and/or label-neutral rename.

    Falls back to the stored pack when raw fields are missing (mixture items)."""
    if "option_texts" not in it:
        return it
    texts = list(it["option_texts"])
    tgt = list(it["target"])
    n = len(texts)
    if rng.random() < float(cfg["order_aug_prob"]) and n > 1:
        perm = list(range(n))
        rng.shuffle(perm)
        texts = [texts[p] for p in perm]
        tgt = [tgt[p] for p in perm]
    if rng.random() < float(cfg["rename_aug_prob"]):
        texts = [t if ": " not in t else "Option %s: %s" % (chr(65 + i), t.split(": ", 1)[1])
                 for i, t in enumerate(texts)]
    ids, markers = pack_sequence(tok, it["qtype"], it["instructions"], texts,
                                 it["state_text"], max_len=int(cfg["max_len"]),
                                 head_max_len=int(cfg["head_max_len"]),
                                 opt_max=int(cfg["option_token_max"]))
    if len(markers) != n:
        return it
    return {"input_ids": ids, "marker_pos": markers, "n_options": n,
            "qtype": it["qtype"], "target": tgt,
            "gold_idx": max(range(n), key=lambda k: tgt[k]),
            "workflow": it.get("workflow", "?")}


def perm_duplicate(items, tok, cfg):
    """Same item, k distinct deterministic option orders (E-65 reflection N4 fix).

    Invariance by construction instead of random repack shuffling (which E-65
    showed does NOT produce block-surgery agreement). Default off; E-66 lever."""
    k = max(1, int(cfg.get("perm_dup_copies", 2)))
    if k <= 1:
        return items
    out = []
    for idx, it in enumerate(items):
        out.append(it)
        if "option_texts" not in it or len(it["option_texts"]) < 2:
            continue
        texts = list(it["option_texts"])
        tgt = list(it["target"])
        n = len(texts)
        for j in range(1, k):
            r = random.Random(((int(cfg["seed"]) * 1000003 + idx) * 131 + j) % 2147483647)
            perm = list(range(n))
            r.shuffle(perm)
            ptexts = [texts[p] for p in perm]
            ptgt = [tgt[p] for p in perm]
            ids, markers = pack_sequence(tok, it["qtype"], it["instructions"], ptexts,
                                         it["state_text"], max_len=int(cfg["max_len"]),
                                         head_max_len=int(cfg["head_max_len"]),
                                         opt_max=int(cfg["option_token_max"]))
            if len(markers) != n:
                continue
            out.append({"input_ids": ids, "marker_pos": markers, "n_options": n,
                        "qtype": it["qtype"], "target": ptgt,
                        "gold_idx": max(range(n), key=lambda kk: ptgt[kk]),
                        "workflow": it.get("workflow", "?")})
    return out


def run_stage(model, cfg, items, eval_items, pad_id, device, tag, epochs, writer,
              eval_fn, tok, aug=False, replay=None):
    out_dir = cfg["out_dir"]
    if tag == "B" and cfg.get("kd_teacher"):
        # E-69 lever P1: blend the L2 teacher's soft distribution (T=2, dumped
        # on TRAIN only) into the gold target - convex combo of two soft-CE
        # losses == soft-CE on the blended target, so no loss change needed.
        kd_w = float(cfg.get("kd_weight", 0.5))
        kd = torch.load(cfg["kd_teacher"], weights_only=False)
        n_kd = 0
        for it in items:
            if it.get("kd_blended") or "qname" not in it:
                continue
            tp = kd.get("%s|%s|%s" % (it.get("workflow", "?"),
                                      it.get("case_id", "?"), it["qname"]))
            if tp is not None and len(tp) == it["n_options"]:
                it["target"] = [(1.0 - kd_w) * g + kd_w * float(t)
                                for g, t in zip(it["target"], tp)]
                it["gold_idx"] = max(range(len(it["target"])),
                                     key=lambda kk: it["target"][kk])
                it["kd_blended"] = True
                n_kd += 1
        print("[l3:%s] KD blend w=%.2f -> %d/%d items" % (tag, kd_w, n_kd, len(items)),
              flush=True)
    if aug and cfg.get("perm_dup"):
        items = perm_duplicate(items, tok, cfg)
        print("[l3:%s] perm-duplicate expansion -> %d items" % (tag, len(items)), flush=True)
    micro = int(cfg["micro_batch"])
    accum = int(cfg["grad_accum"])
    enc_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": float(cfg["lr_encoder"])},
        {"params": head_params, "lr": float(cfg["lr_head"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(int(cfg["seed"]) + (1 if tag == "A" else 2))

    # E-71 lever: per-source balanced replay. Default (flag unset) keeps the
    # E-65 uniform pick, so every prior rung's behavior is bit-identical.
    rep_src = None
    if replay and cfg.get("replay_balanced"):
        groups = {}
        for ri, ritem in enumerate(replay):
            groups.setdefault(ritem.get("workflow", "?"), []).append(ri)
        rep_src = [v for v in groups.values() if v]
        print("[l3:%s] balanced replay: %d sources x ~%d each" %
              (tag, len(rep_src), sum(len(v) for v in rep_src) // len(rep_src)),
              flush=True)

    def pick_replay():
        if rep_src is None:
            return replay[rng.randrange(len(replay))]
        g = rep_src[rng.randrange(len(rep_src))]
        return replay[g[rng.randrange(len(g))]]

    def make_batches():
        order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
        chunks = [order[i:i + micro] for i in range(0, len(order), micro)]
        random.shuffle(chunks)
        return chunks

    micro_per_epoch = math.ceil(len(items) / micro)
    total_updates = math.ceil(micro_per_epoch / accum) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-6)

    start_epoch, step, update, curve = 0, 0, 0, []
    last_path = os.path.join(out_dir, tag + "_last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); sched.load_state_dict(ck["sched"])
        start_epoch, step, update = int(ck["epoch"]), int(ck["step"]), int(ck["update"])
        curve = ck.get("eval_curve", [])
        print("[l3:%s] resumed epoch %d update %d" % (tag, start_epoch, update), flush=True)

    model.train()
    t0, peak_mib, loss_acc, loss_n = time.time(), 0.0, 0.0, 0
    stop = False
    for epoch in range(start_epoch, epochs):
        for chunk in make_batches():
            if aug and not cfg.get("perm_dup"):
                chunk_items = [aug_item(items[i], tok, cfg, rng) for i in chunk]
            else:
                chunk_items = [items[i] for i in chunk]
            if aug and replay:
                rf = float(cfg["replay_frac"])
                chunk_items = [pick_replay()
                               if rng.random() < rf else c
                               for c in chunk_items]
            batch, logp, _ = t1.forward_batch(model, collate(chunk_items, pad_id), device)
            loss = soft_ce_loss(logp, None, batch["targets"], batch["n_options"])
            scaler.scale(loss / accum).backward()
            loss_acc += float(loss.item()); loss_n += 1; step += 1
            if step % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip"]))
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step(); update += 1
                if update % 50 == 0:
                    mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    peak_mib = max(peak_mib, mib)
                    writer.add_scalar("train/%s_loss" % tag, loss_acc / max(1, loss_n), update)
                    print("[l3:%s] update %d/%d loss %.4f peak %.0f MiB %.0fs" %
                          (tag, update, total_updates, loss_acc / max(1, loss_n), mib,
                           time.time() - t0), flush=True)
                    loss_acc, loss_n = 0.0, 0
                if update % int(cfg["save_every_steps"]) == 0:
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                                "step": step, "update": update, "epoch": epoch,
                                "eval_curve": curve}, last_path)
        m = eval_fn(model, eval_items, pad_id, device)
        curve.append({"epoch": epoch + 1, **m})
        print("[l3:%s] epoch %d eval %s" % (tag, epoch + 1, json.dumps(m)[:240]), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                    "step": step, "update": update, "epoch": epoch + 1,
                    "eval_curve": curve}, last_path)
        # E-73 lever: two-phase mixer - in stage B, stop when the transfer
        # metric has decayed below its run peak by the configured drop, so the
        # recipe holds like E-70's transfer while taking E-71/72 style typed
        # gains only while transfer also holds. Default off = prior behavior.
        if tag == "B" and cfg.get("stageB_auroc_stop") and len(curve) >= 3:
            peak = max(c.get("phish_auroc", 0.0) for c in curve)
            drop = float(cfg.get("stageB_auroc_drop", 0.02))
            if curve[-1].get("phish_auroc", 0.0) < peak - drop:
                print("[l3:B] auroc early-stop (peak %.4f, last %.4f) - stopping" %
                      (peak, curve[-1].get("phish_auroc", 0.0)), flush=True)
                stop = True
        if tag == "A" and epoch + 1 >= 3 and len(curve) >= 2:
            gain = curve[-1]["macro"] - curve[-2]["macro"]
            if gain < float(cfg["plateau_min_gain"]):
                print("[l3:A] plateau (gain %.4f) - stopping stage A" % gain, flush=True)
                stop = True
        if stop:
            break
    return {"updates": update, "wall_s": round(time.time() - t0, 1),
            "peak_vram_mib": round(peak_mib, 1), "eval_curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/laya_l3_ladder.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="1-batch validation run: tiny slices, 1 epoch/stage, benches off, out_dir+_smoke")
    args = ap.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(int(cfg["seed"])); random.seed(int(cfg["seed"]))
    device = "cuda"
    out_dir = cfg["out_dir"] + ("_smoke" if args.smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    if args.smoke:
        cfg["out_dir"] = out_dir  # run_stage reads cfg["out_dir"] - keep smoke off the real checkpoints
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    model = LayaDecisionModel(cfg["encoder"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[l3] encoder %s params %.2fM" % (cfg["encoder"], n_params / 1e6), flush=True)
    pad_id = tok.pad_token_id

    mixture_train = torch.load(cfg["mixture_train"], weights_only=False)
    mixture_held = torch.load(cfg["mixture_heldout"], weights_only=False)
    typed_train = torch.load(cfg["typed_train"], weights_only=False)
    typed_test = torch.load(cfg["typed_test"], weights_only=False)
    print("[l3] mixture %d/%d, typed %d/%d" % (len(mixture_train), len(mixture_held),
                                               len(typed_train), len(typed_test)), flush=True)

    if args.smoke:
        mixture_train, mixture_held = mixture_train[:64], mixture_held[:32]
        typed_train, typed_test = typed_train[:64], typed_test[:32]
        print("[l3] SMOKE: tiny slices, 1 epoch/stage, benches off", flush=True)

    bench_log = os.path.join(out_dir, "bench_log.jsonl")

    def log_bench(rec):
        os.makedirs(os.path.dirname(bench_log), exist_ok=True)
        with open(bench_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def eval_A_full(model, items, pad_id, device):
        m = eval_A(model, items, pad_id, device)
        try:
            te = torch.load(cfg["typed_test"], weights_only=False)
            m.update(decision_bench(model, tok, pad_id, device, te, do_phish=False))
        except Exception as e:
            m["bench_err"] = str(e)[:150]
        log_bench({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "stageA", **m})
        return m

    def eval_B_full(model, items, pad_id, device):
        m = t1.evaluate(model, items, pad_id, device)
        try:
            m.update(decision_bench(model, tok, pad_id, device, items, do_phish=True))
        except Exception as e:
            m["bench_err"] = str(e)[:150]
        log_bench({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "stageB", **m})
        return m

    print("[l3] STAGE A: mixture pretrain (max %d epochs, adaptive)" %
          int(cfg["epochs_stageA_max"]), flush=True)
    sa = run_stage(model, cfg, mixture_train, mixture_held, pad_id, device, "A",
                   1 if args.smoke else int(cfg["epochs_stageA_max"]), writer,
                   eval_A_full if not args.smoke else (lambda m, i, p, d: eval_A(m, i, p, d)), tok)
    print("[l3] STAGE B: typed fine-tune + replay %.2f + aug" %
          float(cfg["replay_frac"]), flush=True)
    sb = run_stage(model, cfg, typed_train, typed_test, pad_id, device, "B",
                   1 if args.smoke else int(cfg["epochs_stageB"]), writer,
                   eval_B_full if not args.smoke else (lambda m, i, p, d: t1.evaluate(m, i, p, d)), tok,
                   aug=True, replay=mixture_train)

    final_dir = os.path.join(out_dir, "final"); os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    summary = {"params_m": round(n_params / 1e6, 2), "stageA": sa, "stageB": sb,
               "config": {k: cfg[k] for k in ("encoder", "replay_frac", "order_aug_prob",
                                              "rename_aug_prob", "epochs_stageB")}}
    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[l3] DONE " + json.dumps(summary)[:400], flush=True)


if __name__ == "__main__":
    main()
