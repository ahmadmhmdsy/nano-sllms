r"""E-74: position-symmetric L3 trainer (option-isolated packing).

Window per option:  [CLS] <type> instructions [SEP] MASK <option> state [SEP]
Sub-items carry n_options=1 and a binary target; loss is BCEWithLogits on the
raw marker score. Grouped decision metrics softmax the qid's option scores.
Usage: & .\.venv\Scripts\python.exe laya/scripts/train_l3_sym.py --config configs/laya_l3_e74.yaml
"""
import argparse, json, os, random, time
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_head import LayaDecisionModel, pack_sequence, collate  # noqa: E402


def sym_expand(items):
    out = []
    for it in items:
        if "option_texts" not in it:
            raise ValueError("sym_expand needs raw fields (mixture_raw / typed)")
        qid = "%s|%s|%s" % (it.get("workflow", "?"), it.get("case_id", "?"), it.get("qname", "?"))
        gold = int(it["gold_idx"])
        opts = it["option_texts"]
        for i, opt in enumerate(opts):
            out.append({"qid": qid, "opt_i": i,
                        "qtype": it["qtype"], "n_options": 1,
                        "target": [1.0 if i == gold else 0.0],
                        "group_gold": gold,
                        "workflow": it.get("workflow", "?"),
                        "raw_pack": (it["instructions"], [opt], it["state_text"])})
    return out


def sym_pack_all(items, tok, cfg, batch=4000):
    out, t0 = [], time.time()
    for k in range(0, len(items), batch):
        for it in items[k:k + batch]:
            instr, opts, state = it.pop("raw_pack")
            ids, markers = pack_sequence(tok, it["qtype"], instr, opts, state,
                                         max_len=int(cfg["max_len"]),
                                         head_max_len=int(cfg["head_max_len"]),
                                         opt_max=int(cfg["option_token_max"]))
            if len(markers) != 1:
                continue
            it["input_ids"] = ids; it["marker_pos"] = markers
            out.append(it)
        if (k // batch) % 20 == 0:
            print("[l3s] packed %d/%d (%.0fs)" % (k + len(items[k:k + batch]), len(items), time.time() - t0), flush=True)
    print("[l3s] sym-packed %d sub-items in %.0fs" % (len(out), time.time() - t0), flush=True)
    return out


def forward_batch(model, batch, device):
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(batch["input_ids"], batch["attention_mask"],
                       batch["marker_pos"], batch["marker_batch"], batch["qtype"])
    return logits.float().view(-1)


def evaluate(model, groups, pad_id, device, micro=32):
    model.eval()
    correct, soft_sum, brier_sum, n = 0, 0.0, 0.0, 0
    srcs = {}
    with torch.no_grad():
        gids = list(groups.keys())
        for gid in gids:
            g = groups[gid]
            chunk = [g[i] for i in sorted(range(len(g)), key=lambda j: g[j]["opt_i"])]
            ps_all = []
            for s in range(0, len(chunk), micro):
                part = chunk[s:s + micro]
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in collate(part, pad_id).items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    scores = model(batch["input_ids"], batch["attention_mask"],
                                   batch["marker_pos"], batch["marker_batch"],
                                   batch["qtype"]).float().view(-1)
                ps_all += torch.sigmoid(scores).tolist()
            ps = torch.tensor(ps_all)
            gold = int(chunk[0]["group_gold"])
            pred = int(ps.argmax().item())
            onehot = torch.zeros_like(ps); onehot[gold] = 1.0
            correct += int(pred == gold)
            soft_sum += float(ps[gold].item())
            brier_sum += float(((ps - onehot) ** 2).mean().item())
            w = chunk[0].get("workflow", "?")
            srcs.setdefault(w, []).append(pred == gold)
            n += 1
    per = {k: round(sum(v) / len(v), 4) for k, v in sorted(srcs.items())}
    macro = round(sum(per.values()) / max(1, len(per)), 4)
    model.train()
    return {"acc": round(correct / max(1, n), 4), "soft_acc": round(soft_sum / max(1, n), 4),
            "brier": round(brier_sum / max(1, n), 4), "macro": macro,
            "per_source": per, "n": n}


def run_stage(model, cfg, items, eval_groups, pad_id, device, tag, epochs, writer,
              eval_fn, replay=None):
    micro = int(cfg["micro_batch"]); accum = int(cfg["grad_accum"])
    enc_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": float(cfg["lr_encoder"])},
                             {"params": head_params, "lr": float(cfg["lr_head"])}],
                            weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda")
    last_path = os.path.join(cfg["out_dir"], tag + "_last.pt")
    start_epoch = step = update = 0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, update = int(ck["epoch"]), int(ck["step"]), int(ck["update"])
        print("[l3s:%s] resumed epoch %d update %d" % (tag, start_epoch, update), flush=True)
    model.train()
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        order = list(range(len(items)))
        random.Random(1000 + epoch).shuffle(order)
        for s in range(0, len(order), micro):
            chunk = [items[i] for i in order[s:s + micro]]
            if replay is not None:
                rf = float(cfg["replay_frac"])
                chunk = [replay[random.randrange(len(replay))] if random.random() < rf else c
                         for c in chunk]
            batch = collate(chunk, pad_id)
            scores = forward_batch(model, batch, device)
            target = batch["targets"].to(scores.device).view(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip"]))
            scaler.step(opt); scaler.update(); step += 1; update += 1
            if update % 100 == 0:
                print("[l3s:%s] update %d loss %.4f %.0fs" %
                      (tag, update, loss.item(), time.time() - t0), flush=True)
                writer.add_scalar("train/%s_loss" % tag, loss.item(), update)
            if update % int(cfg["save_every_steps"]) == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "step": step,
                            "update": update, "epoch": epoch}, last_path)
        m = eval_fn(model, eval_groups, pad_id, device)
        print("[l3s:%s] epoch %d eval %s" % (tag, epoch + 1, json.dumps(m)[:260]), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "update": update,
                    "epoch": epoch + 1, "eval_curve": m}, last_path)
    return {"updates": update, "wall_s": round(time.time() - t0, 1),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny slices, 1 epoch/stage, out_dir+_smoke")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)
    if args.smoke:
        cfg["out_dir"] = cfg["out_dir"] + "_smoke"
    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = "cuda"
    torch.manual_seed(int(cfg["seed"])); random.seed(int(cfg["seed"]))
    writer = SummaryWriter(os.path.join(cfg["out_dir"], "logs"))
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    model = LayaDecisionModel(cfg["encoder"]).to(device)
    print("[l3s] encoder %s params %.2fM" % (cfg["encoder"], sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    pad_id = tok.pad_token_id

    def load_sym(kind, src_path):
        out_path = os.path.join("data", "laya", "sym", kind + ".pt")
        if os.path.exists(out_path):
            return torch.load(out_path, weights_only=False)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        raw = torch.load(src_path, weights_only=False)
        packed = sym_pack_all(sym_expand(raw), tok, cfg)
        torch.save(packed, out_path)
        return packed

    mixture_train = load_sym("train", "data/laya/mixture_raw/train.pt")
    mixture_held = load_sym("heldout", "data/laya/mixture_raw/heldout.pt")
    typed_train = load_sym("typed_train", "data/laya/typed_decisions/train.pt")
    typed_test = load_sym("typed_test", "data/laya/typed_decisions/test.pt")
    if args.smoke:
        mixture_train = mixture_train[:256]; mixture_held = mixture_held[:128]
        typed_train = typed_train[:256]; typed_test = typed_test[:128]
        print("[l3s] SMOKE slices", flush=True)

    def groups_of(items):
        g = {}
        for it in items:
            g.setdefault(it["qid"], []).append(it)
        return g

    gt, gm = groups_of(typed_train), groups_of(mixture_held)
    print("[l3s] STAGE A: mixture pretrain (BCE on isolated scores)", flush=True)
    run_stage(model, cfg, mixture_train, gm, pad_id, device, "A",
              1 if args.smoke else int(cfg["epochs_stageA_max"]), writer,
              lambda m, g, p, d: evaluate(m, g, p, d))
    print("[l3s] STAGE B: typed fine-tune", flush=True)
    run_stage(model, cfg, typed_train, gt, pad_id, device, "B",
              1 if args.smoke else int(cfg["epochs_stageB"]), writer,
              lambda m, g, p, d: evaluate(m, g, p, d),
              replay=None if args.smoke else mixture_train)
    final_dir = os.path.join(cfg["out_dir"], "final"); os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    print("[l3s] DONE", flush=True)


if __name__ == "__main__":
    main()
