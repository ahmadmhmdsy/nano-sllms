r"""E-65: from-scratch MLM pretraining of the ~50M L3 encoder (unattended-safe).

Reads a packed uint16 .bin of concatenated token blocks (repo mainline pattern,
src/data.py style), samples random 256-token windows, applies 15% MLM masking
(80/10/10), trains BertForMaskedLM with fp16 autocast + GradScaler.

Auto-resume: every save_every_steps writes out_dir/pretrain_last.pt with model,
optimizer, scaler, step and RNG state; re-running the exact command with zero
flags resumes from it (repo hard requirement).

Usage: & .\.venv\Scripts\python.exe laya/scripts/pretrain_encoder.py --config configs/laya_l3.yaml
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PackedBin(Dataset):
    """Random 256-token windows from a memmapped uint16 token stream."""

    def __init__(self, path, seq):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq = seq
        self.n = max(1, (len(self.data) - 1) // seq)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = int(i) * self.seq
        block = torch.from_numpy(self.data[s:s + self.seq].astype(np.int64))
        return block


def mask_tokens(batch, mask_prob, mask_id, vocab, special):
    labels = batch.clone()
    prob = torch.full(batch.shape, mask_prob, device=batch.device)
    prob[:, 0] = 0.0                                  # never mask the first token
    masked = torch.bernoulli(prob).bool() & ~torch.isin(batch, special)
    labels[~masked] = -100
    r = torch.rand(batch.shape, device=batch.device)
    batch[masked & (r < 0.8)] = mask_id
    rand = masked & (r >= 0.8) & (r < 0.9)
    ri = torch.randint(999, vocab, batch.shape, device=batch.device)
    batch[rand] = ri[rand]
    return batch, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(cfg["vram_fraction"]))

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    special = torch.tensor([tok.cls_token_id, tok.sep_token_id, tok.pad_token_id],
                           device=device)
    cfgm = BertConfig(vocab_size=int(cfg["vocab_size"]), hidden_size=int(cfg["hidden"]),
                      num_hidden_layers=int(cfg["layers"]), num_attention_heads=int(cfg["heads"]),
                      intermediate_size=int(cfg["ffn"]), max_position_embeddings=int(cfg["max_pos"]),
                      pad_token_id=tok.pad_token_id)
    model = BertForMaskedLM(cfgm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[l3pre] params %.1fM seq %d" % (n_params / 1e6, cfg["seq"]), flush=True)

    train = PackedBin(cfg["corpus_bin"], int(cfg["seq"]))
    held = PackedBin(cfg["heldout_bin"], int(cfg["seq"])) if os.path.exists(cfg["heldout_bin"]) else None
    tokens_total = len(train.data)
    # max_epochs caps passes over the corpus (default 1 = pre-E-70 behavior);
    # continuation runs (carried pretrain_last.pt) need >= 2 plus the doubled
    # total_tokens so start < steps actually holds.
    steps = int(min(float(cfg["total_tokens"]) / (cfg["seq"] * cfg["micro_batch"]),
                    (len(train) // cfg["micro_batch"]) * int(cfg.get("max_epochs", 1))))
    steps = max(steps, 100)
    print("[l3pre] corpus %.1fM tokens -> %d steps" % (tokens_total / 1e6, steps), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(out, "logs"))
    except Exception:
        writer = None
    scaler = torch.amp.GradScaler("cuda")

    start, peak = 0, 0.0
    last_path = os.path.join(out, "pretrain_last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start = ck["step"]
        torch.set_rng_state(ck["rng"])
        print("[l3pre] resumed at step %d" % start, flush=True)

    base_lr = float(cfg["lr"]); warm = int(cfg["warmup_steps"])

    def lr_at(s):
        if s < warm:
            return base_lr * s / max(1, warm)
        p = (s - warm) / max(1, steps - warm)
        return 1e-5 + (base_lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    dl = DataLoader(train, batch_size=int(cfg["micro_batch"]), shuffle=True,
                    num_workers=2, drop_last=True, pin_memory=True,
                    persistent_workers=True)
    epochs = math.ceil(steps * cfg["micro_batch"] / max(1, len(train)))
    it = iter(dl)
    model.train()
    bench_log = cfg.get("bench_log", os.path.join(out, "bench_log.jsonl"))
    probe_data = [None, None]   # lazy (typed_train, typed_test) for bench probes
    pad_id = tok.pad_token_id
    t0 = time.time()
    mask_id = tok.mask_token_id
    vocab = int(cfg["vocab_size"])
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next(it, None)
        if batch is None:
            it = iter(dl); batch = next(it)
        batch = batch.to(device, non_blocking=True)
        inp, labels = mask_tokens(batch, float(cfg["mask_prob"]), mask_id, vocab, special)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outm = model(input_ids=inp, attention_mask=torch.ones_like(inp), labels=labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(outm.loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0)
        if (step + 1) % int(cfg["log_every_steps"]) == 0:
            rate = (step + 1 - start) * cfg["seq"] * cfg["micro_batch"] / max(1e-9, time.time() - t0)
            msg = "[l3pre] step %d/%d loss %.4f lr %.2e tok/s %.0f peak %.0f MiB" % (
                step + 1, steps, outm.loss.item(), lr_at(step), rate, peak)
            print(msg, flush=True)
            if writer:
                writer.add_scalar("mlm/loss", outm.loss.item(), step + 1)
                writer.add_scalar("mlm/tok_per_s", rate, step + 1)
        if (step + 1) % int(cfg["eval_every_steps"]) == 0 and held is not None:
            model.eval(); tot, n = 0.0, 0
            with torch.no_grad():
                hd = DataLoader(held, batch_size=int(cfg["micro_batch"]), num_workers=1)
                for hb in hd:
                    hb = hb.to(device)
                    hi, hl = mask_tokens(hb.clone(), 0.15, mask_id, vocab, special)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        ho = model(input_ids=hi, attention_mask=torch.ones_like(hi), labels=hl)
                    tot += ho.loss.item(); n += 1
            if writer:
                writer.add_scalar("mlm/heldout_loss", tot / max(1, n), step + 1)
                writer.add_scalar("mlm/heldout_ppl", float(np.exp(min(20.0, tot / max(1, n)))), step + 1)
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "mlm_heldout",
                   "step": step + 1, "heldout_loss": round(tot / max(1, n), 4),
                   "heldout_ppl": round(float(np.exp(min(20.0, tot / max(1, n)))), 2)}
            with open(bench_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print("[l3pre] heldout loss %.4f ppl %.2f" %
                  (tot / max(1, n), rec["heldout_ppl"]), flush=True)
            model.train()
        if (step + 1) % int(cfg["save_every_steps"]) == 0 or step + 1 == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": step + 1,
                        "rng": torch.get_rng_state()}, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)
        if ((step + 1) % int(cfg.get("probe_every_steps", 4000)) == 0 or step + 1 == steps) \
                and os.path.exists(cfg.get("typed_test", "data/laya/typed_decisions/test.pt")):
            from bench_probe import probe_head_bench
            if probe_data[0] is None:
                probe_data[0] = torch.load(cfg.get("typed_train", "data/laya/typed_decisions/train.pt"),
                                           weights_only=False)
                probe_data[1] = torch.load(cfg.get("typed_test", "data/laya/typed_decisions/test.pt"),
                                           weights_only=False)
            probe_head_bench(model, probe_data[0], probe_data[1], tok, pad_id, device, cfg,
                             tag="pretrain", step_no=step + 1, log_path=bench_log, writer=writer)

    final = os.path.join(out, "pretrain_final")
    os.makedirs(final, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final, "model.pt"))
    model.bert.save_pretrained(os.path.join(final, "encoder"))   # HF-format BertModel for the ladder
    tok.save_pretrained(os.path.join(final, "encoder"))
    summary = {"params_m": round(n_params / 1e6, 2), "steps": steps,
               "tokens_m": round(tokens_total / 1e6, 1), "wall_s": round(time.time() - t0, 1),
               "peak_vram_mib": round(peak, 1)}
    with open(os.path.join(out, "pretrain_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[l3pre] DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
