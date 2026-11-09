"""mex/scripts/train_e60a.py - E-60a Rung A: small deep-narrow trunk, char-LM + mark head co-trained from step 0.
Rules R1 (small trunk), R2 (joint objective, class-weighted), R4 (deep-narrow). No frozen mounts, no pre-trained base.
Data: gold v3q tokens (bare TK ids + per-base mark labels). Saves trunk final/ + head15.safetensors.
"""
from __future__ import annotations
import json as j
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup
from safetensors.torch import save_file

ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))

class V3QJoint:
    def __init__(self, tokens_dir: Path, seq_len: int, split: str, base_ids: set):
        pre = "train" if split == "train" else "val"
        self.x = np.load(tokens_dir / f"{pre}_ids.npy")
        self.y = np.load(tokens_dir / f"{pre}_y.npy")
        self.seq_len = seq_len
        self.base_ids = base_ids
        assert len(self.x) == len(self.y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx: int):
        ids = np.asarray(self.x[idx][: self.seq_len], dtype=np.int64)
        ys = np.asarray(self.y[idx], dtype=np.int64)
        marks = np.full(self.seq_len, -100, dtype=np.int64)
        k = 0
        for i in range(len(ids)):
            if int(ids[i]) in self.base_ids:
                if k < len(ys) and ys[k] >= 0:
                    marks[i] = int(ys[k])
                k += 1
        cl = np.full(self.seq_len, -100, dtype=np.int64)
        cl[: len(ids) - 1] = ids[1:]
        return {"input_ids": torch.from_numpy(ids), "char_labels": torch.from_numpy(cl),
                "mark_labels": torch.from_numpy(marks)}

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dia2j_e60a.yaml"
    cfg = yaml.safe_load(open(ROOT / cfg_path, encoding="utf-8"))
    m = cfg["model"]; t = cfg["train"]
    seq_len = int(m["ctx"])
    out = ROOT / t["output_dir"]; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(t["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from mex.src.vocab import CharVocab
    voc = CharVocab()
    id2c = {v: k for k, v in voc.vocab.items()}
    base_ids = {int(tid) for ch, tid in voc.vocab.items()
                if len(ch) == 1 and (0x0621 <= ord(ch) <= 0x064A or 0x064B <= ord(ch) <= 0x0652)}

    lc = LlamaConfig(vocab_size=97, hidden_size=int(m["hidden"]), intermediate_size=int(m["ffn"]),
                     num_hidden_layers=int(m["layers"]), num_attention_heads=int(m["heads"]),
                     num_key_value_heads=int(m["kv_heads"]), max_position_embeddings=seq_len,
                     rms_norm_eps=1e-5, rope_theta=10000.0)
    base = LlamaForCausalLM(lc).to(device)
    head = nn.Linear(int(m["hidden"]), 15).to(device)
    if t.get("init_from"):
        from safetensors.torch import load_file as _lf
        _bdir = ROOT / t["init_from"]
        base.load_state_dict(_lf(str(_bdir / "model.safetensors")), strict=True)
        print(f"[e60a] trunk init from {_bdir}", flush=True)
    n_par = sum(p.numel() for p in base.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"[e60a] total trainable params: {n_par}", flush=True)

    tdir = ROOT / cfg["data"]["tokens_dir"]
    train_ds = V3QJoint(tdir, seq_len, "train", base_ids)
    val_ds = V3QJoint(tdir, seq_len, "val", base_ids)
    print(f"[e60a] rows train/val: {len(train_ds)}/{len(val_ds)}", flush=True)
    vy = np.load(tdir / "val_y.npy"); vy = vy[vy >= 0]
    counts = np.bincount(vy, minlength=15).astype(np.float64)
    log_prior = torch.tensor(np.log(np.maximum(counts, 1.0) / counts.sum()), dtype=torch.float32, device=device)
    tau = float(t.get("logit_adjust_tau", 1.0))
    ce_mark_fn = nn.CrossEntropyLoss(ignore_index=-100)
    ce_char_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lam = float(t["mark_loss_weight"])

    params = list(base.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=float(t["lr"]), weight_decay=float(t["weight_decay"]),
                            betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, int(t["warmup_steps"]), int(t["max_steps"]))
    train_loader = DataLoader(train_ds, batch_size=int(t["batch"]), shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(t["eval_batch"]), shuffle=False, num_workers=0)
    eval_b = int(t["eval_batch"])

    total = int(t["max_steps"]); acc = int(t["accum"])
    step = 0; micro = 0; loss_acc = 0.0; t0 = time.time()
    opt.zero_grad()
    data_iter = iter(train_loader)
    while step < total:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader); continue
        xi = batch["input_ids"].to(device)
        cl = batch["char_labels"].to(device)
        ml = batch["mark_labels"].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            ob = base(input_ids=xi, output_hidden_states=True)
            h = ob.hidden_states[-1]
        l_char = ce_char_fn(ob.logits[:, :-1].float().reshape(-1, 97), cl[:, 1:].reshape(-1))
        logits_m = head(h.float()).reshape(-1, 15) + tau * log_prior
        l_mark = ce_mark_fn(logits_m, ml.reshape(-1))
        loss = (l_char + lam * l_mark) / acc
        loss = loss.clamp(max=50)
        loss.backward()
        loss_acc += float(loss) * acc
        micro += 1
        if micro == acc:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            micro = 0; step += 1
            if step % int(t["logging_steps"]) == 0 or step == total:
                cps = step * acc * int(t["batch"]) / max(time.time() - t0, 1e-9)
                print(j.dumps({"step": step, "loss": round(loss_acc / max(int(t["logging_steps"]), 1), 4),
                               "lr": float(sched.get_last_lr()[0]), "tok_s": round(cps)}), flush=True)
                loss_acc = 0.0
            if step % int(t["eval_steps"]) == 0 or step == total:
                base.eval()
                correct = 0; tot_ = 0; ce_v = 0.0; nv = 0
                cls_cor = [0]*15; cls_tot = [0]*15
                with torch.no_grad():
                    for vb in val_loader:
                        xi = vb["input_ids"].to(device)
                        cl = vb["char_labels"].to(device)
                        ml = vb["mark_labels"].to(device)
                        ob = base(input_ids=xi, output_hidden_states=True)
                        ce_v += ce_char_fn(ob.logits[:, :-1].float().reshape(-1, 97), cl[:, 1:].reshape(-1)).item(); nv += 1
                        pred = head(ob.hidden_states[-1].float()).argmax(dim=-1).cpu()
                        msk = vb["mark_labels"] >= 0
                        for cls in range(15):
                            csel = msk & (vb["mark_labels"] == cls)
                            if int(csel.sum()):
                                cls_cor[cls] += int(vb["mark_labels"][csel] == pred[csel]).sum if False else int((vb["mark_labels"][csel] == pred[csel]).sum())
                                cls_tot[cls] += int(csel.sum())
                        correct += int((vb["mark_labels"][msk] == pred[msk]).sum())
                        tot_ += int(msk.sum())
                acc15 = correct / max(tot_, 1)
                print(j.dumps({"step": step, "val_ce": round(ce_v / max(nv, 1), 4), "val_base_acc": round(acc15, 4), "cls": [round(c / max(t, 1), 2) for c, t in zip(cls_cor, cls_tot)]}), flush=True)
                base.train()
            if step % int(t["save_steps"]) == 0:
                ck = out / ("checkpoint-%05d" % step); ck.mkdir(exist_ok=True)
                torch.save({"trunk": base.state_dict(), "head": head.state_dict(), "step": step}, ck / "e60a_state.pt")
    (out / "final").mkdir(parents=True, exist_ok=True)
    base.save_pretrained(str(out / "final"))
    save_file(head.state_dict(), str(out / "final" / "head15.safetensors"))
    print(f'[e60a] final saved to {out / "final"}; trunk params={sum(p.numel() for p in base.parameters())}', flush=True)

main();
