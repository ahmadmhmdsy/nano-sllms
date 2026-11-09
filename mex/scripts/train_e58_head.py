"""mex/scripts/train_e58_head.py - E-58: train a 15-label MarkHead on v3q
bare spans (CharVocab ids + per-base-position 15-class labels) with LoRA on
the dia2f_replay trunk. Merges LoRA and saves model.safetensors +
head15.safetensors under runs/mex/dia2g_15head.
Run: python mex/scripts/train_e58_head.py --config configs/dia2g_e58.yaml
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import yaml

ROOT = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diacritizer" / "src"))
import tokenizer as TK
from labels import marks_for_label  # noqa: F401 (documented decode path)
sys.path.insert(0, str(ROOT / "mex"))
from mex.src.vocab import CharVocab

device = "cuda"
args = sys.argv[1:]
cfg_file = ROOT / (args[1] if len(args) > 1 else "configs/dia2g_e58.yaml")
cfg = yaml.safe_load(open(cfg_file, encoding="utf-8"))
T = cfg["train"]
CTX = int(cfg["model"]["ctx"])
voc = CharVocab()
PAD = int(voc.vocab["<pad>"])
SRC = ROOT / cfg["data"]["v3q_src"]
OUT = ROOT / "runs" / "mex" / "dia2h_15head"
OUT.mkdir(parents=True, exist_ok=True)
B = int(T["batch"]); ACC = int(T["accum"]); MAXS = int(T["max_steps"])
torch.manual_seed(42)

base = LlamaForCausalLM.from_pretrained(str(ROOT / cfg["base_run"]))
base.load_state_dict(load_file(str(ROOT / cfg["base_run"] / "model.safetensors")), strict=True)
base = base.to(device).train()

def build_split(split, cap):
    ids = np.load(str(SRC / f"{split}_ids.npy"), mmap_mode="r")
    ys = np.load(str(SRC / f"{split}_y.npy"), mmap_mode="r")
    n = min(cap, len(ids)) if cap else len(ids)
    X = np.full((n, CTX), PAD, dtype=np.int16)
    Y = np.full((n, CTX), -100, dtype=np.int8)
    t0 = time.time()
    for r in range(n):
        s = TK.decode([int(i) for i in ids[r]])[:CTX]
        enc = voc.encode(s)
        L = min(len(enc), CTX)
        X[r, :L] = enc[:L]
        k = 0
        for i, ch in enumerate(s):
            o = ord(ch)
            if 0x0621 <= o <= 0x064A or o == 0x0671:
                lab = int(ys[r][k]) if k < len(ys[r]) else 0
                if i < L and lab >= 0:
                    Y[r, i] = lab
                k += 1
        if (r + 1) % 50000 == 0:
            print(split, r + 1, f"{time.time()-t0:.0f}s", flush=True)
    return X, Y

print("[e58] building train spans ...", flush=True)
Xtr, Ytr = build_split("train", int(cfg["data"]["train_rows"]))
print("[e58] building val spans ...", flush=True)
Xv, Yv = build_split("val", int(cfg["data"]["val_rows"]))
np.save(OUT / "_xv.npy", Xv)
np.save(OUT / "_yv.npy", Yv)
print("[e58] spans ready:", len(Xtr), "train /", len(Xv), "val", flush=True)

pe = LoraConfig(task_type="CAUSAL_LM", r=int(cfg["lora"]["r"]), lora_alpha=int(cfg["lora"]["alpha"]),
                lora_dropout=float(cfg["lora"]["dropout"]), target_modules=list(cfg["lora"]["targets"]))
base = get_peft_model(base, pe)
base.print_trainable_parameters()
head = torch.nn.Linear(int(cfg["model"]["hidden"]), 15, device=device)
lora_p = [p for p in base.parameters() if p.requires_grad]
opt = torch.optim.AdamW([{ "params": lora_p, "lr": float(T["lr"]) }, { "params": head.parameters(), "lr": float(T.get("head_lr", 1e-3)) }], weight_decay=float(T["weight_decay"]))
params = lora_p + list(head.parameters())
sch = get_cosine_schedule_with_warmup(opt, int(T["warmup_steps"]), MAXS)
crit = torch.nn.CrossEntropyLoss(ignore_index=-100)
torch.cuda.set_per_process_memory_fraction(0.95)

def forward(x, y):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        h = base(input_ids=x, output_hidden_states=True).hidden_states[-1]
        logits = head(h.float())
        m = y != -100
        loss = crit(logits[m], y[m])
    return loss

rng = np.random.RandomState(42)
def batch():
    idx = rng.randint(0, len(Xtr), size=B)
    xb = torch.from_numpy(Xtr[idx].astype(np.int64)).to(device)
    lb = torch.from_numpy(Ytr[idx].astype(np.int64)).to(device)
    return xb, lb

def val_acc():
    base.eval(); head.eval()
    ok = tot = 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for bi in range(0, len(Xv) - B + 1, B):
            xb = torch.from_numpy(Xv[bi:bi+B].astype(np.int64)).to(device)
            lb = torch.from_numpy(Yv[bi:bi+B].astype(np.int64)).to(device)
            h = base(input_ids=xb, output_hidden_states=True).hidden_states[-1]
            m = lb != -100
            pred = head(h.float()).argmax(-1)
            ok += int((pred[m] == lb[m]).sum()); tot += int(m.sum())
    base.train(); head.train()
    return ok / max(tot, 1)

opt.zero_grad(set_to_none=True)
gstep = 0
tmark = time.time()
while gstep < MAXS:
    for _ in range(ACC):
        xb, lb = batch()
        loss = forward(xb, lb) / ACC
        loss.backward()
    torch.nn.utils.clip_grad_norm_(params, float(T["max_grad_norm"]))
    opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    gstep += 1
    if gstep % 50 == 0:
        print(json.dumps({"step": gstep, "loss": round(float(loss) * ACC, 4), "lr": round(sch.get_last_lr()[0], 8),
                          "s_per_50": round(time.time() - tmark, 1)}), flush=True)
        tmark = time.time()
    if gstep % int(T["eval_steps"]) == 0 or gstep == MAXS:
        print(json.dumps({"step": gstep, "val_base_acc": round(val_acc(), 4)}), flush=True)
    if gstep % int(T["save_steps"]) == 0 and gstep < MAXS:
        d = OUT / f"ckpt_step{gstep}"
        d.mkdir(exist_ok=True)
        base.save_pretrained(str(d / "adapter"))
        save_file(head.state_dict(), str(d / "head15.safetensors"))

print("[e58] merging + saving final ...", flush=True)
base = base.merge_and_unload()
base.save_pretrained(str(OUT))
save_file(head.state_dict(), str(OUT / "head15.safetensors"))
print("[e58] final saved to", OUT)
