"""Stage-1 char-LM training (next-token CE) on the raw Arabic corpus.

Same discipline contract as train.py: fp32 master + autocast fp16 +
GradScaler skip-on-overflow, zero-flag auto-resume off checkpoint-*
rotation, best.pt at every improved eval, final/ = best weights +
config.yaml. The model is the SAME DiacritizerModel built causal=True
(label head unused; LM logits = h @ embed.T - TIED head, zero extra
params). CE ignores PAD targets.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'diacritizer' / 'src'))
import tokenizer as TK
from model import DiacritizerModel

RUNS = REPO / 'runs' / 'diac'

def _ce(logits, tgt, pad):
    m = tgt != pad
    L2 = logits[m]
    return torch.nn.functional.cross_entropy(L2, tgt[m].long(), reduction='mean')

def latest_checkpoint(run_dir):
    ks = sorted(run_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1)
    return ks[-1] if ks else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    phase = cfg['phase']
    run_dir = RUNS / phase
    run_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = REPO / 'data' / 'diac' / cfg.get('tokens_dir', 'stage1') / 'tokens'
    train_ids = np.load(str(tokens_dir / 'train_ids.npy'))
    val_ids = np.load(str(tokens_dir / 'val_ids.npy'))
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else 'cpu'
    # Hard VRAM ceiling (config key vram_cap_gib). torch.cuda allocator is pinned so
    # the process NEVER allocates past the dedicated VRAM — otherwise Windows WDDM
    # silently spills the tail into shared iGPU memory (RAM+VRAM ballooning, ~10x slowdown).
    if device == 'cuda':
        cap = cfg.get('vram_cap_gib')
        if cap:
            frac = min(0.999, float(cap) / (torch.cuda.get_device_properties(0).total_memory / (1024**3)))
            torch.cuda.set_per_process_memory_fraction(frac)
            print('[VRAM-CAP]', round(frac * (torch.cuda.get_device_properties(0).total_memory / (1024**3)), 2), 'GiB', flush=True)
    mm = cfg['model']
    model = DiacritizerModel(vocab_size=TK.VOCAB_SIZE, hidden=mm['hidden'],
                             layers=mm['layers'], n_heads=mm['n_heads'], n_kv=mm['n_kv'],
                             ffn=mm.get('ffn'), max_ctx=mm.get('ctx', 512), causal=True).to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    print({'phase': phase, 'device': device, 'params': sum(p.numel() for p in model.parameters()),
           'train_windows': len(train_ids), 'val_windows': len(val_ids), 'ctx': train_ids.shape[1]}, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 2e-4), weight_decay=cfg.get('weight_decay', 0.1))
    bs = cfg.get('batch_size', 32)
    accum = cfg.get('accum', 1)          # micro-batches per optimizer step (keeps effective batch = bs_from_cfg)
    eval_bs = cfg.get('eval_bs', 128)    # eval micro-batch (VRAM cap; 64 fits the 6GB card)
    total_steps = cfg.get('total_steps', 12000)
    save_every = cfg.get('save_every', 500)
    eval_every = cfg.get('eval_every', 250)
    patience = cfg.get('early_stop_patience', 12)
    step = 0
    bad = 0
    best_vl = float('inf')
    ck = latest_checkpoint(run_dir)
    if ck:
        st = torch.load(ck / 'state.pt', map_location='cpu')
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        step = st['step']
        best_vl = float(torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False).get('best_val_loss', best_vl)) if (run_dir / 'best.pt').exists() else best_vl
        print('[RESUME]', ck.name, 'step', step, 'best_vl', round(best_vl, 4), flush=True)
    else:
        print('[START] fresh', flush=True)
    PAD = TK.PAD
    rng = np.random.default_rng(777)
    use_amp = device == 'cuda'
    t0 = time.time()
    model.train()
    T = train_ids.shape[1]
    while step < total_steps:
        g_scale = 1.0 / max(1, accum)
        for _a in range(accum):
            idx = rng.integers(0, len(train_ids), bs)
            x = torch.from_numpy(train_ids[idx]).to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                lg = model.hidden(inp) @ model.embed.weight.t()   # TIED LM head
            loss = _ce(lg.float(), tgt, PAD) * g_scale
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sb = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if scaler.get_scale() < sb: continue
        step += 1
        if step % eval_every == 0:
            model.eval()
            tot = ntok = 0.0
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                for i in range(0, min(len(val_ids), 4096), eval_bs):
                    x = torch.from_numpy(val_ids[i:i+eval_bs]).to(device)
                    lg = model.hidden(x[:, :-1]) @ model.embed.weight.t()   # TIED LM head
                    m = x[:, 1:] != PAD
                    vl = _ce(lg.float(), x[:, 1:], PAD)
                    tot += float(vl) * int(m.sum()); ntok += int(m.sum())
            vl = tot / max(1, ntok)
            print(f'step {step} train {float(loss):.4f} val_loss {vl:.4f} elapsed {time.time()-t0:.0f}s', flush=True)
            if vl < best_vl - 1e-4:
                best_vl = float(vl); bad = 0
                torch.save({'model': model.state_dict(), 'step': step, 'best_val_loss': best_vl}, run_dir / 'best.pt')
                print('[BEST]', round(best_vl, 4), flush=True)
            else:
                bad += 1
                if bad >= patience:
                    print({'early_stop': step, 'best_val_loss': best_vl}, flush=True)
                    break
            model.train()
        if step % save_every == 0 or step == total_steps:
            sd = run_dir / f'checkpoint-{step}'
            sd.mkdir(exist_ok=True)
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step, 'config': json.dumps(cfg)}, sd / 'state.pt')
            print('[SAVE]', sd.name, flush=True)
            # rotation: keep only the newest ROT checkpoints (train.py parity;
            # no-rotation variant once filled the disk mid-stage2 run)
            ks = sorted(run_dir.glob('checkpoint-*'), key=lambda q: int(q.name.split('-')[-1]))
            for old in ks[:-3]:
                if old.is_dir():
                    import shutil
                    shutil.rmtree(old, ignore_errors=True)
                    print('[PRUNE]', old.name, flush=True)
            # guard: crash mid-save leaves a torn zip; a later load will fail
            # visibly on resume, which is better than silently mixed state
    md = run_dir / 'final'
    md.mkdir(exist_ok=True)
    torch.save(model.state_dict(), md / 'model.pt')
    (md / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    print({'done': step, 'final': str(md), 'best_val_loss': best_vl}, flush=True)

if __name__ == '__main__':
    main()
