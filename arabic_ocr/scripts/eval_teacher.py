"""Benchmark a VLM teacher on ISIPPT/KHATT (free-running, no teacher forcing)."""
import sys, os, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from src.training.eval import compute_cer, compute_wer

PROMPT = 'استخرج النص المكتوب في الصورة. أجب بالنص فقط.'

def run(bench_dir, proc, model, limit, max_new, device):
    labels = json.load(open(os.path.join(bench_dir, 'labels.json'), encoding='utf-8'))
    if limit:
        labels = labels[:limit]
    preds, refs = [], []
    t0 = time.time()
    for i, r in enumerate(labels):
        img = Image.open(os.path.join(bench_dir, r['image'])).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image'},
                 {'type': 'text', 'text': PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        pred = proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        preds.append(pred.strip()); refs.append(r['text'])
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print('  %d/%d  %.2f img/s  CER_so_far=%.4f' % (
                i + 1, len(labels), (i + 1) / el, compute_cer(preds, refs)), flush=True)
    el = time.time() - t0
    return {'cer': compute_cer(preds, refs), 'wer': compute_wer(preds, refs),
            'n': len(preds), 'img_per_s': len(preds) / el,
            'predictions': preds[:10], 'targets': refs[:10]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='data/teacher/qwen35_08b')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-new-tokens', type=int, default=160)
    ap.add_argument('--out', default='runs/teacher_benchmark.json')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('loading teacher from', args.teacher, flush=True)
    proc = AutoProcessor.from_pretrained(args.teacher)
    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher, dtype=torch.float16, device_map=device)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print('teacher params: %.1fM  device=%s' % (n / 1e6, device), flush=True)
    res = {}
    for name, d in [('isippt', 'data/benchmark/isippt'), ('khatt', 'data/benchmark/khatt')]:
        print('=== %s ===' % name, flush=True)
        res[name] = run(d, proc, model, args.limit, args.max_new_tokens, device)
        r = res[name]
        print('  CER=%.4f (%.2f%%)  WER=%.4f  n=%d  %.2f img/s' % (
            r['cer'], r['cer'] * 100, r['wer'], r['n'], r['img_per_s']), flush=True)
        for p, t in list(zip(r['predictions'], r['targets']))[:3]:
            print('   REF:', t[:80], flush=True)
            print('   HYP:', p[:80], flush=True)
    out = {'model': 'Qwen3.5-0.8B teacher (%dM)' % round(n / 1e6), 'benchmarks': res}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('saved ->', args.out, flush=True)

if __name__ == '__main__':
    main()
