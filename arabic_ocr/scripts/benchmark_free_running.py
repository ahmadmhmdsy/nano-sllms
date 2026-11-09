"""Free-running benchmark: pretrained vs fine-tuned, raw vs padded."""
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

VAL_DIR = "data/hflines/val"
NUM_SAMPLES = 100
OUT_DIR = Path("runs/trocr_finetune")


def pad_to_square(img):
    w, h = img.size
    if w > h:
        new_h = w
        top = (new_h - h) // 2
        p = Image.new("RGB", (w, new_h), (255, 255, 255))
        p.paste(img, (0, top))
    else:
        new_w = h
        left = (new_w - w) // 2
        p = Image.new("RGB", (new_w, h), (255, 255, 255))
        p.paste(img, (left, 0))
    return p


def compute_cer(pred, target):
    import editdistance
    return editdistance.eval(pred, target) / max(len(target), 1)


def compute_wer(pred, target):
    import editdistance
    p, t = pred.split(), target.split()
    return editdistance.eval(p, t) / max(len(t), 1)


def run_eval(processor, model, test_data, use_padding, tag):
    preds, targets = [], []
    start = time.time()
    for item in test_data:
        img = Image.open(Path(VAL_DIR) / item["image"]).convert("RGB")
        if use_padding:
            img = pad_to_square(img)
        pv = processor(img, return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            out = model.generate(pv, max_length=256, num_beams=1)
        preds.append(processor.batch_decode(out, skip_special_tokens=True)[0])
        targets.append(item["text"])
    elapsed = time.time() - start
    cers = [compute_cer(p, t) for p, t in zip(preds, targets)]
    wers = [compute_wer(p, t) for p, t in zip(preds, targets)]
    empty = sum(1 for p in preds if not p.strip())
    return {
        "tag": tag,
        "cer": sum(cers) / len(cers) * 100,
        "wer": sum(wers) / len(wers) * 100,
        "empty_rate": empty / len(preds) * 100,
        "speed": len(preds) / elapsed,
        "preds": preds,
        "targets": targets,
    }


def save(results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "benchmark_free_running.json", "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k not in ("preds", "targets")}
                   for r in results], f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "benchmark_samples.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"=== {r['tag']} ===\n")
            for i in range(min(6, len(r["preds"]))):
                f.write(f"  T: {r['targets'][i]}\n")
                f.write(f"  P: {r['preds'][i]}\n\n")


def main():
    with open(f"{VAL_DIR}/labels.json", encoding="utf-8") as f:
        test_data = json.load(f)[:NUM_SAMPLES]

    results = []
    for model_path, tag in [
        ("RayR1/trocr-base-arabic-handwritten", "pretrained"),
        ("runs/trocr_finetune/checkpoint-189", "finetuned"),
    ]:
        print(f"Loading {tag} from {model_path}...", flush=True)
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path).cuda().eval()
        print(f"  eos_token_id={model.config.eos_token_id} "
              f"decoder_start={model.config.decoder_start_token_id}", flush=True)
        for use_padding in [True, False]:
            name = f"{tag}_{'padded' if use_padding else 'raw'}"
            r = run_eval(processor, model, test_data, use_padding, name)
            results.append(r)
            print(f"  {name}: CER={r['cer']:.1f}% WER={r['wer']:.1f}% "
                  f"empty={r['empty_rate']:.0f}% {r['speed']:.1f} img/s", flush=True)
            save(results)
        del model
        torch.cuda.empty_cache()

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
