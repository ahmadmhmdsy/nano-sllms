"""Fine-Tashkeel (T5 encoder-decoder) inference for E-19 bake-off."""
import argparse, os, re, sys, time
cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(cwd)

def read_lines(p):
    return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]

BAND = "[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0674\u06d6-\u06ed]"
def strip_marks(t):
    return re.sub(BAND, "", t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("models/e19/fine-tashkeel")
    model = AutoModelForSeq2SeqLM.from_pretrained("models/e19/fine-tashkeel", torch_dtype=torch.float16, device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    lines = read_lines(a.pred)
    if a.limit:
        lines = lines[:a.limit]
    t0 = time.time()
    outs = []
    for s in range(0, len(lines), a.batch):
        batch_in = [strip_marks(t) for t in lines[s:s+a.batch]]
        enc = tok(batch_in, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, num_beams=1)
        for g in gen:
            outs.append(tok.decode(g, skip_special_tokens=True).strip())
        done = min(s + a.batch, len(lines))
        print(f"[{done}/{len(lines)}] {(done)/(time.time()-t0):.2f} l/s", flush=True)
        if (done % 500) == 0:  # partial save for resume
            open(a.out, "w", encoding="utf-8").write("\n".join(outs) + "\n")
    open(a.out, "w", encoding="utf-8").write("\n".join(outs) + "\n")
    print("total %.1f s" % (time.time() - t0))

main()