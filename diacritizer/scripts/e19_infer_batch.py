"""Batched external-model inference for E-19 bake-off (OOM-resilient)."""
import argparse, os, re, sys, time
cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(cwd)

def read_lines(p):
    return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]

def strip_marks(t):
    return re.sub("[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0674\u06d6-\u06ed]", "", t)

PY_PROMPT = "\u0642\u0645 \u0628\u062a\u0634\u0643\u064a\u0644 \u0647\u0630\u0627 \u0627\u0644\u0646\u0635" + ":\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    pad_id = tok.pad_token_id or tok.eos_token_id
    lines = read_lines(a.pred)
    if a.limit:
        lines = lines[:a.limit]
    lines = lines[a.start:]
    # resume: skip already written lines (partial-run restart contract)
    done_out = None
    if a.start == 0 and os.path.exists(a.out):
        done_out = [l for l in open(a.out, encoding="utf-8").read().splitlines() if True]
        lines = lines[len([l for l in done_out if not l.startswith("[OOM]")]):]
        outs = [l for l in done_out]
    else:
        outs = []
    tok.padding_side = "left"
    t0 = time.time()
    for i, raw in enumerate(lines):
        text = strip_marks(raw)
        prompt = PY_PROMPT + text
        cap = max(min(400, max(64, int(1.6 * len(text)))) , 64)
        ok = None
        for attempt, plen in enumerate((None, 256, 128)):
            try:
                enc = tok(prompt if plen is None else PY_PROMPT + text[:plen], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=min(cap if plen is None else cap // 2, 400), do_sample=False, num_beams=1, pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
                gen = gen[:, enc["input_ids"].shape[1]:]
                ok = tok.decode(gen[0], skip_special_tokens=True).strip()
                break
            except RuntimeError as e:
                import torch as t
                t.cuda.empty_cache()
                if plen is None:
                    print("[retry truncating]", flush=True)
                else:
                    print("[OOM-skip]", flush=True)
                ok = "[OOM after retries]"
        outs.append(ok if ok is not None else "[OOM]")
        done = a.start + i + 1
        rate = done / max(time.time() - t0, 1e-9)
        print(f"[{done}] {rate:.2f} lines/s | {(ok or '')[:44]}", flush=True)
        if (done % 40) == 0:  # periodic partial save (resume contract)
            with open(a.out, "w", encoding="utf-8") as fh:
                fh.write("\n".join(outs) + "\n")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(outs) + "\n")

main()