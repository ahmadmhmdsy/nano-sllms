import argparse, json, os, sys, re
cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(cwd)

def read_lines(p):
    return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]

def strip_marks(t):
    return re.sub("[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0674\u06d6-\u06ed]", "", t)

def load_model(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model

MODEL_DIR = os.path.join(cwd, "models", "e19", "tashkeel-350m-v2")

def vocalize(tok, model, text, max_new=400):
    import torch
    t = strip_marks(text)
    try:
        messages = [{"role": "user", "content": "\u0642\u0645 \u0628\u062a\u0634\u0643\u064a\u0644 \u0647\u0630\u0627 \u0627\u0644\u0646\u0635" + ":\n" + t}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(model.device) for k, v in ids.items() if hasattr(v, "to")}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, num_beams=1)
        gen = out[0][ids["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()
    except Exception:
        import traceback; return "ERROR: " + traceback.format_exc().replace(chr(10), " | ")[-300:],

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tok, model = load_model(MODEL_DIR)
    bare = read_lines(a.pred)
    if a.limit: bare = bare[:a.limit]
    outs = []
    for i, line in enumerate(bare):
        o = vocalize(tok, model, line)
        if isinstance(o, str): outs.append(o)
        else: outs.append(o[0])
        print(f"[{i+1}/{len(bare)}] {o[:55]}", flush=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(outs) + "\n")

main()