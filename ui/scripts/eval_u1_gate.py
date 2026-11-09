"""U-1 gate: valid@k generation eval (3-layer validator) on held-out test tasks."""
import json, itertools, random, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ui.src.validate import validate_chain

FINAL = "runs/u1_full/final"
N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
K = 4
rows = []
for l in open("data/u1/full/raw/test.jsonl", encoding="utf-8"):
    d = json.loads(l)
    task, _sep, _chain = d["text"].partition("\n")
    rows.append({"task": task})
rows = rows[:N_TASKS]
tok = AutoTokenizer.from_pretrained(FINAL)
model = AutoModelForCausalLM.from_pretrained(FINAL, dtype=torch.float16).cuda().eval()

def gen(task, temp, seeds):
    ids = tok(task.strip() + "\n", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=780, do_sample=temp > 0,
                             temperature=max(temp, 1e-4), top_p=0.95,
                             num_return_sequences=seeds,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    texts = tok.batch_decode(out[:, ids.input_ids.shape[1]:], skip_special_tokens=True)
    return [t.split("<|endoftext|>")[0].strip() for t in texts]

t0 = time.time(); ok1 = 0; ok4 = 0; err_samples = []
val1_times = []
for r in rows:
    chains = gen(r["task"], 0.0, 1)
    spec, issues = None, validate_chain(chains[0])
    valid = not any(i is not None for i in issues)
    ok1 += valid
    val1_times.append(None)
    if not valid and len(err_samples) < 3:
        err_samples.append((r["task"][:60], [str(i)[:60] for i in issues if i is not None][:3], chains[0][:80]))
    samples = chains + gen(r["task"], 0.8, K - 1)
    okk = 0
    for c in samples:
        if not any(i is not None for i in validate_chain(c)):
            okk += 1
    ok4 += okk >= 1
print(json.dumps({
    "rows": len(rows), "greedy_valid": ok1, "greedy_rate": round(ok1 / len(rows), 3),
    "pass_at_4": ok4, "pass_at_4_rate": round(ok4 / len(rows), 3),
    "wall_s": round(time.time() - t0, 1),
}, indent=1), flush=True)
print("ERROR SAMPLES:", json.dumps(err_samples, indent=1))
