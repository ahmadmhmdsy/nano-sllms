r"""E-64c: the 11-assertion probe suite from Luni/laya-jev-benchmark bench/probe.py.

Laya-only published numbers (Jev was never probed - noted honestly):
  Laya base 7 fail / fine-tuned 4 fail / grounded arms 1-2 fail.
Replicated criteria: contradiction |P(A)+P(B)-1| > 0.35 fails; stability needs
all renamed-option variants correct; grounding needs P(yes)>=0.5 to match the
stated fact; overconfidence = wrong grounding answer at confidence >= 0.8.
Deviation: our head has no trained confidence output, so confidence is the max
option probability. Diagnostic only - no pass/fail gate.

Usage: & .\.venv\Scripts\python.exe laya/scripts/eval_probes.py
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya_head import LayaDecisionModel, pack_sequence, collate, qtype_id
import train_l1 as t1

ENCODER = "microsoft/MiniLM-L12-H384-uncased"
MODELS = [("l1_final", "runs/laya/l1/final/model.pt"),
          ("l2_stageA", "runs/laya/l2/A_last.pt"),
          ("l2_final", "runs/laya/l2/final/model.pt")]

TICKET = ("I've been trying to export my data for three days and the button just spins "
          "forever. I'm on the Pro plan and I have a compliance audit on Monday. "
          "This is the second time I've written in.")
PHISH = {"from": "security@arnazon-support.com",
         "subject": "URGENT: your account has been locked",
         "body": "We detected unusual activity. Verify your password at "
                 "http://bit.ly/x9f2 within 24 hours or your account will be closed."}

CONTRADICTIONS = [
    (TICKET, "Does this require a human reply rather than an automated one?",
             "Can an automated bot fully resolve this ticket without any human involvement?"),
    (TICKET, "Is the customer reporting a software bug or defect?",
             "Is the product working exactly as intended for this customer?"),
    (PHISH,  "Is this email a phishing or scam attempt?",
             "Is this email a legitimate message from the real company it claims to be?"),
]
STABILITY = [
    ("Which team should own this ticket?",
     {"engineering": "a product defect", "support": "user needs guidance",
      "account": "billing or plan changes"}, "engineering"),
    ("Which department should handle this ticket?",
     {"technical": "bugs, outages, broken features", "support": "how-to questions",
      "billing": "invoices, payments, plan changes"}, "technical"),
    ("Route this ticket.",
     {"dev_team": "fixes defects in the software", "helpdesk": "answers user questions",
      "finance": "handles invoices"}, "dev_team"),
]
GROUNDING = [
    (TICKET, "Has this customer contacted support about this before?", True),
    (TICKET, "Is the customer on a paid plan?", True),
    (TICKET, "Is the customer asking about a refund?", False),
    (PHISH,  "Does this message contain a shortened or obfuscated link?", True),
    (PHISH,  "Does this message pressure the reader to act within a deadline?", True),
]

NOUL_OPTS = ["false: the statement is false", "true: the statement is true"]


def render(state):
    if isinstance(state, dict):
        return "\n".join("%s: %s" % (k, v) for k, v in state.items())
    return str(state)


def load_model(path, device):
    model = LayaDecisionModel(ENCODER)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd)
    return model.to(device).eval()


@torch.no_grad()
def p_yes(model, tok, pad_id, device, state, instr):
    qt = qtype_id("noul")
    ids, markers = pack_sequence(tok, qt, instr, NOUL_OPTS, render(state))
    it = [{"input_ids": ids, "marker_pos": markers, "n_options": 2, "qtype": qt,
           "target": [1.0, 0.0], "gold_idx": 0}]
    _, logp, _ = t1.forward_batch(model, collate(it, pad_id), device)
    p = logp.view(1, 2).float().exp()[0]
    return float(p[1])


@torch.no_grad()
def choice_pred(model, tok, pad_id, device, state, instr, criteria):
    qt = qtype_id("choice")
    opts = ["%s: %s" % (k, v) for k, v in criteria.items()]
    ids, markers = pack_sequence(tok, qt, instr, opts, render(state))
    it = [{"input_ids": ids, "marker_pos": markers, "n_options": len(opts),
           "qtype": qt, "target": [1.0 / len(opts)] * len(opts), "gold_idx": 0}]
    _, logp, _ = t1.forward_batch(model, collate(it, pad_id), device)
    p = logp.view(1, len(opts)).float().exp()[0]
    return list(criteria.keys())[int(p.argmax())], float(p.max())


def run_model(name, path, tok, pad_id, device):
    model = load_model(path, device)
    res = {"contradictions": [], "stability": [], "grounding": []}
    fails = 0
    for state, qa, qb in CONTRADICTIONS:
        pa = p_yes(model, tok, pad_id, device, state, qa)
        pb = p_yes(model, tok, pad_id, device, state, qb)
        tot = pa + pb
        bad = abs(tot - 1.0) > 0.35
        fails += int(bad)
        res["contradictions"].append({"sum": tot, "fail": bool(bad)})
        print("  %s sum=%.2f  P(A)=%.2f P(B)=%.2f" % ("FAIL" if bad else "ok  ", tot, pa, pb))
    n_right = 0
    for instr, criteria, expect in STABILITY:
        got, conf = choice_pred(model, tok, pad_id, device, TICKET, instr, criteria)
        ok = got == expect
        n_right += int(ok)
        res["stability"].append({"got": got, "expect": expect, "conf": conf})
        print("  %s %s (want %s) @%.2f" % ("ok  " if ok else "FAIL", got, expect, conf))
    fails += int(n_right < len(STABILITY))
    confwrong = []
    for state, q, expect in GROUNDING:
        p = p_yes(model, tok, pad_id, device, state, q)
        ok = (p >= 0.5) == expect
        conf = max(p, 1.0 - p)
        fails += int(not ok)
        res["grounding"].append({"p_yes": p, "expect": expect, "ok": bool(ok), "conf": conf})
        print("  %s p=%.2f (want %s) c=%.2f  %s" % ("ok  " if ok else "FAIL", p,
                                                    ">0.5" if expect else "<0.5", conf, q))
        if (not ok) and conf >= 0.8:
            confwrong.append(q)
    fails += len(confwrong)
    for q in confwrong:
        print("  OVERCONFIDENT: %s" % q)
    res["total_failures"] = fails
    print("  TOTAL FAILURES: %d" % fails)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ENCODER)
    pad_id = tok.pad_token_id
    report = {"note": "probe suite is Laya-only published (7 base / 4 FT); no Jev numbers",
              "models": {}}
    for name, path in MODELS:
        if not os.path.exists(path):
            print("[probes] skip %s (missing %s)" % (name, path), flush=True)
            continue
        print("[probes] %s" % name, flush=True)
        report["models"][name] = run_model(name, path, tok, pad_id, device)
    os.makedirs("runs/laya", exist_ok=True)
    with open("runs/laya/probe_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[probes] saved runs/laya/probe_eval.json", flush=True)


if __name__ == "__main__":
    main()
