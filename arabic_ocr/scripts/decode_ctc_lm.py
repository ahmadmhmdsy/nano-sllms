"""Stage 0.5e (TASKS 139) - CTC prefix beam search + character-LM N-best rescoring.

Two modes:
  --cache   run the acoustic model once per line and store log-softmax logits
            (fp16) so decoder/LM sweeps never touch the GPU again
  --run     beam search over the cached logits -> N-best -> LM rescoring, with
            the fusion weights tuned on the DEV split only, then frozen for test

Metric is imported from the verified TASKS 134 harness (no duplicated metric);
every number is free-running greedy/beam CER, no teacher forcing. Both raw and
norm_ar (E-89) CER are reported for every set: raw is the stricter number, and
published OCR cards report the normalized one (E-99).

--blank-penalty subtracts a constant from the blank logit before decoding (E-98).
It is a decoder knob only: tune it on --dev and apply it unchanged to test.

PRE-REGISTERED GATES (TASKS 139, recorded before any decode):
  G1  muharaf_test raw CER <= 20.61 (>= 1.5 pp better than the 22.11 greedy teacher)
  G2  no eval set worse than its own greedy baseline by more than +0.50 pp
  G3  the adapted ckpt-11700 decoded with the same decoder is reported, not gated
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "arabic_ocr" / "models" / "muharaf_ocr"
sys.path.insert(0, str(MODEL_DIR))

from submission_code.ocr_helpers import (ImageConfig, greedy_ctc_decode,   # noqa: E402
                                         preprocess_image)
from submission_code.torch_models import load_ctc_backbone                  # noqa: E402

DEFAULT_CKPT = MODEL_DIR / "models" / "ctc_backbone" / "model.pt"
DEFAULT_OUT = ROOT / "arabic_ocr" / "runs" / "stage05e"
SETS = {
    "muharaf_val": ROOT / "data" / "hf" / "muharaf" / "validation",
    "muharaf_test": ROOT / "data" / "hf" / "muharaf" / "test",
    "khatt_teklia_test": ROOT / "data" / "hf" / "khatt_teklia" / "test",
    "isippt_bench": ROOT / "data" / "benchmark" / "isippt",
    "khatt_bench": ROOT / "data" / "benchmark" / "khatt",
}
G1_LIMIT = 20.61
G2_SLACK = 0.50


def _load_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load_sibling("char_lm_common", Path(__file__).with_name("char_lm_common.py"))
_M = _load_sibling("ocr_eval_metrics", Path(__file__).with_name("eval_muharaf_hflines.py"))
edit_distance, norm_ar = _M.edit_distance, _M.norm_ar


def _lev(a, b):
    """Levenshtein distance (self-contained; used only for the oracle diagnostic)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def beam_search(logp: np.ndarray, blank: int, topk: int, beam: int, nbest: int,
                len_bonus: float = 0.0, viterbi: bool = False):
    """CTC prefix beam search -> [[ids, log P_acoustic]] best first.

    viterbi=True aggregates each prefix by its BEST alignment (max) instead of the
    sum over alignments. A peaky CTC model ranks better by best path than by summed
    probability, so the summed form loses to greedy; the Viterbi form equals greedy
    at beam=1 and only explores near-miss paths at larger beams.

    len_bonus adds N*len(prefix) to the score used for PRUNING and RANKING. Without
    it P(prefix) strictly decreases as a prefix grows, so a small beam fills with
    short prefixes and loses to greedy; the returned acoustic score stays unmodified.
    """
    beams = {(): [0.0, -math.inf]}
    agg = max if viterbi else C.lse
    for t in range(logp.shape[0]):
        row = logp[t]
        cand = [(int(c), float(row[c])) for c in np.argpartition(row, -topk)[-topk:]]
        blank_lp = float(row[blank])
        new = {}
        for prefix, (pb, pnb) in beams.items():
            total = agg(pb, pnb)
            cur = new.get(prefix)
            v = total + blank_lp
            if cur is None:
                new[prefix] = [v, -math.inf]
            else:
                cur[0] = agg(cur[0], v)
            for c, lp in cand:
                if c == blank:
                    continue
                if prefix and prefix[-1] == c:
                    cur = new.get(prefix)
                    v2 = pnb + lp
                    if cur is None:
                        new[prefix] = [-math.inf, v2]
                    else:
                        cur[1] = agg(cur[1], v2)
                    nxt = prefix + (c,)
                    cur2 = new.get(nxt)
                    v3 = total + lp
                    if cur2 is None:
                        new[nxt] = [-math.inf, v3]
                    else:
                        cur2[1] = agg(cur2[1], v3)
                else:
                    nxt = prefix + (c,)
                    cur2 = new.get(nxt)
                    v3 = total + lp
                    if cur2 is None:
                        new[nxt] = [-math.inf, v3]
                    else:
                        cur2[1] = agg(cur2[1], v3)
        ranked = sorted(new.items(), key=lambda kv: -(agg(kv[1][0], kv[1][1]) + len_bonus * len(kv[0])))[:beam]
        beams = dict(ranked)
    out = sorted(beams.items(), key=lambda kv: -(agg(kv[1][0], kv[1][1]) + len_bonus * len(kv[0])))[:nbest]
    return [[list(k), agg(v[0], v[1])] for k, v in out]


def cache_logits(args, model, charset, device):
    out_dir = args.cache_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in [s for s in args.sets.split(",") if s]:
        path = out_dir / (tag + ".npz")
        if path.exists() and not args.overwrite:
            # The cache is keyed only on the tag filename, so a reused tag would silently
            # serve logits from a DIFFERENT model or height and every downstream CER would
            # be attributed to the wrong configuration. Record the height and refuse to
            # trust a cache that does not match. A cache with no recorded height is of
            # unknown provenance and is recomputed rather than trusted.
            try:
                with np.load(path, allow_pickle=False) as z:
                    cached_h = int(z["height"]) if "height" in z.files else None
            except Exception:  # noqa: BLE001 - an unreadable cache is simply not reusable
                cached_h = None
            if cached_h == int(args.height):
                print("cache keep %s (height %d)" % (path, cached_h))
                continue
            print("cache STALE %s: cached height %r != --height %d, recomputing"
                  % (path, cached_h, args.height), flush=True)
        folder = SETS[tag]
        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        chunks, refs, offs, widths, t0 = [], [], [0], [], time.time()
        with torch.no_grad():
            for n, rec in enumerate(labels, 1):
                with Image.open(folder / rec["image"]) as im:
                    arr = preprocess_image(im, ImageConfig(target_height=args.height,
                                                           grayscale=True, normalize=True))
                tensor = torch.tensor(arr, dtype=torch.float32)[None, None].to(device)
                logp = torch.log_softmax(model(tensor)[:, 0, :].float(), dim=-1)
                chunks.append(logp.half().cpu().numpy())
                refs.append(rec.get("text", ""))
                offs.append(offs[-1] + chunks[-1].shape[0])
                widths.append(int(arr.shape[1]))
        data = np.concatenate(chunks, axis=0)
        np.savez(path, data=data, offsets=np.array(offs, dtype=np.int64),
                 refs=np.array(refs), widths=np.array(widths, dtype=np.int32),
                 height=np.array(int(args.height)))
        print("cache %-20s n=%d T=%d %.1fs" % (tag, len(refs), data.shape[0], time.time() - t0), flush=True)


def ctc_viterbi(logp, labels, blank):
    """Best-path CTC score of a FIXED label sequence, in log space."""
    neg = -math.inf
    length = len(labels)
    if length == 0:
        return float(np.sum(logp[:, blank]))
    labels = np.asarray(labels, dtype=np.int64)
    diff = np.array([labels[j] != labels[j - 1] for j in range(1, length)], dtype=bool)
    prev = np.full(length, neg)
    prev[0] = logp[0][labels[0]]
    prev_blank = logp[0][blank]
    for t in range(1, logp.shape[0]):
        row = logp[t]
        from_prev = np.empty(length)
        from_prev[0] = prev_blank
        if length > 1:
            from_prev[1:] = np.where(diff, prev[:-1], neg)
        np.maximum(prev, from_prev, out=from_prev)
        cur = from_prev + row[labels]
        cur_blank = max(prev_blank, float(prev.max())) + row[blank]
        prev, prev_blank = cur, cur_blank
    return float(max(prev[length - 1], prev_blank))


def greedy_from_cache(data, offs, blank):
    """Collapsed CTC greedy path per line (must use the harness decoder)."""
    hyps = []
    for i in range(len(offs) - 1):
        raw = np.argmax(data[offs[i]:offs[i + 1]], axis=-1).tolist()
        hyps.append(greedy_ctc_decode(raw, blank_id=blank))
    return hyps


def score_rows(pairs):
    edits = refs = edits_norm = refs_norm = 0
    cers = []
    for hyp, ref in pairs:
        e = edit_distance(hyp, ref)
        edits += e
        refs += len(ref)
        cers.append(100.0 * e / max(1, len(ref)))
        nh, nr = norm_ar(hyp), norm_ar(ref)
        edits_norm += edit_distance(nh, nr)
        refs_norm += len(nr)
    return {"cer": round(100.0 * edits / max(1, refs), 2),
            "cer_norm": round(100.0 * edits_norm / max(1, refs_norm), 2),
            "cer_median": round(sorted(cers)[len(cers) // 2], 2) if cers else None}


def run(args, charset):
    sets = [s for s in args.sets.split(",") if s]
    if args.dev and args.dev not in sets:
        raise SystemExit("--dev split must also be in --sets")
    lm = None
    if args.lm and not args.no_lm:
        blob = torch.load(args.lm, map_location="cpu", weights_only=False)
        lm = C.CharLM(blob["vocab"], blob["dim"], blob["layers"])
        lm.load_state_dict(blob["state_dict"])
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        lm.eval().to(device)
        print("lm loaded %s vocab=%d" % (args.lm, blob["vocab"]))
    else:
        device = torch.device("cpu")

    nbest = {}
    greedy_res = {}
    for tag in sets:
        blob = np.load(args.cache_dir / args.tag / (tag + ".npz"), allow_pickle=False)
        data, offs = blob["data"], blob["offsets"]
        refs = [str(r) for r in blob["refs"]]
        if args.blank_penalty:
            data = data.astype(np.float32)
            data[:, charset.blank_id] -= args.blank_penalty
        hyps = greedy_from_cache(data, offs, charset.blank_id)
        greedy_pairs = [(charset.decode(h), r) for h, r in zip(hyps, refs)]
        greedy_res[tag] = score_rows(greedy_pairs) | {"n": len(refs)}
        t0 = time.time()
        rows = []
        for i in range(len(offs) - 1):
            seg = data[offs[i]:offs[i + 1]].astype(np.float32)
            row = beam_search(seg, charset.blank_id, args.topk, args.beam, args.nbest,
                              args.len_bonus, args.viterbi)
            greedy_ids = list(hyps[i])
            if greedy_ids and not any(list(c[0]) == greedy_ids for c in row):
                row.append([greedy_ids, ctc_viterbi(seg, greedy_ids, charset.blank_id)])
            rows.append(row)
        nbest[tag] = {"refs": refs, "rows": rows}
        if args.save_nbest:
            nb_dir = Path(args.out)
            nb_dir.mkdir(parents=True, exist_ok=True)
            with open(nb_dir / ("nbest_%s.jsonl" % tag.replace("/", "_")), "w",
                      encoding="utf-8") as fh:
                for r, ref in zip(rows, refs):
                    fh.write(json.dumps({"ref": ref,
                                         "cands": [[list(c[0]), c[1]] for c in r]},
                                        ensure_ascii=False) + "\n")
        if args.oracle:
            tot_d = 0
            tot_n = 0
            distinct = 0
            for r, ref in zip(rows, refs):
                texts = [charset.decode(c[0]) for c in r]
                distinct += len(set(texts))
                tot_d += min(_lev(t, ref) for t in texts)
                tot_n += len(ref)
            print("oracle %-20s %.2f  (distinct cands/line %.2f)" % (
                tag, 100.0 * tot_d / max(1, tot_n), distinct / max(1, len(refs))), flush=True)
        print("beam %-20s n=%d %.1fs (avg cands %.2f)" % (
            tag, len(rows), time.time() - t0,
            sum(len(r) for r in rows) / max(1, len(rows))), flush=True)

    if lm is not None:
        for tag in sets:
            rows = nbest[tag]["rows"]
            seqs = [[int(x) for x in c[0]] for r in rows for c in r]
            scores = C.score_sequences(lm, seqs, device)
            k = 0
            for r in rows:
                for c in r:
                    c.append(scores[k])
                    k += 1
            print("lm-scored %-20s %d candidates" % (tag, k), flush=True)

    grid = [float(x) for x in args.betas.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]
    dev = args.dev or sets[0]
    sweep = []
    best = None
    for beta in grid:
        for gamma in gammas:
            pairs = []
            for row in nbest[dev]["rows"]:
                pick = pick_candidate(row, beta, gamma)
                pairs.append((charset.decode(pick[0]), nbest[dev]["refs"][len(pairs)]))
            cer = score_rows(pairs)["cer"]
            sweep.append({"beta": beta, "gamma": gamma, "dev_cer": cer})
            if best is None or cer < best[2]:
                best = (beta, gamma, cer)
    print("dev sweep best beta=%.2f gamma=%.2f dev_cer=%.2f" % best)

    results = []
    for tag in sets:
        beta, gamma = best[0], best[1]
        pairs = []
        for i, row in enumerate(nbest[tag]["rows"]):
            pairs.append((charset.decode(pick_candidate(row, beta, gamma)[0]), nbest[tag]["refs"][i]))
        lm_res = score_rows(pairs)
        row = {"tag": tag, "n": len(pairs), "greedy_cer": greedy_res[tag]["cer"],
               "greedy_cer_norm": greedy_res[tag]["cer_norm"],
               "beam_cer": score_rows([(charset.decode(c[0]), nbest[tag]["refs"][i])
                                       for i, r in enumerate(nbest[tag]["rows"])
                                       for c in [pick_candidate(r, 0.0, 0.0)]]),
               "lm_cer": lm_res["cer"], "lm_cer_norm": lm_res["cer_norm"],
               "lm_cer_median": lm_res["cer_median"]}
        row["delta_vs_greedy"] = round(row["lm_cer"] - row["greedy_cer"], 2)
        row["len_hyp"] = round(sum(len(h) for h, _ in pairs) / max(1, len(pairs)), 1)
        row["len_ref"] = round(sum(len(r) for _, r in pairs) / max(1, len(pairs)), 1)
        results.append(row)
        print("%-20s greedy %6.2f -> beam %6.2f -> lm %6.2f (%+0.2f) norm %6.2f" % (
            tag, row["greedy_cer"], row["beam_cer"]["cer"], row["lm_cer"],
            row["delta_vs_greedy"], row["lm_cer_norm"]), flush=True)
        print("   mean len hyp %.1f vs ref %.1f" % (row["len_hyp"], row["len_ref"]), flush=True)

    gates = {"G1_muharaf_test_le_%.2f" % G1_LIMIT: None,
             "G2_no_set_worse_than_greedy_by_%.2f" % G2_SLACK: None,
             "G3_reported_only": True}
    main_row = next((r for r in results if r["tag"] == "muharaf_test"), None)
    if main_row:
        gates["G1_muharaf_test_le_%.2f" % G1_LIMIT] = bool(main_row["lm_cer"] <= G1_LIMIT)
        gates["G2_no_set_worse_than_greedy_by_%.2f" % G2_SLACK] = bool(
            all(r["delta_vs_greedy"] <= G2_SLACK + 1e-9 for r in results))
    scorecard = {"tag": args.tag, "decoder": {"beam": args.beam, "topk": args.topk,
                 "nbest": args.nbest, "blank_penalty": args.blank_penalty,
                 "lm": str(args.lm) if lm is not None else None},
                 "chosen": {"beta": best[0], "gamma": best[1], "dev": dev, "dev_cer": best[2]},
                 "sweep": sweep, "results": results, "GATES": gates}
    args.out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(scorecard, ensure_ascii=False, indent=2)
    (args.out / ("scorecard_%s.json" % args.tag)).write_text(payload, encoding="utf-8")
    with open(args.out / ("perline_%s.jsonl" % args.tag), "w", encoding="utf-8") as fh:
        for tag in sets:
            for i, r in enumerate(nbest[tag]["rows"]):
                pick = pick_candidate(r, best[0], best[1])
                fh.write(json.dumps({"set": tag, "ref": nbest[tag]["refs"][i],
                                     "hyp": charset.decode(pick[0]),
                                     "acoustic": round(pick[1], 4),
                                     "lm": round(pick[2], 4) if len(pick) > 2 else None},
                                    ensure_ascii=False) + "\n")
    print("GATES %s" % json.dumps(gates))


def pick_candidate(row, beta: float, gamma: float):
    best = None
    for c in row:
        lm_score = c[2] if len(c) > 2 else 0.0
        score = c[1] + beta * lm_score + gamma * len(c[0])
        if best is None or score > best[3]:
            best = (c[0], c[1], lm_score, score)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_OUT / "logits")
    ap.add_argument("--tag", default="teacher")
    ap.add_argument("--sets", default="muharaf_val,muharaf_test,khatt_teklia_test,isippt_bench,khatt_bench")
    ap.add_argument("--dev", default="muharaf_val")
    ap.add_argument("--lm", type=Path, default=None)
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--betas", default="0,0.2,0.4,0.6,0.8,1.0,1.4")
    ap.add_argument("--gammas", default="0,0.5,1.0,2.0")
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--nbest", type=int, default=12)
    ap.add_argument("--len-bonus", type=float, default=0.0)
    ap.add_argument("--oracle", action="store_true",
                    help="report the best-CER candidate per line (N-best upper bound)")
    ap.add_argument("--save-nbest", action="store_true")
    ap.add_argument("--viterbi", action="store_true",
                    help="rank prefixes by best alignment (max), not the sum")
    ap.add_argument("--blank-penalty", type=float, default=0.0,
                    help="subtract this from the blank logit before decoding (E-98); tune on --dev only")
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not (args.cache or args.run):
        raise SystemExit("pass --cache and/or --run")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    charset = None
    if args.cache:
        model, charset, _ = load_ctc_backbone(args.ckpt, device)
        cache_logits(args, model, charset, device)
    if args.run:
        if charset is None:
            charset = load_ctc_backbone(args.ckpt, torch.device("cpu"))[1]
        run(args, charset)


if __name__ == "__main__":
    main()
