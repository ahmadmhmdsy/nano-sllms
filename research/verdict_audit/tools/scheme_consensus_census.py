"""E-084 companion probe A: consensus / oracle headroom census over EXISTING arms.

CPU only, no GPU, no training. Reads artifacts that already exist.

WHAT IT DECIDES. The reranking and voting axes (literature MIXED: Niehues
arXiv:1708.00563 - the better output is often already in the list, so DIVERSITY
is the bottleneck; Vassertail & Levy arXiv:2212.08926 - a reranker with no extra
data or params is only slightly better than the token-avg baseline) have never
been tested against the ONE property they depend on: whether our candidate set
actually contains a better sentence than the ensemble already emits.

The seven E-083 weighting schemes are seven DIFFERENT readings of the SAME 24
member probes. They are therefore the cheapest available stand-in for "a
candidate set": if the ORACLE over them is no better than the best of them, then
no selector can win, because there is nothing to select. If the oracle is far
better, selection has headroom and only then is a reranker worth building.

Three exact full-corpus readings per corpus, all through the REGISTERED metric
(score_corpus/summarize), so nothing is reimplemented:
  best-scheme   the seven individual schemes;
  vote          per word, the most frequent scheme word (a real, free arm);
  oracle        per word, a scheme word that matches the gold when one does
                (an UNATTAINABLE upper bound - reported to size the headroom,
                never as an arm);
plus the unanimity split: how often all seven agree, and how much of the error
sits on the words where they do not.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))

import eval_metrics as em      # noqa: E402
import eval_der                # noqa: E402


def lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def score(pred, ref):
    sm = em.summarize(em.score_corpus(pred, ref))
    return {"der_registered": sm.get("der_registered"),
            "der_canonical": sm.get("der_canonical"),
            "der_anyalt": sm.get("der_anyalt"),
            "macro_f1": sm.get("macro_f1")}


def matches(word, gfield):
    return any(eval_der.marks_of(word) == eval_der.marks_of(g)
               for g in gfield.split("/"))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/diac/e83_weighted")
    ap.add_argument("--out", default="runs/diac/_screen_out/consensus_oracle_census.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = full corpus")
    ap.add_argument("--skip-schemes", action="store_true",
                    help="skip re-scoring the 7 individual schemes (already "
                         "recorded by E-083); keeps the vote + oracle readings")
    args = ap.parse_args()
    root = REPO / args.root
    schemes = sorted(p.parent.name for p in root.glob("*/fadel_test.single.txt"))
    res = {"status": "EXACT-CPU-CENSUS", "schemes": schemes, "corpora": {}}
    print("schemes: " + ",".join(schemes), flush=True)
    tout = time.time()
    for pf in sorted((root / schemes[0]).glob("*.single.txt")):
        corpus = pf.name[:-len(".single.txt")]
        ref = lines(root / schemes[0] / (corpus + ".single.ref.txt"))
        preds = {}
        for s in schemes:
            pl = lines(root / s / (corpus + ".single.txt"))
            if args.limit:
                pl = pl[:args.limit]
            preds[s] = pl
        if args.limit:
            ref = ref[:args.limit]
        n = min(len(ref), *(len(v) for v in preds.values()))
        ref = ref[:n]
        preds = {s: v[:n] for s, v in preds.items()}
        rec = {"lines": n, "schemes": {}, "seconds": {}}
        t0 = time.time()
        if not args.skip_schemes:
            for s in schemes:
                rec["schemes"][s] = score(preds[s], ref)
        rec["seconds"]["seven_schemes"] = round(time.time() - t0, 1)
        # per-word agreement structures
        vote, oracle = [], []
        agree = wrong_split = err_agree = err_split = nwords = 0
        for li in range(n):
            ws = [preds[s][li].split() for s in schemes]
            gw = ref[li].split()
            if not all(len(w) == len(ws[0]) for w in ws) or len(gw) != len(ws[0]):
                vote.append(preds[schemes[0]][li])
                oracle.append(preds[schemes[0]][li])
                continue
            vw, ow = [], []
            for wi in range(len(ws[0])):
                cand = [w[wi] for w in ws]
                nwords += 1
                ok = [matches(c, gw[wi]) for c in cand]
                if all(c == cand[0] for c in cand):
                    agree += 1
                    if not ok[0]:
                        err_agree += 1
                else:
                    wrong_split += 1
                    top = Counter(cand).most_common(1)[0][0]
                    if not matches(top, gw[wi]):
                        err_split += 1
                vw.append(Counter(cand).most_common(1)[0][0])
                ow.append(next((c for c, k in zip(cand, ok) if k), cand[0]))
            vote.append(" ".join(vw))
            oracle.append(" ".join(ow))
        rec["seconds"]["selection"] = round(time.time() - t0, 1)
        rec["vote"] = score(vote, ref)
        rec["oracle"] = score(oracle, ref)
        rec["oracle_note"] = ("UNATTAINABLE upper bound over the seven schemes; "
                              "sizes the headroom, is never an arm")
        if rec["schemes"]:
            best = min(rec["schemes"],
                       key=lambda s: rec["schemes"][s]["der_registered"])
            rec["best_scheme"] = {"name": best, **rec["schemes"][best]}
            bd = rec["best_scheme"]["der_registered"]
            rec["vote_vs_best_rel"] = round(
                (rec["vote"]["der_registered"] - bd) / bd, 5)
            rec["oracle_vs_best_rel"] = round(
                (rec["oracle"]["der_registered"] - bd) / bd, 5)
        else:
            # --skip-schemes: the per-scheme DERs are already recorded by E-083, so
            # the relative-to-best fields are simply not recomputed here.
            best = None
            bd = None
        rec["words"] = nwords
        rec["unanimous_words"] = agree
        rec["unanimity_rate"] = round(agree / max(nwords, 1), 4)
        rec["errors_on_agreed_words"] = err_agree
        rec["errors_on_split_words"] = err_split
        rec["error_share_on_agreed"] = round(err_agree / max(err_agree + err_split, 1), 4)
        res["corpora"][corpus] = rec
        print(json.dumps({corpus: {
            "lines": n, "best_scheme": best, "best_der": bd,
            "vote_der": rec["vote"]["der_registered"],
            "vote_vs_best_rel": rec.get("vote_vs_best_rel"),
            "oracle_der": rec["oracle"]["der_registered"],
            "oracle_vs_best_rel": rec.get("oracle_vs_best_rel"),
            "unanimity_rate": rec["unanimity_rate"],
            "error_share_on_agreed_words": rec["error_share_on_agreed"]}},
            ensure_ascii=False), flush=True)
    res["total_seconds"] = round(time.time() - tout, 1)
    Path(REPO / args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                     encoding="utf-8")
    print("wrote " + args.out + " in %.1fs" % res["total_seconds"])


if __name__ == "__main__":
    main()
