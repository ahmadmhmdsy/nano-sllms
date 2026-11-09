"""DA-8 (E-56) ranker test eval: batch-64 R@1 / R@10."""
import json, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.scripts.train_rank import DualBag, read, batch_r1

BASE = pathlib.Path(__file__).resolve().parents[2]


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "ar"
    ckp = str(BASE / "runs" / "langid_da8" / ("rank_%s.pt" % lang))
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    model = DualBag(ck["d"])
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    a, b = read(str(BASE / "data" / "langid" / "title" / lang / "test.tsv"))
    r1, r10 = batch_r1(model, a, b, device, top10=True)
    out = {"lang": lang, "test_pairs_kept": (len(a) // 64) * 64, "batch_R@1": round(r1, 4),
           "batch_R@10": round(r10, 4), "chance": round(1 / 64, 4), "model": ckp}
    json.dump(out, open(ckp + ".rank_eval.json", "w", encoding="utf-8"), indent=1)
    for k, v in out.items():
        print(k, "=", v)


main()