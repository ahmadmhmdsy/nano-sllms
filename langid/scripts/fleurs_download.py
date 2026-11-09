import sys, io, os, pathlib, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"E:\python_projects\nano_SLMs")
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
T = os.environ["HF_TOKEN"]
from huggingface_hub import hf_hub_download
REPO = "google/fleurs"
TARGET = ROOT / "data" / "fleurs"
TARGET.mkdir(parents=True, exist_ok=True)
LANGS = ["ar_eg", "fa_ir", "ur_pk", "ps_af"]
SPLITS = ["train", "dev", "test"]
targets = []
for lang in LANGS:
    for s in SPLITS:
        targets.append(f"data/{lang}/audio/{s}.tar.gz")
        targets.append(f"data/{lang}/{s}.tsv")
for rel in targets:
    out = TARGET / rel
    try:
        p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=rel, token=T, local_dir=str(TARGET))
        print("OK", rel, "->", p, flush=True)
    except Exception as e:
        print("FAIL", rel, type(e).__name__, str(e)[:150], flush=True)
print("done")
