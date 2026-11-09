"""Off-site backup of phase finals to a private HF Hub repo (Milestone F).

DRY-RUN by default: prints whoami + the exact file manifest + total size
and uploads NOTHING. A real upload requires BOTH --repo and --execute and
the repo name must come from the user (TASKS row 15 gate).

Never uploads: checkpoint-* dirs, optimizer.pt, training_args.bin - only
the final model dir's safetensors/config/tokenizer files.

HF_TOKEN comes from the project .env (hf_hub does NOT auto-load .env -
this script loads it explicitly, same pattern as exa_search.py).

Run: .venv/Scripts/python scripts/backup_to_hub.py --dir runs/pilot/final
     .venv/Scripts/python scripts/backup_to_hub.py --repo <user>/<name> --dir runs/pilot/final --execute
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_BLOCKLIST = {"optimizer.pt", "training_args.bin", "trainer_state.json"}


def load_env_token():
    """Load .env from the repo root (repo pattern - no dotenv dependency)."""
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
    return os.environ.get("HF_TOKEN")


def collect_files(d: Path):
    files = []
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        if "checkpoint-" in p.parent.name or p.name in UPLOAD_BLOCKLIST:
            continue
        files.append(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="backup phase finals to HF Hub")
    ap.add_argument("--dir", default="runs/target/final",
                    help="final dir to upload (default runs/target/final)")
    ap.add_argument("--repo", default=None,
                    help="target repo id <user>/<name> (required with --execute)")
    ap.add_argument("--path_in_repo", default=None,
                    help="subfolder in the repo (default: <phase>-final)")
    ap.add_argument("--execute", action="store_true",
                    help="actually upload (default is dry-run)")
    args = ap.parse_args()

    token = load_env_token()
    if not token:
        raise SystemExit("HF_TOKEN not found: add it to the project .env")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    who = api.whoami()
    print(f"[backup] token OK, user: {who.get('name')}", flush=True)

    d = Path(args.dir)
    if not d.is_absolute():
        d = ROOT / d
    if not d.is_dir():
        raise SystemExit(f"final dir not found: {d}")
    files = collect_files(d)
    if not files:
        raise SystemExit(f"no uploadable files under {d}")

    total = sum(p.stat().st_size for p in files)
    print(f"[backup] manifest for {d} ({len(files)} files, "
          f"{total / 2**20:.1f} MiB):", flush=True)
    for p in files:
        print(f"  {p.relative_to(d)}  {p.stat().st_size / 2**20:.2f} MiB", flush=True)

    if not args.execute:
        print("[backup] DRY-RUN complete - nothing was uploaded.", flush=True)
        print("[backup] to upload: pass --repo <user>/<name> --execute "
              "(repo id must be user-approved, TASKS row 15)", flush=True)
        return

    if not args.repo or "/" not in args.repo:
        raise SystemExit("--execute requires --repo <user>/<name>")
    sub = args.path_in_repo or (d.parent.name + "-final")
    api.create_repo(repo_id=args.repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(d), repo_id=args.repo,
                      path_in_repo=sub, repo_type="model",
                      commit_message=f"backup {sub} (nano_SLMs)")
    print(f"[backup] uploaded {len(files)} files -> "
          f"https://huggingface.co/{args.repo}/tree/main/{sub}", flush=True)


if __name__ == "__main__":
    main()
