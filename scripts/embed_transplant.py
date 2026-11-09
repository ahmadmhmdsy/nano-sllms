"""KT-1 (ladder Phase 1, TASKS row 43): SmolLM2-360M embedding transplant.

CPU-ONLY by design (co-run-safe with any GPU job; no CUDA calls anywhere).

Reads the teacher's tied input embedding [49152, 960] from the local HF cache
snapshot, aligns our 32,768 CodeLlama sentencepiece pieces to the teacher's
49,152 Llama-3 BPE pieces at the RAW-BYTE level, lifts d960 -> d1024 with a
seeded isometric matrix, and saves data/kt/embed_init_smol360.pt for the
config-gated train.init_embeddings hook in scripts/train.py.

Method (docs/plans/2026-09-10_kt_ladder_plan.md section 4):
  1. Teacher model.embed_tokens.weight, bf16 on disk -> fp32 extraction
     (our runs stay fp16 per the sm_75 constraint; the hook casts on copy).
  2. Exact piece alignment: pieces are compared as RAW BYTES, which normalizes
     the SP margin-marker vs BPE space-marker conventions automatically (both
     are byte 0x20) plus newline/tab byte pieces (<0x0A> vs the BPE byte char).
  3. Fallback for unmatched pieces: re-tokenize OUR piece (decoded from its
     bytes) with the teacher tokenizer (add_special_tokens=False) and average
     the sub-embeddings (OMP-lite; cf. arXiv 2506.06607).
  4. Still-unmatched rows (incl. specials and ids >= len(tokenizer)): seeded
     Gaussian scaled to the teacher mean row norm - fresh-init semantics.
  5. Isometric lift d960 -> d1024: E_student = E_teacher @ W with
     W [960, 1024] and W @ W.T = I_960 (orthonormal ROWS), from the QR of a
     seeded Gaussian. Norms and pairwise angles are preserved exactly.
     NOTE: the plan text says "row-orthonormal 1024x960"; a 1024x960 matrix
     cannot have orthonormal rows (1024 > 960). The implemented lift is the
     intended isometry - W.T is the 1024x960 matrix with orthonormal columns.

Gate (handoff step 1): report the exact-match rate; investigate if < ~60%.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = Path(
    "C:/Users/AhmadMhmoud/.cache/huggingface/hub/"
    "models--HuggingFaceTB--SmolLM2-360M/snapshots/"
    "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
)
DEFAULT_OUTPUT = ROOT / "data" / "kt" / "embed_init_smol360.pt"
BYTE_PIECE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def bytes_to_unicode() -> dict:
    """GPT-2 byte-level alphabet: byte value -> printable unicode char.

    Deterministic and identical to the reference implementation used by every
    byte-level BPE tokenizer (GPT-2, Llama-3), so the reverse map decodes any
    teacher piece back to its raw bytes.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


CHAR2BYTE = {c: b for b, c in bytes_to_unicode().items()}
SP_SPACE = "\u2581"  # sentencepiece margin marker ("▁")


def sp_piece_bytes(piece) -> bytes | None:
    """CodeLlama sentencepiece piece -> raw bytes (SP space marker = 0x20)."""
    if piece is None or piece == "":
        return None
    m = BYTE_PIECE_RE.match(piece)
    if m:
        return bytes([int(m.group(1), 16)])
    return piece.replace(SP_SPACE, " ").encode("utf-8")


def bpe_piece_bytes(piece, char_to_byte: dict) -> bytes | None:
    """Byte-level BPE piece -> raw bytes (undoes the byte->unicode alphabet)."""
    if piece is None or piece == "":
        return None
    out = bytearray()
    for ch in piece:
        b = char_to_byte.get(ch)
        if b is None:
            out += ch.encode("utf-8")  # non-alphabet unicode: keep as utf-8
        else:
            out.append(b)
    return bytes(out)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_teacher_embeddings(snapshot: Path):
    """Load model.embed_tokens.weight [49152, 960] as fp32 (CPU only)."""
    from safetensors.torch import load_file

    weights = []
    single = snapshot / "model.safetensors"
    index = snapshot / "model.safetensors.index.json"
    if single.is_file() and single.stat().st_size > 0:
        weights = [single]
    elif index.is_file():
        wm = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        weights = [snapshot / s for s in sorted(set(wm.values()))]
    if not weights:
        raise FileNotFoundError(f"no teacher weights found under {snapshot}")
    for w in weights:
        sd = load_file(str(w), device="cpu")
        if "model.embed_tokens.weight" in sd:
            return sd["model.embed_tokens.weight"].to(torch.float32), w
    raise KeyError("model.embed_tokens.weight not in teacher shards")


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER,
                    help="SmolLM2-360M snapshot dir in the local HF cache")
    ap.add_argument("--tokenizer", default="codellama/CodeLlama-7b-hf",
                    help="our sentencepiece tokenizer (repo-wide default)")
    ap.add_argument("--vocab-size", type=int, default=32768,
                    help="our model vocab (padded past len(tokenizer))")
    ap.add_argument("--dim", type=int, default=1024, help="our hidden size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    # --- teacher side -----------------------------------------------------
    emb_t, weights_file = load_teacher_embeddings(args.teacher)
    t_rows, t_dim = emb_t.shape
    print(f"[kt1] teacher embeddings: {tuple(emb_t.shape)} fp32 "
          f"(from {weights_file.name})", flush=True)
    print("[kt1] hashing teacher weights (sha256)...", flush=True)
    teacher_sha = sha256_of(weights_file)

    teach_tok = AutoTokenizer.from_pretrained(str(args.teacher))
    our_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n_our = len(our_tok)
    if args.vocab_size < n_our:
        raise SystemExit(
            f"[kt1] FAIL: vocab_size {args.vocab_size} < len(our tokenizer) "
            f"{n_our} - matrix could not hold real ids")
    print(f"[kt1] tokenizers: teacher={len(teach_tok)} pieces, "
          f"ours={n_our} pieces (model vocab {args.vocab_size})", flush=True)

    teach_ids = list(range(min(len(teach_tok), t_rows)))
    teach_pieces = teach_tok.convert_ids_to_tokens(teach_ids)
    teach_special = set(teach_tok.all_special_ids)
    teacher_by_bytes: dict = {}
    skipped_teacher_specials = 0
    for j, piece in zip(teach_ids, teach_pieces):
        if j in teach_special or (
            isinstance(piece, str) and piece.startswith("<|") and piece.endswith("|>")
        ):
            skipped_teacher_specials += 1
            continue
        b = bpe_piece_bytes(piece, CHAR2BYTE)
        if b and b not in teacher_by_bytes:
            teacher_by_bytes[b] = j  # lowest id wins (most frequent piece)
    print(f"[kt1] teacher byte index: {len(teacher_by_bytes)} unique byte "
          f"strings ({skipped_teacher_specials} specials skipped)", flush=True)

    # --- our side: exact match, then fallback, then random ----------------
    our_ids = list(range(n_our))
    our_pieces = our_tok.convert_ids_to_tokens(our_ids)
    our_special = set(our_tok.all_special_ids)

    E_new = torch.zeros(args.vocab_size, t_dim, dtype=torch.float32)
    gen = torch.Generator().manual_seed(args.seed)
    mean_row_norm = emb_t.norm(dim=1).mean().item()

    n_exact = n_fallback = n_special = 0
    fallback_subtok_counts = []
    random_ids = []
    for i, piece in zip(our_ids, our_pieces):
        if i in our_special:
            n_special += 1
            continue  # filled as random below
        b = sp_piece_bytes(piece)
        tid = teacher_by_bytes.get(b) if b is not None else None
        if tid is not None:
            E_new[i] = emb_t[tid]
            n_exact += 1
            continue
        # fallback: re-tokenize our piece TEXT (decoded from bytes) with the
        # teacher tokenizer; average the sub-embeddings (OMP-lite).
        text = (b.decode("utf-8", errors="replace")
                if b is not None else (piece or ""))
        sub = [s for s in teach_tok(text, add_special_tokens=False)["input_ids"]
               if 0 <= s < t_rows]
        if sub:
            E_new[i] = emb_t[sub].mean(dim=0)
            n_fallback += 1
            fallback_subtok_counts.append(len(sub))
        else:
            random_ids.append(i)

    # random-init rows: unmatched text pieces + specials + padding tail
    for i in random_ids + sorted(our_special) + list(range(n_our, args.vocab_size)):
        E_new[i] = torch.randn(t_dim, generator=gen)
        E_new[i] *= mean_row_norm / E_new[i].norm()
    n_random = len(random_ids)
    n_padding = max(0, args.vocab_size - n_our)

    # --- isometric lift d960 -> d1024 -------------------------------------
    g = torch.Generator().manual_seed(args.seed)
    G = torch.randn(args.dim, t_dim, dtype=torch.float64, generator=g)
    Q, _ = torch.linalg.qr(G)          # [dim, t_dim], orthonormal columns
    W = Q.T                            # [t_dim, dim], orthonormal rows
    ortho_res = (W @ W.T - torch.eye(t_dim, dtype=torch.float64)).abs().max().item()
    E_final = E_new @ W.to(torch.float32)

    # --- report + gate ----------------------------------------------------
    denom_nonspecial = max(1, n_our - n_special)
    rate_all = n_exact / max(1, n_our)
    rate_ns = n_exact / denom_nonspecial
    avg_sub = (sum(fallback_subtok_counts) / len(fallback_subtok_counts)
               if fallback_subtok_counts else 0.0)
    print(f"[kt1] alignment: exact={n_exact:,} fallback={n_fallback:,} "
          f"(avg {avg_sub:.2f} sub-pieces) random={n_random:,} "
          f"special={n_special} padding={n_padding:,}", flush=True)
    print(f"[kt1] exact-match rate: {rate_ns:.1%} of non-special ids "
          f"({rate_all:.1%} of all {n_our})", flush=True)
    gate_ok = rate_ns >= 0.60
    print(f"[kt1] GATE exact-match >= 60%: "
          f"{'PASS' if gate_ok else 'FAIL - investigate alignment'}", flush=True)
    print(f"[kt1] lift: {t_dim}->{args.dim} orthonormal-rows W, "
          f"max|WW^T - I| = {ortho_res:.2e} (isometric, geometry preserved)",
          flush=True)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "KT-1 embedding transplant init (TASKS row 43; plan section 4)",
        "seed": args.seed,
        "teacher_snapshot": str(args.teacher),
        "teacher_safetensors_sha256": teacher_sha,
        "teacher_shape": [t_rows, t_dim],
        "teacher_disk_dtype": "bf16 -> extracted fp32",
        "student_shape": [args.vocab_size, args.dim],
        "student_tokenizer": args.tokenizer,
        "teacher_tokenizer": getattr(teach_tok, "name_or_path", str(args.teacher)),
        "counts": {
            "exact": n_exact, "fallback": n_fallback, "random_text": n_random,
            "special_random": n_special, "padding_random": n_padding,
            "our_ids": n_our, "vocab_size": args.vocab_size,
            "teacher_index_unique_bytes": len(teacher_by_bytes),
        },
        "rates": {"exact_all": rate_all, "exact_nonspecial": rate_ns,
                  "gate_60pct_pass": bool(gate_ok)},
        "lift": {"method": "E_teacher @ W, W [960x1024] orthonormal rows "
                           "(W W^T = I); seeded QR, geometry preserved",
                 "seed": args.seed, "max_wwt_minus_i": ortho_res},
        "random_row_scale": mean_row_norm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embed_tokens": E_final, "meta": meta}, args.output)
    print(f"[kt1] SAVED {args.output} "
          f"({E_final.shape[0]}x{E_final.shape[1]} fp32)", flush=True)
    print("[kt1] meta counts: " + json.dumps(meta["counts"]), flush=True)
    if not gate_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
