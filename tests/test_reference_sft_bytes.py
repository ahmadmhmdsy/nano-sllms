r"""tests for scripts/reference_sft_bytes.py --append-eos (EOS-terminated reference corpus).

Dependency-free runner (repo convention): execute directly with the venv python,
    & .\.venv\Scripts\python.exe tests\test_reference_sft_bytes.py
The CLI checks write to a throwaway directory inside the workspace and clean up after
themselves; no existing corpus under data/ is read or touched. CPU only.
"""
import _common  # noqa: F401  (path setup)
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from _common import run

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reference_sft_bytes.py"

import reference_sft_bytes as rsb  # noqa: E402  (scripts/ is on sys.path via _common)

TEXTS = ["a = 1\n", "b = 2\n", "c = 3\n"]


def _plain_bytes(texts: list) -> bytes:
    joined = "".join(texts)
    values = np.frombuffer(joined.encode("utf-8"), dtype=np.uint8).astype(np.uint16) + rsb.BYTE_OFFSET
    return values.astype("<u2").tobytes()


def _eos_id() -> int:
    from glm53_flash.tokenizer import ByteTokenizer
    return ByteTokenizer.eos_id


def t_pack_texts_default_off_is_byte_identical_to_the_plain_encoding():
    blob, size = rsb.pack_texts(TEXTS, False, None)
    assert blob == _plain_bytes(TEXTS), "default OFF changed the packed bytes"
    assert size == len(blob) // 2


def t_pack_texts_append_eos_terminates_every_example():
    eos = _eos_id()
    blob, size = rsb.pack_texts(TEXTS, True, None)
    values = [int(v) for v in np.frombuffer(blob, dtype="<u2")]
    expected = []
    for text in TEXTS:
        expected.extend(rsb.BYTE_OFFSET + byte for byte in text.encode("utf-8"))
        expected.append(eos)
    assert values == expected
    assert size == len(values)


def _run_cli(out_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(out_dir),
                           "--instances", "3", "--val-instances", "1", *extra],
                          cwd=ROOT, capture_output=True, text=True)


def t_cli_default_off_has_no_eos_and_records_the_flag():
    eos = _eos_id()
    tmp = Path(tempfile.mkdtemp(prefix=".tmp_ref_sft_off_", dir=ROOT))
    try:
        proc = _run_cli(tmp)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        for name in ("train.bin", "val.bin"):
            values = np.frombuffer((tmp / name).read_bytes(), dtype="<u2")
            assert int(np.count_nonzero(values == eos)) == 0, f"{name} contains an EOS token"
        meta = json.loads((tmp / "meta.json").read_text(encoding="utf-8"))
        assert meta["append_eos"] is False and meta["eos_id"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_cli_append_eos_terminates_examples_and_records_the_id():
    eos = _eos_id()
    tmp = Path(tempfile.mkdtemp(prefix=".tmp_ref_sft_on_", dir=ROOT))
    try:
        proc = _run_cli(tmp, "--append-eos")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        meta = json.loads((tmp / "meta.json").read_text(encoding="utf-8"))
        assert meta["append_eos"] is True and meta["eos_id"] == eos
        expected = {"train.bin": meta["train"]["examples"], "val.bin": meta["val"]["examples"]}
        for name, count in expected.items():
            values = np.frombuffer((tmp / name).read_bytes(), dtype="<u2")
            assert int(values[-1]) == eos, f"{name} does not end with EOS"
            assert int(np.count_nonzero(values == eos)) == count, f"{name} EOS count != examples"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_cli_append_eos_rebuild_is_byte_identical():
    def hashes(directory: Path) -> dict:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}

    tmp = Path(tempfile.mkdtemp(prefix=".tmp_ref_sft_rep_", dir=ROOT))
    try:
        first = _run_cli(tmp, "--append-eos")
        assert first.returncode == 0, first.stdout + first.stderr
        before = hashes(tmp)
        second = _run_cli(tmp, "--append-eos", "--force")
        assert second.returncode == 0, second.stdout + second.stderr
        assert hashes(tmp) == before
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run({
        "pack_texts default OFF is byte-identical": t_pack_texts_default_off_is_byte_identical_to_the_plain_encoding,
        "pack_texts append-eos terminates every example": t_pack_texts_append_eos_terminates_every_example,
        "CLI default OFF has no EOS and records the flag": t_cli_default_off_has_no_eos_and_records_the_flag,
        "CLI append-eos terminates examples and records the id": t_cli_append_eos_terminates_examples_and_records_the_id,
        "CLI append-eos rebuild is byte-identical": t_cli_append_eos_rebuild_is_byte_identical,
    })
