"""Task 4: HF pinned-repository network lane (opt-in, never run by default).

The unit lane never touches the network. This lane exercises the ONLY code
that performs live network I/O (HFSourceReader.collect_file_hashes) against
one explicitly pinned test repository revision with token=False, and
persists a report under pipeline_v2/test-artifacts/. The pinned revision
must be declared explicitly via environment variables; nothing here is ever
inferred, and no hash is recorded that was not verifiably downloaded.
"""
import os
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from slm_train.data.sources.huggingface import HFSourceReader

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "test-artifacts"
REPOSITORY_ENV = "PIPELINE_V2_NETWORK_HF_REPOSITORY"
REVISION_ENV = "PIPELINE_V2_NETWORK_HF_REVISION"
FILES_ENV = "PIPELINE_V2_NETWORK_HF_FILES"


def _pinned_declaration():
    repository = os.environ.get(REPOSITORY_ENV, "").strip()
    revision = os.environ.get(REVISION_ENV, "").strip()
    files = [value.strip() for value in os.environ.get(FILES_ENV, "").split(",") if value.strip()]
    if not repository or not revision or not files:
        pytest.skip(
            "network lane requires an explicitly pinned repository, revision, and file "
            "list via PIPELINE_V2_NETWORK_HF_REPOSITORY / _REVISION / _FILES")
    return repository, revision, files


@pytest.mark.network
def test_collect_file_hashes_on_pinned_repository_revision():
    repository, revision, files = _pinned_declaration()
    hashes = HFSourceReader.collect_file_hashes(revision, repository, files)
    assert set(hashes) == set(files)
    for name, digest in sorted(hashes.items()):
        assert isinstance(digest, str) and len(digest) == 64 and int(digest, 16) >= 0
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(ch if ch.isalnum() else "-" for ch in f"{repository}--{revision}")
    report_path = ARTIFACTS_DIR / f"hf-source-{slug}-{stamp}.json"
    report = {
        "kind": "hf-source-network-report",
        "repository": repository,
        "revision": revision,
        "token_used": False,
        "file_digests": hashes,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "lane": "pytest -m network",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert report_path.is_file()
