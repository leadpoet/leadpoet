"""Source previews cannot bypass the submission's 24-hour privacy window."""

import io
import tarfile
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena.source_disclosure import (
    MAX_PREVIEW_FILE_BYTES, SourceDisclosureError, disclosure_status, public_source_code,
)

ACCEPTED = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)


def bundle(files):
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w:gz") as archive:
        for path, content in files.items():
            raw = content.encode("utf-8") if isinstance(content, str) else content
            item = tarfile.TarInfo(path)
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    return target.getvalue()


def submission(**changes):
    return {
        "submission_id": "sub-preview", "status": "frozen",
        "accepted_at": ACCEPTED.isoformat(),
        "created_at": (ACCEPTED - timedelta(days=1)).isoformat(),
        "frozen_at": (ACCEPTED + timedelta(hours=6)).isoformat(),
        "consent": {"public_rerun": True}, "source_ref": "private-source",
        **changes,
    }


class Objects:
    def __init__(self, payload):
        self.payload = payload
        self.reads = 0

    def get_bounded(self, ref, limit):
        self.reads += 1
        assert ref == "private-source" and len(self.payload) <= limit
        return self.payload


def test_exact_24_hour_boundary_and_server_recheck():
    payload = bundle({"harness.py": "def run_icp(icp): return []\n"})
    row = submission(source_size_bytes=len(payload))
    objects = Objects(payload)
    before = ACCEPTED + timedelta(hours=24) - timedelta(microseconds=1)
    assert disclosure_status(row, before) == {
        "available": False, "available_at": "2026-09-10T12:00:00Z", "url": None,
    }
    with pytest.raises(SourceDisclosureError, match="source_not_public"):
        public_source_code(objects, row, before)
    assert objects.reads == 0
    result = public_source_code(objects, row, ACCEPTED + timedelta(hours=24))
    assert result == {"submission_id": "sub-preview", "files": [
        {"path": "harness.py", "content": "def run_icp(icp): return []\n"},
    ], "truncated": False}
    assert disclosure_status(dict(row), before)["available"] is False


@pytest.mark.parametrize("changes", [
    {"status": "uploading"}, {"status": "rejected"}, {"consent": {}},
    {"source_ref": None}, {"accepted_at": None, "frozen_at": None},
    {"accepted_at": "bad", "frozen_at": "bad"},
    {"accepted_at": "2026-09-09T12:00:00", "frozen_at": None},
])
def test_invalid_or_unaccepted_source_stays_private(changes):
    assert disclosure_status(submission(**changes), ACCEPTED + timedelta(days=10))["available"] is False


def test_historical_rows_use_conservative_freeze_not_reservation_time():
    row = submission(accepted_at=None)
    assert disclosure_status(row, ACCEPTED + timedelta(hours=25))["available"] is False
    assert disclosure_status(row, ACCEPTED + timedelta(hours=30))["available"] is True
    row = submission(accepted_at=None, status="accepted", updated_at=ACCEPTED.isoformat())
    assert disclosure_status(row, ACCEPTED + timedelta(hours=24))["available"] is True


def test_preview_is_inert_bounded_text_without_private_object_metadata():
    source = "raise RuntimeError('must not execute')\ndef run_icp(icp): return []\n"
    payload = bundle({
        "wrapped/harness.py": source, "wrapped/README.md": "<script>alert(1)</script>",
        "wrapped/.env.example": "KEY=example", "wrapped/image.png": b"\x00\xff",
        "wrapped/large.py": " " * (MAX_PREVIEW_FILE_BYTES + 1),
    })
    result = public_source_code(Objects(payload), submission(source_size_bytes=len(payload)), ACCEPTED + timedelta(days=1))
    assert result["files"] == [
        {"path": "harness.py", "content": source},
        {"path": "README.md", "content": "<script>alert(1)</script>"},
    ]
    assert result["truncated"] is True
    assert "private-source" not in str(result)


def test_corrupt_or_unsafe_archives_cannot_be_previewed():
    for payload in [b"invalid", bundle({"harness.py": "pass", "../escape.py": "pass"}),
                    bundle({"harness.py": "pass", ".env": "PRIVATE=value"})]:
        with pytest.raises(SourceDisclosureError, match="source_unavailable"):
            public_source_code(Objects(payload), submission(source_size_bytes=len(payload)), ACCEPTED + timedelta(days=1))
