"""Source becomes public with completed evaluation, never by age alone."""

import io
import tarfile
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena.source_disclosure import (
    MAX_PREVIEW_FILE_BYTES, SourceDisclosureError, disclosure_status, public_source_code,
)

ACCEPTED = datetime(2026, 9, 9, 23, 55, tzinfo=timezone.utc)
PUBLISHED = datetime(2026, 9, 10, 0, 30, tzinfo=timezone.utc)


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
        "submission_id": "sub-preview", "round_id": "arena-2026-09-10", "status": "frozen",
        "accepted_at": ACCEPTED.isoformat(),
        "created_at": (ACCEPTED - timedelta(days=1)).isoformat(),
        "frozen_at": (ACCEPTED + timedelta(hours=6)).isoformat(),
        "consent": {"public_rerun": True}, "source_ref": "private-source",
        **changes,
    }


def round_row(**changes):
    return {
        "round_id": "arena-2026-09-10", "status": "published",
        "icp_set_date": "2026-09-09", "evaluation_date": "2026-09-10",
        "configuration_doc": {"schedule": {
            "submission_open": "2026-09-09T00:00:00Z",
            "submission_cutoff": "2026-09-10T00:00:00Z",
        }},
        "published_at": PUBLISHED.isoformat(),
        "publication_doc": {"participants": [{"submission_id": "sub-preview"}]},
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


def test_late_submission_released_at_evaluation_not_after_24_hours():
    payload = bundle({"harness.py": "def run_icp(icp): return []\n"})
    row = submission(source_size_bytes=len(payload))
    objects = Objects(payload)
    before = PUBLISHED - timedelta(microseconds=1)
    assert disclosure_status(row, before, round_row=round_row()) == {
        "available": False, "available_at": "2026-09-10T00:30:00Z", "url": None,
    }
    with pytest.raises(SourceDisclosureError, match="source_not_public"):
        public_source_code(objects, row, before, round_row=round_row())
    assert objects.reads == 0
    result = public_source_code(objects, row, PUBLISHED, round_row=round_row())
    assert result == {"submission_id": "sub-preview", "files": [
        {"path": "harness.py", "content": "def run_icp(icp): return []\n"},
    ], "truncated": False}
    assert disclosure_status(dict(row), before, round_row=round_row())["available"] is False


@pytest.mark.parametrize("changes", [
    {"status": "uploading"}, {"status": "rejected"}, {"consent": {}},
    {"source_ref": None}, {"status": "accepted"},
    {"round_id": "another-round"}, {"submission_id": "not-a-participant"},
])
def test_invalid_or_unaccepted_source_stays_private(changes):
    assert disclosure_status(submission(**changes), PUBLISHED, round_row=round_row())["available"] is False


@pytest.mark.parametrize("status", ["open", "committed", "stage1_scored", "scored", "cancelled"])
def test_age_does_not_release_unpublished_or_failed_evaluation(status):
    result = disclosure_status(submission(), PUBLISHED + timedelta(days=10), round_row=round_row(status=status))
    assert result == {"available": False, "available_at": None, "url": None}


@pytest.mark.parametrize("changes", [
    {"published_at": None}, {"published_at": "bad"},
    {"published_at": "2026-09-10T00:30:00"}, {"publication_doc": {}},
])
def test_missing_publication_or_participant_keeps_source_private(changes):
    assert disclosure_status(submission(), PUBLISHED, round_row=round_row(**changes))["available"] is False
    assert disclosure_status(submission(), PUBLISHED)["available"] is False


def test_same_day_legacy_publication_cannot_release_current_hidden_bank_source():
    legacy = round_row(icp_set_date=None, evaluation_date="2026-09-10")
    assert disclosure_status(submission(), PUBLISHED, round_row=legacy)["available"] is False
    assert disclosure_status(submission(), PUBLISHED + timedelta(days=1), round_row=legacy)["available"] is True


def test_publication_time_must_not_precede_bank_release():
    early = round_row(published_at=ACCEPTED.isoformat())
    assert disclosure_status(submission(), PUBLISHED, round_row=early)["available"] is False


def test_preview_is_inert_bounded_text_without_private_object_metadata():
    source = "raise RuntimeError('must not execute')\ndef run_icp(icp): return []\n"
    payload = bundle({
        "wrapped/harness.py": source, "wrapped/README.md": "<script>alert(1)</script>",
        "wrapped/.env.example": "KEY=example", "wrapped/image.png": b"\x00\xff",
        "wrapped/large.py": " " * (MAX_PREVIEW_FILE_BYTES + 1),
    })
    result = public_source_code(Objects(payload), submission(source_size_bytes=len(payload)), PUBLISHED, round_row=round_row())
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
            public_source_code(Objects(payload), submission(source_size_bytes=len(payload)), PUBLISHED, round_row=round_row())
