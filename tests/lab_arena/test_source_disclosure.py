"""Source becomes public with completed evaluation, never by age alone."""

import io
import tarfile
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceConfig
from lab_arena.store import ArenaStore, PostgrestTransport
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


@pytest.fixture
def source_route():
    """Use the real route/service/store with the existing disclosure fixtures."""
    with ExitStack() as stack:
        def build(fault, *, published):
            payload = bundle({"harness.py": "def run_icp(icp): return []\n"})
            objects = Objects(payload)
            reads = []
            row = round_row()
            row["configuration_doc"]["mode"] = "live"
            sub = submission(status="frozen" if published else "rejected", source_size_bytes=len(payload))

            def handler(request):
                assert request.method == "GET"
                reads.append(request)
                response = fault(request)
                if response is not None:
                    return response
                values = [sub] if request.url.path.endswith("/lab_arena_submissions") else [row]
                return httpx.Response(200, json=values)

            def unused(*args, **kwargs):
                pytest.fail("public source read called a write/provider dependency")

            database = stack.enter_context(httpx.Client(transport=httpx.MockTransport(handler)))
            service = ArenaService(ServiceConfig(
                mode="live", store=ArenaStore(PostgrestTransport("https://project.example", service_key="sb_secret_test", http_client=database)),
                object_store=objects, signer=None, chain=SimpleNamespace(),
                verify_signature=unused, daily_icp_source=unused,
                banned_hotkeys_source=unused, broker_factory=unused, clock=lambda: PUBLISHED,
            ))
            api = stack.enter_context(TestClient(create_app(service), raise_server_exceptions=False))
            return api, objects, reads
        yield build


@pytest.mark.parametrize("error_type", [
    httpx.ReadTimeout, httpx.ReadError, httpx.ConnectTimeout, httpx.ConnectError,
    httpx.WriteTimeout, httpx.WriteError, httpx.RemoteProtocolError,
])
@pytest.mark.parametrize("table", ["lab_arena_submissions", "lab_arena_rounds"])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("recovers", [False, True])
def test_source_route_transient_read_failure_is_bounded(
    source_route, error_type, table, published, recovers,
):
    failures = 0
    armed = True
    def fault(request):
        nonlocal failures
        if armed and request.url.path.endswith("/" + table) and (not recovers or failures == 0):
            failures += 1
            raise error_type("private diagnostic", request=request)
    api, objects, reads = source_route(fault, published=published)
    first = api.get("/arena/v1/submissions/sub-preview/code")
    expected = 200 if published else 403
    assert first.status_code == (expected if recovers else 503)
    assert len(reads) == (3 if recovers else (2 if table == "lab_arena_submissions" else 3))
    if not recovers:
        assert first.json() == {"status": "unavailable", "code": "arena_store_unavailable"}
        assert first.headers["retry-after"] == "1"
        assert objects.reads == 0
    assert first.headers["cache-control"] == "no-store"
    assert first.headers["x-content-type-options"] == "nosniff"
    assert "private diagnostic" not in first.text
    armed = False
    second = api.get("/arena/v1/submissions/sub-preview/code")
    assert second.status_code == expected
    if published:
        assert second.json()["files"] == [{"path": "harness.py", "content": "def run_icp(icp): return []\n"}]
    else:
        assert second.json()["code"] == "source_not_public" and objects.reads == 0


@pytest.mark.parametrize("failure", [httpx.PoolTimeout, 429, 502, 503, 504])
@pytest.mark.parametrize("table", ["lab_arena_submissions", "lab_arena_rounds"])
def test_source_route_temporary_pressure_keeps_bounded_retry(source_route, failure, table):
    armed = True
    def fault(request):
        if armed and request.url.path.endswith("/" + table):
            if isinstance(failure, type):
                raise failure("private diagnostic", request=request)
            return httpx.Response(failure, json={"message": "private diagnostic", "code": "PGRST000"})
    api, objects, reads = source_route(fault, published=False)
    first = api.get("/arena/v1/submissions/sub-preview/code")
    assert first.status_code == 503 and first.headers["retry-after"] == "1"
    assert first.headers["cache-control"] == "no-store"
    assert "private diagnostic" not in first.text
    expected_reads = 1 if table == "lab_arena_submissions" else 2
    # Local pool pressure still fails immediately; transient HTTP reads use
    # the same one-retry budget as ambiguous transport failures.
    assert len(reads) == expected_reads + (0 if failure is httpx.PoolTimeout else 1)
    armed = False
    assert api.get("/arena/v1/submissions/sub-preview/code").status_code == 403
    assert objects.reads == 0


def test_source_route_missing_submission_stays_404(source_route):
    api, objects, reads = source_route(lambda request: httpx.Response(200, json=[]), published=False)
    response = api.get("/arena/v1/submissions/sub-missing/code")
    assert response.status_code == 404 and response.json()["code"] == "submission_missing"
    assert len(reads) == 1 and objects.reads == 0


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
