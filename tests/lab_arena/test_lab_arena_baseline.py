"""The public baseline enters every daily round through source admission."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lab_arena import source_bundle
from lab_arena.service import ArenaService, ServiceError


def _submission(submission_id: str, hotkey: str, *, status: str = "accepted") -> dict:
    return {
        "submission_id": submission_id,
        "round_id": "arena-2026-09-05",
        "miner_hotkey": hotkey,
        "status": status,
        "source_ref": "arena/arena-2026-09-05/sources/%s.tar.gz" % submission_id,
        "source_size_bytes": 123,
        "consent": {"public_rerun": True},
        "is_king": False,
    }


class _Objects:
    def __init__(self) -> None:
        self.values = {}

    def put(self, ref, data):
        self.values[ref] = bytes(data)

    def get_bounded(self, ref, max_bytes):
        if ref not in self.values:
            raise KeyError(ref)
        value = self.values[ref]
        if len(value) > max_bytes:
            raise ValueError("too large")
        return value


class _Store:
    def __init__(self, current: dict, submissions: list, published: list = None) -> None:
        self.rounds = {current["round_id"]: current}
        self.submissions = {row["submission_id"]: row for row in submissions}
        for row in published or []:
            self.rounds[row["round_id"]] = row

    def get_round(self, round_id):
        return self.rounds.get(round_id)

    def list_rounds(self, *, status=None, limit=None, **_kwargs):
        rows = list(reversed(list(self.rounds.values())))
        return [row for row in rows if status is None or row.get("status") == status][
            :limit
        ]

    def list_submissions(self, round_id, *, status=None):
        rows = [
            row
            for row in self.submissions.values()
            if row.get("round_id") == round_id
        ]
        return [dict(row) for row in rows if status is None or row["status"] == status]

    def get_submission(self, submission_id):
        row = self.submissions.get(submission_id)
        return dict(row) if row is not None else None

    def update_submission(
        self, _round_id, submission_id, expected, target, patch=None
    ):
        row = self.submissions[submission_id]
        if row["status"] != expected:
            return {"status": "stale"}
        row.update(patch or {})
        row["status"] = target
        return {"status": "ok"}

    def register_submission(self, round_id, submission_id, hotkey, document):
        self.submissions[submission_id] = {
            "round_id": round_id,
            "submission_id": submission_id,
            "miner_hotkey": hotkey,
            "status": "uploading",
            **document,
        }
        return {
            "status": "registered",
            "submission_id": submission_id,
            "source_ref": document["source_ref"],
        }


def _round(round_id: str = "arena-2026-09-05") -> dict:
    return {
        "round_id": round_id,
        "status": "open",
        "configuration_doc": {
            "mode": "live",
            "max_challengers": 1,
            "baseline_hotkey": "baseline",
            "baseline_source_url": "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz",
        },
    }


def _service(store: _Store, objects: _Objects, payload: bytes) -> ArenaService:
    service = object.__new__(ArenaService)
    service._store = store
    service._objects = objects
    service._config = SimpleNamespace(
        mode="live",
        defaults=SimpleNamespace(
            baseline_source_url="https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz"
        ),
        baseline_source_fetcher=lambda _url, _limit: payload,
    )
    return service


def _archive(tmp_path) -> bytes:
    source = tmp_path / "baseline"
    source.mkdir()
    (source / "harness.py").write_text(
        "def run_icp(icp):\n    return []\n", encoding="utf-8"
    )
    target = tmp_path / "baseline.tar.gz"
    source_bundle.write_source_archive(source, target)
    return target.read_bytes()


def test_baseline_download_uses_the_same_source_checks_and_freezes(tmp_path):
    current = _round()
    challenger = _submission("sub-c", "challenger")
    overflow = _submission("sub-d", "overflow")
    store = _Store(current, [challenger, overflow])
    objects = _Objects()
    service = _service(store, objects, _archive(tmp_path))

    participants = service.freeze_participants(current["round_id"])

    assert [(row["submission_id"], row["is_king"]) for row in participants] == [
        ("sub-c", False),
        ("baseline-2026-09-05", True),
    ]
    assert store.submissions["baseline-2026-09-05"]["status"] == "frozen"
    assert "source_sha256" not in store.submissions["baseline-2026-09-05"]
    assert "source_cache_key" not in store.submissions["baseline-2026-09-05"]
    assert store.submissions["sub-d"]["status"] == "rejected"
    assert list(objects.values) == [
        "arena/arena-2026-09-05/sources/baseline-2026-09-05.tar.gz"
    ]


def test_yesterdays_miner_winner_never_replaces_todays_baseline(tmp_path):
    current = _round()
    baseline = _submission("baseline-2026-09-05", "baseline")
    baseline["is_king"] = True
    previous = {
        "round_id": "arena-2026-09-04",
        "status": "published",
        "configuration_doc": {"mode": "live"},
        "publication_doc": {"king_decision": {"king_submission_id": "old-winner"}},
        "king_hotkey": "miner-winner",
        "king_outcome": "crowned",
    }
    store = _Store(current, [baseline], [previous])
    service = _service(store, _Objects(), _archive(tmp_path))

    participants = service.freeze_participants(current["round_id"])

    assert [(row["miner_hotkey"], row["is_king"]) for row in participants] == [
        ("baseline", True)
    ]
    assert "king-arena-2026-09-05" not in store.submissions


def test_invalid_public_baseline_source_prevents_the_round_from_starting():
    current = _round()
    service = _service(_Store(current, []), _Objects(), b"not a source archive")

    with pytest.raises(ServiceError, match="baseline_source_invalid"):
        service.freeze_participants(current["round_id"])
