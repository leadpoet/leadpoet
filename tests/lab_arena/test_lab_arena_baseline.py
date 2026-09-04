"""The public baseline enters through normal submission and admission."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lab_arena import service as service_module
from lab_arena.service import ArenaService, ServiceError


def _submission(submission_id: str, hotkey: str, *, status: str = "accepted") -> dict:
    return {
        "submission_id": submission_id,
        "miner_hotkey": hotkey,
        "status": status,
        "image_digest": "sha256:" + submission_id[-1] * 64,
        "image_reference": "arena.example/agents/%s@sha256:%s" % (submission_id, submission_id[-1] * 64),
        "submitted_reference": "public.example/agents/%s:latest" % submission_id,
        "consent": {"public_rerun": True},
        "is_king": False,
    }


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
        return [row for row in rows if status is None or row.get("status") == status][:limit]

    def list_submissions(self, round_id, *, status=None):
        rows = [row for row in self.submissions.values() if row.get("round_id", round_id) == round_id]
        return [dict(row) for row in rows if status is None or row["status"] == status]

    def get_submission(self, submission_id):
        row = self.submissions.get(submission_id)
        return dict(row) if row is not None else None

    def update_submission(self, _round_id, submission_id, expected, target, patch):
        row = self.submissions[submission_id]
        if row["status"] != expected:
            return {"status": "stale"}
        row.update(patch)
        row["status"] = target
        return {"status": "ok"}

    def register_submission(self, round_id, submission_id, hotkey, document):
        self.submissions[submission_id] = {
            "round_id": round_id,
            "submission_id": submission_id,
            "miner_hotkey": hotkey,
            "status": "uploaded",
            **document,
        }
        return {"status": "created"}


def _service(store: _Store, *, mode: str = "live", baseline_hotkey: str = "baseline") -> ArenaService:
    service = object.__new__(ArenaService)
    service._store = store
    service._config = SimpleNamespace(mode=mode, defaults=SimpleNamespace(baseline_hotkey=baseline_hotkey))
    return service


def _round(round_id: str = "arena-2026-09-05") -> dict:
    return {
        "round_id": round_id,
        "status": "open",
        "configuration_doc": {"mode": "live", "max_challengers": 1, "baseline_hotkey": "baseline"},
    }


def test_first_same_mode_round_marks_the_normally_admitted_baseline_as_king():
    current = _round()
    baseline = _submission("sub-b", "baseline")
    challenger = _submission("sub-c", "challenger")
    overflow = _submission("sub-d", "overflow")
    # A winner in another mode does not replace this mode's initial baseline.
    shadow = {
        "round_id": "arena-2026-09-04",
        "status": "published",
        "configuration_doc": {"mode": "shadow"},
        "king_hotkey": "shadow-king",
        "king_outcome": "defended",
    }
    store = _Store(current, [baseline, challenger, overflow], [shadow])

    participants = _service(store).freeze_participants(current["round_id"])

    assert [(row["submission_id"], row["is_king"]) for row in participants] == [
        ("sub-b", True),
        ("sub-c", False),
    ]
    assert store.submissions["sub-b"]["status"] == "frozen"
    assert store.submissions["sub-b"]["is_king"] is True
    assert store.submissions["sub-d"]["status"] == "rejected"


def test_initial_baseline_must_complete_normal_image_admission():
    current = _round()
    uploaded = _submission("sub-b", "baseline", status="uploaded")
    challenger = _submission("sub-c", "challenger")
    store = _Store(current, [uploaded, challenger])

    with pytest.raises(ServiceError, match="baseline_submission_missing"):
        _service(store).freeze_participants(current["round_id"])

    assert store.submissions["sub-c"]["status"] == "accepted"


def test_missing_carried_winner_fails_but_a_fresh_incumbent_submission_is_used():
    current = _round()
    previous = {
        "round_id": "arena-2026-09-04",
        "status": "published",
        "configuration_doc": {"mode": "live"},
        "publication_doc": {"king_decision": {"king_submission_id": "old-winner"}},
        "king_hotkey": "incumbent",
        "king_outcome": "defended",
    }
    missing_store = _Store(current, [], [previous])
    with pytest.raises(ServiceError, match="incumbent_submission_missing"):
        _service(missing_store).freeze_participants(current["round_id"])

    fresh = _submission("sub-f", "incumbent")
    fresh_store = _Store(current, [fresh], [previous])
    participants = _service(fresh_store).freeze_participants(current["round_id"])
    assert [(row["submission_id"], row["is_king"]) for row in participants] == [("sub-f", True)]


def test_admission_passes_one_tick_deadline_to_resolve_and_mirror(monkeypatch):
    current = _round()
    uploaded = _submission("sub-b", "baseline", status="uploaded")
    store = _Store(current, [uploaded])
    source = object()
    destination = object()
    service = _service(store)
    service._config.source_registry = source
    service._config.registry = destination
    calls = []
    descriptor = SimpleNamespace(
        reference="public.example/agents/sub-b@sha256:" + "b" * 64,
        to_document=lambda: {"image_digest": "sha256:" + "b" * 64, "image_size_bytes": 123},
    )

    def resolve(client, _reference, _rules, *, deadline):
        calls.append(("resolve", client, deadline))
        return descriptor

    def mirror(client, resolved, repository, *, destination_client, deadline):
        calls.append(("mirror", client, destination_client, repository, deadline))
        assert resolved is descriptor
        return "arena.example/agents/sub-b@sha256:" + "b" * 64

    monkeypatch.setattr(service_module.images, "resolve_image", resolve)
    monkeypatch.setattr(service_module.images, "mirror_image", mirror)

    outcome = service._admit_one(
        current["round_id"],
        uploaded,
        rules=object(),
        repository="arena.example/agents",
        final=False,
        deadline=123.5,
    )

    assert outcome == "accepted"
    assert calls == [
        ("resolve", source, 123.5),
        ("mirror", source, destination, "arena.example/agents", 123.5),
    ]
