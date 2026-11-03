"""A publisher that loses its round must stop holding up later rounds."""

from copy import deepcopy
from datetime import datetime, timezone
import threading
from types import SimpleNamespace

import pytest

from lab_arena import contracts
from lab_arena.driver import _advance_active
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStoreUnavailable


@pytest.fixture
def publication():
    participants = [
        {"submission_id": "baseline", "miner_hotkey": "5" + "A" * 47, "is_king": True},
        {"submission_id": "challenger", "miner_hotkey": "5" + "B" * 47, "is_king": False},
    ]
    row = {
        "round_id": "old", "status": "scored", "participants": participants,
        "finalists": ["challenger"], "configuration_doc": {"schedule": {
            "submission_cutoff": "2026-10-28T00:00:00Z",
            "publication_deadline": "2026-10-28T22:30:00Z",
        }},
    }
    entries = [dict(submission_id=p["submission_id"], hotkey=p["miner_hotkey"],
                    is_king=p["is_king"], stage1_score=score, final_score=score)
               for p, score in zip(participants, (50.0, 60.0))]
    fixture = SimpleNamespace(row=row, assessed=[], transitions=[], reads=0,
                              after_assessment=lambda _: None, read_failure=None,
                              challenger_eligible=True, next_opened=[])

    def get_round(round_id):
        if round_id == "next":
            return dict(deepcopy(row), round_id="next", status="committed")
        fixture.reads += 1
        if fixture.read_failure == "missing":
            return None
        if fixture.read_failure == "error":
            raise ArenaStoreUnavailable("round read unavailable")
        return deepcopy(row)

    def transition(_round_id, old, new, patch):
        fixture.transitions.append((old, new, deepcopy(patch)))
        if row["status"] != old:
            return {"status": "stale", "round_status": row["status"]}
        row.update(deepcopy(patch), status=new)
        return {"status": "ok"}

    def assess(_row, submission_id, *_args, **_kwargs):
        fixture.assessed.append(submission_id)
        fixture.after_assessment(submission_id)
        return {"eligible": submission_id == "baseline" or fixture.challenger_eligible,
                "reasons": [] if submission_id == "baseline" or fixture.challenger_eligible
                else ["capture_incomplete"]}

    service = object.__new__(ArenaService)
    service._lock = threading.Lock()
    service._clock = lambda: datetime(2026, 10, 29, tzinfo=timezone.utc)
    service._require_round_ownership = lambda _: None
    service._require_round_mode = lambda value: value
    service._store = SimpleNamespace(get_round=get_round, list_runs=lambda *_a, **_k: [],
                                     transition_round=transition)
    service._baseline_fully_succeeded = lambda _: True
    service._frozen_unfinished_judgment = lambda _: False
    service._execution_infrastructure_failures = lambda _: {}
    service._participant_incomplete_reasons = lambda *_: {}
    service._score_entries_from_runs = lambda *_a, **_k: deepcopy(entries)
    service._submission_cost_eligibility = lambda *_a, **_k: {
        "eligible": True, "cost_summary": None, "eligibility_reason": "eligible"}
    service._full_thinking_required = lambda _: True
    service._submission_training_eligibility = assess
    service.active_rounds = lambda: [{"round_id": "old"}, {"round_id": "next"}]

    def advance(round_id):
        assert not service._lock.locked()
        return service._advance_round_locked(round_id)

    service.advance_round = advance
    service.open_stage = lambda round_id, _stage: fixture.next_opened.append(round_id) or {"status": "ok"}
    fixture.service = service
    return fixture


@pytest.mark.parametrize("status", ["published", "cancelled"])
def test_stale_participant_boundary_releases_lock_and_advances_next_round(publication, status):
    def another_publisher(_submission):
        assert publication.service._lock.locked()
        publication.row["status"] = status

    publication.after_assessment = another_publisher
    assert "failed" not in _advance_active(publication.service)
    assert publication.assessed == ["baseline"]
    assert publication.transitions == []
    assert not publication.service._lock.locked()
    assert publication.next_opened == ["next"]


@pytest.mark.parametrize("failure, exception", [("missing", ServiceError), ("error", ArenaStoreUnavailable)])
def test_failed_status_refresh_cannot_publish_or_assess_more_participants(publication, failure, exception):
    publication.after_assessment = lambda _: setattr(publication, "read_failure", failure)
    with pytest.raises(exception):
        publication.service._advance_round_locked("old")
    assert publication.assessed == ["baseline"]
    assert publication.transitions == []
    assert not publication.service._lock.locked()


def test_change_after_last_assessment_still_uses_final_cas(publication):
    def another_publisher(submission_id):
        if submission_id == "challenger":
            publication.row.update(status="published", publication_doc={"winner": "other"})

    publication.after_assessment = another_publisher
    assert publication.service._advance_round_locked("old")["status"] == "stale"
    assert publication.assessed == ["baseline", "challenger"]
    assert len(publication.transitions) == 1
    assert publication.row["publication_doc"] == {"winner": "other"}
    assert not publication.service._lock.locked()


@pytest.mark.parametrize("eligible", [True, False])
def test_unchanged_status_preserves_complete_publication(publication, eligible):
    publication.challenger_eligible = eligible
    result = publication.service._advance_round_locked("old")
    challenger = publication.row["participants"][1]["miner_hotkey"]
    decision = {"outcome": "crowned" if eligible else "no_king",
                "king_submission_id": "challenger" if eligible else None,
                "king_hotkey": challenger if eligible else "",
                "winner_submission_id": "challenger" if eligible else None}
    common = {"cost_summary": None, "eligible": True, "eligibility_reason": "eligible",
              "training_data_eligibility": {"eligible": True, "reasons": []}}
    challenger_eligibility = dict(common, eligible=eligible,
        eligibility_reason="eligible" if eligible else "training_data_incomplete",
        training_data_eligibility={"eligible": eligible, "reasons": [] if eligible else ["capture_incomplete"]})
    assert publication.row["publication_doc"] == {
        "schema_version": contracts.PUBLICATION_SCHEMA_VERSION, "round_id": "old",
        "participants": [{"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"],
                          "is_baseline": p["is_king"]} for p in publication.row["participants"]],
        "stage1_ranking": [{"submission_id": "challenger", "stage1_score": 60.0, "rank": 1}],
        "finalists": ["challenger"],
        "final_ranking": [
            dict(rank=1, submission_id="challenger", final_score=60.0, is_baseline=False, **challenger_eligibility),
            dict(rank=2, submission_id="baseline", final_score=50.0, is_baseline=True, **common),
        ],
        "king_decision": decision, "published_at": "2026-10-29T00:00:00Z",
    }
    assert result == {"status": "ok", "king_outcome": decision["outcome"], "king_hotkey": decision["king_hotkey"]}
    assert publication.assessed == ["baseline", "challenger"]
    assert len(publication.transitions) == 1
