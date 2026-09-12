from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena.service import ArenaService, ServiceError


def _service(row):
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(get_submission=lambda _: row)
    return service


@pytest.mark.parametrize("status", [None, "pending", "reviewing", "error", "rejected"])
def test_only_a_complete_pass_can_enter_scoring(status):
    row = {"submission_id": "miner", "miner_hotkey": "miner-owner", "code_review_status": status}
    service = _service(row)
    round_row = {"round_id": "arena-test", "configuration_doc": {"baseline_hotkey": "host"}}
    with pytest.raises(ServiceError, match="code_review_required"):
        service._require_code_review("miner", round_row)
    row["code_review_status"] = "passed"
    service._require_code_review("miner", round_row)


def test_baseline_exemption_requires_all_three_identity_fields():
    row = {"submission_id": "baseline-test", "miner_hotkey": "host", "is_king": True}
    round_row = {"round_id": "arena-test", "configuration_doc": {"baseline_hotkey": "host"}}
    service = _service(row)
    service._require_code_review(row["submission_id"], round_row)
    for field, bad in [("submission_id", "miner"), ("miner_hotkey", "miner"), ("is_king", False)]:
        wrong = dict(row, **{field: bad})
        with pytest.raises(ServiceError, match="code_review_required"):
            _service(wrong)._require_code_review(wrong["submission_id"], round_row)


def test_worker_loads_full_round_identity_and_skips_baseline_and_completed_reviews():
    baseline = {"submission_id": "baseline-test", "miner_hotkey": "host", "is_king": True}
    passed = {"submission_id": "passed", "code_review_status": "passed"}
    pending = {"submission_id": "pending", "code_review_status": "pending"}
    calls = []
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(code_reviewer=SimpleNamespace(
        review=lambda row: calls.append(row["submission_id"]) or {"status": "passed"}
    ))
    service._store = SimpleNamespace(list_submissions=lambda _, status: [baseline, passed, pending] if status == "accepted" else [])
    service.active_rounds = lambda: [{"round_id": "arena-test"}]
    service._round = lambda _: {"round_id": "arena-test", "configuration_doc": {"baseline_hotkey": "host"}}
    assert service.review_pending_submissions() == {"reviewed": 1}
    assert calls == ["pending"]


def test_public_status_does_not_disclose_source_or_findings():
    secret = "never-publish-private-source"
    row = {"code_review_status": "rejected", "code_review_doc": {
        "summary": secret, "findings": [{"evidence": secret}],
        "reviewed_files": [secret], "model": "anthropic/claude-sonnet-5",
        "file_count": 41, "source_bytes": 384634,
    }}
    public = ArenaService._public_code_review(row)
    assert secret not in str(public)
    assert public["status"] == "rejected" and public["file_count"] == 41


def test_inflight_review_can_finish_after_cutoff_within_existing_benchmark_window():
    now = datetime(2026, 9, 11, tzinfo=timezone.utc)
    baseline = {"submission_id": "baseline-test", "miner_hotkey": "host", "is_king": True,
                "source_ref": "baseline", "source_size_bytes": 10, "status": "accepted"}
    miner = {"submission_id": "miner", "miner_hotkey": "miner-owner", "is_king": False,
             "code_review_status": "reviewing", "code_review_attempts": 1,
             "source_ref": "miner", "source_size_bytes": 10, "status": "accepted"}
    round_row = {"round_id": "arena-test", "champion_funding_frozen": True,
                 "configuration_doc": {"baseline_hotkey": "host",
                 "schedule": {"benchmark_deadline": "2026-09-11T00:30:00Z"}}}
    updates = []
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(clock=lambda: now)
    service._clock = lambda: now
    service._round = lambda _: round_row
    service._initial_baseline = lambda _: baseline
    service._store = SimpleNamespace(
        freeze_champion_funding=lambda _round_id: {"status": "existing"},
        list_submissions=lambda _, status: [baseline, miner] if status == "accepted" else [],
        update_submission=lambda *args: updates.append(args) or {"status": "ok"},
    )
    with pytest.raises(ServiceError, match="code_review_pending"):
        service.freeze_participants("arena-test")
    assert not updates
    miner["code_review_status"] = "passed"
    assert len(service.freeze_participants("arena-test")) == 2
    assert all(call[3] == "frozen" for call in updates)
    updates.clear()
    miner["code_review_status"] = "reviewing"
    now += timedelta(minutes=30)
    assert len(service.freeze_participants("arena-test")) == 1
    assert any(call[1] == "miner" and call[4] == {"rejection_rule": "code_review_incomplete"} for call in updates)
