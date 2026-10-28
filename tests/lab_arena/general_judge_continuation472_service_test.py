"""Prior-day work is not rescored; expired unmeasured work cannot publish."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import service
from tests.lab_arena.complete_baseline_final_authority_test import fixture
from tests.lab_arena.icp_fixtures import daily_icps


def test_oct13_unmeasured_round_cancels_without_rescoring_after_deadline():
    instance, row, sources, judges, writes, clock, _ = fixture()
    row["evaluation_date"] = "2026-10-13"
    sources[0]["per_icp_score"] = None
    judges[0].update(status="failed", terminal_cause="judge_error")
    clock[0] = datetime(2026, 10, 14, tzinfo=timezone.utc)
    instance._record_recovered_judgment_scores = lambda _row: (_ for _ in ()).throw(
        AssertionError("prior evaluation must not be rescored"))

    assert instance.publish(row["round_id"]) == {"status": "cancelled"}
    assert writes == [(row["round_id"], "scoring_incomplete")]
    assert sources[0]["per_icp_score"] is None
    assert judges[0]["status"] == "failed"


def test_shadow_rollout_matches_database_reward_gate():
    _, row, *_ = fixture()
    row["evaluation_date"] = "2026-10-13"
    assert not service._judge_continuation_enabled(row)
    row["configuration_doc"]["mode"] = "shadow"
    row["rewards_enabled"] = False
    row["created_at"] = "2026-10-13T16:29:59Z"
    assert not service._judge_continuation_enabled(row)
    row["created_at"] = "2026-10-13T16:31:00Z"
    assert service._judge_continuation_enabled(row)
    row["rewards_enabled"] = True
    assert not service._judge_continuation_enabled(row)
    row["evaluation_date"] = "2026-10-14"
    assert service._judge_continuation_enabled(row)


def test_invalid_company_only_judge_output_is_rejected_before_accept(monkeypatch):
    """Legacy integrity judging must check the group before durable acceptance."""

    scorer_policy = {
        "scoring_adapter_version": "qualification_integrity_v2",
        "max_scored_companies": 5,
    }
    row = {"round_id": "arena-2099-10-14", "configuration_doc": {
        "scorer_policy": scorer_policy}}
    run = {"run_id": "arena-2099-10-14:submission:1:0:score:1",
        "round_id": row["round_id"], "runner_hotkey": "validator",
        "kind": "score", "scored_run_id": "arena-2099-10-14:submission:1:0:1",
        "icp_position": 0, "company_judgment_refs": None}
    instance = object.__new__(service.ArenaService)
    instance._request_round = lambda *_args, **_kwargs: ({"hotkey": "validator",
        "body": {"run_id": run["run_id"], "result": {"terminal_status": "accepted"},
                 "output": {"schema_version": "leadpoet.lab_arena.scoring_output.v1",
                            "scored_run_id": run["scored_run_id"],
                            "breakdowns": [{"final_score": 0.0}]}}}, row)
    instance._require_validator_authority = lambda *_args: None
    instance._lease_token_for_run = lambda *_args: "lease"
    instance.evaluation_icps = lambda *_args: daily_icps()[:1]
    instance._store = SimpleNamespace(get_run=lambda rid: run if rid == run["run_id"]
        else {"output_ref": "accepted-execution"})
    instance._objects = SimpleNamespace(get_bounded=lambda *_args: b"{}",
                                        put=lambda *_args: pytest.fail("invalid judge persisted"))
    monkeypatch.setattr(service.contracts, "validate_run_result", lambda document: document)
    monkeypatch.setattr(service.contracts, "validate_scorer_policy",
                        lambda document: document)
    monkeypatch.setattr(service, "validate_output_document",
                        lambda *_args, **_kwargs: {"companies": []})

    with pytest.raises(service.ServiceError, match="score_invalid"):
        instance.handle_complete({})
