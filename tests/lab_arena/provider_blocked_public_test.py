"""A recovery exclusion remains public and never becomes a measured zero."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from lab_arena import public_dashboard
from lab_arena.service import ServiceError
from tests.lab_arena.public_history_test import fixture
from tests.lab_arena.public_submissions_projection_test import _row, _service


def history(round_id, excluded):
    return {"round_id": round_id, "revisions": [{"generation_floor": 7, "recovery": {
        "excluded_participants": [{**item, "status": "provider_blocked",
            "evaluation": "not_evaluated", "reason": "openrouter_zdr_route_unavailable",
            "private_proof": "private-native-receipt"} for item in excluded],
        "private_actor": "private-actor"}}]}


def assert_blocked(entry):
    assert entry["status"] == "provider_blocked"
    assert entry["score_status"] == "not_evaluated"
    assert entry["stage1_score"] is None and entry["final_score"] is None
    assert entry["eligible"] is False and entry["eligibility_reason"] == "provider_blocked"
    assert entry["is_champion"] is False
    assert entry["evaluation"]["state"] == "provider_blocked"
    assert "cost_summary" not in entry


@pytest.mark.parametrize("status", ["stage2_scoring", "published"])
def test_three_exclusions_remain_visible_despite_old_zero_scores(status):
    row = _row(status)
    row["scoring_generation_floor"] = 7
    excluded = [{"submission_id": "blocked-" + str(i), "miner_hotkey": "5Blocked" + str(i)}
                for i in range(3)]
    service, transport = _service(row)
    service._store.scoring_policy_history = lambda _: history(row["round_id"], excluded)
    transport.submissions.extend({**item, "status": "frozen", "source_ref": "private-source"}
                                 for item in excluded)
    transport.submissions.append({"submission_id": "never-public", "status": "frozen"})
    old_zeros = {item["submission_id"]: {"final_score": 0} for item in excluded}
    with patch.object(public_dashboard, "_completed_scores", return_value=old_zeros), \
            patch.object(public_dashboard, "_stage1_scores", return_value={sid: 0 for sid in old_zeros}), \
            patch.object(public_dashboard.source_disclosure, "disclosure_status", return_value={"available": False}):
        result = public_dashboard.submissions_snapshot(service, row["round_id"])
    by_id = {item["submission_id"]: item for item in result["submissions"]}
    assert len(by_id) == 6 and "never-public" not in by_id
    for item in excluded:
        assert_blocked(by_id[item["submission_id"]])
    if status == "published":
        assert by_id["champion"]["is_champion"] is True
    assert "private-native-receipt" not in json.dumps(result)
    assert "private-actor" not in json.dumps(result)


def test_history_paginates_excluded_entrants_and_preserves_real_zero():
    service, _calls, rows = fixture(1)
    row = rows[0]
    row["scoring_generation_floor"] = 7
    excluded = [{"submission_id": sid, "miner_hotkey": "5Blocked"}
                for sid in ("a-blocked", "m-blocked", "z-blocked")]
    service._store.scoring_policy_history = lambda _: history(row["round_id"], excluded)
    results, cursor = [], None
    while True:
        page = public_dashboard.history_snapshot(service, cursor=cursor, limit=2)
        results.extend(page["submissions"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(results) == 6 and len({item["submission_id"] for item in results}) == 6
    for entry in results:
        if entry["submission_id"].endswith("blocked"):
            assert_blocked(entry)
    assert next(item for item in results if item["submission_id"] == "day1-miner0")["final_score"] == 0
    filtered = public_dashboard.history_snapshot(service, hotkey="Blocked")
    assert len(filtered["submissions"]) == 3
    assert "private-native-receipt" not in json.dumps(results)


def test_history_exclusions_respect_generation_and_readmission():
    service, _calls, rows = fixture(1)
    row = rows[0]
    row["scoring_generation_floor"] = 7
    excluded = [row["publication_doc"]["participants"][1]]
    evidence = history(row["round_id"], excluded)
    evidence["revisions"].append({"generation_floor": 9, "recovery": {"excluded_participants": [
        {"submission_id": "future-blocked", "miner_hotkey": "5Blocked", "status": "provider_blocked",
         "evaluation": "not_evaluated"}]}})
    service._store.scoring_policy_history = lambda _: deepcopy(evidence)
    result = public_dashboard.history_snapshot(service)
    assert len(result["submissions"]) == 3
    assert all(item["status"] != "provider_blocked" for item in result["submissions"])
    evidence["round_id"] = "another-round"
    with pytest.raises(ServiceError, match="recovery_history_invalid"):
        public_dashboard.history_snapshot(service)


def test_recovered_history_uses_current_judge_generation():
    service, _calls, rows = fixture(1)
    row = rows[0]
    row["scoring_generation_floor"] = 7
    service._store.scoring_policy_history = lambda _: history(row["round_id"], [])
    service._store.list_runs = lambda *_a, **_k: [{
        "run_id": "old", "assignment_id": "old-score", "submission_id": "day1-miner1",
        "kind": "score", "status": "accepted", "attempt": 1, "stage_generation": 2,
        "scoring_generation_floor": 0,
    }]
    result = public_dashboard.history_snapshot(service)
    entry = next(item for item in result["submissions"] if item["submission_id"] == "day1-miner1")
    assert entry["evaluation"]["counts"]["completed"] == 0


def test_later_recovery_does_not_hide_previous_excluded_entrants():
    service, _calls, rows = fixture(1)
    row = rows[0]
    row["scoring_generation_floor"] = 9
    evidence = history(row["round_id"], [{"submission_id": "first-blocked", "miner_hotkey": "5First"}])
    later = history(row["round_id"], [{"submission_id": "later-blocked", "miner_hotkey": "5Later"}])["revisions"][0]
    later["generation_floor"] = 9
    evidence["revisions"].append(later)
    service._store.scoring_policy_history = lambda _: evidence
    result = public_dashboard.history_snapshot(service)
    blocked = [item for item in result["submissions"] if item["status"] == "provider_blocked"]
    assert {item["submission_id"] for item in blocked} == {"first-blocked", "later-blocked"}
