"""The paid E2E succeeds only with complete strict exports for both roles."""

import copy

import pytest

from scripts import verify_arena_full_thinking_e2e as e2e


def participant():
    executions = [{"run_id": f"run-{i}", "icp_position": i} for i in range(10)]
    scores = [{"run_id": f"score-{i}", "scored_run_id": f"run-{i}"} for i in range(10)]
    per_icp = [{"icp_position": i, "records": 1, "complete": True,
                "selected_research_complete": True} for i in range(10)]
    splits = {"train": [{} for _ in range(10)], "eval": []}
    return executions, scores, per_icp, splits


@pytest.mark.parametrize("damage,reason", [
    (lambda part: part[0].pop(), "accepted executions and linked real scores"),
    (lambda part: part[1][-1].update(scored_run_id="unlinked"), "accepted executions and linked real scores"),
    (lambda part: part[2][-1].update(icp_position=8), "ICP positions are incomplete"),
    (lambda part: part[2][-1].update(records=0), "complete strict training record"),
    (lambda part: part[2][-1].update(selected_research_complete=False), "complete strict training record"),
    (lambda part: part[3]["train"].pop(), "fewer than ten records"),
])
def test_valid_candidate_cannot_hide_incomplete_baseline(damage, reason):
    candidate = participant()
    baseline = copy.deepcopy(participant())
    damage(baseline)
    with pytest.raises(AssertionError, match="baseline: .*" + reason):
        e2e._require_strict_exports(candidate, baseline)


def test_both_complete_participants_pass_and_candidate_is_checked_too():
    candidate = participant()
    baseline = participant()
    e2e._require_strict_exports(candidate, baseline)
    candidate[2][0]["records"] = 0
    with pytest.raises(AssertionError, match="candidate: .*complete strict training record"):
        e2e._require_strict_exports(candidate, baseline)


def test_ancillary_judge_gap_keeps_raw_incomplete_without_failing_selected_proof():
    candidate = participant()
    baseline = participant()
    candidate[2][0].update(complete=False, selected_research_complete=True)
    e2e._require_strict_exports(candidate, baseline)


def test_frozen_eighty_percent_accepts_two_strict_gaps_and_reports_coverage():
    candidate, baseline = participant(), participant()
    candidate[2][-1].update(records=0, selected_research_complete=False)
    candidate[3]["train"].pop()
    coverage = e2e._require_strict_exports(candidate, baseline, minimum_coverage_percent=80)
    assert coverage["candidate"] == {"complete_icps": 9, "required_icps": 8,
                                     "total_icps": 10, "minimum_coverage_percent": 80}
    assert coverage["baseline"]["complete_icps"] == 10
    candidate[2][-2].update(records=0, selected_research_complete=False)
    candidate[3]["train"].pop()
    coverage = e2e._require_strict_exports(candidate, baseline, minimum_coverage_percent=80)
    assert coverage["candidate"]["complete_icps"] == 8
    candidate[2][-3].update(records=0, selected_research_complete=False)
    with pytest.raises(AssertionError, match="candidate: complete strict training record coverage 7/10"):
        e2e._require_strict_exports(candidate, baseline, minimum_coverage_percent=80)


def test_frozen_eighty_percent_still_requires_all_real_scores():
    candidate, baseline = participant(), participant()
    candidate[1].pop()
    with pytest.raises(AssertionError, match="ten accepted executions and linked real scores"):
        e2e._require_strict_exports(candidate, baseline, minimum_coverage_percent=80)


@pytest.mark.parametrize("role", ["candidate", "baseline"])
def test_sufficient_record_count_cannot_hide_failed_capture_gate(role):
    reports = {name: {"eligible": True, "reasons": []}
               for name in ("candidate", "baseline")}
    reports[role] = {"eligible": False, "reasons": ["selected_run_identity_unverified"]}
    with pytest.raises(AssertionError, match=role + ": stored capture gate failed"):
        e2e._require_strict_exports(participant(), participant(),
                                   minimum_coverage_percent=80, capture_reports=reports)


def test_both_verified_capture_gates_and_strict_exports_pass():
    reports = {name: {"eligible": True, "reasons": []}
               for name in ("candidate", "baseline")}
    e2e._require_strict_exports(participant(), participant(),
                               minimum_coverage_percent=80, capture_reports=reports)
