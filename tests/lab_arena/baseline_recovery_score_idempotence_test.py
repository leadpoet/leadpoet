"""Current continuation materializes missing baseline scores only at publication."""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import Mock

import pytest

from lab_arena import scoring
from lab_arena.service import ServiceError
from qualification.scoring.competition import apply_company_judgment_context
from tests.lab_arena.bounded_baseline_recovery_test import fixture
from tests.lab_arena.cost_eligibility_test import _company
from tests.lab_arena.partial_close_scoring_test import _score_stage_stub
from tests.test_arena_company_quality_policy import _positive_breakdown


@pytest.mark.parametrize("evaluation_date,shadow,created_at,continuation", [
    ("2026-10-13", False, "2026-10-13T17:00:00Z", False),
    ("2026-10-14", False, "2026-10-13T00:00:00Z", True),
    ("2026-10-13", True, "2026-10-13T16:29:59Z", False),
    ("2026-10-13", True, "2026-10-13T16:30:00Z", True),
])
def test_stage_two_uses_frozen_continuation_boundary(monkeypatch, evaluation_date, shadow, created_at, continuation):
    service, _, events = _score_stage_stub(monkeypatch, incomplete_submission=None, stage=2)
    row = service._round("ignored")
    row.update(evaluation_date=evaluation_date, created_at=created_at, rewards_enabled=not shadow)
    row["configuration_doc"]["mode"] = "shadow" if shadow else "live"
    service._score_recovered_baseline = Mock()
    if continuation:
        service._score_recovered_baseline.side_effect = AssertionError("late stage-one write")
    assert service.score_stage(row["round_id"], 2)["status"] == "ok"
    assert events == ["record", "transition"]
    assert service._score_recovered_baseline.call_count == (0 if continuation else 1)


def prepared(*, score=60.0, recovery=False):
    service, row, executions, judges = fixture()
    row.update(evaluation_date="2099-10-13", scoring_generation_floor=0, status="scored")
    row["configuration_doc"].update(scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64)
    row["configuration_doc"]["scorer_policy"] = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2")
    if not recovery:
        executions[:] = executions[:2]
        executions[0].update(attempt=1, run_id=executions[0]["assignment_id"] + ":1")
        judges[0]["scored_run_id"] = executions[0]["run_id"]
    outputs, decisions = {}, {}
    for position, source in enumerate(executions[:2]):
        company = _company(f"company-{position}.example", f"Company {position}")
        company["employee_count"] = "1-10"
        decision = _positive_breakdown(company["company_name"], f"company-{position}.example", f"company-{position}")
        value = score + position
        decision.update(final_score=value, intent_signal_raw=value, intent_signal_final=value)
        decision["intent_signals_detail"][0].update(raw=value, after_decay=value)
        decision = apply_company_judgment_context([company], [decision], company_quality=False)[0]
        outputs[source["run_id"]] = [company]
        decisions[source["run_id"]] = [decision]
        source["per_icp_score"] = float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
        source["qualification_doc"] = {"companies": [{key: decision[key] for key in (
            "company_index", "company_identity_key", "company_qualified", "duplicate_company")} ]}
        judges[position]["judgment_scope_doc"] = {"round_id": row["round_id"], **{
            key: row["configuration_doc"][key] for key in ("scorer_image_digest", "scorer_image_reference")}}
    service.evaluation_icps = lambda _: [{"max_companies": 1, "employee_count": ["1-10"]} for _ in range(2)]
    service._outputs_by_run = Mock(return_value=outputs)
    service._verified_breakdowns = Mock(side_effect=lambda judge, **_: deepcopy(decisions[judge["scored_run_id"]]))
    service._store.record_run_scores = Mock(return_value={"status": "stale", "round_status": "scored"})
    service._store.miner_credit_judgment_terminal = lambda _: False
    return service, row, executions, judges


@pytest.mark.parametrize("score", [0.0, 60.0, 60.12345649, 60.1234565])
def test_publication_preserves_every_accepted_projection_without_rewriting(score):
    service, row, executions, judges = prepared(score=score)
    before = deepcopy((row, executions, judges))
    service._record_recovered_judgment_scores(row)
    assert not service._unfinished_judgments(row)
    service._store.record_run_scores.assert_not_called()
    service._verified_breakdowns.assert_not_called()
    assert (row, executions, judges) == before


@pytest.mark.parametrize("floor", [0, 7])
@pytest.mark.parametrize("recovery", [False, True], ids=["ordinary-first-attempt", "infrastructure-recovery"])
def test_publication_computes_only_missing_current_score(floor, recovery):
    service, row, executions, judges = prepared(recovery=recovery)
    row["scoring_generation_floor"] = floor
    for judge in judges:
        judge["scoring_generation_floor"] = floor
    expected_qualification = deepcopy(executions[0]["qualification_doc"])
    executions[0].update(per_icp_score=None, qualification_doc=None)
    service._store.record_run_scores.return_value = {"status": "ok"}
    before = deepcopy(executions)
    service._record_recovered_judgment_scores(row)
    call = service._store.record_run_scores.call_args
    assert call.args[:2] == (row["round_id"], 1)
    assert [(item["run_id"], item["per_icp_score"]) for item in call.args[2]] == [(executions[0]["run_id"], 60.0)]
    assert call.args[2][0]["qualification_doc"] == expected_qualification
    assert call.kwargs == {"scoring_generation_floor": floor}
    assert service._verified_breakdowns.call_count == 1
    assert executions == before


def test_publication_keeps_guarded_score_write_failure():
    service, row, executions, _ = prepared()
    executions[0].update(per_icp_score=None, qualification_doc=None)
    with pytest.raises(ServiceError, match="^recovered_judgment_scores_not_recorded$"):
        service._record_recovered_judgment_scores(row)
    service._store.record_run_scores.assert_called_once()


def test_historical_judges_cannot_authorize_missing_current_scores():
    service, row, executions, _ = prepared(recovery=True)
    row["scoring_generation_floor"] = 7
    executions[0].update(per_icp_score=None, qualification_doc=None)
    service._record_recovered_judgment_scores(row)
    assert service._unfinished_judgments(row)
    service._store.record_run_scores.assert_not_called()


@pytest.mark.parametrize("missing", [False, True])
def test_invalid_current_judgment_binding_cannot_reach_publication(missing):
    service, row, executions, judges = prepared()
    judges[0]["miner_hotkey"] = "wrong-miner"
    if missing:
        executions[0].update(per_icp_score=None, qualification_doc=None)
    with pytest.raises(ServiceError, match="judgment_binding_invalid"):
        if missing:
            service._record_recovered_judgment_scores(row)
        else:
            service._unfinished_judgments(row)
    service._store.record_run_scores.assert_not_called()


def test_already_scored_baseline_still_waits_for_uncertain_costs():
    service, row, _, _ = prepared()
    service._submission_cost_eligibility = lambda *a, **k: {"eligibility_reason": "provider_cost_uncertain"}
    service._record_recovered_judgment_scores(row)
    assert not service._baseline_final_ready(row)
    service._store.record_run_scores.assert_not_called()
