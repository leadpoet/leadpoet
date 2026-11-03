"""Publication keeps measured scores while training evidence gates contenders."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from lab_arena import contracts, integrity, service, training_reader
from tests.lab_arena.complete_baseline_final_authority_test import fixture as baseline_fixture
from tests.lab_arena.training_eligibility_test import SOURCE_HASH, captured_example, submission
from tests.lab_arena.training_eligibility_test import ten_selected_examples
from tests.lab_arena.training_steps_test import bind_selected_output, seal

ROUND_ID = "arena-2026-10-13"
BASELINE_ID = "baseline-2026-10-13"


@pytest.mark.parametrize("error,eligible", [
    (FileNotFoundError("missing object"), True),
    (training_reader.ExportError("execution_output_hash_mismatch"), True),
    (training_reader.ExportError("ledger_membership_mismatch"), True),
    (ValueError("malformed stored evidence"), True),
    (RuntimeError("unexpected reader failure"), True),
])
@pytest.mark.parametrize("failed_count", [1, 2, 3, 10])
def test_export_integrity_errors_exclude_the_affected_icp_only(monkeypatch, error, eligible, failed_count):
    examples = ten_selected_examples()
    row = {"round_id": ROUND_ID, "configuration_doc": {"stage_1_icp_count": 5, "stage_2_icp_count": 5,
                                "training_minimum_icp_coverage_percent": 80}}
    obj = SimpleNamespace(
        _store=SimpleNamespace(get_submission=lambda _: submission()), _objects=object(),
        _is_daily_baseline=lambda *_: False,
        _selected_accepted_execution_runs=lambda *_: {
            position: {"run_id": run_id} for position, run_id in enumerate(examples)},
    )

    def export(_store, _objects, run_id):
        if int(run_id.split("-")[1]) >= 10 - failed_count:
            raise error
        return examples[run_id]

    monkeypatch.setattr(training_reader, "export_run", export)
    result = service.ArenaService._submission_training_eligibility(obj, row, "sub", [])
    assert result["eligible"] is (eligible and failed_count <= 2)
    assert result["capture"]["counts"]["complete_runs"] == 10 - failed_count
    assert result["capture"]["counts"]["rejected_runs"] == [{
        "run_id": f"run-{position}", "reasons": ["selected_run_export_invalid"],
        "error_class": type(error).__name__,
    } for position in range(10 - failed_count, 10)]


def _replace_exact(value, old, new):
    if isinstance(value, dict):
        return {_replace_exact(key, old, new): _replace_exact(item, old, new)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_exact(item, old, new) for item in value]
    return new if value == old else value


def _export(run_id, submission_id, *, baseline=False):
    example = _replace_exact(captured_example(), "run", run_id)
    example = _replace_exact(example, "sub", submission_id)
    example["role"] = "baseline" if baseline else "miner"
    if baseline:
        example["source"]["hash_origin"] = "computed_from_frozen_object"
    return bind_selected_output(example)


def _publication_harness(monkeypatch):
    obj, round_row, baseline_runs, baseline_judges, writes, _, _ = baseline_fixture()
    round_row["round_id"] = ROUND_ID
    round_row["participants"][0]["submission_id"] = BASELINE_ID
    round_row["configuration_doc"]["baseline_hotkey"] = "baseline-hotkey"
    for run in baseline_runs + baseline_judges:
        run.update(round_id=ROUND_ID, submission_id=BASELINE_ID)
    round_row["configuration_doc"]["reasoning_capture_policy"] = "full_generated_thinking_v1"
    round_row["configuration_doc"]["promotion_margin"] = 1.0
    miner_runs = []
    for index, baseline_run in enumerate(baseline_runs):
        miner_runs.append({**baseline_run, "run_id": f"miner-{index}",
                           "submission_id": "miner", "miner_hotkey": "miner-hotkey",
                           "per_icp_score": 3.0,
                           "output_ref": f"miner-output-{index}"})
    miner_judges = [{**baseline_judges[index], "run_id": f"miner-judge-{index}",
                     "submission_id": "miner", "miner_hotkey": "miner-hotkey",
                     "scored_run_id": f"miner-{index}", "per_icp_score": 3.0}
                    for index in range(2)]
    executions = baseline_runs + miner_runs
    judgments = baseline_judges + miner_judges
    baseline_submission = {"submission_id": BASELINE_ID, "is_king": True,
                           "miner_hotkey": "baseline-hotkey"}
    miner_submission = {**submission(), "submission_id": "miner", "is_king": False}
    submissions = {BASELINE_ID: baseline_submission, "miner": miner_submission}
    examples = {run["run_id"]: _export(run["run_id"], run["submission_id"],
                                        baseline=run["submission_id"] == BASELINE_ID)
                for run in executions}
    exported = []
    def export(_store, _objects, run_id):
        exported.append(run_id)
        return examples.get(run_id)
    monkeypatch.setattr(training_reader, "export_run", export)
    transitions = []
    def transition(round_id, previous, status, patch):
        transitions.append((round_id, previous, status, deepcopy(patch)))
        return {"status": "ok"}
    obj._store = SimpleNamespace(
        list_runs=lambda _rid, kind=None: executions if kind == "execute"
            else judgments if kind == "score" else executions + judgments,
        get_submission=lambda submission_id: submissions.get(submission_id),
        transition_round=transition,
        cancel_round=lambda rid, reason: writes.append((rid, reason)) or {"status": "cancelled"},
    )
    obj._objects = object()
    obj._execution_infrastructure_failures = lambda _reads: {}
    obj._participant_incomplete_reasons = lambda _row, _reads, _blocked: {}
    obj._submission_cost_eligibility = lambda *args, **kwargs: {
        "cost_summary": None, "eligible": True, "eligibility_reason": "eligible"}
    return obj, round_row, submissions, executions, examples, transitions, writes, exported


def _published(harness):
    obj, row, _submissions, _runs, _examples, transitions, writes, _exported = harness
    result = obj.publish(row["round_id"])
    assert result["status"] == "ok"
    assert len(transitions) == 1
    assert transitions[0][1:3] == ("scored", "published")
    assert writes == []
    return transitions[0][3]["publication_doc"]


def _ranking(publication, submission_id):
    return next(item for item in publication["final_ranking"]
                if item["submission_id"] == submission_id)


def test_verified_candidate_is_crowned_with_measured_scores(monkeypatch):
    harness = _publication_harness(monkeypatch)
    publication = _published(harness)
    miner = _ranking(publication, "miner")
    baseline = _ranking(publication, BASELINE_ID)
    assert miner["final_score"] == 3.0
    assert baseline["final_score"] == 0.0
    assert miner["eligible"] is True
    assert miner["training_data_eligibility"]["eligible"] is True
    assert baseline["training_data_eligibility"]["source_review"]["exemption"] == "organizer_baseline"
    assert publication["king_decision"]["outcome"] == "crowned"
    assert publication["king_decision"]["winner_submission_id"] == "miner"


@pytest.mark.parametrize("margin,cost_eligible,audited,decision", [
    (3.0, True, True, "crowned"),  # Equality is a qualifying score.
    (3.01, True, False, "no_king"),
    (3.0, False, False, "no_king"),
])
def test_per_icp_full_thinking_audits_every_possible_winner_only(
    monkeypatch, margin, cost_eligible, audited, decision,
):
    harness = _publication_harness(monkeypatch)
    obj, row, _submissions, _runs, _examples, _transitions, _writes, exported = harness
    row["configuration_doc"].update(
        integrity_policy=integrity.POLICY,
        sourcing_cost_eligibility_policy=contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
        promotion_margin=margin,
    )
    obj._submission_cost_eligibility = lambda _row, submission_id, *_args, **_kwargs: {
        "cost_summary": None,
        "eligible": submission_id == BASELINE_ID or cost_eligible,
        "eligibility_reason": "eligible" if submission_id == BASELINE_ID or cost_eligible
                              else "cost_per_company_exceeded",
    }
    obj._qualified_company_counts = lambda *_args, **_kwargs: {0: 1, 1: 1}
    obj._per_icp_cost_eligibility = lambda *_args, **_kwargs: {"eligible": True}
    publication = _published(harness)
    miner = _ranking(publication, "miner")
    baseline = _ranking(publication, BASELINE_ID)
    assert publication["king_decision"]["outcome"] == decision
    assert miner["final_score"] == 3.0
    assert miner["eligible"] is cost_eligible
    if audited:
        assert miner["training_data_eligibility"]["eligible"] is True
        assert exported == ["miner-0", "miner-1", "exec-0", "exec-1"]
    else:
        assert miner["training_data_eligibility"] == {
            "status": "not_assessed", "eligible": None, "reasons": [],
        }
        assert exported == []
    if audited:
        assert baseline["training_data_eligibility"]["eligible"] is True
    else:
        assert baseline["training_data_eligibility"] == {
            "status": "not_assessed", "eligible": None, "reasons": [],
        }


def test_crowned_round_keeps_baseline_training_reward_gate(monkeypatch):
    harness = _publication_harness(monkeypatch)
    obj, row, _submissions, _runs, examples = harness[:5]
    row["configuration_doc"].update(
        integrity_policy=integrity.POLICY,
        sourcing_cost_eligibility_policy=contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
    )
    obj._qualified_company_counts = lambda *_args, **_kwargs: {0: 1, 1: 1}
    obj._per_icp_cost_eligibility = lambda *_args, **_kwargs: {"eligible": True}
    examples.pop("exec-1")
    publication = _published(harness)
    assert publication["king_decision"]["outcome"] == "crowned"
    baseline = _ranking(publication, BASELINE_ID)
    assert baseline["final_score"] == 0.0
    assert baseline["eligible"] is False
    assert baseline["eligibility_reason"] == "training_data_incomplete"
    assert "selected_run_export_missing" in baseline["training_data_eligibility"]["reasons"]


def test_all_qualifiers_failing_training_leave_baseline_unassessed(monkeypatch):
    harness = _publication_harness(monkeypatch)
    obj, row, submissions = harness[:3]
    row["configuration_doc"].update(
        integrity_policy=integrity.POLICY,
        sourcing_cost_eligibility_policy=contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
    )
    obj._qualified_company_counts = lambda *_args, **_kwargs: {0: 1, 1: 1}
    obj._per_icp_cost_eligibility = lambda *_args, **_kwargs: {"eligible": True}
    submissions["miner"]["code_review_doc"] = {}
    publication = _published(harness)
    miner = _ranking(publication, "miner")
    assert miner["eligible"] is False
    assert miner["eligibility_reason"] == "training_data_incomplete"
    assert publication["king_decision"]["outcome"] == "no_king"
    assert _ranking(publication, BASELINE_ID)["training_data_eligibility"] == {
        "status": "not_assessed", "eligible": None, "reasons": [],
    }


def test_tied_qualifiers_are_both_audited_before_stable_winner(monkeypatch):
    harness = _publication_harness(monkeypatch)
    obj, row, submissions, executions = harness[:4]
    row["configuration_doc"].update(
        integrity_policy=integrity.POLICY,
        sourcing_cost_eligibility_policy=contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
        promotion_margin=3.0,
    )
    row["participants"].append({"submission_id": "miner-2", "miner_hotkey": "miner-2-hotkey",
                                "is_king": False})
    submissions["miner-2"] = {**submissions["miner"], "submission_id": "miner-2",
                              "miner_hotkey": "miner-2-hotkey"}
    executions.extend({**run, "run_id": run["run_id"].replace("miner-", "miner-2-"),
                       "submission_id": "miner-2", "miner_hotkey": "miner-2-hotkey"}
                      for run in list(executions) if run["submission_id"] == "miner")
    obj._qualified_company_counts = lambda *_args, **_kwargs: {0: 1, 1: 1}
    obj._per_icp_cost_eligibility = lambda *_args, **_kwargs: {"eligible": True}
    audited = []

    def training(_round, submission_id, _runs, **_kwargs):
        audited.append(submission_id)
        return {"eligible": True, "reasons": [], "source_review": {"eligible": True},
                "capture": {"eligible": True}}

    obj._submission_training_eligibility = training
    publication = _published(harness)
    assert set(audited) == {"miner", "miner-2", BASELINE_ID}
    assert publication["king_decision"]["winner_submission_id"] == "miner"
    assert _ranking(publication, "miner-2")["training_data_eligibility"]["eligible"] is True


def test_baseline_cost_penalty_zero_is_the_qualifier_threshold(monkeypatch):
    harness = _publication_harness(monkeypatch)
    obj, row, _submissions, executions = harness[:4]
    for run in executions:
        if run["submission_id"] == BASELINE_ID:
            run["per_icp_score"] = 4.0
    row["configuration_doc"].update(
        integrity_policy=integrity.POLICY,
        sourcing_cost_eligibility_policy=contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
    )
    obj._qualified_company_counts = lambda *_args, **_kwargs: {0: 1, 1: 1}
    obj._per_icp_cost_eligibility = lambda *_args, **_kwargs: {"eligible": True}
    obj._submission_cost_eligibility = lambda _round, submission_id, *_args, **_kwargs: {
        "cost_summary": None,
        "eligible": submission_id != BASELINE_ID,
        "eligibility_reason": "cost_per_company_exceeded" if submission_id == BASELINE_ID
                              else "eligible",
    }
    audited = []

    def training(_round, submission_id, _runs, **_kwargs):
        audited.append(submission_id)
        return {"eligible": True, "reasons": [], "source_review": {"eligible": True},
                "capture": {"eligible": True}}

    obj._submission_training_eligibility = training
    publication = _published(harness)
    assert _ranking(publication, BASELINE_ID)["final_score"] == 0.0
    assert _ranking(publication, "miner")["final_score"] == 3.0
    assert publication["king_decision"]["outcome"] == "crowned"
    assert audited == ["miner", BASELINE_ID]


@pytest.mark.parametrize("damage,reason", [
    (lambda submissions, examples: submissions["miner"].update(code_review_doc={}),
     "source_review_not_passed"),
    (lambda submissions, examples: submissions["miner"]["code_review_doc"]["research_control"].update(
        source_archive_sha256=contracts.hash_bytes(b"stale archive")),
     "model_directed_source_unverified"),
    (lambda submissions, examples: examples.pop("miner-1"),
     "selected_run_export_missing"),
])
def test_missing_source_or_capture_excludes_candidate_without_cancel_or_score_change(
    monkeypatch, damage, reason,
):
    harness = _publication_harness(monkeypatch)
    damage(harness[2], harness[4])
    publication = _published(harness)
    miner = _ranking(publication, "miner")
    assert miner["final_score"] == 3.0
    assert miner["eligible"] is False
    assert miner["eligibility_reason"] == "training_data_incomplete"
    assert reason in miner["training_data_eligibility"]["reasons"]
    assert publication["king_decision"]["outcome"] == "no_king"


def test_baseline_missing_capture_is_reported_but_still_measured_reference(monkeypatch):
    harness = _publication_harness(monkeypatch)
    for run in harness[3]:
        if run["submission_id"] == BASELINE_ID:
            run["per_icp_score"] = 4.0
    harness[4].pop("exec-1")
    publication = _published(harness)
    baseline = _ranking(publication, BASELINE_ID)
    assert baseline["final_score"] == 4.0
    assert baseline["training_data_eligibility"]["eligible"] is False
    assert "selected_run_export_missing" in baseline["training_data_eligibility"]["reasons"]
    assert publication["king_decision"]["outcome"] == "no_king"
    assert publication["king_decision"]["winner_submission_id"] is None


def test_publication_checks_final_selected_accepted_attempt(monkeypatch):
    harness = _publication_harness(monkeypatch)
    runs, examples, exported = harness[3], harness[4], harness[7]
    retry = {**next(run for run in runs if run["run_id"] == "miner-0"),
             "run_id": "miner-0-retry", "attempt": 2,
             "output_ref": "miner-output-0-retry"}
    runs.append(retry)
    examples[retry["run_id"]] = _export(retry["run_id"], "miner")
    publication = _published(harness)
    assert _ranking(publication, "miner")["eligible"] is True
    assert "miner-0-retry" in exported
    assert "miner-0" not in exported
    assert _ranking(publication, "miner")["final_score"] == 3.0


def test_missing_selected_position_keeps_legacy_full_coverage_requirement(monkeypatch):
    harness = _publication_harness(monkeypatch)
    obj, row, _, runs, _, _, _, exported = harness
    result = obj._submission_training_eligibility(
        row, "miner", [run for run in runs if run["run_id"] != "miner-1"])
    assert result["eligible"] is False
    assert result["reasons"] == ["selected_execution_coverage_incomplete"]
    assert exported == ["miner-0"]
    assert result["capture"]["counts"]["missing_selected_runs"] == 1


def test_pre_scoring_review_gate_rejects_stale_pass(monkeypatch):
    harness = _publication_harness(monkeypatch)
    obj, row, submissions = harness[:3]
    obj._require_code_review("miner", row)
    submissions["miner"]["code_review_doc"]["research_control"]["source_archive_sha256"] = (
        contracts.hash_bytes(b"older archive"))
    with pytest.raises(service.ServiceError) as failure:
        obj._require_code_review("miner", row)
    assert failure.value.code == "code_review_required"


def test_legacy_publication_never_reads_training_exports(monkeypatch):
    harness = _publication_harness(monkeypatch)
    harness[1]["configuration_doc"].pop("reasoning_capture_policy")
    monkeypatch.setattr(training_reader, "export_run", lambda *_: pytest.fail("legacy capture read"))
    publication = _published(harness)
    assert _ranking(publication, "miner")["final_score"] == 3.0
    assert "training_data_eligibility" not in _ranking(publication, "miner")


def test_frozen_stale_source_proof_finishes_scoring_but_cannot_win(monkeypatch):
    harness = _publication_harness(monkeypatch)
    obj, row, submissions = harness[:3]
    submissions["miner"]["status"] = "frozen"
    submissions["miner"]["code_review_doc"].pop("research_control")
    # Both lease delivery and scoring call this gate for frozen entrants.
    obj._require_code_review("miner", row)
    publication = _published(harness)
    assert _ranking(publication, "miner")["eligible"] is False
    assert _ranking(publication, "miner")["final_score"] == 3.0
    assert publication["king_decision"]["outcome"] == "no_king"


@pytest.mark.parametrize('actual,eligible', [(0, False), (7, False), (8, True), (10, True)])
def test_selected_execution_coverage_uses_the_whole_icp_bank(monkeypatch, actual, eligible):
    examples = ten_selected_examples()
    selected = {position: {'run_id': f'run-{position}'} for position in range(actual)}
    row = {'round_id': ROUND_ID, 'configuration_doc': {
        'stage_1_icp_count': 5, 'stage_2_icp_count': 5,
        'training_minimum_icp_coverage_percent': 80}}
    obj = SimpleNamespace(_store=SimpleNamespace(get_submission=lambda _: submission()),
        _objects=object(), _is_daily_baseline=lambda *_: False,
        _selected_accepted_execution_runs=lambda *_: selected)
    exported = []
    def export(_store, _objects, run_id):
        exported.append(run_id)
        return examples[run_id]
    monkeypatch.setattr(training_reader, 'export_run', export)
    result = service.ArenaService._submission_training_eligibility(obj, row, 'sub', [])
    assert result['eligible'] is eligible, result
    assert result['capture']['counts']['complete_runs'] == actual
    assert result['capture']['counts']['incomplete_runs'] == 10 - actual
    assert len(exported) == actual
    if actual < 10:
        assert result['capture']['counts']['missing_icp_positions'] == list(range(actual, 10))


@pytest.mark.parametrize('damage', ['unexpected_position', 'duplicate_run', 'source_review', 'configuration'])
def test_coverage_allowance_cannot_hide_global_selection_source_or_policy_errors(monkeypatch, damage):
    examples = ten_selected_examples()
    selected = {position: {'run_id': f'run-{position}'} for position in range(8)}
    source = submission()
    configuration = {'stage_1_icp_count': 5, 'stage_2_icp_count': 5,
                     'training_minimum_icp_coverage_percent': 80}
    if damage == 'unexpected_position':
        selected[10] = {'run_id': 'run-8'}
    elif damage == 'duplicate_run':
        selected[7] = selected[6]
    elif damage == 'source_review':
        source['code_review_status'] = 'failed'
    else:
        configuration['training_minimum_icp_coverage_percent'] = True
    obj = SimpleNamespace(_store=SimpleNamespace(get_submission=lambda _: source),
        _objects=object(), _is_daily_baseline=lambda *_: False,
        _selected_accepted_execution_runs=lambda *_: selected)
    monkeypatch.setattr(training_reader, 'export_run', lambda _store, _objects, run_id: examples[run_id])
    result = service.ArenaService._submission_training_eligibility(
        obj, {'round_id': ROUND_ID, 'configuration_doc': configuration}, 'sub', [])
    assert not result['eligible']
