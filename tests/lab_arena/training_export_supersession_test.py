"""A regrade retains original training inputs without retaining old labels."""
import copy

import pytest

from lab_arena import contracts, training_input
from lab_arena.training_export import ExportError, _select_judge, assemble_example
from scripts.export_arena_training import export_run
from tests.lab_arena.training_export_test import CLAIM_HASH, fixture, save_input


def regraded_example():
    values = fixture()
    old = copy.deepcopy(values["round_row"]["configuration_doc"]["scorer_policy"])
    new = copy.deepcopy(old)
    new.setdefault("judge_models", {})["role_batch_check"] = "google/gemini-replacement"
    new.setdefault("env_bindings", {})["ARENA_ROLE_BATCH_MODEL"] = "google/gemini-replacement"
    row = values["round_row"]
    row.update(scoring_generation_floor=7, stage_generation=9)
    row["configuration_doc"]["scorer_policy"] = new
    values["scoring_policy_history"] = {"round_id": row["round_id"], "revisions": [{
        "previous_generation_floor": 0, "generation_floor": 7,
        "old_scorer_policy": old, "new_scorer_policy": new}]}
    original = values["runs"][1]
    original["stage_generation"] = 2
    replacement = copy.deepcopy(original)
    replacement.update(run_id="judge:2", attempt=2, stage_generation=9, scoring_generation_floor=7)
    values["runs"].append(replacement)
    document = training_input.read_input(values["objects"], original["run_id"], "score", 1, CLAIM_HASH)["input"]
    document["scorer_policy"] = copy.deepcopy(new)
    save_input(values["objects"], replacement["run_id"], "score", document)
    return values


def test_regrade_exports_current_label_and_both_exact_policy_inputs():
    values = regraded_example()
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert result["score_run_id"] == "judge:2"
    assert result["scoring_generation_floor"] == 7
    assert result["scoring_policy_history"] == values["scoring_policy_history"]
    revisions = values["scoring_policy_history"]["revisions"]
    attempts = {item["run_id"]: item for item in result["attempts"]}
    assert attempts["judge:1"]["scoring_policy_hash"] == contracts.document_hash(revisions[0]["old_scorer_policy"])
    assert attempts["judge:2"]["scoring_policy_hash"] == contracts.document_hash(revisions[0]["new_scorer_policy"])
    assert result["costs"]["all_related_attempts_included"]
    assert {"judge:1", "judge:2"}.issubset(result["input_snapshots"])


def test_pending_regrade_never_falls_back_to_previously_accepted_label():
    values = regraded_example()
    values["runs"][-1].update(status="pending", terminal_cause=None, output_ref=None, result_doc={})
    values["runs"][0]["per_icp_score"] = None
    result = assemble_example(**values)
    assert "score_run_id" not in result
    assert "final_judge_unavailable" in result["completeness"]["incomplete_reasons"]
    assert "judge:1" in result["input_snapshots"]


@pytest.mark.parametrize("mutation", ["missing", "wrong_round", "missing_floor", "gap", "current_policy", "bool_floor"])
def test_missing_or_inconsistent_protected_policy_history_fails_closed(mutation):
    values = regraded_example()
    history = values["scoring_policy_history"]
    if mutation == "missing":
        values.pop("scoring_policy_history")
    elif mutation == "wrong_round":
        history["round_id"] += "-other"
    elif mutation == "missing_floor":
        history["revisions"][0]["generation_floor"] = 6
    elif mutation == "gap":
        history["revisions"][0]["previous_generation_floor"] = 1
    elif mutation == "bool_floor":
        history["revisions"][0]["previous_generation_floor"] = False
    else:
        history["revisions"][0]["new_scorer_policy"] = {}
    with pytest.raises(ExportError, match="scoring_policy_history"):
        assemble_example(**values)


def test_old_input_must_match_recorded_old_policy_not_current_policy():
    values = regraded_example()
    wrong = training_input.read_input(values["objects"], "judge:2", "score", 1, CLAIM_HASH)["input"]
    save_input(values["objects"], "judge:1", "score", wrong)
    result = assemble_example(**values)
    assert not result["completeness"]["complete"]
    assert "judge_input_context_mismatch" in result["completeness"]["incomplete_reasons"]


def test_multiple_revisions_preserve_the_intermediate_attempt_policy():
    values = regraded_example()
    middle = copy.deepcopy(values["round_row"]["configuration_doc"]["scorer_policy"])
    newest = copy.deepcopy(middle)
    newest["judge_models"]["role_batch_check"] = "google/gemini-next"
    newest["env_bindings"]["ARENA_ROLE_BATCH_MODEL"] = "google/gemini-next"
    values["round_row"].update(scoring_generation_floor=12, stage_generation=14)
    values["round_row"]["configuration_doc"]["scorer_policy"] = newest
    values["scoring_policy_history"]["revisions"].append({"previous_generation_floor": 7,
        "generation_floor": 12, "old_scorer_policy": middle, "new_scorer_policy": newest})
    current = copy.deepcopy(values["runs"][-1])
    current.update(run_id="judge:3", attempt=3, stage_generation=14, scoring_generation_floor=12)
    values["runs"].append(current)
    document = training_input.read_input(values["objects"], "judge:2", "score", 1, CLAIM_HASH)["input"]
    document["scorer_policy"] = newest
    save_input(values["objects"], "judge:3", "score", document)
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert result["score_run_id"] == "judge:3"
    assert next(item for item in result["attempts"] if item["run_id"] == "judge:2")["scoring_policy_hash"] == contracts.document_hash(middle)


def test_stale_round_read_never_uses_future_revision_judgment():
    values = regraded_example()
    execution = values["runs"][0]
    assert _select_judge(execution, values["runs"], scoring_generation_floor=0)["run_id"] == "judge:1"
    assert _select_judge(execution, values["runs"], scoring_generation_floor=7)["run_id"] == "judge:2"
    values["round_row"]["scoring_generation_floor"] = 0
    values["round_row"]["configuration_doc"]["scorer_policy"] = values["scoring_policy_history"]["revisions"][0]["old_scorer_policy"]
    values.pop("scoring_policy_history")
    with pytest.raises(ExportError, match="scoring_policy_history_invalid"):
        assemble_example(**values)


@pytest.mark.parametrize("revision", [-1, True, "7", None])
def test_invalid_judge_revision_fails_closed(revision):
    values = regraded_example()
    values["runs"][-1]["scoring_generation_floor"] = revision
    with pytest.raises(ExportError, match="scoring_generation_floor_invalid"):
        assemble_example(**values)


def test_read_only_export_loads_protected_history_for_regraded_round():
    values = regraded_example()
    history_reads = []

    class Store:
        def get_run(self, run_id):
            assert run_id == values["run_id"]
            return values["runs"][0]

        def get_round(self, round_id):
            assert round_id == values["round_row"]["round_id"]
            return values["round_row"]

        def get_submission(self, submission_id):
            assert submission_id == values["submission_row"]["submission_id"]
            return values["submission_row"]

        def scoring_policy_history(self, round_id):
            history_reads.append(round_id)
            return values["scoring_policy_history"]

        def list_runs(self, round_id):
            assert round_id == values["round_row"]["round_id"]
            return values["runs"]

        def list_ledger(self, **identity):
            return [row for row in values["ledger_rows"] if row["run_id"] == identity["run_id"]]

    result = export_run(Store(), values["objects"], values["run_id"])
    assert history_reads == [values["round_row"]["round_id"]]
    assert result["completeness"]["complete"]
    assert result["score_run_id"] == "judge:2"
    assert {"judge:1", "judge:2"}.issubset(result["input_snapshots"])
