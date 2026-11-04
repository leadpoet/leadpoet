"""Verified empty and truncated replies share the frozen ICP coverage budget."""

from copy import deepcopy
from itertools import permutations

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import (
    refresh_response_links, submission, ten_selected_examples,
)
from tests.lab_arena.training_steps_test import seal


PATTERNS = [
    ("captured", "captured", "captured"),
    ("empty", "empty", "empty"),
    ("truncated", "truncated", "truncated"),
    *sorted(set(permutations(("truncated", "empty", "empty")))),
    *sorted(set(permutations(("truncated", "truncated", "empty")))),
    ("truncated", "empty", "captured"),
]


def _apply_pattern(example, pattern):
    for index, returned in enumerate(pattern):
        response = example["provider_calls"][index]["response"]
        if returned == "truncated":
            response["choices"][0]["finish_reason"] = "length"
        elif returned == "empty":
            message = response["choices"][0]["message"]
            for field in ("reasoning", "reasoning_content", "reasoning_details"):
                message.pop(field, None)
        refresh_response_links(example, index)
    return seal(example)


@pytest.mark.parametrize("pattern", PATTERNS, ids=lambda pattern: "-".join(pattern))
@pytest.mark.parametrize("percent", [80, 100])
@pytest.mark.parametrize("enabled", [False, True])
def test_verified_reply_gaps_count_one_icp(pattern, percent, enabled):
    examples = ten_selected_examples()
    _apply_pattern(examples["run-9"], pattern)
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=enabled,
        minimum_coverage_percent=percent,
    )
    complete = all(returned == "captured" for returned in pattern)
    assert result["eligible"] is (not enabled or complete or percent == 80), result
    if enabled and percent == 80:
        assert result["counts"]["complete_runs"] == 9 + int(complete)
        assert result["counts"]["incomplete_runs"] == int(not complete)
        assert result["counts"]["required_complete_runs"] == 8


@pytest.mark.parametrize("pattern", PATTERNS[1:])
def test_verified_gap_icp_stays_out_of_strict_export(pattern):
    examples = ten_selected_examples()
    _apply_pattern(examples["run-9"], pattern)
    strict, _ = training_thinking.build(
        list(examples.values()), training_steps.Rules(production_rounds_only=False),
    )
    strict_runs = {ref["run_id"] for split in ("train", "eval")
                   for record in strict[split] for ref in record["source_refs"]}
    expected = set(examples) if "captured" in pattern else set(examples) - {"run-9"}
    assert strict_runs == expected


@pytest.mark.parametrize("damage", ["route", "receipt", "response_hash", "request_hash", "provenance"])
@pytest.mark.parametrize("index", range(3))
def test_mixed_gaps_do_not_excuse_invalid_evidence(damage, index):
    examples = ten_selected_examples()
    example = _apply_pattern(examples["run-9"], ("truncated", "empty", "empty"))
    call = example["provider_calls"][index]
    wrong_hash = contracts.hash_bytes(b"different evidence")
    if damage == "route":
        call["reasoning_capture"]["provider_route"] = "wrong-route"
    elif damage == "receipt":
        call["response_receipts"][0]["response_sha256"] = wrong_hash
    elif damage == "response_hash":
        call["response_sha256"] = wrong_hash
    elif damage == "request_hash":
        call["request"]["outbound"]["body_sha256"] = wrong_hash
    else:
        call["response_provenance"]["agent_receipt"] = "not_reported"
    seal(example)
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80,
    )
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == 9
    assert result["counts"]["rejected_runs"][0]["run_id"] == "run-9"


def test_mixed_gaps_still_require_source_identity():
    examples = ten_selected_examples()
    example = _apply_pattern(examples["run-9"], ("truncated", "empty", "empty"))
    example["source"]["sha256"] = contracts.hash_bytes(b"different source")
    seal(example)
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80,
    )
    assert result["eligible"] is True
    assert result["counts"]["complete_runs"] == 9
    assert result["counts"]["rejected_runs"] == [{
        "run_id": "run-9", "reasons": ["selected_run_identity_unverified"]}]


@pytest.mark.parametrize("incomplete", [1, 2, 3])
def test_mixed_gaps_still_require_eighty_percent_complete(incomplete):
    examples = ten_selected_examples()
    for run_id in list(examples)[-incomplete:]:
        _apply_pattern(examples[run_id], ("truncated", "empty", "empty"))
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True, minimum_coverage_percent=80,
    )
    assert result["eligible"] is (incomplete <= 2), result
    assert result["reasons"] == ([] if incomplete <= 2 else ["training_coverage_incomplete"])
    assert result["counts"]["complete_runs"] == 10 - incomplete


def _ten_icp_publication(monkeypatch):
    from tests.lab_arena.training_eligibility_publication_test import (
        BASELINE_ID, _export, _publication_harness,
    )
    harness = _publication_harness(monkeypatch)
    obj, row, submissions, executions, examples = harness[:5]
    row["configuration_doc"].update(
        stage_1_icp_count=5, stage_2_icp_count=5,
        training_minimum_icp_coverage_percent=80,
    )
    judgments = obj._store.list_runs(row["round_id"], kind="score")
    for submission_id, prefix in ((BASELINE_ID, "exec"), ("miner", "miner")):
        original = next(run for run in executions if run["submission_id"] == submission_id)
        judge = next(run for run in judgments if run["submission_id"] == submission_id)
        for position in range(2, 10):
            run = {**original, "run_id": f"{prefix}-{position}", "icp_position": position,
                   "output_ref": f"{prefix}-output-{position}"}
            executions.append(run)
            judgments.append({**judge, "run_id": f"{prefix}-judge-{position}",
                              "icp_position": position, "scored_run_id": run["run_id"]})
            examples[run["run_id"]] = _export(
                run["run_id"], submission_id, baseline=submission_id == BASELINE_ID,
            )
    submissions["peer"] = {**deepcopy(submissions["miner"]), "submission_id": "peer",
                           "miner_hotkey": "peer-hotkey"}
    row["participants"].append({"submission_id": "peer", "miner_hotkey": "peer-hotkey",
                                "is_king": False})
    for original in list(executions):
        if original["submission_id"] != "miner":
            continue
        position = original["icp_position"]
        run = {**original, "submission_id": "peer", "miner_hotkey": "peer-hotkey",
               "run_id": f"peer-{position}", "output_ref": f"peer-output-{position}",
               "per_icp_score": 2.0}
        executions.append(run)
        judgments.append({**run, "run_id": f"peer-judge-{position}", "kind": "score",
                          "scored_run_id": run["run_id"]})
        examples[run["run_id"]] = _export(run["run_id"], "peer")
    _apply_pattern(examples["miner-9"], ("truncated", "empty", "empty"))
    return harness


@pytest.mark.parametrize("policy", ["eighty", "historical_hundred", "off"])
def test_mixed_gap_publication_keeps_scores_and_peer_eligibility(monkeypatch, policy):
    from tests.lab_arena.training_eligibility_publication_test import _published, _ranking
    harness = _ten_icp_publication(monkeypatch)
    configuration = harness[1]["configuration_doc"]
    if policy == "historical_hundred":
        configuration.pop("training_minimum_icp_coverage_percent")
    elif policy == "off":
        configuration.pop("reasoning_capture_policy")
    publication = _published(harness)
    miner, peer = (_ranking(publication, submission_id) for submission_id in ("miner", "peer"))
    assert miner["final_score"] == 3.0
    assert peer["final_score"] == 2.0
    assert peer["eligible"] is True
    assert miner["eligible"] is (policy != "historical_hundred")
    expected_winner = "peer" if policy == "historical_hundred" else "miner"
    assert publication["king_decision"]["winner_submission_id"] == expected_winner
    if policy == "eighty":
        training = miner["training_data_eligibility"]
        assert training["source_review"]["eligible"] is True
        assert training["capture"]["counts"]["complete_runs"] == 9
    elif policy == "off":
        assert "training_data_eligibility" not in miner
