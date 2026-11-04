"""Exact model pairs do not manufacture missing full-runtime inputs."""

from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import captured_example, refresh_response_links, submission
from tests.lab_arena.training_steps_test import seal


RULES = training_steps.Rules(production_rounds_only=False)


def example_with_missing_input(missing="execute", *, hosted=True):
    example = captured_example()
    example["configuration"]["training_minimum_icp_coverage_percent"] = 80
    if hosted:
        example["configuration"]["training_maximum_missing_reasoning_percent"] = 1
    identities = {"execute": example["run_id"], "score": example["score_run_id"]}
    for kind, identity in identities.items():
        snapshot = example["input_snapshots"][identity]
        snapshot.update(schema_version="leadpoet.lab_arena.training_input.v1",
                        run_id=identity, kind=kind, lease_generation=1,
                        claim_request_hash=contracts.hash_bytes(identity.encode()))
        next(row for row in example["attempts"] if row["run_id"] == identity)["lease_generation"] = 1
    missing_ids = [identities[kind] for kind in identities if missing in (kind, "both")]
    for identity in missing_ids:
        del example["input_snapshots"][identity]
    if missing_ids:
        proof = example["completeness"]
        proof["complete"] = False
        proof["dimensions"]["exact_inputs"] = False
        proof["reasons_by_dimension"]["exact_inputs"] = ["runtime_input_snapshot_missing"]
        proof["incomplete_reasons"] = sorted(set(proof["incomplete_reasons"]) | {"runtime_input_snapshot_missing"})
    return seal(example), missing_ids


def built(example):
    splits, manifest = training_thinking.build([seal(example)], RULES)
    return splits["train"] + splits["eval"], manifest


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("missing", ["execute", "score", "both"])
def test_absent_runtime_inputs_keep_exact_thinking_pairs_but_not_full_icp(missing, hosted):
    example, missing_ids = example_with_missing_input(missing, hosted=hosted)
    original = deepcopy(example)
    records, _ = built(example)
    assert len(records) == 2
    assert example == original
    assert training_thinking.selected_research_evidence(example) is None
    assessment = training_eligibility.assess_capture(
        submission(), {"run": example}, ["run"], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1 if hosted else 0)
    assert not assessment["eligible"] and assessment["counts"]["complete_runs"] == 0
    baseline, _ = built(example_with_missing_input("none", hosted=hosted)[0])
    for record, expected in zip(records, baseline):
        assert all(record[key] == expected[key] for key in ("input", "target", "generated_thinking", "prior_observations", "candidate_id"))
        assert record["quality"] == {"status": "unreviewed", "correct_action": "not_established"}
        ref = record["source_refs"][0]
        assert ref["source_quality"] == example["completeness"]
        assert ref["outcome"] == expected["source_refs"][0]["outcome"]
        proof = ref["selected_research_evidence"]
        assert proof["complete"] is False
        assert proof["scope"] == "individual_model_request_result"
        assert proof["missing_runtime_input_run_ids"] == sorted(missing_ids)
        assert "runtime_input_snapshot_missing" not in proof["ancillary_judge_gaps"]


@pytest.mark.parametrize("damage", ["empty", "null", "hash", "privacy", "privacy_list", "run_id", "kind", "lease", "claim", "schema", "score_target", "companies"])
def test_corrupt_present_snapshot_is_not_treated_as_absent(damage):
    example, _ = example_with_missing_input("execute")
    identity = example["score_run_id"]
    snapshot = example["input_snapshots"][identity]
    if damage == "empty": example["input_snapshots"][identity] = {}
    elif damage == "null": example["input_snapshots"][identity] = None
    elif damage == "hash": snapshot["input_sha256"] = contracts.hash_bytes(b"bad")
    elif damage == "privacy": snapshot["privacy"] = {"redacted": True, "redactions": ["credential"]}
    elif damage == "privacy_list": snapshot["privacy"]["redactions"] = ["credential"]
    elif damage == "run_id": snapshot["run_id"] = "other-run"
    elif damage == "kind": snapshot["kind"] = "execute"
    elif damage == "lease": snapshot["lease_generation"] = 2
    elif damage == "claim": snapshot["claim_request_hash"] = "invalid"
    elif damage == "schema": snapshot["schema_version"] = "unknown"
    elif damage == "score_target": snapshot["input"]["scored_run_id"] = "other-run"
    else: snapshot["input"]["companies"] = [{"different": "output"}]
    if damage in {"score_target", "companies"}:
        snapshot["input_sha256"] = contracts.document_hash(snapshot["input"])
    assert built(example)[0] == []


@pytest.mark.parametrize("damage", ["output", "execution_identity", "score_identity", "source", "context", "contradiction", "invalid_policy"])
def test_missing_snapshot_does_not_waive_other_source_checks(damage):
    example, _ = example_with_missing_input()
    if damage == "output": example["output"]["changed"] = True
    elif damage == "execution_identity": example["attempts"][0]["run_id"] = "other"
    elif damage == "score_identity": example["score_run_id"] = "other"
    elif damage in {"source", "context"}:
        dimension = "source_identity" if damage == "source" else "context"
        example["completeness"]["dimensions"][dimension] = False
    elif damage == "contradiction": example["completeness"]["incomplete_reasons"].append("runtime_input_context_mismatch")
    else: example["configuration"]["training_maximum_missing_reasoning_percent"] = False
    assert built(example)[0] == []


@pytest.mark.parametrize("damage", ["request", "response", "credential", "truncated", "missing", "summary"])
def test_missing_snapshot_keeps_per_call_integrity_and_thinking_distinctions(damage):
    example, _ = example_with_missing_input()
    call = example["provider_calls"][1]
    if damage == "request": call["request"]["outbound"]["body_sha256"] = contracts.hash_bytes(b"bad")
    elif damage == "response": call["response"]["changed"] = True
    elif damage == "credential": call["response_receipts"][0]["privacy_redactions"] = ["credential"]
    else:
        response = call["response"]
        message = response["choices"][0]["message"]
        if damage == "truncated": response["choices"][0]["finish_reason"] = "length"
        else:
            message.pop("reasoning", None)
            message.pop("reasoning_content", None)
            message["reasoning_details"] = ([{"type": "reasoning.summary", "summary": "A summary only."}]
                                             if damage == "summary" else [])
        refresh_response_links(example, 1)
    records, _ = built(example)
    assert len(records) == 1
    assert records[0]["source_refs"][0]["call_identity"] != call["call_identity"]


@pytest.mark.parametrize("reason", ["runtime_input_snapshot_invalid", "runtime_input_privacy_nonexact",
                                    "runtime_input_context_mismatch", "judge_input_context_mismatch"])
@pytest.mark.parametrize("dimension", ["exact_inputs", "research_process"])
def test_invalid_removed_snapshot_stays_invalid_after_missing_flag(reason, dimension):
    example, _ = example_with_missing_input()
    proof = example["completeness"]
    proof["incomplete_reasons"] = sorted(proof["incomplete_reasons"] + [reason])
    proof["reasons_by_dimension"].setdefault(dimension, []).append(reason)
    proof["dimensions"][dimension] = False
    assert built(example)[0] == []


def test_prior_attempt_input_gap_is_diagnostic_not_selected_authority():
    example, _ = example_with_missing_input()
    proof = example["completeness"]
    reason = "prior_attempt_runtime_input_snapshot_invalid"
    proof["incomplete_reasons"].append(reason)
    proof["reasons_by_dimension"]["research_process"] = [reason]
    proof["dimensions"]["research_process"] = False
    assert len(built(example)[0]) == 2
    # Attributing an unprefixed selected error to another run cannot waive it.
    proof["incomplete_reasons"].append("runtime_input_snapshot_invalid")
    proof["reasons_by_dimension"]["research_process"].append("runtime_input_snapshot_invalid")
    example["related_run_gap_attribution"] = {"prior": {"research_process_gaps": ["runtime_input_snapshot_invalid"]}}
    assert built(example)[0] == []


@pytest.mark.parametrize("damage", ["missing_reason", "unattributed_reason", "dimension", "missing_dimension"])
def test_inconsistent_missing_diagnostics_do_not_grant_exception(damage):
    example, _ = example_with_missing_input()
    proof = example["completeness"]
    if damage == "missing_reason": proof["incomplete_reasons"] = []
    elif damage == "unattributed_reason": proof["incomplete_reasons"].append("unattributed")
    elif damage == "dimension": proof["dimensions"]["exact_inputs"] = True
    else: del proof["reasons_by_dimension"]["exact_inputs"]
    assert built(example)[0] == []


@pytest.mark.parametrize("damage", ["empty", "privacy", "privacy_type", "hash", "claim_binding", "lease_type", "icp", "date"])
def test_present_execution_snapshot_is_also_verified(damage):
    example, _ = example_with_missing_input("score")
    identity = example["run_id"]
    snapshot = example["input_snapshots"][identity]
    if damage == "empty": example["input_snapshots"][identity] = {}
    elif damage == "privacy": snapshot["privacy"] = {"redacted": True, "redactions": ["credential"]}
    elif damage == "privacy_type": snapshot["privacy"]["redacted"] = 0
    elif damage == "hash": snapshot["input_sha256"] = contracts.hash_bytes(b"bad")
    elif damage == "lease_type":
        next(row for row in example["attempts"] if row["run_id"] == identity)["lease_generation"] = True
    elif damage in {"icp", "date"}:
        snapshot["input"]["icp" if damage == "icp" else "evaluation_date"] = {"industry": "wrong"} if damage == "icp" else "1900-01-01"
        snapshot["input_sha256"] = contracts.document_hash(snapshot["input"])
    else:
        example["worker_observation_artifacts"][identity]["claim_request_hash"] = snapshot["claim_request_hash"]
        snapshot["claim_request_hash"] = contracts.hash_bytes(b"different claim")
    assert built(example)[0] == []


@pytest.mark.parametrize("damage", ["missing_judgment", "failed_judge", "missing_selected_judge"])
def test_absent_snapshot_exception_does_not_admit_unlabelled_sources(damage):
    example, _ = example_with_missing_input()
    if damage == "missing_judgment": example.pop("judgment")
    elif damage == "failed_judge":
        next(row for row in example["attempts"] if row["kind"] == "score")["status"] = "failed"
    else: example.pop("score_run_id")
    assert built(example)[0] == []


def test_complete_input_path_and_global_dedup_stay_unchanged():
    complete, _ = example_with_missing_input("none")
    before = deepcopy(complete)
    records, _ = built(complete)
    assert len(records) == 2 and complete == before
    assert all(row["source_refs"][0]["selected_research_evidence"]["complete"] is True for row in records)
    missing, _ = example_with_missing_input()
    once, manifest = training_thinking.build([missing], RULES)
    twice, duplicate_manifest = training_thinking.build([missing, deepcopy(missing)], RULES)
    assert once == twice
    assert manifest["split_source_hashes"] == duplicate_manifest["split_source_hashes"]
