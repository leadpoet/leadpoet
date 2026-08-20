from __future__ import annotations

from copy import deepcopy

import pytest

from gateway.research_lab.routing_experiment_attestation import (
    RoutingExperimentAttestationError,
    build_routing_experiment_attestation_input_v2,
    execute_routing_experiment_attestation_v2,
    validate_routing_experiment_attestation_input_v2,
)
from gateway.research_lab.routing_experiment_runtime import (
    AttestedScoringV2RoutingEvaluationAuthority,
    RoutingExperimentRuntimeError,
)
from research_lab.canonical import sha256_json
from tests.routing_experiment_authority_fixture import authority_fixture, _signed_receipt


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _payload() -> dict[str, object]:
    fixture = authority_fixture()
    return build_routing_experiment_attestation_input_v2(
        spec_doc=fixture["spec"].to_dict(),
        evaluation_doc=fixture["evaluation"].to_dict(),
        gold_label_authority=fixture["labels"],
        artifact_lineage=fixture["lineage"],
        execution_envelope=fixture["execution_envelope"],
        decision_receipts=fixture["decisions"],
        provider_attempts=fixture["attempts"],
        budget_events=fixture["budgets"],
    )


def _valid_executor(request):
    assert request["operation"] == "attest_routing_experiment_v2"
    assert request["purpose"] == "research_lab.routing_experiment.v2"
    result = execute_routing_experiment_attestation_v2(request["payload"])
    receipt = _signed_receipt(
        purpose=request["purpose"],
        input_root=result["input_root"],
        output_root=result["output_root"],
        index=99,
    )
    return {
        "status": "succeeded",
        "operation": request["operation"],
        "purpose": request["purpose"],
        "result": result,
        "receipt": receipt,
    }


def test_attestation_payload_requires_exact_nonempty_canonical_authority_rows():
    payload = _payload()
    validate_routing_experiment_attestation_input_v2(payload)
    assert execute_routing_experiment_attestation_v2(payload)["reconciled"] is True

    duplicate = dict(payload)
    duplicate["decision_receipts"] = [payload["decision_receipts"][0]] * 2
    with pytest.raises(RoutingExperimentAttestationError, match="not canonical"):
        validate_routing_experiment_attestation_input_v2(duplicate)

    missing = dict(payload)
    missing["provider_attempts"] = []
    with pytest.raises(RoutingExperimentAttestationError, match="provider attempts is invalid"):
        validate_routing_experiment_attestation_input_v2(missing)


@pytest.mark.parametrize(
    ("field", "mutation", "message"),
    (
        ("evaluation_doc", lambda doc: {**doc, "selected_variant_id": "candidate"}, "evaluation differs"),
        (
            "evaluation_doc",
            lambda doc: {
                **doc,
                "variants": [
                    {
                        **item,
                        "holdout": {**item["holdout"], "precision": 0.12345678},
                    }
                    for item in doc["variants"]
                ],
            },
            "evaluation differs",
        ),
        (
            "provider_attempts",
            lambda rows: [
                {**rows[0], "billing_state": "uncertain", "authoritative_billed_credit_microunits": None},
                *rows[1:],
            ],
            "billing is unresolved",
        ),
        (
            "budget_events",
            lambda rows: [item for item in rows if item["event_type"] != "settle"],
            "budget chain",
        ),
    ),
)
def test_attestation_rejects_forged_metrics_selection_billing_and_missing_settlement(
    field, mutation, message
):
    payload = _payload()
    payload[field] = mutation(payload[field])
    with pytest.raises(RoutingExperimentAttestationError, match=message):
        validate_routing_experiment_attestation_input_v2(payload)


@pytest.mark.parametrize("mutation", ["missing_terminal", "terminal_substitution", "admission_substitution"])
def test_attestation_rejects_provider_terminal_chain_substitution(mutation):
    payload = _payload()
    attempts = [deepcopy(item) for item in payload["provider_attempts"]]
    document = attempts[0]["attempt_doc"]
    if mutation == "missing_terminal":
        del document["terminal_proof"]
    elif mutation == "terminal_substitution":
        document["terminal_proof"] = deepcopy(attempts[1]["attempt_doc"]["terminal_proof"])
    else:
        document["admission_bundle"] = deepcopy(document["admission_bundle"])
        document["admission_bundle"]["job_id"] = "substituted-job"
    payload["provider_attempts"] = attempts
    with pytest.raises(RoutingExperimentAttestationError):
        validate_routing_experiment_attestation_input_v2(payload)


def test_code_side_attested_operation_rejects_fabricated_result_and_fails_closed_without_tee():
    payload = _payload()
    with pytest.raises(RoutingExperimentRuntimeError, match="not released"):
        AttestedScoringV2RoutingEvaluationAuthority().attest(payload)

    authority = AttestedScoringV2RoutingEvaluationAuthority(executor=_valid_executor)
    value = authority.attest(payload)
    assert value["result"] == execute_routing_experiment_attestation_v2(payload)

    def forged_executor(request):
        result = dict(execute_routing_experiment_attestation_v2(request["payload"]))
        result["provider_attempts_root"] = _hash("7")
        value = _valid_executor(request)
        value["result"] = result
        return value

    with pytest.raises(RoutingExperimentRuntimeError, match="result is not exact"):
        AttestedScoringV2RoutingEvaluationAuthority(executor=forged_executor).attest(payload)

    def missing_receipt_hash_executor(request):
        value = _valid_executor(request)
        value["receipt"]["receipt_hash"] = ""
        return value

    with pytest.raises(RoutingExperimentRuntimeError, match="receipt is invalid"):
        AttestedScoringV2RoutingEvaluationAuthority(
            executor=missing_receipt_hash_executor
        ).attest(payload)
