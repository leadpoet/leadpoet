"""Rewritten claims retain both original judgment and current acceptance proof."""
from copy import deepcopy
import json
import pytest

from lab_arena import contracts, evidence_judgments as ej, signal_judgments
from lab_arena.training_export import ExportError, _signal_authority_evidence
from lab_arena.training_input import read_input
from tests.lab_arena.training_export_signal_test import _signal_fixture
from tests.lab_arena.training_export_test import save_input


def fixture(*, fit=False, negative=False, rewrite=True):
    values = _signal_fixture()
    execute, judge, source = values["runs"]
    config = values["round_row"]["configuration_doc"]
    policy = config["scorer_policy"]
    policy["evidence_judgment_policy"] = ej.POLICY
    policy["env_bindings"]["ARENA_EVIDENCE_JUDGMENT_POLICY"] = ej.POLICY
    saved = read_input(values["objects"], judge["run_id"], "score", judge["lease_generation"], judge["claim_request_hash"])["input"]
    fields = ("schema_version", "scored_run_id", "icp", "companies", "scorer_policy", "evaluation_date")
    current = {key: deepcopy(saved[key]) for key in fields}
    current["scorer_policy"] = deepcopy(policy)
    original = deepcopy(current)
    original["scored_run_id"] = source["scored_run_id"]
    if rewrite:
        current["companies"][0]["intent_signals"][0]["description"] = "Acme introduced a new product."
    raw = deepcopy(values["signal_judgment_rows"][0]["judgment_doc"])
    trace = raw["judge_verdict"]["verification_trace"]
    trace.update(stage1={"status": "skipped_source_grounded"}, stage3={"status": "supported", "decision": "approve"})
    trace["intent_verdict"]["signal_evaluations"][0]["signal_id"] = "signal-1"
    snapshot = {"schema_version": "leadpoet.intent_evidence_input.v1",
                "evaluation_date": current["evaluation_date"],
                "model": policy["judge_models"]["intent_fetched_evidence"],
                "row": {"signal_date": original["companies"][0]["intent_signals"][0]["date"],
                        "claimed_source_urls": raw["evidence_urls"]},
                "contents": {"results": deepcopy(trace["verified_source_context"]), "statuses": []}}
    if fit:
        from tests.lab_arena.evidence_judgments_v2_test import fit_snapshot
        envelope = fit_snapshot(negative=negative)
        snapshot = envelope["evidence_manifest"]
    def scope(doc):
        return ej.build_scope(scoring_input=doc, company_index=0, signal_index=-1 if fit else 0,
            verified_company_identity={"decision": "match", "observed_name": "Acme", "observed_domain": "acme.example"},
            verified_funding_stage_context=[], round_id=judge["round_id"], network_name="finney", netuid=71,
            scorer_image_digest=config["scorer_image_digest"], scorer_image_reference=config["scorer_image_reference"],
            runtime_image_digest=config["scorer_image_digest"], runtime_image_reference=config["scorer_image_reference"],
            evidence_input=snapshot, evidence_hash=contracts.document_hash(snapshot))
    original_scope, current_scope = scope(original), scope(current)
    if not fit:
        envelope = {"schema_version": ej.SIGNAL_SCHEMA, "raw_judgment": raw,
                    "verifier_result": {"decision": "approve", "client_ready": True,
                        "stage1": trace["stage1"], "stage3": trace["stage3"], "verdict": trace["intent_verdict"]},
                    "evidence_input": snapshot, "claim": original_scope["claim"]}
    ej.validate_judgment(envelope, original_scope)
    reference = signal_judgments.reference(original_scope, 0)
    proof = None if original_scope["claim"] == current_scope["claim"] else {
        "schema_version": ej.EQUIVALENCE_SCHEMA, "source_claim_hash": contracts.document_hash(original_scope["claim"]),
        "current_claim_hash": contracts.document_hash(current_scope["claim"]),
        "model": policy["judge_models"]["intent_fetched_evidence"], "decision": "equivalent", "material_differences": []}
    row = {**values["signal_judgment_rows"][0], **{k: reference[k] for k in ("cache_key", "authority_slot", "signal_input_hash")},
           "scope_doc": original_scope, "judgment_doc": envelope, "judgment_hash": contracts.document_hash(envelope)}
    use = {**values["signal_judgment_uses"][0], **{k: row[k] for k in ("cache_key", "authority_slot", "signal_input_hash", "judgment_hash")},
           "signal_index": -1 if fit else 0,
           "acceptance_doc": {"schema_version": "leadpoet.lab_arena.evidence_use.v2", "current_scope": current_scope,
                              "candidate_reference": reference, "equivalence_receipt": proof}}
    breakdown = ({"verifier_gate_receipts": [envelope["company_fit_receipt"]]} if fit else {
        "intent_signals_detail": [{**raw, "signal_judgment_reference": reference, "counted_in_aggregate": True}]})
    inputs = {judge["run_id"]: current, source["run_id"]: original}
    return dict(judge=judge, breakdowns=[breakdown], scored_company_indexes=[0], scoring_input=current,
                uses=[use], authority_rows=[row], company_rows=[], runs={r["run_id"]: r for r in values["runs"]},
                round_row=values["round_row"], source_input_reader=lambda run: inputs[run["run_id"]]), inputs


@pytest.mark.parametrize("fit,negative", [(False, False), (True, False), (True, True)])
def test_original_native_judgment_and_source_survive_cross_submission_rewrite(fit, negative):
    args, _ = fixture(fit=fit, negative=negative)
    result = _signal_authority_evidence(**args)
    assert result[0]["authority"] == args["authority_rows"][0]
    assert result[0]["use"]["acceptance_doc"] == args["uses"][0]["acceptance_doc"]


@pytest.mark.parametrize("fault", ["missing_proof", "wrong_proof", "source_claim", "current_claim", "source_body", "source_runner", "source_input_missing"])
def test_changed_or_missing_frozen_provenance_fails_closed(fault):
    args, inputs = fixture()
    if fault == "missing_proof": args["uses"][0]["acceptance_doc"] = None
    elif fault == "wrong_proof": args["uses"][0]["acceptance_doc"]["equivalence_receipt"]["decision"] = "different"
    elif fault == "source_claim": inputs["judge:source"]["companies"][0]["intent_signals"][0]["description"] = "Acme acquired a company."
    elif fault == "current_claim": args["scoring_input"]["companies"][0]["intent_signals"][0]["description"] = "Acme acquired a company."
    elif fault == "source_body": args["authority_rows"][0]["scope_doc"]["evidence_input"]["contents"]["results"][0]["text"] = "Changed evidence"
    elif fault == "source_runner": args["authority_rows"][0]["source_runner_hotkey"] = "other-validator"
    else: args["source_input_reader"] = None
    with pytest.raises(ExportError): _signal_authority_evidence(**args)


def test_exact_variant_can_use_existing_completion_without_paraphrase_receipt():
    args, _ = fixture(rewrite=False)
    args["uses"][0]["acceptance_doc"] = None
    assert len(_signal_authority_evidence(**args)) == 1
