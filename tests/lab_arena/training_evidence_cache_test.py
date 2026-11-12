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


def test_same_run_source_search_reaches_later_company():
    args, _ = fixture(rewrite=False)
    judge = args["judge"]
    judge["runner_hotkey"] = "validator-current"
    unrelated = deepcopy(args["scoring_input"]["companies"][0])
    unrelated["intent_signals"][0]["url"] = "https://acme.example/other-launch"
    args["scoring_input"]["companies"].insert(0, unrelated)
    args["scored_company_indexes"] = [1]
    use = args["uses"][0]
    use.update(company_index=1, source_score_run_id=judge["run_id"])
    row = args["authority_rows"][0]
    for key in ("scored_run_id", "runner_hotkey", "claim_request_hash", "lease_generation"):
        row["source_" + key] = judge[key]
    row["source_score_run_id"] = judge["run_id"]
    row["runner_authority_exclusions"] = [judge["runner_hotkey"]]

    result = _signal_authority_evidence(**args)

    assert len(result) == 1
    assert result[0]["company_index"] == 1
    assert result[0]["authority"]["source_score_run_id"] == judge["run_id"]


def test_source_search_with_no_exact_evidence_match_still_rejects():
    args, inputs = fixture()
    source = inputs["judge:source"]
    source["companies"][0]["intent_signals"][0]["date"] = "2000-01-01"

    with pytest.raises(ExportError) as error:
        _signal_authority_evidence(**args)

    assert str(error.value.__cause__) == "evidence_original_input_mismatch"


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


def transitive_runtime_fixture(*, fit=False, source_index=0):
    args, inputs = fixture(fit=fit)
    from lab_arena import company_judgments, reviewer_runtime
    judge = args["judge"]
    config = args["round_row"]["configuration_doc"]
    accepted = deepcopy(judge)
    accepted["run_id"] = "judge:company-source"
    args["runs"][accepted["run_id"]] = accepted
    inputs[accepted["run_id"]] = deepcopy(args["scoring_input"])
    ref = company_judgments.build_company_scopes(
        scoring_input=args["scoring_input"], round_id=judge["round_id"],
        network_name="finney", netuid=71,
        scorer_image_digest=config["scorer_image_digest"],
        scorer_image_reference=config["scorer_image_reference"],
        integrity_policy="arena_integrity_v1", company_judgment_policy=company_judgments.POLICY,
    )[0]
    judge["company_judgment_refs"] = [ref]
    accepted["company_judgment_refs"] = [{**ref, "company_index": source_index}]
    if source_index:
        inputs[accepted["run_id"]]["companies"].insert(0, deepcopy(args["scoring_input"]["companies"][0]))
    judge["claim_response"] = {"company_judgment_cache": {"hits": [{
        **{key: ref[key] for key in ("company_index", "cache_key", "company_input_hash")},
        "authority_slot": 0,
    }]}}
    args["company_rows"] = [{"cache_key": ref["cache_key"], "authority_slot": 0,
                             "source_score_run_id": accepted["run_id"]}]
    args["uses"][0]["run_id"] = accepted["run_id"]
    args["uses"][0]["company_index"] = source_index
    judge["reviewer_runtime_doc"] = {
        "schema_version": reviewer_runtime.SCHEMA_VERSION,
        "run_id": judge["run_id"], "lease_generation": judge["lease_generation"],
        "grading_image_digest": config["scorer_image_digest"],
        "grading_image_reference": config["scorer_image_reference"],
        "runtime_image_digest": "sha256:" + "d" * 64,
        "runtime_image_reference": "registry.example/runtime@sha256:" + "d" * 64,
    }
    return args


@pytest.mark.parametrize("fit", [False, True])
@pytest.mark.parametrize("source_index", [0, 1])
def test_transitive_company_use_keeps_original_runtime_after_reviewer_repair(fit, source_index):
    args = transitive_runtime_fixture(fit=fit, source_index=source_index)
    result = _signal_authority_evidence(**args)
    assert result[0]["authority"] == args["authority_rows"][0]
    assert result[0]["use"] == args["uses"][0]
    assert result[0]["authority"]["scope_doc"]["runtime_image_digest"] != args["judge"]["reviewer_runtime_doc"]["runtime_image_digest"]


@pytest.mark.parametrize("fault", ["evidence", "requirements", "receipt", "source_identity", "direct_use"])
def test_transitive_runtime_repair_keeps_content_and_authority_checks(fault):
    args = transitive_runtime_fixture()
    if fault == "evidence":
        args["scoring_input"]["companies"][0]["intent_signals"][0]["snippet"] = "Different evidence"
    elif fault == "requirements":
        args["scoring_input"]["icp"]["industry"] = "Different buyer requirements"
    elif fault == "receipt":
        args["uses"][0]["acceptance_doc"]["equivalence_receipt"]["decision"] = "different"
    elif fault == "source_identity":
        args["authority_rows"][0]["source_runner_hotkey"] = "other-validator"
    else:
        args["uses"][0]["run_id"] = args["judge"]["run_id"]
    with pytest.raises(ExportError):
        _signal_authority_evidence(**args)


@pytest.mark.parametrize("earlier_candidate", ["company", "signal"])
@pytest.mark.parametrize("source_status", ["accepted", "failed"])
def test_source_search_reaches_exact_signal_after_unrelated_evidence(
    earlier_candidate, source_status,
):
    args, inputs = fixture()
    source = inputs["judge:source"]
    args["runs"]["judge:source"]["status"] = source_status
    unrelated = deepcopy(source["companies"][0])
    unrelated["intent_signals"][0]["date"] = "2000-01-01"
    unrelated["intent_signals"][0]["url"] = "https://acme.example/other-launch"
    if earlier_candidate == "company":
        source["companies"].insert(0, unrelated)
    else:
        source["companies"][0]["intent_signals"].insert(
            0, unrelated["intent_signals"][0],
        )

    result = _signal_authority_evidence(**args)

    assert len(result) == 1
    assert result[0]["authority"] == args["authority_rows"][0]
    assert result[0]["use"]["acceptance_doc"] == args["uses"][0]["acceptance_doc"]
