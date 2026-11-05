"""Prospective evidence identities on the existing private signal authority."""
from __future__ import annotations

from typing import Any, Mapping
from lab_arena import company_judgments, contracts, signal_judgments

POLICY = "evidence_judgment_v2"
FIT_KIND = "company_fit_evidence_v2"
SIGNAL_KIND = "signal_evidence_v2"
FIT_SCHEMA = "leadpoet.lab_arena.company_fit_evidence.v2"
SIGNAL_SCHEMA = "leadpoet.lab_arena.signal_evidence.v2"
EQUIVALENCE_SCHEMA = "leadpoet.lab_arena.claim_equivalence.v1"
Error = signal_judgments.SignalJudgmentError
_copy = signal_judgments._copy
_hash = contracts.document_hash


def enabled(policy: Mapping[str, Any]) -> bool:
    return policy.get("evidence_judgment_policy") == POLICY


def normalize_claim(signal: Mapping[str, Any]) -> dict[str, str]:
    """Preserve semantics; only trim the edges of the two claim fields."""
    if any(not isinstance(signal.get(key, ""), str) for key in ("description", "snippet")):
        raise Error("signal claim is invalid")
    return {"description": signal.get("description", "").strip(), "date": signal.get("date"),
            "evidence_bundle": [{"url": signal.get("url"),
                "description": signal.get("description", "").strip(),
                "snippet": signal.get("snippet", "").strip(), "date": signal.get("date")}]}


def fit_projection(effective: Mapping[str, Any]) -> dict[str, Any]:
    copied = _copy(effective)
    company = copied["company"]
    if "intent_details" in company:
        # Native fit uses presence/non-nullness to choose its source policy.
        company["intent_details"] = None if company["intent_details"] is None else ""
    for signal in company.get("intent_signals", []):
        signal.pop("description", None)
        signal.pop("snippet", None)
    return copied


def build_scope(*, scoring_input: Mapping[str, Any], company_index: int,
                signal_index: int, round_id: str, network_name: str, netuid: int,
                scorer_image_digest: str, scorer_image_reference: str,
                runtime_image_digest: str, runtime_image_reference: str,
                evidence_input: Mapping[str, Any] | None = None, evidence_hash: str | None = None,
                verified_company_identity: Mapping[str, Any] | None = None,
                verified_funding_stage_context: list[Any] | None = None,
                verified_source_pages: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = scoring_input.get("scorer_policy") or {}
    if not enabled(policy):
        raise Error("evidence policy is not frozen")
    rows = company_judgments.effective_company_inputs(scoring_input)
    if type(company_index) is not int or not 0 <= company_index < len(rows):
        raise Error("company index is outside scoring input")
    effective = rows[company_index]
    fit = signal_index == -1
    if fit:
        native = fit_projection(effective)
        claim = None
        expectations = {}
    else:
        # Keep the established native signal target and strict identity rules.
        legacy = signal_judgments.build_scope(
            scoring_input=scoring_input, company_index=company_index, signal_index=signal_index,
            round_id=round_id, network_name=network_name, netuid=netuid,
            scorer_image_digest=scorer_image_digest, scorer_image_reference=scorer_image_reference,
            verified_company_identity=verified_company_identity,
            verified_funding_stage_context=verified_funding_stage_context,
            verified_source_pages=verified_source_pages)
        signal = _copy(effective["company"]["intent_signals"][signal_index])
        claim = normalize_claim(signal)
        signal.pop("description", None); signal.pop("snippet", None)
        native = {"buyer_requirements": effective["buyer_requirements"],
                  "company_identity": {key: effective["company"].get(key) for key in
                      ("company_name", "company_website", "company_linkedin")},
                  "signal": signal, "verified_company_identity": _copy(verified_company_identity),
                  "verified_funding_stage_context": _copy(verified_funding_stage_context)}
        expectations = {key: legacy[key] for key in
                        ("expected_urls", "expected_context_urls", "expected_index", "expected_evidence_type")}
    family = {"schema_version": "leadpoet.lab_arena.evidence_family.v2",
              "round_id": str(round_id), "network_name": str(network_name), "netuid": int(netuid),
              "evaluation_date": effective["evaluation_date"],
              "scorer_image_digest": scorer_image_digest, "scorer_image_reference": scorer_image_reference,
              "runtime_image_digest": runtime_image_digest, "runtime_image_reference": runtime_image_reference,
              "evidence_judgment_policy": POLICY, "scorer_policy": _copy(policy),
              "judgment_kind": FIT_KIND if fit else SIGNAL_KIND, "native_input": native}
    if not all(family.get(key) for key in ("round_id", "network_name", "evaluation_date",
            "scorer_image_digest", "scorer_image_reference", "runtime_image_digest", "runtime_image_reference")):
        raise Error("evidence family is incomplete")
    scope = {"schema_version": signal_judgments.SCOPE_VERSION,
             "signal_judgment_policy": signal_judgments.POLICY,
             **{key: family[key] for key in ("round_id", "network_name", "netuid", "evaluation_date",
                     "scorer_image_digest", "scorer_image_reference", "runtime_image_digest", "runtime_image_reference")},
             "evidence_judgment_policy": POLICY, "family_hash": _hash(family),
             "family_doc": family, "claim": claim, "claim_hash": _hash(claim), **expectations}
    scope["checkpoint_kind" if fit else "judgment_kind"] = FIT_KIND if fit else SIGNAL_KIND
    if evidence_input is None and fit and evidence_hash is None:
        return scope  # bounded prefetch only, never authority or a use
    if not isinstance(evidence_input, Mapping) or not evidence_input:
        raise Error("current evidence snapshot is required")
    snapshot = _copy(evidence_input)
    if not fit:
        models = policy.get("judge_models") or {}
        model = models.get("intent_fetched_evidence") or models.get("intent_three_stage_stage3")
        row = snapshot.get("row")
        if (snapshot.get("schema_version") != "leadpoet.intent_evidence_input.v1"
            or snapshot.get("model") != model or not model
            or snapshot.get("evaluation_date") != effective["evaluation_date"]
            or not isinstance(row, Mapping)
            or row.get("signal_date") != claim["date"]
            or row.get("claimed_source_urls") != expectations["expected_urls"]):
            raise Error("native evidence input differs from frozen signal")
    if evidence_hash != _hash(snapshot):
        raise Error("current evidence snapshot hash differs")
    scope.update(evidence_hash=evidence_hash, evidence_input=snapshot)
    scope["signal_input_hash"] = _hash({"family_hash": scope["family_hash"],
                                        "evidence_hash": evidence_hash, "claim": claim})
    scope["cache_key"] = _hash(scope)
    return scope


def validate_equivalence(receipt: Any, source_claim: Any, current_claim: Any,
                         policy: Mapping[str, Any]) -> dict[str, Any] | None:
    if source_claim == current_claim:
        return None
    models = policy.get("judge_models") or {}
    model = models.get("intent_fetched_evidence") or models.get("intent_three_stage_stage3")
    expected = {"schema_version": EQUIVALENCE_SCHEMA, "source_claim_hash": _hash(source_claim),
                "current_claim_hash": _hash(current_claim), "model": model,
                "decision": "equivalent", "material_differences": []}
    if not model or receipt != expected:
        raise Error("claim equivalence is not bound to current evidence and model")
    return _copy(expected)


def validate_fit(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "company_fit_receipt",
            "linkedin_profile_source", "matched_company_sources", "required_attribute_sources",
            "evidence_manifest", "source_snapshot_hash"}:
        raise Error("fit evidence fields changed")
    copied = _copy(value)
    if copied["schema_version"] != FIT_SCHEMA or not isinstance(copied["evidence_manifest"], Mapping) or not copied["evidence_manifest"]:
        raise Error("fit evidence snapshot is invalid")
    if copied["source_snapshot_hash"] != _hash(copied["evidence_manifest"]):
        raise Error("fit evidence snapshot hash differs")
    receipt = copied["company_fit_receipt"]
    from qualification.scoring.company_fit_decision import aggregate_company_fit_decisions, reconcile_company_fit_decisions
    if (not isinstance(receipt, Mapping) or receipt.get("gate") != "company_fit"
        or receipt.get("contract_id") != "company-fit-decision:v1"
        or receipt.get("contract_version") != "company-fit-decision:v1"
        or type(receipt.get("company_fit_stage_required")) is not bool):
        raise Error("fit evidence receipt is invalid")
    decision = receipt.get("decision")
    dimensions = receipt.get("company_fit_dimensions")
    if (decision not in {"match", "mismatch"} or receipt.get("company_fit_decision") != decision
        or not isinstance(dimensions, Mapping)
        or reconcile_company_fit_decisions([aggregate_company_fit_decisions(dimensions, stage_required=receipt["company_fit_stage_required"]), receipt.get("required_attribute_decision")]) != decision
        or receipt.get("failure_class") or receipt.get("failure_reason_code")):
        raise Error("fit evidence is not terminal")
    for field in ("linkedin_profile_source", "matched_company_sources", "required_attribute_sources"):
        if not isinstance(copied[field], Mapping):
            raise Error("fit evidence source context is invalid")
    if decision == "match":
        legacy = {key: value for key, value in copied.items() if key not in {"evidence_manifest", "source_snapshot_hash"}}
        legacy["schema_version"] = company_judgments.FIT_CHECKPOINT_SCHEMA_VERSION
        try:
            company_judgments.validate_verified_fit_checkpoint(legacy)
        except company_judgments.CompanyJudgmentError as exc:
            raise Error("positive fit evidence is incomplete") from exc
    else:
        required = {"identity", "employee_size", "industry", "geography"}
        if receipt["company_fit_stage_required"]:
            required.add("stage")
        evidence = receipt.get("dimension_evidence")
        if (not isinstance(evidence, Mapping) or not any(
                (dimensions.get(key) == "mismatch" or (key == "required_attribute" and receipt.get("required_attribute_decision") == "mismatch")) and isinstance(evidence.get(key), Mapping)
                and evidence[key].get("decision") == "mismatch" for key in required | {"required_attribute"})):
            raise Error("negative fit lacks native mismatch evidence")
    return copied


def validate_judgment(value: Any, scope: Mapping[str, Any]) -> dict[str, Any]:
    if scope.get("checkpoint_kind") == FIT_KIND:
        copied = validate_fit(value)
        if copied["evidence_manifest"] != scope["evidence_input"] or copied["source_snapshot_hash"] != scope["evidence_hash"]:
            raise Error("fit judgment differs from current evidence")
        return copied
    if (not isinstance(value, Mapping) or set(value) != {"schema_version", "raw_judgment",
            "verifier_result", "evidence_input", "claim"} or value.get("schema_version") != SIGNAL_SCHEMA
        or value.get("claim") != scope.get("claim") or value.get("evidence_input") != scope.get("evidence_input")
        or not isinstance(value.get("verifier_result"), Mapping) or not value["verifier_result"]):
        raise Error("signal evidence envelope is invalid")
    from qualification.scoring.intent_evidence_reuse import reusable_evidence_terminal_judgment, terminal_result
    raw, native = value["raw_judgment"], value["verifier_result"]
    if not reusable_evidence_terminal_judgment(
            raw, evidence_input=scope["evidence_input"], expected_urls=scope["expected_urls"],
            expected_index=scope["expected_index"], expected_evidence_type=scope["expected_evidence_type"],
            allowed_context_urls=scope.get("expected_context_urls", [])):
        raise Error("signal evidence is not terminal and reusable")
    contents = scope["evidence_input"].get("contents")
    trace = raw["judge_verdict"]["verification_trace"]
    if (not isinstance(contents, Mapping)
        or not terminal_result(native, contents=contents, source_url=scope["expected_urls"][0])
        or trace.get("intent_verdict") != native.get("verdict")
        or trace.get("stage1") != native.get("stage1")
        or trace.get("stage3") != native.get("stage3")
        or trace.get("final_disposition") != native.get("decision")):
        raise Error("full native verifier result differs from terminal signal")
    return _copy(value)
