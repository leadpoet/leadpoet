"""Offline reconstruction of both inputs to prospective evidence reuse."""
from __future__ import annotations

from collections.abc import Mapping

from lab_arena import company_judgments, contracts, evidence_judgments as evidence, reviewer_runtime, signal_judgments


def validate_use(*, use, row, judge, scoring_input, company_index, signal_index,
                 runs, round_row, source_input_reader, fields, runtime_fields):
    """Return original authority only after proving source and destination joins.

    Candidate keys alone cannot establish that a rewritten claim was accepted.
    Reconstruct the current scope and its original source from retained inputs;
    changed claims additionally require the persisted equivalence receipt.
    """
    config = round_row["configuration_doc"]
    source = runs.get(row.get("source_score_run_id"))
    use_judge = runs.get(use.get("run_id"))
    if source is None or use_judge is None:
        raise ValueError("evidence_source_missing")

    def frozen(run):
        value = scoring_input if run["run_id"] == judge["run_id"] else (
            source_input_reader(run) if source_input_reader is not None else None)
        if (not isinstance(value, Mapping) or set(value) - runtime_fields != set(fields)
                or value.get("scored_run_id") != run.get("scored_run_id")):
            raise ValueError("evidence_frozen_input_invalid")
        return {key: value[key] for key in fields}

    original = row.get("scope_doc")
    if not isinstance(original, Mapping) or not evidence.enabled(original):
        raise ValueError("evidence_scope_invalid")
    native = (original.get("family_doc") or {}).get("native_input") or {}

    def scope(run, value, ci, si):
        selected = reviewer_runtime.selected_image(run, config)
        return evidence.build_scope(
            scoring_input=value, company_index=ci, signal_index=si,
            round_id=judge["round_id"], network_name=round_row["arena_network_name"],
            netuid=int(round_row.get("arena_netuid") or 0),
            scorer_image_digest=config["scorer_image_digest"],
            scorer_image_reference=config["scorer_image_reference"],
            runtime_image_digest=selected["image_digest"],
            runtime_image_reference=selected["image_reference"],
            evidence_input=original.get("evidence_input"), evidence_hash=original.get("evidence_hash"),
            verified_company_identity=native.get("verified_company_identity"),
            verified_funding_stage_context=native.get("verified_funding_stage_context"),
        )

    # Source indices are deliberately absent from content keys. Find the exact
    # source input by reconstruction; duplicate identical rows are harmless.
    source_input = frozen(source)
    found = False
    for ci, effective in enumerate(company_judgments.effective_company_inputs(source_input)):
        indexes = [-1] if signal_index == -1 else range(len(effective["company"].get("intent_signals") or []))
        for si in indexes:
            if scope(source, source_input, ci, si) == original:
                found = True
                break
        if found:
            break
    if not found:
        raise ValueError("evidence_original_input_mismatch")

    current = scope(judge, scoring_input, company_index, signal_index)
    accepted = scope(use_judge, frozen(use_judge), use["company_index"], use["signal_index"])
    if current != accepted or any(current.get(key) != original.get(key) for key in (
            "family_doc", "family_hash", "evidence_input", "evidence_hash")):
        raise ValueError("evidence_current_input_mismatch")
    reference = signal_judgments.reference(original, row["authority_slot"])
    proof = use.get("acceptance_doc")
    if proof is None:
        if current != original:
            raise ValueError("evidence_acceptance_missing")
    else:
        receipt = evidence.validate_equivalence(
            proof.get("equivalence_receipt"), original.get("claim"), current.get("claim"),
            scoring_input["scorer_policy"])
        if proof != {"schema_version": "leadpoet.lab_arena.evidence_use.v2",
                     "current_scope": current, "candidate_reference": reference,
                     "equivalence_receipt": receipt}:
            raise ValueError("evidence_acceptance_invalid")
    judgment = evidence.validate_judgment(row.get("judgment_doc"), original)
    if (any(use.get(key) != row.get(key) for key in (
            "cache_key", "authority_slot", "signal_input_hash", "source_score_run_id", "judgment_hash"))
            or row.get("cache_key") != original["cache_key"]
            or row.get("signal_input_hash") != original["signal_input_hash"]
            or row.get("judgment_hash") != contracts.document_hash(judgment)):
        raise ValueError("evidence_authority_mismatch")
    return original, judgment
