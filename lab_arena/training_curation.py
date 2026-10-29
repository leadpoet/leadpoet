"""Offline, exact-record review of already eligible Arena training candidates.

Review decisions are human assertions, not inferred truth or training permission.
This module never changes raw candidates, split assignment, or source proof.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping

from lab_arena import contracts, training_dataset

SCHEMA_VERSION = "leadpoet.lab_arena.training_curation.v1"
_STATUSES = frozenset({"accepted", "rejected", "unreviewed"})


def _identity(record):
    return record.get("record_id") or record.get("candidate_id")


def project_judgment(judgment):
    """Retain bounded target-side intent facts without putting them in input."""
    if not isinstance(judgment, Mapping):
        return {"intent_details_status": "missing", "signals": []}
    details = judgment.get("intent_signals_detail")
    if not isinstance(details, list):
        return {"intent_details_status": "missing", "signals": []}
    signals = []
    for detail in details:
        if not isinstance(detail, Mapping):
            continue
        verdict = detail.get("judge_verdict") if isinstance(detail.get("judge_verdict"), Mapping) else {}
        trace = verdict.get("verification_trace") if isinstance(verdict.get("verification_trace"), Mapping) else {}
        overall = trace.get("intent_verdict") if isinstance(trace.get("intent_verdict"), Mapping) else {}
        evaluations = overall.get("signal_evaluations")
        evaluations = evaluations if isinstance(evaluations, list) else []
        signals.append({"matched_icp_signal": detail.get("matched_icp_signal"),
                        "after_decay": detail.get("after_decay"),
                        "decision": verdict.get("decision"),
                        "date_verdict": detail.get("date_verdict"),
                        "overall_verdict": overall.get("overall_verdict"),
                        "overall_risks": copy.deepcopy(overall.get("missing_or_risks")),
                        "evidence_urls": copy.deepcopy(detail.get("evidence_urls")),
                        "signal_evaluations": [{"signal_status": item.get("signal_status"),
                                                "risk_notes": copy.deepcopy(item.get("risk_notes")),
                                                "evidence_urls_used": copy.deepcopy(item.get("evidence_urls_used"))}
                                               for item in evaluations if isinstance(item, Mapping)]})
    return {"intent_details_status": "present", "signals": signals}


def qualification_flags(record, projection=None):
    """Flag uncertain primary proof and stored verdict conflicts for human review."""
    schema = record.get("schema_version")
    if schema == "leadpoet.lab_arena.agent_step_candidate.v1":
        return ["observed_action_quality_unreviewed"]
    if schema == "leadpoet.lab_arena.full_thinking_step.v1":
        return ["generated_thinking_quality_unreviewed"]
    if schema != training_dataset.RECORD_SCHEMA_VERSION:
        return []
    label = record.get("label") or {}
    projection = projection or {"intent_details_status": "missing", "signals": []}
    signals = projection["signals"]
    primary = [item for item in signals if isinstance(item, Mapping)
               and type(item.get("matched_icp_signal")) is int
               and item["matched_icp_signal"] == 0]
    reasons = []
    if label.get("class") == "qualified":
        if (projection["intent_details_status"] != "present"
                or not primary
                or any(item.get("overall_verdict") is None for item in primary)):
            reasons.append("primary_intent_source_context_unknown")
        if not any(item.get("decision") == "verified"
                   and isinstance(item.get("after_decay"), (int, float))
                   and not isinstance(item.get("after_decay"), bool)
                   and math.isfinite(item["after_decay"])
                   and item["after_decay"] > 0
                   and item.get("date_verdict") == "in_window"
                   and item.get("overall_verdict") == "qualified" for item in primary):
            reasons.append("required_primary_without_verified_in_window_qualified_support")
        if any(item.get("overall_verdict") == "needs_review" for item in primary):
            reasons.append("primary_overall_needs_review")
        if any(item.get("overall_verdict") == "disqualified" for item in primary):
            reasons.append("primary_overall_disqualified")
        if any(item.get("overall_risks") or any(
                evaluation.get("risk_notes") for evaluation in item.get("signal_evaluations") or []
                if isinstance(evaluation, Mapping)) for item in primary):
            reasons.append("primary_intent_risk_notes")
    elif label.get("class") == "wrong_fit":
        # A source quote can pass structural proof while its company semantics
        # remain disputed. The stored negative is never changed automatically.
        reasons.append("negative_fit_semantics_need_human_review")
    if record.get("selection_exclusion"):
        reasons.append("existing_selection_exclusion")
    return sorted(set(reasons))


def flag_manifest(splits, source_version, projections=None):
    entries = []
    for split in ("train", "eval", "analysis"):
        for record in splits[split]:
            projection = (projections or {}).get(_identity(record))
            reasons = qualification_flags(record, projection)
            if reasons or projection is not None:
                entries.append({"record_id": _identity(record),
                                "record_hash": contracts.document_hash(record),
                                "source_example_hash": (record.get("meta") or {}).get("example_hash"),
                                "split": split, "reasons": reasons,
                                "retained_verdict": projection})
    result = {"schema_version": SCHEMA_VERSION, "source_dataset_version": source_version,
              "training_permission": copy.deepcopy(training_dataset.TRAINING_PERMISSION),
              "source_record_hashes": {name: [contracts.document_hash(row) for row in splits[name]]
                                       for name in ("train", "eval", "analysis")},
              "flagged_records": sorted(entries, key=lambda item: (item["split"], item["record_id"]))}
    result["curation_version"] = contracts.document_hash(result)
    return result


def review_sample(splits, size, source_manifest, curation_manifest):
    """Use the established qualification strata; adapt other candidate types."""
    _check_manifest_binding(source_manifest, curation_manifest)
    _check_split_binding(splits, source_manifest, curation_manifest)
    first = next((row for name in ("train", "eval", "analysis") for row in splits[name]
                  if "record_id" in row or "candidate_id" in row), None)
    if first is not None and "record_id" in first:
        result, summary = training_dataset.review_sample(splits, size)
        for row in result:
            row["source_dataset_version"] = source_manifest["dataset_version"]
            row["curation_version"] = curation_manifest["curation_version"]
        return result, {**summary, "source_dataset_version": source_manifest["dataset_version"],
                        "curation_version": curation_manifest["curation_version"]}
    if type(size) is not int or size < 0:
        raise ValueError("invalid_review_sample_size")
    records = [(name, row) for name in ("train", "eval") for row in splits[name]]
    records.sort(key=lambda pair: contracts.document_hash([pair[0], pair[1]["candidate_id"]]))
    result = [{"record_id": row["candidate_id"], "record_hash": contracts.document_hash(row),
               "record_file": name + ".jsonl", "source_refs": copy.deepcopy(row["source_refs"]),
               "source_dataset_version": source_manifest["dataset_version"],
               "curation_version": curation_manifest["curation_version"],
               "human_review_status": "unreviewed", "reviewed_label": None,
               "review_reason": None, "reviewer": None} for name, row in records[:size]]
    return result, {"requested": size, "selected": len(result), "human_review_status": "unreviewed",
                    "source_dataset_version": source_manifest["dataset_version"],
                    "curation_version": curation_manifest["curation_version"]}


def _check_manifest_binding(source_manifest, curation_manifest):
    if not isinstance(source_manifest, Mapping) or not isinstance(curation_manifest, Mapping):
        raise ValueError("curation_manifest_mismatch")
    if source_manifest.get("schema_version") == training_dataset.SCHEMA_VERSION:
        keys = ("schema_version", "record_schema_version", "rules", "sources", "record_hashes")
        valid_source = all(key in source_manifest for key in keys) and (
            source_manifest.get("dataset_version") == contracts.document_hash(
                {key: source_manifest[key] for key in keys}))
    elif source_manifest.get("schema_version") in {
            "leadpoet.lab_arena.agent_step_dataset.v1",
            "leadpoet.lab_arena.full_thinking_dataset.v1"}:
        valid_source = source_manifest.get("dataset_version") == contracts.document_hash({
            key: value for key, value in source_manifest.items()
            if key not in {"dataset_version", "review_sample"}})
    else:
        valid_source = False
    if (not valid_source
            or not isinstance(source_manifest.get("dataset_version"), str)
            or not source_manifest["dataset_version"]
            or curation_manifest.get("schema_version") != SCHEMA_VERSION
            or curation_manifest.get("source_dataset_version") != source_manifest["dataset_version"]
            or source_manifest.get("training_permission") != training_dataset.TRAINING_PERMISSION
            or curation_manifest.get("training_permission") != source_manifest.get("training_permission")
            or curation_manifest.get("curation_version") != contracts.document_hash({
                key: value for key, value in curation_manifest.items() if key != "curation_version"})):
        raise ValueError("curation_manifest_mismatch")


def _check_split_binding(splits, source_manifest, curation_manifest):
    names = ("train", "eval", "analysis")
    if not isinstance(splits, Mapping) or set(splits) != set(names):
        raise ValueError("curation_split_mismatch")
    hashes = {name: [contracts.document_hash(row) for row in splits[name]] for name in names}
    if curation_manifest.get("source_record_hashes") != hashes:
        raise ValueError("curation_split_mismatch")
    if source_manifest["schema_version"] == training_dataset.SCHEMA_VERSION:
        if source_manifest.get("record_hashes") != hashes:
            raise ValueError("curation_split_mismatch")
    elif source_manifest["schema_version"] == "leadpoet.lab_arena.agent_step_dataset.v1":
        if source_manifest.get("split_hashes") != {
                name: contracts.document_hash(splits[name]) for name in names}:
            raise ValueError("curation_split_mismatch")


def apply_review(splits, source_manifest, decisions, curation_manifest):
    """Return a separate reviewed projection; reject ambiguous or stale decisions."""
    _check_manifest_binding(source_manifest, curation_manifest)
    _check_split_binding(splits, source_manifest, curation_manifest)
    candidates = {}
    for split in ("train", "eval"):
        for record in splits[split]:
            identity = _identity(record)
            if not isinstance(identity, str) or not identity or identity in candidates:
                raise ValueError("ambiguous_candidate_identity")
            candidates[identity] = (split, record)
    selected = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("invalid_review_decision")
        if (decision.get("source_dataset_version") != source_manifest["dataset_version"]
                or decision.get("curation_version") != curation_manifest["curation_version"]):
            raise ValueError("stale_review_manifest")
        identity = decision.get("record_id")
        if identity not in candidates or identity in selected:
            raise ValueError("unknown_or_duplicate_review_decision")
        split, record = candidates[identity]
        if (decision.get("record_hash") != contracts.document_hash(record)
                or decision.get("record_file") != split + ".jsonl"):
            raise ValueError("stale_review_decision")
        if "record_id" in record:
            if decision.get("input_provenance") != record.get("meta"):
                raise ValueError("review_source_mismatch")
        elif decision.get("source_refs") != record.get("source_refs"):
            raise ValueError("review_source_mismatch")
        if "original_label" in decision and decision["original_label"] != record.get("label"):
            raise ValueError("review_original_label_mismatch")
        status = decision.get("human_review_status")
        if status not in _STATUSES:
            raise ValueError("invalid_review_status")
        if status != "unreviewed" and (not isinstance(decision.get("reviewer"), str)
                                       or not decision["reviewer"].strip()
                                       or not isinstance(decision.get("review_reason"), str)
                                       or not decision["review_reason"].strip()):
            raise ValueError("reviewer_and_reason_required")
        reviewed_label = decision.get("reviewed_label")
        original = record.get("label")
        if status == "accepted":
            if original is None:
                if reviewed_label is not None:
                    raise ValueError("candidate_label_override_forbidden")
            elif reviewed_label != original:
                # Corrections need an independent evidence check and class-specific
                # semantics. Reject them until that contract exists.
                raise ValueError("qualification_label_override_forbidden")
            if record.get("selection_exclusion"):
                raise ValueError("ineligible_review_acceptance")
            if original is not None and (original.get("class") not in training_dataset.TRAINABLE
                                         or training_dataset._evidence_gap(record)):
                raise ValueError("ineligible_review_acceptance")
        elif reviewed_label is not None:
            raise ValueError("nonaccepted_reviewed_label")
        selected[identity] = decision
    output = {"train": [], "eval": [], "analysis": copy.deepcopy(splits["analysis"])}
    for split in ("train", "eval"):
        for record in splits[split]:
            decision = selected.get(_identity(record))
            reviewed = copy.deepcopy(record)
            if decision and decision["human_review_status"] == "accepted":
                reviewed["reviewed_label"] = copy.deepcopy(decision.get("reviewed_label"))
                reviewed["human_review_status"] = "accepted"
                output[split].append(reviewed)
            else:
                reviewed["human_review_status"] = (decision or {}).get("human_review_status", "unreviewed")
                reviewed["reviewed_label"] = None
                reviewed["selection_exclusion"] = "training_review_" + reviewed["human_review_status"]
                output["analysis"].append(reviewed)
    counts = {name: len(rows) for name, rows in output.items()}
    status_counts = Counter((decision.get("human_review_status") or "unreviewed")
                            for decision in selected.values())
    manifest = {"schema_version": SCHEMA_VERSION, "source_dataset_version": source_manifest["dataset_version"],
                "source_curation_version": curation_manifest["curation_version"],
                "source_schema_version": source_manifest["schema_version"],
                "training_permission": copy.deepcopy(source_manifest["training_permission"]),
                "counts": counts, "review_decision_counts": dict(sorted(status_counts.items())),
                "decision_hashes": sorted(contracts.document_hash(dict(item)) for item in selected.values()),
                "record_hashes": {name: [contracts.document_hash(row) for row in rows]
                                  for name, rows in output.items()}}
    manifest["dataset_version"] = contracts.document_hash(manifest)
    return output, manifest
