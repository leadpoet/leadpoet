"""Company-qualification training records from Arena training exports.

Each exported execution becomes one record per submitted company. The input
is what the agent had: the ICP it received, the evaluation date and its own
company claim. The label is the accepted judgment's verdict for that company.

Labels come only from the stored judgment. A zero score alone is never a
negative label, and judge-side evidence never enters the input. Identical
claims judged more than once count once; a claim judged both ways is left
out. Train and evaluation splits never share a company cluster, so the
evaluation set holds companies the model has not seen.

Read-only and deterministic: the same exports and rules give the same
records, split and manifest hash.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from lab_arena import contact_policy, contracts, integrity

SCHEMA_VERSION = "leadpoet.lab_arena.training_dataset.v1"
RECORD_SCHEMA_VERSION = "leadpoet.lab_arena.company_qualification_record.v1"
EXAMPLE_SCHEMA_VERSION = "leadpoet.lab_arena.training_example.v1"
TRAINABLE = ("qualified", "wrong_fit", "insufficient_evidence")
INTEGRITY_ADAPTERS = frozenset({"qualification_integrity_v2", "qualification_contacts_v3"})
# Daily production rounds; a suffix marks a probe, replay or shadow round.
_PRODUCTION_ROUND_RE = re.compile(r"^arena-\d{4}-\d{2}-\d{2}$")
_LABEL_AUTHORITY_GAPS = frozenset({"scorer_image_identity_missing", "reviewer_runtime_identity_unverified"})
_INPUT_CONFLICTS = frozenset({
    "runtime_input_context_mismatch", "judge_input_context_mismatch", "runtime_input_snapshot_invalid",
})
# The exporter's redaction marker, literal or percent-encoded inside a URL.
_REDACTION_RE = re.compile(r"\[REDACTED\]|%5[Bb]REDACTED%5[Dd]")
_CLAIM_CODES = ("company_linkedin_required_or_invalid", "us_headquarters_state_required_or_invalid",
                "state_must_be_text")
TRAINING_PERMISSION = {
    "status": "unconfirmed",
    "reason": "Submission consent records public rerun only; confirm training use before training.",
}


@dataclass(frozen=True)
class Rules:
    """Selection rules, recorded verbatim in the manifest."""

    eval_percent: int = 20
    seed: str = "leadpoet-arena-dataset-v1"
    production_rounds_only: bool = True
    scorer_image_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.eval_percent) is not int or not 0 < self.eval_percent < 100:
            raise ValueError("eval_percent must be between 1 and 99")
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError("seed is required")


def _hash(value: Any) -> str:
    return contracts.document_hash(value)


def example_exclusion(example: Mapping[str, Any], rules: Rules) -> Optional[str]:
    """Why an exported execution cannot supply labels, or ``None``."""

    if example.get("schema_version") != EXAMPLE_SCHEMA_VERSION:
        return "example_schema_unsupported"
    # The exporter hashes every other field; an edited file no longer matches.
    if example.get("example_hash") != _hash({key: value for key, value in example.items() if key != "example_hash"}):
        return "example_hash_mismatch"
    configuration = example.get("configuration") or {}
    if rules.production_rounds_only and (
        not _PRODUCTION_ROUND_RE.fullmatch(str(example.get("round_id") or ""))
        or example.get("round_status") != "published"
    ):
        return "not_a_published_production_round"
    adapter = (configuration.get("scorer_policy") or {}).get("scoring_adapter_version")
    if adapter not in INTEGRITY_ADAPTERS:
        return "scorer_adapter_without_company_labels"
    if rules.scorer_image_digest and configuration.get("scorer_image_digest") != rules.scorer_image_digest:
        return "outside_scorer_cohort"
    completeness = example.get("completeness") or {}
    dimensions = completeness.get("dimensions") or {}
    if dimensions.get("context") is not True or dimensions.get("input_output_label") is not True:
        return "label_unverified"
    reasons = set(completeness.get("incomplete_reasons") or ())
    if reasons & _LABEL_AUTHORITY_GAPS:
        return "label_authority_unverified"
    if reasons & _INPUT_CONFLICTS:
        return "input_conflict"
    return None


def _receipt(judgment: Mapping[str, Any], gate: str) -> Mapping[str, Any]:
    for receipt in judgment.get("verifier_gate_receipts") or ():
        if isinstance(receipt, Mapping) and receipt.get("gate") == gate:
            return receipt
    return {}


def _mismatched_fit(fit: Mapping[str, Any]) -> str:
    dimensions = fit.get("company_fit_dimensions") or {}
    names = sorted(name for name, value in dimensions.items() if value == "mismatch")
    if fit.get("required_attribute_decision") == "mismatch":
        names.append("required_attribute")
    return ",".join(names) or "fit"


def classify(row: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    """Return (label class, sub-reason) for one exported company row."""

    from qualification.scoring import competition

    status = row.get("label_status")
    if status == "outside_evaluated_slice":
        return "outside_scored_slice", None
    if status == "skipped":
        return "bucket_skipped", "submitted_employee_bucket"
    judgment = row.get("judgment")
    if status != "scored" or not isinstance(judgment, Mapping):
        return "unjudged", str(status or "missing")
    if judgment.get("duplicate_company") is True:
        return "duplicate", None
    if judgment.get("company_qualified") is True:
        return "qualified", None
    fit = _receipt(judgment, "company_fit")
    failure_class = fit.get("failure_class")
    reason = str(judgment.get("failure_reason") or "")
    if failure_class == "company_verification_exhausted" or (
        competition.scorer_breakdown_has_retryable_infrastructure_failure(judgment, integrity_policy=True)
    ):
        return "provider_error", failure_class or "retryable_infrastructure"
    if (failure_class == "model_contract_incompatible"
            or reason.lower().startswith("company model contract incompatible")
            or any(code in reason for code in _CLAIM_CODES)):
        return "invalid_submission", failure_class or "claim_contract"
    if fit.get("decision") == "mismatch":
        return "wrong_fit", _mismatched_fit(fit)
    if fit.get("decision") == "unavailable":
        return "insufficient_evidence", "fit_" + str(failure_class or "unavailable")
    details = judgment.get("intent_signals_detail")
    if isinstance(details, list) and not competition.has_verified_primary_intent(details):
        return "insufficient_evidence", "primary_intent_unverified"
    intent_details = _receipt(judgment, "intent_details")
    if intent_details and intent_details.get("decision") != "match":
        return "insufficient_evidence", "intent_details_" + str(intent_details.get("decision") or "unavailable")
    if judgment.get("contact_qualified") is False:
        return "contact_failed", None
    return "unclassified", None


def _label(judgment: Mapping[str, Any], label_class: str, sub_reason: Optional[str]) -> Dict[str, Any]:
    from qualification.scoring import competition

    fit = _receipt(judgment, "company_fit")
    intent_details = _receipt(judgment, "intent_details")
    details = judgment.get("intent_signals_detail")
    signals = []
    for detail in details if isinstance(details, list) else ():
        if not isinstance(detail, Mapping):
            continue
        verdict = detail.get("judge_verdict") if isinstance(detail.get("judge_verdict"), Mapping) else {}
        signals.append({"matched_icp_signal": detail.get("matched_icp_signal"),
                        "decision": verdict.get("decision"), "rejection_reason": verdict.get("rejection_reason"),
                        "date_verdict": detail.get("date_verdict"), "evidence_urls": detail.get("evidence_urls")})
    return {
        "class": label_class, "sub_reason": sub_reason,
        "company_qualified": judgment.get("company_qualified") is True,
        "final_score": judgment.get("final_score"),
        "failure_reason": judgment.get("failure_reason"),
        "fit": {key: fit.get(key) for key in (
            "decision", "company_fit_dimensions", "required_attribute_decision", "failure_class")},
        "primary_intent_verified": (competition.has_verified_primary_intent(details)
                                    if isinstance(details, list) else None),
        "signals": signals,
        "intent_details_decision": intent_details.get("decision") if intent_details else None,
    }


def _contacts(configuration: Mapping[str, Any]) -> bool:
    try:
        return contact_policy.enabled(configuration)
    except ValueError:
        return False


def agent_icp(example: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    """The ICP the agent received: its stored input, else the same projection."""

    snapshot = (example.get("input_snapshots") or {}).get(example.get("run_id")) or {}
    icp = (snapshot.get("input") or {}).get("icp") if isinstance(snapshot, Mapping) else None
    if isinstance(icp, Mapping) and not (snapshot.get("privacy") or {}).get("redacted"):
        return dict(icp), "input_snapshot"
    contacts = _contacts(example.get("configuration") or {})
    return integrity.agent_visible_icp(example.get("icp") or {}, contacts_required=contacts), "agent_visible_projection"


def company_keys(company: Mapping[str, Any]) -> List[str]:
    """Label-free handles for one submitted company. Splits merge on any."""

    from leadpoet_verifier.identity.normalization import NormalizationError, normalize_linkedin_company_url
    from qualification.scoring.arena_integrity import canonical_company_identity

    identity = canonical_company_identity(company)
    keys = []
    if identity.registrable_domain or identity.normalized_name:
        keys.append(identity.key)
    if identity.registrable_domain:
        keys.append("domain:" + identity.registrable_domain)
    try:
        slug = normalize_linkedin_company_url(str(company.get("company_linkedin") or "")).normalized_slug
    except (NormalizationError, TypeError, ValueError):
        slug = ""  # An absent or malformed submitted URL adds no handle.
    if slug:
        keys.append("linkedin:" + slug)
    return keys


def _provenance(example: Mapping[str, Any], index: int) -> Dict[str, Any]:
    judgment = example.get("judgment") or {}
    score_run = example.get("score_run_id")
    if judgment.get("judgment_cache_source_run_id"):
        return {"judgment_source": "run_cache", "source_score_run_id": judgment["judgment_cache_source_run_id"]}
    refs = {ref.get("company_index"): ref for ref in judgment.get("company_judgment_refs") or ()
            if isinstance(ref, Mapping)}
    evidence = {item.get("cache_key"): item for item in judgment.get("company_authority_evidence") or ()
                if isinstance(item, Mapping)}
    ref = refs.get(index) or {}
    source = ((evidence.get(ref.get("cache_key")) or {}).get("evidence_doc") or {}).get("source_score_run_id")
    if source and source != score_run:
        return {"judgment_source": "company_cache", "source_score_run_id": source}
    return {"judgment_source": "fresh", "source_score_run_id": score_run}


def records_from_example(example: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One record per submitted company of a usable exported execution."""

    configuration = example.get("configuration") or {}
    icp, icp_source = agent_icp(example)
    fingerprint = integrity.requirement_fingerprint(example.get("icp") or {},
                                                    contacts_required=_contacts(configuration))
    rows = list(example.get("companies") or ())
    if not rows:  # No verified judgment: every submitted company is unjudged.
        rows = [{"company_index": index, "company": company, "label_status": "unjudged", "judgment": None}
                for index, company in enumerate((example.get("output") or {}).get("companies") or ())]
    records = []
    for row in rows:
        company = row.get("company") or {}
        index = row.get("company_index")
        label_class, sub_reason = classify(row)
        judgment = row.get("judgment")
        records.append({
            "schema_version": RECORD_SCHEMA_VERSION,
            "record_id": _hash({"example_id": example.get("example_id"), "company_index": index}),
            "meta": {
                "example_id": example.get("example_id"), "example_hash": example.get("example_hash"),
                "run_id": example.get("run_id"), "score_run_id": example.get("score_run_id"),
                "round_id": example.get("round_id"), "submission_id": example.get("submission_id"),
                "role": example.get("role"), "stage": example.get("stage"),
                "icp_position": example.get("icp_position"), "company_index": index,
                "evaluation_date": example.get("evaluation_date"),
                "scorer_image_digest": configuration.get("scorer_image_digest"),
                "scorer_policy_hash": _hash(configuration.get("scorer_policy") or {}),
            },
            "input": {"icp": icp, "icp_source": icp_source,
                      "evaluation_date": example.get("evaluation_date"), "company": company},
            "label": (_label(judgment, label_class, sub_reason) if isinstance(judgment, Mapping)
                      else {"class": label_class, "sub_reason": sub_reason}),
            "label_provenance": _provenance(example, index),
            "group_keys": {"company_keys": company_keys(company), "icp_fingerprint": fingerprint,
                           "source_family": (example.get("source") or {}).get("normalized_sha256")
                           or (example.get("source") or {}).get("sha256")},
        })
    return records


class _Clusters:
    """Union-find over label-free company handles."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, keys: Iterable[str]) -> None:
        keys = list(keys)
        for other in keys[1:]:
            first, second = self.find(keys[0]), self.find(other)
            if first != second:
                self.parent[max(first, second)] = min(first, second)


def build(examples: Iterable[Mapping[str, Any]], rules: Rules = Rules()) -> Tuple[Dict[str, List[dict]], Dict[str, Any]]:
    """Return ({train, eval, analysis}: records) and the dataset manifest."""

    sources, records, privacy_filtered = [], [], 0
    for example in examples:
        reason = example_exclusion(example, rules)
        sources.append({"example_id": example.get("example_id"), "example_hash": example.get("example_hash"),
                        "run_id": example.get("run_id"), "round_id": example.get("round_id"),
                        "used": reason is None, "exclusion": reason})
        if reason is None:
            for record in records_from_example(example):
                # Redaction elsewhere in the run, mostly URL tokens in fetched
                # pages, never reaches these fields. One inside the record's
                # own input would train on a placeholder, so that record goes.
                if _REDACTION_RE.search(contracts.canonical_json(record["input"])):
                    privacy_filtered += 1
                    continue
                records.append(record)
    # Identical ICP requirements, evaluation date and claim: replays, retries
    # and cached judgments count once. The date is part of the input because
    # intent recency depends on it. Only verdicts can conflict; a claim judged
    # both ways is left out. Non-verdict outcomes group by their own class.
    groups: Dict[str, List[dict]] = {}
    for record in sorted(records, key=lambda item: item["record_id"]):
        claim = _hash({"icp": record["group_keys"]["icp_fingerprint"],
                       "evaluation_date": record["input"]["evaluation_date"],
                       "company": record["input"]["company"]})
        record["group_keys"]["claim_key"] = claim
        label = record["label"]["class"]
        groups.setdefault(claim + "|" + ("verdict" if label in TRAINABLE else label), []).append(record)
    kept, duplicates, conflicts = [], 0, 0
    for group in groups.values():
        if len({item["label"]["class"] for item in group}) > 1:
            conflicts += 1
            continue
        kept.append(group[0])
        duplicates += len(group) - 1
    clusters = _Clusters()
    for record in kept:
        clusters.union(record["group_keys"]["company_keys"] or [record["record_id"]])
    splits: Dict[str, List[dict]] = {"train": [], "eval": [], "analysis": []}
    for record in kept:
        handles = record["group_keys"]["company_keys"] or [record["record_id"]]
        cluster = clusters.find(handles[0])
        record["group_keys"]["company_cluster"] = cluster
        if record["label"]["class"] not in TRAINABLE:
            splits["analysis"].append(record)
            continue
        bucket = int(hashlib.sha256((rules.seed + "\x00" + cluster).encode("utf-8")).hexdigest(), 16) % 100
        splits["eval" if bucket < rules.eval_percent else "train"].append(record)
    for name in splits:
        splits[name].sort(key=lambda item: item["record_id"])
    counts = {name: {} for name in splits}
    for name, items in splits.items():
        for record in items:
            label = record["label"]["class"]
            counts[name][label] = counts[name].get(label, 0) + 1
    exclusions: Dict[str, int] = {}
    for source in sources:
        if source["exclusion"]:
            exclusions[source["exclusion"]] = exclusions.get(source["exclusion"], 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION, "record_schema_version": RECORD_SCHEMA_VERSION,
        "rules": asdict(rules), "training_permission": dict(TRAINING_PERMISSION),
        "sources": sorted(sources, key=lambda item: str(item["example_id"])),
        "counts": {"examples_used": sum(source["used"] for source in sources),
                   "examples_excluded": exclusions, "records_by_split_and_label": counts,
                   "records_privacy_filtered": privacy_filtered,
                   "duplicate_records_merged": duplicates, "conflicting_label_groups_excluded": conflicts},
        "record_hashes": {name: [_hash(record) for record in items] for name, items in splits.items()},
    }
    manifest["dataset_version"] = _hash({key: manifest[key] for key in (
        "schema_version", "record_schema_version", "rules", "sources", "record_hashes")})
    return splits, manifest


__all__ = [
    "RECORD_SCHEMA_VERSION", "Rules", "SCHEMA_VERSION", "TRAINABLE", "agent_icp", "build", "classify",
    "company_keys", "example_exclusion", "records_from_example",
]
