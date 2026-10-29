"""Company-qualification training records from Arena training exports.

Each exported execution becomes one record per submitted company. The input
pairs the ICP and submitted claim with bounded, retained source-page excerpts.
The label is the accepted judgment's verdict for that company.

Labels come only from the stored judgment. A zero score alone is never a
negative label, and judge decisions never enter the input. Identical
claims judged more than once count once; a claim judged both ways is left
out. Train and evaluation splits never share a company cluster, so the
evaluation set holds companies the model has not seen.

Read-only and deterministic: the same exports and rules give the same
records, split and manifest hash.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from lab_arena import contact_policy, contracts, integrity

SCHEMA_VERSION = "leadpoet.lab_arena.training_dataset.v2"
RECORD_SCHEMA_VERSION = "leadpoet.lab_arena.company_qualification_record.v2"
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
_MAX_SOURCE_CONTEXTS = 12
_MAX_SOURCE_BYTES = 64 * 1024
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
    scorer_policy_hash: Optional[str] = None

    eval_mode: str = "company"
    eval_after: Optional[str] = None

    def __post_init__(self) -> None:
        if self.eval_mode not in {"company", "company_and_icp"}:
            raise ValueError("unsupported eval_mode")
        if self.eval_after is not None and _date(self.eval_after) is None:
            raise ValueError("eval_after must be YYYY-MM-DD")
        if type(self.eval_percent) is not int or not 0 < self.eval_percent < 100:
            raise ValueError("eval_percent must be between 1 and 99")
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError("seed is required")


def _date(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


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
    if rules.scorer_policy_hash and _hash(configuration.get("scorer_policy") or {}) != rules.scorer_policy_hash:
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


def _source_url(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 2048 or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username and not parsed.password
    except ValueError:
        return False


def _source_evidence(judgment: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Project retained page excerpts and fit citations without clipping proof."""

    if not isinstance(judgment, Mapping):
        return [], None
    candidates = []
    fit = _receipt(judgment, "company_fit")
    dimensions = fit.get("dimension_evidence")
    if isinstance(dimensions, Mapping):
        from qualification.scoring.linkedin_company_size import (
            STRUCTURED_PROFILE_PROVIDER, STRUCTURED_PROFILE_SOURCE_FIELD,
        )

        identity_detail = dimensions.get("identity")
        identity = (identity_detail.get("web_identity_receipt")
                    if isinstance(identity_detail, Mapping) else None)
        if (isinstance(identity, Mapping)
                and identity.get("evidence_source") in {
                    "company_homepage", "company_web_reverification"}
                and isinstance(identity.get("observed_name"), str)
                and identity.get("observed_name")
                and isinstance(identity.get("observed_domain"), str)
                and identity.get("observed_domain")):
            candidates.append({
                "content_kind": "company_identity_observation", "dimension": "identity",
                "observed_name": identity.get("observed_name"),
                "observed_domain": identity.get("observed_domain"),
                "observed_linkedin_slug": identity.get("observed_linkedin_slug"),
                "provenance": "company_fit_receipt",
            })
        for dimension in ("identity", "employee_size", "industry", "geography", "stage", "required_attribute"):
            detail = dimensions.get(dimension)
            web = detail.get("web_evidence") if isinstance(detail, Mapping) else None
            if not isinstance(web, Mapping):
                continue
            if (dimension == "employee_size" and set(web) == {
                    "employee_count", "provider", "source_field", "url", "website"}
                    and web.get("provider") == STRUCTURED_PROFILE_PROVIDER
                    and web.get("source_field") == STRUCTURED_PROFILE_SOURCE_FIELD
                    and isinstance(web.get("employee_count"), str)
                    and _source_url(web.get("url")) and _source_url(web.get("website"))):
                candidates.append({
                    "content_kind": "structured_provider_fact", "dimension": dimension,
                    "url": web["url"], "website": web["website"],
                    "employee_count": web["employee_count"], "provider": web["provider"],
                    "source_field": web["source_field"], "provenance": "company_fit_receipt",
                })
                continue
            url, quote = web.get("url"), web.get("quote")
            if _source_url(url) and isinstance(quote, str) and quote.strip():
                candidates.append({
                    "url": url, "text": quote, "content_kind": "verifier_source_quote",
                    "dimension": dimension, "source_publication_date": None,
                    "retrieved_at": None, "provenance": "company_fit_receipt",
                })
    for detail in judgment.get("intent_signals_detail") or ():
        if not isinstance(detail, Mapping):
            continue
        verdict = detail.get("judge_verdict")
        trace = verdict.get("verification_trace") if isinstance(verdict, Mapping) else None
        contexts = trace.get("verified_source_context") if isinstance(trace, Mapping) else None
        if not isinstance(contexts, list):
            continue
        for context in contexts:
            if not isinstance(context, Mapping):
                continue
            url, source_text = context.get("url"), context.get("text")
            if not _source_url(url) or not isinstance(source_text, str) or not source_text.strip():
                continue
            candidates.append({
                "url": url,
                "text": source_text,
                "content_kind": "retained_page_excerpt",
                "matched_icp_signal": detail.get("matched_icp_signal"),
                "source_publication_date": context.get("source_publication_date") or None
                if isinstance(context.get("source_publication_date"), str) else None,
                "retrieved_at": context.get("retrieved_at") or None
                if isinstance(context.get("retrieved_at"), str) else None,
                "provenance": "verified_source_context",
            })
    # Reject over-bound records rather than silently removing decisive proof.
    unique = {contracts.canonical_json(item): item for item in candidates}
    evidence = [unique[key] for key in sorted(unique)]
    if (len(evidence) > _MAX_SOURCE_CONTEXTS
            or len(contracts.canonical_json(evidence).encode("utf-8")) > _MAX_SOURCE_BYTES):
        return [], "retained_source_evidence_overflow"
    return evidence, None


def _evidence_gap(record: Mapping[str, Any]) -> Optional[str]:
    label = record["label"]
    evidence = record["input"]["source_evidence"]
    if record.get("source_evidence_error"):
        return record["source_evidence_error"]
    if not evidence:
        return "missing_retained_source_evidence"
    label_class = label["class"]
    fit = label.get("fit") or {}
    quoted = {item.get("dimension") for item in evidence
              if item.get("content_kind") == "verifier_source_quote"}
    supported = quoted | {item.get("dimension") for item in evidence
                          if item.get("content_kind") == "structured_provider_fact"}
    if label_class == "wrong_fit":
        mismatches = {key for key, value in (fit.get("company_fit_dimensions") or {}).items()
                      if value == "mismatch"}
        if fit.get("required_attribute_decision") == "mismatch":
            mismatches.add("required_attribute")
        if not mismatches or not mismatches.issubset(quoted):
            return "missing_wrong_fit_source_evidence"
    elif label_class == "qualified" or (label_class == "insufficient_evidence"
                                        and not str(label.get("sub_reason") or "").startswith("fit_")):
        dimensions = fit.get("company_fit_dimensions") or {}
        icp = record["input"]["icp"]
        required = set()
        if icp.get("employee_count"):
            required.add("employee_size")
        if any(icp.get(key) for key in ("industry", "sub_industry", "product_service")):
            required.add("industry")
        if icp.get("geography") or icp.get("country"):
            required.add("geography")
        if icp.get("company_stage"):
            required.add("stage")
        if icp.get("required_attribute"):
            required.add("required_attribute")
        if (dimensions.get("identity") != "match"
                or not any(item.get("content_kind") == "company_identity_observation" for item in evidence)
                or not required or any(dimensions.get(key) != "match" for key in required if key != "required_attribute")
                or ("required_attribute" in required and fit.get("required_attribute_decision") != "match")
                or not required.issubset(supported)):
            return "missing_company_fit_source_evidence"
        if not any(item.get("content_kind") == "retained_page_excerpt"
                   and item.get("matched_icp_signal") == 0 for item in evidence):
            return "missing_primary_signal_source_evidence"
    elif label_class == "insufficient_evidence":
        unavailable = {key for key, value in (fit.get("company_fit_dimensions") or {}).items()
                       if value == "unavailable"}
        if fit.get("required_attribute_decision") == "unavailable":
            unavailable.add("required_attribute")
        if not unavailable or not unavailable.issubset(supported):
            return "missing_fit_gap_source_evidence"
    return None


def _label_signature(label: Mapping[str, Any]) -> str:
    """Stable decision fields; ignore narrative and score variation."""

    fit = label.get("fit") or {}
    signals = label.get("signals") or ()
    return _hash({
        "class": label.get("class"), "sub_reason": label.get("sub_reason"),
        "fit": {key: fit.get(key) for key in (
            "decision", "company_fit_dimensions", "required_attribute_decision", "failure_class")},
        "primary_intent_verified": label.get("primary_intent_verified"),
        "signals": sorted(({"matched_icp_signal": item.get("matched_icp_signal"),
                            "decision": item.get("decision"),
                            "date_verdict": item.get("date_verdict")}
                           for item in signals if isinstance(item, Mapping)),
                          key=contracts.canonical_json),
        "intent_details_decision": label.get("intent_details_decision"),
    })


def _company_keys(company: Mapping[str, Any], judgment: Any) -> List[str]:
    keys = company_keys(company)
    if isinstance(judgment, Mapping):
        for alias in judgment.get("company_identity_alias_keys") or ():
            if isinstance(alias, str) and alias:
                keys.append(alias)
        identity = judgment.get("company_identity_key")
        if isinstance(identity, str) and identity:
            keys.append(identity)
    return sorted(set(keys))


def _provenance(example: Mapping[str, Any], index: int) -> Dict[str, Any]:
    judgment = example.get("judgment") or {}
    score_run = example.get("score_run_id")
    signal_sources = []
    for item in judgment.get("signal_authority_evidence") or ():
        if not isinstance(item, Mapping) or item.get("company_index") != index:
            continue
        source = (item.get("authority") or {}).get("source_score_run_id")
        if isinstance(source, str) and source:
            signal_sources.append({"signal_index": item.get("signal_index"),
                                   "source_score_run_id": source,
                                   "source": "cached" if source != score_run else "fresh"})
    signal_sources.sort(key=lambda item: (str(item["signal_index"]), item["source_score_run_id"]))
    if judgment.get("judgment_cache_source_run_id"):
        result = {"judgment_source": "run_cache", "source_score_run_id": judgment["judgment_cache_source_run_id"]}
        result["signal_sources"] = signal_sources
        return result
    refs = {ref.get("company_index"): ref for ref in judgment.get("company_judgment_refs") or ()
            if isinstance(ref, Mapping)}
    evidence = {item.get("cache_key"): item for item in judgment.get("company_authority_evidence") or ()
                if isinstance(item, Mapping)}
    ref = refs.get(index) or {}
    source = ((evidence.get(ref.get("cache_key")) or {}).get("evidence_doc") or {}).get("source_score_run_id")
    if source and source != score_run:
        return {"judgment_source": "company_cache", "source_score_run_id": source,
                "signal_sources": signal_sources}
    return {"judgment_source": "fresh", "source_score_run_id": score_run,
            "signal_sources": signal_sources}


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
        source_evidence, source_error = _source_evidence(judgment)
        records.append({
            "schema_version": RECORD_SCHEMA_VERSION,
            "record_id": _hash({"example_id": example.get("example_id"),
                                "example_hash": example.get("example_hash"), "company_index": index}),
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
                      "evaluation_date": example.get("evaluation_date"), "company": company,
                      "source_evidence": source_evidence},
            "label": (_label(judgment, label_class, sub_reason) if isinstance(judgment, Mapping)
                      else {"class": label_class, "sub_reason": sub_reason}),
            "label_provenance": _provenance(example, index),
            **({"source_evidence_error": source_error} if source_error else {}),
            "group_keys": {"company_keys": _company_keys(company, judgment), "icp_fingerprint": fingerprint,
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


def build(examples: Iterable[Mapping[str, Any]], rules: Rules = Rules(), *,
          curation_metadata: Optional[dict] = None) -> Tuple[Dict[str, List[dict]], Dict[str, Any]]:
    """Return ({train, eval, analysis}: records) and the dataset manifest."""

    sources, records, privacy_filtered = [], [], 0
    for example in examples:
        reason = example_exclusion(example, rules)
        sources.append({"example_id": example.get("example_id"), "example_hash": example.get("example_hash"),
                        "run_id": example.get("run_id"), "round_id": example.get("round_id"),
                        "used": reason is None, "exclusion": reason})
        if reason is None:
            judgments = ({row.get("company_index"): row.get("judgment")
                          for row in example.get("companies") or () if isinstance(row, Mapping)}
                         if curation_metadata is not None else {})
            for record in records_from_example(example):
                # Redaction elsewhere in the run, mostly URL tokens in fetched
                # pages, never reaches these fields. One inside the record's
                # own input would train on a placeholder, so that record goes.
                if _REDACTION_RE.search(contracts.canonical_json(record["input"])):
                    privacy_filtered += 1
                    continue
                if curation_metadata is not None:
                    from lab_arena.training_curation import project_judgment
                    curation_metadata[record["record_id"]] = project_judgment(
                        judgments.get(record["meta"]["company_index"]))
                records.append(record)
    # Identical ICP requirements, evaluation date and claim: replays, retries
    # and cached judgments count once. The date is part of the input because
    # intent recency depends on it. Only verdicts can conflict; a claim judged
    # both ways is left out. Non-verdict outcomes group by their own class.
    groups: Dict[str, List[dict]] = {}
    for record in sorted(records, key=lambda item: (item["record_id"], item["meta"]["example_hash"])):
        claim = _hash({"icp": record["group_keys"]["icp_fingerprint"],
                       "evaluation_date": record["input"]["evaluation_date"],
                       "company": record["input"]["company"]})
        record["group_keys"]["claim_key"] = claim
        label = record["label"]["class"]
        groups.setdefault(claim + "|" + ("verdict" if label in TRAINABLE else label), []).append(record)
    kept, duplicates, conflicts = [], 0, 0
    for group in groups.values():
        if len({_label_signature(item["label"]) for item in group}) > 1:
            conflicts += 1
            continue
        kept.append(min(group, key=lambda item: (
            bool(_evidence_gap(item)) if item["label"]["class"] in TRAINABLE else False,
            item["record_id"], _hash(item))))
        duplicates += len(group) - 1
    cohorts = {(record["meta"]["scorer_policy_hash"], record["meta"]["scorer_image_digest"])
               for record in kept if record["label"]["class"] in TRAINABLE and not _evidence_gap(record)}
    if len(cohorts) > 1:
        raise ValueError("mixed_scorer_cohorts")
    clusters = _Clusters()
    for record in kept:
        clusters.union(record["group_keys"]["company_keys"] or [record["record_id"]])
    split_clusters = _Clusters()
    for record in kept:
        handles = list(record["group_keys"]["company_keys"]) or [record["record_id"]]
        if rules.eval_mode == "company_and_icp":
            handles.append("icp:" + record["group_keys"]["icp_fingerprint"])
        split_clusters.union(handles)
    def split_group(record):
        handles = record["group_keys"]["company_keys"] or [record["record_id"]]
        return split_clusters.find(handles[0])
    future_groups = {split_group(record) for record in kept
                     if rules.eval_after and _date(record["input"]["evaluation_date"])
                     and record["input"]["evaluation_date"] > rules.eval_after
                     and record["label"]["class"] in TRAINABLE and not _evidence_gap(record)}
    splits: Dict[str, List[dict]] = {"train": [], "eval": [], "analysis": []}
    evidence_exclusions: Dict[str, int] = {}
    for record in kept:
        handles = record["group_keys"]["company_keys"] or [record["record_id"]]
        cluster = clusters.find(handles[0])
        record["group_keys"]["company_cluster"] = cluster
        evidence_gap = _evidence_gap(record) if record["label"]["class"] in TRAINABLE else None
        if record["label"]["class"] not in TRAINABLE or evidence_gap:
            if evidence_gap:
                record["selection_exclusion"] = evidence_gap
                evidence_exclusions[evidence_gap] = evidence_exclusions.get(evidence_gap, 0) + 1
            splits["analysis"].append(record)
            continue
        group = split_group(record)
        if rules.eval_after:
            observed = _date(record["input"]["evaluation_date"])
            reason = ("temporal_date_missing_or_invalid" if observed is None else
                      "temporal_group_purged_from_train" if observed <= rules.eval_after and group in future_groups else None)
            if reason:
                record["selection_exclusion"] = reason
                splits["analysis"].append(record)
            else:
                splits["eval" if observed > rules.eval_after else "train"].append(record)
            continue
        bucket = int(hashlib.sha256((rules.seed + "\x00" + group).encode("utf-8")).hexdigest(), 16) % 100
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
    rule_document = asdict(rules)
    # Keep the established default dataset identity and record bytes.
    if rules.eval_mode == "company":
        rule_document.pop("eval_mode")
    if rules.eval_after is None:
        rule_document.pop("eval_after")
    manifest = {
        "schema_version": SCHEMA_VERSION, "record_schema_version": RECORD_SCHEMA_VERSION,
        "rules": rule_document, "training_permission": dict(TRAINING_PERMISSION),
        "sources": sorted(sources, key=lambda item: (str(item["example_id"]),
                                                    str(item["example_hash"]),
                                                    str(item["run_id"]), str(item["round_id"]))),
        "counts": {"examples_used": sum(source["used"] for source in sources),
                   "examples_excluded": exclusions, "records_by_split_and_label": counts,
                   "records_privacy_filtered": privacy_filtered,
                   "trainable_records_by_evidence_exclusion": evidence_exclusions,
                   "duplicate_records_merged": duplicates, "conflicting_label_groups_excluded": conflicts},
        "record_hashes": {name: [_hash(record) for record in items] for name, items in splits.items()},
    }
    manifest["dataset_version"] = _hash({key: manifest[key] for key in (
        "schema_version", "record_schema_version", "rules", "sources", "record_hashes")})
    manifest["evaluation_checks"] = evaluation_checks(splits, rules)
    return splits, manifest


def evaluation_checks(splits, rules):
    """Observed coverage only; this does not approve training or label quality."""
    def values(name, key):
        return {record["group_keys"][key] for record in splits[name]
                if record["group_keys"].get(key)}
    overlaps = {key: len(values("train", key) & values("eval", key))
                for key in ("company_cluster", "icp_fingerprint", "source_family")}
    ranges = {}
    for name, records in splits.items():
        dates = sorted(filter(None, (_date(record["input"]["evaluation_date"]) for record in records)))
        ranges[name] = {"min": dates[0] if dates else None, "max": dates[-1] if dates else None,
                        "missing_or_invalid": len(records) - len(dates)}
    reasons = ["empty_" + name for name in ("train", "eval") if not splits[name]]
    if overlaps["company_cluster"] or (rules.eval_mode == "company_and_icp" and overlaps["icp_fingerprint"]):
        reasons.append("group_overlap")
    return {"split_viable": not reasons, "insufficient_split_reasons": reasons,
            "date_ranges": ranges, "train_eval_group_overlap_counts": overlaps,
            "source_family_counts": {name: len(values(name, "source_family")) for name in splits}}


def review_sample(splits, size, seed="leadpoet-arena-review-v1"):
    """Deterministic round-robin strata; originals are referenced, never edited."""
    if type(size) is not int or size < 0:
        raise ValueError("review sample size must be nonnegative")
    strata = {}
    for split, records in splits.items():
        for record in records:
            kinds = tuple(sorted({item["content_kind"] for item in record["input"]["source_evidence"]}))
            key = (split, record["label"]["class"], kinds)
            strata.setdefault(key, []).append(record)
    for records in strata.values():
        records.sort(key=lambda record: _hash([seed, record["record_id"], _hash(record)]))
    selected = []
    keys = sorted(strata)
    while keys and len(selected) < size:
        remaining = []
        for key in keys:
            if len(selected) >= size:
                break
            record = strata[key].pop(0)
            selected.append({"record_id": record["record_id"], "record_hash": _hash(record),
                             "record_file": key[0] + ".jsonl", "original_label": copy.deepcopy(record["label"]),
                             "label_provenance": copy.deepcopy(record["label_provenance"]), "input_provenance": copy.deepcopy(record["meta"]),
                             "evidence_kinds": list(key[2]), "human_review_status": "unreviewed",
                             "reviewed_label": None, "review_reason": None, "reviewer": None})
            if strata[key]:
                remaining.append(key)
        keys = remaining
    return selected, {"requested": size, "selected": len(selected), "available_strata": len(strata),
                      "covered_strata": len({(item["record_file"], item["original_label"]["class"],
                                               tuple(item["evidence_kinds"])) for item in selected}),
                      "human_review_status": "unreviewed"}


__all__ = [
    "RECORD_SCHEMA_VERSION", "Rules", "SCHEMA_VERSION", "TRAINABLE", "agent_icp", "build", "classify",
    "company_keys", "example_exclusion", "records_from_example",
]
