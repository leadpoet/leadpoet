"""Per-company accepted-judgment identities and evidence envelopes.

The gateway builds reusable identities from the exact normalized input seen by
the company-quality scorer.  Company position and run metadata stay in refs,
outside the reusable scope.  A trusted claim assigns the authority slot; the
validator may return a raw company judgment only for a miss in that claim.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from lab_arena import contact_policy, contracts, quality_policy


CACHE_SCOPE_SCHEMA_VERSION = "leadpoet.lab_arena.company_judgment_scope.v1"
COMPANY_REF_SCHEMA_VERSION = "leadpoet.lab_arena.company_judgment_ref.v1"
LEASE_SCHEMA_VERSION = "leadpoet.lab_arena.company_judgment_lease.v1"
EVIDENCE_SCHEMA_VERSION = "leadpoet.lab_arena.company_judgment_evidence.v1"

_BASE_SCORING_INPUT_FIELDS = (
    "schema_version",
    "scored_run_id",
    "icp",
    "companies",
    "scorer_policy",
    "evaluation_date",
)
_CONTEXT_FIELDS = frozenset({
    "company_index",
    "company_qualified",
    "duplicate_company",
    "duplicate_of_index",
})


class CompanyJudgmentError(ValueError):
    """A company cache identity, lease, or evidence envelope is malformed."""


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(contracts.canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise CompanyJudgmentError(
            "company judgment value is not canonical JSON"
        ) from exc


def _hash(value: Any) -> str:
    return contracts.document_hash(value)


def _require_hash(value: Any, field: str) -> str:
    text = str(value or "")
    if (
        len(text) != 71
        or not text.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise CompanyJudgmentError(f"{field} is invalid")
    return text


def _input_fields(policy: Mapping[str, Any]) -> tuple[str, ...]:
    fields = _BASE_SCORING_INPUT_FIELDS
    if contact_policy.scorer_enabled(policy):
        fields += ("contact_source_evidence",)
    return fields


def effective_company_inputs(document: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Project order-independent single-company inputs consumed by the scorer."""

    if not isinstance(document, Mapping):
        raise CompanyJudgmentError("scoring input must be an object")
    policy = document.get("scorer_policy")
    if not isinstance(policy, Mapping) or not quality_policy.scorer_enabled(policy):
        raise CompanyJudgmentError("company quality scorer policy is required")
    if not contact_policy.integrity_adapter(
        str(policy.get("scoring_adapter_version") or "")
    ):
        raise CompanyJudgmentError("company quality requires integrity scoring")
    if tuple(document.keys()) != _input_fields(policy):
        raise CompanyJudgmentError("scoring input fields or field order changed")
    if document.get("schema_version") != "leadpoet.lab_arena.scoring_input.v1":
        raise CompanyJudgmentError("scoring input schema changed")
    companies = document.get("companies")
    icp = document.get("icp")
    if (
        not isinstance(companies, Sequence)
        or isinstance(companies, (str, bytes, bytearray))
        or not isinstance(icp, Mapping)
    ):
        raise CompanyJudgmentError("scoring companies or ICP are invalid")

    from qualification.scoring.competition import effective_competition_input

    try:
        effective = effective_competition_input(
            companies,
            icp,
            contacts_required=contact_policy.scorer_enabled(policy),
            contact_source_evidence=document.get("contact_source_evidence"),
            company_quality=True,
        )
    except (TypeError, ValueError) as exc:
        raise CompanyJudgmentError(
            "effective company scoring input is invalid"
        ) from exc
    rows = effective.get("companies")
    buyer_requirements = effective.get("icp")
    if not isinstance(rows, list) or not isinstance(buyer_requirements, Mapping):
        raise CompanyJudgmentError("effective company scoring input changed")
    copied_policy = _json_copy(policy)
    evaluation_date = str(document.get("evaluation_date") or "")
    if not evaluation_date:
        raise CompanyJudgmentError("evaluation date is missing")
    return [
        {
            "schema_version": "leadpoet.lab_arena.effective_company_input.v1",
            "company_quality_policy": quality_policy.POLICY,
            "buyer_requirements": _json_copy(buyer_requirements),
            "company": _json_copy(row),
            "scorer_policy": copied_policy,
            "evaluation_date": evaluation_date,
        }
        for row in rows
    ]


def build_company_scopes(
    *,
    scoring_input: Mapping[str, Any],
    round_id: str,
    network_name: str,
    netuid: int,
    scorer_image_digest: str,
    scorer_image_reference: str,
    integrity_policy: str,
    company_quality_policy: str,
) -> list[Dict[str, Any]]:
    """Build reusable company refs while keeping destination indexes outside keys."""

    if integrity_policy != "arena_integrity_v1":
        raise CompanyJudgmentError("company judgments require integrity policy")
    if company_quality_policy != quality_policy.POLICY:
        raise CompanyJudgmentError("unsupported company quality policy")
    identity = {
        "round_id": str(round_id),
        "network_name": str(network_name),
        "netuid": int(netuid),
        "scorer_image_digest": str(scorer_image_digest),
        "scorer_image_reference": str(scorer_image_reference),
    }
    if (
        not identity["round_id"]
        or not identity["network_name"]
        or identity["netuid"] < 0
        or not identity["scorer_image_digest"]
        or not identity["scorer_image_reference"]
    ):
        raise CompanyJudgmentError("company judgment scope is incomplete")

    refs = []
    for company_index, effective in enumerate(effective_company_inputs(scoring_input)):
        company_input_hash = _hash(effective)
        scope = {
            "schema_version": CACHE_SCOPE_SCHEMA_VERSION,
            "integrity_policy": integrity_policy,
            "company_quality_policy": company_quality_policy,
            **identity,
            "evaluation_date": effective["evaluation_date"],
            "company_input_hash": company_input_hash,
        }
        cache_key = _hash(scope)
        refs.append({
            "schema_version": COMPANY_REF_SCHEMA_VERSION,
            "company_index": company_index,
            "cache_key": cache_key,
            "company_input_hash": company_input_hash,
            "scope_doc": {**scope, "cache_key": cache_key},
        })
    return refs


def validate_company_ref(document: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {
        "schema_version", "company_index", "cache_key", "company_input_hash",
        "scope_doc",
    }:
        raise CompanyJudgmentError("company judgment ref fields changed")
    copied = _json_copy(document)
    if copied["schema_version"] != COMPANY_REF_SCHEMA_VERSION:
        raise CompanyJudgmentError("company judgment ref schema changed")
    if type(copied["company_index"]) is not int or copied["company_index"] < 0:
        raise CompanyJudgmentError("company judgment index is invalid")
    cache_key = _require_hash(copied["cache_key"], "cache key")
    input_hash = _require_hash(copied["company_input_hash"], "company input hash")
    scope = copied["scope_doc"]
    if not isinstance(scope, Mapping):
        raise CompanyJudgmentError("company judgment scope is invalid")
    scope_without_key = dict(scope)
    scope_cache_key = scope_without_key.pop("cache_key", None)
    if scope_cache_key != cache_key or _hash(scope_without_key) != cache_key:
        raise CompanyJudgmentError("company judgment scope hash mismatch")
    if scope.get("company_input_hash") != input_hash:
        raise CompanyJudgmentError("company judgment input hash mismatch")
    return copied


def raw_judgment_is_cacheable(raw_judgment: Mapping[str, Any]) -> bool:
    """Accept substantive positive or negative results, never provider errors."""

    if not isinstance(raw_judgment, Mapping) or not raw_judgment:
        return False
    if _CONTEXT_FIELDS.intersection(raw_judgment):
        return False
    try:
        copied = _json_copy(raw_judgment)
    except CompanyJudgmentError:
        return False
    score = copied.get("final_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0.0 <= float(score) <= 100.0
    ):
        return False
    from qualification.scoring.arena_integrity import (
        company_fit_verified,
        verified_identity_receipt,
    )
    from qualification.scoring.competition import (
        company_quality_identity_verified,
        scorer_breakdown_has_retryable_infrastructure_failure,
    )

    if company_fit_verified(copied) and not company_quality_identity_verified(
        verified_identity_receipt(copied.get("verifier_gate_receipts"))
    ):
        return False

    return not scorer_breakdown_has_retryable_infrastructure_failure(
        copied, integrity_policy=True
    )


def _validate_lease_item(
    item: Mapping[str, Any], *, hit: bool
) -> Dict[str, Any]:
    common = {
        "company_index", "cache_key", "company_input_hash", "authority_slot"
    }
    expected = common | ({"evidence_hash", "evidence_doc"} if hit else set())
    if not isinstance(item, Mapping) or set(item) != expected:
        raise CompanyJudgmentError("company judgment lease item fields changed")
    copied = _json_copy(item)
    if type(copied["company_index"]) is not int or copied["company_index"] < 0:
        raise CompanyJudgmentError("company judgment lease index is invalid")
    _require_hash(copied["cache_key"], "cache key")
    _require_hash(copied["company_input_hash"], "company input hash")
    if type(copied["authority_slot"]) is not int or copied["authority_slot"] < 0:
        raise CompanyJudgmentError("company judgment authority slot is invalid")
    if hit:
        validate_evidence_snapshot(
            copied["evidence_doc"],
            cache_key=copied["cache_key"],
            company_input_hash=copied["company_input_hash"],
            authority_slot=copied["authority_slot"],
            evidence_hash=_require_hash(copied["evidence_hash"], "evidence hash"),
        )
    return copied


def validate_lease_context(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the cache material attached by the trusted claim RPC."""

    if not isinstance(document, Mapping) or set(document) != {
        "schema_version", "hits", "misses"
    }:
        raise CompanyJudgmentError("company judgment lease fields changed")
    if document.get("schema_version") != LEASE_SCHEMA_VERSION:
        raise CompanyJudgmentError("company judgment lease schema changed")
    if not isinstance(document.get("hits"), list) or not isinstance(
        document.get("misses"), list
    ):
        raise CompanyJudgmentError("company judgment lease lists are invalid")
    hits = [_validate_lease_item(item, hit=True) for item in document["hits"]]
    misses = [_validate_lease_item(item, hit=False) for item in document["misses"]]
    all_items = sorted(hits + misses, key=lambda item: item["company_index"])
    indexes = [item["company_index"] for item in all_items]
    if indexes != list(range(len(indexes))):
        raise CompanyJudgmentError("company judgment lease indexes are incomplete")
    decisions: dict[str, tuple[str, int, str]] = {}
    for status, items in (("hit", hits), ("miss", misses)):
        for item in items:
            decision = (
                status, item["authority_slot"], item.get("evidence_hash", "")
            )
            previous = decisions.setdefault(item["cache_key"], decision)
            if previous != decision:
                raise CompanyJudgmentError(
                    "one company key has inconsistent lease decisions"
                )
    return {
        "schema_version": LEASE_SCHEMA_VERSION,
        "hits": hits,
        "misses": misses,
    }


def validate_new_company_judgments(
    rows: Sequence[Mapping[str, Any]], *, lease_context: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    """Bind one returned raw judgment to every unique cache miss."""

    lease = validate_lease_context(lease_context)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise CompanyJudgmentError("new company judgments must be a list")
    expected: dict[str, Dict[str, Any]] = {}
    for miss in lease["misses"]:
        existing = expected.get(miss["cache_key"])
        if existing is None or miss["company_index"] < existing["company_index"]:
            expected[miss["cache_key"]] = miss
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "company_index", "cache_key", "company_input_hash",
            "authority_slot", "raw_judgment",
        }:
            raise CompanyJudgmentError("new company judgment fields changed")
        copied = _json_copy(row)
        cache_key = _require_hash(copied["cache_key"], "cache key")
        miss = expected.get(cache_key)
        if (
            miss is None
            or cache_key in seen
            or copied["company_index"] != miss["company_index"]
            or copied["company_input_hash"] != miss["company_input_hash"]
            or copied["authority_slot"] != miss["authority_slot"]
            or not raw_judgment_is_cacheable(copied["raw_judgment"])
        ):
            raise CompanyJudgmentError(
                "new company judgment is not bound to a cacheable miss"
            )
        seen.add(cache_key)
        result.append(copied)
    if seen != set(expected):
        raise CompanyJudgmentError("new company judgments do not cover every miss")
    return sorted(result, key=lambda item: item["company_index"])


def build_evidence_snapshot(
    *,
    new_judgment: Mapping[str, Any],
    company_ref: Mapping[str, Any],
    source_score_run_id: str,
    source_scored_run_id: str,
    source_output_ref: str,
    source_output_hash: str,
    source_runner_hotkey: str,
    source_claim_request_id: str,
    source_claim_request_hash: str,
    source_lease_generation: int,
    source_completion_request_hash: str,
    runner_authority_exclusions: Sequence[str] | None,
) -> Dict[str, Any]:
    """Freeze one accepted raw verdict with its signed run provenance."""

    ref = validate_company_ref(company_ref)
    row = _json_copy(new_judgment)
    if set(row) != {
        "company_index", "cache_key", "company_input_hash", "authority_slot",
        "raw_judgment",
    }:
        raise CompanyJudgmentError("new company judgment fields changed")
    if (
        row["company_index"] != ref["company_index"]
        or row["cache_key"] != ref["cache_key"]
        or row["company_input_hash"] != ref["company_input_hash"]
        or type(row["authority_slot"]) is not int
        or row["authority_slot"] < 0
        or not raw_judgment_is_cacheable(row["raw_judgment"])
    ):
        raise CompanyJudgmentError("new company judgment ref mismatch")
    if (
        not isinstance(runner_authority_exclusions, Sequence)
        or isinstance(runner_authority_exclusions, (str, bytes, bytearray))
    ):
        raise CompanyJudgmentError("company judgment authority is incomplete")
    exclusions = sorted(set(runner_authority_exclusions))
    if (
        not exclusions
        or any(not isinstance(item, str) or not item for item in exclusions)
        or source_runner_hotkey not in exclusions
    ):
        raise CompanyJudgmentError("company judgment authority is incomplete")
    provenance = {
        "source_score_run_id": str(source_score_run_id),
        "source_scored_run_id": str(source_scored_run_id),
        "source_output_ref": str(source_output_ref),
        "source_output_hash": _require_hash(source_output_hash, "source output hash"),
        "source_runner_hotkey": str(source_runner_hotkey),
        "source_claim_request_id": str(source_claim_request_id),
        "source_claim_request_hash": _require_hash(
            source_claim_request_hash, "source claim request hash"
        ),
        "source_lease_generation": source_lease_generation,
        "source_completion_request_hash": _require_hash(
            source_completion_request_hash, "source completion request hash"
        ),
    }
    if (
        any(not provenance[key] for key in (
            "source_score_run_id", "source_scored_run_id", "source_output_ref",
            "source_runner_hotkey", "source_claim_request_id",
        ))
        or type(source_lease_generation) is not int
        or source_lease_generation < 1
    ):
        raise CompanyJudgmentError("company judgment provenance is incomplete")
    snapshot = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "cache_key": ref["cache_key"],
        "company_input_hash": ref["company_input_hash"],
        "authority_slot": row["authority_slot"],
        **provenance,
        "runner_authority_exclusions": exclusions,
        "raw_judgment": row["raw_judgment"],
    }
    return snapshot


def validate_evidence_snapshot(
    document: Mapping[str, Any], *, cache_key: str, company_input_hash: str,
    authority_slot: int, evidence_hash: str,
) -> Dict[str, Any]:
    """Validate immutable database evidence before exposing it in a lease."""

    fields = {
        "schema_version", "cache_key", "company_input_hash", "authority_slot",
        "source_score_run_id", "source_scored_run_id", "source_output_ref",
        "source_output_hash", "source_runner_hotkey", "source_claim_request_id",
        "source_claim_request_hash", "source_lease_generation",
        "source_completion_request_hash", "runner_authority_exclusions",
        "raw_judgment",
    }
    if not isinstance(document, Mapping) or set(document) != fields:
        raise CompanyJudgmentError("company judgment evidence fields changed")
    copied = _json_copy(document)
    if copied["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise CompanyJudgmentError("company judgment evidence schema changed")
    if (
        copied["cache_key"] != _require_hash(cache_key, "cache key")
        or copied["company_input_hash"]
        != _require_hash(company_input_hash, "company input hash")
        or copied["authority_slot"] != authority_slot
        or _hash(copied) != _require_hash(evidence_hash, "evidence hash")
    ):
        raise CompanyJudgmentError("company judgment evidence binding mismatch")
    for field in (
        "source_output_hash", "source_claim_request_hash",
        "source_completion_request_hash",
    ):
        _require_hash(copied[field], field)
    if (
        type(copied["source_lease_generation"]) is not int
        or copied["source_lease_generation"] < 1
        or any(not copied[field] for field in (
            "source_score_run_id", "source_scored_run_id", "source_output_ref",
            "source_runner_hotkey", "source_claim_request_id",
        ))
        or not raw_judgment_is_cacheable(copied["raw_judgment"])
    ):
        raise CompanyJudgmentError("company judgment evidence is incomplete")
    exclusions = copied["runner_authority_exclusions"]
    if (
        not isinstance(exclusions, list)
        or not exclusions
        or exclusions != sorted(set(exclusions))
        or any(not isinstance(item, str) or not item for item in exclusions)
        or copied["source_runner_hotkey"] not in exclusions
    ):
        raise CompanyJudgmentError("company judgment authority is incomplete")
    return copied
