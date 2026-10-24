"""Exact, gateway-owned identities for reusable Arena intent signal results."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from lab_arena import company_judgments, contracts
from qualification.scoring.signal_judgment_cache import reusable_terminal_judgment


POLICY = "signal_judgment_v1"
SCOPE_VERSION = "leadpoet.lab_arena.signal_judgment_scope.v1"
REFERENCE_VERSION = "leadpoet.lab_arena.signal_judgment_reference.v1"


class SignalJudgmentError(ValueError):
    """A signal identity or reusable result violates the frozen protocol."""


def _copy(value: Any) -> Any:
    try:
        return json.loads(contracts.canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise SignalJudgmentError("signal value is not canonical JSON") from exc


def _hash(value: Any) -> str:
    return contracts.document_hash(value)


def _require_hash(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:") or any(
        char not in "0123456789abcdef" for char in value[7:]
    ):
        raise SignalJudgmentError("signal hash is invalid")
    return value


def build_scope(
    *, scoring_input: Mapping[str, Any], company_index: int, signal_index: int,
    verified_company_identity: Mapping[str, Any],
    verified_funding_stage_context: list[Any], round_id: str, network_name: str,
    netuid: int, scorer_image_digest: str, scorer_image_reference: str,
    verified_source_pages: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Hash one effective signal; sibling submitted signals never enter its key."""

    if type(company_index) is not int or company_index < 0 or type(signal_index) is not int or signal_index < 0:
        raise SignalJudgmentError("signal position is invalid")
    policy = scoring_input.get("scorer_policy") if isinstance(scoring_input, Mapping) else None
    if not isinstance(policy, Mapping) or policy.get("signal_judgment_policy") != POLICY:
        raise SignalJudgmentError("signal policy is not frozen")
    if not isinstance(verified_company_identity, Mapping) or verified_company_identity.get("decision") != "match":
        raise SignalJudgmentError("verified company identity is missing")
    if not isinstance(verified_funding_stage_context, list):
        raise SignalJudgmentError("funding context must be a list")
    effective = company_judgments.effective_company_inputs(scoring_input)
    if company_index >= len(effective):
        raise SignalJudgmentError("company index is outside scoring input")
    company = effective[company_index]["company"]
    signals = company.get("intent_signals")
    # The signal judge reads these identity fields. Company fit, paragraph
    # review, and other submitted prose are evaluated independently.
    signal_company = {
        key: company.get(key) for key in
        ("company_name", "company_website", "company_linkedin")
    }
    if not isinstance(signals, list) or signal_index >= len(signals):
        raise SignalJudgmentError("signal index is outside effective evidence")
    signal = signals[signal_index]
    if not isinstance(signal, Mapping):
        raise SignalJudgmentError("effective signal is invalid")
    matched_index = signal.get("matched_icp_signal")
    buyer = effective[company_index]["buyer_requirements"]
    targets = buyer.get("intent_signals")
    types = buyer.get("intent_signal_evidence_types")
    if type(matched_index) is not int or not isinstance(targets, list) or not 0 <= matched_index < len(targets):
        raise SignalJudgmentError("signal target is invalid")
    evidence_type = str(types[matched_index] or "UNSPECIFIED").upper() if isinstance(types, list) and matched_index < len(types) else "UNSPECIFIED"
    context_urls: list[str] = []
    if evidence_type == "FUNDING":
        for context in verified_funding_stage_context:
            if not isinstance(context, Mapping):
                raise SignalJudgmentError("funding context item is invalid")
            for url in (context.get("url"), context.get("final_url")):
                if url is None:
                    continue
                if not isinstance(url, str) or not url.startswith("https://"):
                    raise SignalJudgmentError("funding context URL is invalid")
                if url not in context_urls:
                    context_urls.append(url)
        if len(context_urls) > 12:
            raise SignalJudgmentError("funding context is too large")
    if verified_source_pages is not None and not isinstance(verified_source_pages, Mapping):
        raise SignalJudgmentError("retained source pages must be a mapping")
    retained_context = ({"verified_source_pages": _copy(verified_source_pages)}
                        if verified_source_pages else {})
    input_doc = {
        "schema_version": "leadpoet.lab_arena.signal_judgment_input.v1",
        "buyer_requirements": buyer,
        "signal_company_identity": signal_company,
        "signal": signal,
        "verified_company_identity": _copy(verified_company_identity),
        "verified_funding_stage_context": _copy(verified_funding_stage_context),
        **retained_context,
        "scorer_policy": _copy(policy),
    }
    input_hash = _hash(input_doc)
    scope = {
        "schema_version": SCOPE_VERSION,
        "round_id": str(round_id), "network_name": str(network_name),
        "netuid": int(netuid),
        "scorer_image_digest": str(scorer_image_digest),
        "scorer_image_reference": str(scorer_image_reference),
        "evaluation_date": str(effective[company_index]["evaluation_date"]),
        "signal_judgment_policy": POLICY,
        "signal_input_hash": input_hash,
        "verified_company_identity": _copy(verified_company_identity),
        "verified_funding_stage_context": _copy(verified_funding_stage_context),
        **retained_context,
        "expected_urls": [str(signal.get("url") or "")],
        "expected_context_urls": context_urls,
        "expected_index": matched_index,
        "expected_evidence_type": evidence_type,
    }
    if not all(scope[key] for key in ("round_id", "network_name", "scorer_image_digest", "scorer_image_reference", "evaluation_date", "expected_urls")):
        raise SignalJudgmentError("signal scope is incomplete")
    return {**scope, "cache_key": _hash(scope)}


def reference(scope: Mapping[str, Any], authority_slot: int) -> dict[str, Any]:
    if type(authority_slot) is not int or authority_slot < 0:
        raise SignalJudgmentError("authority slot is invalid")
    return {
        "schema_version": REFERENCE_VERSION,
        "cache_key": _require_hash(scope.get("cache_key")),
        "signal_input_hash": _require_hash(scope.get("signal_input_hash")),
        "authority_slot": authority_slot,
    }


def validate_reference(value: Any, scope: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "cache_key", "signal_input_hash", "authority_slot"
    }:
        raise SignalJudgmentError("signal reference fields changed")
    copied = _copy(value)
    expected = reference(scope, copied["authority_slot"])
    if copied != expected:
        raise SignalJudgmentError("signal reference differs from frozen scope")
    return copied


def validate_judgment(value: Any, scope: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SignalJudgmentError("signal judgment is not an object")
    copied = _copy(value)
    if not reusable_terminal_judgment(
        copied, expected_urls=scope["expected_urls"],
        expected_index=scope["expected_index"],
        expected_evidence_type=scope["expected_evidence_type"],
        allowed_context_urls=scope.get("expected_context_urls", []),
    ):
        raise SignalJudgmentError("signal judgment is not terminal and reusable")
    return copied
