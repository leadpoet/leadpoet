"""In-enclave entrypoint for Research Lab allocation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Mapping

SCORING_EXECUTOR_SCHEMA_VERSION = "leadpoet.gateway_scoring_executor.v1"
OP_RESEARCH_LAB_ALLOCATION = "research_lab_allocation"

SUPPORTED_OPERATIONS = frozenset({OP_RESEARCH_LAB_ALLOCATION})

# Only values that can change scoring behavior are committed here. Provider
# credentials and infrastructure locations are intentionally excluded; their
# accepted responses are committed separately through evidence roots.
SCORING_CONFIG_ENV_NAMES = (
    "INTENT_GATE_STRICT_JUDGE_ENABLED",
    "INTENT_THREE_STAGE_S1_MODEL",
    "INTENT_THREE_STAGE_S3_MODEL",
    "INTENT_VERIFIER_REVIEW_AS_ACCEPT",
    "QUAL_INTENT_CACHE_TTL_DAYS",
    "QUAL_INTENT_CONFIDENCE_THRESHOLD",
    "QUAL_INTENT_SIGNAL_DECAY_25_PCT_MONTHS",
    "QUAL_INTENT_SIGNAL_DECAY_50_PCT_MONTHS",
    "QUAL_LEADS_PER_ICP",
    "QUAL_MAX_COST_PER_LEAD_USD",
    "QUAL_MAX_TIME_PER_LEAD_SECONDS",
)

SCORING_SECRET_ENV_NAMES = (
    "EXA_API_KEY",
    "FULFILLMENT_OPENROUTER_API_KEY",
    "GITHUB_TOKEN",
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "QUALIFICATION_SCRAPINGDOG_API_KEY",
    "SCRAPINGDOG_API_KEY",
)
SCORING_RUNTIME_ENV_NAMES = tuple(
    sorted(set(SCORING_CONFIG_ENV_NAMES + SCORING_SECRET_ENV_NAMES))
)
MAX_RUNTIME_ENV_VALUE_BYTES = 16 * 1024
MAX_RUNTIME_ENV_TOTAL_BYTES = 128 * 1024


class ScoringExecutorError(ValueError):
    """Raised when a scoring operation or payload is unsupported."""


class ScoringExecutionResult:
    """Internal result plus evidence roots derived inside the enclave."""

    def __init__(self, result: Mapping[str, Any], evidence_roots: Mapping[str, str]) -> None:
        self.result = dict(result)
        self.evidence_roots = dict(evidence_roots)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScoringExecutorError("scoring value is not canonical JSON") from exc


def _manifest_configuration_env_names() -> tuple:
    # The import manifest records every literal env reference reachable from
    # broad shared modules, including unrelated gateway/validator operations.
    # Only this reviewed list can alter an enclave scoring operation.
    return SCORING_RUNTIME_ENV_NAMES


def normalize_runtime_environment(values: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(values, Mapping) or set(values) != set(SCORING_RUNTIME_ENV_NAMES):
        raise ScoringExecutorError("scoring runtime environment fields do not match the schema")
    normalized = {}
    total_bytes = 0
    for name in SCORING_RUNTIME_ENV_NAMES:
        value = values.get(name)
        if value is None:
            normalized[name] = None
            continue
        if not isinstance(value, str) or "\x00" in value:
            raise ScoringExecutorError("scoring runtime environment value is invalid")
        encoded_size = len(value.encode("utf-8"))
        if encoded_size > MAX_RUNTIME_ENV_VALUE_BYTES:
            raise ScoringExecutorError("scoring runtime environment value exceeds limit")
        total_bytes += encoded_size
        normalized[name] = value
    if total_bytes > MAX_RUNTIME_ENV_TOTAL_BYTES:
        raise ScoringExecutorError("scoring runtime environment exceeds total limit")
    return normalized


def runtime_environment_values() -> Dict[str, Any]:
    return {name: os.environ.get(name) for name in SCORING_RUNTIME_ENV_NAMES}


def configuration_snapshot(values: Mapping[str, Any] = None) -> Dict[str, Any]:
    source = (
        normalize_runtime_environment(values)
        if values is not None
        else runtime_environment_values()
    )
    environment = {}
    for name in _manifest_configuration_env_names():
        value = source.get(name)
        if name in SCORING_SECRET_ENV_NAMES:
            environment[name] = {
                "configured": bool(value),
                "value_sha256": (
                    "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
                    if value
                    else None
                ),
            }
        else:
            environment[name] = value
    from gateway.tee.egress_policy import destination_policy_hash

    return {
        "schema_version": SCORING_EXECUTOR_SCHEMA_VERSION,
        "environment": environment,
        "egress_policy_hash": destination_policy_hash(),
    }


def configuration_hash(values: Mapping[str, Any] = None) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(configuration_snapshot(values))).hexdigest()


def purpose_allowed_for_operation(operation: str, purpose: str) -> bool:
    allowed = {OP_RESEARCH_LAB_ALLOCATION: {"research_lab.allocation.v1"}}
    return purpose in allowed.get(operation, set())


async def execute_scoring_operation(operation: str, payload: Mapping[str, Any]) -> Any:
    """Execute one existing pure/scoring entrypoint without changing its logic."""

    if operation not in SUPPORTED_OPERATIONS:
        raise ScoringExecutorError("unsupported scoring operation")
    if not isinstance(payload, Mapping):
        raise ScoringExecutorError("scoring payload must be an object")

    from leadpoet_verifier.economics import allocate_research_lab_epoch

    policy = payload.get("policy")
    reimbursements = payload.get("active_reimbursement_obligations")
    champions = payload.get("active_champion_obligations")
    source_add = payload.get("active_source_add_obligations", [])
    fallback_reimbursements = payload.get(
        "fallback_reimbursement_obligations",
        [],
    )
    if not isinstance(policy, Mapping):
        raise ScoringExecutorError("policy must be an object")
    if (
        not isinstance(reimbursements, list)
        or not isinstance(champions, list)
        or not isinstance(source_add, list)
        or not isinstance(fallback_reimbursements, list)
    ):
        raise ScoringExecutorError("allocation obligations must be lists")
    allocation = allocate_research_lab_epoch(
        int(payload.get("epoch", -1)),
        policy,
        reimbursements,
        champions,
        active_source_add_obligations=source_add,
        fallback_reimbursement_obligations=fallback_reimbursements,
    )
    allocation_hash = str(allocation.get("allocation_hash") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", allocation_hash):
        raise ScoringExecutorError("allocation hash is invalid")
    return ScoringExecutionResult(
        {"allocation": allocation},
        {"allocation": allocation_hash},
    )
