"""Provider capability catalog used by SOURCE_ADD and provider transport."""

from __future__ import annotations

import base64

import binascii

from dataclasses import dataclass

import hashlib

import json

import logging

import math

import os

import re

import threading

import time

import urllib.parse

import urllib.request

from typing import Any, Callable, Mapping, Sequence

from gateway.research_lab.source_add_execution_plan import SourceAddExecutionPlanError, bind_source_add_execution_plan_to_probes, is_supported_source_add_execution_plan, normalize_source_add_execution_plan

from research_lab.canonical import sha256_json

logger = logging.getLogger(__name__)

CAPABILITY_CATALOG_ENABLED_ENV = "RESEARCH_LAB_PROVIDER_CAPABILITY_CATALOG_ENABLED"

CAPABILITY_ENFORCEMENT_ENV = "RESEARCH_LAB_PROVIDER_CAPABILITY_ENFORCEMENT"

CAPABILITY_REFRESH_SECONDS_ENV = "RESEARCH_LAB_PROVIDER_CAPABILITY_REFRESH_SECONDS"

MODEL_CATALOG_TTL_SECONDS_ENV = "RESEARCH_LAB_OPENROUTER_MODEL_CATALOG_TTL_SECONDS"

_TRUTHY = {"1", "true", "yes", "on"}

_VALID_AUTH_KINDS = {"header", "query", "bearer", "none"}

_VALID_METHODS = {"GET", "POST"}

_VALID_ENFORCEMENT = {"observe", "enforce"}

_SECRET_MARKERS = (
    "sk-or-",
    "sb_secret",
    "service_role",
    "raw_secret",
    "raw_credential",
    "hidden_prompt",
    "provider_output",
    "request_body",
    "response_body",
    "page_content",
    "raw_content",
    "judge_prompt",
    "private_manifest",
    "private_repo",
)

_ENV_REF_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")

_ROUTING_FEATURE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,95}$")

_SOURCE_ADD_BINDING_MANIFEST_SCHEMA_VERSION = (
    "leadpoet.intent-source-binding-manifest:v1"
)

_SOURCE_ADD_REGISTRATION_FIELDS = (
    "provider_id",
    "stage",
    "revision",
    "manifest_sha256",
    "execution_mode",
    "priority",
    "capabilities",
    "idempotency",
    "cost_class",
    "unit_cost",
    "max_calls",
    "max_results",
    "timeout_seconds",
    "intent_categories",
    "evidence_types",
    "category_contracts",
    "binding_requirements",
    "best_for",
    "avoid_when",
    "best_for_description",
    "avoid_when_description",
)

_SOURCE_ADD_EXECUTION_PLAN_FIELD = "execution_plan_identity"

_SOURCE_ADD_EXECUTION_MODES = frozenset({"invoke", "observe", "virtual"})

_SOURCE_ADD_IDEMPOTENCY_MODES = frozenset(
    {"idempotent", "resume_safe", "non_idempotent"}
)

_SOURCE_ADD_COST_CLASSES = frozenset({"free", "metered", "paid"})

_SOURCE_ADD_V8_PLANNER_FIELDS = (
    "stage",
    "execution_mode",
    "priority",
    "capabilities",
    "idempotency",
    "cost_class",
    "unit_cost",
    "max_calls",
    "max_results",
    "timeout_seconds",
    "intent_categories",
    "evidence_types",
    "category_contracts",
    "binding_requirements",
    "best_for_features",
    "avoid_when_features",
    "best_for",
    "avoid_when",
    _SOURCE_ADD_EXECUTION_PLAN_FIELD,
)

def _source_add_model_string_tuple(
    values: Any,
    *,
    field_name: str,
    maximum: int,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Mirror the retained v8 ``routing.contracts._string_tuple`` behavior.

    This compatibility adapter is intentionally mechanical. Parity tests
    execute the retained v8 fixtures.
    """

    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a literal sequence")
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not _ROUTING_FEATURE_RE.fullmatch(text):
            raise ValueError(f"{field_name} contains an invalid value")
        if text not in output:
            output.append(text)
    if len(output) > maximum or (not allow_empty and not output):
        raise ValueError(f"{field_name} is out of bounds")
    return tuple(output)

def _source_add_manifest_string_tuple(
    values: Any,
    *,
    field_name: str,
    maximum: int,
) -> tuple[str, ...]:
    """Normalize a bounded SOURCE_ADD string sequence."""

    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a literal sequence")
    normalized = tuple(
        dict.fromkeys(
            str(value or "").strip()
            for value in values
            if str(value or "").strip()
        )
    )
    if (
        not normalized
        or len(normalized) > maximum
        or any(len(value) > 160 for value in normalized)
    ):
        raise ValueError(f"{field_name} is out of bounds")
    return normalized

def _source_add_bounded_int(
    value: Any,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} is out of bounds")
    return value

def _source_add_bounded_float(
    value: Any,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
    precision: int | None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{field_name} is out of bounds")
    return round(number, precision) if precision is not None else number

def _source_add_description(
    value: Any,
    *,
    field_name: str,
    default: str,
) -> str:
    text = " ".join(str(value or "").split()) or default
    if len(text) > 500:
        raise ValueError(f"{field_name} exceeds 500 characters")
    return text

def _source_add_binding_manifest(
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    stage = str(normalized["stage"])
    provider_id = str(normalized["provider_id"])
    manifest = {
        "schema_version": _SOURCE_ADD_BINDING_MANIFEST_SCHEMA_VERSION,
        "tool_id": (
            "candidate" if stage == "candidate_acquisition" else "intent"
        )
        + ".source_add."
        + provider_id,
        "provider_id": provider_id,
        "stage": stage,
        "execution_mode": normalized["execution_mode"],
        "cost_class": normalized["cost_class"],
        "unit_cost": normalized["unit_cost"],
        "max_calls": normalized["max_calls"],
        "max_results": normalized["max_results"],
        "timeout_seconds": normalized["timeout_seconds"],
        "capabilities": list(normalized["capabilities"]),
        "intent_categories": list(normalized["intent_categories"]),
        "evidence_types": list(normalized["evidence_types"]),
        "category_contracts": [
            {
                "category": contract["category"],
                "capabilities": list(contract["capabilities"]),
                "evidence_types": list(contract["evidence_types"]),
                "requirements": list(contract["requirements"]),
            }
            for contract in sorted(
                normalized["category_contracts"],
                key=lambda item: str(item["category"]),
            )
        ],
        "binding_requirements": list(normalized["binding_requirements"]),
    }
    execution_plan = normalized.get(_SOURCE_ADD_EXECUTION_PLAN_FIELD)
    if execution_plan:
        manifest[_SOURCE_ADD_EXECUTION_PLAN_FIELD] = dict(execution_plan)
    return manifest

def _source_add_binding_manifest_digest(
    normalized: Mapping[str, Any],
) -> str:
    rendered = json.dumps(
        _source_add_binding_manifest(normalized),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

def _normalize_source_add_v8_registration(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one SOURCE_ADD provider registration."""

    normalized = dict(value)
    provider_id = str(normalized.get("provider_id") or "").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{1,79}", provider_id):
        raise ValueError("provider_id is invalid")
    stage = normalized.get("stage")
    if stage not in {"candidate_acquisition", "intent_evidence"}:
        raise ValueError("stage is invalid")
    normalized["provider_id"] = provider_id
    normalized["stage"] = stage
    normalized["priority"] = _source_add_bounded_int(
        normalized.get("priority"),
        field_name="priority",
        minimum=0,
        maximum=10_000,
    )
    normalized["capabilities"] = _source_add_model_string_tuple(
        normalized.get("capabilities", ()),
        field_name="capabilities",
        maximum=32,
        allow_empty=False,
    )
    execution_mode = normalized.get("execution_mode", "invoke")
    if execution_mode not in _SOURCE_ADD_EXECUTION_MODES:
        raise ValueError("execution_mode is invalid")
    normalized["execution_mode"] = execution_mode
    idempotency = normalized.get("idempotency", "idempotent")
    if idempotency not in _SOURCE_ADD_IDEMPOTENCY_MODES:
        raise ValueError("idempotency is invalid")
    normalized["idempotency"] = idempotency
    cost_class = normalized.get("cost_class", "metered")
    if cost_class not in _SOURCE_ADD_COST_CLASSES:
        raise ValueError("cost_class is invalid")
    normalized["cost_class"] = cost_class
    unit_cost = _source_add_bounded_float(
        normalized.get("unit_cost", 0.01),
        field_name="unit_cost",
        minimum=0.0,
        maximum=10_000.0,
        precision=None,
    )
    if (
        cost_class == "free" and unit_cost != 0.0
    ) or (
        cost_class != "free" and unit_cost <= 0.0
    ):
        raise ValueError("cost_class and unit_cost differ")
    normalized["unit_cost"] = round(unit_cost, 6)
    # Reject a positive cost that rounds to zero at the stored precision.
    if cost_class != "free" and normalized["unit_cost"] <= 0.0:
        raise ValueError("cost_class and unit_cost differ after normalization")
    normalized["max_calls"] = _source_add_bounded_int(
        normalized.get("max_calls", 1),
        field_name="max_calls",
        minimum=1,
        maximum=10_000,
    )
    normalized["max_results"] = _source_add_bounded_int(
        normalized.get("max_results", 1),
        field_name="max_results",
        minimum=1,
        maximum=100_000,
    )
    normalized["timeout_seconds"] = _source_add_bounded_float(
        normalized.get("timeout_seconds", 30.0),
        field_name="timeout_seconds",
        minimum=0.1,
        maximum=3_600.0,
        precision=3,
    )
    raw_categories = normalized.get("intent_categories", ())
    if not isinstance(raw_categories, (list, tuple)):
        raise ValueError("intent_categories must be a literal sequence")
    categories = tuple(
        dict.fromkeys(
            str(item or "").strip().upper()
            for item in raw_categories
            if str(item or "").strip()
        )
    )
    if len(categories) > 64 or any(len(item) > 80 for item in categories):
        raise ValueError("intent_categories are out of bounds")
    normalized["intent_categories"] = categories
    normalized["evidence_types"] = _source_add_model_string_tuple(
        normalized.get("evidence_types", ()),
        field_name="evidence_types",
        maximum=24,
        allow_empty=True,
    )
    default_best_for = (
        ("icp.structured_eligible",)
        if stage == "candidate_acquisition"
        else ("intent.general",)
    )
    raw_best_for = normalized.get("best_for") or default_best_for
    normalized["best_for"] = _source_add_model_string_tuple(
        raw_best_for,
        field_name="best_for",
        maximum=32,
        allow_empty=False,
    )
    normalized["avoid_when"] = _source_add_model_string_tuple(
        normalized.get("avoid_when", ()),
        field_name="avoid_when",
        maximum=32,
        allow_empty=True,
    )
    source_add_best_description = (
        "Approved SOURCE_ADD company-discovery provider for structured ICP "
        "acquisition."
        if stage == "candidate_acquisition"
        else "Approved SOURCE_ADD provider for company-scoped intent-evidence "
        "discovery."
    )
    raw_best_description = normalized.get("best_for_description")
    if not raw_best_description:
        raw_best_description = source_add_best_description
    normalized["best_for_description"] = _source_add_description(
        raw_best_description,
        field_name="best_for_description",
        default=(
            "Use when the route context matches: "
            + ", ".join(normalized["best_for"])
        ),
    )
    raw_avoid_description = normalized.get("avoid_when_description")
    if not raw_avoid_description:
        raw_avoid_description = (
            "Avoid when the consumer binding is unavailable, unhealthy, "
            "outside its approved categories, or over budget."
        )
    normalized["avoid_when_description"] = _source_add_description(
        raw_avoid_description,
        field_name="avoid_when_description",
        default=(
            "No additional structured exclusions; availability, category, "
            "budget, and policy gates remain authoritative."
        ),
    )

    raw_contracts = normalized.get("category_contracts", ())
    if not isinstance(raw_contracts, (list, tuple)):
        raise ValueError("category_contracts must be a literal sequence")
    category_contracts: list[dict[str, Any]] = []
    for raw_contract in raw_contracts:
        if not isinstance(raw_contract, Mapping) or set(raw_contract) != {
            "category",
            "capabilities",
            "evidence_types",
            "requirements",
        }:
            raise ValueError("category contract fields differ from the contract")
        category = str(raw_contract.get("category") or "").strip().upper()
        if not category or len(category) > 80:
            raise ValueError("category contract is invalid")
        category_contracts.append(
            {
                "category": category,
                "capabilities": _source_add_manifest_string_tuple(
                    raw_contract.get("capabilities"),
                    field_name="category capabilities",
                    maximum=24,
                ),
                "evidence_types": _source_add_manifest_string_tuple(
                    raw_contract.get("evidence_types"),
                    field_name="category evidence_types",
                    maximum=24,
                ),
                "requirements": _source_add_manifest_string_tuple(
                    raw_contract.get("requirements"),
                    field_name="category requirements",
                    maximum=24,
                ),
            }
        )
    contract_categories = tuple(item["category"] for item in category_contracts)
    if len(contract_categories) != len(set(contract_categories)):
        raise ValueError("duplicate category contract")
    if category_contracts and set(contract_categories) != set(categories):
        raise ValueError("category contracts must match intent_categories")
    if any(
        not set(item["capabilities"]) <= set(normalized["capabilities"])
        or not set(item["evidence_types"]) <= set(normalized["evidence_types"])
        for item in category_contracts
    ):
        raise ValueError("category contract exceeds the tool definition")
    normalized["category_contracts"] = tuple(category_contracts)

    raw_requirements = normalized.get("binding_requirements", ())
    normalized["binding_requirements"] = (
        _source_add_manifest_string_tuple(
            raw_requirements,
            field_name="binding requirements",
            maximum=32,
        )
        if raw_requirements
        else ()
    )
    raw_execution_plan = normalized.get(_SOURCE_ADD_EXECUTION_PLAN_FIELD)
    if raw_execution_plan:
        if not is_supported_source_add_execution_plan(raw_execution_plan):
            raise ValueError("execution_plan_identity schema is unsupported")
        try:
            normalized[_SOURCE_ADD_EXECUTION_PLAN_FIELD] = (
                normalize_source_add_execution_plan(
                    raw_execution_plan,
                    provider_id=provider_id,
                    tool_id=(
                        "candidate" if stage == "candidate_acquisition" else "intent"
                    )
                    + f".source_add.{provider_id}",
                    stage=stage,
                    execution_mode=normalized["execution_mode"],
                    intent_categories=normalized["intent_categories"],
                    max_calls=normalized["max_calls"],
                    max_results=normalized["max_results"],
                )
            )
        except SourceAddExecutionPlanError as exc:
            raise ValueError(str(exc)) from exc
    else:
        normalized.pop(_SOURCE_ADD_EXECUTION_PLAN_FIELD, None)
    computed_manifest = _source_add_binding_manifest_digest(normalized)
    configured_manifest = str(normalized.get("manifest_sha256") or "").strip()
    if configured_manifest and configured_manifest != computed_manifest:
        raise ValueError("manifest_sha256 does not match binding manifest")
    normalized["manifest_sha256"] = computed_manifest
    expected_revision = f"source-add-{computed_manifest[:12]}"
    configured_revision = str(normalized.get("revision") or "").strip()
    if configured_revision and configured_revision != expected_revision:
        raise ValueError("revision does not match binding manifest")
    normalized["revision"] = expected_revision
    result = {
        field: normalized[field]
        for field in _SOURCE_ADD_REGISTRATION_FIELDS
    }
    if _SOURCE_ADD_EXECUTION_PLAN_FIELD in normalized:
        result[_SOURCE_ADD_EXECUTION_PLAN_FIELD] = normalized[
            _SOURCE_ADD_EXECUTION_PLAN_FIELD
        ]
    return result

def normalize_source_add_planner_contract(
    provider_id: str,
    contract: Mapping[str, Any],
    *,
    estimated_cost_microusd_per_call: int = 0,
    probe_endpoints: Sequence[Mapping[str, Any]] = (),
    tested_probes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate a provisioning-time SOURCE_ADD planner contract.

    The admin API persists this JSON-safe projection in ``planner_summary``.
    Each explicit contract names one stage and uses only reviewed registration
    fields.
    """

    if not isinstance(contract, Mapping) or not contract:
        raise ValueError("routing_contract must be a non-empty object")
    allowed = (
        set(_SOURCE_ADD_REGISTRATION_FIELDS)
        | {_SOURCE_ADD_EXECUTION_PLAN_FIELD}
    ) - {
        "provider_id",
        "revision",
        "manifest_sha256",
    }
    unknown = set(contract) - allowed
    if unknown or "stage" not in contract:
        raise ValueError("routing_contract fields differ from the v8 contract")
    stage = contract.get("stage")
    candidate_stage = stage == "candidate_acquisition"
    try:
        microusd = max(0, int(estimated_cost_microusd_per_call))
    except (TypeError, ValueError) as exc:
        raise ValueError("estimated provider cost is invalid") from exc
    values = {
        "provider_id": provider_id,
        "stage": stage,
        "execution_mode": contract.get("execution_mode", "invoke"),
        "priority": contract.get(
            "priority", 80 if candidate_stage else 35
        ),
        "capabilities": contract.get(
            "capabilities",
            (
                "candidate.provider_discovery"
                if candidate_stage
                else "intent.provider_evidence",
            ),
        ),
        "idempotency": contract.get("idempotency", "idempotent"),
        "cost_class": contract.get(
            "cost_class", "metered" if microusd else "free"
        ),
        "unit_cost": contract.get(
            "unit_cost",
            (
                round(max(microusd / 1_000_000, 0.000001), 6)
                if microusd
                else 0.0
            ),
        ),
        "max_calls": contract.get("max_calls", 1),
        "max_results": contract.get(
            "max_results", 100 if candidate_stage else 1
        ),
        "timeout_seconds": contract.get(
            "timeout_seconds", 60.0 if candidate_stage else 30.0
        ),
        "intent_categories": contract.get("intent_categories", ()),
        "evidence_types": contract.get(
            "evidence_types",
            ("provider_database" if candidate_stage else "external",),
        ),
        "category_contracts": contract.get("category_contracts", ()),
        "binding_requirements": contract.get("binding_requirements", ()),
        "best_for": contract.get(
            "best_for",
            (
                ("icp.structured_eligible",)
                if candidate_stage
                else ("intent.general",)
            ),
        ),
        "avoid_when": contract.get("avoid_when", ()),
        "best_for_description": contract.get("best_for_description"),
        "avoid_when_description": contract.get("avoid_when_description"),
        _SOURCE_ADD_EXECUTION_PLAN_FIELD: contract.get(
            _SOURCE_ADD_EXECUTION_PLAN_FIELD
        ),
    }
    normalized = _normalize_source_add_v8_registration(values)
    execution_plan = normalized.get(_SOURCE_ADD_EXECUTION_PLAN_FIELD)
    if execution_plan:
        try:
            bind_source_add_execution_plan_to_probes(
                execution_plan,
                provider_id=provider_id,
                probe_endpoints=probe_endpoints,
                tested_probes=tested_probes,
            )
        except SourceAddExecutionPlanError as exc:
            raise ValueError(str(exc)) from exc
    result = {
        "stage": normalized["stage"],
        "execution_mode": normalized["execution_mode"],
        "priority": normalized["priority"],
        "capabilities": list(normalized["capabilities"]),
        "idempotency": normalized["idempotency"],
        "cost_class": normalized["cost_class"],
        "unit_cost": normalized["unit_cost"],
        "max_calls": normalized["max_calls"],
        "max_results": normalized["max_results"],
        "timeout_seconds": normalized["timeout_seconds"],
        "intent_categories": list(normalized["intent_categories"]),
        "evidence_types": list(normalized["evidence_types"]),
        "category_contracts": [
            {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in item.items()
            }
            for item in normalized["category_contracts"]
        ],
        "binding_requirements": list(normalized["binding_requirements"]),
        "best_for_features": list(normalized["best_for"]),
        "avoid_when_features": list(normalized["avoid_when"]),
        "best_for": normalized["best_for_description"],
        "avoid_when": normalized["avoid_when_description"],
    }
    if execution_plan:
        result[_SOURCE_ADD_EXECUTION_PLAN_FIELD] = dict(execution_plan)
    return result

def capability_catalog_enabled() -> bool:
    return str(os.getenv(CAPABILITY_CATALOG_ENABLED_ENV, "true") or "").strip().lower() in _TRUTHY

def capability_enforcement_mode() -> str:
    value = str(os.getenv(CAPABILITY_ENFORCEMENT_ENV, "observe") or "observe").strip().lower()
    return value if value in _VALID_ENFORCEMENT else "observe"

def capability_refresh_seconds() -> int:
    try:
        return max(10, int(os.getenv(CAPABILITY_REFRESH_SECONDS_ENV, "60") or 60))
    except (TypeError, ValueError):
        return 60

def model_catalog_ttl_seconds() -> int:
    try:
        return max(60, int(os.getenv(MODEL_CATALOG_TTL_SECONDS_ENV, "900") or 900))
    except (TypeError, ValueError):
        return 900

def _contains_secret_material(value: Any) -> bool:
    try:
        text = json.dumps(value, sort_keys=True, default=str).lower()
    except Exception:
        text = str(value).lower()
    return any(marker in text for marker in _SECRET_MARKERS)

def _slug(value: Any) -> str:
    return str(value or "").strip()

def _string_tuple(value: Any, *, limit: int = 100) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        values = list(value)
    else:
        values = []
    return tuple(str(item).strip() for item in values[:limit] if str(item or "").strip())

def _safe_route_path(value: Any, *, allow_prefix: bool = False) -> bool:
    path = str(value or "").strip()
    if allow_prefix and path == "/":
        return True
    if not path.startswith("/") or "?" in path or "#" in path or "\\" in path:
        return False
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        return False
    decoded = urllib.parse.unquote(path)
    if "\\" in decoded or any(part in {".", ".."} for part in decoded.split("/")):
        return False
    return not urllib.parse.urlsplit(path).netloc

def normalize_candidate_route(rest: str) -> tuple[str, str] | None:
    """Return normalized (path, query) or None for an unsafe proxy route."""

    raw = str(rest or "")
    if any(ord(char) < 32 or ord(char) == 127 for char in raw) or "\\" in raw:
        return None
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        return None
    path = parsed.path or "/"
    decoded = urllib.parse.unquote(path)
    decoded_query = urllib.parse.unquote(parsed.query)
    if not path.startswith("/") or "\\" in decoded:
        return None
    if any(part in {".", ".."} for part in decoded.split("/")):
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in decoded):
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in decoded_query):
        return None
    normalized = urllib.parse.quote(decoded, safe="/-._~")
    return normalized, parsed.query

def _route_doc_valid(route: Mapping[str, Any]) -> bool:
    method = str(route.get("method") or "").upper()
    path = str(route.get("path") or "")
    prefix = str(route.get("path_prefix") or "")
    if method not in _VALID_METHODS:
        return False
    if bool(path) == bool(prefix):
        return False
    return _safe_route_path(path or prefix, allow_prefix=bool(prefix))

def validate_capability_provider_doc(provider: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    provider_id = _slug(provider.get("id"))
    if not provider_id or not provider_id.replace("_", "").replace("-", "").isalnum():
        errors.append("provider_id_invalid")
    base_url = str(provider.get("base_url") or "").strip()
    parsed_base = urllib.parse.urlsplit(base_url)
    if parsed_base.scheme != "https" or not parsed_base.netloc or parsed_base.username or parsed_base.password:
        errors.append("provider_base_url_invalid")
    if parsed_base.query or parsed_base.fragment:
        errors.append("provider_base_url_must_not_include_query")
    auth_kind = str(provider.get("auth_kind") or "none").strip().lower()
    if auth_kind not in _VALID_AUTH_KINDS:
        errors.append("provider_auth_kind_invalid")
    auth_name = str(provider.get("auth_name") or "").strip()
    refs = _string_tuple(provider.get("credential_ref"), limit=12)
    if auth_kind in {"header", "query"} and not auth_name:
        errors.append("provider_auth_name_missing")
    if auth_kind != "none" and not refs:
        errors.append("provider_credential_ref_missing")
    if any(
        not ref
        or "=" in ref
        or ref != ref.strip()
        or (not ref.startswith("encrypted_ref:") and not _ENV_REF_RE.fullmatch(ref))
        for ref in refs
    ):
        errors.append("provider_credential_ref_invalid")
    try:
        if int(provider.get("per_day_quota") or 0) < 0:
            errors.append("provider_quota_invalid")
    except (TypeError, ValueError):
        errors.append("provider_quota_invalid")
    policy = provider.get("capability_policy")
    if policy is not None and not isinstance(policy, Mapping):
        errors.append("provider_capability_policy_invalid")
        policy = {}
    policy = dict(policy or {})
    routes = policy.get("routes") or []
    blocked_routes = policy.get("blocked_routes") or []
    if not isinstance(routes, list) or not all(isinstance(item, Mapping) and _route_doc_valid(item) for item in routes):
        errors.append("provider_routes_invalid")
    if not isinstance(blocked_routes, list) or not all(
        isinstance(item, Mapping) and _route_doc_valid(item) for item in blocked_routes
    ):
        errors.append("provider_blocked_routes_invalid")
    unlisted_methods = {str(item).upper() for item in _string_tuple(policy.get("unlisted_methods"), limit=4)}
    if policy.get("allow_unlisted_paths") is True and (
        not unlisted_methods or not unlisted_methods.issubset(_VALID_METHODS)
    ):
        errors.append("provider_unlisted_methods_invalid")
    model_policy = policy.get("model_policy") or {}
    if not isinstance(model_policy, Mapping):
        errors.append("provider_model_policy_invalid")
    elif str(model_policy.get("kind") or "none") not in {"none", "live_text_catalog"}:
        errors.append("provider_model_policy_kind_invalid")
    if _contains_secret_material(provider):
        errors.append("provider_doc_contains_forbidden_material")
    origin = str(provider.get("origin") or "")
    if origin not in {"builtin", "source_add", "legacy_fallback"}:
        errors.append("provider_origin_invalid")
    if origin == "source_add" and provider.get("reward_eligible") is not True:
        errors.append("source_add_provider_must_be_reward_eligible")
    if origin != "source_add" and provider.get("reward_eligible") is not False:
        errors.append("non_source_add_provider_must_not_be_reward_eligible")
    return sorted(set(errors))

@dataclass(frozen=True)
class EffectiveProviderCapabilities:
    providers: tuple[dict[str, Any], ...]
    capability_hash: str
    private_registry_hash: str = ""
    private_snapshot_loaded: bool = False
    source_add_provider_count: int = 0
    warning_codes: tuple[str, ...] = ()

    def by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item.get("id") or ""): dict(item) for item in self.providers}

    def diagnostic(self) -> dict[str, Any]:
        ready = sum(1 for item in self.providers if item.get("credential_ready") is True)
        return {
            "capability_hash": self.capability_hash,
            "private_registry_hash": self.private_registry_hash,
            "private_snapshot_loaded": self.private_snapshot_loaded,
            "provider_count": len(self.providers),
            "credential_ready_count": ready,
            "source_add_provider_count": self.source_add_provider_count,
            "warning_count": len(self.warning_codes),
        }

def _credential_ready(provider: Mapping[str, Any]) -> bool:
    if str(provider.get("auth_kind") or "none").lower() == "none":
        return True
    cost_model = provider.get("cost_model") if isinstance(provider.get("cost_model"), Mapping) else {}
    envelope = cost_model.get("source_add_credential_envelope") if isinstance(cost_model, Mapping) else {}
    if isinstance(envelope, Mapping):
        ciphertext = str(envelope.get("ciphertext_b64") or "").strip()
        kms_key_id = str(envelope.get("kms_key_id") or "").strip()
        credential_ref = str(envelope.get("credential_ref") or "").strip()
        try:
            decoded = base64.b64decode(ciphertext, validate=True)
        except (binascii.Error, ValueError):
            decoded = b""
        if (
            kms_key_id
            and credential_ref.startswith("encrypted_ref:source_add:")
            and 8 <= len(decoded) <= 16_384
        ):
            return True
    for ref in _string_tuple(provider.get("credential_ref"), limit=12):
        if ref.startswith("encrypted_ref:"):
            continue
        key_split = str(os.getenv("RESEARCH_LAB_PROVIDER_KEY_SPLIT", "") or "").strip().lower() in _TRUTHY
        if key_split and not ref.startswith("RESEARCH_LAB_"):
            continue
        if str(os.getenv(ref) or "").strip():
            return True
    return False

def _resolved_credential_ready(
    provider: Mapping[str, Any],
    resolver: Callable[[Mapping[str, Any]], bool | None] | None,
) -> bool:
    if resolver is not None:
        resolved = resolver(provider)
        if resolved is not None:
            return bool(resolved)
    return _credential_ready(provider)

def _provider_doc_from_source_row(
    row: Mapping[str, Any],
    *,
    credential_ready_resolver: Callable[[Mapping[str, Any]], bool | None]
    | None = None,
) -> dict[str, Any] | None:
    if str(row.get("provision_status") or "") != "provisioned_autoresearch_eligible":
        return None
    provision = row.get("provision_doc") if isinstance(row.get("provision_doc"), Mapping) else {}
    raw = provision.get("provider_registry_entry") if isinstance(provision.get("provider_registry_entry"), Mapping) else {}
    if not raw:
        return None
    provider = dict(raw)
    envelope = row.get("credential_envelope") if isinstance(row.get("credential_envelope"), Mapping) else {}
    cost_model = dict(provider.get("cost_model") or {})
    if envelope:
        cost_model["source_add_credential_envelope"] = dict(envelope)
        cost_model["source_add_miner_hotkey"] = str(row.get("miner_hotkey") or "")
        cost_model["source_add_adapter_ref"] = f"source_add:{str(row.get('adapter_id') or '')}"
    provider["cost_model"] = cost_model
    probe_endpoints = provision.get("probe_endpoints") if isinstance(provision.get("probe_endpoints"), list) else []
    routes = []
    for endpoint in probe_endpoints:
        if not isinstance(endpoint, Mapping):
            continue
        routes.append(
            {
                "method": str(endpoint.get("method") or "GET").upper(),
                "path": str(endpoint.get("path") or ""),
            }
        )
    raw_planner = (
        provider.get("planner_summary")
        if isinstance(provider.get("planner_summary"), Mapping)
        else {}
    )
    planner_summary: dict[str, Any] = {
        "provider_alias": str(
            raw_planner.get("provider_alias") or provider.get("id") or ""
        )[:80],
        "endpoint_families": [
            {
                "endpoint_id": str(item.get("endpoint_id") or "")[:120],
                "description": str(item.get("description") or "")[:200],
            }
            for item in probe_endpoints
            if isinstance(item, Mapping)
        ],
        "model_policy": "",
        "probe_metadata": [
            str(item.get("endpoint_id") or "")[:120]
            for item in probe_endpoints
            if isinstance(item, Mapping)
        ],
    }
    for field_name in _SOURCE_ADD_V8_PLANNER_FIELDS:
        if field_name not in raw_planner:
            continue
        # Round-trip through strict JSON so the durable provider projection
        # retains only public, serializable model-routing metadata. The v8
        # normalizer later validates every semantic value and fails closed.
        planner_summary[field_name] = json.loads(
            json.dumps(
                raw_planner[field_name],
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    provider.update(
        {
            "origin": "source_add",
            "reward_eligible": True,
            "source_add_provisioning_provenance_sha256": sha256_json(
                dict(row)
            ).split(":", 1)[-1],
            "probe_endpoints": [dict(item) for item in probe_endpoints if isinstance(item, Mapping)],
            "capability_policy": {
                "routes": routes,
                "blocked_routes": [],
                "allow_unlisted_paths": False,
                "unlisted_methods": [],
                "model_policy": {"kind": "none"},
            },
            "planner_summary": planner_summary,
        }
    )
    provider["credential_ready"] = _resolved_credential_ready(
        provider,
        credential_ready_resolver,
    )
    return provider

def _load_private_snapshot_rows_sync() -> list[Mapping[str, Any]]:
    from gateway.db.client import get_write_client

    response = (
        get_write_client()
        .table("research_lab_provider_registry")
        .select("registry_hash,provider_count,registry_doc,created_at")
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    return [dict(row) for row in rows]

def _parse_private_snapshot(
    row: Mapping[str, Any],
    *,
    credential_ready_resolver: Callable[[Mapping[str, Any]], bool | None]
    | None = None,
) -> tuple[list[dict[str, Any]], str]:
    doc = row.get("registry_doc") if isinstance(row.get("registry_doc"), Mapping) else {}
    raw_providers = doc.get("providers") if isinstance(doc.get("providers"), list) else None
    if not raw_providers:
        raise ValueError("private capability snapshot has no providers")
    providers = [dict(item) for item in raw_providers if isinstance(item, Mapping)]
    if len(providers) != len(raw_providers) or len(providers) != int(row.get("provider_count") or 0):
        raise ValueError("private capability snapshot provider_count mismatch")
    seen: set[str] = set()
    for provider in providers:
        provider_id = str(provider.get("id") or "")
        if provider_id in seen:
            raise ValueError("private capability snapshot has duplicate provider ids")
        seen.add(provider_id)
        if str(provider.get("origin") or "") != "builtin":
            raise ValueError("private capability snapshot providers must use builtin origin")
        errors = validate_capability_provider_doc(provider)
        if errors:
            raise ValueError("invalid private capability provider: " + ";".join(errors))
        provider["credential_ready"] = _resolved_credential_ready(
            provider,
            credential_ready_resolver,
        )
    expected_hash = sha256_json(doc)
    registry_hash = str(row.get("registry_hash") or "")
    if registry_hash != expected_hash:
        raise ValueError("private capability snapshot hash mismatch")
    return providers, registry_hash

def _legacy_provider_doc(
    value: Mapping[str, Any],
    *,
    credential_ready_resolver: Callable[[Mapping[str, Any]], bool | None]
    | None = None,
) -> dict[str, Any]:
    provider = dict(value)
    provider.setdefault("origin", "legacy_fallback")
    provider.setdefault("reward_eligible", False)
    provider.setdefault("capability_policy", {})
    provider.setdefault("planner_summary", {})
    provider.setdefault("probe_endpoints", [])
    provider["credential_ready"] = _resolved_credential_ready(
        provider,
        credential_ready_resolver,
    )
    return provider

def load_effective_provider_capabilities_sync(
    static_provider_docs: Sequence[Mapping[str, Any]],
    *,
    strict_remote: bool = False,
    private_row_loader: Callable[[], Mapping[str, Any] | Sequence[Mapping[str, Any]] | None] | None = None,
    source_row_loader: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    credential_ready_resolver: Callable[[Mapping[str, Any]], bool | None]
    | None = None,
) -> EffectiveProviderCapabilities:
    """Merge private snapshot, ready SOURCE_ADD rows, then continuity fallback."""

    static_docs = [
        _legacy_provider_doc(
            item,
            credential_ready_resolver=credential_ready_resolver,
        )
        for item in static_provider_docs
    ]
    private_loaded = False
    private_hash = ""
    private_docs: list[dict[str, Any]] = []
    warning_codes: list[str] = []
    try:
        loaded_rows = (private_row_loader or _load_private_snapshot_rows_sync)()
        if isinstance(loaded_rows, Mapping):
            candidate_rows = [loaded_rows]
        elif isinstance(loaded_rows, Sequence):
            candidate_rows = [item for item in loaded_rows if isinstance(item, Mapping)]
        else:
            candidate_rows = []
        last_error: Exception | None = None
        for candidate_row in candidate_rows:
            try:
                private_docs, private_hash = _parse_private_snapshot(
                    candidate_row,
                    credential_ready_resolver=credential_ready_resolver,
                )
                private_loaded = True
                break
            except Exception as exc:
                last_error = exc
                warning_codes.append("private_snapshot_invalid_skipped")
        if candidate_rows and not private_loaded and last_error is not None:
            raise last_error
    except Exception:
        if strict_remote:
            raise
        logger.warning("research_lab_provider_capability_private_load_failed", exc_info=True)
        warning_codes.append("private_snapshot_unavailable")

    try:
        if source_row_loader is None:
            from gateway.research_lab.source_add_catalog import load_provisioned_source_rows_sync

            source_rows = load_provisioned_source_rows_sync(raise_on_error=strict_remote)
        else:
            source_rows = list(source_row_loader())
    except Exception:
        if strict_remote:
            raise
        logger.warning("research_lab_provider_capability_source_add_load_failed", exc_info=True)
        source_rows = []
        warning_codes.append("source_add_snapshot_unavailable")

    reserved_ids = {str(item.get("id") or "") for item in static_docs + private_docs}
    source_docs: list[dict[str, Any]] = []
    for row in source_rows:
        provider = _provider_doc_from_source_row(
            row,
            credential_ready_resolver=credential_ready_resolver,
        )
        if provider is None:
            warning_codes.append("source_add_provider_missing_registry_entry")
            continue
        provider_id = str(provider.get("id") or "")
        if provider_id in reserved_ids or any(str(item.get("id") or "") == provider_id for item in source_docs):
            warning_codes.append("source_add_provider_id_collision")
            logger.warning("research_lab_source_add_provider_collision provider_hash=%s", sha256_json({"id": provider_id}))
            continue
        errors = validate_capability_provider_doc(provider)
        if errors or provider.get("credential_ready") is not True:
            warning_codes.append("source_add_provider_not_runtime_ready")
            continue
        source_docs.append(provider)

    merged: dict[str, dict[str, Any]] = {}
    for provider in private_docs:
        merged[str(provider.get("id") or "")] = provider
    for provider in source_docs:
        merged[str(provider.get("id") or "")] = provider
    for provider in static_docs:
        merged.setdefault(str(provider.get("id") or ""), provider)
    providers = tuple(dict(merged[key]) for key in sorted(merged) if key)
    capability_hash = sha256_json(
        {
            "providers": [
                {key: value for key, value in provider.items() if key != "credential_ready"}
                for provider in providers
            ],
            "private_registry_hash": private_hash,
            "private_snapshot_loaded": private_loaded,
        }
    )
    return EffectiveProviderCapabilities(
        providers=providers,
        capability_hash=capability_hash,
        private_registry_hash=private_hash,
        private_snapshot_loaded=private_loaded,
        source_add_provider_count=len(source_docs),
        warning_codes=tuple(sorted(warning_codes)),
    )

def _route_matches(route: Mapping[str, Any], method: str, path: str) -> bool:
    if str(route.get("method") or "").upper() != method.upper():
        return False
    exact = str(route.get("path") or "")
    prefix = str(route.get("path_prefix") or "")
    return bool(exact and path == exact) or bool(prefix and path.startswith(prefix))

def provider_request_allowed(provider: Mapping[str, Any], method: str, rest: str) -> tuple[bool, str, str]:
    normalized = normalize_candidate_route(rest)
    if normalized is None:
        return False, "unsafe_route", ""
    path, _query = normalized
    policy = provider.get("capability_policy")
    if not isinstance(policy, Mapping) or not policy:
        return True, "legacy_continuity", path
    blocked = policy.get("blocked_routes") if isinstance(policy.get("blocked_routes"), list) else []
    if any(isinstance(item, Mapping) and _route_matches(item, method, path) for item in blocked):
        return False, "blocked_route", path
    routes = policy.get("routes") if isinstance(policy.get("routes"), list) else []
    if any(isinstance(item, Mapping) and _route_matches(item, method, path) for item in routes):
        return True, "allowed_route", path
    unlisted_methods = {
        str(item).upper()
        for item in _string_tuple(policy.get("unlisted_methods"), limit=4)
    }
    if policy.get("allow_unlisted_paths") is True and method.upper() in unlisted_methods:
        return True, "allowed_unlisted_route", path
    return False, "route_not_allowed", path

class LiveTextModelCatalog:
    """Thread-safe live text-model cache with last-known-good fallback."""

    def __init__(
        self,
        *,
        ttl_seconds: int | None = None,
        fetch_json: Callable[[str, Mapping[str, str]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._ttl_seconds = max(60, int(ttl_seconds or model_catalog_ttl_seconds()))
        self._fetch_json = fetch_json or self._default_fetch_json
        self._lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._models: dict[str, set[str]] = {}
        self._fetched_at: dict[str, float] = {}
        self._status: dict[str, str] = {}

    @staticmethod
    def _default_fetch_json(url: str, headers: Mapping[str, str]) -> Mapping[str, Any]:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("model catalog response must be an object")
        return data

    @staticmethod
    def _model_ids(doc: Mapping[str, Any]) -> set[str]:
        raw = doc.get("data") if isinstance(doc.get("data"), list) else []
        ids: set[str] = set()
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            model_id = str(item.get("id") or "").strip()
            architecture = item.get("architecture") if isinstance(item.get("architecture"), Mapping) else {}
            outputs = architecture.get("output_modalities") if isinstance(architecture, Mapping) else None
            if model_id and (not isinstance(outputs, list) or "text" in outputs):
                ids.add(model_id)
        return ids

    def refresh(self, provider: Mapping[str, Any], *, credential: str = "", force: bool = False) -> tuple[set[str], str]:
        provider_id = str(provider.get("id") or "")
        policy = provider.get("capability_policy") if isinstance(provider.get("capability_policy"), Mapping) else {}
        model_policy = policy.get("model_policy") if isinstance(policy.get("model_policy"), Mapping) else {}
        if str(model_policy.get("kind") or "none") != "live_text_catalog":
            return set(), "not_required"
        now = time.monotonic()
        with self._lock:
            current = set(self._models.get(provider_id, set()))
            fetched_at = self._fetched_at.get(provider_id, 0.0)
            if not force and current and now - fetched_at < self._ttl_seconds:
                return current, self._status.get(provider_id, "live")
        with self._fetch_lock:
            with self._lock:
                current = set(self._models.get(provider_id, set()))
                fetched_at = self._fetched_at.get(provider_id, 0.0)
                if not force and current and now - fetched_at < self._ttl_seconds:
                    return current, self._status.get(provider_id, "live")
            catalog_path = str(model_policy.get("catalog_path") or "")
            if not catalog_path.startswith("/"):
                return current, "catalog_unconfigured"
            url = str(provider.get("base_url") or "").rstrip("/") + catalog_path
            headers = {"Accept": "application/json"}
            if credential:
                headers["Authorization"] = "Bearer " + credential
            try:
                models = self._model_ids(self._fetch_json(url, headers))
                if not models:
                    raise ValueError("live text-model catalog returned no models")
            except Exception as exc:
                with self._lock:
                    fallback = set(self._models.get(provider_id, set()))
                    if fallback:
                        self._status[provider_id] = "last_known_good"
                        logger.warning(
                            "research_lab_text_model_catalog_refresh_failed provider_hash=%s fallback=last_known_good error_class=%s",
                            sha256_json({"provider": provider_id}),
                            type(exc).__name__,
                        )
                        return fallback, "last_known_good"
                bootstrap = set(_string_tuple(model_policy.get("bootstrap_model_ids"), limit=100))
                logger.warning(
                    "research_lab_text_model_catalog_refresh_failed provider_hash=%s fallback=%s error_class=%s",
                    sha256_json({"provider": provider_id}),
                    "bootstrap" if bootstrap else "unavailable",
                    type(exc).__name__,
                )
                return bootstrap, "bootstrap_fallback" if bootstrap else "unavailable"
            with self._lock:
                self._models[provider_id] = set(models)
                self._fetched_at[provider_id] = now
                self._status[provider_id] = "live"
            return set(models), "live"

    def validate_model(
        self,
        provider: Mapping[str, Any],
        model_id: str,
        *,
        credential: str = "",
    ) -> tuple[bool, str]:
        model = str(model_id or "").strip()
        if not model or len(model) > 200 or any(ord(char) < 32 for char in model):
            return False, "model_id_invalid"
        models, status = self.refresh(provider, credential=credential)
        if model in models:
            return True, status
        policy = provider.get("capability_policy") if isinstance(provider.get("capability_policy"), Mapping) else {}
        model_policy = policy.get("model_policy") if isinstance(policy.get("model_policy"), Mapping) else {}
        lookup_template = str(model_policy.get("lookup_path_template") or "")
        if status == "live" and lookup_template and "{model_id}" in lookup_template:
            lookup_path = lookup_template.replace("{model_id}", urllib.parse.quote(model, safe="/"))
            url = str(provider.get("base_url") or "").rstrip("/") + lookup_path
            headers = {"Accept": "application/json"}
            if credential:
                headers["Authorization"] = "Bearer " + credential
            try:
                doc = self._fetch_json(url, headers)
                resolved = str(doc.get("id") or (doc.get("data") or {}).get("id") or "")
                if resolved == model:
                    with self._lock:
                        self._models.setdefault(str(provider.get("id") or ""), set()).add(model)
                    return True, "live_lookup"
            except Exception as exc:
                logger.warning(
                    "research_lab_text_model_lookup_failed provider_hash=%s model_hash=%s error_class=%s",
                    sha256_json({"provider": str(provider.get("id") or "")}),
                    sha256_json({"model": model}),
                    type(exc).__name__,
                )
                return False, "model_lookup_failed"
        return False, "model_not_in_live_text_catalog"

    def model_ids(self, provider_id: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._models.get(str(provider_id), set())))
