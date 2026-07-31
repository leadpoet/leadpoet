"""Private provider-capability snapshots for Research Lab runtimes.

The committed module defines only the generic contract and validation rules.
Production provider inventories are append-only rows in Supabase and are never
embedded here or projected into miner-visible activity.
"""

from __future__ import annotations

import ast
import base64
import binascii
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Sequence

from gateway.research_lab.code_build import CodeEditBuildError, _run_git_apply
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
_NETWORK_IMPORT_RE = re.compile(
    r"^\+\s*(?:from|import)\s+(?:aiohttp|httpx|requests|urllib3|http\.client)(?:\b|\.)",
    re.IGNORECASE,
)
_ENV_READ_RE = re.compile(r"\b(?:os\.getenv|os\.environ|environ\.get|getenv)\s*[\[(]", re.IGNORECASE)
_CREDENTIAL_NAME_RE = re.compile(r"\b(?:api[_-]?key|access[_-]?token|credential|client[_-]?secret)\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_ENV_REF_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_SOURCE_ADD_MANIFEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ROUTING_FEATURE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,95}$")
_SOURCE_ADD_RUNTIME_PATH = "sourcing_model/routing/runtime.py"
_SOURCE_ADD_REGISTRATION_FIELDS = (
    "provider_id",
    "stage",
    "revision",
    "manifest_sha256",
    "priority",
    "capabilities",
    "idempotency",
    "cost_class",
    "unit_cost",
    "max_calls",
    "max_results",
    "timeout_seconds",
    "evidence_types",
    "best_for",
    "avoid_when",
    "best_for_description",
    "avoid_when_description",
)
_SOURCE_ADD_LEGACY_REGISTRATION_FIELDS = tuple(
    field
    for field in _SOURCE_ADD_REGISTRATION_FIELDS
    if field
    not in {
        "best_for",
        "avoid_when",
        "best_for_description",
        "avoid_when_description",
    }
)
_COMPANY_DISCOVERY_PATTERNS = (
    re.compile(r"\b(?:company|candidate|account)\s+(?:discovery|sourcing|acquisition)\b"),
    re.compile(r"\b(?:discover|source|find)\s+(?:companies|candidates|accounts)\b"),
)
_INTENT_DISCOVERY_PATTERNS = (
    re.compile(r"\b(?:intent|signal|evidence)\s+(?:discovery|sourcing|acquisition)\b"),
    re.compile(r"\b(?:discover|source|find)\s+(?:intent|signals|evidence)\b"),
)


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

    def prompt_summary(
        self,
        *,
        live_model_ids: Mapping[str, Sequence[str]] | None = None,
        miner_focus: str = "",
    ) -> dict[str, Any]:
        """Private planner context; callers persist only diagnostic hash/count."""

        summaries: list[dict[str, Any]] = []
        live_ids = dict(live_model_ids or {})
        for provider in self.providers:
            if not provider.get("active", True) or provider.get("credential_ready") is not True:
                continue
            planner = provider.get("planner_summary")
            if not isinstance(planner, Mapping):
                planner = {}
            summary = {
                "provider_alias": str(planner.get("provider_alias") or provider.get("id") or "")[:80],
                "endpoint_families": list(planner.get("endpoint_families") or [])[:40],
                "model_policy": str(planner.get("model_policy") or "")[:500],
                "probe_metadata": list(planner.get("probe_metadata") or [])[:40],
                "best_for": str(planner.get("best_for") or "")[:500],
                "avoid_when": str(planner.get("avoid_when") or "")[:500],
                "best_for_features": [
                    str(item)
                    for item in (
                        planner.get("best_for_features")
                        if isinstance(
                            planner.get("best_for_features"), (list, tuple)
                        )
                        else ()
                    )
                    if _ROUTING_FEATURE_RE.fullmatch(str(item))
                ][:16],
                "avoid_when_features": [
                    str(item)
                    for item in (
                        planner.get("avoid_when_features")
                        if isinstance(
                            planner.get("avoid_when_features"), (list, tuple)
                        )
                        else ()
                    )
                    if _ROUTING_FEATURE_RE.fullmatch(str(item))
                ][:16],
            }
            provider_id = str(provider.get("id") or "")
            if provider_id in live_ids:
                summary["live_text_model_ids"] = sorted({str(item) for item in live_ids[provider_id]})[:100]
            if str(provider.get("origin") or "") == "source_add":
                summary.update(
                    {
                        "governance_origin": "source_add",
                        "provider_id": provider_id,
                        "manifest_sha256": str(
                            provider.get("source_add_manifest_sha256") or ""
                        ),
                    }
                )
            summaries.append(summary)
        output = {
            "schema_version": "1.0",
            "capability_hash": self.capability_hash,
            "provider_count": len(summaries),
            "providers": summaries,
            "rules": {
                "existing_provider_modules_only": True,
                "new_credentials_forbidden": True,
                "new_network_clients_forbidden": True,
                "new_dependencies_forbidden": True,
            },
        }
        source_context = approved_source_router_suggestions(
            miner_focus,
            self.providers,
        )
        if source_context["requests"] or source_context["clarifications"]:
            output["routerverse_source_incorporation"] = source_context
        return output


def _normalized_words(value: Any) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split()
    )


def _source_mention_forms(provider: Mapping[str, Any]) -> tuple[str, ...]:
    planner = (
        provider.get("planner_summary")
        if isinstance(provider.get("planner_summary"), Mapping)
        else {}
    )
    forms = {
        _normalized_words(provider.get("id")),
        _normalized_words(planner.get("provider_alias")),
    }
    return tuple(sorted(item for item in forms if len(item) >= 3))


def _mentioned_source_add_providers(
    miner_focus: str,
    providers: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    focus = f" {_normalized_words(miner_focus)} "
    matches: list[dict[str, Any]] = []
    for provider in providers:
        if (
            str(provider.get("origin") or "") != "source_add"
            or provider.get("active", True) is not True
            or provider.get("credential_ready") is not True
        ):
            continue
        manifest = str(provider.get("source_add_manifest_sha256") or "")
        if not _SOURCE_ADD_MANIFEST_RE.fullmatch(manifest):
            continue
        if any(f" {form} " in focus for form in _source_mention_forms(provider)):
            matches.append(dict(provider))
    return tuple(
        sorted(matches, key=lambda item: str(item.get("id") or ""))
    )


def approved_source_router_suggestions(
    miner_focus: str,
    providers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Translate an explicit miner mention into attested router registrations.

    Only active, runtime-ready SOURCE_ADD providers can produce a request.
    The miner must also name company discovery and/or intent discovery; a
    provider mention without a stage is retained only as a clarification and
    cannot authorize a model edit.
    """

    normalized_focus = _normalized_words(miner_focus)
    stages: list[str] = []
    if any(pattern.search(normalized_focus) for pattern in _COMPANY_DISCOVERY_PATTERNS):
        stages.append("candidate_acquisition")
    if any(pattern.search(normalized_focus) for pattern in _INTENT_DISCOVERY_PATTERNS):
        stages.append("intent_evidence")
    providers_mentioned = _mentioned_source_add_providers(
        miner_focus,
        providers,
    )
    focus_with_boundaries = f" {normalized_focus} "
    directly_named = tuple(
        provider
        for provider in providers_mentioned
        if f" {_normalized_words(provider.get('id'))} "
        in focus_with_boundaries
    )
    if directly_named:
        providers_mentioned = directly_named
    elif len(providers_mentioned) > 1:
        return {
            "schema_version": "leadpoet.routerverse_source_suggestions.v2",
            "requests": [],
            "clarifications": [
                {
                    "provider_id": "",
                    "provider_alias": "",
                    "reason_code": "approved_source_mention_ambiguous",
                }
            ],
            "rules": {
                "approved_source_add_only": True,
                "explicit_discovery_stage_required": True,
                "model_registration_required": True,
                "consumer_binding_activation_separate": True,
            },
        }
    requests: list[dict[str, Any]] = []
    clarifications: list[dict[str, str]] = []
    for provider in providers_mentioned:
        provider_id = str(provider.get("id") or "")
        planner = (
            provider.get("planner_summary")
            if isinstance(provider.get("planner_summary"), Mapping)
            else {}
        )
        best_for = [
            str(item).strip()
            for item in (
                planner.get("best_for_features")
                if isinstance(planner.get("best_for_features"), (list, tuple))
                else ()
            )
            if _ROUTING_FEATURE_RE.fullmatch(str(item).strip())
        ][:16]
        avoid_when = [
            str(item).strip()
            for item in (
                planner.get("avoid_when_features")
                if isinstance(planner.get("avoid_when_features"), (list, tuple))
                else ()
            )
            if _ROUTING_FEATURE_RE.fullmatch(str(item).strip())
        ][:16]
        provider_alias = str(
            planner.get("provider_alias") or provider_id
        )[:80]
        if not stages:
            clarifications.append(
                {
                    "provider_id": provider_id,
                    "provider_alias": provider_alias,
                    "reason_code": "discovery_stage_not_explicit",
                }
            )
            continue
        cost_model = (
            provider.get("cost_model")
            if isinstance(provider.get("cost_model"), Mapping)
            else {}
        )
        try:
            microusd = max(
                0,
                int(cost_model.get("est_cost_microusd_per_call") or 0),
            )
        except (TypeError, ValueError):
            microusd = 0
        manifest = str(provider.get("source_add_manifest_sha256") or "")
        for stage in stages:
            candidate_stage = stage == "candidate_acquisition"
            requests.append(
                {
                    "schema_version": (
                        "leadpoet.routerverse_source_incorporation.v2"
                    ),
                    "provider_id": provider_id,
                    "provider_alias": provider_alias,
                    "stage": stage,
                    "tool_id": (
                        f"candidate.source_add.{provider_id}"
                        if candidate_stage
                        else f"intent.source_add.{provider_id}"
                    ),
                    "registration_symbol": (
                        "sourcing_model/routing/runtime.py::"
                        "SOURCE_ADD_ROUTING_REGISTRATIONS"
                    ),
                    "registration_type": "SourceAddRoutingRegistration",
                    "revision": f"source-add-{manifest[:12]}",
                    "manifest_sha256": manifest,
                    "priority": 80 if candidate_stage else 35,
                    "capabilities": [
                        (
                            "candidate.provider_discovery"
                            if candidate_stage
                            else "intent.provider_evidence"
                        )
                    ],
                    "idempotency": "idempotent",
                    "cost_class": "metered" if microusd else "free",
                    "unit_cost": (
                        round(max(microusd / 1_000_000, 0.000001), 6)
                        if microusd
                        else 0.0
                    ),
                    "max_calls": 1,
                    "max_results": 100 if candidate_stage else 1,
                    "timeout_seconds": 60.0 if candidate_stage else 30.0,
                    "evidence_types": (
                        ["provider_database"]
                        if candidate_stage
                        else ["external"]
                    ),
                    "best_for": (
                        best_for
                        or (
                            ["icp.structured_eligible"]
                            if candidate_stage
                            else ["intent.general"]
                        )
                    ),
                    "avoid_when": avoid_when,
                    "best_for_description": str(
                        planner.get("best_for")
                        or (
                            "Approved SOURCE_ADD company-discovery provider "
                            "for structured ICP acquisition."
                            if candidate_stage
                            else "Approved SOURCE_ADD provider for "
                            "company-scoped intent-evidence discovery."
                        )
                    )[:500],
                    "avoid_when_description": str(
                        planner.get("avoid_when")
                        or "Avoid when the consumer binding is unavailable, "
                        "unhealthy, outside its approved categories, or over "
                        "budget."
                    )[:500],
                    "runtime_binding_id": provider_id,
                }
            )
    return {
        "schema_version": "leadpoet.routerverse_source_suggestions.v2",
        "requests": requests,
        "clarifications": clarifications,
        "rules": {
            "approved_source_add_only": True,
            "explicit_discovery_stage_required": True,
            "model_registration_required": True,
            "consumer_binding_activation_separate": True,
        },
    }


def validate_source_add_registration_diff(
    unified_diff: str,
    source_context: Mapping[str, Any] | None,
    *,
    existing_runtime_source: str = "",
) -> list[str]:
    """Validate the complete patched Routerverse registration assignment.

    Added-line parsing is insufficient here: changing one field in an existing
    registration, adding an unknown keyword, or placing an approved constructor
    in dead code can all look valid in isolation. Apply only the runtime file's
    diff to the exact parent source, then compare the complete typed registry.
    """

    def function_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def normalize_registration(value: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(value)
        provider_id = normalized.get("provider_id")
        stage = normalized.get("stage")
        revision = normalized.get("revision")
        manifest = normalized.get("manifest_sha256")
        if not isinstance(provider_id, str) or not re.fullmatch(
            r"[a-z][a-z0-9_-]{1,79}", provider_id
        ):
            raise ValueError("provider_id is invalid")
        if stage not in {"candidate_acquisition", "intent_evidence"}:
            raise ValueError("stage is invalid")
        if (
            not isinstance(manifest, str)
            or not _SOURCE_ADD_MANIFEST_RE.fullmatch(manifest)
            or revision != f"source-add-{manifest[:12]}"
        ):
            raise ValueError("registration revision is not manifest-bound")
        for name in ("capabilities", "evidence_types", "best_for"):
            field_value = normalized.get(name)
            if (
                not isinstance(field_value, (list, tuple))
                or not field_value
                or any(
                    not isinstance(item, str) or not item
                    for item in field_value
                )
            ):
                raise ValueError(f"{name} must be a literal sequence")
            normalized[name] = tuple(field_value)
        avoid_when = normalized.get("avoid_when")
        if (
            not isinstance(avoid_when, (list, tuple))
            or any(
                not isinstance(item, str)
                or not _ROUTING_FEATURE_RE.fullmatch(item)
                for item in avoid_when
            )
        ):
            raise ValueError("avoid_when must be a literal feature sequence")
        normalized["avoid_when"] = tuple(avoid_when)
        for name in ("best_for",):
            if any(
                not _ROUTING_FEATURE_RE.fullmatch(item)
                for item in normalized[name]
            ):
                raise ValueError(f"{name} contains an invalid routing feature")
        for name in ("best_for_description", "avoid_when_description"):
            field_value = normalized.get(name)
            if (
                not isinstance(field_value, str)
                or not field_value.strip()
                or len(field_value) > 500
            ):
                raise ValueError(f"{name} must contain 1-500 characters")
            normalized[name] = " ".join(field_value.split())
        for name in ("priority", "max_calls", "max_results"):
            field_value = normalized.get(name)
            if isinstance(field_value, bool) or not isinstance(field_value, int):
                raise ValueError(f"{name} must be an integer")
        for name in ("unit_cost", "timeout_seconds"):
            field_value = normalized.get(name)
            if isinstance(field_value, bool) or not isinstance(
                field_value, (int, float)
            ):
                raise ValueError(f"{name} must be numeric")
            field_value = float(field_value)
            if not math.isfinite(field_value) or field_value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            normalized[name] = field_value
        expected_by_stage = {
            "candidate_acquisition": {
                "priority": 80,
                "capabilities": ("candidate.provider_discovery",),
                "max_results": 100,
                "timeout_seconds": 60.0,
                "evidence_types": ("provider_database",),
            },
            "intent_evidence": {
                "priority": 35,
                "capabilities": ("intent.provider_evidence",),
                "max_results": 1,
                "timeout_seconds": 30.0,
                "evidence_types": ("external",),
            },
        }
        expected = expected_by_stage[str(stage)]
        if any(normalized.get(name) != expected_value for name, expected_value in expected.items()):
            raise ValueError("registration stage contract is invalid")
        if (
            normalized.get("idempotency") != "idempotent"
            or normalized.get("cost_class") not in {"free", "metered"}
            or normalized.get("max_calls") != 1
            or (
                normalized.get("cost_class") == "free"
                and normalized.get("unit_cost") != 0.0
            )
            or (
                normalized.get("cost_class") == "metered"
                and normalized.get("unit_cost") <= 0.0
            )
        ):
            raise ValueError("registration execution contract is invalid")
        return normalized

    def registration_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
        provider_id = request.get("provider_id")
        stage = request.get("stage")
        expected_tool_id = (
            f"candidate.source_add.{provider_id}"
            if stage == "candidate_acquisition"
            else f"intent.source_add.{provider_id}"
        )
        if (
            request.get("schema_version")
            != "leadpoet.routerverse_source_incorporation.v2"
            or request.get("registration_symbol")
            != (
                "sourcing_model/routing/runtime.py::"
                "SOURCE_ADD_ROUTING_REGISTRATIONS"
            )
            or request.get("registration_type")
            != "SourceAddRoutingRegistration"
            or request.get("runtime_binding_id") != provider_id
            or request.get("tool_id") != expected_tool_id
            or not isinstance(request.get("provider_alias"), str)
            or not 0 < len(request["provider_alias"]) <= 80
        ):
            raise ValueError("approved request metadata is invalid")
        return normalize_registration(
            {
                field: request.get(field)
                for field in _SOURCE_ADD_REGISTRATION_FIELDS
            }
        )

    def registration_from_call(
        node: ast.Call,
    ) -> tuple[dict[str, Any], bool]:
        if function_name(node) != "SourceAddRoutingRegistration":
            raise ValueError("registry contains a non-registration value")
        if node.args or any(keyword.arg is None for keyword in node.keywords):
            raise ValueError("registration must use explicit keyword literals")
        keyword_names = [str(keyword.arg) for keyword in node.keywords]
        keyword_set = set(keyword_names)
        if (
            len(keyword_names) != len(set(keyword_names))
            or keyword_set
            not in (
                set(_SOURCE_ADD_REGISTRATION_FIELDS),
                set(_SOURCE_ADD_LEGACY_REGISTRATION_FIELDS),
            )
        ):
            raise ValueError("registration fields differ from the contract")
        values: dict[str, Any] = {}
        for keyword in node.keywords:
            try:
                values[str(keyword.arg)] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise ValueError(
                    "registration fields must be literal"
                ) from exc
        legacy_guidance = keyword_set == set(
            _SOURCE_ADD_LEGACY_REGISTRATION_FIELDS
        )
        if legacy_guidance:
            stage = values.get("stage")
            values.update(
                {
                    "best_for": (
                        ("icp.structured_eligible",)
                        if stage == "candidate_acquisition"
                        else ("intent.general",)
                    ),
                    "avoid_when": (),
                    "best_for_description": (
                        "Approved SOURCE_ADD company-discovery provider for "
                        "structured ICP acquisition."
                        if stage == "candidate_acquisition"
                        else "Approved SOURCE_ADD provider for company-scoped "
                        "intent-evidence discovery."
                    ),
                    "avoid_when_description": (
                        "Avoid when the consumer binding is unavailable, "
                        "unhealthy, outside its approved categories, or over "
                        "budget."
                    ),
                }
            )
        return normalize_registration(values), legacy_guidance

    def registrations_from_source(
        source: str,
    ) -> tuple[
        dict[tuple[str, str], dict[str, Any]],
        set[tuple[str, str]],
    ]:
        tree = ast.parse(source)
        assignments: list[ast.AST] = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == "SOURCE_ADD_ROUTING_REGISTRATIONS"
                for target in node.targets
            ):
                assignments.append(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "SOURCE_ADD_ROUTING_REGISTRATIONS"
            ):
                assignments.append(node.value)
        if len(assignments) != 1 or not isinstance(
            assignments[0], (ast.List, ast.Tuple)
        ):
            raise ValueError("registration assignment is missing or ambiguous")
        elements = assignments[0].elts
        registration_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and function_name(node) == "SourceAddRoutingRegistration"
        ]
        if len(registration_calls) != len(elements):
            raise ValueError("registration constructor exists outside the registry")
        registrations: dict[tuple[str, str], dict[str, Any]] = {}
        legacy_guidance_keys: set[tuple[str, str]] = set()
        for element in elements:
            if not isinstance(element, ast.Call):
                raise ValueError("registry contains a dynamic value")
            registration, legacy_guidance = registration_from_call(element)
            key = (
                str(registration.get("provider_id") or ""),
                str(registration.get("stage") or ""),
            )
            if (
                not re.fullmatch(r"[a-z][a-z0-9_-]{1,79}", key[0])
                or key[1] not in {"candidate_acquisition", "intent_evidence"}
                or key in registrations
            ):
                raise ValueError("registration identity is invalid or duplicated")
            registrations[key] = registration
            if legacy_guidance:
                legacy_guidance_keys.add(key)
        return registrations, legacy_guidance_keys

    def diff_sections(diff_text: str) -> list[tuple[str, str, str]]:
        sections: list[tuple[str, str, str]] = []
        current: list[str] = []
        old_path = ""
        new_path = ""
        for line in diff_text.splitlines(keepends=True):
            match = re.fullmatch(
                r"diff --git a/(\S+) b/(\S+)\r?\n?",
                line,
            )
            if match is not None:
                if current:
                    sections.append((old_path, new_path, "".join(current)))
                old_path, new_path = match.groups()
                current = [line]
            elif current:
                current.append(line)
        if current:
            sections.append((old_path, new_path, "".join(current)))
        return sections

    def patched_runtime_source(
        source: str,
        runtime_diff: str,
    ) -> str:
        with tempfile.TemporaryDirectory(
            prefix="leadpoet-source-registration-check-"
        ) as raw:
            root = Path(raw)
            runtime_path = root / _SOURCE_ADD_RUNTIME_PATH
            runtime_path.parent.mkdir(parents=True)
            runtime_path.write_text(source, encoding="utf-8")
            patch_path = root / "runtime.diff"
            patch_path.write_text(runtime_diff, encoding="utf-8")
            _run_git_apply(
                patch_path,
                cwd=root,
                timeout_seconds=5,
                check=False,
            )
            return runtime_path.read_text(encoding="utf-8")

    requests = (
        source_context.get("requests")
        if isinstance(source_context, Mapping)
        and isinstance(source_context.get("requests"), list)
        else []
    )
    diff_text = str(unified_diff or "")
    sections = diff_sections(diff_text)
    runtime_sections = [
        section
        for old_path, new_path, section in sections
        if old_path == _SOURCE_ADD_RUNTIME_PATH
        and new_path == _SOURCE_ADD_RUNTIME_PATH
    ]
    touches_runtime = bool(runtime_sections)
    added = "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    approved_tool_ids = {
        str(item.get("tool_id") or "")
        for item in requests
        if isinstance(item, Mapping)
    }
    added_tool_ids = {
        match.group(1)
        for match in re.finditer(
            r"['\"]((?:candidate|intent)\.source_add\.[a-z][a-z0-9_-]{1,79})['\"]",
            added,
        )
    }
    errors: list[str] = []
    if added_tool_ids - approved_tool_ids:
        errors.append("source_add_registration_unapproved_tool")
    if (
        "ToolDefinition(" in added
        and (
            "ORIGIN_SOURCE_ADD" in added
            or "source_add" in added
            or bool(added_tool_ids)
        )
    ):
        errors.append("source_add_registration_direct_definition_forbidden")
    source_markers_outside_runtime = any(
        (
            "SourceAddRoutingRegistration" in section
            or "SOURCE_ADD_ROUTING_REGISTRATIONS" in section
        )
        for old_path, new_path, section in sections
        if (
            old_path != _SOURCE_ADD_RUNTIME_PATH
            or new_path != _SOURCE_ADD_RUNTIME_PATH
        )
    )
    if source_markers_outside_runtime:
        errors.append("source_add_registration_wrong_model_target")

    needs_registry_validation = bool(requests) or touches_runtime or any(
        marker in added or marker in removed
        for marker in (
            "SourceAddRoutingRegistration",
            "SOURCE_ADD_ROUTING_REGISTRATIONS",
        )
    )
    if not needs_registry_validation:
        return sorted(set(errors))
    if not existing_runtime_source:
        errors.append("source_add_registration_source_unavailable")
        return sorted(set(errors))
    try:
        existing, existing_legacy_guidance = registrations_from_source(
            existing_runtime_source
        )
    except (SyntaxError, ValueError):
        errors.append("source_add_registration_existing_source_invalid")
        return sorted(set(errors))

    patched_source = existing_runtime_source
    if touches_runtime:
        if len(runtime_sections) != 1:
            errors.append("source_add_registration_runtime_diff_ambiguous")
            return sorted(set(errors))
        try:
            patched_source = patched_runtime_source(
                existing_runtime_source,
                runtime_sections[0],
            )
        except (CodeEditBuildError, OSError, ValueError):
            errors.append("source_add_registration_runtime_diff_unapplicable")
            return sorted(set(errors))
    try:
        observed, observed_legacy_guidance = registrations_from_source(
            patched_source
        )
    except (SyntaxError, ValueError):
        errors.append("source_add_registration_patched_source_invalid")
        return sorted(set(errors))
    if any(
        key not in existing
        or key not in existing_legacy_guidance
        or observed.get(key) != existing.get(key)
        for key in observed_legacy_guidance
    ):
        errors.append("source_add_registration_patched_source_invalid")
        return sorted(set(errors))

    expected: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for request in requests:
            if not isinstance(request, Mapping):
                raise ValueError("approved request is invalid")
            registration = registration_from_request(request)
            key = (
                str(registration.get("provider_id") or ""),
                str(registration.get("stage") or ""),
            )
            if key in expected:
                raise ValueError("approved request is duplicated")
            expected[key] = registration
    except ValueError:
        errors.append("source_add_approved_request_invalid")
        return sorted(set(errors))

    desired = dict(existing)
    desired.update(expected)
    if not requests:
        if set(existing) - set(observed):
            errors.append("source_add_registration_removal_forbidden")
        if observed != existing:
            errors.append("source_add_registration_without_approved_request")
        return sorted(set(errors))

    if any(observed.get(key) != registration for key, registration in expected.items()):
        errors.append("source_add_registration_missing_approved_request")
    if set(existing) - set(expected) - set(observed):
        errors.append("source_add_registration_removal_forbidden")
    if any(
        key not in desired or desired[key] != registration
        for key, registration in observed.items()
    ):
        errors.append("source_add_registration_unapproved_registration")
    if any(
        str(item.get("provider_id") or "") not in {
            str(request.get("provider_id") or "")
            for request in requests
            if isinstance(request, Mapping)
        }
        for item in observed.values()
        if item not in existing.values()
    ):
        errors.append("source_add_registration_unapproved_provider")
    if expected and not touches_runtime and any(
        existing.get(key) != registration
        for key, registration in expected.items()
    ):
        errors.append("source_add_registration_wrong_model_target")
    return sorted(set(errors))


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
    provider.update(
        {
            "origin": "source_add",
            "reward_eligible": True,
            "source_add_manifest_sha256": sha256_json(dict(row)).split(
                ":", 1
            )[-1],
            "probe_endpoints": [dict(item) for item in probe_endpoints if isinstance(item, Mapping)],
            "capability_policy": {
                "routes": routes,
                "blocked_routes": [],
                "allow_unlisted_paths": False,
                "unlisted_methods": [],
                "model_policy": {"kind": "none"},
            },
            "planner_summary": {
                "provider_alias": str(provider.get("id") or "")[:80],
                "endpoint_families": [
                    {
                        "endpoint_id": str(item.get("endpoint_id") or "")[:120],
                        "description": str(item.get("description") or "")[:200],
                    }
                    for item in probe_endpoints
                    if isinstance(item, Mapping)
                ],
                "model_policy": "",
                "probe_metadata": [str(item.get("endpoint_id") or "")[:120] for item in probe_endpoints if isinstance(item, Mapping)],
            },
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


def validate_candidate_provider_diff(
    unified_diff: str,
    capabilities: EffectiveProviderCapabilities,
) -> list[str]:
    """Static guard for provider-related additions before candidate build."""

    added_lines = []
    for line in str(unified_diff or "").splitlines():
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            added_lines.append(line)
    added = "\n".join(added_lines)
    errors: list[str] = []
    if any(_NETWORK_IMPORT_RE.search(line) for line in added_lines):
        errors.append("candidate_adds_new_network_client_import")
    if _ENV_READ_RE.search(added) or _CREDENTIAL_NAME_RE.search(added):
        errors.append("candidate_adds_new_credential_or_env_reference")
    provider_by_host: dict[str, dict[str, Any]] = {}
    for provider in capabilities.providers:
        parsed = urllib.parse.urlsplit(str(provider.get("base_url") or ""))
        if parsed.hostname:
            provider_by_host[parsed.hostname.lower()] = provider
    for match in _URL_RE.findall(added):
        parsed = urllib.parse.urlsplit(match.rstrip(".,);]"))
        host = (parsed.hostname or "").lower()
        provider = provider_by_host.get(host)
        if provider is None:
            errors.append("candidate_adds_unknown_provider_host:" + sha256_json({"host": host})[-16:])
            continue
        allowed_for_any_method = any(
            provider_request_allowed(provider, method, parsed.path or "/")[0]
            for method in _VALID_METHODS
        )
        if not allowed_for_any_method:
            errors.append("candidate_adds_disallowed_provider_path")
    for provider in capabilities.providers:
        policy = provider.get("capability_policy") if isinstance(provider.get("capability_policy"), Mapping) else {}
        for route in policy.get("blocked_routes") or []:
            if not isinstance(route, Mapping):
                continue
            blocked_path = str(route.get("path") or route.get("path_prefix") or "")
            if blocked_path and blocked_path in added:
                errors.append("candidate_adds_blocked_provider_route")
    return sorted(set(errors))


def summary_mentions_private_capability(
    summary: str,
    capabilities: EffectiveProviderCapabilities,
) -> bool:
    text = str(summary or "").lower()
    markers: set[str] = set()
    for provider in capabilities.providers:
        parsed = urllib.parse.urlsplit(str(provider.get("base_url") or ""))
        if parsed.hostname:
            markers.add(parsed.hostname.lower())
        planner = provider.get("planner_summary") if isinstance(provider.get("planner_summary"), Mapping) else {}
        alias = str(planner.get("provider_alias") or "").strip().lower()
        if len(alias) >= 4:
            markers.add(alias)
        policy = provider.get("capability_policy") if isinstance(provider.get("capability_policy"), Mapping) else {}
        model_policy = policy.get("model_policy") if isinstance(policy.get("model_policy"), Mapping) else {}
        markers.update(
            model.lower()
            for model in _string_tuple(model_policy.get("bootstrap_model_ids"), limit=100)
            if len(model) >= 6
        )
    return any(marker and marker in text for marker in markers)


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
