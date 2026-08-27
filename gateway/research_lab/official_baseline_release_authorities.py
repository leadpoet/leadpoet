"""Fixed exact-release bindings for official baseline model actions.

The immutable Sourcing artifact owns the binding catalog, provider request
compiler, verifier, response parser, and qualification semantics.  This module
only supplies the gateway-owned transport and append-only physical custody.
Provider traffic can leave the gateway only through the configured Research
Lab evidence proxy; provider credentials are never read or persisted here.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
import re
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

from gateway.research_lab.common_model_experiment import ProtectedModelActionResult
from gateway.research_lab.official_baseline_authority import (
    PROTECTED_ACTION_AUTHORITY_SHA256,
    PROTECTED_ACTION_AUTHORITY_SCHEMA_VERSION,
    GatewayLocalProtectedActionBridge,
    OfficialBaselineProtectedAuthorityError,
    OfficialBaselineProtectedPreparation,
    OfficialBaselineProtectedTerminal,
    OfficialBaselineReleaseComponents,
    _protected_result_document,
)
from gateway.research_lab.official_baseline_custody import (
    S3OfficialBaselineDocumentCustody,
)
from gateway.research_lab.official_baseline_model_runner import (
    ArtifactProtocolBenchmarkProjector,
    OfficialBaselineAuthorityUnavailable,
    OfficialBaselineDependencyContext,
)
from gateway.research_lab.official_baseline_store import (
    official_baseline_action_replay_identity,
)
from research_lab.canonical import sha256_json
from research_lab.common_model_runner_host import HostActionResult
from research_lab.docker_model_runner_transport import DockerModelRunnerTransport
from research_lab.eval.private_runtime import DockerPrivateModelRunner
from research_lab.model_runner_protocol import (
    ExactModelRunnerRegistration,
    ResearchLabModelRunnerProtocol,
)
from research_lab.routing_experiments import (
    ProviderOutcome,
    ProviderReceipt,
    ReceiptExecutionMode,
    validate_provider_receipt,
)


OFFICIAL_BINDING_CATALOG_SCHEMA_VERSION = (
    "model-runner-official-host-binding-catalog:v1"
)
PROTECTED_PROVIDER_PROGRESS_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_provider_progress.v1"
)
PROTECTED_PROVIDER_REQUEST_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_protected_request.v1"
)
PROTECTED_PROVIDER_TERMINAL_RECEIPT_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_terminal_receipt.v1"
)
PROTECTED_PROVIDER_REPLAY_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_provider_replay.v1"
)
SITE_PROTECTED_ACTION_AUTHORITY_SOURCE_COMMIT = (
    "f705fe57b61ea81188c42f3d2a0f04b310a33cd8"
)
SITE_PROTECTED_ACTION_AUTHORITY_SOURCE_CALLABLE = (
    "sourcing-worker/leadpoet_sourcing_worker/"
    "site_model_action_authority.py:"
    "protected_action_authority_contract_identity"
)

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
_BARE_HASH_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}")
_PROVIDER_ACTION_TYPES = frozenset(
    {
        "normalize_icp",
        "execute_candidate_tool",
        "execute_intent_tool",
        "execute_contact_tool",
    }
)
_VERIFIER_ACTION_TYPES = frozenset(
    {"verify_company", "verify_intent", "verify_contact"}
)
_PROXY_ROUTE_BY_PROVIDER = {
    "openrouter": "or",
    "scrapingdog": "sd",
    "deepline": "deepline",
    "exa": "exa",
}
_UPSTREAM_HOST_BY_PROVIDER = {
    "openrouter": "openrouter.ai",
    "scrapingdog": "api.scrapingdog.com",
    "deepline": "code.deepline.com",
    "exa": "api.exa.ai",
}
_ARTIFACT_CREDENTIAL_BINDING_BY_PROVIDER = {
    "openrouter": ("header", "Authorization", "OPENROUTER_API_KEY"),
    "scrapingdog": ("query", "api_key", "SCRAPINGDOG_API_KEY"),
    "deepline": ("header", "Authorization", "DEEPLINE_API_KEY"),
    "exa": ("header", "x-api-key", "EXA_API_KEY"),
}
_HTTP_METHODS = frozenset({"GET", "POST"})
_MAX_HTTP_RESPONSE_BYTES = 8 * 1024 * 1024
_DEEPLINE_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled"}
)


def _bare_hash(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    if _BARE_HASH_RE.fullmatch(text) is None:
        raise OfficialBaselineAuthorityUnavailable(f"{label} is invalid")
    return text


def _prefixed_hash(value: Any, label: str) -> str:
    return "sha256:" + _bare_hash(value, label)


def protected_action_authority_contract_identity() -> dict[str, Any]:
    """Return the exact static Site authority document frozen at f705fe57.

    This is a source copy of the endpoint-independent Site callable named by
    ``SITE_PROTECTED_ACTION_AUTHORITY_SOURCE_CALLABLE``.  Keeping the document
    local avoids a runtime dependency on Site code, its database, or its API.
    """

    payload: dict[str, Any] = {
        "schema_version": PROTECTED_ACTION_AUTHORITY_SCHEMA_VERSION,
        "protocol": "site-model-action-service:v3",
        "model_action_schema_version": "model-runner-action:v2",
        "action_request_schema_version": "site-model-action-request:v2",
        "prepared_action_schema_version": "site-model-action-prepared:v1",
        "dispatch_claim_schema_version": (
            "site-model-action-dispatch-claim:v1"
        ),
        "reconciliation_state_schema_version": (
            "site-model-action-reconciliation-state:v1"
        ),
        "reconciliation_schema_version": (
            "site-model-action-reconciliation:v1"
        ),
        "provider_receipt_schema_version": "site-model-action-receipt:v2",
        "model_completion_schema_version": "model-runner-completion:v3",
        "contract_hash_scope": "all_fields_except_contract_sha256",
        "service_operations": ["prepare", "execute_prepared", "reconcile"],
        "host_workflow": [
            "prepare",
            "outer_reserve",
            "execute_prepared_or_reconcile",
        ],
        "authority_context": {
            "owner": "host_runtime",
            "implementation": "host_specific_durable_authority",
            "static_site_run_schema_required": False,
            "credential_free_model_action": True,
            "raw_credentials_in_durable_preparation": False,
            "durable_credential_identity": "one_way_hash_only",
        },
        "ordering": {
            "required": [
                "prepare",
                "outer_reserve",
                "execute_prepared",
            ],
            "prepare_before_outer_reserve": True,
            "execute_requires_fresh_outer_reserve_or_authoritative_not_started": True,
            "prepare_provider_side_effects": 0,
        },
        "prepare": {
            "idempotency_scope": [
                "host_authority_scope",
                "action_sha256",
                "idempotency_key",
                "binding_contract_sha256",
            ],
            "request_body": "exact_canonical_model_action",
            "request_body_hash": "sha256",
            "response_closed_fields": [
                "schema_version",
                "prepared_action_ref",
                "prepared_request_sha256",
                "action_sha256",
            ],
            "same_identity_returns_same_reference": True,
            "different_identity_fails_closed": True,
            "provider_or_network_side_effects": False,
        },
        "outer_reserve": {
            "owner": "model_runner_host",
            "exact_action_and_continuation_required": True,
            "fresh_reserve_allows_execute_prepared": True,
            "existing_reserve_requires_reconcile": True,
            "foreign_or_unbound_reserve_allows_dispatch": False,
        },
        "execute_prepared": {
            "required_identity_fields": [
                "prepared_action_ref",
                "prepared_request_sha256",
                "action_sha256",
            ],
            "atomic_dispatch_claim": True,
            "maximum_dispatch_claims": 1,
            "physical_call_authorization_after_claim": True,
            "concurrent_or_repeated_claim": "fail_closed",
            "unresolved_claim_redispatch": False,
        },
        "reconcile": {
            "states": ["not_started", "known", "uncertain"],
            "response_closed_fields": [
                "schema_version",
                "state",
                "reason_code",
                "prepared_action_ref",
                "prepared_request_sha256",
                "action_sha256",
                "completion",
            ],
            "not_started_requires": [
                "exact_prepared_identity",
                "no_dispatch_claim",
                "no_physical_call_authorization",
            ],
            "known_requires": "exact_durable_model_completion",
            "claimed_without_durable_completion": "uncertain",
            "uncertain_redispatch": False,
            "not_started_allows_single_execute_prepared": True,
        },
        "provider_actions": sorted(
            {
                "normalize_icp",
                "execute_candidate_tool",
                "execute_intent_tool",
                "execute_contact_tool",
            }
        ),
        "verifier_actions": sorted(
            {
                "verify_company",
                "verify_intent",
                "verify_contact",
            }
        ),
        "accounting": {
            "provider_success_or_empty_minimum_calls": 1,
            "provider_receipt_custody_required": True,
            "verifier_outcome": "succeeded",
            "verifier_calls": 0,
            "verifier_cost_credits": 0,
            "verifier_provider_receipt_custody": False,
        },
        "artifact_request_authority": {
            "provider_request_compiler_owner": "model_artifact",
            "host_prompt_or_query_compilation": False,
            "exact_prepared_provider_bytes_required": True,
            "binding_without_artifact_compiler": "capability_closed",
            "adapter_semantics": "send_exact_bytes_and_return_raw_envelope",
        },
        "verifier_zero_call": {
            "network_or_source_retrieval": False,
            "provider_authorization": False,
            "provider_receipt": False,
            "evidence_authority": "exact_model_carried_custodied_evidence",
            "verifier_artifact": "exact_release_bound_immutable_artifact",
            "calls": 0,
            "cost_credits": 0,
            "company_result_schema_version": "company-verifier-response:v2",
            "missing_or_insufficient_evidence": "closed_rejection",
        },
        "physical_call_authority": {
            "authorization_timing": "immediately_before_each_call",
            "provider_idempotency_key": "authority_derived",
            "durable_authorization_unique_by": [
                "reservation_id",
                "call_ordinal",
            ],
            "ambiguous_call_requires_reconciliation": True,
            "fabricated_provider_result_allowed": False,
        },
        "runtime_identity_fields": [
            "release_identity_sha256",
            "manifest_sha256",
            "registry_sha256",
            "binding_contract_sha256",
            "service_release_sha256",
            "service_image_digest",
            "endpoint_identity_sha256",
            "handler_sha256",
        ],
        "runtime_identity_source": "verified_signed_registry",
        "runtime_identity_validation": "separate_from_static_contract_hash",
        "endpoint_or_manifest_values_in_static_contract": False,
        "legacy_execute": {
            "included_in_protected_contract": False,
            "old_release_drain_only": True,
            "new_official_baseline_allowed": False,
        },
        "canonical_json": "utf8-json-sort-keys-compact-ascii-no-nan",
        "hash_algorithm": "sha256",
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    contract_sha256 = hashlib.sha256(encoded).hexdigest()
    if "sha256:" + contract_sha256 != PROTECTED_ACTION_AUTHORITY_SHA256:
        raise OfficialBaselineAuthorityUnavailable(
            "protected action authority contract identity drifted"
        )
    return {**payload, "contract_sha256": contract_sha256}


def _validated_protected_action_authority_sha256() -> str:
    identity = protected_action_authority_contract_identity()
    if not isinstance(identity, Mapping):
        raise OfficialBaselineAuthorityUnavailable(
            "protected action authority contract is unavailable"
        )
    body = dict(identity)
    claimed = _bare_hash(
        body.pop("contract_sha256", None),
        "protected action authority contract hash",
    )
    computed = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    expected = _bare_hash(
        PROTECTED_ACTION_AUTHORITY_SHA256,
        "protected action authority identity",
    )
    if (
        body.get("schema_version")
        != PROTECTED_ACTION_AUTHORITY_SCHEMA_VERSION
        or claimed != computed
        or computed != expected
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "protected action authority contract identity differs"
        )
    return "sha256:" + computed


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline provider value is not canonical JSON"
        ) from exc


def _load_json_object(value: bytes) -> Mapping[str, Any]:
    def _closed_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError("duplicate JSON key")
            output[key] = item
        return output

    try:
        decoded = json.loads(value.decode("utf-8"), object_pairs_hook=_closed_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline provider returned invalid JSON"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline provider returned a non-object response"
        )
    return dict(decoded)


def _release_identity(context: OfficialBaselineDependencyContext) -> Mapping[str, Any]:
    return _artifact_release_identity(context.artifact)


def _artifact_release_identity(
    artifact: Any,
) -> Mapping[str, Any]:
    extensions = getattr(artifact, "signed_extensions", None)
    value = (
        extensions.get("model_release_identity")
        if isinstance(extensions, Mapping)
        else None
    )
    if not isinstance(value, Mapping):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline signed model release identity is unavailable"
        )
    return dict(value)


def preflight_official_baseline_artifact_protocol(
    *,
    artifact: Any,
    selection: Any,
    spec: Any,
) -> None:
    """Validate the signed semantic role map before starting host services."""

    release_identity = _artifact_release_identity(artifact)
    if (
        not getattr(selection, "is_exact", False)
        or getattr(spec, "image_digest", None)
        != getattr(artifact, "image_digest", None)
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact protocol preflight context differs"
        )
    protocol = ResearchLabModelRunnerProtocol(
        transport=DockerModelRunnerTransport(DockerPrivateModelRunner(spec)),
        expected_release_identity=release_identity,
    )
    generation = protocol.protocol_generation
    if (
        not generation.supports_official_baseline
        or generation.protocol_generation_sha256
        != selection.selection_document.get("protocol_generation_sha256")
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline signed protocol generation differs"
        )


def _catalog_bindings(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(catalog, Mapping) or set(catalog) != {
        "schema_version",
        "bindings",
        "binding_contracts_sha256",
        "catalog_sha256",
    } or catalog.get("schema_version") != OFFICIAL_BINDING_CATALOG_SCHEMA_VERSION:
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact binding catalog is invalid"
        )
    bindings = catalog.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact binding catalog is empty"
        )
    normalized: list[dict[str, Any]] = []
    keys: list[tuple[str, str]] = []
    required = {
        "schema_version",
        "action_type",
        "tool_id",
        "binding_contract_sha256",
        "response_schema_version",
        "idempotency",
        "max_response_bytes",
    }
    for value in bindings:
        if not isinstance(value, Mapping) or set(value) != required:
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline artifact binding row is invalid"
            )
        row = dict(value)
        action_type = str(row.get("action_type") or "")
        tool_id = str(row.get("tool_id") or "")
        if (
            action_type not in _PROVIDER_ACTION_TYPES | _VERIFIER_ACTION_TYPES
            or _SAFE_REF_RE.fullmatch(tool_id) is None
            or row.get("idempotency") != "idempotent"
            or type(row.get("max_response_bytes")) is not int
            or not 1 <= row["max_response_bytes"] <= _MAX_HTTP_RESPONSE_BYTES
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline artifact binding row differs"
            )
        row["binding_contract_sha256"] = _bare_hash(
            row.get("binding_contract_sha256"),
            "official baseline binding contract hash",
        )
        keys.append((action_type, tool_id))
        normalized.append(row)
    if (
        keys != sorted(keys)
        or len(keys) != len(set(keys))
        or len({tool_id for _action_type, tool_id in keys}) != len(keys)
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact binding catalog ordering differs"
        )
    if (
        _bare_hash(
            catalog.get("binding_contracts_sha256"),
            "official baseline binding catalog hash",
        )
        != sha256_json(normalized).removeprefix("sha256:")
        or _bare_hash(
            catalog.get("catalog_sha256"),
            "official baseline catalog identity",
        )
        != sha256_json(
            {
                "schema_version": OFFICIAL_BINDING_CATALOG_SCHEMA_VERSION,
                "bindings": normalized,
                "binding_contracts_sha256": catalog[
                    "binding_contracts_sha256"
                ],
            }
        ).removeprefix("sha256:")
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact binding catalog hash differs"
        )
    return normalized


def _validate_inventory_catalog(
    inventory: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    entries = inventory.get("entries") if isinstance(inventory, Mapping) else None
    if (
        not isinstance(entries, list)
        or _bare_hash(
            inventory.get("entries_sha256"),
            "official baseline compiler entries hash",
        )
        != sha256_json(entries).removeprefix("sha256:")
        or _bare_hash(
            inventory.get("inventory_sha256"),
            "official baseline compiler inventory hash",
        )
        != sha256_json(
            {
                key: item
                for key, item in inventory.items()
                if key != "inventory_sha256"
            }
        ).removeprefix("sha256:")
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact compiler inventory is invalid"
        )
    indexed = {
        (str(row.get("action_type") or ""), str(row.get("tool_id") or "")): dict(row)
        for row in entries
        if isinstance(row, Mapping)
    }
    if len(indexed) != len(entries):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact compiler inventory is duplicated"
        )
    for binding in _catalog_bindings(catalog):
        key = (binding["action_type"], binding["tool_id"])
        entry = indexed.get(key)
        expected_status = (
            "supported" if key[0] in _PROVIDER_ACTION_TYPES else "virtual"
        )
        if (
            not isinstance(entry, Mapping)
            or entry.get("status") != expected_status
            or (
                expected_status == "supported"
                and _bare_hash(
                    entry.get("compiler_contract_sha256"),
                    "official baseline compiler contract hash",
                )
                != binding["binding_contract_sha256"]
            )
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline catalog and compiler inventory differ"
            )
    return indexed


def _reviewed_provider_transport_available(provider: Any) -> bool:
    provider_id = str(provider or "").strip()
    route = _PROXY_ROUTE_BY_PROVIDER.get(provider_id)
    upstream_host = _UPSTREAM_HOST_BY_PROVIDER.get(provider_id)
    credential = _ARTIFACT_CREDENTIAL_BINDING_BY_PROVIDER.get(provider_id)
    return bool(
        re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", provider_id)
        and isinstance(route, str)
        and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", route)
        and isinstance(upstream_host, str)
        and re.fullmatch(
            r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?",
            upstream_host,
        )
        and isinstance(credential, tuple)
        and len(credential) == 3
        and credential[0] in {"header", "query"}
        and all(isinstance(item, str) and item for item in credential)
    )


def _official_host_availability(
    *,
    catalog: Mapping[str, Any],
    inventory: Mapping[str, Any],
    ready_provider_ids: Sequence[str],
) -> dict[str, bool]:
    """Derive claims only from artifact facts and reviewed proxy capability."""

    bindings = _catalog_bindings(catalog)
    indexed = _validate_inventory_catalog(inventory, catalog)
    ready_providers = frozenset(str(item) for item in ready_provider_ids)
    availability: dict[str, bool] = {}
    for binding in bindings:
        action_type = binding["action_type"]
        entry = indexed[(action_type, binding["tool_id"])]
        if action_type in _PROVIDER_ACTION_TYPES:
            available = bool(
                entry.get("status") == "supported"
                and entry.get("execution_mode") == "invoke"
                and isinstance(entry.get("compiler_id"), str)
                and _SAFE_REF_RE.fullmatch(entry.get("compiler_id") or "")
                and _reviewed_provider_transport_available(
                    entry.get("provider")
                )
                and _PROXY_ROUTE_BY_PROVIDER.get(
                    str(entry.get("provider") or "")
                )
                in ready_providers
            )
        else:
            available = bool(
                action_type in {"verify_company", "verify_intent"}
                and entry.get("status") == "virtual"
                and entry.get("execution_mode") == "verify"
                and entry.get("provider") == "model_artifact"
            )
        availability[binding["tool_id"]] = available
    return availability


def _proxy_base_url(value: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = urllib.parse.urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline evidence proxy URL is invalid"
        ) from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or port is None
        or port < 1
        or port > 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline evidence proxy URL is invalid"
        )
    return text.rstrip("/")


class _GatewayEvidenceProxyClient:
    """Credential-free, bounded transport to the gateway evidence proxy."""

    def __init__(
        self,
        *,
        proxy_url: str,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self._proxy_url = _proxy_base_url(proxy_url)
        self._opener = opener

    def _proxied_url(self, *, provider: str, upstream_url: str) -> str:
        route = _PROXY_ROUTE_BY_PROVIDER.get(provider)
        expected_host = _UPSTREAM_HOST_BY_PROVIDER.get(provider)
        try:
            parsed = urllib.parse.urlsplit(upstream_url)
        except ValueError as exc:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider URL is invalid"
            ) from exc
        if (
            route is None
            or parsed.scheme != "https"
            or (parsed.hostname or "").casefold() != expected_host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider URL is outside the evidence proxy"
            )
        path = parsed.path or "/"
        query = "?" + parsed.query if parsed.query else ""
        return f"{self._proxy_url}/{route}{path}{query}"

    def request(
        self,
        *,
        provider: str,
        method: str,
        upstream_url: str,
        static_headers: Mapping[str, Any],
        body: Mapping[str, Any] | None,
        query: Mapping[str, Any] | None,
        timeout_seconds: float,
        max_response_bytes: int,
        cost_scope: str,
        replay_only: bool = False,
    ) -> tuple[int, Mapping[str, Any], Mapping[str, str]]:
        normalized_method = str(method or "").upper()
        if normalized_method not in _HTTP_METHODS:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider method is unsupported"
            )
        if not isinstance(static_headers, Mapping) or any(
            str(key).casefold()
            in {"authorization", "x-api-key", "proxy-authorization", "cookie"}
            for key in static_headers
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider headers contain host credentials"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < float(timeout_seconds) <= 900
            or type(max_response_bytes) is not int
            or not 1 <= max_response_bytes <= _MAX_HTTP_RESPONSE_BYTES
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider transport bounds are invalid"
            )
        target = self._proxied_url(provider=provider, upstream_url=upstream_url)
        if query:
            if not isinstance(query, Mapping):
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline provider query is invalid"
                )
            split = urllib.parse.urlsplit(target)
            existing = urllib.parse.parse_qsl(split.query, keep_blank_values=True)
            supplied: list[tuple[str, str]] = []
            for key, value in query.items():
                if isinstance(value, (list, tuple)):
                    supplied.extend((str(key), str(item)) for item in value)
                else:
                    supplied.append((str(key), str(value)))
            target = urllib.parse.urlunsplit(
                split._replace(query=urllib.parse.urlencode(existing + supplied))
            )
        encoded = None if body is None else _canonical_bytes(body)
        headers = {
            str(key): str(value)
            for key, value in static_headers.items()
            if str(key).casefold() != "x-research-lab-replay-only"
        }
        headers.update(
            {
                "Accept": headers.get("Accept", "application/json"),
                "X-Research-Lab-Cost-Scope": cost_scope,
                "X-Research-Lab-Budget-Soft-Stop": "1",
            }
        )
        if replay_only:
            headers["X-Research-Lab-Replay-Only"] = "1"
        request = urllib.request.Request(
            target,
            data=encoded,
            headers=headers,
            method=normalized_method,
        )
        status = 0
        raw = b""
        response_headers: Mapping[str, str] = {}
        try:
            response = self._opener(request, timeout=float(timeout_seconds))
            response_status = getattr(response, "status", None)
            status = int(
                response_status
                if response_status is not None
                else response.getcode()
            )
            try:
                raw = response.read(max_response_bytes + 1)
                response_headers = dict(response.headers.items())
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read(max_response_bytes + 1)
            response_headers = dict(exc.headers.items()) if exc.headers else {}
        except Exception as exc:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline evidence proxy request failed"
            ) from exc
        if len(raw) > max_response_bytes:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider response exceeds artifact bound"
            )
        return status, _load_json_object(raw), response_headers


class ArtifactPreparedActionExecutor:
    """Release-bound provider/verifier executor under the durable S3 claim."""

    def __init__(
        self,
        *,
        registration: ExactModelRunnerRegistration,
        catalog: Mapping[str, Any],
        inventory: Mapping[str, Any],
        custody: S3OfficialBaselineDocumentCustody,
        proxy_url: str,
        proxy_client: _GatewayEvidenceProxyClient | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            not isinstance(registration, ExactModelRunnerRegistration)
            or not isinstance(custody, S3OfficialBaselineDocumentCustody)
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline release executor dependencies are invalid"
            )
        bindings = _catalog_bindings(catalog)
        _validate_inventory_catalog(inventory, catalog)
        authority_identity_sha256 = (
            _validated_protected_action_authority_sha256()
        )
        self._registration = registration
        self._protocol = registration.protocol
        self._catalog = {
            (row["action_type"], row["tool_id"]): row for row in bindings
        }
        self._inventory = {
            (str(row.get("action_type") or ""), str(row.get("tool_id") or "")): dict(row)
            for row in inventory["entries"]
        }
        self._custody = custody
        self._authority_identity_sha256 = authority_identity_sha256
        self._proxy = proxy_client or _GatewayEvidenceProxyClient(
            proxy_url=proxy_url
        )
        self._clock = clock
        self._sleep = sleep

    def _binding(self, action: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        key = (
            str(action.get("action_type") or ""),
            str(action.get("tool_id") or ""),
        )
        binding = self._catalog.get(key)
        inventory = self._inventory.get(key)
        if (
            binding is None
            or inventory is None
            or _bare_hash(
                action.get("binding_contract_sha256"),
                "official baseline action binding hash",
            )
            != binding["binding_contract_sha256"]
            or action.get("response_schema_version")
            != binding["response_schema_version"]
            or action.get("max_response_bytes") != binding["max_response_bytes"]
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline action differs from artifact catalog"
            )
        return binding, inventory

    def _provider_dispatch(self, action: Mapping[str, Any]) -> dict[str, Any]:
        dispatch = self._protocol.prepare_provider_request(action)
        if not isinstance(dispatch, Mapping) or set(dispatch) != {
            "schema_version",
            "action_sha256",
            "action_type",
            "tool_id",
            "compiler_id",
            "compiler_contract_sha256",
            "provider",
            "request",
            "request_sha256",
            "response_contract",
            "budgets",
            "idempotency_key",
            "dispatch_sha256",
        }:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline artifact provider dispatch is not closed"
            )
        value = dict(dispatch)
        body = dict(value)
        claimed = _bare_hash(
            body.pop("dispatch_sha256"),
            "official baseline provider dispatch hash",
        )
        request = value.get("request")
        if (
            value.get("action_sha256") != action.get("action_sha256")
            or value.get("action_type") != action.get("action_type")
            or value.get("tool_id") != action.get("tool_id")
            or _bare_hash(
                value.get("compiler_contract_sha256"),
                "official baseline compiler contract hash",
            )
            != _bare_hash(
                action.get("binding_contract_sha256"),
                "official baseline action binding hash",
            )
            or not isinstance(request, Mapping)
            or _bare_hash(
                value.get("request_sha256"),
                "official baseline provider request hash",
            )
            != sha256_json(dict(request)).removeprefix("sha256:")
            or claimed != sha256_json(body).removeprefix("sha256:")
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline artifact provider dispatch identity differs"
            )
        provider = str(value.get("provider") or "")
        inventory = self._inventory.get(
            (
                str(value.get("action_type") or ""),
                str(value.get("tool_id") or ""),
            )
        )
        expected_credential = _ARTIFACT_CREDENTIAL_BINDING_BY_PROVIDER.get(
            provider
        )
        credential = request.get("credential_binding")
        if (
            not isinstance(inventory, Mapping)
            or inventory.get("status") != "supported"
            or inventory.get("execution_mode") != "invoke"
            or inventory.get("provider") != provider
            or inventory.get("compiler_id") != value.get("compiler_id")
            or expected_credential is None
            or not isinstance(credential, Mapping)
            or credential.get("location") != expected_credential[0]
            or credential.get("name") != expected_credential[1]
            or credential.get("source") != expected_credential[2]
            or credential.get("persist") is not False
            or not str(credential.get("scheme") or "").strip()
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline artifact credential binding differs"
            )
        return value

    def _recompute(
        self,
        *,
        run_sha256: str,
        unit_ref: str,
        action: Mapping[str, Any],
    ) -> tuple[OfficialBaselineProtectedPreparation, Mapping[str, Any] | None]:
        binding, inventory = self._binding(action)
        identity = official_baseline_action_replay_identity(
            run_sha256=_prefixed_hash(
                run_sha256, "official baseline run identity"
            ),
            unit_ref=unit_ref,
            action=action,
        )
        action_type = str(action.get("action_type") or "")
        dispatch: Mapping[str, Any] | None = None
        if action_type in _PROVIDER_ACTION_TYPES:
            dispatch = self._provider_dispatch(action)
            budgets = dispatch.get("budgets")
            if not isinstance(budgets, Mapping):
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline artifact provider budgets are invalid"
                )
            call_cap = budgets.get("call_cap")
            credit_cap = budgets.get("credit_cap")
            timeout_seconds = budgets.get("timeout_seconds")
            request_body_sha256 = _prefixed_hash(
                dispatch.get("request_sha256"),
                "official baseline provider request hash",
            )
            request_value: Mapping[str, Any] = dispatch
        else:
            call_cap = 0
            credit_cap = 0.0
            timeout_seconds = inventory.get("timeout_seconds")
            request_value = {
                "schema_version": "model-runner-verifier-preparation:v1",
                "action_sha256": action.get("action_sha256"),
                "action_type": action_type,
                "tool_id": action.get("tool_id"),
                "binding_contract_sha256": action.get(
                    "binding_contract_sha256"
                ),
            }
            request_body_sha256 = sha256_json(request_value)
        if (
            type(call_cap) is not int
            or not 0 <= call_cap <= 100_000
            or isinstance(credit_cap, bool)
            or not isinstance(credit_cap, (int, float))
            or not 0 <= float(credit_cap) <= 100
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < float(timeout_seconds) <= 900
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline protected action budget is invalid"
            )
        protected_request = {
            "schema_version": PROTECTED_PROVIDER_REQUEST_SCHEMA_VERSION,
            "authority_identity_sha256": self._authority_identity_sha256,
            "run_sha256": identity["run_sha256"],
            "unit_ref": unit_ref,
            "action_idempotency_sha256": identity[
                "action_idempotency_sha256"
            ],
            "action_sha256": identity["action_sha256"],
            "binding_contract_sha256": _prefixed_hash(
                binding["binding_contract_sha256"],
                "official baseline binding contract hash",
            ),
            "request_body_sha256": request_body_sha256,
            "request": request_value,
        }
        protected_request_sha256 = sha256_json(protected_request)
        job_digest = protected_request_sha256.removeprefix("sha256:")
        preparation = OfficialBaselineProtectedPreparation(
            authority_identity_sha256=self._authority_identity_sha256,
            run_sha256=identity["run_sha256"],
            unit_ref=unit_ref,
            action_idempotency_sha256=identity["action_idempotency_sha256"],
            action_sha256=identity["action_sha256"],
            action_sequence=int(action.get("sequence")),
            action_type=action_type,
            tool_id=str(action.get("tool_id") or ""),
            binding_contract_sha256=_prefixed_hash(
                binding["binding_contract_sha256"],
                "official baseline binding contract hash",
            ),
            request_fingerprint_sha256=identity[
                "request_fingerprint_sha256"
            ],
            request_body_sha256=request_body_sha256,
            call_cap=call_cap,
            credit_cap_microunits=int(round(float(credit_cap) * 1_000_000)),
            timeout_ms=int(round(float(timeout_seconds) * 1000)),
            protected_job_ref="official_action:" + job_digest,
            protected_request_sha256=protected_request_sha256,
        )
        return preparation, dispatch

    def prepare(
        self,
        *,
        run_identity: Mapping[str, Any],
        unit_ref: str,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedPreparation:
        preparation, _dispatch = self._recompute(
            run_sha256=sha256_json(dict(run_identity)),
            unit_ref=unit_ref,
            action=action,
        )
        return preparation

    @staticmethod
    def _validate_preparation(
        actual: OfficialBaselineProtectedPreparation,
        expected: OfficialBaselineProtectedPreparation,
    ) -> None:
        if not isinstance(actual, OfficialBaselineProtectedPreparation) or actual != expected:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline protected preparation differs on replay"
            )

    def _preparation_for_action(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
    ) -> tuple[OfficialBaselineProtectedPreparation, Mapping[str, Any] | None]:
        expected, dispatch = self._recompute(
            run_sha256=preparation.run_sha256,
            unit_ref=preparation.unit_ref,
            action=action,
        )
        self._validate_preparation(preparation, expected)
        return expected, dispatch

    @staticmethod
    def _request_ref(dispatch: Mapping[str, Any]) -> str:
        return "provider_request:" + _bare_hash(
            dispatch.get("dispatch_sha256"),
            "official baseline dispatch hash",
        )

    @staticmethod
    def _deepline_status(value: Mapping[str, Any], depth: int = 0) -> str:
        if depth > 8:
            return ""
        for name in ("status", "run_status", "state"):
            status = str(value.get(name) or "").strip().casefold()
            if status:
                return status
        for name in (
            "run",
            "workflow",
            "data",
            "result",
            "response",
            "payload",
            "output",
            "outputs",
            "value",
        ):
            nested = value.get(name)
            if isinstance(nested, Mapping):
                status = ArtifactPreparedActionExecutor._deepline_status(
                    nested, depth + 1
                )
                if status:
                    return status
        return ""

    @staticmethod
    def _deepline_run_id(
        value: Mapping[str, Any],
        reconciliation: Mapping[str, Any],
    ) -> str:
        """Resolve the provider run ID from the exact model-owned pointer."""

        pointer = reconciliation.get("run_id_json_pointer")
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline Deepline run id pointer is invalid"
            )
        current: Any = value
        for raw_segment in pointer[1:].split("/"):
            if not raw_segment or re.search(r"~(?:[^01]|$)", raw_segment):
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline Deepline run id pointer is invalid"
                )
            segment = raw_segment.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, Mapping) or segment not in current:
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline Deepline run id is unavailable"
                )
            current = current[segment]
        run_id = current.strip() if isinstance(current, str) else ""
        if _SAFE_REF_RE.fullmatch(run_id) is None:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline Deepline run id is invalid"
            )
        return run_id

    @staticmethod
    def _deepline_poll_path_component(
        progress: Mapping[str, Any],
    ) -> str:
        """Encode the run ID exactly as declared by the model artifact."""

        reconciliation = progress.get("reconciliation")
        if not isinstance(reconciliation, Mapping):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline Deepline poll contract is invalid"
            )
        layers = reconciliation.get("run_id_path_encoding_layers", 1)
        if type(layers) is not int or not 1 <= layers <= 4:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline Deepline run id path encoding is invalid"
            )
        encoded = str(progress.get("run_id") or "")
        for _ in range(layers):
            encoded = urllib.parse.quote(encoded, safe="")
        return encoded

    @staticmethod
    def _progress_document(
        *,
        preparation: OfficialBaselineProtectedPreparation,
        dispatch: Mapping[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        request = dispatch.get("request")
        reconciliation = (
            request.get("reconciliation") if isinstance(request, Mapping) else None
        )
        if not isinstance(reconciliation, Mapping):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline Deepline reconciliation contract is missing"
            )
        body = {
            "schema_version": PROTECTED_PROVIDER_PROGRESS_SCHEMA_VERSION,
            "preparation_sha256": preparation.preparation_sha256,
            "action_sha256": preparation.action_sha256,
            "request_body_sha256": preparation.request_body_sha256,
            "dispatch_sha256": _prefixed_hash(
                dispatch.get("dispatch_sha256"),
                "official baseline Deepline dispatch hash",
            ),
            "provider": "deepline",
            "provider_run_ref": "deepline_run:" + run_id,
            "run_id": run_id,
            "reconciliation": dict(reconciliation),
        }
        return {**body, "progress_sha256": sha256_json(body)}

    @staticmethod
    def _validate_progress(
        value: Mapping[str, Any],
        *,
        preparation: OfficialBaselineProtectedPreparation,
        dispatch: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "preparation_sha256",
            "action_sha256",
            "request_body_sha256",
            "dispatch_sha256",
            "provider",
            "provider_run_ref",
            "run_id",
            "reconciliation",
            "progress_sha256",
        }:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline Deepline progress is not closed"
            )
        run_id = str(value.get("run_id") or "")
        expected = ArtifactPreparedActionExecutor._progress_document(
            preparation=preparation,
            dispatch=dispatch,
            run_id=run_id,
        )
        if _SAFE_REF_RE.fullmatch(run_id) is None or dict(value) != expected:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline Deepline progress differs"
            )
        return expected

    def _proxy_request(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        provider: str,
        request: Mapping[str, Any],
        timeout_seconds: float,
        replay_only: bool,
    ) -> tuple[int, Mapping[str, Any], Mapping[str, str]]:
        method = str(request.get("method") or "")
        if method == "BATCH_GET":
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline nested batch request is invalid"
            )
        return self._proxy.request(
            provider=provider,
            method=method,
            upstream_url=str(request.get("url") or ""),
            static_headers=(
                request.get("static_headers")
                if isinstance(request.get("static_headers"), Mapping)
                else {}
            ),
            body=(
                request.get("body")
                if isinstance(request.get("body"), Mapping)
                else None
            ),
            query=(
                request.get("query")
                if isinstance(request.get("query"), Mapping)
                else None
            ),
            timeout_seconds=timeout_seconds,
            max_response_bytes=int(
                self._catalog[(preparation.action_type, preparation.tool_id)][
                    "max_response_bytes"
                ]
            ),
            cost_scope="official-baseline-" + preparation.unit_ref.split(":", 1)[-1][
                :32
            ],
            replay_only=replay_only,
        )

    def _one_request(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        provider: str,
        request: Mapping[str, Any],
        timeout_seconds: float,
    ) -> tuple[int, Mapping[str, Any]]:
        status, body, _headers = self._proxy_request(
            preparation=preparation,
            provider=provider,
            request=request,
            timeout_seconds=timeout_seconds,
            replay_only=False,
        )
        return status, body

    def _replay_one_request(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        provider: str,
        request: Mapping[str, Any],
        timeout_seconds: float,
    ) -> tuple[int, Mapping[str, Any]] | None:
        status, body, headers = self._proxy_request(
            preparation=preparation,
            provider=provider,
            request=request,
            timeout_seconds=timeout_seconds,
            replay_only=True,
        )
        evidence = next(
            (
                str(value).strip().casefold()
                for key, value in headers.items()
                if str(key).casefold() == "x-research-lab-evidence"
            ),
            "",
        )
        if evidence == "hit":
            return status, body
        if (
            status == 409
            and evidence in {"", "replay_miss"}
            and dict(body) == {"error": "replay_miss"}
        ):
            return None
        raise OfficialBaselineProtectedAuthorityError(
            "official baseline evidence proxy replay response is invalid"
        )

    def _batch_request(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        provider: str,
        request: Mapping[str, Any],
        timeout_seconds: float,
        replay_only: bool = False,
    ) -> tuple[int, Mapping[str, Any], int] | None:
        rows = request.get("requests")
        if (
            request.get("method") != "BATCH_GET"
            or not isinstance(rows, list)
            or not rows
            or len(rows) > preparation.call_cap
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline artifact batch request is invalid"
            )
        responses: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "method",
                "url",
                "query",
                "segment_id",
                "request_sha256",
            }:
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline artifact batch member is invalid"
                )
            member_request = {
                "method": row["method"],
                "url": row["url"],
                "query": row["query"],
                "static_headers": request.get("static_headers") or {},
            }
            terminal = (
                self._replay_one_request(
                    preparation=preparation,
                    provider=provider,
                    request=member_request,
                    timeout_seconds=timeout_seconds,
                )
                if replay_only
                else self._one_request(
                    preparation=preparation,
                    provider=provider,
                    request=member_request,
                    timeout_seconds=timeout_seconds,
                )
            )
            if terminal is None:
                return None
            status, body = terminal
            responses.append(
                {
                    "request_sha256": row["request_sha256"],
                    "segment_id": row["segment_id"],
                    "status_code": status,
                    "provider_payload": dict(body),
                }
            )
            if not 200 <= status < 300:
                return status, {"responses": responses}, len(responses)
        return 200, {"responses": responses}, len(responses)

    def _poll_deepline(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        dispatch: Mapping[str, Any],
        progress: Mapping[str, Any],
        started: float,
    ) -> tuple[int, Mapping[str, Any]] | None:
        reconciliation = progress["reconciliation"]
        run_id = self._deepline_poll_path_component(progress)
        deadline = started + preparation.timeout_ms / 1000
        first = True
        while first or self._clock() < deadline:
            first = False
            remaining = max(0.001, deadline - self._clock())
            primary = reconciliation.get("primary_poll")
            fallback = reconciliation.get("fallback_poll")
            if not isinstance(primary, Mapping) or not isinstance(fallback, Mapping):
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline Deepline poll contract is invalid"
                )
            request = {
                "method": primary.get("method"),
                "url": str(primary.get("url_template") or "").replace(
                    "{run_id}", run_id
                ),
                "static_headers": {},
            }
            status, body = self._one_request(
                preparation=preparation,
                provider="deepline",
                request=request,
                timeout_seconds=min(remaining, 30.0),
            )
            if status in {404, 405}:
                status, body = self._one_request(
                    preparation=preparation,
                    provider="deepline",
                    request={
                        "method": fallback.get("method"),
                        "url": str(fallback.get("url_template") or "").replace(
                            "{run_id}", run_id
                        ),
                        "static_headers": {},
                    },
                    timeout_seconds=min(remaining, 30.0),
                )
            if not 200 <= status < 300:
                return status, body
            state = self._deepline_status(body)
            if state in _DEEPLINE_TERMINAL_STATUSES:
                return status, body
            if self._clock() >= deadline:
                break
            self._sleep(min(1.0, max(0.0, deadline - self._clock())))
        return None

    def _execute_provider(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        dispatch: Mapping[str, Any],
        action: Mapping[str, Any],
        reconcile_only: bool,
    ) -> OfficialBaselineProtectedTerminal:
        request = dispatch["request"]
        provider = str(dispatch.get("provider") or "")
        request_ref = self._request_ref(dispatch)
        started = self._clock()
        calls = 1
        terminal: tuple[int, Mapping[str, Any]] | None
        if provider == "deepline":
            stored = self._custody.load_protected_action_progress(
                preparation_sha256=preparation.preparation_sha256
            )
            if stored is None:
                if reconcile_only:
                    return self._uncertain(request_ref)
                status, body = self._one_request(
                    preparation=preparation,
                    provider=provider,
                    request=request,
                    timeout_seconds=preparation.timeout_ms / 1000,
                )
                if not 200 <= status < 300:
                    terminal = (status, body)
                    return self._known_provider_terminal(
                        preparation=preparation,
                        dispatch=dispatch,
                        action=action,
                        status=status,
                        body=body,
                        calls=1,
                        started=started,
                    )
                reconciliation = request.get("reconciliation")
                if not isinstance(reconciliation, Mapping):
                    raise OfficialBaselineProtectedAuthorityError(
                        "official baseline Deepline reconciliation contract is missing"
                    )
                run_id = self._deepline_run_id(body, reconciliation)
                progress = self._progress_document(
                    preparation=preparation,
                    dispatch=dispatch,
                    run_id=run_id,
                )
                self._custody.append_protected_action_progress(
                    preparation_sha256=preparation.preparation_sha256,
                    progress=progress,
                )
                stored = self._custody.load_protected_action_progress(
                    preparation_sha256=preparation.preparation_sha256
                )
                if stored != progress:
                    raise OfficialBaselineProtectedAuthorityError(
                        "official baseline Deepline progress readback differs"
                    )
                if self._deepline_status(body) in _DEEPLINE_TERMINAL_STATUSES:
                    terminal = (status, body)
                else:
                    terminal = self._poll_deepline(
                        preparation=preparation,
                        dispatch=dispatch,
                        progress=progress,
                        started=started,
                    )
            else:
                progress = self._validate_progress(
                    stored,
                    preparation=preparation,
                    dispatch=dispatch,
                )
                terminal = self._poll_deepline(
                    preparation=preparation,
                    dispatch=dispatch,
                    progress=progress,
                    started=started,
                )
            if terminal is None:
                return self._uncertain(request_ref)
            status, body = terminal
        elif request.get("method") == "BATCH_GET":
            terminal = self._batch_request(
                preparation=preparation,
                provider=provider,
                request=request,
                timeout_seconds=preparation.timeout_ms / 1000,
                replay_only=reconcile_only,
            )
            if terminal is None:
                return self._uncertain(request_ref)
            status, body, calls = terminal
        else:
            terminal = (
                self._replay_one_request(
                    preparation=preparation,
                    provider=provider,
                    request=request,
                    timeout_seconds=preparation.timeout_ms / 1000,
                )
                if reconcile_only
                else self._one_request(
                    preparation=preparation,
                    provider=provider,
                    request=request,
                    timeout_seconds=preparation.timeout_ms / 1000,
                )
            )
            if terminal is None:
                return self._uncertain(request_ref)
            status, body = terminal
        return self._known_provider_terminal(
            preparation=preparation,
            dispatch=dispatch,
            action=action,
            status=status,
            body=body,
            calls=calls,
            started=started,
        )

    @staticmethod
    def _uncertain(provider_request_ref: str) -> OfficialBaselineProtectedTerminal:
        return OfficialBaselineProtectedTerminal(
            state="uncertain",
            protected_action_result=None,
            protected_result_sha256=None,
            protected_terminal_receipt_ref=None,
            protected_terminal_receipt_sha256=None,
            provider_request_ref=provider_request_ref,
            model_provider_response_sha256=None,
            uncertainty_sha256=sha256_json(
                {
                    "schema_version": (
                        "leadpoet.research_lab.provider_reconciliation_pending.v1"
                    ),
                    "provider_request_ref": provider_request_ref,
                }
            ),
        )

    def _known_provider_terminal(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        dispatch: Mapping[str, Any],
        action: Mapping[str, Any],
        status: int,
        body: Mapping[str, Any],
        calls: int,
        started: float,
    ) -> OfficialBaselineProtectedTerminal:
        provider = str(dispatch["provider"])
        elapsed_ms = max(0, int(math.ceil((self._clock() - started) * 1000)))
        if elapsed_ms > preparation.timeout_ms:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider exceeded the artifact timeout"
            )
        succeeded = 200 <= status < 300 and (
            provider != "deepline"
            or self._deepline_status(body) == "completed"
        )
        response = (
            {
                "schema_version": "host-provider-response:v1",
                "provider": provider,
                "status_code": status,
                "body": dict(body),
            }
            if succeeded
            else None
        )
        if (
            response is not None
            and len(_canonical_bytes(response))
            > self._catalog[(preparation.action_type, preparation.tool_id)][
                "max_response_bytes"
            ]
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider response exceeds artifact bound"
            )
        ingestion: Mapping[str, Any] | None = None
        response_rejected = False
        if response is not None:
            try:
                ingestion = self._protocol.ingest_provider_response(
                    action,
                    response,
                )
            except Exception:
                # The provider has returned a known terminal response. A
                # model-contract rejection is therefore a known adapter
                # failure, not an unknown paid-call outcome. Retain only the
                # response hash in the receipt and let the model-owned
                # waterfall handle the failed completion.
                response = None
                succeeded = False
                response_rejected = True
            else:
                if not isinstance(ingestion, Mapping):
                    response = None
                    succeeded = False
                    response_rejected = True
                    ingestion = None
                else:
                    ingestion = dict(ingestion)
        outcome = (
            ProviderOutcome.VERIFIED.value
            if succeeded
            else ProviderOutcome.ADAPTER_FAILURE.value
        )
        cost_credits = round(
            preparation.credit_cap_microunits / 1_000_000
            * calls
            / max(1, preparation.call_cap),
            6,
        )
        request_fingerprint = _prefixed_hash(
            preparation.request_fingerprint_sha256,
            "official baseline request fingerprint",
        )
        receipt_identity = {
            "binding_id": "official_baseline." + preparation.tool_id,
            "tool_id": preparation.tool_id,
            "binding_version": str(dispatch["compiler_id"]),
            "source_lineage_id": "model_release:" + str(
                self._registration.artifact_identity["commit_sha"]
            ),
            "unit_ref": preparation.unit_ref,
            "request_fingerprint": request_fingerprint,
            "outcome": outcome,
            "evidence_hash": sha256_json(
                response
                if response is not None
                else {
                    "provider": provider,
                    "status_code": status,
                    "body_sha256": sha256_json(body),
                }
            ),
            "credit_microunits": int(round(cost_credits * 1_000_000)),
            "latency_ms": elapsed_ms,
            "execution_mode": ReceiptExecutionMode.MEASURED_LAB.value,
            "call_count": calls,
        }
        receipt = ProviderReceipt(
            receipt_ref="provider_receipt:"
            + sha256_json(receipt_identity).removeprefix("sha256:")[:16],
            **receipt_identity,
        )
        if validate_provider_receipt(receipt):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider receipt is invalid"
            )
        provider_identity = hashlib.sha256(provider.encode("utf-8")).hexdigest()
        host = HostActionResult(
            outcome="succeeded" if succeeded else "failed",
            reason_code=(
                "protected_provider_verified"
                if succeeded
                else (
                    "protected_provider_response_rejected"
                    if response_rejected
                    else "protected_provider_adapter_failure"
                )
            ),
            provider_response=response,
            calls=calls,
            cost_credits=cost_credits,
            latency_ms=float(elapsed_ms),
            provider_request_id=self._request_ref(dispatch),
            provider_receipt_ref=receipt.receipt_ref,
            provider_identity_sha256=provider_identity,
        )
        binding_host = (
            replace(
                host,
                model_provider_response_ingestion=ingestion,
            )
            if getattr(
                self._protocol,
                "requires_raw_provider_response_custody",
                False,
            )
            else host
        )
        binding = self._protocol.build_provider_receipt_binding(
            action, binding_host
        )
        if (
            not isinstance(binding, Mapping)
            or binding.get("provider_receipt_ref") != host.provider_receipt_ref
            or binding.get("provider_identity_sha256")
            != host.provider_identity_sha256
            or _BARE_HASH_RE.fullmatch(str(binding.get("receipt_sha256") or ""))
            is None
            or (
                ingestion is not None
                and binding.get("provider_response_sha256")
                != ingestion.get("parsed_response_sha256")
            )
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline artifact provider receipt binding differs"
            )
        bound_host = replace(
            host,
            provider_receipt_sha256=str(binding["receipt_sha256"]),
        )
        protected = ProtectedModelActionResult(
            host_result=bound_host,
            provider_receipt=receipt,
            replay_ref={
                "schema_version": PROTECTED_PROVIDER_REPLAY_SCHEMA_VERSION,
                "preparation_sha256": preparation.preparation_sha256,
                "protected_job_ref": preparation.protected_job_ref,
                "dispatch_sha256": _prefixed_hash(
                    dispatch.get("dispatch_sha256"),
                    "official baseline dispatch hash",
                ),
                "provider_request_ref": self._request_ref(dispatch),
                "provider_receipt_ref": bound_host.provider_receipt_ref,
                "provider_receipt_sha256": bound_host.provider_receipt_sha256,
            },
            model_provider_response_ingestion=ingestion,
        )
        return self._known_terminal(
            preparation=preparation,
            protected=protected,
            provider_request_ref=self._request_ref(dispatch),
        )

    def _verifier_terminal(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
        started: float,
    ) -> OfficialBaselineProtectedTerminal:
        execution = self._protocol.execute_verifier_action(action)
        if not isinstance(execution, Mapping) or set(execution) != {
            "schema_version",
            "action_sha256",
            "action_type",
            "calls",
            "cost_credits",
            "provider_receipt_allowed",
            "result",
            "result_sha256",
            "execution_sha256",
        }:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline artifact verifier result is not closed"
            )
        body = dict(execution)
        claimed = _bare_hash(
            body.pop("execution_sha256"),
            "official baseline verifier execution hash",
        )
        result = execution.get("result")
        if (
            execution.get("schema_version")
            != "model-runner-verifier-execution:v1"
            or execution.get("action_sha256") != action.get("action_sha256")
            or execution.get("action_type") != action.get("action_type")
            or execution.get("calls") != 0
            or execution.get("cost_credits") != 0.0
            or execution.get("provider_receipt_allowed") is not False
            or not isinstance(result, Mapping)
            or _bare_hash(
                execution.get("result_sha256"),
                "official baseline verifier result hash",
            )
            != sha256_json(dict(result)).removeprefix("sha256:")
            or claimed != sha256_json(body).removeprefix("sha256:")
        ):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline artifact verifier identity differs"
            )
        latency_ms = max(0, int(math.ceil((self._clock() - started) * 1000)))
        reason = str(result.get("reason_code") or "artifact_verifier_completed")
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,127}", reason):
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline artifact verifier reason is invalid"
            )
        protected = ProtectedModelActionResult(
            host_result=HostActionResult(
                outcome="succeeded",
                reason_code=reason,
                provider_response=dict(result),
                calls=0,
                cost_credits=0.0,
                latency_ms=float(latency_ms),
            )
        )
        return self._known_terminal(
            preparation=preparation,
            protected=protected,
            provider_request_ref=None,
        )

    @staticmethod
    def _known_terminal(
        *,
        preparation: OfficialBaselineProtectedPreparation,
        protected: ProtectedModelActionResult,
        provider_request_ref: str | None,
    ) -> OfficialBaselineProtectedTerminal:
        result_sha256 = sha256_json(_protected_result_document(protected))
        terminal_body = {
            "schema_version": PROTECTED_PROVIDER_TERMINAL_RECEIPT_SCHEMA_VERSION,
            "preparation_sha256": preparation.preparation_sha256,
            "protected_job_ref": preparation.protected_job_ref,
            "protected_result_sha256": result_sha256,
            "provider_request_ref": provider_request_ref,
            "model_provider_response_sha256": sha256_json(
                protected.host_result.provider_response
            ),
        }
        terminal_sha256 = sha256_json(terminal_body)
        return OfficialBaselineProtectedTerminal(
            state="known",
            protected_action_result=protected,
            protected_result_sha256=result_sha256,
            protected_terminal_receipt_ref=(
                "official_terminal:" + terminal_sha256.removeprefix("sha256:")
            ),
            protected_terminal_receipt_sha256=terminal_sha256,
            provider_request_ref=provider_request_ref,
            model_provider_response_sha256=terminal_body[
                "model_provider_response_sha256"
            ],
        )

    def execute_prepared(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedTerminal:
        _expected, dispatch = self._preparation_for_action(
            preparation=preparation,
            action=action,
        )
        if preparation.action_type in _PROVIDER_ACTION_TYPES:
            if dispatch is None:
                raise OfficialBaselineProtectedAuthorityError(
                    "official baseline provider dispatch is unavailable"
                )
            return self._execute_provider(
                preparation=preparation,
                dispatch=dispatch,
                action=action,
                reconcile_only=False,
            )
        return self._verifier_terminal(
            preparation=preparation,
            action=action,
            started=self._clock(),
        )

    def reconcile(
        self,
        *,
        preparation: OfficialBaselineProtectedPreparation,
        action: Mapping[str, Any],
    ) -> OfficialBaselineProtectedTerminal:
        _expected, dispatch = self._preparation_for_action(
            preparation=preparation,
            action=action,
        )
        if preparation.action_type not in _PROVIDER_ACTION_TYPES:
            # This is a release-bound, zero-network pure verifier. Recomputing
            # it after an interrupted terminal write is a deterministic replay,
            # not a physical redispatch.
            return self._verifier_terminal(
                preparation=preparation,
                action=action,
                started=self._clock(),
            )
        if dispatch is None:
            raise OfficialBaselineProtectedAuthorityError(
                "official baseline provider dispatch is unavailable"
            )
        return self._execute_provider(
            preparation=preparation,
            dispatch=dispatch,
            action=action,
            reconcile_only=True,
        )


def load_official_baseline_release_components(
    *,
    context: OfficialBaselineDependencyContext,
    custody: S3OfficialBaselineDocumentCustody,
) -> OfficialBaselineReleaseComponents:
    """Build one exact artifact-derived registration, projector, and bridge."""

    if (
        not isinstance(context, OfficialBaselineDependencyContext)
        or not isinstance(custody, S3OfficialBaselineDocumentCustody)
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline release component context is invalid"
        )
    authority_identity_sha256 = _validated_protected_action_authority_sha256()
    context.validate()
    release_identity = _release_identity(context)
    runner = DockerPrivateModelRunner(context.spec)
    transport = DockerModelRunnerTransport(runner)
    protocol = ResearchLabModelRunnerProtocol(
        transport=transport,
        expected_release_identity=release_identity,
    )
    for method_name in (
        "official_host_binding_catalog",
        "build_official_host_capability_manifest",
        "prepare_provider_request",
        "ingest_provider_response",
        "execute_verifier_action",
    ):
        if not callable(getattr(protocol, method_name, None)):
            raise OfficialBaselineAuthorityUnavailable(
                f"official baseline artifact operation {method_name} is unavailable"
            )
    catalog = protocol.official_host_binding_catalog()
    bindings = _catalog_bindings(catalog)
    inventory = protocol.provider_compiler_inventory()
    _validate_inventory_catalog(inventory, catalog)
    proxy_url = _proxy_base_url(context.evidence_proxy_url)
    availability = _official_host_availability(
        catalog=catalog,
        inventory=inventory,
        ready_provider_ids=context.evidence_proxy_ready_provider_ids,
    )
    manifest = protocol.build_official_host_capability_manifest(availability)
    expected_manifest_bindings = [
        {**row, "available": availability[row["tool_id"]]} for row in bindings
    ]
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("bindings") != expected_manifest_bindings
        or _bare_hash(
            manifest.get("binding_contracts_sha256"),
            "official baseline host binding manifest hash",
        )
        != _bare_hash(
            catalog.get("binding_contracts_sha256"),
            "official baseline artifact catalog hash",
        )
        or _bare_hash(
            release_identity.get("tool_binding_manifest_sha256"),
            "official baseline release binding manifest hash",
        )
        != _bare_hash(
            catalog.get("binding_contracts_sha256"),
            "official baseline artifact catalog hash",
        )
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact host capability manifest differs"
        )
    artifact_identity = {
        "repository": "leadpoet/Sourcing_model",
        "branch": context.source_branch,
        "commit_sha": context.artifact.git_commit_sha,
        "model_artifact_hash": context.artifact.model_artifact_hash,
        "manifest_hash": context.artifact.manifest_hash,
        "routing_contract_hash": "sha256:"
        + _bare_hash(
            release_identity.get("consumer_contract_sha256"),
            "official baseline consumer contract hash",
        ),
        "routing_catalog_hash": "sha256:"
        + _bare_hash(
            release_identity.get("catalog_sha256"),
            "official baseline routing catalog hash",
        ),
        "routing_policy_hash": "sha256:"
        + _bare_hash(
            release_identity.get("policy_sha256"),
            "official baseline routing policy hash",
        ),
        "feature_schema_hash": "sha256:"
        + _bare_hash(
            release_identity.get("feature_schema_sha256"),
            "official baseline feature schema hash",
        ),
    }
    if (
        not _COMMIT_RE.fullmatch(context.artifact.git_commit_sha)
        or release_identity.get("source_commit")
        != context.artifact.git_commit_sha
        or _bare_hash(
            release_identity.get("verifier_artifact_digest"),
            "official baseline verifier artifact digest",
        )
        != _bare_hash(
            release_identity.get("model_artifact_digest"),
            "official baseline model artifact digest",
        )
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact or verifier release identity differs"
        )
    registration = ExactModelRunnerRegistration(
        artifact_identity=artifact_identity,
        protocol=protocol,
        host_capability_manifest=dict(manifest),
    )
    registration.validate_identity()
    generation = registration.protocol_generation
    if not generation.supports_official_baseline:
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact exact protocol is unavailable"
        )
    selection = protocol.build_official_baseline_execution(
        protected_action_authority_sha256=authority_identity_sha256
    )
    if (
        selection != dict(context.selection.selection_document)
        or selection.get("protocol_generation_sha256")
        != generation.protocol_generation_sha256
        or selection.get("protected_action_authority_sha256")
        != authority_identity_sha256
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline signed release selection differs"
        )
    registration.preflight(execution_mode="full_company")
    compiler_preflight = protocol.provider_compiler_preflight(manifest)
    preflight_payload = (
        {
            key: item
            for key, item in compiler_preflight.items()
            if key != "preflight_sha256"
        }
        if isinstance(compiler_preflight, Mapping)
        else {}
    )
    if (
        not isinstance(compiler_preflight, Mapping)
        or compiler_preflight.get("manifest_sha256")
        != manifest.get("manifest_sha256")
        or compiler_preflight.get("inventory_sha256")
        != inventory.get("inventory_sha256")
        or compiler_preflight.get("full_company_ready") is not True
        or compiler_preflight.get("available_closed_bindings") != []
        or compiler_preflight.get("reason_codes") != []
        or _bare_hash(
            compiler_preflight.get("preflight_sha256"),
            "official baseline compiler preflight hash",
        )
        != sha256_json(preflight_payload).removeprefix("sha256:")
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact compiler preflight failed"
        )
    projector = ArtifactProtocolBenchmarkProjector(registration)
    if (
        projector.protocol_generation_sha256
        != selection["protocol_generation_sha256"]
        or projector.projection_identity_sha256
        != selection["benchmark_projection_sha256"]
    ):
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline artifact projector identity differs"
        )
    executor = ArtifactPreparedActionExecutor(
        registration=registration,
        catalog=catalog,
        inventory=inventory,
        custody=custody,
        proxy_url=proxy_url,
    )
    bridge = GatewayLocalProtectedActionBridge(
        custody=custody,
        executor=executor,
    )
    if bridge.authority_identity_sha256 != selection[
        "protected_action_authority_sha256"
    ]:
        raise OfficialBaselineAuthorityUnavailable(
            "official baseline protected bridge identity differs"
        )
    return OfficialBaselineReleaseComponents(
        registration=registration,
        projector=projector,
        protected_bridge=bridge,
    )


__all__ = [
    "ArtifactPreparedActionExecutor",
    "OFFICIAL_BINDING_CATALOG_SCHEMA_VERSION",
    "PROTECTED_PROVIDER_PROGRESS_SCHEMA_VERSION",
    "SITE_PROTECTED_ACTION_AUTHORITY_SOURCE_CALLABLE",
    "SITE_PROTECTED_ACTION_AUTHORITY_SOURCE_COMMIT",
    "load_official_baseline_release_components",
    "preflight_official_baseline_artifact_protocol",
    "protected_action_authority_contract_identity",
]
