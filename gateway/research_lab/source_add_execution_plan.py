"""Fail-closed validation for executable SOURCE_ADD model plans.

The model plan is a public semantic contract. It carries one credential-free
request and a bounded response projection; the gateway binds the provider id
to the already tested/provisioned private transport.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Mapping, Sequence


SOURCE_ADD_STATIC_JSON_INTENT_PLAN_SCHEMA_VERSION = (
    "source-add-static-json-intent-plan:v1"
)
SOURCE_ADD_STATIC_JSON_INTENT_COMPILER_ID = (
    "source_add.static_json_intent:v1"
)
SOURCE_ADD_SIGNAL_BOUND_JSON_INTENT_PLAN_SCHEMA_VERSION = (
    "source-add-signal-bound-json-intent-plan:v1"
)
SOURCE_ADD_SIGNAL_BOUND_JSON_INTENT_COMPILER_ID = (
    "source_add.signal_bound_json_intent:v1"
)
SOURCE_ADD_STATIC_JSON_INTENT_MAX_RESPONSE_BYTES = 1_048_576

_SUPPORTED_SCHEMA_VERSIONS = frozenset({
    SOURCE_ADD_STATIC_JSON_INTENT_PLAN_SCHEMA_VERSION,
    SOURCE_ADD_SIGNAL_BOUND_JSON_INTENT_PLAN_SCHEMA_VERSION,
})

_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,79}$")
_TOOL_ID_RE = re.compile(r"^intent\.source_add\.[a-z][a-z0-9_-]{1,79}$")
_FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_QUERY_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._~-]+$")
_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|bearer|cookie|credential|password|secret|token)",
    re.IGNORECASE,
)
_CANONICAL_SOURCE_DOMAIN_FIELD = "canonical_source_domain"
_LEGACY_CANONICAL_URL_HOST_FIELD = "canonical_url_host"


class SourceAddExecutionPlanError(ValueError):
    """An executable SOURCE_ADD plan violated the public contract."""


def _exact_mapping(
    value: Any,
    *,
    field_name: str,
    fields: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SourceAddExecutionPlanError(
            f"{field_name} fields differ from the contract"
        )
    return dict(value)


def _canonical_path(value: Any, *, field_name: str) -> str:
    path = str(value or "").strip()
    if (
        not path.startswith("/")
        or "?" in path
        or "#" in path
        or "\\" in path
        or "//" in path
    ):
        raise SourceAddExecutionPlanError(f"{field_name} is invalid")
    segments = [segment for segment in path.split("/") if segment]
    if not segments or any(
        segment in {".", ".."}
        or _PATH_SEGMENT_RE.fullmatch(segment) is None
        for segment in segments
    ):
        raise SourceAddExecutionPlanError(f"{field_name} is invalid")
    return "/" + "/".join(segments)


def _public_domain(value: Any) -> str:
    domain = str(value or "").strip().casefold().rstrip(".")
    if (
        _DOMAIN_RE.fullmatch(domain) is None
        or domain.endswith((".local", ".localhost", ".internal", ".example"))
    ):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD response canonical source domain is invalid"
        )
    try:
        ipaddress.ip_address(domain)
    except ValueError:
        return domain
    raise SourceAddExecutionPlanError(
        "SOURCE_ADD response canonical source domain must not be an IP address"
    )


def _field_name(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if _FIELD_RE.fullmatch(text) is None or _SECRET_KEY_RE.search(text):
        raise SourceAddExecutionPlanError(f"{field_name} is invalid")
    return text


def _query_key(value: Any) -> str:
    key = str(value or "").strip()
    if _QUERY_KEY_RE.fullmatch(key) is None or _SECRET_KEY_RE.search(key):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD execution request query is invalid"
        )
    return key


def _signal_parameter_binding(value: Any) -> dict[str, Any]:
    binding = _exact_mapping(
        value,
        field_name="SOURCE_ADD signal-bound query",
        fields=frozenset({"source", "name", "max_length"}),
    )
    name = _field_name(
        binding["name"],
        field_name="SOURCE_ADD signal-bound query parameter",
    )
    max_length = binding["max_length"]
    if (
        binding["source"] != "signal_temporal_parameter"
        or isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or not 1 <= max_length <= 256
    ):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD signal-bound query is invalid"
        )
    return {
        "source": "signal_temporal_parameter",
        "name": name,
        "max_length": max_length,
    }


def _string_sequence(
    value: Any,
    *,
    field_name: str,
    maximum: int,
) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
    ):
        raise SourceAddExecutionPlanError(f"{field_name} must be an array")
    output: list[str] = []
    for item in value:
        normalized = _field_name(item, field_name=field_name)
        if normalized not in output:
            output.append(normalized)
    if not output or len(output) > maximum:
        raise SourceAddExecutionPlanError(f"{field_name} is out of bounds")
    return tuple(output)


def normalize_source_add_execution_plan(
    value: Mapping[str, Any],
    *,
    provider_id: str,
    tool_id: str,
    stage: str,
    execution_mode: str,
    intent_categories: Sequence[str],
    max_calls: int,
    max_results: int,
) -> dict[str, Any]:
    """Mirror the signed model's one-call GET/JSON plan exactly."""

    normalized_provider = str(provider_id or "").strip().casefold()
    normalized_tool = str(tool_id or "").strip()
    if (
        _PROVIDER_ID_RE.fullmatch(normalized_provider) is None
        or _TOOL_ID_RE.fullmatch(normalized_tool) is None
        or normalized_tool != f"intent.source_add.{normalized_provider}"
        or stage != "intent_evidence"
        or execution_mode != "invoke"
        or type(max_calls) is not int
        or max_calls != 1
        or type(max_results) is not int
        or max_results != 1
    ):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD JSON plan registration is incompatible"
        )
    plan = _exact_mapping(
        value,
        field_name="SOURCE_ADD execution plan",
        fields=frozenset(
            {
                "schema_version",
                "provider_id",
                "tool_id",
                "request",
                "response_projection",
            }
        ),
    )
    schema_version = str(plan["schema_version"] or "")
    if (
        schema_version not in _SUPPORTED_SCHEMA_VERSIONS
        or str(plan["provider_id"] or "").strip().casefold()
        != normalized_provider
        or str(plan["tool_id"] or "").strip() != normalized_tool
    ):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD execution plan identity differs"
        )
    request = _exact_mapping(
        plan["request"],
        field_name="SOURCE_ADD execution request",
        fields=frozenset({"method", "path", "query"}),
    )
    if request["method"] != "GET":
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD JSON plan method must be GET"
        )
    path = _canonical_path(
        request["path"], field_name="SOURCE_ADD execution request path"
    )
    raw_query = request["query"]
    if not isinstance(raw_query, Mapping) or not 1 <= len(raw_query) <= 8:
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD execution request query is out of bounds"
        )
    query: dict[str, Any] = {}
    seen_query_keys: set[str] = set()
    for raw_key, raw_value in raw_query.items():
        key = _query_key(raw_key)
        folded = key.casefold()
        if folded in seen_query_keys:
            raise SourceAddExecutionPlanError(
                "SOURCE_ADD execution request query is invalid"
            )
        seen_query_keys.add(folded)
        if schema_version == SOURCE_ADD_STATIC_JSON_INTENT_PLAN_SCHEMA_VERSION:
            value_text = str(raw_value or "").strip()
            if (
                not value_text
                or len(value_text) > 256
                or any(ord(character) < 32 for character in value_text)
            ):
                raise SourceAddExecutionPlanError(
                    "SOURCE_ADD execution request query is invalid"
                )
            query[key] = value_text
        else:
            query[key] = _signal_parameter_binding(raw_value)

    raw_projection = plan["response_projection"]
    projection_fields = {
        "kind",
        "category",
        "object_field",
        "identity_field",
        "canonical_url_field",
        "canonical_url_path_prefix",
        "excerpt_fields",
    }
    expected_identity_field = (
        "expected_identity"
        if schema_version == SOURCE_ADD_STATIC_JSON_INTENT_PLAN_SCHEMA_VERSION
        else "expected_identity_query_key"
    )
    if not isinstance(raw_projection, Mapping):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD response projection fields differ from the contract"
        )
    domain_fields = {
        _CANONICAL_SOURCE_DOMAIN_FIELD,
        _LEGACY_CANONICAL_URL_HOST_FIELD,
    } & set(raw_projection)
    if (
        len(domain_fields) != 1
        or set(raw_projection)
        != projection_fields | domain_fields | {expected_identity_field}
    ):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD response projection fields differ from the contract"
        )
    projection = dict(raw_projection)
    source_domain = _public_domain(projection[next(iter(domain_fields))])
    category = str(projection["category"] or "").strip().upper()
    categories = tuple(
        str(item or "").strip().upper() for item in intent_categories
    )
    expected_identity = ""
    expected_identity_query_key = ""
    if schema_version == SOURCE_ADD_STATIC_JSON_INTENT_PLAN_SCHEMA_VERSION:
        expected_identity = " ".join(
            str(projection["expected_identity"] or "").split()
        )
    else:
        expected_identity_query_key = _query_key(
            projection["expected_identity_query_key"]
        )
    if (
        projection["kind"] != "technology_context"
        or not category
        or category not in categories
        or (
            schema_version == SOURCE_ADD_STATIC_JSON_INTENT_PLAN_SCHEMA_VERSION
            and not 1 <= len(expected_identity) <= 160
        )
        or (
            schema_version
            == SOURCE_ADD_SIGNAL_BOUND_JSON_INTENT_PLAN_SCHEMA_VERSION
            and expected_identity_query_key not in query
        )
    ):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD response projection identity is invalid"
        )
    object_field = _field_name(
        projection["object_field"],
        field_name="SOURCE_ADD response object_field",
    )
    identity_field = _field_name(
        projection["identity_field"],
        field_name="SOURCE_ADD response identity_field",
    )
    canonical_url_field = _field_name(
        projection["canonical_url_field"],
        field_name="SOURCE_ADD response canonical_url_field",
    )
    excerpt_fields = _string_sequence(
        projection["excerpt_fields"],
        field_name="SOURCE_ADD response excerpt_fields",
        maximum=4,
    )
    if len(
        {object_field, identity_field, canonical_url_field, *excerpt_fields}
    ) != 3 + len(excerpt_fields):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD response projection fields overlap"
        )
    path_prefix = _canonical_path(
        projection["canonical_url_path_prefix"],
        field_name="SOURCE_ADD response canonical_url_path_prefix",
    )
    if not path_prefix.endswith("/"):
        path_prefix += "/"

    return {
        "schema_version": schema_version,
        "provider_id": normalized_provider,
        "tool_id": normalized_tool,
        "request": {
            "method": "GET",
            "path": path,
            "query": {key: query[key] for key in sorted(query)},
        },
        "response_projection": {
            "kind": "technology_context",
            "category": category,
            "object_field": object_field,
            "identity_field": identity_field,
            **(
                {"expected_identity": expected_identity}
                if schema_version
                == SOURCE_ADD_STATIC_JSON_INTENT_PLAN_SCHEMA_VERSION
                else {
                    "expected_identity_query_key": expected_identity_query_key
                }
            ),
            "canonical_url_field": canonical_url_field,
            # This is a public citation-domain constraint, not the provider's
            # private transport host. Canonicalize the legacy input name so
            # model release metadata retains its strict host-binding ban.
            _CANONICAL_SOURCE_DOMAIN_FIELD: source_domain,
            "canonical_url_path_prefix": path_prefix,
            "excerpt_fields": list(excerpt_fields),
        },
    }


def bind_source_add_execution_plan_to_probes(
    plan: Mapping[str, Any],
    *,
    provider_id: str,
    probe_endpoints: Sequence[Mapping[str, Any]],
    tested_probes: Sequence[Mapping[str, Any]],
) -> None:
    """Require the model request to be the exact operator-tested route."""

    request = plan.get("request")
    if not isinstance(request, Mapping):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD execution request is unavailable"
        )
    method = str(request.get("method") or "")
    path = str(request.get("path") or "")
    query = request.get("query")
    if not isinstance(query, Mapping):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD execution request query is invalid"
        )
    matching_endpoints = [
        endpoint
        for endpoint in probe_endpoints
        if isinstance(endpoint, Mapping)
        and endpoint.get("provider_id") == provider_id
        and str(endpoint.get("method") or "").upper() == method
        and endpoint.get("path") == path
    ]
    if len(matching_endpoints) != 1:
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD execution plan must match one provisioned endpoint"
        )
    endpoint = matching_endpoints[0]
    raw_params = endpoint.get("params") or []
    if isinstance(raw_params, Mapping):
        params = [
            {"name": name, **(dict(spec) if isinstance(spec, Mapping) else {})}
            for name, spec in raw_params.items()
        ]
    elif isinstance(raw_params, Sequence) and not isinstance(
        raw_params, (str, bytes)
    ):
        params = [dict(item) for item in raw_params if isinstance(item, Mapping)]
    else:
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD provisioned endpoint params are invalid"
        )
    parameter_names: dict[str, Mapping[str, Any]] = {}
    for param in params:
        name = str(param.get("name") or "")
        folded = name.casefold()
        if not name or folded in parameter_names:
            raise SourceAddExecutionPlanError(
                "SOURCE_ADD provisioned endpoint params are invalid"
            )
        parameter_names[folded] = param
    query_names = {str(key).casefold() for key in query}
    if (
        any(
            key not in parameter_names
            or parameter_names[key].get("location", "query") != "query"
            for key in query_names
        )
        or any(
            param.get("location", "query") == "body" for param in params
        )
        or any(
            bool(param.get("required"))
            and str(param.get("name") or "").casefold() not in query_names
            for param in params
        )
    ):
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD execution query differs from the provisioned endpoint"
        )
    schema_version = str(plan.get("schema_version") or "")
    dynamic_query = (
        schema_version
        == SOURCE_ADD_SIGNAL_BOUND_JSON_INTENT_PLAN_SCHEMA_VERSION
    )
    if dynamic_query:
        for raw_key, raw_binding in query.items():
            binding = _signal_parameter_binding(raw_binding)
            endpoint_parameter = parameter_names[str(raw_key).casefold()]
            endpoint_type = endpoint_parameter.get("type")
            endpoint_max_length = endpoint_parameter.get("max_length")
            if endpoint_type not in {None, "", "string"}:
                raise SourceAddExecutionPlanError(
                    "SOURCE_ADD execution query differs from the provisioned endpoint"
                )
            if endpoint_max_length is not None and (
                isinstance(endpoint_max_length, bool)
                or not isinstance(endpoint_max_length, int)
                or endpoint_max_length < binding["max_length"]
            ):
                raise SourceAddExecutionPlanError(
                    "SOURCE_ADD execution query differs from the provisioned endpoint"
                )
    expected_query = {str(key): str(value) for key, value in query.items()}

    def tested_query_matches(probe: Mapping[str, Any]) -> bool:
        raw_test_query = probe.get("query")
        if not isinstance(raw_test_query, Mapping):
            return False
        if not dynamic_query:
            return {
                str(key): str(value)
                for key, value in raw_test_query.items()
            } == expected_query
        if set(raw_test_query) != set(query):
            return False
        for key, binding in query.items():
            value = raw_test_query.get(key)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > int(binding["max_length"])
                or any(ord(character) < 32 for character in value)
            ):
                return False
        return True

    matching_tests = [
        probe
        for probe in tested_probes
        if isinstance(probe, Mapping)
        and str(probe.get("method") or "").upper() == method
        and probe.get("path") == path
        and probe.get("body_json") is None
        and tested_query_matches(probe)
    ]
    if len(matching_tests) != 1:
        raise SourceAddExecutionPlanError(
            "SOURCE_ADD execution plan must match one successful test probe"
        )


def is_supported_source_add_execution_plan(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("schema_version") in _SUPPORTED_SCHEMA_VERSIONS
    )


def source_add_execution_plan_compiler_id(value: Mapping[str, Any]) -> str:
    schema_version = (
        value.get("schema_version") if isinstance(value, Mapping) else None
    )
    if schema_version == SOURCE_ADD_STATIC_JSON_INTENT_PLAN_SCHEMA_VERSION:
        return SOURCE_ADD_STATIC_JSON_INTENT_COMPILER_ID
    if schema_version == SOURCE_ADD_SIGNAL_BOUND_JSON_INTENT_PLAN_SCHEMA_VERSION:
        return SOURCE_ADD_SIGNAL_BOUND_JSON_INTENT_COMPILER_ID
    raise SourceAddExecutionPlanError(
        "SOURCE_ADD execution plan schema is unsupported"
    )


__all__ = [
    "SOURCE_ADD_STATIC_JSON_INTENT_COMPILER_ID",
    "SOURCE_ADD_STATIC_JSON_INTENT_MAX_RESPONSE_BYTES",
    "SOURCE_ADD_STATIC_JSON_INTENT_PLAN_SCHEMA_VERSION",
    "SOURCE_ADD_SIGNAL_BOUND_JSON_INTENT_COMPILER_ID",
    "SOURCE_ADD_SIGNAL_BOUND_JSON_INTENT_PLAN_SCHEMA_VERSION",
    "SourceAddExecutionPlanError",
    "bind_source_add_execution_plan_to_probes",
    "is_supported_source_add_execution_plan",
    "normalize_source_add_execution_plan",
    "source_add_execution_plan_compiler_id",
]
