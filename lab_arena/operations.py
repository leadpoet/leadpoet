"""Closed provider operation table for Lab Arena V1 (labarena.md section 7.4).

Pure data plus validation; nothing here performs I/O. The table is the only
way a model reaches a provider: every request is matched to exactly one
operation by method, constant host, and exact path, its parameters are
validated against a closed schema, and only the operation id plus validated
parameters cross the sandbox socket. The broker (``lab_arena/broker.py``)
builds the outbound URL, headers, and credentials itself from the constants
in this table; no request field can alter host, path, method, headers, or
credentials, and the tests prove it field by field.

Cost rules: TAO-funded operations (Exa, ScrapingDog) carry a fixed published
price from ``PROVIDER_PRICE_LIST`` and every price-determining parameter is
pinned in ``fixed_params``, so the reserved estimate is the settled price.
``openrouter.chat`` is priced by the broker from the round's pinned price
table; this module only caps the model-controllable output tokens.

``OPERATION_TABLE_HASH`` and ``PROVIDER_PRICE_LIST_HASH`` are bound into the
signed round configuration (``operation_table_hash`` and
``provider_price_list_hash``).

Structural limits: ``contracts.REQUEST_LIMITS`` caps strings at 8 KiB, which
contradicts the 32,000-character chat ``content`` field, so operation
parameters use ``OPERATION_LIMITS`` below and then the per-operation
``max_request_bytes`` cap.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit

from lab_arena import contracts

OPERATION_TABLE_SCHEMA_VERSION = "leadpoet.lab_arena.operation_table.v1"
PRICE_LIST_SCHEMA_VERSION = "leadpoet.lab_arena.provider_price_list.v1"

PROVIDERS = ("exa", "scrapingdog", "openrouter")
FUNDING_SOURCES = ("tao", "openrouter")
METHODS = ("GET", "POST")
PARAMETER_LOCATIONS = ("body", "query")
RESPONSE_SANITIZERS = ("json", "text")
FIELD_KINDS = ("str", "int", "float", "bool", "list[str]", "list[object]", "object")
FIELD_FORMATS = ("https_url", "iso_date", "domain", "model_id")

OPENROUTER_MAX_OUTPUT_TOKENS = 4096
OPENROUTER_MAX_MESSAGES = 64
OPENROUTER_MAX_CONTENT_CHARS = 32_000
# Copied from gateway/research_lab/key_vault.py (section 3.1): the broker
# injects this policy into every chat body; it is table data so its hash is
# bound into the round configuration.
OPENROUTER_STRICT_PROVIDER_POLICY: Mapping[str, Any] = MappingProxyType(
    {"data_collection": "deny", "allow_fallbacks": False}
)

# Structural limits for operation parameters (see module docstring). They are
# deliberately looser than every field schema so the schema produces the
# precise error code; ``max_total_bytes`` is the ceiling on any operation's
# ``max_request_bytes``, which is enforced per operation.
OPERATION_LIMITS = contracts.StrictLimits(
    max_depth=6,
    max_list_items=128,
    max_object_keys=64,
    max_string_bytes=4 * OPENROUTER_MAX_CONTENT_CHARS,
    max_total_bytes=1_000_000,
)
# The structural pass never applies the total-size check itself: size is the
# per-operation cap below and reports ``request_too_large``.
_STRUCTURE_LIMITS = dataclasses.replace(OPERATION_LIMITS, max_total_bytes=2 ** 31)

ERROR_CODES = frozenset(
    {
        "no_matching_operation",
        "invalid_request",
        "request_too_large",
        "invalid_body",
        "invalid_query",
        "unknown_header",
        "forbidden_header",
        "unknown_field",
        "forbidden_field",
        "missing_field",
        "invalid_field",
        "invalid_url",
        "invalid_response",
        "response_too_large",
    }
)

# Request headers a client may send. Everything else is rejected: the broker
# constructs outbound headers itself and forwards none of these.
ALLOWED_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "accept-language",
        "cache-control",
        "connection",
        "content-length",
        "content-type",
        "date",
        "expect",
        "host",
        "keep-alive",
        "pragma",
        "te",
        "user-agent",
    }
)
# Caller-supplied credentials are refused rather than stripped (section 18.4).
CREDENTIAL_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "cookie2",
        "x-api-key",
        "api-key",
        "apikey",
        "x-auth-token",
        "x-access-token",
        "x-openrouter-api-key",
        "x-csrf-token",
        "x-amz-security-token",
    }
)

# Parameter names refused at the top level of every operation unless the
# operation declares them. Compared after lowercasing and mapping ``-`` to
# ``_`` so ``X-API-Key``, ``x_api_key`` and ``apikey`` all match.
FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "host",
        "hostname",
        "url",
        "urls",
        "base_url",
        "endpoint",
        "path",
        "method",
        "port",
        "scheme",
        "headers",
        "header",
        "authorization",
        "apikey",
        "api_key",
        "x_api_key",
        "cookie",
        "cookies",
        "redirect",
        "redirects",
        "follow_redirects",
        "allow_redirects",
        "proxy",
        "proxies",
        "auth",
        "credentials",
        "token",
        "bearer",
        # OpenRouter surfaces whose cost or routing the broker owns (7.3).
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "plugins",
        "provider",
        "models",
        "route",
        "stream",
        "stream_options",
        "reasoning",
        "transforms",
        "web_search_options",
        "modalities",
        "audio",
        "image",
        "images",
        "prediction",
    }
)

# Statuses that can only describe the Arena's own credential or account
# state; the model never sees the provider's explanation (section 7.4).
CREDENTIAL_STATUSES = frozenset({401, 403, 407})
GENERIC_UNAVAILABLE_STATUS = 502
GENERIC_UNAVAILABLE_BODY = b'{"error":{"code":"provider_unavailable"}}'

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DOMAIN_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_CONTENT_TYPE_RE = re.compile(r"^[a-z0-9.+-]+/[a-z0-9.+-]+$")
_INT_QUERY_RE = re.compile(r"^-?[0-9]{1,15}$")


class OperationError(RuntimeError):
    """Base error: ``code`` is always drawn from ``ERROR_CODES``.

    ``field`` names a schema-owned field path (never a model-supplied value)
    so the message stays generic and bounded.
    """

    def __init__(self, code: str, field: Optional[str] = None) -> None:
        if code not in ERROR_CODES:
            raise ValueError("unknown operation error code")
        self.code = code
        self.field = field
        super().__init__(code if field is None else "%s (%s)" % (code, field))


class OperationRequestError(OperationError):
    """A model request matched no operation or violated its schema."""


class OperationResponseError(OperationError):
    """A provider response could not be sanitized; the broker fails the call."""


# ---------------------------------------------------------------------------
# Table types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One request field.

    ``kind`` is from ``FIELD_KINDS``. ``max_length`` bounds characters for
    ``str`` and items for lists. ``format`` adds a semantic check from
    ``FIELD_FORMATS``. ``item`` constrains ``list[str]`` items and ``fields``
    is the closed schema of ``object`` and ``list[object]`` members.
    """

    kind: str
    required: bool = False
    min_length: int = 0
    max_length: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[Tuple[str, ...]] = None
    format: Optional[str] = None
    item: Optional["FieldSpec"] = None
    fields: Optional[Mapping[str, "FieldSpec"]] = None

    def __post_init__(self) -> None:
        if self.kind not in FIELD_KINDS:
            raise ValueError("unknown field kind")
        if self.format is not None and self.format not in FIELD_FORMATS:
            raise ValueError("unknown field format")
        if self.kind == "list[str]" and self.item is None:
            raise ValueError("list[str] needs an item spec")
        if self.kind in ("list[object]", "object") and self.fields is None:
            raise ValueError("object fields need a schema")
        if self.fields is not None:
            object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


@dataclass(frozen=True)
class CredentialPlacement:
    """Where the broker places the server-side credential: table data only."""

    location: str  # "header" | "query"
    name: str  # header name or query parameter name
    scheme: str = ""  # e.g. "Bearer" for Authorization headers


@dataclass(frozen=True)
class Operation:
    operation_id: str
    provider: str
    method: str
    host: str
    path: str
    parameter_location: str
    request_fields: Mapping[str, FieldSpec]
    fixed_params: Mapping[str, Any]
    defaults: Mapping[str, Any]
    timeout_seconds: int
    max_request_bytes: int
    max_response_bytes: int
    cost_rule: Mapping[str, Any]
    response_sanitizer: str
    funding_source: str
    credential: CredentialPlacement

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS or self.method not in METHODS:
            raise ValueError("operation provider or method is invalid")
        if self.parameter_location not in PARAMETER_LOCATIONS:
            raise ValueError("operation parameter location is invalid")
        if self.method == "GET" and self.parameter_location != "query":
            raise ValueError("GET operations carry query parameters only")
        if self.method == "POST" and self.parameter_location != "body":
            raise ValueError("POST operations carry a JSON body only")
        if self.response_sanitizer not in RESPONSE_SANITIZERS:
            raise ValueError("operation sanitizer is invalid")
        if self.funding_source not in FUNDING_SOURCES:
            raise ValueError("operation funding source is invalid")
        if not self.path.startswith("/") or "?" in self.path or "#" in self.path:
            raise ValueError("operation path is invalid")
        if not _is_dns_hostname(self.host):
            raise ValueError("operation host is invalid")
        if set(self.fixed_params) & set(self.request_fields):
            raise ValueError("fixed params may not overlap request fields")
        if not set(self.defaults) <= set(self.request_fields):
            raise ValueError("defaults must name request fields")
        if self.timeout_seconds <= 0 or self.max_request_bytes <= 0 or self.max_response_bytes <= 0:
            raise ValueError("operation bounds are invalid")
        object.__setattr__(self, "request_fields", MappingProxyType(dict(self.request_fields)))
        object.__setattr__(self, "fixed_params", MappingProxyType(_deep_copy_json(self.fixed_params)))
        object.__setattr__(self, "defaults", MappingProxyType(_deep_copy_json(self.defaults)))
        object.__setattr__(self, "cost_rule", MappingProxyType(_deep_copy_json(self.cost_rule)))

    @property
    def forbidden_names(self) -> frozenset:
        declared = {_normalize_name(name) for name in self.request_fields}
        return FORBIDDEN_FIELD_NAMES - declared


def _deep_copy_json(value: Any) -> Any:
    def _plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError("value is not JSON data")

    return json.loads(json.dumps(value, sort_keys=True, default=_plain))


def _normalize_name(name: str) -> str:
    return str(name).lower().replace("-", "_")


def _is_dns_hostname(host: str) -> bool:
    """A public DNS name: lowercase labels, no IP literal, alphabetic TLD."""

    if not isinstance(host, str) or not 1 <= len(host) <= 253:
        return False
    labels = host.split(".")
    if len(labels) < 2:
        return False
    if not all(_DOMAIN_LABEL_RE.match(label) for label in labels):
        return False
    return labels[-1].isalpha()


# ---------------------------------------------------------------------------
# Price list (section 7.5: TAO-funded settlement equals the published price)
# ---------------------------------------------------------------------------

PROVIDER_PRICE_LIST: Mapping[str, int] = MappingProxyType(
    {
        "exa.search": 5_000,
        # 1,000 microUSD per result times the fixed maximum of five URLs.
        "exa.contents": 5_000,
        "scrapingdog.scrape": 2_000,
        "scrapingdog.google": 5_000,
    }
)


def _fixed_cost(operation_id: str) -> Mapping[str, Any]:
    return {"kind": "fixed_microusd", "microusd": int(PROVIDER_PRICE_LIST[operation_id])}


# ---------------------------------------------------------------------------
# V1 operations
# ---------------------------------------------------------------------------

_EXA_CATEGORIES = (
    "company",
    "research paper",
    "news",
    "pdf",
    "github",
    "tweet",
    "personal site",
    "linkedin profile",
    "financial report",
)
_DOMAIN_ITEM = FieldSpec("str", min_length=1, max_length=253, format="domain")
_HTTPS_URL_ITEM = FieldSpec("str", min_length=8, max_length=2000, format="https_url")
_GOOGLE_COUNTRIES = ("us", "gb", "ca", "au", "de", "fr", "nl", "ie", "in", "sg")

_EXA_CREDENTIAL = CredentialPlacement("header", "x-api-key")
_SCRAPINGDOG_CREDENTIAL = CredentialPlacement("query", "api_key")
_OPENROUTER_CREDENTIAL = CredentialPlacement("header", "authorization", scheme="Bearer")

_OPERATION_LIST = (
    Operation(
        operation_id="exa.search",
        provider="exa",
        method="POST",
        host="api.exa.ai",
        path="/search",
        parameter_location="body",
        request_fields={
            "query": FieldSpec("str", required=True, min_length=1, max_length=1000),
            "type": FieldSpec("str", choices=("auto", "neural", "keyword")),
            "category": FieldSpec("str", choices=_EXA_CATEGORIES),
            "includeDomains": FieldSpec("list[str]", max_length=20, item=_DOMAIN_ITEM),
            "excludeDomains": FieldSpec("list[str]", max_length=20, item=_DOMAIN_ITEM),
            "startPublishedDate": FieldSpec("str", format="iso_date", min_length=10, max_length=10),
            "endPublishedDate": FieldSpec("str", format="iso_date", min_length=10, max_length=10),
        },
        fixed_params={"numResults": 10, "contents": {"text": {"maxCharacters": 2000}}},
        defaults={},
        timeout_seconds=30,
        max_request_bytes=16_384,
        max_response_bytes=1_048_576,
        cost_rule=_fixed_cost("exa.search"),
        response_sanitizer="json",
        funding_source="tao",
        credential=_EXA_CREDENTIAL,
    ),
    Operation(
        operation_id="exa.contents",
        provider="exa",
        method="POST",
        host="api.exa.ai",
        path="/contents",
        parameter_location="body",
        request_fields={
            "urls": FieldSpec("list[str]", required=True, min_length=1, max_length=5, item=_HTTPS_URL_ITEM),
        },
        fixed_params={"text": {"maxCharacters": 4000}},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=16_384,
        max_response_bytes=1_048_576,
        cost_rule=_fixed_cost("exa.contents"),
        response_sanitizer="json",
        funding_source="tao",
        credential=_EXA_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.scrape",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/scrape",
        parameter_location="query",
        request_fields={
            "url": FieldSpec("str", required=True, min_length=8, max_length=2000, format="https_url"),
            "dynamic": FieldSpec("bool"),
        },
        fixed_params={"premium": False},
        defaults={"dynamic": False},
        timeout_seconds=60,
        max_request_bytes=4_096,
        max_response_bytes=2_097_152,
        cost_rule=_fixed_cost("scrapingdog.scrape"),
        response_sanitizer="text",
        funding_source="tao",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.google",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/google",
        parameter_location="query",
        request_fields={
            "query": FieldSpec("str", required=True, min_length=1, max_length=500),
            "country": FieldSpec("str", choices=_GOOGLE_COUNTRIES),
        },
        fixed_params={"results": 10},
        defaults={"country": "us"},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=1_048_576,
        cost_rule=_fixed_cost("scrapingdog.google"),
        response_sanitizer="json",
        funding_source="tao",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="openrouter.chat",
        provider="openrouter",
        method="POST",
        host="openrouter.ai",
        path="/api/v1/chat/completions",
        parameter_location="body",
        request_fields={
            # Validated against the round's allowed model list by the broker.
            "model": FieldSpec("str", required=True, min_length=3, max_length=128, format="model_id"),
            "messages": FieldSpec(
                "list[object]",
                required=True,
                min_length=1,
                max_length=OPENROUTER_MAX_MESSAGES,
                fields={
                    "role": FieldSpec("str", required=True, choices=("system", "user", "assistant")),
                    "content": FieldSpec("str", required=True, max_length=OPENROUTER_MAX_CONTENT_CHARS),
                },
            ),
            "temperature": FieldSpec("float", minimum=0, maximum=2),
            "max_tokens": FieldSpec("int", minimum=1, maximum=OPENROUTER_MAX_OUTPUT_TOKENS),
            "top_p": FieldSpec("float", minimum=0, maximum=1),
            "stop": FieldSpec("list[str]", max_length=4, item=FieldSpec("str", min_length=1, max_length=64)),
            "seed": FieldSpec("int", minimum=-(2 ** 31), maximum=2 ** 31 - 1),
            "response_format": FieldSpec("object", fields={"type": FieldSpec("str", required=True, choices=("text",))}),
        },
        fixed_params={"stream": False, "provider": dict(OPENROUTER_STRICT_PROVIDER_POLICY)},
        # Absent output caps default to the server cap so cost stays bounded.
        defaults={"max_tokens": OPENROUTER_MAX_OUTPUT_TOKENS},
        timeout_seconds=120,
        max_request_bytes=1_000_000,
        max_response_bytes=1_048_576,
        cost_rule={"kind": "openrouter_price_table", "max_output_tokens": OPENROUTER_MAX_OUTPUT_TOKENS},
        response_sanitizer="json",
        funding_source="openrouter",
        credential=_OPENROUTER_CREDENTIAL,
    ),
)

OPERATIONS: Mapping[str, Operation] = MappingProxyType(
    {operation.operation_id: operation for operation in _OPERATION_LIST}
)
if len(OPERATIONS) != len(_OPERATION_LIST):
    raise RuntimeError("operation ids must be unique")
for _operation in _OPERATION_LIST:
    if _operation.funding_source == "tao" and _operation.operation_id not in PROVIDER_PRICE_LIST:
        raise RuntimeError("TAO-funded operation without a published price")
    if _operation.cost_rule["kind"] == "fixed_microusd" and _operation.cost_rule["microusd"] != PROVIDER_PRICE_LIST[_operation.operation_id]:
        raise RuntimeError("operation price differs from the price list")
    if _operation.cost_rule["kind"] not in ("fixed_microusd", "openrouter_price_table"):
        raise RuntimeError("operation cost rule is invalid")
    if _operation.max_request_bytes > OPERATION_LIMITS.max_total_bytes:
        raise RuntimeError("operation request cap exceeds the structural ceiling")
del _operation


def _operation(operation_id: Any) -> Operation:
    operation = OPERATIONS.get(operation_id) if isinstance(operation_id, str) else None
    if operation is None:
        raise OperationRequestError("no_matching_operation")
    return operation


# ---------------------------------------------------------------------------
# Deterministic documents and hashes
# ---------------------------------------------------------------------------


def field_spec_document(spec: FieldSpec) -> Dict[str, Any]:
    document: Dict[str, Any] = {"kind": spec.kind, "required": spec.required, "min_length": spec.min_length}
    if spec.max_length is not None:
        document["max_length"] = spec.max_length
    if spec.minimum is not None:
        document["minimum"] = spec.minimum
    if spec.maximum is not None:
        document["maximum"] = spec.maximum
    if spec.choices is not None:
        document["choices"] = list(spec.choices)
    if spec.format is not None:
        document["format"] = spec.format
    if spec.item is not None:
        document["item"] = field_spec_document(spec.item)
    if spec.fields is not None:
        document["fields"] = {name: field_spec_document(inner) for name, inner in spec.fields.items()}
    return document


def operation_document(operation: Operation) -> Dict[str, Any]:
    return {
        "operation_id": operation.operation_id,
        "provider": operation.provider,
        "method": operation.method,
        "host": operation.host,
        "port": 443,
        "path": operation.path,
        "parameter_location": operation.parameter_location,
        "request_fields": {name: field_spec_document(spec) for name, spec in operation.request_fields.items()},
        "fixed_params": _deep_copy_json(operation.fixed_params),
        "defaults": _deep_copy_json(operation.defaults),
        "timeout_seconds": operation.timeout_seconds,
        "max_request_bytes": operation.max_request_bytes,
        "max_response_bytes": operation.max_response_bytes,
        "cost_rule": _deep_copy_json(operation.cost_rule),
        "response_sanitizer": operation.response_sanitizer,
        "funding_source": operation.funding_source,
        "credential": {
            "location": operation.credential.location,
            "name": operation.credential.name,
            "scheme": operation.credential.scheme,
        },
    }


def operation_table_document() -> Dict[str, Any]:
    """The full table in operation-id order; hashed into the round configuration."""

    return {
        "schema_version": OPERATION_TABLE_SCHEMA_VERSION,
        "operation_limits": {
            "max_depth": OPERATION_LIMITS.max_depth,
            "max_list_items": OPERATION_LIMITS.max_list_items,
            "max_object_keys": OPERATION_LIMITS.max_object_keys,
            "max_string_bytes": OPERATION_LIMITS.max_string_bytes,
            "max_total_bytes": OPERATION_LIMITS.max_total_bytes,
        },
        "allowed_request_headers": sorted(ALLOWED_REQUEST_HEADERS),
        "credential_headers": sorted(CREDENTIAL_HEADERS),
        "forbidden_field_names": sorted(FORBIDDEN_FIELD_NAMES),
        "credential_statuses": sorted(CREDENTIAL_STATUSES),
        "operations": [operation_document(OPERATIONS[key]) for key in sorted(OPERATIONS)],
    }


def price_list_document() -> Dict[str, Any]:
    return {
        "schema_version": PRICE_LIST_SCHEMA_VERSION,
        "unit": "microusd",
        "prices": {key: int(PROVIDER_PRICE_LIST[key]) for key in sorted(PROVIDER_PRICE_LIST)},
    }


OPERATION_TABLE_HASH = contracts.document_hash(operation_table_document())
PROVIDER_PRICE_LIST_HASH = contracts.document_hash(price_list_document())


def fixed_cost_microusd(operation_id: str) -> Optional[int]:
    """The fixed maximum cost of a TAO-funded operation; ``None`` for OpenRouter.

    The value never depends on parameters: every price-determining parameter is
    pinned in ``fixed_params``.
    """

    rule = _operation(operation_id).cost_rule
    if rule["kind"] == "fixed_microusd":
        return int(rule["microusd"])
    return None


# ---------------------------------------------------------------------------
# URL and field validation
# ---------------------------------------------------------------------------


def validate_https_url(value: Any, *, max_length: int = 2000, field: str = "url") -> str:
    """Accept only ``https://<dns-name>[:443]/path[?query]``.

    Rejects non-string values, non-ASCII or whitespace, IP literals (IPv4,
    IPv6, and integer hosts), userinfo, fragments, explicit ports other than
    443, and backslashes. The string is returned unchanged.
    """

    if not isinstance(value, str) or not value or len(value) > max_length:
        raise OperationRequestError("invalid_url", field)
    if not value.isascii() or any(char.isspace() for char in value) or "\\" in value:
        raise OperationRequestError("invalid_url", field)
    if any(ord(char) < 0x21 or ord(char) == 0x7F for char in value):
        raise OperationRequestError("invalid_url", field)
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise OperationRequestError("invalid_url", field) from exc
    if parts.scheme != "https" or parts.fragment or "#" in value:
        raise OperationRequestError("invalid_url", field)
    if "@" in parts.netloc or not parts.hostname:
        raise OperationRequestError("invalid_url", field)
    if port is not None and port != 443:
        raise OperationRequestError("invalid_url", field)
    if parts.netloc.count(":") > 1 or parts.netloc.startswith("["):
        raise OperationRequestError("invalid_url", field)
    if not _is_dns_hostname(parts.hostname):
        raise OperationRequestError("invalid_url", field)
    if parts.path and not parts.path.startswith("/"):
        raise OperationRequestError("invalid_url", field)
    return value


def _check_format(spec: FieldSpec, value: str, path: str) -> str:
    if spec.format == "https_url":
        return validate_https_url(value, max_length=spec.max_length or 2000, field=path)
    if spec.format == "iso_date":
        if not _ISO_DATE_RE.match(value):
            raise OperationRequestError("invalid_field", path)
        try:
            _datetime.date.fromisoformat(value)
        except ValueError as exc:
            raise OperationRequestError("invalid_field", path) from exc
        return value
    if spec.format == "domain":
        if not _is_dns_hostname(value.lower()):
            raise OperationRequestError("invalid_field", path)
        return value.lower()
    if spec.format == "model_id":
        if not _MODEL_ID_RE.match(value):
            raise OperationRequestError("invalid_field", path)
        return value
    return value


def _validate_field(spec: FieldSpec, value: Any, path: str) -> Any:
    kind = spec.kind
    if kind == "bool":
        if not isinstance(value, bool):
            raise OperationRequestError("invalid_field", path)
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise OperationRequestError("invalid_field", path)
        if (spec.minimum is not None and value < spec.minimum) or (spec.maximum is not None and value > spec.maximum):
            raise OperationRequestError("invalid_field", path)
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OperationRequestError("invalid_field", path)
        number = float(value)
        if not math.isfinite(number):
            raise OperationRequestError("invalid_field", path)
        if (spec.minimum is not None and number < spec.minimum) or (spec.maximum is not None and number > spec.maximum):
            raise OperationRequestError("invalid_field", path)
        return number
    if kind == "str":
        if not isinstance(value, str):
            raise OperationRequestError("invalid_field", path)
        if len(value) < spec.min_length or (spec.max_length is not None and len(value) > spec.max_length):
            raise OperationRequestError("invalid_field", path)
        if spec.choices is not None and value not in spec.choices:
            raise OperationRequestError("invalid_field", path)
        return _check_format(spec, value, path)
    if kind in ("list[str]", "list[object]"):
        if not isinstance(value, (list, tuple)):
            raise OperationRequestError("invalid_field", path)
        if len(value) < spec.min_length or (spec.max_length is not None and len(value) > spec.max_length):
            raise OperationRequestError("invalid_field", path)
        if kind == "list[str]":
            assert spec.item is not None
            return [_validate_field(spec.item, item, "%s[%d]" % (path, index)) for index, item in enumerate(value)]
        assert spec.fields is not None
        return [_validate_object(spec.fields, item, "%s[%d]" % (path, index)) for index, item in enumerate(value)]
    if kind == "object":
        assert spec.fields is not None
        return _validate_object(spec.fields, value, path)
    raise OperationRequestError("invalid_field", path)


def _validate_object(
    fields: Mapping[str, FieldSpec],
    value: Any,
    path: str,
    *,
    forbidden: frozenset = frozenset(),
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OperationRequestError("invalid_field", path)
    for name in value:
        if not isinstance(name, str):
            raise OperationRequestError("invalid_field", path)
        if _normalize_name(name) in forbidden:
            raise OperationRequestError("forbidden_field", path)
        if name not in fields:
            raise OperationRequestError("unknown_field", path)
    out: Dict[str, Any] = {}
    for name, spec in fields.items():
        if name not in value or value[name] is None:
            if spec.required:
                raise OperationRequestError("missing_field", "%s.%s" % (path, name))
            continue
        out[name] = _validate_field(spec, value[name], "%s.%s" % (path, name))
    return out


def validate_operation_request(operation_id: str, parameters: Any) -> Dict[str, Any]:
    """Validate and normalize ``parameters`` for one operation.

    Unknown fields, forbidden names, wrong types, out-of-range values, and
    invalid URLs raise ``OperationRequestError`` with a generic code. Absent
    optional fields with a table default are filled. The normalized document
    must fit the operation's ``max_request_bytes`` in canonical JSON.
    """

    operation = _operation(operation_id)
    if not isinstance(parameters, Mapping):
        raise OperationRequestError("invalid_request")
    try:
        contracts.check_strict_document(parameters, _STRUCTURE_LIMITS)
    except contracts.ArenaContractError as exc:
        raise OperationRequestError("invalid_request") from exc
    if len(contracts.canonical_json(parameters).encode("utf-8")) > operation.max_request_bytes:
        raise OperationRequestError("request_too_large")
    normalized = _validate_object(operation.request_fields, parameters, "$", forbidden=operation.forbidden_names)
    for name, default in operation.defaults.items():
        if name not in normalized:
            normalized[name] = _deep_copy_json(default)
    encoded = contracts.canonical_json(normalized).encode("utf-8")
    if len(encoded) > operation.max_request_bytes:
        raise OperationRequestError("request_too_large")
    return normalized


# ---------------------------------------------------------------------------
# Request matching (used by the shim; the broker re-validates the frame)
# ---------------------------------------------------------------------------


def check_request_headers(headers: Mapping[str, Any]) -> None:
    """Refuse caller credentials and any header outside the allowlist."""

    if headers is None:
        return
    if not isinstance(headers, Mapping):
        raise OperationRequestError("invalid_request")
    content_type = None
    for name in headers:
        lowered = str(name).strip().lower()
        if lowered in CREDENTIAL_HEADERS:
            raise OperationRequestError("forbidden_header")
        if lowered not in ALLOWED_REQUEST_HEADERS:
            raise OperationRequestError("unknown_header")
        if lowered == "content-type":
            content_type = str(headers[name])
    if content_type is not None:
        media = content_type.split(";", 1)[0].strip().lower()
        if media and media not in ("application/json", "application/x-www-form-urlencoded", "text/plain"):
            raise OperationRequestError("invalid_body")


def _coerce_query_value(spec: FieldSpec, raw: str, path: str) -> Any:
    if spec.kind == "bool":
        lowered = raw.strip().lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
        raise OperationRequestError("invalid_field", path)
    if spec.kind == "int":
        if not _INT_QUERY_RE.match(raw.strip()):
            raise OperationRequestError("invalid_field", path)
        return int(raw.strip())
    if spec.kind == "float":
        try:
            return float(raw.strip())
        except ValueError as exc:
            raise OperationRequestError("invalid_field", path) from exc
    return raw


def _match_operation(method: str, url: str) -> Operation:
    if not isinstance(method, str) or not isinstance(url, str):
        raise OperationRequestError("no_matching_operation")
    method = method.upper()
    if method not in METHODS or not url.isascii() or any(char.isspace() for char in url):
        raise OperationRequestError("no_matching_operation")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise OperationRequestError("no_matching_operation") from exc
    if parts.scheme != "https" or parts.fragment or "#" in url or "@" in parts.netloc:
        raise OperationRequestError("no_matching_operation")
    if port is not None and port != 443:
        raise OperationRequestError("no_matching_operation")
    host = parts.hostname
    if not host or not _is_dns_hostname(host):
        raise OperationRequestError("no_matching_operation")
    for operation in _OPERATION_LIST:
        if operation.method == method and operation.host == host and operation.path == parts.path:
            return operation
    raise OperationRequestError("no_matching_operation")


def match_request(
    method: str,
    url: str,
    body: Optional[bytes],
    headers: Optional[Mapping[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Map one client request to ``(operation_id, normalized_parameters)``.

    Exactly one operation matches by method, constant host, and exact path
    (port 443 only, no userinfo, no fragment). POST operations take a JSON
    object body and no query string; GET operations take declared query
    parameters and no body. Headers are checked against the allowlist and
    never forwarded.
    """

    operation = _match_operation(method, url)
    check_request_headers(headers or {})
    parts = urlsplit(url)
    raw = b"" if body is None else bytes(body)
    if operation.parameter_location == "body":
        if parts.query:
            raise OperationRequestError("invalid_query")
        if len(raw) > operation.max_request_bytes:
            raise OperationRequestError("request_too_large")
        try:
            parameters = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise OperationRequestError("invalid_body") from exc
        if not isinstance(parameters, dict):
            raise OperationRequestError("invalid_body")
    else:
        if raw:
            raise OperationRequestError("invalid_body")
        if len(url) > operation.max_request_bytes:
            raise OperationRequestError("request_too_large")
        try:
            pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True) if parts.query else []
        except ValueError as exc:
            raise OperationRequestError("invalid_query") from exc
        parameters = {}
        for name, value in pairs:
            if name in parameters:
                raise OperationRequestError("invalid_query")
            spec = operation.request_fields.get(name)
            parameters[name] = _coerce_query_value(spec, value, "$." + name) if spec is not None else value
    return operation.operation_id, validate_operation_request(operation.operation_id, parameters)


# ---------------------------------------------------------------------------
# Outbound construction (credential-free; the broker adds the credential)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutboundTarget:
    method: str
    scheme: str
    host: str
    port: int
    path: str


@dataclass(frozen=True)
class OutboundRequest:
    target: OutboundTarget
    url: str
    query: Mapping[str, str]
    body: bytes
    content_type: Optional[str]
    credential: CredentialPlacement


def outbound_target(operation_id: str, parameters: Any) -> OutboundTarget:
    """The constant target of an operation; parameters are validated only."""

    operation = _operation(operation_id)
    validate_operation_request(operation_id, parameters)
    return OutboundTarget(operation.method, "https", operation.host, 443, operation.path)


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise OperationRequestError("invalid_field")


def build_outbound_request(operation_id: str, parameters: Any) -> OutboundRequest:
    """Build the credential-free outbound request from validated parameters.

    Fixed params are merged last so nothing model-supplied can shadow them.
    The broker places the credential per ``operation.credential`` and adds
    nothing else from the model request.
    """

    operation = _operation(operation_id)
    normalized = validate_operation_request(operation_id, parameters)
    target = OutboundTarget(operation.method, "https", operation.host, 443, operation.path)
    base_url = "https://%s%s" % (operation.host, operation.path)
    if operation.parameter_location == "body":
        document = dict(normalized)
        document.update(_deep_copy_json(operation.fixed_params))
        body = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        return OutboundRequest(target, base_url, MappingProxyType({}), body, "application/json", operation.credential)
    query: Dict[str, str] = {name: _query_value(value) for name, value in normalized.items()}
    for name, value in operation.fixed_params.items():
        query[name] = _query_value(value)
    ordered = MappingProxyType({key: query[key] for key in sorted(query)})
    return OutboundRequest(target, base_url + "?" + urlencode(ordered), ordered, b"", None, operation.credential)


# ---------------------------------------------------------------------------
# Response sanitizer
# ---------------------------------------------------------------------------


def _truncate_utf8(body: bytes, limit: int) -> bytes:
    cut = body[:limit]
    # Drop a trailing partial multi-byte sequence so the text stays decodable.
    for back in range(1, 5):
        try:
            cut.decode("utf-8")
            return cut
        except UnicodeDecodeError:
            cut = body[: max(0, limit - back)]
    return cut


def _provider_content_type(headers: Mapping[str, Any]) -> Optional[str]:
    for name, value in headers.items():
        if str(name).lower() == "content-type":
            media = str(value).split(";", 1)[0].strip().lower()
            if _CONTENT_TYPE_RE.match(media):
                return media
            return None
    return None


def sanitize_response(
    operation_id: str,
    status: Any,
    headers: Optional[Mapping[str, Any]],
    body: Any,
) -> Tuple[int, Dict[str, str], bytes]:
    """Return the model-visible ``(status, headers, body)`` for a provider reply.

    Every provider header is dropped except a validated ``content-type``;
    ``content-length`` is recomputed. Credential statuses become one generic
    502. JSON operations require a JSON object or array within the size cap
    and pass the body through unchanged; text operations truncate at the cap
    on a UTF-8 boundary.
    """

    operation = _operation(operation_id)
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        raise OperationResponseError("invalid_response")
    if headers is not None and not isinstance(headers, Mapping):
        raise OperationResponseError("invalid_response")
    if not isinstance(body, (bytes, bytearray, memoryview)):
        raise OperationResponseError("invalid_response")
    raw = bytes(body)
    generic = status in CREDENTIAL_STATUSES or (status == 402 and operation.funding_source == "tao")
    if generic:
        return (
            GENERIC_UNAVAILABLE_STATUS,
            {"content-type": "application/json", "content-length": str(len(GENERIC_UNAVAILABLE_BODY))},
            GENERIC_UNAVAILABLE_BODY,
        )
    if operation.response_sanitizer == "json":
        if len(raw) > operation.max_response_bytes:
            raise OperationResponseError("response_too_large")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise OperationResponseError("invalid_response") from exc
        if not isinstance(parsed, (dict, list)):
            raise OperationResponseError("invalid_response")
        content_type = "application/json"
    else:
        if len(raw) > operation.max_response_bytes:
            raw = _truncate_utf8(raw, operation.max_response_bytes)
        content_type = _provider_content_type(headers or {}) or "text/plain"
    return status, {"content-type": content_type, "content-length": str(len(raw))}, raw


__all__ = [
    "ALLOWED_REQUEST_HEADERS",
    "CREDENTIAL_HEADERS",
    "CREDENTIAL_STATUSES",
    "CredentialPlacement",
    "ERROR_CODES",
    "FIELD_FORMATS",
    "FIELD_KINDS",
    "FORBIDDEN_FIELD_NAMES",
    "FieldSpec",
    "OPENROUTER_MAX_OUTPUT_TOKENS",
    "OPENROUTER_STRICT_PROVIDER_POLICY",
    "OPERATIONS",
    "OPERATION_LIMITS",
    "OPERATION_TABLE_HASH",
    "OPERATION_TABLE_SCHEMA_VERSION",
    "Operation",
    "OperationError",
    "OperationRequestError",
    "OperationResponseError",
    "OutboundRequest",
    "OutboundTarget",
    "PRICE_LIST_SCHEMA_VERSION",
    "PROVIDER_PRICE_LIST",
    "PROVIDER_PRICE_LIST_HASH",
    "build_outbound_request",
    "check_request_headers",
    "field_spec_document",
    "fixed_cost_microusd",
    "match_request",
    "operation_document",
    "operation_table_document",
    "outbound_target",
    "price_list_document",
    "sanitize_response",
    "validate_https_url",
    "validate_operation_request",
]
