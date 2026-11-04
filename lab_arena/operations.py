"""Closed provider operation table for Lab Arena V1 (labarena.md section 7.4).

Pure data plus validation; nothing here performs I/O. The table is the only
way a model reaches a provider: every request is matched to exactly one
operation by method, constant host, and exact path, its parameters are
validated against a closed schema, and only the operation id plus validated
parameters cross the sandbox socket. The broker (``lab_arena/broker.py``)
builds the outbound URL, headers, and credentials itself from the constants
in this table; no request field can alter host, path, method, headers, or
credentials, and the tests prove it field by field.

Provider calls use host credentials and bounded per-provider call quotas.
``openrouter.chat`` is priced by the broker from the round's pinned price
table; this module only caps the model-controllable output tokens.

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
from dataclasses import dataclass, field
from html.parser import HTMLParser
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit

from lab_arena import contracts

OPERATION_TABLE_SCHEMA_VERSION = "leadpoet.lab_arena.operation_table.v1"
PRICE_LIST_SCHEMA_VERSION = "leadpoet.lab_arena.provider_price_list.v1"
# Added by the broker only after provider headers have been sanitized. The
# worker carries it inside the existing response-header map, and the scorer
# shim consumes it without exposing it as a provider response header.
TRUSTED_RESPONSE_URL_HEADER = "x-lab-arena-response-url"
BUDGET_ADMISSION_MAX_SECONDS = 20.0
# A paid OpenRouter or Deepline request can consume its full network timeout
# before returning an identity without final billing. Keep a separate bounded
# window for exact post-response billing reconciliation; never resend the paid
# request during this allowance.
PROVIDER_BILLING_RECONCILIATION_SECONDS = 30.0
PROVIDER_API_TIMEOUT_GRACE_SECONDS = 15.0

# A public SDK-compatible handle, not a provider credential. The worker drops
# this exact value before the gateway resolves the submission-owned key.
SCRAPINGDOG_RUNTIME_HANDLE = "lab-arena-brokered-scrapingdog"

PROVIDERS = contracts.PROVIDERS
FUNDING_SOURCES = ("host",)
METHODS = ("GET", "POST")
PARAMETER_LOCATIONS = ("body", "query")
RESPONSE_SANITIZERS = ("json", "text")
BODY_WRAPPERS = ("", "deepline_execute", "deepline_exa_compat")
RESPONSE_TRANSFORMS = ("", "deepline_unwrap")
RESPONSE_SCRUBS = ("", "deepline_person_entities")
FIELD_KINDS = ("str", "int", "float", "bool", "list[str]", "list[object]", "object", "any")
FIELD_FORMATS = ("https_url", "iso_date", "domain", "model_id")

OPENROUTER_MAX_OUTPUT_TOKENS = 4096
OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS = 32_768
OPENROUTER_RESPONSES_TIMEOUT_SECONDS = 600
OPENROUTER_MAX_MESSAGES = 128
# A native Responses history can add reasoning, call, and call-output items
# for each brokered call. Keep enough top-level history for the 200-call
# execution profile while the existing byte and nested-list limits still
# bound the request.
OPENROUTER_RESPONSES_MAX_INPUT_ITEMS = 768
OPENROUTER_MAX_CONTENT_CHARS = 32_000
OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS = 1
OPENROUTER_WEB_SEARCH_MAX_TOTAL_RESULTS = 5
# The broker injects this policy into every chat body; it is table data so its hash is
# bound into the round configuration.
# Zero data retention is the judge's own request for the page content it
# sends to a model; the Arena pins it for every OpenRouter call.
OPENROUTER_STRICT_PROVIDER_POLICY: Mapping[str, Any] = MappingProxyType(
    {"data_collection": "deny", "allow_fallbacks": False, "zdr": True}
)
OPENROUTER_OUTBOUND_HEADERS: Mapping[str, str] = MappingProxyType(
    {"X-OpenRouter-Metadata": "enabled"}
)

# Structural limits for operation parameters (see module docstring). They are
# deliberately looser than every field schema so the schema produces the
# precise error code; ``max_total_bytes`` is the ceiling on any operation's
# ``max_request_bytes``, which is enforced per operation.
# Depth allows a caller's JSON schema (the judge's verifier nests ten levels);
# bytes stay bounded per operation.
OPERATION_LIMITS = contracts.StrictLimits(
    max_depth=12,
    max_list_items=128,
    max_object_keys=64,
    max_string_bytes=4 * OPENROUTER_MAX_CONTENT_CHARS,
    max_total_bytes=1_000_000,
)
# The structural pass never applies the total-size check itself: size is the
# per-operation cap below and reports ``request_too_large``.
_STRUCTURE_LIMITS = dataclasses.replace(OPERATION_LIMITS, max_total_bytes=2 ** 31)
RESPONSES_OPERATION_LIMITS = dataclasses.replace(
    OPERATION_LIMITS,
    max_depth=24,
    max_list_items=OPENROUTER_RESPONSES_MAX_INPUT_ITEMS,
)
_RESPONSES_STRUCTURE_LIMITS = dataclasses.replace(RESPONSES_OPERATION_LIMITS, max_total_bytes=2 ** 31)


def _request_structure_limits(operation_id: str, allow_full_thinking: bool) -> contracts.StrictLimits:
    limits = _RESPONSES_STRUCTURE_LIMITS if operation_id == "openrouter.responses" else _STRUCTURE_LIMITS
    if allow_full_thinking and operation_id in {"openrouter.chat", "openrouter.responses"}:
        # Token limits do not bound generated character counts. Keep every
        # replay byte under the existing complete request cap instead.
        return dataclasses.replace(limits, max_string_bytes=contracts.FULL_THINKING_MAX_STRING_BYTES)
    return limits

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
        # OpenRouter's app-attribution headers; the judge sends them, the
        # broker never forwards a caller header.
        "http-referer",
        "keep-alive",
        "pragma",
        "te",
        "user-agent",
        "x-title",
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
        # OpenRouter surfaces whose cost or routing the broker owns.
        "functions",
        "function_call",
        "plugins",
        "provider",
        "models",
        "route",
        "stream",
        "stream_options",
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
HOST_ACCOUNT_STATUSES = frozenset({401, 402, 403, 407})


def provider_status_is_infrastructure(status: Any) -> bool:
    """Shared account, rate-limit, and provider-server failures are host faults."""

    return isinstance(status, int) and not isinstance(status, bool) and (
        status in HOST_ACCOUNT_STATUSES or status == 429 or 500 <= status <= 599
    )
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
        if self.kind == "list[object]" and self.fields is None:
            raise ValueError("list[object] members need a schema")
        # An ``object`` without ``fields`` is an opaque provider-owned document
        # bounded by the structural limits and the forbidden-name scan only.
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
    # Request fields substituted into the path (one ``{name}`` placeholder
    # each); they must be closed-choice strings so the path stays enumerable.
    path_fields: Tuple[str, ...] = ()
    # Constant headers the broker adds to the outbound request (never from the model).
    outbound_headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    # Named body transform applied after validation; "" sends the fields as-is.
    body_wrapper: str = ""
    # Named response scrub applied after sanitization; "" passes the body through.
    response_scrub: str = ""
    # Compatibility routes match one host and path but are sent elsewhere.
    outbound_host: str = ""
    outbound_path: str = ""
    # For Deepline-backed routes: the fixed tool the request becomes.
    deepline_tool: str = ""
    # Named reply transform applied before the scrub; "" keeps the provider body.
    response_transform: str = ""
    # Body fields a caller may send only as a subset of the table's own value
    # for the same fixed param; they are dropped and the fixed param stands.
    # The judge asks OpenRouter for the privacy policy the table already pins.
    superseded_fields: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS or self.method not in METHODS:
            raise ValueError("operation provider or method is invalid")
        for name in self.path_fields:
            spec = self.request_fields.get(name)
            if spec is None or spec.kind != "str" or not spec.choices or not spec.required:
                raise ValueError("path fields must be required closed-choice strings")
            if self.path.count("{%s}" % name) != 1:
                raise ValueError("path must carry each path field exactly once")
        residual = re.sub(r"\{[a-z_]+\}", "", self.path)
        if "{" in residual or "}" in residual:
            raise ValueError("path placeholders must all be declared path fields")
        if self.body_wrapper not in BODY_WRAPPERS:
            raise ValueError("operation body wrapper is invalid")
        if self.response_scrub not in RESPONSE_SCRUBS:
            raise ValueError("operation response scrub is invalid")
        if self.response_transform not in RESPONSE_TRANSFORMS:
            raise ValueError("operation response transform is invalid")
        if self.outbound_host and not _is_dns_hostname(self.outbound_host):
            raise ValueError("operation outbound host is invalid")
        if self.outbound_path and (not self.outbound_path.startswith("/") or "?" in self.outbound_path or "#" in self.outbound_path or "{" in self.outbound_path):
            raise ValueError("operation outbound path is invalid")
        if self.deepline_tool and self.deepline_tool not in DEEPLINE_TOOLS:
            raise ValueError("operation deepline tool is invalid")
        if self.body_wrapper == "deepline_exa_compat" and (not self.deepline_tool or not self.outbound_path):
            raise ValueError("Exa compatibility routes need a Deepline tool and outbound path")
        for name, value in self.outbound_headers.items():
            if _normalize_name(name) in {_normalize_name(h) for h in CREDENTIAL_HEADERS} or not isinstance(value, str):
                raise ValueError("outbound headers may not carry credentials")
        object.__setattr__(self, "outbound_headers", MappingProxyType(dict(self.outbound_headers)))
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
        for name, allowed in self.superseded_fields.items():
            fixed = self.fixed_params.get(name)
            if name in self.request_fields or not isinstance(fixed, Mapping) or not isinstance(allowed, Mapping):
                raise ValueError("superseded fields must name an object-valued fixed param")
            if any(key not in fixed or fixed[key] != value for key, value in allowed.items()):
                raise ValueError("a superseded field may only accept values the fixed param already carries")
        object.__setattr__(self, "superseded_fields", MappingProxyType({name: _deep_copy_json(allowed) for name, allowed in self.superseded_fields.items()}))
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


# Names refused at any depth inside an opaque provider payload. Top-level
# request fields use the wider FORBIDDEN_FIELD_NAMES because they could steer
# the request; inside a tool payload the target is fixed by the table, so only
# credential-shaped names are refused (a tool may legitimately take ``urls``).
PAYLOAD_FORBIDDEN_FIELD_NAMES = frozenset({_normalize_name(name) for name in CREDENTIAL_HEADERS} | {"api_key", "apikey", "token", "access_token", "secret", "password", "bearer"})


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

# Non-OpenRouter operations use the host account; the Arena
# enforces fairness with the per-provider call quotas pinned in
# ``contracts.CALL_QUOTAS_PER_ICP``.
CALL_QUOTA_COST_RULE: Mapping[str, Any] = MappingProxyType({"kind": "call_quota"})


# ---------------------------------------------------------------------------
# V1 operations
# ---------------------------------------------------------------------------

_DOMAIN_ITEM = FieldSpec("str", min_length=1, max_length=253, format="domain")
_HTTPS_URL_ITEM = FieldSpec("str", min_length=8, max_length=2000, format="https_url")
_GOOGLE_COUNTRIES = ("us", "gb", "ca", "au", "de", "fr", "nl", "ie", "in", "sg")

_SCRAPINGDOG_CREDENTIAL = CredentialPlacement("query", "api_key")
_OPENROUTER_CREDENTIAL = CredentialPlacement("header", "authorization", scheme="Bearer")
_DEEPLINE_CREDENTIAL = CredentialPlacement("header", "authorization", scheme="Bearer")
# Deepline tools that the sourcing bundles and judge may execute. Deepline
# owns each tool's payload schema.
DEEPLINE_TOOLS = (
    "contextdev_get_web_scrape_markdown",
    "contextdev_post_news_search",
    "contextdev_post_web_search",
    "exa_answer",
    "exa_company_search",
    "exa_contents",
    "exa_people_search",
    "exa_search",
    "firecrawl_scrape",
    "free_simple_company_search",
    "generic_http_request",
    "harvestapi_get_company",
    "harvestapi_get_profile",
    "harvestapi_search_leads",
    "hunter_email_finder",
    "limadata_find_work_email",
    "datagma_find_email",
    "leadmagic_email_finder",
    "zerobounce_validate",
    "bounceban_verify_single",
    "bounceban_get_single_status",
    "harvestapi_get_job",
    "harvestapi_get_post",
    "hunter_discover",
    "predictleads_company_financing_events",
    "predictleads_company_job_openings",
    "predictleads_company_news_events",
    "twitterapi_tweets_by_ids",
)
# Deepline's execute body names the upstream provider of each tool; pinned
# from the official client (deepline_core 0.3.20) and the tool contracts.
DEEPLINE_TOOL_PROVIDERS: Mapping[str, str] = MappingProxyType({
    "contextdev_get_web_scrape_markdown": "contextdev",
    "contextdev_post_news_search": "contextdev",
    "contextdev_post_web_search": "contextdev",
    "exa_answer": "exa",
    "exa_company_search": "exa",
    "exa_contents": "exa",
    "exa_people_search": "exa",
    "exa_search": "exa",
    "firecrawl_scrape": "firecrawl",
    "free_simple_company_search": "deepline_native",
    "generic_http_request": "generic_http",
    "harvestapi_get_company": "harvestapi",
    "harvestapi_get_profile": "harvestapi",
    "harvestapi_search_leads": "harvestapi",
    "hunter_email_finder": "hunter",
    "limadata_find_work_email": "limadata",
    "datagma_find_email": "datagma",
    "leadmagic_email_finder": "leadmagic",
    "zerobounce_validate": "zerobounce",
    "bounceban_verify_single": "bounceban",
    "bounceban_get_single_status": "bounceban",
    "harvestapi_get_job": "harvestapi",
    "harvestapi_get_post": "harvestapi",
    "hunter_discover": "hunter",
    "predictleads_company_financing_events": "predictleads",
    "predictleads_company_job_openings": "predictleads",
    "predictleads_company_news_events": "predictleads",
    "twitterapi_tweets_by_ids": "twitterapi",
})
CONTACT_FINDER_INPUT_SCHEMA_VERSION = "leadpoet.lab_arena.contact_finder_input.v1"
DEEPLINE_CONTACT_FINDER_IDENTITY_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "hunter_email_finder": (
        "company", "domain", "first_name", "full_name", "last_name",
        "linkedin_handle",
    ),
    "limadata_find_work_email": ("company_domain", "full_name"),
    "datagma_find_email": (
        "companyDomain", "companyName", "company_domain", "company_name",
        "domain", "firstName", "first_name", "fullName", "full_name",
        "lastName", "last_name", "linkedInSlug",
    ),
    "leadmagic_email_finder": (
        "company_domain", "company_name", "domain", "first_name", "last_name",
    ),
})
# Asks Deepline for the provider's raw response under result.data instead of a
# normalized view; captured from the official client and verified live.
DEEPLINE_EXECUTE_HEADERS: Mapping[str, str] = MappingProxyType({"x-deepline-execute-response-intent": "raw"})
# The generic HTTP tool's native receipt carries target URL, status, headers,
# and body only under rawV2. Intent alone returns a bare result.data string
# with no target provenance. Keep other tools on their existing contracts,
# including Exa clients that unwrap result.data. The same builder is used for
# initial dispatch and a proved same-key recovery. Callers must select this
# contract explicitly; omission retains the legacy result.data envelope.
DEEPLINE_GENERIC_HTTP_HEADERS: Mapping[str, str] = MappingProxyType({
    "x-deepline-execute-response-contract": "raw-v2",
    "x-deepline-tool-error-schema": "1",
})
# Target-page headers, selected only by the broker's HTML source fallback.
# Caller payload headers remain forbidden by the frozen company-only policy.
SOURCE_HTML_REQUEST_HEADERS: Mapping[str, str] = MappingProxyType({
    "User-Agent": "Leadpoet-Arena/1.0",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
})
# Exa attaches extracted ``entities`` (people with employers and education) to
# search and contents results. Nothing in the judge reads them, so the broker
# drops person entities before a model or the ledger sees them; the people
# search tool exists to return people and is exempt.
DEEPLINE_PERSON_ENTITY_TOOLS = frozenset({"exa_people_search"})

_OPERATION_LIST = (
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
            # The judge's escalation tiers (dynamic render, premium stealth);
            # every tier is bounded by the same host-enforced quota.
            "wait": FieldSpec("int", minimum=0, maximum=15000),
            "premium": FieldSpec("bool"),
            "stealth_mode": FieldSpec("bool"),
        },
        fixed_params={},
        defaults={"dynamic": False},
        timeout_seconds=60,
        max_request_bytes=4_096,
        max_response_bytes=2_097_152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
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
        timeout_seconds=65,
        max_request_bytes=4_096,
        max_response_bytes=1_048_576,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="json",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.google_news",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/google_news",
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
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="json",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.google_jobs",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/google_jobs",
        parameter_location="query",
        request_fields={
            "query": FieldSpec("str", required=True, min_length=1, max_length=500),
            "country": FieldSpec("str", choices=_GOOGLE_COUNTRIES),
        },
        fixed_params={},
        defaults={"country": "us"},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=1_048_576,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="json",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.x_post",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/x/post",
        parameter_location="query",
        request_fields={
            "tweetId": FieldSpec("str", required=True, min_length=1, max_length=64),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.x_profile",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/x/profile",
        parameter_location="query",
        request_fields={
            "profileId": FieldSpec("str", required=True, min_length=1, max_length=64),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.profile",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/profile",
        parameter_location="query",
        request_fields={
            "type": FieldSpec("str", required=True, choices=("company", "profile")),
            "id": FieldSpec("str", required=True, min_length=1, max_length=200),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.profile_post",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/profile/post",
        parameter_location="query",
        request_fields={
            "id": FieldSpec("str", required=True, min_length=1, max_length=200),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.linkedinjobs",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/linkedinjobs",
        parameter_location="query",
        request_fields={
            "job_id": FieldSpec("str", required=True, min_length=1, max_length=32),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.jobs",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/jobs",
        parameter_location="query",
        request_fields={
            "job_id": FieldSpec("str", required=True, min_length=1, max_length=64),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.indeed",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/indeed",
        parameter_location="query",
        request_fields={
            "url": FieldSpec("str", required=True, min_length=8, max_length=2000, format="https_url"),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.instagram_profile",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/instagram/profile",
        parameter_location="query",
        request_fields={
            "username": FieldSpec("str", required=True, min_length=1, max_length=64),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.tiktok_profile",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/tiktok/profile",
        parameter_location="query",
        request_fields={
            "username": FieldSpec("str", required=True, min_length=1, max_length=64),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.youtube_video",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/youtube/video",
        parameter_location="query",
        request_fields={
            "v": FieldSpec("str", required=True, min_length=1, max_length=32),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.youtube_transcripts",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/youtube/transcripts",
        parameter_location="query",
        request_fields={
            "v": FieldSpec("str", required=True, min_length=1, max_length=32),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.youtube_channel",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/youtube/channel",
        parameter_location="query",
        request_fields={
            "channel_id": FieldSpec("str", required=True, min_length=1, max_length=64),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    Operation(
        operation_id="scrapingdog.youtube_search",
        provider="scrapingdog",
        method="GET",
        host="api.scrapingdog.com",
        path="/youtube/search",
        parameter_location="query",
        request_fields={
            "search_query": FieldSpec("str", required=True, min_length=1, max_length=500),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=45,
        max_request_bytes=4_096,
        max_response_bytes=2097152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="text",
        funding_source="host",
        credential=_SCRAPINGDOG_CREDENTIAL,
    ),
    # Exa compatibility: matched on api.exa.ai so Exa-shaped clients (the Lab
    # judge, Exa SDKs) work unchanged, sent to Deepline's exa_search on the miner's
    # Deepline key, and answered with the raw Exa reply unwrapped from the envelope.
    Operation(
        operation_id="exa.search",
        provider="deepline",
        method="POST",
        host="api.exa.ai",
        path="/search",
        parameter_location="body",
        request_fields={
            "query": FieldSpec("str", required=True, min_length=1, max_length=2000),
            "additionalQueries": FieldSpec("list[str]", max_length=10, item=FieldSpec("str", min_length=1, max_length=2000)),
            "type": FieldSpec("str", choices=("auto", "fast", "deep", "neural", "instant", "keyword")),
            "category": FieldSpec("str", choices=("company", "people", "news", "research paper", "tweet", "personal site", "financial report", "pdf", "github", "linkedin profile")),
            "numResults": FieldSpec("int", minimum=1, maximum=100),
            "includeDomains": FieldSpec("list[str]", max_length=100, item=FieldSpec("str", min_length=1, max_length=253)),
            "excludeDomains": FieldSpec("list[str]", max_length=100, item=FieldSpec("str", min_length=1, max_length=253)),
            "startCrawlDate": FieldSpec("str", max_length=40),
            "endCrawlDate": FieldSpec("str", max_length=40),
            "startPublishedDate": FieldSpec("str", max_length=40),
            "endPublishedDate": FieldSpec("str", max_length=40),
            "includeText": FieldSpec("list[str]", max_length=5, item=FieldSpec("str", min_length=1, max_length=200)),
            "excludeText": FieldSpec("list[str]", max_length=5, item=FieldSpec("str", min_length=1, max_length=200)),
            "userLocation": FieldSpec("str", min_length=2, max_length=2),
            "contents": FieldSpec("any"),
            "context": FieldSpec("any"),
            "moderation": FieldSpec("bool"),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=60,
        max_request_bytes=65_536,
        max_response_bytes=2_097_152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="json",
        funding_source="host",
        credential=_DEEPLINE_CREDENTIAL,
        outbound_headers=DEEPLINE_EXECUTE_HEADERS,
        body_wrapper="deepline_exa_compat",
        response_scrub="deepline_person_entities",
        outbound_host="code.deepline.com",
        outbound_path="/api/v2/integrations/exa_search/execute",
        deepline_tool="exa_search",
        response_transform="deepline_unwrap",
    ),
    # Exa compatibility: matched on api.exa.ai so Exa-shaped clients (the Lab
    # judge, Exa SDKs) work unchanged, sent to Deepline's exa_contents on the miner's
    # Deepline key, and answered with the raw Exa reply unwrapped from the envelope.
    Operation(
        operation_id="exa.contents",
        provider="deepline",
        method="POST",
        host="api.exa.ai",
        path="/contents",
        parameter_location="body",
        request_fields={
            "ids": FieldSpec("list[str]", max_length=100, item=FieldSpec("str", min_length=1, max_length=2000)),
            "urls": FieldSpec("list[str]", max_length=100, item=FieldSpec("str", min_length=1, max_length=2000)),
            "text": FieldSpec("any"),
            "highlights": FieldSpec("any"),
            "summary": FieldSpec("any"),
            "livecrawl": FieldSpec("str", choices=("never", "fallback", "preferred", "always")),
            "livecrawlTimeout": FieldSpec("int", minimum=0, maximum=120000),
            "maxAgeHours": FieldSpec("int", minimum=0, maximum=100000),
            "subpages": FieldSpec("int", minimum=0, maximum=10),
            "subpageTarget": FieldSpec("any"),
            "extras": FieldSpec("any"),
            "context": FieldSpec("any"),
        },
        fixed_params={},
        defaults={},
        timeout_seconds=60,
        max_request_bytes=65_536,
        max_response_bytes=2_097_152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="json",
        funding_source="host",
        credential=_DEEPLINE_CREDENTIAL,
        outbound_headers=DEEPLINE_EXECUTE_HEADERS,
        body_wrapper="deepline_exa_compat",
        response_scrub="deepline_person_entities",
        outbound_host="code.deepline.com",
        outbound_path="/api/v2/integrations/exa_contents/execute",
        deepline_tool="exa_contents",
        response_transform="deepline_unwrap",
    ),
    Operation(
        operation_id="deepline.execute",
        provider="deepline",
        method="POST",
        host="code.deepline.com",
        path="/api/v2/integrations/{tool}/execute",
        parameter_location="body",
        request_fields={
            "tool": FieldSpec("str", required=True, choices=DEEPLINE_TOOLS),
            # Opaque tool payload: Deepline owns each tool's schema, the
            # structural limits bound it, credential-shaped names are refused.
            "payload": FieldSpec("object", required=True),
            # Closed opt-in for generic_http_request; not a provider payload field.
            "response_contract": FieldSpec("str", choices=("raw-v2",)),
        },
        fixed_params={},
        defaults={},
        # Paid integrations can need several minutes before returning a job
        # identity. Preserve the caller's shorter timeout; allow the native
        # Deepline 240-second window without changing the ICP hard deadline.
        timeout_seconds=240,
        max_request_bytes=65_536,
        max_response_bytes=2_097_152,
        cost_rule=CALL_QUOTA_COST_RULE,
        response_sanitizer="json",
        funding_source="host",
        credential=_DEEPLINE_CREDENTIAL,
        path_fields=("tool",),
        outbound_headers=DEEPLINE_EXECUTE_HEADERS,
        body_wrapper="deepline_execute",
        response_scrub="deepline_person_entities",
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
                    "role": FieldSpec("str", required=True, choices=("system", "developer", "user", "assistant", "tool")),
                    "content": FieldSpec("any"),
                    "name": FieldSpec("str", min_length=1, max_length=128),
                    "tool_call_id": FieldSpec("str", min_length=1, max_length=256),
                    # Echo provider-returned reasoning on assistant turns so
                    # signatures and tool continuity survive the next call.
                    "reasoning": FieldSpec("any"),
                    "reasoning_content": FieldSpec("any"),
                    "refusal": FieldSpec("any"),
                    "reasoning_details": FieldSpec("list[object]", max_length=64, fields={
                        "type": FieldSpec("str", required=True, choices=("reasoning.text", "reasoning.summary", "reasoning.encrypted")),
                        "text": FieldSpec("str", max_length=OPENROUTER_MAX_CONTENT_CHARS),
                        "summary": FieldSpec("str", max_length=OPENROUTER_MAX_CONTENT_CHARS),
                        "data": FieldSpec("str", max_length=OPENROUTER_MAX_CONTENT_CHARS),
                        "signature": FieldSpec("any"),
                        "id": FieldSpec("any"),
                        "format": FieldSpec("str", max_length=128),
                        "index": FieldSpec("int", minimum=0),
                    }),
                    "tool_calls": FieldSpec(
                        "list[object]",
                        max_length=64,
                        fields={
                            "id": FieldSpec("str", required=True, min_length=1, max_length=256),
                            "type": FieldSpec("str", required=True, choices=("function",)),
                            "index": FieldSpec("int", minimum=0),
                            "function": FieldSpec(
                                "object",
                                required=True,
                                fields={
                                    "name": FieldSpec("str", required=True, min_length=1, max_length=128),
                                    "arguments": FieldSpec("str", required=True, max_length=OPENROUTER_MAX_CONTENT_CHARS),
                                },
                            ),
                        },
                    ),
                },
            ),
            "tools": FieldSpec(
                "list[object]",
                max_length=64,
                fields={
                    "type": FieldSpec("str", required=True, choices=("function",)),
                    "function": FieldSpec("object", required=True),
                },
            ),
            "tool_choice": FieldSpec("any"),
            "parallel_tool_calls": FieldSpec("bool"),
            "reasoning": FieldSpec("object"),
            "reasoning_effort": FieldSpec("str", min_length=1, max_length=32),
            "temperature": FieldSpec("float", minimum=0, maximum=2),
            "max_tokens": FieldSpec("int", minimum=1, maximum=OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS),
            "top_p": FieldSpec("float", minimum=0, maximum=1),
            "stop": FieldSpec("list[str]", max_length=4, item=FieldSpec("str", min_length=1, max_length=64)),
            "seed": FieldSpec("int", minimum=-(2 ** 31), maximum=2 ** 31 - 1),
            # The judge asks for JSON replies; a JSON schema is the caller's own
            # bounded object, never interpreted by the Arena.
            "response_format": FieldSpec(
                "object",
                fields={
                    "type": FieldSpec("str", required=True, choices=("text", "json_object", "json_schema")),
                    "json_schema": FieldSpec("object"),
                },
            ),
            # The judge's three-stage verifier asks for reasoning tokens; they
            # are completion tokens under the same output cap.
            "include_reasoning": FieldSpec("bool"),
        },
        fixed_params={"stream": False, "provider": dict(OPENROUTER_STRICT_PROVIDER_POLICY)},
        superseded_fields={"provider": {"data_collection": "deny", "zdr": True, "allow_fallbacks": False}},
        # Absent output caps default to the server cap so cost stays bounded.
        defaults={"max_tokens": OPENROUTER_MAX_OUTPUT_TOKENS},
        timeout_seconds=120,
        max_request_bytes=1_000_000,
        max_response_bytes=1_048_576,
        cost_rule={"kind": "openrouter_price_table", "max_output_tokens": OPENROUTER_MAX_OUTPUT_TOKENS},
        response_sanitizer="json",
        funding_source="host",
        credential=_OPENROUTER_CREDENTIAL,
        outbound_headers=OPENROUTER_OUTBOUND_HEADERS,
    ),
    Operation(
        operation_id="openrouter.responses",
        provider="openrouter",
        method="POST",
        host="openrouter.ai",
        path="/api/v1/responses",
        parameter_location="body",
        request_fields={
            "model": FieldSpec("str", required=True, min_length=3, max_length=128, format="model_id"),
            # Union members are checked below; one source-controlled hosted
            # search tool is allowed alongside local tools and text.
            "input": FieldSpec("any", required=True),
            "instructions": FieldSpec("str", max_length=OPENROUTER_MAX_CONTENT_CHARS),
            "tools": FieldSpec("any"),
            "tool_choice": FieldSpec("str", choices=("auto", "none", "required")),
            "parallel_tool_calls": FieldSpec("bool"),
            "max_tool_calls": FieldSpec(
                "int", minimum=1,
                maximum=OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS,
            ),
            "reasoning": FieldSpec("object", fields={
                "effort": FieldSpec("str", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max")),
                "summary": FieldSpec("str", choices=("auto", "concise", "detailed")),
                "context": FieldSpec("str", choices=("auto", "all_turns", "current_turn")),
            }),
            "text": FieldSpec("object", fields={
                "verbosity": FieldSpec("str", choices=("low", "medium", "high")),
                "format": FieldSpec("object", fields={
                    "type": FieldSpec("str", required=True, choices=("text", "json_schema")),
                    "name": FieldSpec("str", max_length=128),
                    "schema": FieldSpec("object"),
                    "strict": FieldSpec("bool"),
                }),
            }),
            "include": FieldSpec("list[str]", max_length=1, item=FieldSpec("str", choices=("reasoning.encrypted_content",))),
            "max_output_tokens": FieldSpec("int", minimum=1, maximum=OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS),
            "prompt_cache_key": FieldSpec("str", max_length=256),
        },
        fixed_params={"stream": False, "store": False, "provider": dict(OPENROUTER_STRICT_PROVIDER_POLICY)},
        defaults={"max_output_tokens": OPENROUTER_MAX_OUTPUT_TOKENS},
        timeout_seconds=OPENROUTER_RESPONSES_TIMEOUT_SECONDS,
        max_request_bytes=1_000_000,
        max_response_bytes=1_048_576,
        cost_rule={"kind": "openrouter_price_table", "max_output_tokens": OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS},
        response_sanitizer="json",
        funding_source="host",
        credential=_OPENROUTER_CREDENTIAL,
        outbound_headers=OPENROUTER_OUTBOUND_HEADERS,
    ),
)

OPERATIONS: Mapping[str, Operation] = MappingProxyType(
    {operation.operation_id: operation for operation in _OPERATION_LIST}
)
if len(OPERATIONS) != len(_OPERATION_LIST):
    raise RuntimeError("operation ids must be unique")
for _operation in _OPERATION_LIST:
    if _operation.cost_rule["kind"] not in ("call_quota", "openrouter_price_table"):
        raise RuntimeError("operation cost rule is invalid")
    if _operation.provider == "openrouter" and _operation.cost_rule["kind"] != "openrouter_price_table":
        raise RuntimeError("OpenRouter operations must bound cost with the price table")
    if _operation.max_request_bytes > OPERATION_LIMITS.max_total_bytes:
        raise RuntimeError("operation request cap exceeds the structural ceiling")
del _operation


# Frozen metadata is served locally by the broker. Keep these routes outside
# the legacy operation table so existing round hashes remain unchanged.
_CATALOG_OPERATION_LIST = (
    Operation(
        operation_id="deepline.tools.list",
        provider="deepline", method="GET", host="code.deepline.com",
        path="/api/v2/tools", parameter_location="query",
        request_fields={"compact": FieldSpec("bool"), "categories": FieldSpec("str", max_length=512)},
        fixed_params={}, defaults={"compact": True}, timeout_seconds=5,
        max_request_bytes=4096, max_response_bytes=8_388_608,
        cost_rule=CALL_QUOTA_COST_RULE, response_sanitizer="json",
        funding_source="host", credential=_DEEPLINE_CREDENTIAL,
    ),
    Operation(
        operation_id="deepline.tools.search",
        provider="deepline", method="GET", host="code.deepline.com",
        path="/api/v2/tools/search", parameter_location="query",
        request_fields={"query": FieldSpec("str", max_length=512),
                        "compact": FieldSpec("bool"), "categories": FieldSpec("str", max_length=512),
                        "search_terms": FieldSpec("str", max_length=512),
                        "task": FieldSpec("str", max_length=1024),
                        "limit": FieldSpec("int", minimum=1, maximum=50),
                        "offset": FieldSpec("int", minimum=0),
                        "search_mode": FieldSpec("str", choices=("v1", "v2")),
                        "include_search_debug": FieldSpec("bool")},
        fixed_params={}, defaults={"compact": True}, timeout_seconds=5,
        max_request_bytes=4096, max_response_bytes=8_388_608,
        cost_rule=CALL_QUOTA_COST_RULE, response_sanitizer="json",
        funding_source="host", credential=_DEEPLINE_CREDENTIAL,
    ),
    Operation(
        operation_id="deepline.tools.get",
        provider="deepline", method="GET", host="code.deepline.com",
        path="/api/v2/integrations/{tool}/get", parameter_location="query",
        request_fields={"tool": FieldSpec("str", required=True, choices=DEEPLINE_TOOLS)},
        fixed_params={}, defaults={}, timeout_seconds=5,
        max_request_bytes=4096, max_response_bytes=262_144,
        cost_rule=CALL_QUOTA_COST_RULE, response_sanitizer="json",
        funding_source="host", credential=_DEEPLINE_CREDENTIAL,
        path_fields=("tool",),
    ),
)
CATALOG_OPERATIONS: Mapping[str, Operation] = MappingProxyType(
    {operation.operation_id: operation for operation in _CATALOG_OPERATION_LIST}
)


def _operation(operation_id: Any) -> Operation:
    operation = (OPERATIONS.get(operation_id) or CATALOG_OPERATIONS.get(operation_id)) if isinstance(operation_id, str) else None
    if operation is None:
        raise OperationRequestError("no_matching_operation")
    return operation


def get_operation(operation_id: Any) -> Optional[Operation]:
    """Resolve fixed provider operations and locally served catalog routes."""
    return (OPERATIONS.get(operation_id) or CATALOG_OPERATIONS.get(operation_id)) if isinstance(operation_id, str) else None


# ---------------------------------------------------------------------------
# Public operation description
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
        "path_fields": list(operation.path_fields),
        "outbound_headers": dict(operation.outbound_headers),
        "body_wrapper": operation.body_wrapper,
        "response_scrub": operation.response_scrub,
        "response_scrub_exempt_tools": sorted(DEEPLINE_PERSON_ENTITY_TOOLS) if operation.response_scrub else [],
        "outbound_host": operation.outbound_host,
        "outbound_path": operation.outbound_path,
        "deepline_tool": operation.deepline_tool,
        "response_transform": operation.response_transform,
        "superseded_fields": {name: dict(allowed) for name, allowed in operation.superseded_fields.items()},
        "credential": {
            "location": operation.credential.location,
            "name": operation.credential.name,
            "scheme": operation.credential.scheme,
        },
    }


def operation_table_document() -> Dict[str, Any]:
    """The full table in operation-id order."""

    return {
        "schema_version": OPERATION_TABLE_SCHEMA_VERSION,
        "operation_limits": {
            "max_depth": OPERATION_LIMITS.max_depth,
            "max_list_items": OPERATION_LIMITS.max_list_items,
            "max_object_keys": OPERATION_LIMITS.max_object_keys,
            "max_string_bytes": OPERATION_LIMITS.max_string_bytes,
            "max_total_bytes": OPERATION_LIMITS.max_total_bytes,
        },
        "operation_limit_overrides": {
            "openrouter.responses": {
                "max_depth": RESPONSES_OPERATION_LIMITS.max_depth,
                "max_list_items": RESPONSES_OPERATION_LIMITS.max_list_items,
            },
        },
        "allowed_request_headers": sorted(ALLOWED_REQUEST_HEADERS),
        "credential_headers": sorted(CREDENTIAL_HEADERS),
        "forbidden_field_names": sorted(FORBIDDEN_FIELD_NAMES),
        "payload_forbidden_field_names": sorted(PAYLOAD_FORBIDDEN_FIELD_NAMES),
        "host_account_statuses": sorted(HOST_ACCOUNT_STATUSES),
        "operations": [operation_document(OPERATIONS[key]) for key in sorted(OPERATIONS)],
        "deepline_tool_response_contracts": {
            "generic_http_request": {"raw-v2": dict(DEEPLINE_GENERIC_HTTP_HEADERS)},
        },
    }


# Every host the table matches or sends to. The trusted-scorer shim refuses to
# turn a request for one of these into a page fetch: a provider is reached only
# through its own closed operation.
PROVIDER_HOSTS = frozenset(
    host.lower()
    for operation in OPERATIONS.values()
    for host in (operation.host, operation.outbound_host)
    if host
)
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


def _validate_field(spec: FieldSpec, value: Any, path: str, *, limits: contracts.StrictLimits = _STRUCTURE_LIMITS) -> Any:
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
            return [_validate_field(spec.item, item, "%s[%d]" % (path, index), limits=limits) for index, item in enumerate(value)]
        assert spec.fields is not None
        return [_validate_object(spec.fields, item, "%s[%d]" % (path, index), limits=limits) for index, item in enumerate(value)]
    if kind == "object":
        if spec.fields is None:
            return _validate_opaque_object(value, path, limits=limits)
        return _validate_object(spec.fields, value, path, limits=limits)
    if kind == "any":
        return _validate_opaque_object({"value": value}, path, limits=limits)["value"]
    raise OperationRequestError("invalid_field", path)


def _validate_opaque_object(value: Any, path: str, *, limits: contracts.StrictLimits = _STRUCTURE_LIMITS) -> Dict[str, Any]:
    """A provider-owned document: bounded, JSON-plain, and free of credential-shaped names at any depth."""

    if not isinstance(value, Mapping):
        raise OperationRequestError("invalid_field", path)
    try:
        contracts.check_strict_document(value, limits)
    except contracts.ArenaContractError:
        raise OperationRequestError("invalid_field", path) from None

    def walk(node: Any, node_path: str) -> Any:
        if isinstance(node, Mapping):
            out: Dict[str, Any] = {}
            for name, item in node.items():
                if not isinstance(name, str):
                    raise OperationRequestError("invalid_field", node_path)
                if _normalize_name(name) in PAYLOAD_FORBIDDEN_FIELD_NAMES:
                    raise OperationRequestError("forbidden_field", node_path)
                out[name] = walk(item, "%s.%s" % (node_path, name))
            return out
        if isinstance(node, (list, tuple)):
            return [walk(item, "%s[%d]" % (node_path, index)) for index, item in enumerate(node)]
        if isinstance(node, bool) or node is None or isinstance(node, (int, str)):
            return node
        if isinstance(node, float):
            if not math.isfinite(node):
                raise OperationRequestError("invalid_field", node_path)
            return node
        raise OperationRequestError("invalid_field", node_path)

    return walk(value, path)


def _validate_object(
    fields: Mapping[str, FieldSpec],
    value: Any,
    path: str,
    *,
    forbidden: frozenset = frozenset(),
    limits: contracts.StrictLimits = _STRUCTURE_LIMITS,
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
        out[name] = _validate_field(spec, value[name], "%s.%s" % (path, name), limits=limits)
    return out


def _drop_superseded_fields(operation: Operation, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """Accept a superseded field only as a subset of the table's value, then drop it."""

    if not operation.superseded_fields:
        return dict(parameters)
    remaining = dict(parameters)
    for name, allowed in operation.superseded_fields.items():
        if name not in remaining:
            continue
        value = remaining.pop(name)
        if not isinstance(value, Mapping) or not value:
            raise OperationRequestError("invalid_request")
        for key, item in value.items():
            if not isinstance(key, str) or key not in allowed or allowed[key] != item or type(allowed[key]) is not type(item):
                raise OperationRequestError("invalid_request")
    return remaining


def validate_operation_request(operation_id: str, parameters: Any, *, deepline_catalog: Optional[Mapping[str, Any]] = None,
                               allow_full_thinking: bool = False) -> Dict[str, Any]:
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
        limits = _request_structure_limits(operation_id, allow_full_thinking)
        contracts.check_strict_document(parameters, limits)
    except contracts.ArenaContractError as exc:
        raise OperationRequestError("invalid_request") from exc
    if len(contracts.canonical_json(parameters).encode("utf-8")) > operation.max_request_bytes:
        raise OperationRequestError("request_too_large")
    selected_provider = None
    if allow_full_thinking and operation_id in {"openrouter.chat", "openrouter.responses"}:
        preference = parameters.get("provider")
        if isinstance(preference, Mapping) and "only" in preference:
            fixed = dict(operation.superseded_fields.get("provider") or {})
            selected = preference["only"]
            if (set(preference) - (set(fixed) | {"only"})
                    or any(type(value) is not type(fixed[key]) or value != fixed[key]
                           for key, value in preference.items() if key != "only")
                    or not isinstance(selected, list) or len(selected) != 1
                    or not isinstance(selected[0], str)
                    or re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,127}", selected[0]) is None):
                raise OperationRequestError("invalid_request")
            # This is an exact route selector, not a caller-owned provider
            # policy. The outbound builder still installs every host control.
            selected_provider = {"only": [selected[0]]}
            parameters = {key: value for key, value in parameters.items() if key != "provider"}
    parameters = _drop_superseded_fields(operation, parameters)
    request_fields = operation.request_fields
    if operation_id == "openrouter.chat" and not allow_full_thinking:
        # Existing rounds must retain the exact Chat schema and normalized
        # request hash they used before full-thinking protocol replay.
        message_spec = request_fields["messages"]
        assert message_spec.fields is not None
        message_fields = dict(message_spec.fields)
        message_fields["content"] = FieldSpec("str", max_length=OPENROUTER_MAX_CONTENT_CHARS)
        for name in ("reasoning", "reasoning_content", "reasoning_details", "refusal"):
            message_fields.pop(name)
        call_spec = message_fields["tool_calls"]
        assert call_spec.fields is not None
        call_fields = dict(call_spec.fields)
        call_fields.pop("index")
        message_fields["tool_calls"] = dataclasses.replace(call_spec, fields=call_fields)
        request_fields = dict(request_fields,
                              messages=dataclasses.replace(message_spec, fields=message_fields))
    elif operation_id == "openrouter.chat" and allow_full_thinking:
        # Copy the nested schema; the shared catalog and policy-off hashes
        # retain their original field definitions.
        message_spec = request_fields["messages"]
        message_fields = dict(message_spec.fields)
        detail_spec = message_fields["reasoning_details"]
        detail_fields = dict(detail_spec.fields)
        for name in ("text", "summary", "data"):
            detail_fields[name] = dataclasses.replace(detail_fields[name], max_length=contracts.FULL_THINKING_MAX_STRING_BYTES)
        message_fields["reasoning_details"] = dataclasses.replace(detail_spec, fields=detail_fields)
        call_spec = message_fields["tool_calls"]
        call_fields = dict(call_spec.fields)
        function_spec = call_fields["function"]
        function_fields = dict(function_spec.fields)
        function_fields["arguments"] = dataclasses.replace(function_fields["arguments"], max_length=contracts.FULL_THINKING_MAX_STRING_BYTES)
        call_fields["function"] = dataclasses.replace(function_spec, fields=function_fields)
        message_fields["tool_calls"] = dataclasses.replace(call_spec, fields=call_fields)
        request_fields = dict(request_fields, messages=dataclasses.replace(message_spec, fields=message_fields))
    if operation_id.startswith("deepline.tools.") and deepline_catalog is None:
        raise OperationRequestError("no_matching_operation")
    if operation_id in {"deepline.execute", "deepline.tools.get"} and deepline_catalog is not None:
        from lab_arena import deepline_catalog as catalog
        try:
            request_fields = dict(request_fields, tool=FieldSpec("str", required=True, choices=catalog.allowed_tool_ids(deepline_catalog)))
        except catalog.CatalogError as exc:
            raise OperationRequestError("invalid_request") from exc
    normalized = _validate_object(request_fields, parameters, "$", forbidden=operation.forbidden_names, limits=limits)
    if selected_provider is not None:
        normalized["provider"] = selected_provider
    if (operation_id == "openrouter.chat" and not allow_full_thinking
            and normalized.get("max_tokens", 0) > OPENROUTER_MAX_OUTPUT_TOKENS):
        raise OperationRequestError("invalid_field", "$.max_tokens")
    for name, default in operation.defaults.items():
        if name not in normalized:
            normalized[name] = _deep_copy_json(default)
    if operation_id == "deepline.tools.search":
        # The SDK sends q="" for category/term-only discovery. Keep empty
        # filters invalid and require at least one useful discovery input.
        if any(name in normalized and not normalized[name].strip()
               for name in ("categories", "search_terms", "task")):
            raise OperationRequestError("invalid_field")
        if not any(normalized.get(name, "").strip()
                   for name in ("query", "categories", "search_terms")):
            raise OperationRequestError("invalid_field")
    if operation_id == "openrouter.responses":
        _validate_responses(normalized, allow_full_thinking=allow_full_thinking)
    if operation_id == "openrouter.chat" and allow_full_thinking:
        # Generic object validation omits JSON nulls. A returned assistant
        # tool turn may carry them, and replay must keep the provider's shape.
        for original, message in zip(parameters["messages"], normalized["messages"]):
            if isinstance(original, Mapping):
                for field in ("content", "reasoning", "reasoning_content", "refusal"):
                    if field in original and original[field] is None:
                        message[field] = None
                for source, detail in zip(original.get("reasoning_details") or (),
                                          message.get("reasoning_details") or ()):
                    if isinstance(source, Mapping):
                        for field in ("id", "signature"):
                            if field in source and source[field] is None:
                                detail[field] = None
        for index, message in enumerate(normalized["messages"]):
            content = message.get("content")
            if not ((isinstance(content, str) and len(content) <= (contracts.FULL_THINKING_MAX_STRING_BYTES
                        if message["role"] == "assistant" else OPENROUTER_MAX_CONTENT_CHARS))
                    or (content is None and message["role"] == "assistant" and message.get("tool_calls"))
                    or ("content" not in message and message["role"] == "assistant" and message.get("tool_calls"))):
                raise OperationRequestError("invalid_field", "$.messages[%d].content" % index)
            history = any(key in message for key in ("reasoning", "reasoning_content", "reasoning_details"))
            if history and message["role"] != "assistant":
                raise OperationRequestError("invalid_field", "$.messages[%d].reasoning" % index)
            for field in ("reasoning", "reasoning_content", "refusal"):
                if field in message and message[field] is not None and (
                        not isinstance(message[field], str) or len(message[field]) > contracts.FULL_THINKING_MAX_STRING_BYTES):
                    raise OperationRequestError("invalid_field", "$.messages[%d].%s" % (index, field))
            for detail in message.get("reasoning_details", []):
                required = {"reasoning.text": "text", "reasoning.summary": "summary", "reasoning.encrypted": "data"}[detail["type"]]
                if required not in detail or any(key in detail for key in ({"text", "summary", "data"} - {required})):
                    raise OperationRequestError("invalid_field", "$.messages[%d].reasoning_details" % index)
                if any(detail.get(key) is not None and (not isinstance(detail[key], str) or len(detail[key]) > 256)
                       for key in ("id", "signature")):
                    raise OperationRequestError("invalid_field", "$.messages[%d].reasoning_details" % index)
    if operation.deepline_tool and deepline_catalog is not None:
        from lab_arena import deepline_catalog as catalog
        try:
            catalog.validate_payload(deepline_catalog, operation.deepline_tool, normalized)
        except catalog.CatalogError as exc:
            raise OperationRequestError("invalid_field", "$.payload") from exc
    if operation_id == "deepline.execute":
        tool = normalized.get("tool")
        payload = normalized.get("payload")
        if "response_contract" in parameters and (
                tool != "generic_http_request" or "response_contract" not in normalized):
            raise OperationRequestError("invalid_field", "$.response_contract")
        if deepline_catalog is not None:
            from lab_arena import deepline_catalog as catalog
            try:
                normalized["payload"] = catalog.validate_payload(deepline_catalog, tool, payload)
            except catalog.CatalogError as exc:
                raise OperationRequestError("invalid_field", "$.payload") from exc
        # These additions are read-only single-record operations. In particular,
        # no caller-controlled webhook URL is needed for synchronous judging.
        contact_fields = {
            "harvestapi_get_company": {"url", "universalName", "search"},
            "zerobounce_validate": {"email", "ip_address"},
            "bounceban_verify_single": {"email", "mode", "disable_catchall_verify"},
            "bounceban_get_single_status": {"id"},
            "harvestapi_get_profile": {"url", "publicIdentifier", "profileId", "findEmail", "main", "skipSmtp", "includeAboutProfile"},
        }
        if deepline_catalog is None and tool in contact_fields:
            if not isinstance(payload, Mapping) or set(payload) - contact_fields[tool]:
                raise OperationRequestError("invalid_field", "$.payload")
            if any(not isinstance(value, str) or not value or len(value) > 2048 for value in payload.values()):
                raise OperationRequestError("invalid_field", "$.payload")
            required = "id" if tool == "bounceban_get_single_status" else "email"
            if tool == "harvestapi_get_company":
                if len(payload) != 1:
                    raise OperationRequestError("invalid_field", "$.payload")
            elif tool == "harvestapi_get_profile":
                if not any(payload.get(key) for key in ("url", "publicIdentifier", "profileId")):
                    raise OperationRequestError("invalid_field", "$.payload")
            elif not payload.get(required):
                raise OperationRequestError("invalid_field", "$.payload")
    encoded = contracts.canonical_json(normalized).encode("utf-8")
    if len(encoded) > operation.max_request_bytes:
        raise OperationRequestError("request_too_large")
    return normalized


def _validate_responses(parameters: Mapping[str, Any], *, allow_full_thinking: bool = False) -> None:
    """Stateless text, local tools, and one bounded hosted web-search tool."""

    limits = _request_structure_limits("openrouter.responses", allow_full_thinking)
    text = FieldSpec("str", max_length=OPENROUTER_MAX_CONTENT_CHARS)
    history_text = dataclasses.replace(text, max_length=contracts.FULL_THINKING_MAX_STRING_BYTES) if allow_full_thinking else text
    identifier = FieldSpec("str", required=True, min_length=1, max_length=256)
    status = FieldSpec("str", choices=("in_progress", "completed", "incomplete"))
    item_fields = {
        "message": {"role": FieldSpec("str", required=True, choices=("system", "developer", "user", "assistant")), "content": FieldSpec("any", required=True), "status": status, "phase": FieldSpec("str", choices=("commentary", "final_answer"))},
        "function_call": {"call_id": identifier, "name": identifier, "namespace": dataclasses.replace(identifier, required=False), "arguments": dataclasses.replace(history_text, required=True), "status": status},
        "function_call_output": {"call_id": identifier, "output": FieldSpec("any", required=True), "status": status},
        "custom_tool_call": {"call_id": identifier, "name": identifier, "namespace": dataclasses.replace(identifier, required=False), "input": dataclasses.replace(history_text, required=True), "status": status},
        "custom_tool_call_output": {"call_id": identifier, "output": FieldSpec("any", required=True)},
        "additional_tools": {"role": FieldSpec("str", required=True, choices=("developer",)), "tools": FieldSpec("any", required=True)},
        "reasoning": {"summary": FieldSpec("any"), "encrypted_content": FieldSpec("str", max_length=contracts.FULL_THINKING_MAX_STRING_BYTES if allow_full_thinking else 128_000), "content": FieldSpec("any"), "status": status},
        "web_search_call": {
            "status": status,
            "action": FieldSpec("object", required=True),
        },
    }

    if allow_full_thinking:
        # Preserve the opaque provider tag only for the new protocol. Existing
        # rounds retain the previous release's accepted schema and request hash.
        item_fields["reasoning"]["format"] = FieldSpec("str", min_length=1, max_length=128)
        # OpenRouter's server tool returns this provider-specific item type.
        # Admit its observed search shape only; old rounds retain their schema.
        item_fields["openrouter:web_search"] = {
            "status": dataclasses.replace(status, required=True),
            "action": FieldSpec("object", required=True),
        }

    local_tool_fields = {"type": FieldSpec("str", required=True, choices=("function", "custom")), "name": identifier, "description": text}

    def validate_local_tools(
        tools: Any, path: str, *, namespaces_allowed: bool,
        hosted_web_search_allowed: bool = False,
    ) -> tuple[int, int]:
        if not isinstance(tools, list) or len(tools) > 64:
            raise OperationRequestError("invalid_field", path)
        total = 0
        web_searches = 0
        for tool in tools:
            kind = tool.get("type") if isinstance(tool, dict) else None
            if kind == "openrouter:web_search" and hosted_web_search_allowed:
                _validate_object({
                    "type": FieldSpec(
                        "str", required=True,
                        choices=("openrouter:web_search",),
                    ),
                    "parameters": FieldSpec("object", required=True, fields={
                        "engine": FieldSpec(
                            "str", required=True, choices=("native", "exa"),
                        ),
                        "max_uses": FieldSpec(
                            "int", required=True, minimum=1,
                            maximum=OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS,
                        ),
                        "max_total_results": FieldSpec(
                            "int", required=True, minimum=1,
                            maximum=OPENROUTER_WEB_SEARCH_MAX_TOTAL_RESULTS,
                        ),
                    }),
                }, tool, path, limits=limits)
                web_searches += 1
                continue
            if kind == "namespace" and namespaces_allowed:
                _validate_object({
                    "type": FieldSpec("str", required=True, choices=("namespace",)),
                    "name": identifier,
                    "description": dataclasses.replace(text, required=True),
                    "tools": FieldSpec("any", required=True),
                }, tool, path, limits=limits)
                total += 1
                nested_count, nested_web_searches = validate_local_tools(
                    tool["tools"], path + ".tools", namespaces_allowed=False,
                )
                total += nested_count
                web_searches += nested_web_searches
                continue
            fields = dict(local_tool_fields)
            if kind == "function":
                fields.update(
                    parameters=FieldSpec("object"), strict=FieldSpec("bool"),
                    output_schema=FieldSpec("object"), async_=FieldSpec("bool"),
                    defer_loading=FieldSpec("bool"),
                    allowed_callers=FieldSpec("list[str]", max_length=2, item=FieldSpec("str", choices=("direct", "programmatic"))),
                )
                # ``async`` is a schema field, while ``async_`` is a Python identifier.
                fields["async"] = fields.pop("async_")
            elif kind == "custom":
                fields["format"] = FieldSpec("object", fields={
                    "type": FieldSpec("str", required=True, choices=("text", "grammar")),
                    "syntax": FieldSpec("str", choices=("lark", "regex")),
                    "definition": text,
                })
            else:
                raise OperationRequestError("invalid_field", path + ".type")
            _validate_object(fields, tool, path, limits=limits)
            total += 1
        return total, web_searches

    def validate_web_search_action(action: Any) -> None:
        if not isinstance(action, Mapping):
            raise OperationRequestError("invalid_field", "$.input.action")
        action_type = action.get("type")
        fields = {"type": FieldSpec("str", required=True)}
        if action_type == "search":
            fields.update(
                query=FieldSpec("str", max_length=OPENROUTER_MAX_CONTENT_CHARS),
                queries=FieldSpec(
                    "list[str]", max_length=128,
                    item=FieldSpec("str", min_length=1,
                                   max_length=OPENROUTER_MAX_CONTENT_CHARS),
                ),
                sources=FieldSpec(
                    "list[object]", max_length=128, fields={
                        "type": FieldSpec(
                            "str", required=True, choices=("url",),
                        ),
                        "url": FieldSpec(
                            "str", required=True, max_length=8_192,
                        ),
                    },
                ),
            )
        elif action_type == "open_page":
            fields["url"] = FieldSpec(
                "str", required=True, max_length=8_192,
            )
        elif action_type == "find_in_page":
            fields.update(
                url=FieldSpec(
                    "str", required=True, max_length=8_192,
                ),
                pattern=FieldSpec(
                    "str", required=True,
                    max_length=OPENROUTER_MAX_CONTENT_CHARS,
                ),
            )
        else:
            raise OperationRequestError("invalid_field", "$.input.action.type")
        normalized_action = _validate_object(
            fields, action, "$.input.action",
            limits=limits,
        )
        if "url" in normalized_action:
            validate_response_url(normalized_action["url"], "$.input.action.url")
        for index, source in enumerate(normalized_action.get("sources", [])):
            validate_response_url(
                source["url"], "$.input.action.sources[%d].url" % index,
            )

    def validate_response_url(value: Any, path: str) -> None:
        """Validate bounded provider-returned HTTP(S) links without rewriting."""

        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 8_192
            or any(char.isspace() or ord(char) < 0x20 for char in value)
        ):
            raise OperationRequestError("invalid_field", path)
        try:
            parts = urlsplit(value)
        except ValueError as exc:
            raise OperationRequestError("invalid_field", path) from exc
        if parts.scheme not in ("http", "https") or not parts.netloc or not parts.hostname:
            raise OperationRequestError("invalid_field", path)

    def citation_annotations(value: Any) -> None:
        if not isinstance(value, list) or len(value) > 128:
            raise OperationRequestError("invalid_field", "$.input.content.annotations")
        for annotation in value:
            _validate_object({
                "type": FieldSpec(
                    "str", required=True, choices=("url_citation",),
                ),
                "url": FieldSpec(
                    "str", required=True, max_length=8_192,
                ),
                "title": FieldSpec(
                    "str", required=True,
                    max_length=OPENROUTER_MAX_CONTENT_CHARS,
                ),
                "start_index": FieldSpec("int", minimum=0),
                "end_index": FieldSpec("int", minimum=0),
                "content": FieldSpec(
                    "str", max_length=OPENROUTER_MAX_CONTENT_CHARS,
                ),
            }, annotation, "$.input.content.annotations",
               limits=limits)
            validate_response_url(
                annotation["url"], "$.input.content.annotations.url",
            )

    def content_parts(value: Any, *, reasoning: bool = False, assistant: bool = False) -> None:
        content_text = history_text if reasoning or assistant else text
        if isinstance(value, str) and not reasoning:
            _validate_field(content_text, value, "$.input.content")
            return
        if not isinstance(value, list) or len(value) > OPENROUTER_MAX_MESSAGES:
            raise OperationRequestError("invalid_field", "$.input.content")
        for part in value:
            fields = {
                "type": FieldSpec("str", required=True, choices=(("summary_text", "reasoning_text") if reasoning else ("input_text", "output_text"))),
                "text": dataclasses.replace(content_text, required=True),
                "annotations": FieldSpec("any"),
            }
            if allow_full_thinking and isinstance(part, Mapping) and part.get("type") == "output_text":
                # Providers include this empty transport field even when no
                # token probabilities were requested. Preserve replay shape.
                fields["logprobs"] = FieldSpec("list[object]", max_length=OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS,
                    fields={"token": text, "logprob": FieldSpec("float"),
                            "bytes": FieldSpec("any"), "top_logprobs": FieldSpec("list[object]", max_length=20,
                                fields={"token": text, "logprob": FieldSpec("float"), "bytes": FieldSpec("any")})})
            normalized_part = _validate_object(fields, part, "$.input.content", limits=limits)
            if "annotations" in normalized_part:
                citation_annotations(normalized_part["annotations"])

    def tool_output(value: Any) -> None:
        if isinstance(value, str):
            _validate_field(text, value, "$.input.output")
        elif isinstance(value, list) and 1 <= len(value) <= OPENROUTER_MAX_MESSAGES:
            for part in value:
                _validate_object({
                    "type": FieldSpec("str", required=True, choices=("input_text",)),
                    "text": dataclasses.replace(text, required=True),
                }, part, "$.input.output")
        else:
            raise OperationRequestError("invalid_field", "$.input.output")

    value = parameters["input"]
    additional_tool_count = 0
    if isinstance(value, str):
        _validate_field(text, value, "$.input")
    elif isinstance(value, list) and 1 <= len(value) <= OPENROUTER_RESPONSES_MAX_INPUT_ITEMS:
        for item in value:
            kind = item.get("type", "message") if isinstance(item, dict) else None
            if kind not in item_fields:
                raise OperationRequestError("invalid_field", "$.input.type")
            _validate_object(dict(item_fields[kind], type=FieldSpec("str", choices=(kind,)), id=dataclasses.replace(identifier, required=kind == "openrouter:web_search")), item, "$.input", limits=limits)
            if kind == "message":
                content_parts(item["content"], assistant=allow_full_thinking and item["role"] == "assistant")
                if "phase" in item and item["role"] != "assistant":
                    raise OperationRequestError("invalid_field", "$.input.phase")
            elif kind == "additional_tools":
                count, hosted_count = validate_local_tools(
                    item["tools"], "$.input.tools", namespaces_allowed=True,
                )
                additional_tool_count += count
                if hosted_count:
                    raise OperationRequestError("invalid_field", "$.input.tools")
                if additional_tool_count > 64:
                    raise OperationRequestError("invalid_field", "$.input.tools")
            elif kind in ("function_call_output", "custom_tool_call_output"):
                tool_output(item["output"])
            elif kind == "reasoning":
                for name in ("summary", "content"):
                    if item.get(name) is not None:
                        content_parts(item[name], reasoning=True)
            elif kind in ("web_search_call", "openrouter:web_search"):
                if kind == "openrouter:web_search":
                    action = item["action"]
                    if action.get("type") != "search":
                        raise OperationRequestError("invalid_field", "$.input.action.type")
                    if not action.get("query") and not action.get("queries"):
                        raise OperationRequestError("invalid_field", "$.input.action.query")
                validate_web_search_action(item["action"])
    else:
        raise OperationRequestError("invalid_field", "$.input")
    root_tool_count, web_search_count = validate_local_tools(
        parameters.get("tools", []), "$.tools", namespaces_allowed=True,
        hosted_web_search_allowed=True,
    )
    if additional_tool_count + root_tool_count > 64:
        raise OperationRequestError("invalid_field", "$.tools")
    if web_search_count not in (0, 1):
        raise OperationRequestError("invalid_field", "$.tools")
    if web_search_count:
        if parameters.get("max_tool_calls") != OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS:
            raise OperationRequestError("invalid_field", "$.max_tool_calls")
    elif "max_tool_calls" in parameters:
        raise OperationRequestError("invalid_field", "$.max_tool_calls")


# ---------------------------------------------------------------------------
# Request matching (used by the shim; the broker re-validates the frame)
# ---------------------------------------------------------------------------


def check_request_headers(headers: Mapping[str, Any], *, catalog_metadata: bool = False,
                          deepline_generic_http: bool = False) -> None:
    """Refuse caller credentials and any header outside the allowlist."""

    if headers is None:
        return
    if not isinstance(headers, Mapping):
        raise OperationRequestError("invalid_request")
    content_type = None
    native_headers = set()
    for name in headers:
        lowered = str(name).strip().lower()
        if lowered in CREDENTIAL_HEADERS:
            raise OperationRequestError("forbidden_header")
        if deepline_generic_http and lowered in {**DEEPLINE_EXECUTE_HEADERS, **DEEPLINE_GENERIC_HTTP_HEADERS}:
            expected = {**DEEPLINE_EXECUTE_HEADERS, **DEEPLINE_GENERIC_HTTP_HEADERS}[lowered]
            if lowered in native_headers or headers[name] != expected:
                raise OperationRequestError("invalid_request")
            native_headers.add(lowered)
            continue
        if lowered not in ALLOWED_REQUEST_HEADERS and not (catalog_metadata and lowered == "x-deepline-tool-meta-only"):
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


def _match_operation(method: str, url: str, *, catalog_metadata: bool = False) -> Tuple[Operation, Dict[str, str]]:
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
    for operation in _OPERATION_LIST + (_CATALOG_OPERATION_LIST if catalog_metadata else ()):
        if operation.method != method or operation.host != host:
            continue
        if not operation.path_fields:
            if operation.path == parts.path:
                return operation, {}
            continue
        matched = _path_pattern(operation).fullmatch(parts.path)
        if matched is not None:
            return operation, dict(matched.groupdict())
    raise OperationRequestError("no_matching_operation")


def _path_pattern(operation: Operation) -> "re.Pattern[str]":
    pattern = re.escape(operation.path)
    for name in operation.path_fields:
        pattern = pattern.replace(re.escape("{%s}" % name), "(?P<%s>[A-Za-z0-9_]{1,80})" % name)
    return re.compile("^" + pattern + "$")


def _render_path(operation: Operation, normalized: Mapping[str, Any]) -> str:
    path = operation.path
    for name in operation.path_fields:
        value = normalized[name]  # validated against the closed choices
        path = path.replace("{%s}" % name, str(value))
    return path


def match_request(
    method: str,
    url: str,
    body: Optional[bytes],
    headers: Optional[Mapping[str, Any]],
    *,
    deepline_catalog: Optional[Mapping[str, Any]] = None,
    allow_full_thinking: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Map one client request to ``(operation_id, normalized_parameters)``.

    Exactly one operation matches by method, constant host, and exact path
    (port 443 only, no userinfo, no fragment). POST operations take a JSON
    object body and no query string; GET operations take declared query
    parameters and no body. Headers are checked against the allowlist and
    never forwarded.
    """

    operation, path_parameters = _match_operation(method, url, catalog_metadata=deepline_catalog is not None)
    generic_http = (operation.operation_id == "deepline.execute"
                    and path_parameters.get("tool") == "generic_http_request")
    check_request_headers(headers or {}, catalog_metadata=operation.operation_id in CATALOG_OPERATIONS,
                          deepline_generic_http=generic_http)
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
        for name, value in path_parameters.items():
            if name in parameters:
                raise OperationRequestError("invalid_body")
            parameters[name] = value
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
        seen = set()
        for name, value in pairs:
            # The public search route uses q; the local broker uses query.
            # Normalize before duplicate checks so both names cannot conflict.
            if operation.operation_id == "deepline.tools.search" and name == "q":
                name = "query"
            if name in seen:
                raise OperationRequestError("invalid_query")
            seen.add(name)
            if (
                operation.provider == "scrapingdog"
                and name == "api_key"
                and value == SCRAPINGDOG_RUNTIME_HANDLE
            ):
                continue
            spec = operation.request_fields.get(name)
            parameters[name] = _coerce_query_value(spec, value, "$." + name) if spec is not None else value
        for name, value in path_parameters.items():
            if name in parameters:
                raise OperationRequestError("invalid_query")
            parameters[name] = value
    if generic_http:
        native_headers = {str(name).strip().lower(): value for name, value in (headers or {}).items()}
        contract = native_headers.get("x-deepline-execute-response-contract")
        if contract is not None:
            if "response_contract" in parameters and parameters["response_contract"] != contract:
                raise OperationRequestError("invalid_body")
            parameters["response_contract"] = contract
        if ("x-deepline-tool-error-schema" in native_headers
                and parameters.get("response_contract") != "raw-v2"):
            raise OperationRequestError("invalid_request")
    return operation.operation_id, validate_operation_request(
        operation.operation_id, parameters, deepline_catalog=deepline_catalog,
        allow_full_thinking=allow_full_thinking,
    )


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
    headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


def outbound_target(operation_id: str, parameters: Any, *, deepline_catalog: Optional[Mapping[str, Any]] = None) -> OutboundTarget:
    """The constant target of an operation; parameters are validated only."""

    operation = _operation(operation_id)
    normalized = validate_operation_request(operation_id, parameters, deepline_catalog=deepline_catalog)
    return OutboundTarget(operation.method, "https", operation.outbound_host or operation.host, 443, operation.outbound_path or _render_path(operation, normalized))


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise OperationRequestError("invalid_field")


def _explicit_responses_open_maps(document: Dict[str, Any]) -> bool:
    """Make only simple native open-map defaults explicit on a copied wire.

    References, composition and unevaluated rules can make annotations affect
    validation. Leave those schemas intact. Never walk example/default data.
    https://json-schema.org/understanding-json-schema/reference/object#additionalproperties
    """
    unsupported = {
        "$ref", "$dynamicRef", "$recursiveRef", "$defs", "definitions",
        "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
        "unevaluatedProperties", "unevaluatedItems", "patternProperties",
        "dependentSchemas", "dependencies", "propertyNames", "contains",
        "prefixItems", "additionalItems", "contentSchema",
    }

    def children(schema):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            yield from properties.values()
        items = schema.get("items")
        if isinstance(items, (dict, bool)):
            yield items
        elif isinstance(items, list):
            yield from items

    def simple(schema):
        if isinstance(schema, bool):
            return True
        if not isinstance(schema, dict) or unsupported.intersection(schema):
            return False
        extra = schema.get("additionalProperties")
        return (all(simple(child) for child in children(schema))
                and (not isinstance(extra, dict) or simple(extra)))

    def expand(schema):
        if not isinstance(schema, dict):
            return False
        changed = False
        if (schema.get("type") == "object" and schema.get("properties") == {}
                and "additionalProperties" not in schema):
            schema["additionalProperties"] = True
            changed = True
        for child in children(schema):
            changed |= expand(child)
        return changed

    def tools(values):
        changed = False
        for tool in values:
            if tool.get("type") == "namespace":
                changed |= tools(tool["tools"])
            elif tool.get("type") == "function" and tool.get("strict") is not True:
                schema = tool.get("parameters")
                if isinstance(schema, dict) and simple(schema):
                    changed |= expand(schema)
        return changed

    changed = tools(document.get("tools", []))
    if isinstance(document.get("input"), list):
        for item in document["input"]:
            if item.get("type") == "additional_tools":
                changed |= tools(item["tools"])
    return changed


def build_outbound_request(
    operation_id: str,
    parameters: Any,
    *,
    openrouter_provider_policy: Optional[Mapping[str, Any]] = None,
    deepline_catalog: Optional[Mapping[str, Any]] = None,
    allow_full_thinking: bool = False,
    source_html_profile: bool = False,
    industry_schema_order: bool = False,
) -> OutboundRequest:
    """Build the credential-free outbound request from validated parameters.

    Fixed params are merged last so nothing model-supplied can shadow them.
    The broker may narrow an OpenRouter call with one host-owned provider
    policy derived from its source-controlled model route.  The broker places
    the credential per ``operation.credential`` and adds nothing else from the
    model request. Only the broker's HTML fallback selects the fixed target
    header profile; it is not a caller request field.
    """

    operation = _operation(operation_id)
    normalized = validate_operation_request(operation_id, parameters, deepline_catalog=deepline_catalog,
                                            allow_full_thinking=allow_full_thinking)
    if type(industry_schema_order) is not bool or industry_schema_order and (
        operation_id != "openrouter.chat"
        or normalized.get("model") != "openai/gpt-6.1-sol"
        or openrouter_provider_policy is None
    ):
        raise OperationRequestError("invalid_request")
    if type(source_html_profile) is not bool or source_html_profile and (
        operation_id != "deepline.execute"
        or normalized.get("tool") != "generic_http_request"
        or normalized["payload"].get("method") != "GET"
        or "headers" in normalized["payload"]
    ):
        raise OperationRequestError("invalid_request")
    if allow_full_thinking and (operation_id not in {"openrouter.chat", "openrouter.responses"}
                                or openrouter_provider_policy is None):
        raise OperationRequestError("invalid_request")
    path = operation.outbound_path or _render_path(operation, normalized)
    host = operation.outbound_host or operation.host
    target = OutboundTarget(operation.method, "https", host, 443, path)
    base_url = "https://%s%s" % (host, path)
    if operation.parameter_location == "body":
        document = {name: value for name, value in normalized.items() if name not in operation.path_fields}
        document.update(_deep_copy_json(operation.fixed_params))
        schema_expanded = False
        if allow_full_thinking and operation_id == "openrouter.responses":
            # Admission/hash/capture parameters remain the caller's originals.
            # Only this outbound copy receives the equivalent open-map default.
            document = _deep_copy_json(document)
            schema_expanded = _explicit_responses_open_maps(document)
        if (allow_full_thinking and operation_id == "openrouter.responses"
                and document.get("parallel_tool_calls") is True):
            # The conventional default permits parallel calls; it does not
            # require them. Explicit true can exclude otherwise capable pinned
            # endpoints under require_parameters. Keep the caller's original
            # intent in hashed/captured parameters and preserve explicit false.
            # https://openrouter.ai/docs/api_reference/parameters#parallel-tool-calls
            document.pop("parallel_tool_calls")
        if (allow_full_thinking and operation_id == "openrouter.responses"
                and document.get("parallel_tool_calls") is False
                and document.get("tools") == []
                and document.get("tool_choice", "auto") in ("auto", "none")
                and not (isinstance(document["input"], list) and any(
                    item.get("type") == "additional_tools"
                    for item in document["input"]
                ))):
            # A validated request with no callable tools cannot make parallel
            # calls. Omit only its irrelevant wire control; retain the original
            # false in admission/hash/capture. Never omit false for real tools
            # or additional declarations, including empty declarations.
            document.pop("parallel_tool_calls")
        if openrouter_provider_policy is not None:
            if operation_id not in {"openrouter.chat", "openrouter.responses"} or not isinstance(
                operation.fixed_params.get("provider"), Mapping
            ):
                raise OperationRequestError("invalid_request")
            policy = _deep_copy_json(openrouter_provider_policy)
            price_fields = {"prompt", "completion", "request"}
            policy_fields = {"data_collection", "zdr", "allow_fallbacks", "max_price"}
            policy_shapes = (
                policy_fields,
                policy_fields | {"order"},
                policy_fields | {"only"},
            )
            exact_full_route = (
                allow_full_thinking
                and isinstance(policy, dict)
                and set(policy) == policy_fields | {"only", "require_parameters"}
                and isinstance(policy.get("only"), list)
                and len(policy["only"]) == 1
                and isinstance(policy["only"][0], str)
                and re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,127}", policy["only"][0]) is not None
                and policy.get("allow_fallbacks") is False
                and policy.get("require_parameters") is True
            )
            industry_region_route = (
                not allow_full_thinking
                and operation_id == "openrouter.chat"
                and normalized.get("model") == "openai/gpt-6.1-sol"
                and isinstance(policy, dict)
                and set(policy) == policy_fields | {"only"}
                and policy.get("only") == ["azure/us", "azure/eu"]
                and policy.get("allow_fallbacks") is True
            )
            if (
                not isinstance(policy, dict)
                or (set(policy) not in policy_shapes and not exact_full_route)
                or (allow_full_thinking and not exact_full_route)
                or (
                    set(policy) != policy_fields
                    and not exact_full_route
                    and not industry_region_route
                    and (
                        normalized.get("model") not in {
                            "openai/gpt-5.6-luna",
                            "openai/gpt-6-luna",
                        }
                        or (
                            policy.get("order") != ["azure/us"]
                            and policy.get("only")
                            != ["azure/us", "azure/eu"]
                        )
                    )
                )
                or policy.get("data_collection") != "deny"
                or policy.get("zdr") is not True
                or (operation_id == "openrouter.chat" and not exact_full_route and not industry_region_route)
                or type(policy.get("allow_fallbacks")) is not bool
                or not isinstance(policy.get("max_price"), dict)
                or set(policy["max_price"]) != price_fields
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= value <= 1_000_000
                    for value in policy["max_price"].values()
                )
            ):
                raise OperationRequestError("invalid_request")
            if allow_full_thinking and normalized.get("provider") is not None:
                if normalized["provider"]["only"] != policy.get("only"):
                    raise OperationRequestError("invalid_request")
            document["provider"] = policy
        document = _wrap_body(operation, normalized, document, deepline_catalog=deepline_catalog)
        if source_html_profile:
            document = {**document, "payload": {
                **document["payload"], "headers": dict(SOURCE_HTML_REQUEST_HEADERS),
            }}
            if deepline_catalog is not None:
                from jsonschema import Draft202012Validator
                from lab_arena import deepline_catalog as catalog
                schema = catalog.tool_entry(deepline_catalog, normalized["tool"])["input_schema"]
                if not Draft202012Validator(schema).is_valid(document["payload"]):
                    raise OperationRequestError("invalid_request")
        body = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        if industry_schema_order:
            # Structured outputs follow property order. Canonical storage sorts
            # maps, but preserves the industry's facts-before-verdict sequence
            # in `required`. Restore only that top-level sequence on the wire.
            ordered = json.loads(body)
            response_format = ordered.get("response_format", {})
            declaration = response_format.get("json_schema", {})
            schema = declaration.get("schema", {})
            schema = schema if isinstance(schema, dict) else {}
            properties, required = schema.get("properties"), schema.get("required")
            if (response_format.get("type") == "json_schema"
                    and declaration.get("name") == "industry_proposal"
                    and declaration.get("strict") is True
                    and isinstance(properties, dict) and isinstance(required, list)
                    and all(isinstance(key, str) for key in required)
                    and len(required) == len(properties) and set(required) == set(properties)):
                schema["properties"] = {key: properties[key] for key in required}
                body = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False,
                                  allow_nan=False).encode("utf-8")
        if schema_expanded:
            try:
                contracts.check_strict_document(document, _request_structure_limits(operation_id, allow_full_thinking))
            except contracts.ArenaContractError as exc:
                raise OperationRequestError("invalid_request") from exc
            if len(body) > operation.max_request_bytes:
                raise OperationRequestError("request_too_large")
        headers = operation.outbound_headers
        if (operation_id == "deepline.execute"
                and normalized.get("response_contract") == "raw-v2"):
            headers = MappingProxyType({**headers, **DEEPLINE_GENERIC_HTTP_HEADERS})
        return OutboundRequest(target, base_url, MappingProxyType({}), body, "application/json", operation.credential, headers)
    query: Dict[str, str] = {name: _query_value(value) for name, value in normalized.items()}
    for name, value in operation.fixed_params.items():
        query[name] = _query_value(value)
    ordered = MappingProxyType({key: query[key] for key in sorted(query)})
    return OutboundRequest(target, base_url + "?" + urlencode(ordered), ordered, b"", None, operation.credential, operation.outbound_headers)


def _wrap_body(operation: Operation, normalized: Mapping[str, Any], document: Dict[str, Any], *, deepline_catalog: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Apply the operation's named body transform (table data, never model data)."""

    if operation.body_wrapper == "deepline_execute":
        tool = str(normalized["tool"])
        if deepline_catalog is not None:
            from lab_arena import deepline_catalog as catalog
            provider = catalog.tool_entry(deepline_catalog, tool)["provider"]
        else:
            provider = DEEPLINE_TOOL_PROVIDERS[tool]
        return {"provider": provider, "operation": tool, "payload": document["payload"]}
    if operation.body_wrapper == "deepline_exa_compat":
        # The validated Exa request is the Deepline payload as-is: Deepline's
        # Exa tools accept Exa's own field names (``ids`` included).
        return {"provider": DEEPLINE_TOOL_PROVIDERS[operation.deepline_tool], "operation": operation.deepline_tool, "payload": document}
    return document


# ---------------------------------------------------------------------------
# Response sanitizer
# ---------------------------------------------------------------------------


def _project_oversized_html_scripts(original: bytes, limit: int) -> bytes:
    """Keep complete oversized script payloads out of the HTML text allowance.

    The provider reply already has its upstream response bound. Locate
    spans with the stdlib parser, but preserve the original tags and all other
    bytes rather than serializing HTML or joining text across removed tags.
    """
    if len(original) <= limit:
        return original
    try:
        raw_html = original.decode("utf-8")
    except UnicodeDecodeError:
        # Preserve the existing byte-cap behavior for non-UTF-8 source data.
        return original

    class ScriptSpans(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.content_start: Optional[int] = None
            self.spans: list[tuple[int, int]] = []
            self.line = 1
            self.line_start = 0

        def position(self) -> int:
            line, column = self.getpos()
            # Positions advance monotonically. Avoid a per-line index whose
            # memory would grow with every newline in the bounded reply.
            while self.line < line:
                self.line_start = raw_html.index("\n", self.line_start) + 1
                self.line += 1
            return self.line_start + column

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            if tag == "script":
                self.content_start = self.position() + len(self.get_starttag_text())

        def handle_startendtag(self, tag: str, attrs: Any) -> None:
            # A self-closing token does not delimit a script content span.
            pass

        def handle_endtag(self, tag: str) -> None:
            if tag != "script" or self.content_start is None:
                return
            end = self.position()
            closing_end = raw_html.find(">", end)
            if (
                closing_end >= 0
                and re.fullmatch(r"</script\s*>", raw_html[end:closing_end + 1], re.I)
                and len(raw_html[self.content_start:end].encode("utf-8")) > limit
            ):
                self.spans.append((self.content_start, end))
            self.content_start = None

    parser = ScriptSpans()
    try:
        parser.feed(raw_html)
        parser.close()
    except (AssertionError, ValueError):
        # HTMLParser rejects malformed marked sections. Do not use partial
        # spans from a page that could not be parsed; retain the old response.
        return original
    if not parser.spans:
        return original
    parts = []
    previous = 0
    for start, end in parser.spans:
        parts.append(raw_html[previous:start])
        previous = end
    parts.append(raw_html[previous:])
    return "".join(parts).encode("utf-8")


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


def scrub_person_entities(value: Any) -> Any:
    """Drop ``entities`` items whose ``type`` is ``person`` at any depth; everything else is unchanged."""

    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "entities" and isinstance(item, list):
                out[key] = [scrub_person_entities(entry) for entry in item if not (isinstance(entry, Mapping) and str(entry.get("type") or "").lower() == "person")]
            else:
                out[key] = scrub_person_entities(item)
        return out
    if isinstance(value, list):
        return [scrub_person_entities(item) for item in value]
    return value


def sanitize_response(
    operation_id: str,
    status: Any,
    headers: Optional[Mapping[str, Any]],
    body: Any,
    parameters: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """Return the model-visible ``(status, headers, body)`` for a provider reply.

    Every provider header is dropped except a validated ``content-type``;
    ``content-length`` is recomputed. Host-account, rate-limit, and provider
    server failures become one generic 502. JSON operations require an object or array within the size cap
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
    generic = provider_status_is_infrastructure(status)
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
        if operation.response_transform == "deepline_unwrap" and isinstance(parsed, dict):
            # A completed Deepline envelope becomes the provider's own reply so
            # Exa-shaped clients read it unchanged; anything else passes through.
            inner = parsed.get("result")
            if parsed.get("status") == "completed" and isinstance(inner, dict) and isinstance(inner.get("data"), (dict, list)):
                parsed = inner["data"]
                raw = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        if operation.response_scrub == "deepline_person_entities":
            tool = operation.deepline_tool or str((parameters or {}).get("tool") or "")
            if tool not in DEEPLINE_PERSON_ENTITY_TOOLS:
                scrubbed = scrub_person_entities(parsed)
                if scrubbed != parsed:  # bodies without person entities pass through byte-for-byte
                    parsed = scrubbed
                    raw = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        content_type = "application/json"
    else:
        content_type = _provider_content_type(headers or {}) or "text/plain"
        if (operation_id == "scrapingdog.scrape" and 200 <= status < 300
                and content_type in {"text/html", "application/xhtml+xml"}):
            raw = _project_oversized_html_scripts(raw, operation.max_response_bytes)
        if len(raw) > operation.max_response_bytes:
            raw = _truncate_utf8(raw, operation.max_response_bytes)
    return status, {"content-type": content_type, "content-length": str(len(raw))}, raw


__all__ = [
    "ALLOWED_REQUEST_HEADERS",
    "BUDGET_ADMISSION_MAX_SECONDS",
    "PROVIDER_API_TIMEOUT_GRACE_SECONDS",
    "PROVIDER_BILLING_RECONCILIATION_SECONDS",
    "CREDENTIAL_HEADERS",
    "HOST_ACCOUNT_STATUSES",
    "CredentialPlacement",
    "ERROR_CODES",
    "FIELD_FORMATS",
    "FIELD_KINDS",
    "FORBIDDEN_FIELD_NAMES",
    "FieldSpec",
    "OPENROUTER_MAX_OUTPUT_TOKENS",
    "OPENROUTER_RESPONSES_MAX_INPUT_ITEMS",
    "OPENROUTER_RESPONSES_MAX_OUTPUT_TOKENS",
    "OPENROUTER_STRICT_PROVIDER_POLICY",
    "OPENROUTER_OUTBOUND_HEADERS",
    "OPERATIONS",
    "CATALOG_OPERATIONS",
    "get_operation",
    "OPERATION_LIMITS",
    "RESPONSES_OPERATION_LIMITS",
    "DEEPLINE_TOOLS",
    "DEEPLINE_TOOL_PROVIDERS",
    "DEEPLINE_EXECUTE_HEADERS",
    "PAYLOAD_FORBIDDEN_FIELD_NAMES",
    "OPERATION_TABLE_SCHEMA_VERSION",
    "TRUSTED_RESPONSE_URL_HEADER",
    "Operation",
    "OperationError",
    "OperationRequestError",
    "OperationResponseError",
    "OutboundRequest",
    "OutboundTarget",
    "scrub_person_entities",
    "DEEPLINE_PERSON_ENTITY_TOOLS",
    "build_outbound_request",
    "check_request_headers",
    "field_spec_document",
    "match_request",
    "operation_document",
    "operation_table_document",
    "provider_status_is_infrastructure",
    "outbound_target",
    "sanitize_response",
    "validate_https_url",
    "validate_operation_request",
]
