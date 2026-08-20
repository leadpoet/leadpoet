"""Closed, fail-closed PredictLeads composite intent workflows.

The model catalog describes three PredictLeads SourceAdd tools whose first
operation is not sufficient to produce evidence.  This module contains the
host-side execution graph for those tools.  It intentionally accepts action
IDs and sanitized payloads only; a caller cannot provide an HTTP endpoint,
provider URL, credential, or arbitrary operation.

The provider callable is injected by the runtime.  It receives one reviewed
action ID, one generated payload, and the route timeout.  It is called at most
once for each action.  A reservation callback must reserve the complete route
before the first provider call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import re
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit


WORKFLOW_SCHEMA_VERSION = "leadpoet.predictleads.intent-workflow:v1"
ROUTE_TIMEOUT_MS = 30_000
PREDICTLEADS_CREDIT_MICROUNITS = 560_000
EXA_CREDIT_MICROUNITS = 560_000

ROUTE_CONNECTIONS = "intent.source_add.predictleads_connections"
ROUTE_NEWS = "intent.source_add.predictleads_news"
ROUTE_TECHNOLOGY = "intent.source_add.predictleads_technology"

ACTION_COMPANY = "predictleads_company"
ACTION_CONNECTIONS = "predictleads_company_connections"
ACTION_NEWS = "predictleads_company_news_events"
ACTION_DETECTIONS = "predictleads_company_technology_detections"
ACTION_TECHNOLOGY = "predictleads_technology"
ACTION_JOB = "predictleads_job_opening"
ACTION_EXA = "exa_search"


class PredictLeadsWorkflowError(ValueError):
    """Raised only for an invalid injected runtime contract."""


@dataclass(frozen=True)
class ReviewedAction:
    """One closed provider operation allowed by these workflows."""

    action_id: str
    allowed_fields: frozenset[str]
    fixed_fields: Mapping[str, Any]

    def payload(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(values, Mapping):
            raise PredictLeadsWorkflowError("action payload is not a mapping")
        if set(values) - self.allowed_fields:
            raise PredictLeadsWorkflowError("action payload contains an unreviewed field")
        result = dict(self.fixed_fields)
        result.update(values)
        if set(result) - self.allowed_fields:
            raise PredictLeadsWorkflowError("action fixed field is not reviewed")
        return result


# This is deliberately not a URL registry.  A signed runtime catalog may map
# these IDs to private transport bindings.  User data can select neither an
# action nor an endpoint.
REVIEWED_ACTIONS: Mapping[str, ReviewedAction] = MappingProxyType(
    {
        ACTION_COMPANY: ReviewedAction(
            ACTION_COMPANY,
            frozenset({"id_or_domain"}),
            {},
        ),
        ACTION_CONNECTIONS: ReviewedAction(
            ACTION_CONNECTIONS,
            frozenset({
                "company_id_or_domain",
                "first_seen_at_from",
                "first_seen_at_until",
                "categories",
                "page",
                "limit",
            }),
            {"page": 1, "limit": 25},
        ),
        ACTION_NEWS: ReviewedAction(
            ACTION_NEWS,
            frozenset({
                "company_id_or_domain",
                "found_at_from",
                "found_at_until",
                "categories",
                "page",
                "limit",
            }),
            {"page": 1, "limit": 25},
        ),
        ACTION_DETECTIONS: ReviewedAction(
            ACTION_DETECTIONS,
            frozenset({
                "company_id_or_domain",
                "first_seen_at_from",
                "first_seen_at_until",
                "last_seen_at_from",
                "last_seen_at_until",
                "page",
                "limit",
            }),
            {"page": 1, "limit": 25},
        ),
        ACTION_TECHNOLOGY: ReviewedAction(
            ACTION_TECHNOLOGY,
            frozenset({"id_or_fuzzy_name"}),
            {},
        ),
        ACTION_JOB: ReviewedAction(
            ACTION_JOB,
            frozenset({"id"}),
            {},
        ),
        ACTION_EXA: ReviewedAction(
            ACTION_EXA,
            frozenset({"query", "numResults", "startPublishedDate", "endPublishedDate", "category"}),
            {"numResults": 5, "category": "news"},
        ),
    }
)

_ROUTE_ACTIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        ROUTE_CONNECTIONS: (ACTION_COMPANY, ACTION_CONNECTIONS, ACTION_COMPANY),
        ROUTE_NEWS: (ACTION_NEWS, ACTION_COMPANY, ACTION_EXA),
        ROUTE_TECHNOLOGY: (ACTION_DETECTIONS, ACTION_COMPANY, ACTION_TECHNOLOGY, ACTION_JOB),
    }
)
_ROUTE_MAX_CALLS: Mapping[str, int] = MappingProxyType(
    {ROUTE_CONNECTIONS: 3, ROUTE_NEWS: 3, ROUTE_TECHNOLOGY: 4}
)
_ROUTE_CREDIT_CEILINGS: Mapping[str, int] = MappingProxyType(
    {
        ROUTE_CONNECTIONS: PREDICTLEADS_CREDIT_MICROUNITS * 3,
        ROUTE_NEWS: PREDICTLEADS_CREDIT_MICROUNITS * 2 + EXA_CREDIT_MICROUNITS,
        ROUTE_TECHNOLOGY: PREDICTLEADS_CREDIT_MICROUNITS * 4,
    }
)
# Public read-only projections used by the provider binding adapter.  The
# adapter must reserve these values before it invokes this module.
ROUTE_ACTION_ORDER = _ROUTE_ACTIONS
ROUTE_MAX_CALLS = _ROUTE_MAX_CALLS
ROUTE_CREDIT_CEILINGS = _ROUTE_CREDIT_CEILINGS


def _json_safe(value: Any) -> Any:
    """Convert reviewed immutable contract values to canonical JSON data."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_json_safe(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


# These are the exact Deepline contracts read from the live catalog.  Keep the
# projection small and immutable: the signed binding catalog owns transport
# details, while this module owns the workflow's semantic boundary.
_ACTION_CONTRACTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        ACTION_COMPANY: MappingProxyType(
            {
                "input": {"required": ("id_or_domain",), "fields": ("id_or_domain",)},
                "output": {"resource_type": "company", "top_level": ("data", "included", "meta")},
            }
        ),
        ACTION_CONNECTIONS: MappingProxyType(
            {
                "input": {
                    "required": ("company_id_or_domain",),
                    "fields": (
                        "company_id_or_domain",
                        "first_seen_at_from",
                        "first_seen_at_until",
                        "categories",
                        "page",
                        "limit",
                    ),
                },
                "output": {
                    "resource_type": "connection",
                    "top_level": ("data", "included", "meta"),
                    "relationships": ("company1", "company2"),
                },
            }
        ),
        ACTION_NEWS: MappingProxyType(
            {
                "input": {
                    "required": ("company_id_or_domain",),
                    "fields": (
                        "company_id_or_domain",
                        "found_at_from",
                        "found_at_until",
                        "categories",
                        "page",
                        "limit",
                    ),
                },
                "output": {
                    "resource_type": "news_event",
                    "top_level": ("data", "included", "meta"),
                    "relationships": ("company1", "company2", "most_relevant_source"),
                },
            }
        ),
        ACTION_DETECTIONS: MappingProxyType(
            {
                "input": {
                    "required": ("company_id_or_domain",),
                    "fields": (
                        "company_id_or_domain",
                        "first_seen_at_from",
                        "first_seen_at_until",
                        "last_seen_at_from",
                        "last_seen_at_until",
                        "page",
                        "limit",
                    ),
                },
                "output": {
                    "resource_type": "technology_detection",
                    "top_level": ("data", "included", "meta"),
                    "relationships": ("company", "technology", "seen_on_job_openings"),
                },
            }
        ),
        ACTION_TECHNOLOGY: MappingProxyType(
            {
                "input": {"required": ("id_or_fuzzy_name",), "fields": ("id_or_fuzzy_name",)},
                "output": {"resource_type": "technology", "top_level": ("data", "included", "meta")},
            }
        ),
        ACTION_JOB: MappingProxyType(
            {
                "input": {"required": ("id",), "fields": ("id",)},
                "output": {"resource_type": "job_opening", "top_level": ("data", "included", "meta")},
            }
        ),
        ACTION_EXA: MappingProxyType(
            {
                "input": {
                    "required": ("query",),
                    "fields": ("query", "numResults", "startPublishedDate", "endPublishedDate", "category"),
                },
                "output": {"resource_type": "search_result", "top_level": ("results", "requestId", "context")},
            }
        ),
    }
)
ACTION_CONTRACT_HASHES: Mapping[str, str] = MappingProxyType(
    {action_id: _sha256_json(contract) for action_id, contract in _ACTION_CONTRACTS.items()}
)


@dataclass(frozen=True)
class WorkflowActionManifest:
    action_id: str
    input_output_contract_hash: str
    optional: bool
    credit_ceiling_microcredits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "input_output_contract_hash": self.input_output_contract_hash,
            "optional": self.optional,
            "credit_ceiling_microcredits": self.credit_ceiling_microcredits,
        }


@dataclass(frozen=True)
class PredictLeadsWorkflowManifest:
    workflow_id: str
    schema_version: str
    ordered_actions: tuple[WorkflowActionManifest, ...]
    max_calls: int
    timeout_ms: int
    credit_ceiling_microcredits: int
    branch_optional_actions: tuple[str, ...]
    manifest_hash: str

    def payload(self) -> dict[str, Any]:
        """Return the hash-covered, JSON-safe manifest payload."""

        return {
            "workflow_id": self.workflow_id,
            "schema_version": self.schema_version,
            "ordered_actions": [item.to_dict() for item in self.ordered_actions],
            "max_calls": self.max_calls,
            "timeout_ms": self.timeout_ms,
            "credit_ceiling_microcredits": self.credit_ceiling_microcredits,
            "branch_optional_actions": list(self.branch_optional_actions),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.payload()
        result["manifest_hash"] = self.manifest_hash
        return result


def _make_manifest(route: str, optional_actions: frozenset[str] = frozenset()) -> PredictLeadsWorkflowManifest:
    actions = tuple(
        WorkflowActionManifest(
            action_id=action_id,
            input_output_contract_hash=ACTION_CONTRACT_HASHES[action_id],
            optional=action_id in optional_actions,
            credit_ceiling_microcredits=EXA_CREDIT_MICROUNITS
            if action_id == ACTION_EXA
            else PREDICTLEADS_CREDIT_MICROUNITS,
        )
        for action_id in _ROUTE_ACTIONS[route]
    )
    payload = {
        "workflow_id": route,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "ordered_actions": [
            {
                "action_id": item.action_id,
                "input_output_contract_hash": item.input_output_contract_hash,
                "optional": item.optional,
                "credit_ceiling_microcredits": item.credit_ceiling_microcredits,
            }
            for item in actions
        ],
        "max_calls": _ROUTE_MAX_CALLS[route],
        "timeout_ms": ROUTE_TIMEOUT_MS,
        "credit_ceiling_microcredits": _ROUTE_CREDIT_CEILINGS[route],
        "branch_optional_actions": sorted(optional_actions),
    }
    return PredictLeadsWorkflowManifest(
        workflow_id=route,
        schema_version=WORKFLOW_SCHEMA_VERSION,
        ordered_actions=actions,
        max_calls=_ROUTE_MAX_CALLS[route],
        timeout_ms=ROUTE_TIMEOUT_MS,
        credit_ceiling_microcredits=_ROUTE_CREDIT_CEILINGS[route],
        branch_optional_actions=tuple(sorted(optional_actions)),
        manifest_hash=_sha256_json(payload),
    )


ROUTE_MANIFESTS: Mapping[str, PredictLeadsWorkflowManifest] = MappingProxyType(
    {
        ROUTE_CONNECTIONS: _make_manifest(ROUTE_CONNECTIONS),
        ROUTE_NEWS: _make_manifest(ROUTE_NEWS, frozenset({ACTION_EXA})),
        ROUTE_TECHNOLOGY: _make_manifest(ROUTE_TECHNOLOGY),
    }
)


def workflow_manifest(route: str) -> PredictLeadsWorkflowManifest:
    """Return the reviewed manifest for one model SourceAdd route."""

    try:
        return ROUTE_MANIFESTS[route]
    except KeyError as exc:
        raise PredictLeadsWorkflowError("workflow route is not reviewed") from exc


def export_workflow_manifests() -> dict[str, dict[str, Any]]:
    """Export deterministic JSON-ready manifests for signed catalog dispatch."""

    return {
        route: ROUTE_MANIFESTS[route].to_dict()
        for route in sorted(ROUTE_MANIFESTS)
    }


_WORKFLOW_INPUT_FIELDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        ROUTE_CONNECTIONS: frozenset({"company_domain", "minimum_date", "maximum_date"}),
        ROUTE_NEWS: frozenset(
            {"company_domain", "intent_category", "minimum_date", "maximum_date"}
        ),
        ROUTE_TECHNOLOGY: frozenset(
            {"company_domain", "technology", "minimum_date", "maximum_date"}
        ),
    }
)


def validate_workflow_input(route: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the signed, provider-independent input for one workflow.

    This function does not call a provider and does not accept a branch,
    endpoint, or request body.  The binding compiler uses it before a route
    can be handed to a runtime dispatcher.
    """

    if route not in _WORKFLOW_INPUT_FIELDS:
        raise PredictLeadsWorkflowError("workflow route is not reviewed")
    if not isinstance(values, Mapping) or set(values) != _WORKFLOW_INPUT_FIELDS[route]:
        raise PredictLeadsWorkflowError("workflow input fields are not reviewed")
    domain = _domain(values.get("company_domain"))
    minimum = _date(values.get("minimum_date"), "minimum_date")
    maximum = _date(values.get("maximum_date"), "maximum_date")
    if minimum > maximum:
        raise PredictLeadsWorkflowError("date range is inverted")
    result: dict[str, Any] = {
        "company_domain": domain,
        "minimum_date": minimum.isoformat(),
        "maximum_date": maximum.isoformat(),
    }
    if route == ROUTE_NEWS:
        category = str(values.get("intent_category") or "").strip().upper()
        if category not in _NEWS_CATEGORIES:
            raise PredictLeadsWorkflowError("news category is unsupported")
        result["intent_category"] = category
    elif route == ROUTE_TECHNOLOGY:
        technology = _technology_name(values.get("technology"))
        if not technology:
            raise PredictLeadsWorkflowError("technology is required")
        result["technology"] = technology
    return result

_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")
_PRIVATE_HOSTS = frozenset(
    {
        "deepline.com",
        "code.deepline.com",
        "predictleads.com",
        "api.predictleads.com",
        "exa.ai",
        "api.exa.ai",
    }
)
_CONNECTION_CATEGORIES = frozenset({"partner"})
_NEWS_CATEGORIES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "ACQUISITION": frozenset({"acquires", "merges_with", "sells_assets_to"}),
        "PARTNERSHIP": frozenset({"partners_with"}),
        "PRODUCT_LAUNCH": frozenset({"launches"}),
        "LEADERSHIP_CHANGE": frozenset({"hires", "promotes"}),
        "MARKET_EXPANSION": frozenset({"expands_offices_in", "expands_offices_to", "expands_facilities"}),
        "FACILITY_OPENING": frozenset({"opens_new_location", "expands_facilities"}),
    }
)
_NEWS_SEARCH_TERMS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "ACQUISITION": frozenset({"acquire", "acquired", "acquisition", "merger", "merged"}),
        "PARTNERSHIP": frozenset({"partner", "partners", "partnered", "partnership"}),
        "PRODUCT_LAUNCH": frozenset({"launch", "launched", "launches", "released", "release"}),
        "LEADERSHIP_CHANGE": frozenset({"hired", "hires", "promoted", "promotes", "appointed"}),
        "MARKET_EXPANSION": frozenset({"expands", "expanded", "expansion", "opened", "opens"}),
        "FACILITY_OPENING": frozenset({"opened", "opens", "opening", "facility"}),
    }
)
_FORBIDDEN_CATEGORY_WORDS = frozenset({"ends", "ending", "developing", "develops", "attends", "vendor", "integration", "investor", "parent"})


@dataclass(frozen=True)
class PredictLeadsWorkflowResult:
    """Redacted route result suitable for an append-only execution receipt."""

    route: str
    status: str
    reason_code: str
    calls: tuple[str, ...] = ()
    evidence: Mapping[str, Any] | None = None

    @property
    def qualified(self) -> bool:
        return self.status == "qualified" and self.evidence is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "route": self.route,
            "status": self.status,
            "reason_code": self.reason_code,
            "calls": list(self.calls),
            "evidence": dict(self.evidence) if self.evidence else None,
        }


Reserve = Callable[[str, int, int, int], Any]
Call = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _slug(value: Any) -> str:
    return _norm(value).replace("_", " ").replace("-", " ")


def _domain(value: Any) -> str:
    result = str(value or "").strip().casefold().rstrip(".")
    if not _DOMAIN_RE.fullmatch(result):
        raise PredictLeadsWorkflowError("requested company domain is invalid")
    return result


def _date(value: Any, field: str) -> date:
    text = str(value or "")[:10]
    if not _DATE_RE.fullmatch(text):
        raise PredictLeadsWorkflowError(f"{field} is invalid")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise PredictLeadsWorkflowError(f"{field} is invalid") from exc


def _date_in_range(value: Any, minimum: date, maximum: date) -> bool:
    parsed = str(value or "")[:10]
    if not _DATE_RE.fullmatch(parsed):
        return False
    try:
        actual = date.fromisoformat(parsed)
    except ValueError:
        return False
    return minimum <= actual <= maximum


def _date_on_or_after(value: Any, lower_bound: date, upper_bound: date) -> bool:
    parsed = str(value or "")[:10]
    if not _DATE_RE.fullmatch(parsed):
        return False
    try:
        actual = date.fromisoformat(parsed)
    except ValueError:
        return False
    return lower_bound <= actual <= upper_bound


def _text(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_text(item) for item in value if isinstance(item, (str, int, float)))
    return ""


def _attrs(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("attributes")
    return value if isinstance(value, Mapping) else row


def _rows(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for path in (
        ("result", "data", "data"),
        ("result", "data"),
        ("data", "data"),
        ("data",),
        ("results",),
    ):
        node: Any = response
        for part in path:
            if not isinstance(node, Mapping) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, list):
            return [row for row in node if isinstance(row, Mapping)]
        if isinstance(node, Mapping) and node.get("type"):
            return [node]
    return []


def _included(response: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    for path in (("result", "data", "included"), ("result", "included"), ("included",)):
        node: Any = response
        for part in path:
            if not isinstance(node, Mapping) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, list):
            return {
                (str(item.get("type") or ""), str(item.get("id") or "")): item
                for item in node
                if isinstance(item, Mapping) and item.get("type") and item.get("id")
            }
    return {}


def _relationship(row: Mapping[str, Any], name: str) -> tuple[str, str] | None:
    relationships = row.get("relationships")
    value = relationships.get(name) if isinstance(relationships, Mapping) else None
    data = value.get("data") if isinstance(value, Mapping) else None
    if isinstance(data, Mapping) and data.get("id") and data.get("type"):
        return str(data["type"]), str(data["id"])
    return None


def _public_url(*values: Any) -> str | None:
    candidates: list[Any] = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            candidates.extend(value)
        else:
            candidates.append(value)
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            parsed = urlsplit(value.strip())
        except ValueError:
            continue
        host = (parsed.hostname or "").casefold().rstrip(".")
        if (
            parsed.scheme != "https"
            or not host
            or host in _PRIVATE_HOSTS
            or parsed.username
            or parsed.password
            or not parsed.netloc
            or not parsed.path
        ):
            continue
        return value.strip()
    return None


def _source_url(attrs: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> str | None:
    row = row or {}
    return _public_url(
        attrs.get("source_url"),
        attrs.get("source_urls"),
        attrs.get("url"),
        row.get("source_url"),
        row.get("source_urls"),
        row.get("url"),
    )


def _relationship_ids(row: Mapping[str, Any], name: str) -> tuple[tuple[str, str], ...]:
    """Return one or more JSON:API relationship identifiers."""

    relationships = row.get("relationships")
    value = relationships.get(name) if isinstance(relationships, Mapping) else None
    data = value.get("data") if isinstance(value, Mapping) else None
    if isinstance(data, Mapping) and data.get("id") and data.get("type"):
        return ((str(data["type"]), str(data["id"])),)
    if isinstance(data, list):
        return tuple(
            (str(item["type"]), str(item["id"]))
            for item in data
            if isinstance(item, Mapping) and item.get("id") and item.get("type")
        )
    return ()


def _record_id(row: Mapping[str, Any]) -> str | None:
    value = str(row.get("id") or "").strip()
    return value if _SAFE_ID_RE.fullmatch(value) else None


def _company_from_response(response: Mapping[str, Any], expected_domain: str, expected_id: str | None = None) -> Mapping[str, Any] | None:
    for row in _rows(response):
        if str(row.get("type") or "company") != "company":
            continue
        attrs = _attrs(row)
        domain = str(attrs.get("domain") or attrs.get("company_domain") or "").casefold().rstrip(".")
        row_id = str(row.get("id") or "")
        if domain == expected_domain and (expected_id is None or row_id == expected_id):
            return row
    return None


def _company_by_id(response: Mapping[str, Any], expected_id: str) -> Mapping[str, Any] | None:
    """Resolve one exact company resource when a response has no ``included``."""

    for row in _rows(response):
        if str(row.get("type") or "company") != "company":
            continue
        if str(row.get("id") or "") != expected_id:
            continue
        attrs = _attrs(row)
        domain = str(attrs.get("domain") or attrs.get("company_domain") or "").casefold().rstrip(".")
        name = _text(attrs.get("company_name") or attrs.get("friendly_company_name") or attrs.get("name"))
        if _DOMAIN_RE.fullmatch(domain) and name:
            return row
    return None


def _reserve(route: str, reserve: Reserve) -> bool:
    if not callable(reserve):
        raise PredictLeadsWorkflowError("route reservation callback is required")
    try:
        result = reserve(
            route,
            _ROUTE_MAX_CALLS[route],
            _ROUTE_CREDIT_CEILINGS[route],
            ROUTE_TIMEOUT_MS,
        )
    except Exception:
        return False
    return result is not False


class _Runner:
    def __init__(self, route: str, reserve: Reserve, call: Call) -> None:
        if route not in _ROUTE_ACTIONS:
            raise PredictLeadsWorkflowError("route is not reviewed")
        if not callable(call):
            raise PredictLeadsWorkflowError("provider callback is required")
        self.route = route
        self.call = call
        self.calls: list[str] = []
        self.failed = False
        self.deadline = time.monotonic() + ROUTE_TIMEOUT_MS / 1000.0
        self.reserved = _reserve(route, reserve)

    def invoke(self, action_id: str, values: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self.failed or not self.reserved:
            return None
        if len(self.calls) >= _ROUTE_MAX_CALLS[self.route]:
            self.failed = True
            return None
        allowed = _ROUTE_ACTIONS[self.route]
        if action_id not in allowed:
            self.failed = True
            return None
        # One action can occur more than once in a route only where the model
        # requires it (connections resolves the requested and counterparty
        # companies).  No retry loop exists here.
        try:
            payload = REVIEWED_ACTIONS[action_id].payload(values)
        except Exception:
            self.failed = True
            return None
        remaining_ms = int((self.deadline - time.monotonic()) * 1000)
        if remaining_ms < 1:
            self.failed = True
            return None
        self.calls.append(action_id)
        try:
            response = self.call(action_id, payload, min(ROUTE_TIMEOUT_MS, remaining_ms))
        except Exception:
            self.failed = True
            return None
        if not isinstance(response, Mapping):
            self.failed = True
            return None
        return response


def _miss(route: str, runner: _Runner, reason: str) -> PredictLeadsWorkflowResult:
    return PredictLeadsWorkflowResult(route, "miss", reason, tuple(runner.calls))


def _blocked(route: str) -> PredictLeadsWorkflowResult:
    return PredictLeadsWorkflowResult(route, "blocked", "route_reservation_rejected")


def _qualified(route: str, runner: _Runner, evidence: Mapping[str, Any]) -> PredictLeadsWorkflowResult:
    return PredictLeadsWorkflowResult(route, "qualified", "verified_evidence", tuple(runner.calls), dict(evidence))


def run_predictleads_connections(
    *,
    company_domain: str,
    minimum_date: str,
    maximum_date: str,
    reserve: Reserve,
    call: Call,
) -> PredictLeadsWorkflowResult:
    """Run company → connections → counterparty company, at most three calls."""

    try:
        domain = _domain(company_domain)
        minimum = _date(minimum_date, "minimum_date")
        maximum = _date(maximum_date, "maximum_date")
        if minimum > maximum:
            raise PredictLeadsWorkflowError("date range is inverted")
    except PredictLeadsWorkflowError:
        return PredictLeadsWorkflowResult(ROUTE_CONNECTIONS, "miss", "invalid_request")
    runner = _Runner(ROUTE_CONNECTIONS, reserve, call)
    if not runner.reserved:
        return _blocked(ROUTE_CONNECTIONS)
    company_response = runner.invoke(ACTION_COMPANY, {"id_or_domain": domain})
    company = _company_from_response(company_response or {}, domain)
    company_id = str(company.get("id") or "") if company else ""
    if not company_id:
        return _miss(ROUTE_CONNECTIONS, runner, "company_identity_unverified")
    connection_response = runner.invoke(
        ACTION_CONNECTIONS,
        {
            "company_id_or_domain": company_id,
            "first_seen_at_from": minimum.isoformat(),
            "first_seen_at_until": maximum.isoformat(),
            "categories": ["partner"],
        },
    )
    if connection_response is None:
        return _miss(ROUTE_CONNECTIONS, runner, "provider_call_failed")
    selected: tuple[Mapping[str, Any], str, str] | None = None
    for row in _rows(connection_response):
        attrs = _attrs(row)
        category = _slug(attrs.get("category") or attrs.get("connection_type") or attrs.get("type"))
        if category not in _CONNECTION_CATEGORIES or any(
            _slug(word) in category for word in _FORBIDDEN_CATEGORY_WORDS
        ):
            continue
        company1 = _relationship(row, "company1")
        company2 = _relationship(row, "company2")
        if company1 is None or company2 is None:
            continue
        if company1[0] != "company" or company2[0] != "company":
            continue
        matching_sides = [
            relation
            for relation in (company1, company2)
            if relation[1] == company_id
        ]
        if len(matching_sides) != 1:
            continue
        relation = company2 if matching_sides[0] == company1 else company1
        if relation[1] == company_id:
            continue
        first_seen = attrs.get("first_seen_at") or attrs.get("first_seen")
        last_seen = attrs.get("last_seen_at") or attrs.get("last_seen")
        if not _date_in_range(first_seen, minimum, maximum) or not _date_on_or_after(
            last_seen, date(2024, 2, 1), maximum
        ):
            continue
        if str(last_seen)[:10] < "2024-02-01" or str(last_seen)[:10] < str(first_seen)[:10]:
            continue
        source = _source_url(attrs, row)
        if not source:
            continue
        counterparty_id = str(relation[1] or "")
        if not _SAFE_ID_RE.fullmatch(counterparty_id):
            continue
        selected = (row, counterparty_id, source)
        break
    if selected is None:
        return _miss(ROUTE_CONNECTIONS, runner, "no_verified_current_partner")
    row, counterparty_id, source = selected
    counterparty_response = runner.invoke(ACTION_COMPANY, {"id_or_domain": counterparty_id})
    resolved = _company_by_id(counterparty_response or {}, counterparty_id)
    if resolved is None:
        return _miss(ROUTE_CONNECTIONS, runner, "counterparty_identity_unverified")
    resolved_attrs = _attrs(resolved)
    counterparty_domain = str(resolved_attrs.get("domain") or resolved_attrs.get("company_domain") or "").casefold().rstrip(".")
    if counterparty_domain == domain:
        return _miss(ROUTE_CONNECTIONS, runner, "counterparty_identity_unverified")
    attrs = _attrs(row)
    evidence = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "source_tool": ROUTE_CONNECTIONS,
        "evidence_type": "partner_connection",
        "intent_category": "PARTNERSHIP",
        "company_domain": domain,
        "counterparty_company_id": counterparty_id,
        "counterparty_domain": counterparty_domain,
        "first_seen_at": str(attrs.get("first_seen_at") or attrs.get("first_seen")),
        "last_seen_at": str(attrs.get("last_seen_at") or attrs.get("last_seen")),
        "source_url": source,
        "provider_record_id": _record_id(row),
    }
    if not evidence["provider_record_id"]:
        return _miss(ROUTE_CONNECTIONS, runner, "provider_record_id_missing")
    return _qualified(ROUTE_CONNECTIONS, runner, evidence)


def _news_named_entity(
    attrs: Mapping[str, Any],
    category: str,
    row: Mapping[str, Any],
    included: Mapping[tuple[str, str], Mapping[str, Any]],
    expected_domain: str,
) -> tuple[str, str] | None:
    keys = {
        "ACQUISITION": ("counterparty", "acquired_company", "target"),
        "PARTNERSHIP": ("counterparty", "partner", "partner_company"),
        "PRODUCT_LAUNCH": ("product", "product_name", "launched_product"),
        "LEADERSHIP_CHANGE": ("executive", "person", "named_executive"),
        "MARKET_EXPANSION": ("location", "market", "city"),
        "FACILITY_OPENING": ("location", "facility", "city"),
    }[category]
    for key in keys:
        value = attrs.get(key)
        if isinstance(value, Mapping):
            value = value.get("name") or value.get("label")
        text = _text(value)
        if text:
            return key, text
    # PredictLeads represents the other company as company2 rather than as
    # an ad-hoc attribute.  Resolve the included lite company when present.
    for relation_name in ("company2", "company1"):
        relation = _relationship(row, relation_name)
        related = included.get(relation) if relation else None
        if related is None:
            continue
        related_attrs = _attrs(related)
        related_domain = str(
            related_attrs.get("domain") or related_attrs.get("company_domain") or ""
        ).casefold().rstrip(".")
        # Do not turn the requested company itself into the named event
        # entity when the provider omitted the category-specific attribute.
        if related_domain == expected_domain:
            continue
        text = _text(
            related_attrs.get("company_name")
            or related_attrs.get("friendly_company_name")
            or related_attrs.get("name")
        )
        if text:
            return relation_name, text
    entities = attrs.get("entities")
    if isinstance(entities, Mapping):
        for key in keys:
            text = _text(entities.get(key))
            if text:
                return key, text
    return None


def _news_category(attrs: Mapping[str, Any], requested: str) -> bool:
    raw = _slug(attrs.get("category") or attrs.get("event_type") or attrs.get("type"))
    return raw in {_slug(item) for item in _NEWS_CATEGORIES.get(requested, ())} and not any(
        word in raw.split() for word in _FORBIDDEN_CATEGORY_WORDS
    )


def _event_match(
    row: Mapping[str, Any],
    requested: str,
    domain: str,
    minimum: date,
    maximum: date,
    included: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[Mapping[str, Any], tuple[str, str], str, str] | None:
    attrs = _attrs(row)
    if not _news_category(attrs, requested):
        return None
    if not (attrs.get("planning") is False or attrs.get("is_planned") is False or str(attrs.get("status") or "").casefold() in {"completed", "complete", "published", "active"}):
        return None
    relations = tuple(
        relation
        for name in ("company1", "company2")
        for relation in _relationship_ids(row, name)
        if relation[0] == "company"
    )
    if not relations:
        return None
    found_at = attrs.get("found_at")
    if not _date_in_range(found_at, minimum, maximum):
        return None
    event_date = attrs.get("effective_date") or found_at
    if event_date is None or not _date_in_range(event_date, minimum, maximum):
        return None
    named = _news_named_entity(attrs, requested, row, included, domain)
    if named is None:
        return None
    subject_relations = [
        relation
        for relation in relations
        if str(
            _attrs(included.get(relation, {})).get("domain")
            or _attrs(included.get(relation, {})).get("company_domain")
            or ""
        ).casefold().rstrip(".") == domain
    ]
    if len(subject_relations) > 1:
        return None
    return attrs, subject_relations[0] if subject_relations else relations[0], str(event_date), named[1]


def _exa_match(response: Mapping[str, Any], *, company_name: str, domain: str, category: str, entity: str, minimum: date, maximum: date) -> tuple[Mapping[str, Any], str] | None:
    for row in _rows(response):
        url = _public_url(row.get("url"), row.get("source_url"))
        if not url:
            continue
        published = row.get("publishedDate") or row.get("published_at") or row.get("date")
        if not _date_in_range(published, minimum, maximum):
            continue
        text = _norm(" ".join(_text(row.get(key)) for key in ("title", "text", "highlights", "summary")))
        if _norm(company_name) not in text or _norm(entity) not in text:
            continue
        category_terms = _NEWS_SEARCH_TERMS.get(category, ())
        if not any(_norm(term) in text for term in category_terms):
            continue
        return row, url
    return None


def run_predictleads_news(
    *,
    company_domain: str,
    intent_category: str,
    minimum_date: str,
    maximum_date: str,
    reserve: Reserve,
    call: Call,
) -> PredictLeadsWorkflowResult:
    """Run events → company → optional Exa source resolution, at most three calls."""

    try:
        domain = _domain(company_domain)
        category = str(intent_category or "").strip().upper()
        if category not in _NEWS_CATEGORIES:
            raise PredictLeadsWorkflowError("news category is unsupported")
        minimum = _date(minimum_date, "minimum_date")
        maximum = _date(maximum_date, "maximum_date")
        if minimum > maximum:
            raise PredictLeadsWorkflowError("date range is inverted")
    except PredictLeadsWorkflowError:
        return PredictLeadsWorkflowResult(ROUTE_NEWS, "miss", "invalid_request")
    runner = _Runner(ROUTE_NEWS, reserve, call)
    if not runner.reserved:
        return _blocked(ROUTE_NEWS)
    event_response = runner.invoke(
        ACTION_NEWS,
        {
            "company_id_or_domain": domain,
            "found_at_from": minimum.isoformat(),
            "found_at_until": maximum.isoformat(),
            "categories": sorted(_NEWS_CATEGORIES[category]),
        },
    )
    if event_response is None:
        return _miss(ROUTE_NEWS, runner, "provider_call_failed")
    selected: tuple[Mapping[str, Any], Mapping[str, Any], str, str] | None = None
    included = _included(event_response)
    for row in _rows(event_response):
        match = _event_match(row, category, domain, minimum, maximum, included)
        if match is None:
            continue
        attrs, relation, event_date, named_entity = match
        company = included.get(relation)
        source = _source_url(attrs, row)
        if not source:
            article_relation = _relationship(row, "most_relevant_source")
            article = included.get(article_relation) if article_relation else None
            if article is not None:
                source = _source_url(_attrs(article), article)
        if company is None:
            # The company endpoint below is required to establish the exact
            # relation.  It must still be called even when the event has URL.
            company = {"type": relation[0], "id": relation[1], "attributes": {}}
        selected = (row, company, source or "", named_entity)
        break
    if selected is None:
        return _miss(ROUTE_NEWS, runner, "no_verified_news_event")
    row, company_hint, included_source, named_entity = selected
    company_id = str(company_hint.get("id") or "")
    if not _SAFE_ID_RE.fullmatch(company_id):
        return _miss(ROUTE_NEWS, runner, "company_relationship_invalid")
    company_response = runner.invoke(ACTION_COMPANY, {"id_or_domain": company_id})
    company = _company_from_response(company_response or {}, domain, company_id)
    if company is None:
        return _miss(ROUTE_NEWS, runner, "company_identity_unverified")
    company_name = _text(
        _attrs(company).get("company_name")
        or _attrs(company).get("friendly_company_name")
        or _attrs(company).get("name")
    )
    attrs = _attrs(row)
    source = included_source
    exa_row: Mapping[str, Any] | None = None
    if not source:
        query = f'"{company_name or domain}" "{named_entity}" {category.replace("_", " ")}'
        exa_response = runner.invoke(
            ACTION_EXA,
            {
                "query": query,
                "startPublishedDate": str(minimum),
                "endPublishedDate": str(maximum),
            },
        )
        match = _exa_match(
            exa_response or {},
            company_name=company_name or domain,
            domain=domain,
            category=category,
            entity=named_entity,
            minimum=minimum,
            maximum=maximum,
        )
        if match is None:
            return _miss(ROUTE_NEWS, runner, "original_source_unresolved")
        exa_row, source = match
    evidence = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "source_tool": ROUTE_NEWS,
        "evidence_type": "news_event",
        "intent_category": category,
        "company_domain": domain,
        "event_date": str(attrs.get("effective_date") or attrs.get("announced_on") or attrs.get("published_at") or attrs.get("date")),
        "named_entity": named_entity,
        "source_url": source,
        "provider_record_id": _record_id(row),
    }
    if exa_row is not None:
        evidence["source_resolution"] = "exa_fallback"
    if not evidence["provider_record_id"]:
        return _miss(ROUTE_NEWS, runner, "provider_record_id_missing")
    return _qualified(ROUTE_NEWS, runner, evidence)


def _technology_name(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("vendor_name") or value.get("technology_name")
    return _norm(value)


def run_predictleads_technology(
    *,
    company_domain: str,
    technology: str,
    minimum_date: str,
    maximum_date: str,
    reserve: Reserve,
    call: Call,
) -> PredictLeadsWorkflowResult:
    """Run detections → company → technology → job, at most four calls."""

    try:
        domain = _domain(company_domain)
        requested_technology = _technology_name(technology)
        if not requested_technology:
            raise PredictLeadsWorkflowError("technology is required")
        minimum = _date(minimum_date, "minimum_date")
        maximum = _date(maximum_date, "maximum_date")
        if minimum > maximum:
            raise PredictLeadsWorkflowError("date range is inverted")
    except PredictLeadsWorkflowError:
        return PredictLeadsWorkflowResult(ROUTE_TECHNOLOGY, "miss", "invalid_request")
    runner = _Runner(ROUTE_TECHNOLOGY, reserve, call)
    if not runner.reserved:
        return _blocked(ROUTE_TECHNOLOGY)
    detection_response = runner.invoke(
        ACTION_DETECTIONS,
        {
            "company_id_or_domain": domain,
            "first_seen_at_from": minimum.isoformat(),
            "first_seen_at_until": maximum.isoformat(),
            "last_seen_at_from": minimum.isoformat(),
            "last_seen_at_until": maximum.isoformat(),
        },
    )
    if detection_response is None:
        return _miss(ROUTE_TECHNOLOGY, runner, "provider_call_failed")
    selected: tuple[Mapping[str, Any], str, str, str] | None = None
    for row in _rows(detection_response):
        attrs = _attrs(row)
        relation = _relationship(row, "company")
        tech_relation = _relationship(row, "technology")
        job_relations = _relationship_ids(row, "seen_on_job_openings")
        if not relation or relation[0] != "company" or not tech_relation or not job_relations:
            continue
        # PredictLeads may omit ``included`` from the detection response.
        # Relationship IDs are authoritative inputs for the required detail
        # calls below.  If optional included rows are present, use them only
        # as an early consistency check; never require them for admission.
        included = _included(detection_response)
        company_hint = included.get(relation)
        if company_hint is not None and str(
            _attrs(company_hint).get("domain") or ""
        ).casefold().rstrip(".") != domain:
            continue
        tech_hint = included.get(tech_relation)
        if tech_hint is not None and _technology_name(
            _attrs(tech_hint).get("name") or _attrs(tech_hint).get("vendor_name")
        ) != requested_technology:
            continue
        source_count = attrs.get("source_count") or attrs.get("sources_count")
        try:
            if int(source_count) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        if _slug(attrs.get("source_type")) not in {"job opening", "job openings"}:
            continue
        last_seen = attrs.get("last_seen_at")
        if not _date_in_range(last_seen, minimum, maximum):
            continue
        selected = (row, str(relation[1]), str(tech_relation[1]), str(job_relations[0][1]))
        break
    if selected is None:
        return _miss(ROUTE_TECHNOLOGY, runner, "no_verified_technology_detection")
    row, company_id, technology_id, job_id = selected
    company_response = runner.invoke(ACTION_COMPANY, {"id_or_domain": company_id})
    company = _company_from_response(company_response or {}, domain, company_id)
    if company is None:
        return _miss(ROUTE_TECHNOLOGY, runner, "company_identity_unverified")
    technology_response = runner.invoke(ACTION_TECHNOLOGY, {"id_or_fuzzy_name": technology_id})
    technology_rows = _rows(technology_response or {})
    technology_row = next(
        (item for item in technology_rows if str(item.get("id") or "") == technology_id),
        None,
    )
    if technology_row is None or _technology_name(_attrs(technology_row).get("name") or _attrs(technology_row).get("vendor_name")) != requested_technology:
        return _miss(ROUTE_TECHNOLOGY, runner, "technology_identity_unverified")
    job_response = runner.invoke(ACTION_JOB, {"id": job_id})
    job_rows = _rows(job_response or {})
    job = next((item for item in job_rows if str(item.get("id") or "") == job_id), None)
    if job is None:
        return _miss(ROUTE_TECHNOLOGY, runner, "job_identity_unverified")
    job_attrs = _attrs(job)
    job_company_relation = _relationship(job, "company")
    if not job_company_relation or job_company_relation != ("company", company_id):
        return _miss(ROUTE_TECHNOLOGY, runner, "job_company_relationship_mismatch")
    status = job_attrs.get("status")
    if str(job.get("type") or "job_opening") != "job_opening" or status is not None:
        return _miss(ROUTE_TECHNOLOGY, runner, "job_not_active")
    job_text = _norm(" ".join(_text(job_attrs.get(key)) for key in ("title", "normalized_title", "description")))
    if requested_technology not in job_text:
        return _miss(ROUTE_TECHNOLOGY, runner, "job_technology_mismatch")
    posted = job_attrs.get("posted_at") or job_attrs.get("first_seen_at")
    if not _date_in_range(posted, minimum, maximum):
        return _miss(ROUTE_TECHNOLOGY, runner, "job_date_out_of_range")
    source = _source_url(job_attrs, job)
    if not source:
        return _miss(ROUTE_TECHNOLOGY, runner, "job_source_missing")
    detection_attrs = _attrs(row)
    evidence = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "source_tool": ROUTE_TECHNOLOGY,
        "evidence_type": "technology_detection",
        "intent_category": "TECHSTACK",
        "company_domain": domain,
        "technology": technology,
        "last_seen_at": str(detection_attrs.get("last_seen_at")),
        "job_opening_id": job_id,
        "source_url": source,
        "provider_record_id": _record_id(row),
    }
    if not evidence["provider_record_id"]:
        return _miss(ROUTE_TECHNOLOGY, runner, "provider_record_id_missing")
    return _qualified(ROUTE_TECHNOLOGY, runner, evidence)


__all__ = [
    "ACTION_COMPANY",
    "ACTION_CONNECTIONS",
    "ACTION_DETECTIONS",
    "ACTION_EXA",
    "ACTION_JOB",
    "ACTION_NEWS",
    "ACTION_TECHNOLOGY",
    "PredictLeadsWorkflowError",
    "PredictLeadsWorkflowResult",
    "PredictLeadsWorkflowManifest",
    "REVIEWED_ACTIONS",
    "ACTION_CONTRACT_HASHES",
    "ROUTE_MANIFESTS",
    "ROUTE_ACTION_ORDER",
    "ROUTE_CREDIT_CEILINGS",
    "ROUTE_CONNECTIONS",
    "ROUTE_MAX_CALLS",
    "ROUTE_NEWS",
    "ROUTE_TECHNOLOGY",
    "export_workflow_manifests",
    "validate_workflow_input",
    "workflow_manifest",
    "run_predictleads_connections",
    "run_predictleads_news",
    "run_predictleads_technology",
]
