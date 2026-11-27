"""The tool market's catalog: which vendor operations a buyer may call.

Each operation names one vendor endpoint, the parameters a buyer may send,
how its usage is counted, and the most it may count for one call. The vault
enforces the same host and route shape again (``lead_order_vault.tools.TOOLS``);
this table is narrower, so a buyer reaches only the listed operations.

Usage units:

- ``call``: the vendor charges credits per request. A miner asks a price in
  micro-USD per credit. A 2xx answer counts ``call_units`` credits, or what
  the operation's ``cost`` reader finds the vendor charged (no email found
  is often free).
- ``vendor_usd``: the vendor reports the request's own cost in its answer.
  A miner asks a price in micro-USD per vendor USD (``900000`` sells at 90%
  of the vendor's price), and the call counts the reported micro-USD.

Adding a tool is data, not code: one ``Operation`` per endpoint here, plus
the tool's key placement and route rule in ``lead_order_vault.tools.TOOLS``,
which the attested vault enforces on its own.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode

UNIT_CALL = "call"
UNIT_VENDOR_USD = "vendor_usd"
UNIT_SCALE = {UNIT_CALL: 1, UNIT_VENDOR_USD: 1_000_000}


class CatalogError(ValueError):
    """A buyer request the catalog does not allow; ``code`` is safe to return."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Param:
    kind: str  # "str", "int", "bool", "str_list" or "enum"
    required: bool = False
    max_length: int = 500
    low: int = 1
    high: int = 1
    choices: Tuple[str, ...] = ()

    def read(self, value: Any) -> Any:
        if self.kind == "str":
            if not isinstance(value, str) or not 1 <= len(value) <= self.max_length:
                raise CatalogError("param_invalid")
            return value
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int) or not self.low <= value <= self.high:
                raise CatalogError("param_invalid")
            return value
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise CatalogError("param_invalid")
            return value
        if self.kind == "enum":
            if value not in self.choices:
                raise CatalogError("param_invalid")
            return value
        if self.kind == "str_list":
            if not isinstance(value, list) or not self.low <= len(value) <= self.high:
                raise CatalogError("param_invalid")
            for item in value:
                if not isinstance(item, str) or not 1 <= len(item) <= self.max_length:
                    raise CatalogError("param_invalid")
            return list(value)
        raise AssertionError(self.kind)


def _usd_field(*path: Any) -> Callable[[Any], Optional[int]]:
    """Read a vendor-reported USD cost and round it up to micro-USD."""

    def read(document: Any) -> Optional[int]:
        value = document
        for step in path:
            if isinstance(step, int):
                value = value[step] if isinstance(value, list) and len(value) > step else None
            else:
                value = value.get(step) if isinstance(value, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return None
        try:
            usd = Decimal(str(value))
        except InvalidOperation:
            return None
        if not usd.is_finite() or usd < 0:
            return None
        return int(math.ceil(usd * 1_000_000))

    return read


@dataclass(frozen=True)
class Operation:
    tool: str
    name: str
    method: str
    url: str
    unit: str
    # Upper bound on the units one call may count; the reserve is priced on it.
    max_units: int
    params: Mapping[str, Param]
    # Units a 2xx answer counts: ``cost(document)`` when set, else ``call_units``.
    call_units: int = 1
    cost: Optional[Callable[[Any], Optional[int]]] = field(default=None, compare=False)
    # How the checked params become the vendor's JSON body (default: as is).
    wrap: Optional[Callable[[Dict[str, Any]], Any]] = field(default=None, compare=False)
    # A vendor whose status codes say something else, such as an error in a
    # 200 or "out of credits" as 429: the status its answer means.
    real_status: Optional[Callable[[int, Any], int]] = field(default=None, compare=False)

    @property
    def unit_scale(self) -> int:
        return UNIT_SCALE[self.unit]

    def build(self, params: Any) -> Tuple[str, Dict[str, str], bytes]:
        """``(url, headers, body)`` for the vendor, or ``CatalogError``."""

        if not isinstance(params, Mapping) or not set(params) <= set(self.params):
            raise CatalogError("param_unknown")
        for name, param in self.params.items():
            if param.required and name not in params:
                raise CatalogError("param_missing")
        values = {name: self.params[name].read(value) for name, value in params.items()}
        headers = {"accept": "application/json"}
        if self.method == "GET":
            return self.url + ("?" + urlencode(sorted(values.items())) if values else ""), headers, b""
        headers["content-type"] = "application/json"
        document = self.wrap(values) if self.wrap is not None else values
        return self.url, headers, json.dumps(document, separators=(",", ":")).encode("utf-8")

    def status(self, http_status: int, document: Any) -> int:
        """The answer's real status, in the meaning the market acts on."""

        return http_status if self.real_status is None else self.real_status(http_status, document)

    def units(self, status: int, document: Any) -> Optional[int]:
        """Units an answer counts, given ``status()``: 0 for a refusal, None when unknown."""

        if 400 <= status < 500:
            return 0
        if not 200 <= status < 300:
            return None
        if self.cost is None:
            return self.call_units
        reported = self.cost(document)
        return None if reported is None else min(reported, self.max_units)


def _get(document: Any, *path: str) -> Any:
    for step in path:
        document = document.get(step) if isinstance(document, dict) else None
    return document


def _zerobounce_status(status: int, document: Any) -> int:
    """ZeroBounce answers 200 with an ``error`` for a dead key or an empty
    account, and for a bad request too; only the first fails the key."""

    error = _get(document, "error") if status == 200 and _get(document, "status") is None else None
    if not isinstance(error, str) or not error:
        return status
    return 402 if re.search(r"api key|credits", error, re.IGNORECASE) else 400


def _zerobounce_cost(document: Any) -> Optional[int]:
    result = _get(document, "status")
    return None if not isinstance(result, str) else 0 if result == "unknown" else 1


def _prospeo_status(status: int, document: Any) -> int:
    return {"INSUFFICIENT_CREDITS": 402, "INVALID_API_KEY": 401}.get(_get(document, "error_code"), status)


def _prospeo_cost(document: Any) -> Optional[int]:
    if _get(document, "error") is not False:
        return None
    return 0 if _get(document, "free_enrichment") is True else 1


def _found_email(*path: str) -> Callable[[Any], Optional[int]]:
    """One credit when the answer holds an email at ``path``; finding none is free."""

    return lambda document: 1 if _get(document, *path) else 0


def _prospeo_body(values: Dict[str, Any]) -> Dict[str, Any]:
    person = dict(values)
    flag = person.pop("only_verified_email", None)
    return {"data": person} if flag is None else {"data": person, "only_verified_email": flag}


_PROSPEO_FIELDS = ("first_name", "last_name", "full_name", "linkedin_url", "email", "company_name",
                   "company_website", "company_linkedin_url", "person_id")
_NAME = Param("str", max_length=200)
_URL = Param("str", required=True, max_length=300)
_ZB = "https://api.zerobounce.net"
_HARVEST = "https://api.harvestapi.io"
_EXA = "https://api.exa.ai"

_OPERATIONS: Sequence[Operation] = (
    # ZeroBounce: one credit per check; an ``unknown`` result is free.
    Operation("zerobounce", "validate", "GET", _ZB + "/v2/validate", UNIT_CALL, 1,
              {"email": Param("str", required=True, max_length=320), "ip_address": Param("str", max_length=45)},
              cost=_zerobounce_cost, real_status=_zerobounce_status),
    # Prospeo: one credit per match; no match, or a free re-enrichment, is free.
    Operation("prospeo", "enrich_person", "POST", "https://api.prospeo.io/enrich-person", UNIT_CALL, 1,
              dict({name: _NAME for name in _PROSPEO_FIELDS}, only_verified_email=Param("bool")),
              cost=_prospeo_cost, real_status=_prospeo_status,
              wrap=_prospeo_body),
    # Findymail: one credit per email found. 423 is a paused account.
    Operation("findymail", "find_from_linkedin", "POST", "https://app.findymail.com/api/search/business-profile",
              UNIT_CALL, 1, {"linkedin_url": _URL}, cost=_found_email("contact", "email"),
              real_status=lambda status, _: 402 if status == 423 else status),
    Operation("findymail", "find_from_name", "POST", "https://app.findymail.com/api/search/name", UNIT_CALL, 1,
              {"name": Param("str", required=True, max_length=200), "domain": Param("str", required=True, max_length=253)},
              cost=_found_email("contact", "email"), real_status=lambda status, _: 402 if status == 423 else status),
    # Hunter: one search per email found. Its 403 is a rate limit and its
    # 429 an empty account.
    Operation("hunter", "email_finder", "GET", "https://api.hunter.io/v2/email-finder", UNIT_CALL, 1,
              {"domain": Param("str", max_length=253), "company": _NAME, "linkedin_handle": _NAME,
               "first_name": _NAME, "last_name": _NAME, "full_name": _NAME,
               "max_duration": Param("int", low=3, high=20)},
              cost=_found_email("data", "email"), real_status=lambda status, _: {403: 429, 429: 402}.get(status, status)),
    # Exa: each answer carries ``costDollars.total``, which Exa calls an estimate.
    Operation("exa", "search", "POST", _EXA + "/search", UNIT_VENDOR_USD, 50_000,
              {"query": Param("str", required=True, max_length=2000), "numResults": Param("int", high=25),
               "type": Param("enum", choices=("auto", "neural", "keyword", "fast")),
               "category": Param("enum", choices=("company", "people", "news", "research paper", "pdf",
                                                  "github", "tweet", "personal site", "linkedin profile",
                                                  "financial report")),
               "includeDomains": Param("str_list", max_length=200, high=100),
               "excludeDomains": Param("str_list", max_length=200, high=100),
               "startPublishedDate": Param("str", max_length=40), "endPublishedDate": Param("str", max_length=40)},
              cost=_usd_field("costDollars", "total")),
    Operation("exa", "contents", "POST", _EXA + "/contents", UNIT_VENDOR_USD, 50_000,
              {"urls": Param("str_list", required=True, max_length=500, high=10), "text": Param("bool"),
               "highlights": Param("bool")},
              cost=_usd_field("costDollars", "total")),
    # HarvestAPI: per-request credits. Its prices are not published per
    # endpoint, so launch keeps the two profile reads only.
    Operation("harvestapi", "linkedin_profile", "GET", _HARVEST + "/linkedin/profile", UNIT_CALL, 1,
              {"url": _URL}),
    Operation("harvestapi", "linkedin_company", "GET", _HARVEST + "/linkedin/company", UNIT_CALL, 1,
              {"url": _URL}),
)

OPERATIONS: Dict[Tuple[str, str], Operation] = {(op.tool, op.name): op for op in _OPERATIONS}
TOOLS: Dict[str, str] = {op.tool: op.unit for op in _OPERATIONS}
_NAME_RE = re.compile(r"^[a-z_]{1,40}$")

# One unit per tool, so one ask prices every operation of that tool. A
# vendor-USD operation must read the vendor's cost.
assert all(op.unit == TOOLS[op.tool] and (op.unit == UNIT_CALL or op.cost) for op in _OPERATIONS)
assert all(_NAME_RE.fullmatch(op.tool) and _NAME_RE.fullmatch(op.name) for op in _OPERATIONS)


def operation(tool: Any, name: Any) -> Operation:
    found = OPERATIONS.get((tool, name)) if isinstance(tool, str) and isinstance(name, str) else None
    if found is None:
        raise CatalogError("operation_unknown")
    return found


__all__ = ["CatalogError", "OPERATIONS", "Operation", "Param", "TOOLS", "UNIT_CALL", "UNIT_VENDOR_USD", "operation"]
