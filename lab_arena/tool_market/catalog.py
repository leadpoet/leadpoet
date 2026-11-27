"""The tool market's catalog: which vendor operations a buyer may call.

Each operation names one vendor endpoint, the parameters a buyer may send,
how its usage is counted, and the most it may count for one call. The vault
enforces the same host and route shape again (``lead_order_vault.vault.TOOLS``);
this table is narrower, so a buyer reaches only the listed operations.

Usage units:

- ``call``: the vendor charges a fixed number of credits per request. A
  miner asks a price in micro-USD per unit, and a 2xx answer counts
  ``call_units`` units.
- ``vendor_usd``: the vendor reports the request's own cost in its answer.
  A miner asks a price in micro-USD per vendor USD (``900000`` sells at 90%
  of the vendor's price), and the call counts the reported micro-USD.
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
    # ``call`` unit: units a 2xx answer counts. ``vendor_usd``: the cost reader.
    call_units: int = 1
    cost: Optional[Callable[[Any], Optional[int]]] = field(default=None, compare=False)
    # DataForSEO takes a list of tasks; the catalog sends exactly one.
    one_task_array: bool = False

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
        document: Any = [values] if self.one_task_array else values
        return self.url, headers, json.dumps(document, separators=(",", ":")).encode("utf-8")

    def units(self, status: int, document: Any) -> Optional[int]:
        """Units a vendor answer counts: 0 for a refusal, None when unknown."""

        if 400 <= status < 500:
            return 0
        if not 200 <= status < 300:
            return None
        if self.unit == UNIT_CALL:
            return self.call_units
        reported = self.cost(document) if self.cost is not None else None
        return None if reported is None else min(reported, self.max_units)


_URL = Param("str", required=True, max_length=300)
_SC = "https://api.scrapecreators.com"
_DFS = "https://api.dataforseo.com/v3"
_HARVEST = "https://api.harvestapi.io"
_EXA = "https://api.exa.ai"

_OPERATIONS: Sequence[Operation] = (
    # ScrapeCreators: one credit per request.
    Operation("scrapecreators", "linkedin_profile", "GET", _SC + "/v1/linkedin/profile", UNIT_CALL, 1,
              {"url": _URL}),
    Operation("scrapecreators", "linkedin_company", "GET", _SC + "/v1/linkedin/company", UNIT_CALL, 1,
              {"url": _URL}),
    Operation("scrapecreators", "google_search", "GET", _SC + "/v1/google/search", UNIT_CALL, 1,
              {"query": Param("str", required=True, max_length=300), "region": Param("str", max_length=2),
               "page": Param("int", low=1, high=11)}),
    # DataForSEO: each answer carries its own USD ``cost``.
    Operation("dataforseo", "google_organic_serp", "POST", _DFS + "/serp/google/organic/live/advanced",
              UNIT_VENDOR_USD, 50_000,
              {"keyword": Param("str", required=True, max_length=700), "location_code": Param("int", high=10**8),
               "language_code": Param("str", max_length=8), "depth": Param("int", high=200)},
              cost=_usd_field("cost"), one_task_array=True),
    Operation("dataforseo", "google_ads_search_volume", "POST", _DFS + "/keywords_data/google_ads/search_volume/live",
              UNIT_VENDOR_USD, 150_000,
              {"keywords": Param("str_list", required=True, max_length=80, high=1000),
               "location_code": Param("int", high=10**8), "language_code": Param("str", max_length=8)},
              cost=_usd_field("cost"), one_task_array=True),
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

# One unit per tool, so one ask prices every operation of that tool.
assert all(op.unit == TOOLS[op.tool] for op in _OPERATIONS)
assert all(_NAME_RE.fullmatch(op.tool) and _NAME_RE.fullmatch(op.name) for op in _OPERATIONS)


def operation(tool: Any, name: Any) -> Operation:
    found = OPERATIONS.get((tool, name)) if isinstance(tool, str) and isinstance(name, str) else None
    if found is None:
        raise CatalogError("operation_unknown")
    return found


__all__ = ["CatalogError", "OPERATIONS", "Operation", "Param", "TOOLS", "UNIT_CALL", "UNIT_VENDOR_USD", "operation"]
