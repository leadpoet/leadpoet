"""The tool market's catalog: which vendor operations a buyer may call.

Each operation names one vendor endpoint, the parameters a buyer may send,
how its usage is counted, and the most it may count for one call. The vault
enforces the same host and route shape again (``lead_order_vault.tools.TOOLS``);
this table is narrower, so a buyer reaches only the listed operations.

Usage units:

- ``call``: the vendor charges credits per request. A miner asks a price in
  micro-USD per credit. A 2xx answer counts 1 credit, or what the
  operation's ``cost`` reader finds the vendor charged (no email found is
  often free).
- ``vendor_usd``: the vendor reports the request's own cost in its answer.
  A miner asks a price in micro-USD per vendor USD (``900000`` sells at 90%
  of the vendor's price), and the call counts the reported micro-USD.

Adding a tool is data, not code: one ``Operation`` per endpoint and a
``REQUEST_PRIVACY`` entry here, plus the tool's key placement and route rule
in ``lead_order_vault.tools.TOOLS``, which the attested vault enforces on its own.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote_plus, urlencode

UNIT_CALL = "call"
UNIT_VENDOR_USD = "vendor_usd"
UNIT_SCALE = {UNIT_CALL: 1, UNIT_VENDOR_USD: 1_000_000}
# The highest ask a miner may post. Per call: above every launch vendor's
# own list price. Per vendor USD: the vendor's own price.
MAX_ASK = {UNIT_CALL: 250_000, UNIT_VENDOR_USD: 1_000_000}
# The vault refuses a query value longer than this once URL-encoded, and a
# URL longer than MAX_URL.
MAX_ENCODED_QUERY_VALUE = 2000
MAX_URL = 4096


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
    vendor_name: Optional[str] = None

    def _text(self, value: Any) -> bool:
        # JSON allows a lone surrogate, which no request can encode.
        if not isinstance(value, str) or not 1 <= len(value) <= self.max_length:
            return False
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True

    def schema(self) -> Dict[str, Any]:
        """The JSON Schema of what ``read`` accepts (an operation's cross-field rules aside)."""

        text = {"type": "string", "minLength": 1, "maxLength": self.max_length}
        return {"str": text, "int": {"type": "integer", "minimum": self.low, "maximum": self.high},
                "bool": {"type": "boolean"}, "enum": {"enum": list(self.choices)},
                "str_list": {"type": "array", "minItems": self.low, "maxItems": self.high, "items": text}}[self.kind]

    def read(self, value: Any) -> Any:
        if self.kind == "str":
            if not self._text(value):
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
            if not all(self._text(item) for item in value):
                raise CatalogError("param_invalid")
            return list(value)
        raise AssertionError(self.kind)


def _usd_field(*path: str) -> Callable[[Any], Optional[int]]:
    """Read a vendor-reported USD cost and round it up to micro-USD."""

    def read(document: Any) -> Optional[int]:
        value = _get(document, *path)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return None
        try:
            usd = Decimal(str(value))
        except InvalidOperation:
            return None
        if not usd.is_finite() or usd < 0 or usd > 1_000_000:  # also keeps the product finite
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
    # Units a 2xx answer counts: ``cost(document)`` when set, else 1.
    cost: Optional[Callable[[Any], Optional[int]]] = field(default=None, compare=False)
    # Translate checked params to the vendor's body or query fields.
    wrap: Optional[Callable[[Dict[str, Any]], Any]] = field(default=None, compare=False)
    # A vendor whose status codes say something else, such as an error in a
    # 200 or "out of credits" as 429: the status its answer means.
    real_status: Optional[Callable[[int, Any], int]] = field(default=None, compare=False)
    # One line on the job it does, so an agent picks by task, not vendor.
    does: str = ""
    timeout_seconds: float = 30.0

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
        values = {self.params[name].vendor_name or name: self.params[name].read(value)
                  for name, value in params.items()}
        if self.wrap is not None:
            values = self.wrap(values)
        headers = {"accept": "application/json"}
        if self.method == "GET":
            if any(len(quote_plus(str(v))) > MAX_ENCODED_QUERY_VALUE for v in values.values()):
                raise CatalogError("param_invalid")
            url = self.url + ("?" + urlencode(sorted(values.items())) if values else "")
            if len(url) > MAX_URL:
                raise CatalogError("param_invalid")
            return url, headers, b""
        headers["content-type"] = "application/json"
        return self.url, headers, json.dumps(values, separators=(",", ":")).encode("utf-8")

    def status(self, http_status: int, document: Any) -> int:
        """The answer's real status, in the meaning the market acts on."""

        return http_status if self.real_status is None else self.real_status(http_status, document)

    def units(self, status: int, document: Any) -> Optional[int]:
        """Units an answer counts, given ``status()``: 0 for a refusal, None when unknown."""

        if 400 <= status < 500:
            return 0
        if not 200 <= status < 300 or document is None:
            return None  # a 2xx the gateway cannot send back is unknown too
        if self.cost is None:
            return 1
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
    if result == "unknown":
        return 0
    return 1 if result in ("valid", "invalid", "catch-all", "spamtrap", "abuse", "do_not_mail") else None


def _findymail_status(status: int, _: Any) -> int:
    return 401 if status == 423 else status


def _hunter_status(status: int, _: Any) -> int:
    return {403: 429, 429: 402}.get(status, status)


def _prospeo_status(status: int, document: Any) -> int:
    code = _get(document, "error_code")
    return {"INSUFFICIENT_CREDITS": 402, "INVALID_API_KEY": 401}.get(code, status) if isinstance(code, str) else status


def _prospeo_cost(document: Any) -> Optional[int]:
    if _get(document, "error") is not False:
        return None
    free = _get(document, "free_enrichment")
    return int(not free) if isinstance(free, bool) else None


def _found_email(container: str) -> Callable[[Any], Optional[int]]:
    """One credit when the answer holds an email in ``container``; finding none is free."""

    def read(document: Any) -> Optional[int]:
        if not isinstance(document, dict) or container not in document:
            return None
        result = document[container]
        if result is None:
            return 0
        if not isinstance(result, dict) or "email" not in result:
            return None
        email = result["email"]
        if email is None:
            return 0
        return int(bool(email.strip())) if isinstance(email, str) else None

    return read


def _exa_search_body(values: Dict[str, Any]) -> Dict[str, Any]:
    category = values.get("category")
    if category in ("company", "people") and ("startPublishedDate" in values or "endPublishedDate" in values):
        raise CatalogError("param_invalid")
    if category == "people" and "excludeDomains" in values:
        raise CatalogError("param_invalid")
    return values


def _prospeo_body(values: Dict[str, Any]) -> Dict[str, Any]:
    person = dict(values)
    flag = person.pop("only_verified_email", None)
    return {"data": person} if flag is None else {"data": person, "only_verified_email": flag}


_PROSPEO_FIELDS = ("first_name", "last_name", "full_name", "linkedin_url", "email", "company_name",
                   "company_website", "company_linkedin_url", "person_id")
_NAME = Param("str", max_length=200)
_URL = Param("str", required=True, max_length=300)
# Harvest names the vendor query parameter "query"; keep the buyer-facing "url".
_HARVEST_URL = Param("str", required=True, max_length=300, vendor_name="query")


def _harvest_cost(document: Any) -> Optional[int]:
    element = _get(document, "element")
    if _get(document, "error") or not isinstance(element, dict) or not element:
        return None
    return 1 if _get(document, "status") in (200, "200") else None

def _harvest_page_cost(document: Any) -> Optional[int]:
    elements = _get(document, "elements")
    if _get(document, "error") or _get(document, "status") not in (200, "200"):
        return None
    if not isinstance(elements, list) or not all(isinstance(row, dict) for row in elements):
        return None
    return 1


# One lead-search page, with the subset of vendor filters Lead Search uses.
# A page after the first needs the sessionId that ToolMarket._session pins
# to the key that served page one.
_HARVEST_LEAD_LIMITS = {
    "currentCompanies": 50, "currentJobTitles": 70, "excludeCurrentJobTitles": 70,
    "locations": 70, "lastNames": 70, "pastCompanies": 50, "schools": 50,
    "industryIds": 70, "seniorityLevelIds": 20, "companyHeadcount": 20,
}


def _lead_search_query(values: Dict[str, Any]) -> Dict[str, Any]:
    if values.get("page", 1) > 1 and "sessionId" not in values:
        raise CatalogError("session_required")
    if not any(name in values for name in _HARVEST_LEAD_LIMITS):
        raise CatalogError("param_missing")
    for name, limit in _HARVEST_LEAD_LIMITS.items():
        if name in values:
            items = values[name].split(",")
            if len(items) > limit or any(not item.strip() for item in items):
                raise CatalogError("param_invalid")
    # Query flags sent by urllib must have the vendor's lowercase spelling.
    if "recentlyChangedJobs" in values:
        values["recentlyChangedJobs"] = "true" if values["recentlyChangedJobs"] else "false"
    return values


_ZB = "https://api.zerobounce.net"
_HARVEST = "https://api.harvestapi.io"
_EXA = "https://api.exa.ai"

_OPERATIONS: Sequence[Operation] = (
    # ZeroBounce: one credit per check; an ``unknown`` result is free.
    Operation("zerobounce", "validate", "GET", _ZB + "/v2/validate", UNIT_CALL, 1,
              {"email": Param("str", required=True, max_length=320), "ip_address": Param("str", max_length=45)},
              cost=_zerobounce_cost, real_status=_zerobounce_status,
              does="Check whether an email address can receive mail."),
    # Prospeo: one credit per match; no match, or a free re-enrichment, is free.
    Operation("prospeo", "enrich_person", "POST", "https://api.prospeo.io/enrich-person", UNIT_CALL, 1,
              dict({name: _NAME for name in _PROSPEO_FIELDS}, only_verified_email=Param("bool")),
              cost=_prospeo_cost, real_status=_prospeo_status,
              wrap=_prospeo_body, does="Find a person's work email and profile from a name, company or LinkedIn URL."),
    # Findymail: one credit per email found. 423 is a paused account, which no free check sees, so it counts as a bad key.
    Operation("findymail", "find_from_linkedin", "POST", "https://app.findymail.com/api/search/business-profile",
              UNIT_CALL, 1, {"linkedin_url": _URL}, cost=_found_email("contact"),
              real_status=_findymail_status, does="Find a person's work email from their LinkedIn URL."),
    Operation("findymail", "find_from_name", "POST", "https://app.findymail.com/api/search/name", UNIT_CALL, 1,
              {"name": Param("str", required=True, max_length=200), "domain": Param("str", required=True, max_length=253)},
              cost=_found_email("contact"), real_status=_findymail_status,
              does="Find a person's work email from their name and company domain."),
    # Hunter: one search per email found. Its 403 is a rate limit and its
    # 429 an empty account.
    Operation("hunter", "email_finder", "GET", "https://api.hunter.io/v2/email-finder", UNIT_CALL, 1,
              {"domain": Param("str", max_length=253), "company": _NAME, "linkedin_handle": _NAME,
               "first_name": _NAME, "last_name": _NAME, "full_name": _NAME,
               "max_duration": Param("int", low=3, high=20)},
              cost=_found_email("data"), real_status=_hunter_status,
              does="Find a person's work email from their name and company."),
    # Exa: each answer carries ``costDollars.total``, which Exa calls an estimate.
    Operation("exa", "search", "POST", _EXA + "/search", UNIT_VENDOR_USD, 50_000,
              {"query": Param("str", required=True, max_length=2000), "numResults": Param("int", high=25),
               "type": Param("enum", choices=("auto", "fast")),
               "category": Param("enum", choices=("company", "people", "news", "publication", "research paper", "pdf",
                                                  "github", "tweet", "personal site", "linkedin profile",
                                                  "financial report")),
               "includeDomains": Param("str_list", max_length=200, high=100),
               "excludeDomains": Param("str_list", max_length=200, high=100),
               "startPublishedDate": Param("str", max_length=40), "endPublishedDate": Param("str", max_length=40)},
              cost=_usd_field("costDollars", "total"), wrap=_exa_search_body, does="Search the web for pages, companies or people."),
    Operation("exa", "contents", "POST", _EXA + "/contents", UNIT_VENDOR_USD, 50_000,
              {"urls": Param("str_list", required=True, max_length=500, high=10), "text": Param("bool"),
               "highlights": Param("bool")},
              cost=_usd_field("costDollars", "total"), does="Read the text of up to 10 web pages."),
    Operation("harvestapi", "get_job", "GET", _HARVEST + "/linkedin/job", UNIT_CALL, 1,
              {"jobId": Param("str", required=True, max_length=80)}, cost=_harvest_cost,
              does="Read a LinkedIn job posting by ID."),
    Operation("harvestapi", "search_jobs", "GET", _HARVEST + "/linkedin/job-search", UNIT_CALL, 1,
              {"search": _NAME, "location": _NAME, "industryId": Param("str", max_length=500),
               "sortBy": Param("enum", choices=("relevance", "date")),
               "postedLimit": Param("enum", choices=("24h", "week", "month")), "page": Param("int", high=40)},
              cost=_harvest_page_cost, does="Search one page of LinkedIn jobs with date and industry filters."),
    Operation("harvestapi", "search_posts", "GET", _HARVEST + "/linkedin/post-search", UNIT_CALL, 1,
              {"search": Param("str", max_length=1000), "companyId": Param("str", vendor_name="company"),
               "authorsCompany": Param("str"), "postedLimit": Param("enum", choices=("24h", "week", "month")),
               "sortBy": Param("enum", choices=("relevance", "date")), "page": Param("int", high=100),
               "contentType": Param("enum", choices=("videos", "images", "live_videos", "documents",
                                                     "collaborative_articles", "jobs"))},
              cost=_harvest_page_cost, does="Search one page of posts by phrase, company or author employer."),
    Operation("harvestapi", "profile_posts", "GET", _HARVEST + "/linkedin/profile-posts", UNIT_CALL, 1,
              {"profileId": Param("str", required=True, vendor_name="profile"), "page": Param("int", high=100),
               "paginationToken": Param("str", max_length=1500)},
              cost=_harvest_page_cost, does="Read one page of posts from a LinkedIn member."),
    *(Operation("harvestapi", name, "GET", _HARVEST + "/linkedin/" + path, UNIT_CALL, 1,
                {"post": _URL, "page": Param("int", high=100)}, cost=_harvest_page_cost, does=description)
      for name, path, description in (
          ("post_reactions", "post-reactions", "Read one page of reactions to a LinkedIn post."),
          ("post_comments", "post-comments", "Read one page of comments on a LinkedIn post."))),
    # HarvestAPI: per-request credits. A tool-wide ask covers only the profile
    # and company reads (book.GENERAL_ASK_OPERATIONS); every other SKU needs its own ask.
    Operation("harvestapi", "linkedin_profile", "GET", _HARVEST + "/linkedin/profile", UNIT_CALL, 1,
              {"url": _HARVEST_URL, "main": Param("enum", choices=("true",))}, cost=_harvest_cost, does="Read a LinkedIn profile: name, headline, current jobs."),
    Operation("harvestapi", "linkedin_profile_email", "GET", _HARVEST + "/linkedin/profile", UNIT_CALL, 1,
              {"url": _HARVEST_URL, "main": Param("enum", choices=("true",))},
              wrap=lambda values: dict(values, findEmail="true"), cost=_harvest_cost,
              does="Read a LinkedIn profile with the vendor's email discovery and SMTP checks."),
    Operation("harvestapi", "search_leads", "GET", _HARVEST + "/linkedin/lead-search", UNIT_CALL, 1,
              {**{key: Param("str", max_length=2000) for key in _HARVEST_LEAD_LIMITS},
               "recentlyChangedJobs": Param("bool"), "sessionId": Param("str", max_length=200), "page": Param("int", high=100)},
              wrap=_lead_search_query, cost=_harvest_page_cost, timeout_seconds=90.0,
              does="Search a Sales Navigator lead page by company, role or location."),
    Operation("harvestapi", "linkedin_company", "GET", _HARVEST + "/linkedin/company", UNIT_CALL, 1,
              {"url": _HARVEST_URL}, cost=_harvest_cost, does="Read a LinkedIn company page."),
)

OPERATIONS: Dict[Tuple[str, str], Operation] = {(op.tool, op.name): op for op in _OPERATIONS}
TOOLS: Dict[str, str] = {op.tool: op.unit for op in _OPERATIONS}

# How private a tool's requests are from the key's owner, in the vendor's own
# dashboard or history. "visible": the vendor documents a request history.
# "unverified": not documented, so treat it as visible. "private" is set only
# after Leadpoet checks a vendor account itself. Vendor docs, Nov 27 2026.
REQUEST_PRIVACY: Dict[str, str] = {
    "zerobounce": "visible",  # validation results, kept up to 30 days
    "hunter": "visible",  # a request history per call
    "prospeo": "unverified",
    "findymail": "unverified",
    "harvestapi": "unverified",
    "exa": "unverified",
}
assert set(REQUEST_PRIVACY) == set(TOOLS) and set(REQUEST_PRIVACY.values()) <= {"visible", "unverified", "private"}
_NAME_RE = re.compile(r"^[a-z_]{1,40}$")

# One unit per tool, so its ask ceiling and unit hold for every operation. A
# vendor-USD operation must read the vendor's cost. Agents choose by task,
# so every operation says what it does.
assert all(op.unit == TOOLS[op.tool] and (op.unit == UNIT_CALL or op.cost) and op.does for op in _OPERATIONS)
assert all(_NAME_RE.fullmatch(op.tool) and _NAME_RE.fullmatch(op.name) for op in _OPERATIONS)


def operation(tool: Any, name: Any) -> Operation:
    found = OPERATIONS.get((tool, name)) if isinstance(tool, str) and isinstance(name, str) else None
    if found is None:
        raise CatalogError("operation_unknown")
    return found


__all__ = ["CatalogError", "MAX_ASK", "OPERATIONS", "Operation", "TOOLS", "operation"]
