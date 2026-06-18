"""Deterministic L0 checks for Research Lab evidence snapshots.

The open verifier checks only mechanics that can be rerun byte-for-byte:
schema-shaped signal fields, URL/source/domain consistency, verbatim snippet
grounding, date-in-content checks, prompt-injection patterns, freshness caps,
and obvious bot/login-wall pages. It does not perform semantic judging or any
network fetches.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class Finding:
    check_id: str
    severity: str
    message: str
    details: Dict[str, Any]


@dataclass(frozen=True)
class L0Result:
    passed: bool
    findings: Tuple[Finding, ...]
    metrics: Dict[str, Any]


_PROMPT_INJECTION_PATTERNS = [
    re.compile(
        r"\b(?:ignore|disregard|forget|skip|bypass|override|nullify|cancel)\s+"
        r"(?:all\s+|any\s+|the\s+|every\s+|whatever\s+|what\s+(?:was\s+)?)?"
        r"(?:previous|prior|above|earlier|preceding|former|original|initial)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:ignore|disregard)\s+(?:everything|all)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(?:everything|all|what|that)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:new|updated?|revised?|fresh|different)\s+"
        r"(?:instructions?|task|prompt|rules?|directives?|orders?|guidelines?)\s*"
        r"(?:[:.]|are|is|to|that)",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\|(?:im_(?:start|end)|endoftext|fim_[a-z]+|begin_of_text|end_of_text)\|>",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|\n)\s*(?:system|assistant|user)\s*[:>]", re.IGNORECASE),
    re.compile(
        r"\b(?:return|respond|reply|output|give|set|make|use|score|assign)\s+"
        r"(?:with\s+|this\s+|a\s+|the\s+)?(?:score|value|rating)?\s*"
        r"(?:of\s+|=\s*|:\s*|to\s+)?\s*(?:5\d|60)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bscore\s*[:=]\s*(?:5\d|60)\b", re.IGNORECASE),
    re.compile(r"\bmatched_icp_signal_idx\s*[:=]", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:a\s+)?(?:different|new)", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(?:a\s+)?(?:different|new)", re.IGNORECASE),
    re.compile(r"\bfollow\s+(?:these|the)\s+new\b", re.IGNORECASE),
]

_CONTROL_TOKEN_RE = re.compile(
    r"<\|(?:im_(?:start|end)|endoftext|fim_[a-z]+|begin_of_text|end_of_text)\|>",
    re.IGNORECASE,
)
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u2028-\u202F\u2060-\u206F\uFEFF]")
_TRIPLE_BACKTICK_RE = re.compile(r"```")

_INVALID_URL_PATTERNS = [
    r"/alternatives(?:\b|/|\?|$)",
    r"/competitors(?:\b|/|\?|$)",
    r"indeed\.com/hire/job-description/",
    r"github\.com/[^/]+/[^/]+/labels(?:/|$)",
    r"github\.com/[^/]+/[^/]+/discussions/\d+(?:/|$)",
]
_INVALID_URL_RE = re.compile("|".join(_INVALID_URL_PATTERNS), re.IGNORECASE)

_ANTIBOT_PATTERNS = [
    r"access denied",
    r"verifying your connection",
    r"verifying.{0,30}browser",
    r"just a moment",
    r"enable javascript",
    r"please enable js",
    r"additional verification required",
    r"verifying you are human",
    r"sign in to (?:linkedin|see|join|view|continue)",
    r"join linkedin to",
    r"create an account to (?:see|join)",
    r"page can.?t be found",
    r"this page (?:doesn.?t|does not) exist",
    r"403\s*(?:[-:|]|--)?\s*forbidden",
    r"404\s*(?:[-:|]|--)?\s*(?:not\s*found|page.*not.*found)",
    r"this content isn.?t available",
]
_ANTIBOT_RE = re.compile("|".join(_ANTIBOT_PATTERNS), re.IGNORECASE)
_ANTIBOT_MAX_LEN = 4000

_FRESHNESS_WINDOWS = {
    "in the last few weeks": 60,
    "in the last 30 days": 45,
    "in the last 60 days": 75,
    "in the last 90 days": 105,
    "in the last 6 months": 200,
    "in the last 12 months": 400,
    "in the past few weeks": 60,
    "in the past 30 days": 45,
    "in the past 60 days": 75,
    "in the past 90 days": 105,
    "in the past 6 months": 200,
    "in the past 12 months": 400,
    "last few weeks": 60,
    "last 30 days": 45,
    "last 60 days": 75,
    "last 90 days": 105,
    "last 6 months": 200,
    "last 12 months": 400,
    "past 30 days": 45,
    "past 60 days": 75,
    "past 90 days": 105,
    "past 6 months": 200,
    "past 12 months": 400,
    "recently": 180,
}

_SOURCE_DOMAIN_ALLOWLIST = {
    "linkedin": frozenset({"linkedin.com"}),
    "github": frozenset({"github.com", "raw.githubusercontent.com"}),
    "wikipedia": frozenset({"en.wikipedia.org", "wikipedia.org"}),
}

_SOURCE_HOST_SUFFIX_ALLOWLIST = {
    "job_board": (
        "greenhouse.io",
        "lever.co",
        "ashbyhq.com",
        "workdayjobs.com",
        "smartrecruiters.com",
        "bamboohr.com",
        "indeed.com",
    ),
}

_COMPANY_DOMAIN_REQUIRED_SOURCES = frozenset({"company_website"})

_SIGNAL_WORDS = {
    "launched",
    "announced",
    "expanded",
    "expanding",
    "partnered",
    "partnership",
    "merged",
    "acquisition",
    "acquired",
    "hired",
    "hiring",
    "recruited",
    "recruiting",
    "opening",
    "openings",
    "funding",
    "funded",
    "raised",
    "secured",
    "closed",
    "obtained",
    "invested",
    "investment",
    "seed",
    "series",
}

GENERIC_INTENT_PATTERNS = [
    r"is\s+actively\s+operating\s+in\s+\w+",
    r"visible\s+market\s+activity",
    r"market\s+activity\s+and\s+company\s+updates",
    r"business\s+operations\s+and\s+updates",
    r"^.{0,50}\s+is\s+(?:actively\s+)?(?:operating|expanding|growing)",
    r"company\s+(?:updates|activities|operations)",
    r"market\s+(?:activity|presence|operations)",
]

SPECIFIC_INTENT_KEYWORDS = [
    "hiring",
    "recruit",
    "job",
    "position",
    "opening",
    "launch",
    "released",
    "announced",
    "introduced",
    "raised",
    "funding",
    "series",
    "investment",
    "partnership",
    "partnered",
    "collaboration",
    "acquired",
    "acquisition",
    "merger",
    "expansion",
    "opened",
    "new office",
    "new location",
    "migrating",
    "adopting",
    "implementing",
]

_MONTH_NAMES = {
    1: ("january", "jan"),
    2: ("february", "feb"),
    3: ("march", "mar"),
    4: ("april", "apr"),
    5: ("may", "may"),
    6: ("june", "jun"),
    7: ("july", "jul"),
    8: ("august", "aug"),
    9: ("september", "sep"),
    10: ("october", "oct"),
    11: ("november", "nov"),
    12: ("december", "dec"),
}


def extract_domain(url: str) -> str:
    """Return the registrable-ish domain used by current scoring code."""
    if not url:
        return ""
    try:
        clean = url.strip()
        if not clean.lower().startswith(("http://", "https://")):
            clean = "https://" + clean
        hostname = urlparse(clean).hostname or ""
        hostname = hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        parts = hostname.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return hostname
    except Exception:
        return url.lower().strip()


def _extract_hostname(url: str) -> str:
    if not url:
        return ""
    clean = url.strip()
    if not clean.lower().startswith(("http://", "https://")):
        clean = "https://" + clean
    hostname = urlparse(clean).hostname or ""
    return hostname.lower().lstrip("www.")


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_snippet_overlap(snippet: str, content: str) -> float:
    """Fraction of snippet 4-word n-grams found verbatim in source content."""
    norm_snippet = _normalize_text(snippet)
    norm_content = _normalize_text(content)
    snippet_words = norm_snippet.split()
    if len(snippet_words) < 4:
        return 1.0

    content_words = norm_content.split()
    content_set = {
        tuple(content_words[i : i + 4])
        for i in range(max(0, len(content_words) - 3))
    }
    total = len(snippet_words) - 3
    matches = sum(
        1
        for i in range(total)
        if tuple(snippet_words[i : i + 4]) in content_set
    )
    return matches / total if total > 0 else 1.0


def check_description_grounding(description: str, source_content: str) -> float:
    stop_words = {
        "about",
        "after",
        "being",
        "between",
        "could",
        "during",
        "every",
        "first",
        "their",
        "these",
        "those",
        "through",
        "under",
        "using",
        "which",
        "while",
        "would",
        "other",
        "there",
        "where",
        "should",
        "company",
        "business",
        "service",
        "services",
        "solution",
        "solutions",
        "based",
        "including",
        "across",
        "within",
    }
    content_words = set(_normalize_text(source_content).split())
    desc_words = [
        word
        for word in _normalize_text(description).split()
        if len(word) >= 5 and word not in stop_words
    ]
    if len(desc_words) < 3:
        return 1.0
    found = sum(1 for word in desc_words if word in content_words)
    return found / len(desc_words)


def check_signal_word_grounding(text: str, source_content: str) -> Tuple[int, int, List[str]]:
    content_words = set(_normalize_text(source_content).split())
    text_words = set(_normalize_text(text).split())
    signal_words = text_words & _SIGNAL_WORDS
    if not signal_words:
        return 0, 0, []
    grounded = signal_words & content_words
    return len(grounded), len(signal_words), sorted(signal_words - content_words)


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    if not text:
        return False, ""
    for pattern in _PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, match.group(0)[:80]
    return False, ""


def sanitize_miner_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _CONTROL_TOKEN_RE.sub(" ", text)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = _TRIPLE_BACKTICK_RE.sub("''' ", cleaned)
    cleaned = re.sub(r"[ \t]{3,}", " ", cleaned)
    return cleaned.strip()


def is_generic_intent_description(description: str) -> Tuple[bool, str]:
    desc_lower = (description or "").lower().strip()
    for pattern in GENERIC_INTENT_PATTERNS:
        if re.search(pattern, desc_lower, re.IGNORECASE):
            return True, f"generic pattern: {pattern[:40]}"

    has_specific = any(keyword in desc_lower for keyword in SPECIFIC_INTENT_KEYWORDS)
    if len(desc_lower) < 80 and not has_specific:
        return True, "too short and lacks specific intent keywords"

    templated = r"^\w+(?:\s+\w+){0,3}\s+is\s+\w+ing\s+(?:in\s+)?\w+\s*\.?$"
    if re.match(templated, desc_lower) and not has_specific:
        return True, "templated structure with no specific details"
    return False, "description appears specific"


def check_url_structural_validity(url: str) -> Optional[str]:
    match = _INVALID_URL_RE.search(url or "")
    if not match:
        return None
    return f"URL path {match.group()!r} is structurally invalid evidence"


def check_antibot_wall(content: str) -> Optional[str]:
    if not content:
        return None
    head = content[:5000].lower()
    match = _ANTIBOT_RE.search(head)
    if match and len(content) < _ANTIBOT_MAX_LEN:
        return f"page returned anti-bot/login-wall content: {match.group()[:40]!r}"
    return None


def _claim_max_age_days(claim_text: str) -> Optional[int]:
    if not claim_text:
        return None
    lowered = claim_text.lower()
    best = None
    for phrase, days in _FRESHNESS_WINDOWS.items():
        if phrase in lowered and (best is None or days < best):
            best = days
    return best


def _parse_signal_datetime(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        parsed = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        try:
            parsed = datetime.strptime(str(date_str), "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except (TypeError, ValueError):
            return None
    if parsed.year < 2000:
        return None
    return parsed


def check_evidence_freshness(
    claim_text: str,
    signal_date: Optional[str],
    content_found_date: Optional[str] = None,
    buyer_cap_days: Optional[int] = None,
    *,
    now: Optional[datetime] = None,
) -> Optional[str]:
    max_age = buyer_cap_days
    if max_age is None:
        max_age = _claim_max_age_days(claim_text)
    if max_age is None:
        return None

    date_str = signal_date or content_found_date
    parsed = _parse_signal_datetime(date_str)
    if parsed is None:
        if buyer_cap_days is not None:
            return f"buyer requires evidence within {buyer_cap_days} days but signal has no valid date"
        return None

    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    age_days = (clock - parsed).days
    if age_days > max_age:
        return f"signal date {date_str} is {age_days} days old; max allowed is {max_age}"
    return None


def strip_copyright_founded_years(content: str) -> str:
    content = re.sub(
        r"(?:\u00a9|\(c\)|copyright)\s*(?:\u00a9|\(c\))?\s*(?:19|20)\d{2}(?:\s*[-]\s*(?:19|20)\d{2})?",
        "XXXX",
        content or "",
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"(?:founded|established|since|est\.?)\s+(?:in\s+)?(?:19|20)\d{2}\b",
        "XXXX",
        content,
        flags=re.IGNORECASE,
    )
    return content


def strip_dynamic_boilerplate_dates(content: str) -> str:
    iso = r"(?:19|20)\d{2}-\d{2}-\d{2}"
    slash = r"\d{1,2}/\d{1,2}/(?:19|20)\d{2}"
    month = (
        r"(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    )
    named = rf"{month}\s+\d{{1,2}}[,]?\s*(?:19|20)\d{{2}}"
    named_dmy = rf"\d{{1,2}}\s+{month}[,]?\s*(?:19|20)\d{{2}}"
    any_date = rf"(?:{iso}|{slash}|{named}|{named_dmy})"
    prefixes = (
        r"(?:last\s+)?(?:updated|modified|refreshed|generated|retrieved|accessed|fetched)"
        r"|as\s+of"
        r"|current\s+(?:as\s+of|date)"
        r"|page\s+(?:updated|generated|modified)"
        r"|date\s*:"
    )
    return re.sub(
        rf"(?:{prefixes})\s*(?:on\s+)?:?\s*{any_date}",
        "XXXX",
        content or "",
        flags=re.IGNORECASE,
    )


_TIMESTAMP_ANCHOR_RE = re.compile(
    r"\b(?:posted|published|shared|uploaded|edited|updated|created|written|date(?:d)?)"
    r"\b\s*(?:on|at|in)?\s*[:*.\-\u2022\u00b7]?\s*"
    r"(?:about\s+|approximately\s+|~\s*)?"
    r"(\d+)\s*(year|month|week|day|hour|minute)s?\s+ago",
    re.IGNORECASE,
)
_LINKEDIN_TIMESTAMP_BADGE_RE = re.compile(
    r"(?<![A-Za-z])(\d+)\s*(mo|w|d|h|m)\s*(?:[*.\u2022\u00b7]|\bedited\b)",
    re.IGNORECASE,
)
_ANCHOR_RELATIVE_WORDS_RE = re.compile(
    r"\b(?:posted|published|shared|uploaded|edited|created|written|date(?:d)?)"
    r"\b\s*(?:on|at|in)?\s*[:*.\-\u2022\u00b7]?\s*"
    r"(yesterday|today|last\s+week|this\s+week|last\s+month|this\s+month|last\s+year)",
    re.IGNORECASE,
)
_UNIT_TO_DAYS = {
    "year": 365,
    "month": 30,
    "week": 7,
    "day": 1,
    "hour": 0,
    "minute": 0,
    "mo": 30,
    "w": 7,
    "d": 1,
    "h": 0,
    "m": 30,
}
_WORD_TO_DAYS = {
    "yesterday": 1,
    "today": 0,
    "last week": 7,
    "this week": 0,
    "last month": 30,
    "this month": 0,
    "last year": 365,
}


def _extract_implied_ages_days(content: str) -> List[int]:
    out = []
    for match in _TIMESTAMP_ANCHOR_RE.finditer(content or ""):
        out.append(int(match.group(1)) * _UNIT_TO_DAYS.get(match.group(2).lower(), 0))
    for match in _LINKEDIN_TIMESTAMP_BADGE_RE.finditer(content or ""):
        out.append(int(match.group(1)) * _UNIT_TO_DAYS.get(match.group(2).lower(), 0))
    for match in _ANCHOR_RELATIVE_WORDS_RE.finditer(content or ""):
        word = " ".join(match.group(1).lower().split())
        if word in _WORD_TO_DAYS:
            out.append(_WORD_TO_DAYS[word])
    return out


def _relative_tolerance_days(claimed_age_days: int) -> int:
    if claimed_age_days <= 7:
        return 2
    if claimed_age_days <= 30:
        return 5
    if claimed_age_days <= 90:
        return 10
    if claimed_age_days <= 180:
        return 15
    return 30


def check_date_precision(
    claimed_date: Optional[str],
    content: str,
    *,
    today: Optional[date] = None,
) -> str:
    """Return verified, approximate, year_only, or no_match."""
    try:
        dt = datetime.strptime(str(claimed_date).strip()[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return "no_match"

    year = dt.year
    month = dt.month
    day = dt.day
    year_str = str(year)
    content = strip_dynamic_boilerplate_dates(strip_copyright_founded_years(content or ""))
    content_lower = content.lower()

    iso_date = f"{year:04d}-{month:02d}-{day:02d}"
    if iso_date in content:
        return "verified"

    month_names = _MONTH_NAMES.get(month, ())
    day_str = str(day)
    day_padded = f"{day:02d}"
    for month_name in month_names:
        if re.search(rf"\b{month_name}\s+{day_str}\b[,]?\s*{year_str}", content_lower):
            return "verified"
        if day_padded != day_str and re.search(
            rf"\b{month_name}\s+{day_padded}\b[,]?\s*{year_str}",
            content_lower,
        ):
            return "verified"
        if re.search(rf"\b{day_str}\s+{month_name}\b[,]?\s*{year_str}", content_lower):
            return "verified"
        if day_padded != day_str and re.search(
            rf"\b{day_padded}\s+{month_name}\b[,]?\s*{year_str}",
            content_lower,
        ):
            return "verified"

    if re.search(rf"date\w*[\"']?\s*[:=]\s*[\"']?{re.escape(iso_date)}", content_lower):
        return "verified"

    slash_mdy = f"{month:02d}/{day:02d}/{year}"
    slash_dmy = f"{day:02d}/{month:02d}/{year}"
    if slash_mdy in content or slash_dmy in content:
        return "verified"

    clock = today or date.today()
    claimed_age = (clock - dt).days
    if claimed_age >= 0:
        implied = _extract_implied_ages_days(content)
        if implied:
            tolerance = _relative_tolerance_days(claimed_age)
            if any(abs(age - claimed_age) <= tolerance for age in implied):
                return "verified"

    month_year_found = False
    for month_name in month_names:
        if re.search(rf"\b{month_name}\s+{year_str}\b", content_lower):
            month_year_found = True
            break
    if f"{year:04d}-{month:02d}" in content:
        month_year_found = True
    if f"{month:02d}/{year}" in content:
        month_year_found = True
    if month_year_found:
        return "approximate" if day == 1 else "verified"

    if re.search(rf"\b{year_str}\b", content_lower):
        return "year_only"
    return "no_match"


def _source_domain_reason(source: str, url: str, company_website: str) -> Optional[str]:
    source_lower = (source or "").lower().strip()
    url_domain = extract_domain(url)
    url_host = _extract_hostname(url)

    allowed = _SOURCE_DOMAIN_ALLOWLIST.get(source_lower)
    if allowed and url_domain not in allowed and url_host not in allowed:
        return f"source {source_lower!r} does not match URL domain {url_domain!r}"

    suffixes = _SOURCE_HOST_SUFFIX_ALLOWLIST.get(source_lower, ())
    if suffixes and not any(url_host.endswith(suffix) for suffix in suffixes):
        company_domain = extract_domain(company_website)
        if company_domain and url_domain == company_domain:
            return f"company URL classified as {source_lower!r}; expected company_website"

    if source_lower in _COMPANY_DOMAIN_REQUIRED_SOURCES and company_website:
        company_domain = extract_domain(company_website)
        if company_domain and url_domain and url_domain != company_domain:
            return f"company source domain {url_domain!r} differs from company domain {company_domain!r}"
    return None


def _field(signal: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in signal:
            return signal[name]
    return None


def _add_find(
    findings: List[Finding],
    check_id: str,
    severity: str,
    message: str,
    **details: Any,
) -> None:
    findings.append(Finding(check_id, severity, message, dict(details)))


def run_l0_checks(
    signal: Dict[str, Any],
    snapshot: Dict[str, Any],
    *,
    today: Optional[date] = None,
    now: Optional[datetime] = None,
    snippet_overlap_threshold: float = 0.60,
    description_grounding_threshold: float = 0.25,
) -> L0Result:
    """Run deterministic L0 checks over a submitted signal and snapshot.

    Args:
        signal: Miner output signal dict. Expected keys include url, source,
            description, snippet, date, matched_icp_signal, and company_website.
        snapshot: Notary snapshot dict. Expected keys include url and content.
        today/now: Optional clocks for golden-vector determinism.
    """
    findings: List[Finding] = []
    metrics: Dict[str, Any] = {}

    url = str(_field(signal, "url") or "")
    source = str(_field(signal, "source") or "")
    description = str(_field(signal, "description") or "")
    snippet = str(_field(signal, "snippet") or "")
    content = str(_field(snapshot, "content", "normalized_text") or "")
    company_website = str(_field(signal, "company_website", "lead_company_website") or "")
    claim = str(_field(signal, "matched_icp_signal", "claim", "buyer_icp_signal") or description)
    signal_date = _field(signal, "date", "signal_date")
    content_found_date = _field(signal, "content_found_date")
    buyer_cap_days = _field(signal, "buyer_cap_days")

    reason = check_url_structural_validity(url)
    if reason:
        _add_find(findings, "url_structural_validity", "fail", reason, url=url)

    reason = _source_domain_reason(source, url, company_website)
    if reason:
        _add_find(findings, "domain_source_match", "fail", reason, source=source, url=url)

    snapshot_url = str(_field(snapshot, "url") or "")
    if snapshot_url and extract_domain(snapshot_url) != extract_domain(url):
        _add_find(
            findings,
            "snapshot_url_match",
            "fail",
            "snapshot URL domain differs from signal URL domain",
            signal_url=url,
            snapshot_url=snapshot_url,
        )

    reason = check_antibot_wall(content)
    if reason:
        _add_find(findings, "antibot_wall", "fail", reason)

    for field_name, text in (("description", description), ("snippet", snippet)):
        injected, excerpt = detect_prompt_injection(text)
        if injected:
            _add_find(
                findings,
                "prompt_injection",
                "fail",
                f"prompt-injection pattern in {field_name}",
                field=field_name,
                excerpt=excerpt,
            )

    is_generic, generic_reason = is_generic_intent_description(description)
    if is_generic:
        _add_find(
            findings,
            "generic_intent_description",
            "fail",
            "generic or templated intent description",
            reason=generic_reason,
        )

    overlap = compute_snippet_overlap(snippet, content)
    metrics["snippet_overlap"] = round(overlap, 6)
    if snippet and overlap < snippet_overlap_threshold:
        _add_find(
            findings,
            "snapshot_verbatim_snippet",
            "fail",
            "snippet is not sufficiently verbatim in the evidence snapshot",
            overlap=metrics["snippet_overlap"],
            threshold=snippet_overlap_threshold,
        )

    desc_grounding = check_description_grounding(description, content)
    metrics["description_grounding"] = round(desc_grounding, 6)
    if description and desc_grounding < description_grounding_threshold:
        _add_find(
            findings,
            "description_grounding",
            "fail",
            "description terms are not grounded in snapshot content",
            grounding=metrics["description_grounding"],
            threshold=description_grounding_threshold,
        )

    desc_grounded, desc_total, desc_ungrounded = check_signal_word_grounding(
        description, content
    )
    snip_grounded, snip_total, snip_ungrounded = check_signal_word_grounding(
        snippet, content
    )
    metrics["description_signal_words"] = {
        "grounded": desc_grounded,
        "total": desc_total,
        "ungrounded": desc_ungrounded,
    }
    metrics["snippet_signal_words"] = {
        "grounded": snip_grounded,
        "total": snip_total,
        "ungrounded": snip_ungrounded,
    }
    if desc_total and desc_grounded == 0:
        _add_find(
            findings,
            "description_signal_word_grounding",
            "fail",
            "description signal words are absent from snapshot content",
            ungrounded_words=desc_ungrounded,
        )
    if snip_total and snip_grounded == 0:
        _add_find(
            findings,
            "snippet_signal_word_grounding",
            "fail",
            "snippet signal words are absent from snapshot content",
            ungrounded_words=snip_ungrounded,
        )

    freshness_reason = check_evidence_freshness(
        claim,
        signal_date,
        content_found_date,
        buyer_cap_days if isinstance(buyer_cap_days, int) else None,
        now=now,
    )
    if freshness_reason:
        _add_find(
            findings,
            "freshness_window",
            "fail",
            freshness_reason,
            claim=claim,
            signal_date=signal_date,
        )

    if signal_date:
        date_status = check_date_precision(signal_date, content, today=today)
        metrics["date_precision"] = date_status
        if date_status in {"year_only", "no_match"}:
            _add_find(
                findings,
                "date_in_content",
                "fail",
                "claimed date is not precisely supported by snapshot content",
                signal_date=signal_date,
                date_status=date_status,
            )
        elif date_status == "approximate":
            _add_find(
                findings,
                "date_in_content",
                "warn",
                "claimed date is supported only approximately by snapshot content",
                signal_date=signal_date,
                date_status=date_status,
            )
    else:
        metrics["date_precision"] = "date_omitted"

    return L0Result(
        passed=not any(f.severity == "fail" for f in findings),
        findings=tuple(findings),
        metrics=metrics,
    )


def finding_ids(findings: Iterable[Finding]) -> List[str]:
    return [finding.check_id for finding in findings]
