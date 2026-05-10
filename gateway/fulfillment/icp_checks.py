"""
ICP matching checks for fulfillment leads.

Tier 1:  Deterministic, free — exact match + containment on miner's claimed values.
Tier 1.5: LLM-based — semantic sub-industry matching + location validation/geography.

Both tiers run before the expensive Tier 2 (LinkedIn, email, person verification)
to reject obvious mismatches early and save API cost.
"""

import ast
import asyncio
import logging
import os
import re
from typing import Any, List, Optional, Set, Tuple

import aiohttp

from gateway.fulfillment.models import FulfillmentLead, FulfillmentICP
from gateway.qualification.models import LeadOutput

logger = logging.getLogger(__name__)


def _coerce_industry_list(v: Any) -> List[str]:
    """Normalize industry/sub_industry input to a clean ``List[str]``.

    Tolerates four shapes:
      * ``None`` / ``""`` / ``[]``                  → ``[]``
      * ``List[str]`` (the canonical form)          → cleaned list
      * single ``str`` (legacy single-industry ICP) → ``[s]``
      * Python-repr stringified list ``"['X', 'Y']"`` (the bug shape that
        leaks in via legacy DB rows or CSV round-trips) → parsed list

    Used at every consumer of icp.industry / icp.sub_industry as a defense-
    in-depth measure: even if some upstream caller mis-serializes a list,
    the comparison still gets a clean ``List[str]``.
    """
    if v is None or v == "" or v == []:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
        except (ValueError, SyntaxError):
            pass
        # Last-resort: comma-split, strip surrounding brackets/quotes/whitespace.
        return [t.strip(" '\"[]") for t in s.split(",") if t.strip(" '\"[]")]
    return [s]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LLM_MODEL = "google/gemini-2.5-flash-lite"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_SUB_INDUSTRY_SEMANTIC_PROMPT = (
    'Industry: "{lead_industry}"\n'
    'Lead sub-industry: "{lead_sub_industry}"\n\n'
    'ICP target sub-industries: {icp_sub_industries}\n\n'
    'Does the lead\'s sub-industry mean the same thing as any target?\n'
    'Match on: spelling variations, abbreviations, singular/plural, '
    'hyphenation differences, or terms that refer to the same business category.\n'
    'Do NOT match unrelated categories even if they sound similar.\n\n'
    'Return JSON only: {{"match": true/false, "matched_sub_industry": "the matched target or empty"}}'
)

_LOCATION_CHECK_PROMPT = (
    'ICP target geography: "{icp_geography}"\n\n'
    'Lead HQ: city="{hq_city}", state="{hq_state}", country="{hq_country}"\n\n'
    '1. Is this a real, valid location? (city/state/country must exist)\n'
    '2. Is this location within the ICP\'s target geography?\n\n'
    'Return JSON only: {{"valid": true/false, "match": true/false}}'
)


# ---------------------------------------------------------------------------
# Country normalization
# ---------------------------------------------------------------------------

def _normalize_country(c: str) -> str:
    """Simple country alias normalization."""
    aliases = {
        "us": "united states", "usa": "united states", "u.s.": "united states",
        "u.s.a.": "united states", "uk": "united kingdom",
        "gb": "united kingdom", "great britain": "united kingdom",
    }
    c = c.strip().lower()
    return aliases.get(c, c)


# ---------------------------------------------------------------------------
# Fuzzy role matching
# ---------------------------------------------------------------------------

_ROLE_TITLE_EQUIVALENTS = {
    "vp": ["vp", "vice president", "v.p."],
    "svp": ["svp", "senior vice president", "senior vp"],
    "evp": ["evp", "executive vice president"],
    "director": ["director", "dir"],
    "head": ["head", "head of"],
    "cro": ["cro", "chief revenue officer"],
    "coo": ["coo", "chief operating officer"],
    "cmo": ["cmo", "chief marketing officer"],
    "cto": ["cto", "chief technology officer"],
    "cfo": ["cfo", "chief financial officer"],
    "ceo": ["ceo", "chief executive officer"],
    "cio": ["cio", "chief information officer"],
    "manager": ["manager", "mgr"],
    "gm": ["gm", "general manager"],
    "md": ["md", "managing director"],
}

_ROLE_FUNCTION_EQUIVALENTS = {
    "sales": ["sales", "revenue", "commercial", "business development", "gtm", "go-to-market", "go to market"],
    "marketing": ["marketing", "growth", "demand generation", "brand"],
    "engineering": ["engineering", "software engineering", "development", "r&d"],
    "product": ["product", "product management"],
    "operations": ["operations", "ops"],
    "hr": ["hr", "human resources", "people", "talent"],
    "finance": ["finance", "financial"],
    "it": ["it", "information technology"],
    "customer success": ["customer success", "client success", "cx"],
    "partnerships": ["partnerships", "alliances", "channel"],
}


def _normalize_role_tokens(role: str) -> set:
    """Break a role into normalized tokens, expanding equivalents."""
    role_lower = role.lower().strip()
    role_lower = re.sub(r'[/,&]+', ' ', role_lower)
    role_lower = re.sub(r'\s+of\s+', ' ', role_lower)
    role_lower = re.sub(r'\s+', ' ', role_lower).strip()

    tokens = set(role_lower.split())

    expanded = set()
    for token in tokens:
        for canonical, equivalents in _ROLE_TITLE_EQUIVALENTS.items():
            if token in equivalents:
                expanded.update(equivalents)
                break
        for canonical, equivalents in _ROLE_FUNCTION_EQUIVALENTS.items():
            if token in equivalents:
                expanded.update(equivalents)
                break

    return tokens | expanded


def _fuzzy_role_match(lead_role: str, target_roles: list) -> bool:
    """Check if lead_role is a fuzzy match for any target role."""
    if not lead_role or not target_roles:
        return False

    lead_tokens = _normalize_role_tokens(lead_role)

    for target in target_roles:
        target_tokens = _normalize_role_tokens(target)

        lead_titles = set()
        lead_functions = set()
        target_titles = set()
        target_functions = set()

        for token in lead_tokens:
            for _, equivs in _ROLE_TITLE_EQUIVALENTS.items():
                if token in equivs:
                    lead_titles.add(token)
            for _, equivs in _ROLE_FUNCTION_EQUIVALENTS.items():
                if token in equivs:
                    lead_functions.add(token)

        for token in target_tokens:
            for _, equivs in _ROLE_TITLE_EQUIVALENTS.items():
                if token in equivs:
                    target_titles.add(token)
            for _, equivs in _ROLE_FUNCTION_EQUIVALENTS.items():
                if token in equivs:
                    target_functions.add(token)

        title_overlap = bool(lead_titles & target_titles)
        function_overlap = bool(lead_functions & target_functions)

        if title_overlap and function_overlap:
            return True

        overlap = lead_tokens & target_tokens
        min_size = min(len(lead_tokens), len(target_tokens))
        if min_size > 0 and len(overlap) / min_size >= 0.5:
            return True

    return False


# ---------------------------------------------------------------------------
# Tier 1: ICP Fit Gate (free, deterministic)
# ---------------------------------------------------------------------------

def tier1_check(
    lead: FulfillmentLead,
    lead_output: LeadOutput,
    icp: FulfillmentICP,
    seen_companies: Set[str],
) -> Optional[str]:
    """
    Return failure_reason string if the lead fails any ICP check, else None.
    Returns "sub_industry_needs_llm" if sub-industry needs Tier 1.5 LLM check.
    """
    allowed_inds = _coerce_industry_list(icp.industry)
    if allowed_inds:
        if lead.industry not in allowed_inds:
            return "industry_mismatch"

    allowed_subs = _coerce_industry_list(icp.sub_industry)
    if allowed_subs:
        if lead.sub_industry not in allowed_subs:
            # Try containment match before flagging for LLM
            lead_sub_lower = lead.sub_industry.lower().strip()
            containment_match = False
            for allowed in allowed_subs:
                allowed_lower = allowed.lower().strip()
                if allowed_lower in lead_sub_lower or lead_sub_lower in allowed_lower:
                    containment_match = True
                    break
            if not containment_match:
                return "sub_industry_needs_llm"

    if icp.excluded_companies and lead.business:
        excluded_keys = {c.strip().lower() for c in icp.excluded_companies if c and c.strip()}
        if lead.business.strip().lower() in excluded_keys:
            return "company_excluded"

    if icp.target_role_types and lead.role_type not in icp.target_role_types:
        return "role_type_mismatch"

    if icp.target_roles and lead.role not in icp.target_roles:
        if not _fuzzy_role_match(lead.role, icp.target_roles):
            return "role_mismatch"

    if icp.target_seniority:
        try:
            from gateway.qualification.models import Seniority
            lead_sen = lead_output.seniority.value if hasattr(lead_output.seniority, "value") else str(lead_output.seniority)
            try:
                target_sen = Seniority(icp.target_seniority).value
            except (ValueError, KeyError):
                target_sen = icp.target_seniority
            if lead_sen.lower() != target_sen.lower():
                return "seniority_mismatch"
        except Exception:
            return "seniority_mismatch"

    # Multi-country support: ``icp.country`` is a ``List[str]`` (the field
    # validator coerces legacy single-string values to ``[str]``).  An empty
    # list means "any country accepted" and skips the check.  Otherwise the
    # lead's HQ country must match ANY listed target after alias
    # normalization (so "US" / "USA" / "United States" all collide).
    if icp.country and lead.company_hq_country:
        targets = icp.country if isinstance(icp.country, list) else [icp.country]
        target_set = {_normalize_country(c) for c in targets if c}
        if target_set and _normalize_country(lead.company_hq_country) not in target_set:
            return "country_mismatch"

    if icp.employee_count and lead.employee_count:
        allowed = icp.employee_count if isinstance(icp.employee_count, list) else [icp.employee_count]
        if lead.employee_count not in allowed:
            return "employee_count_mismatch"

    if icp.company_stage and lead_output.role:
        pass

    biz_lower = lead_output.business.strip().lower()
    if not biz_lower:
        return "data_quality"
    if biz_lower in seen_companies:
        return "duplicate_company"
    seen_companies.add(biz_lower)

    return None


# ---------------------------------------------------------------------------
# Tier 1.5: LLM-based checks
# ---------------------------------------------------------------------------

def _get_openrouter_key() -> str:
    return (
        os.environ.get("FULFILLMENT_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_KEY", "")
    )


async def _llm_call(prompt: str, or_key: str) -> Optional[dict]:
    """Make a single LLM call with 3-retry logic. Returns parsed JSON or None."""
    import json as _json

    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    _OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {or_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": _LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 50,
                        "temperature": 0,
                    },
                ) as resp:
                    if resp.status == 429 and attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                        return None
                    body = await resp.json()
                    content = body["choices"][0]["message"]["content"]
                    m = re.search(r"\{[^}]+\}", content)
                    if m:
                        return _json.loads(m.group())
                    if attempt < 2:
                        continue
                    return None
        except Exception:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            return None
    return None


async def semantic_sub_industry_match(
    lead_industry: str,
    lead_sub_industry: str,
    icp_sub_industries: list,
) -> Tuple[bool, str]:
    """Semantic sub-industry match via LLM. Returns (matched, matched_sub_industry)."""
    or_key = _get_openrouter_key()
    if not or_key:
        return False, ""

    safe_ind = lead_industry.replace("{", "{{").replace("}", "}}")
    safe_sub = lead_sub_industry.replace("{", "{{").replace("}", "}}")
    subs_str = ", ".join(f'"{s}"' for s in icp_sub_industries)

    prompt = _SUB_INDUSTRY_SEMANTIC_PROMPT.format(
        lead_industry=safe_ind,
        lead_sub_industry=safe_sub,
        icp_sub_industries=subs_str,
    )

    result = await _llm_call(prompt, or_key)
    if result:
        matched = bool(result.get("match", False))
        matched_sub = str(result.get("matched_sub_industry", ""))
        return matched, matched_sub
    return False, ""


async def validate_lead_geography(
    hq_city: str,
    hq_state: str,
    hq_country: str,
    icp_geography: str,
) -> Tuple[bool, bool]:
    """Location validation + geography match via LLM. Returns (valid, match)."""
    or_key = _get_openrouter_key()
    if not or_key:
        return False, False

    safe_city = (hq_city or "").replace("{", "{{").replace("}", "}}")
    safe_state = (hq_state or "").replace("{", "{{").replace("}", "}}")
    safe_country = (hq_country or "").replace("{", "{{").replace("}", "}}")
    safe_geo = icp_geography.replace("{", "{{").replace("}", "}}")

    prompt = _LOCATION_CHECK_PROMPT.format(
        icp_geography=safe_geo,
        hq_city=safe_city,
        hq_state=safe_state,
        hq_country=safe_country,
    )

    result = await _llm_call(prompt, or_key)
    if result:
        valid = bool(result.get("valid", False))
        match = bool(result.get("match", False))
        return valid, match
    return False, False
