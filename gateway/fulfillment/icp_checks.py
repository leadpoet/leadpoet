"""
ICP matching checks for fulfillment leads.

Tier 1:  Deterministic, free — exact match + containment on miner's claimed values.
Tier 1.5: LLM-based — semantic sub-industry matching + location validation/geography.

Both tiers run before the expensive Tier 2 (LinkedIn, email, person verification)
to reject obvious mismatches early and save API cost.
"""

import ast
import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

from gateway.fulfillment.models import FulfillmentLead, FulfillmentICP
from gateway.qualification.models import LeadOutput

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Industry equivalence map — Tier 1 fallback for parent-industry synonyms
# ─────────────────────────────────────────────────────────────────────
# Loaded once from gateway/utils/industry_equivalence.json.  Each class is a
# set of parent industries treated as equivalent for the Tier 1 industry-match
# check.  Behavior is purely additive: if the direct membership check
# (``lead.industry in icp.industry``) already passes, this map is never
# consulted.  If the direct check fails, the equivalence map gets a chance.
# JSON missing or malformed → empty map → no behavior change vs. pre-change.
# ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_industry_equivalence_map() -> Dict[str, Set[str]]:
    """Build ``{industry: set_of_equivalent_industries}`` from the JSON.

    An industry MAY appear in multiple equivalence classes; the per-industry
    set is the UNION of all classes containing it.  This intentionally avoids
    transitive matching across classes: if Software is in class1 with Apps
    AND class2 with Privacy/Security, Apps still does NOT auto-match
    Privacy/Security (only Software does).  Each class is self-contained.

    Returns:
        ``{}`` if JSON file missing or malformed (silent fail-open — the
        layer is additive, never load-bearing).  Otherwise the lookup map.
    """
    path = Path(__file__).resolve().parent.parent / "utils" / "industry_equivalence.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"industry_equivalence.json malformed: {e} — falling back to empty map")
        return {}
    out: Dict[str, Set[str]] = {}
    for cls in data.get("equivalence_classes", []):
        members = set(cls)
        for m in cls:
            out.setdefault(m, set()).update(members)
    return out


def _industry_in_equivalence_class(lead_industry: str, allowed_industries: List[str]) -> bool:
    """Return True if ``lead_industry`` shares an equivalence class with any
    of ``allowed_industries``.  Used as Tier 1 fallback AFTER the direct
    membership check fails.  Empty equivalence map → always False.
    """
    if not lead_industry or not allowed_industries:
        return False
    equiv = _load_industry_equivalence_map()
    lead_class = equiv.get(lead_industry)
    if not lead_class:
        return False
    return any(ai in lead_class for ai in allowed_industries)


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
    """Country alias normalization for the Tier 1 country gate.

    Delegates to gateway.utils.geo_normalize.normalize_country so the full
    alias map (ISO-2, ISO-3, common variants for all 199 supported
    countries) is the single source of truth. Returns lowercase for
    case-insensitive equality against the lead's normalized HQ country.
    """
    if not c:
        return ""
    from gateway.utils.geo_normalize import normalize_country as _gn
    return _gn(c).lower()


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


_NON_ENGLISH_ROLE_RE = re.compile(
    r"[Ͱ-Ͽ"
    r"Ѐ-ӿ"
    r"֐-׿"
    r"؀-ۿ"
    r"　-ヿ"
    r"一-鿿"
    r"가-힯]"
    r"|[áéíóúñçãõâêîôûäëïöüàèìòùýÁÉÍÓÚÑÇÃÕÂÊÎÔÛÄËÏÖÜÀÈÌÒÙÝ"
    r"ßæøåłąęćśźżńÆØÅŁŚŹŻŃœŒ]"
    r"|\b("
    r"de|la|el|los|las|para|por|con|sin|y|o|ou|et|ed|"
    r"do|da|dos|das|com|sem|"
    r"du|le|les|des|avec|sans|"
    r"der|die|das|und|mit|ohne|"
    r"di|della|delle|degli|"
    r"van|den|het|en|"
    r"comercial|mercadeo|ventas|gerente|jefe|presidente|"
    r"ingeniero|desarrollo|desarrollador|tecnologia|"
    r"coordinador|vicepresidente|encargado|auxiliar|asistente|"
    r"analista|subdirector|"
    r"diretor|diretora|desenvolvimento|atendimento|programador|coordenador|"
    r"directeur|directrice|gestion|ingenieur|developpement|coordinateur|"
    r"direttore|sviluppo|gestione|responsabile|coordinatore|sottodirettore"
    r")\b",
    re.IGNORECASE,
)


def _looks_definitely_english(text: str) -> bool:
    if not text:
        return True
    return not _NON_ENGLISH_ROLE_RE.search(text)


def _fuzzy_role_match(lead_role: str, target_roles: list) -> bool:
    """Legacy fuzzy match.  Returns True if the role would have been
    accepted under the pre-2026-06-03 logic (title+function OR 50%
    token overlap).  Retained for backward compatibility; new code
    should use ``classify_role`` instead.
    """
    verdict = classify_role(lead_role, target_roles)
    return verdict in ("strict_match", "gray_zone")


def classify_role(lead_role: str, target_roles: list) -> str:
    """Three-way classification of how a lead's role relates to the buyer's
    target_roles list.

      "strict_match"  — title-overlap AND function-overlap on at least one
                        target role.  High-confidence accept, no LLM needed.
      "gray_zone"     — neither title+function overlap nor exact match, but
                        raw token overlap with some target is >= 50%.
                        Used to be auto-accepted (Path 2); now routed
                        through batched LLM judgment in
                        ``score_fulfillment_batch``'s pre-pass.
      "no_match"      — fails both Path 1 and the 50% overlap check.
                        Deterministically reject — same outcome as today.

    See bug audit 2026-06-03: Path 2 (50% raw token overlap) was producing
    ~30 of ~78 FPs across the five Revamped labels — accepting CTO against
    a Sales target, Executive Assistant to CEO against a CEO target, etc.
    Routing the gray zone through an LLM judge keeps legitimate variants
    that share tokens-but-mean-the-same-thing while rejecting tokens-but-
    different-function cases.
    """
    if not lead_role or not target_roles:
        return "no_match"

    lead_tokens = _normalize_role_tokens(lead_role)

    gray_zone_hit = False
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

        # Path 1: high-confidence accept.
        if title_overlap and function_overlap:
            return "strict_match"

        # Path 2 detection (used to auto-accept; now flags the gray zone).
        if not gray_zone_hit:
            overlap = lead_tokens & target_tokens
            min_size = min(len(lead_tokens), len(target_tokens))
            if min_size > 0 and len(overlap) / min_size >= 0.5:
                gray_zone_hit = True
        # Continue scanning — a later target may yield a strict_match.

    return "gray_zone" if gray_zone_hit else "no_match"


def _role_decision_key(role: str) -> str:
    """Return the stable cache key for a role-batch judgment."""
    return " ".join((role or "").split()).casefold()


# ---------------------------------------------------------------------------
# Tier 1: ICP Fit Gate (free, deterministic)
# ---------------------------------------------------------------------------

def tier1_check(
    lead: FulfillmentLead,
    lead_output: LeadOutput,
    icp: FulfillmentICP,
    seen_companies: Set[str],
    role_decisions: Optional[dict] = None,
) -> Optional[str]:
    """
    Return failure_reason string if the lead fails any ICP check, else None.
    Returns "sub_industry_needs_llm" if sub-industry needs Tier 1.5 LLM check.

    ``role_decisions`` is an optional ``{normalized_role: bool}`` cache
    pre-populated by ``score_fulfillment_batch``'s LLM pre-pass.  When
    provided, judged leads accept iff the cache says True.  When absent
    (e.g., called outside a batch), they fail-closed → ``role_mismatch``.
    """
    allowed_inds = _coerce_industry_list(icp.industry)
    if allowed_inds:
        if lead.industry not in allowed_inds:
            # Fallback: industry equivalence map (e.g., Software ≈ IT, Real
            # Estate ≈ Physical Infrastructure).  See
            # gateway/utils/industry_equivalence.json.  Purely additive — if
            # JSON missing or no class matches, behaves exactly as the
            # original hard check.
            if not _industry_in_equivalence_class(lead.industry, allowed_inds):
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
        from gateway.fulfillment.normalize import normalize_company
        excluded_simple: Set[str] = set()
        excluded_norm: Set[str] = set()
        for c in icp.excluded_companies:
            if not c or not c.strip():
                continue
            excluded_simple.add(c.strip().lower())
            n = normalize_company(c)
            if n:
                excluded_norm.add(n)
        biz_simple = lead.business.strip().lower()
        biz_norm = normalize_company(lead.business)
        if biz_simple in excluded_simple or (biz_norm and biz_norm in excluded_norm):
            return "company_excluded"

    if icp.target_role_types and lead.role_type not in icp.target_role_types:
        return "role_type_mismatch"

    if icp.target_roles:
        if not _looks_definitely_english(lead.role or ""):
            return "role_not_english"
        if lead.role not in icp.target_roles:
            verdict = classify_role(lead.role, icp.target_roles)
            if verdict == "no_match":
                return "role_mismatch"
            # Both "strict_match" (Path 1 lexicon overlap) and "gray_zone"
            # (Path 2 50% token overlap) defer to the LLM cache populated
            # by score_fulfillment_batch's pre-pass.  See classify_role's
            # docstring for why strict_match isn't trusted on its own.
            # Missing cache (e.g., called outside the batch path) or
            # explicit False → reject (fail-closed).
            role_key = _role_decision_key(lead.role)
            if not (role_decisions and role_decisions.get(role_key) is True):
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

    # Tier 1a — COMPANY country gate.  ``icp.company_country`` is a
    # ``List[str]`` (the field validator coerces legacy single-string
    # values).  An empty list = "any company country accepted" → check
    # skipped.  Otherwise lead.company_hq_country must match ANY listed
    # target after alias normalization (US / USA / United States all
    # collide via gateway/utils/geo_normalize).
    #
    # Fail-closed when the buyer specified company_country but the lead
    # didn't carry a company HQ country: the missing data prevents any
    # meaningful verification, so reject.  (Previously the check was
    # skipped silently when lead.company_hq_country was empty, letting
    # half-populated leads slip through.)
    if icp.company_country:
        if not lead.company_hq_country:
            return "company_location_missing"
        targets = icp.company_country
        target_set = {_normalize_country(c) for c in targets if c}
        if target_set and _normalize_country(lead.company_hq_country) not in target_set:
            return "country_mismatch"

    # Tier 1b — CONTACT country gate.  Symmetric to 1a but reads the
    # PERSON-level country from the lead (``lead.country`` — set by the
    # miner from LinkedIn ``location`` parsing).  Most buyers won't set
    # this; only person-targeting ICPs (e.g. Adedeji 5's retirement-IUL
    # campaign) need it.
    if icp.contact_country:
        if not lead.country:
            return "contact_location_missing"
        targets = icp.contact_country
        target_set = {_normalize_country(c) for c in targets if c}
        if target_set and _normalize_country(lead.country) not in target_set:
            return "contact_country_mismatch"

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
