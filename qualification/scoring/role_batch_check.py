"""Batched LLM role-match judge.

Called from ``gateway/fulfillment/scoring.py::score_fulfillment_batch`` as
a pre-pass over leads whose role classification fell into the gray zone
(Path 2 token-overlap, no Path 1 title+function match).  See bug audit
2026-06-03: the previous behaviour was to auto-accept the gray zone,
which produced ~30 false positives across the five Revamped labels
(CTO vs Sales-CRO target, Executive Assistant to CEO vs CEO target, etc.).

Design contract:
  - Sequential per-chunk (no concurrency) — keeps debugging tractable
    and avoids bursty 429s.
  - ``BATCH_SIZE`` leads per LLM call (default 10).
  - Per-chunk retry on 429 / 5xx / timeout / malformed JSON
    (``NUM_RETRIES`` attempts with linear backoff).
  - Fail-closed: chunk-level error after retries → every lead in the
    chunk is recorded as ``False``.  Same conservative direction as
    deleting Path 2 entirely.
  - Missing ``OPENROUTER_KEY`` → all gray-zone leads recorded as ``False``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

MODEL = "google/gemini-2.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TIMEOUT_SECONDS = 30
NUM_RETRIES = 3
BATCH_SIZE = 10  # leads per LLM call, per user spec

_SYS_MESSAGE = "You are a strict B2B role-match judge. Return JSON only."

_PROMPT_TEMPLATE = """Judge whether each LEAD's role matches ANY of the buyer's TARGET ROLES.

TARGET ROLES (what the buyer wants):
{target_roles_list}

LEADS to judge ({n} total):
{leads_block}

DEFAULT TO REJECT.  Accept only when the lead's role clearly belongs to
the same function family AND the same seniority bucket as at least one
target.  Title-word overlap (CTO vs CRO, VP Eng vs VP Sales) does NOT
imply a match — function and seniority must both line up.

==== NORMALIZATION (apply BEFORE every rule below) ====

Treat the following C-level abbreviations as EQUIVALENT to their full
forms in BOTH the candidate's title AND the target_roles list.  This
applies before R0/R1/F1/F2 evaluation — when the candidate's title says
"COO" and a target says "Chief Operating Officer", they are the SAME
string for all rule purposes.  Match the abbreviation only when it
appears as a standalone token (separated by spaces, slashes, commas,
or other separators) — not as part of an unrelated word.

  CEO  ↔ Chief Executive Officer
  COO  ↔ Chief Operating Officer
  CTO  ↔ Chief Technology Officer
  CIO  ↔ Chief Information Officer
  CFO  ↔ Chief Financial Officer
  CMO  ↔ Chief Marketing Officer
  CRO  ↔ Chief Revenue Officer (NOT Conversion Rate Optimization —
                                see EDGE CASE GUIDANCE on marketing CROs)
  CSO  ↔ Chief Sales Officer (or Chief Strategy Officer — judge by context)
  CCO  ↔ Chief Customer Officer (or Chief Commercial Officer)
  CPO  ↔ Chief Product Officer (or Chief People Officer — judge by context)
  CDO  ↔ Chief Data Officer (or Chief Digital Officer — judge by context)
  CISO ↔ Chief Information Security Officer
  CHRO ↔ Chief Human Resources Officer

  VP   ↔ Vice President
  SVP  ↔ Senior Vice President
  EVP  ↔ Executive Vice President
  AVP  ↔ Associate / Assistant Vice President  (lower seniority — see SENIORITY BUCKETS)

  Director equivalents:
    Dir, Sr Dir, Sr Director, Senior Director  ↔ Director

  Functional acronyms in titles (treat as the full noun):
    BD   = Business Development
    GTM  = Go-to-Market
    RevOps = Revenue Operations
    SalesOps = Sales Operations
    IT   = Information Technology

EXAMPLE OF APPLYING NORMALIZATION:
  Candidate: "COO | Operations Director | Operational Excellence | CX |"
  Target:    "Chief Operating Officer"
  → After NORMALIZATION, candidate's primary segment "COO" reads as
    "Chief Operating Officer", which EXACTLY equals the target.
  → R1's "primary role is first segment" still applies — primary is
    still the FIRST segment ("COO" → "Chief Operating Officer"), the
    other segments are still secondary.
  → F1 fires (target string equals primary string after normalization) → ACCEPT.

==== ABSOLUTE REJECTS (apply BEFORE fast-path accept rules) ====

REJECT R0 — NO-LONGER-IN-ROLE markers (case-insensitive, anywhere
in the title): "former", "formerly", "ex-", "ex ", "retired",
"past", "previous", "previously", "alum", "alumna", "alumnus".
These ALWAYS reject regardless of any other rule.
   REJECT: "Former President and CEO", "CEO at Acme - Retired",
           "Ex-CRO, now Advisor", "Previously VP Sales"

REJECT R1 — PRIMARY-ROLE MUST BE THE TARGET:
A target token appearing in a multi-function laundry-list title does
NOT count as the candidate's primary role. If the title is a long
list separated by "|", "/", "♦", "•", "·", "—", "-", or commas, the
candidate's PRIMARY job is the FIRST listed role (before the first
separator).  Apply F1/F2 only against the PRIMARY role segment.
   REJECT: "Founder & CEO ♦ Client Advisory Services ♦ Bookkeeping |
           Payroll | HR | Recruiting" → primary is "Founder & CEO" of
           a bookkeeping/HR/recruiting firm; not a sales-target match.
   REJECT: "Marketing Director / Sales Director" if target is "Sales
           Director" → primary is Marketing (listed first).
   ACCEPT: "President & Chief Revenue Officer" → both segments are
           target-role-equivalent; primary segment "President & Chief
           Revenue Officer" contains "Chief Revenue Officer".
   ACCEPT: "VP of Sales, Prime Accounts" → "Prime Accounts" is a
           qualifier on the role, not a separate role.

==== FAST-PATH ACCEPT RULES (apply AFTER absolute rejects) ====

RULE F1 — TARGET-SUPERSET MATCH (qualifier additions):
If the candidate's title CONTAINS every word of any target role
(case-insensitive; ignore connectors "and / & / | / , / of / the / a /
to"), accept — provided the extra qualifier is a REGION, INDUSTRY
DOMAIN, SECONDARY FUNCTION, or SECONDARY TITLE, NOT a SENIORITY OR
FUNCTION CHANGE.
   ACCEPT examples (qualifier is region/domain/secondary function/co-title):
     target "VP of Sales"           ← "VP of Sales & Marketing"
     target "VP of Sales"           ← "Vice President of Security Sales"
     target "VP of Sales"           ← "Vice President of Sales, Prime Accounts"
     target "Chief Revenue Officer" ← "President & Chief Revenue Officer"
     target "Chief Revenue Officer" ← "Founder & Chief Revenue Officer"
     target "Director of Sales"     ← "Sales Director, West"
     target "Director of Sales"     ← "Sales Director, Southwest U.S."
     target "Head of Sales"         ← "Head of Sales and Business Development"
     target "Director of Business Development" ← "Director of Business Development and Marketing"
   REJECT examples (qualifier changes seniority or function):
     target "VP of Sales"           ✗ "Assistant VP of Sales"            (lower seniority)
     target "VP of Sales"           ✗ "VP of Sales Engineering"          (eng = different function)
     target "VP of Sales"           ✗ "VP Sales Trainee"                 (Trainee = junior IC)
     target "VP of Sales"           ✗ "Sales Manager"                    (lower seniority)
     target "Chief Revenue Officer" ✗ "Marketing CRO"                    (Conversion Rate Optimization, not Chief Revenue Officer)
     target "Chief Revenue Officer" ✗ "Chief Revenue Officer's EA"       (EA, not CRO)
   PRECEDENCE: F1 wins over the "industry-specific role" reject rule
   (line "Industry-specific role when ICP doesn't require...") because
   the rejected industry qualifier here is attached to a TARGET ROLE,
   not the candidate's primary job.
   *** CRITICAL: If a candidate's primary segment, after NORMALIZATION,
   EXACTLY contains every word of a target role, ACCEPT regardless of
   trailing industry qualifiers — "Chief Operating Officer - Banking
   and Insurance" matches target "Chief Operating Officer" even though
   "Banking" and "Insurance" are industry qualifiers.  Do not invoke
   the industry-mismatch reject rule when F1 fires. ***

RULE F2 — SALES-FAMILY EQUIVALENT (when ANY target is sales-family):
If the target_roles list contains ANY role with the words "Sales",
"Revenue", "Business Development", "BD", "GTM", "Go-to-Market",
"Partnerships", "Alliances", "Channel", "Account Management", or
"Strategic Accounts", then a candidate whose PRIMARY function is in
this same family ACCEPTS — provided the candidate's seniority bucket
(C-level / VP / Director / Head-of) is also present in the target list.
   ACCEPT examples:
     targets include "VP of Sales"                  ← "Vice President of GTM Operations"
     targets include "Director of Sales"            ← "Director Of Business Development"
     targets include "Head of Sales"                ← "Head Of Business Development"
     targets include "Director of Sales"            ← "Director of Business Development and Marketing"
     targets include "VP of Sales"                  ← "VP Sales & Partnerships"
   STILL REJECT (seniority mismatch):
     targets include only VP/Director ✗ "Business Development Manager"     (Manager < Director)
     targets include only VP/Director ✗ "BDR"                              (junior IC)
   STILL REJECT (function mismatch):
     targets include only "VP of Sales" ✗ "Director of Revenue Marketing"  (Marketing primary, not Sales-rev)
     targets include only "VP of Sales" ✗ "Head of Partnerships & Growth" (Partnerships+Growth fine BUT verify it's revenue-aligned)

==== END FAST-PATH RULES ====

==== FUNCTION FAMILIES ====

SALES / REVENUE / BD / PARTNERSHIPS family (treat as one):
  Sales, Revenue, Commercial, Business Development, GTM, Go-to-Market,
  Growth (when sales-aligned), Partnerships, Alliances, Channel, Account
  Management, Strategic Accounts.

SALES-OPS / REV-OPS family (treat as one):
  Sales Operations, Revenue Operations, RevOps, GTM Operations, GTM Ops,
  Sales Strategy & Operations, Revenue Strategy.

DIFFERENT FUNCTIONS — do NOT match a Sales/RevOps target:
  Engineering, Software Engineering, R&D, DevOps, Site Reliability,
  Product Management, Product Design, Design, Research, Data Science,
  Marketing (when brand/comms only, no revenue lens), Finance, Treasury,
  Accounting, Legal, Compliance, Risk, HR, People, Talent, Recruiting,
  IT, Security, Infrastructure, Real Estate, Facilities, Operations
  (general — manufacturing, supply chain, mining, construction; NOT
  sales/revenue ops), Customer Support (purely reactive), Procurement.

==== SENIORITY BUCKETS ====

  C-level    : CEO, CRO, CSO, CCO, CMO, CFO, COO, CTO, CIO, Chief X Officer
  VP-level   : VP, SVP, EVP, AVP, "Vice President", "Sr. Vice President"
  Director   : Director, Sr Director, Senior Director, Dir
  Head of X  : Head of <function> ≈ VP-level OR Director-level
  Manager    : Manager (without VP/Director/Senior prefix) — does NOT
               match VP+ or Director+ targets
  Junior/IC  : Account Executive, SDR, BDR, Analyst, Associate, Specialist,
               Coordinator, Representative — does NOT match VP+/Director+

==== ALWAYS REJECT ====

  - Executive Assistant to <X>, EA to <X>, Personal Assistant, Assistant to <X>
  - Chief of Staff to <X> (even when X is a target)
  - Reports to <X>, Supporting the <X>, Working with the <X>
  - Board Director, Advisor, Non-Executive Director, Board Member
    (when target is the operating role, e.g., CEO/CRO)
  - Country-specific BD/Sales role when target_roles list is global AND
    the country isn't on the buyer's stated geography (judge from
    context; absent that, accept if function matches)
  - Industry-specific role when ICP doesn't require that industry vertical

==== FUNCTIONAL EQUIVALENTS — accept these as matches ====

  Head of Revenue          = CRO
  Chief Sales Officer      = CRO   (same function family)
  Chief Customer Officer   = CRO   IF function = revenue/growth (judge by role detail)
  Commercial Director      = Director of Sales
  Sales Director           = Director of Sales
  GTM Lead / Head of GTM   = Sales/Revenue leader
  Sales Operations         = Revenue Operations = RevOps = GTM Operations
  Director of RevOps       = Director of Revenue Operations
  Partnerships VP          = VP of Partnerships
  Head of <X>              = VP of X OR Director of X (when function matches)

==== EDGE CASE GUIDANCE ====

  "Sales Engineering"      — this is pre-sales/solutions consulting, NOT
                             individual contributor sales.  Accept iff
                             target list includes "Sales Engineering"
                             explicitly OR a generic "Sales" VP/Director
                             target (interpret broadly).
  "Customer Success"       — reactive support → reject.  Expansion-led
                             (renewals, upsell quota) → accept if target
                             includes Sales/Revenue.
  "Growth Marketing"       — accept if target includes Growth/Marketing
                             AND seniority matches; reject if target is
                             explicit sales (Sales VP).
  "VP Strategic Development" — corporate dev / M&A focus → reject for
                             sales targets unless role clearly = BD.
  "Sales Development"      — SDR/BDR organisation management.  Roles
                             like "Director of Sales Development" /
                             "Head of Sales Development" / "VP of Sales
                             Development" are SALES family — accept for
                             sales VP/Director targets.  Do NOT confuse
                             with "Business Development" (which is also
                             sales family — both accept).
  Acronym ambiguity        — "CRO" usually means Chief Revenue Officer
                             but in MARKETING context can mean Conversion
                             Rate Optimization.  When role title contains
                             "Marketing CRO" or "Digital Marketing CRO" or
                             "Marketing & CRO", treat as marketing
                             specialist and REJECT for sales-family
                             targets unless target list explicitly
                             includes a marketing role.

WHEN UNCERTAIN, REJECT.

OUTPUT — JSON array of length {n} in the SAME ORDER as input:
[{{"id": <id>, "match": true|false, "reason": "<<=120 chars>"}}, ...]
Do not return anything else."""


def _build_prompt(target_roles: List[str], chunk: List[Dict[str, Any]]) -> str:
    leads_block = "\n".join(
        f"  {i+1}. id={l['id']} role={l['role']!r}" for i, l in enumerate(chunk)
    )
    target_block = "\n".join(f"  - {t}" for t in target_roles)
    return _PROMPT_TEMPLATE.format(
        target_roles_list=target_block,
        leads_block=leads_block,
        n=len(chunk),
    )


async def _judge_chunk(
    http: httpx.AsyncClient,
    api_key: str,
    target_roles: List[str],
    chunk: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Judge one chunk with retry.  Returns parsed list, or None on failure."""
    prompt = _build_prompt(target_roles, chunk)
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": _SYS_MESSAGE},
            {"role": "user",   "content": prompt},
        ],
        "provider": {
            "data_collection": "deny",
            "zdr": True,
        },
    }
    last_err = "retries_exhausted"
    for attempt in range(NUM_RETRIES):
        try:
            r = await http.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=TIMEOUT_SECONDS,
            )
            if r.status_code == 429:
                wait_s = 2 * (attempt + 1)
                logger.warning(
                    "role-batch 429 rate-limited attempt=%d/%d sleeping=%ds",
                    attempt + 1, NUM_RETRIES, wait_s,
                )
                await asyncio.sleep(wait_s)
                last_err = "rate_limited_429"
                continue
            if r.status_code in (500, 502, 503, 504):
                wait_s = 1 + attempt
                logger.warning(
                    "role-batch HTTP %s attempt=%d/%d sleeping=%ds",
                    r.status_code, attempt + 1, NUM_RETRIES, wait_s,
                )
                await asyncio.sleep(wait_s)
                last_err = f"http_{r.status_code}"
                continue
            if r.status_code != 200:
                logger.warning(
                    "role-batch HTTP %s body=%s — fail-closed",
                    r.status_code, r.text[:200],
                )
                return None
            content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
            parsed = _parse_response(content)
            if parsed is not None and len(parsed) == len(chunk):
                return parsed
            logger.warning(
                "role-batch parse / length-mismatch (got=%s expected=%d) — retrying",
                None if parsed is None else len(parsed), len(chunk),
            )
            last_err = "parse_or_length_mismatch"
            continue
        except (httpx.TimeoutException, httpx.HTTPError) as e:
            logger.warning(
                "role-batch %s attempt=%d/%d", type(e).__name__,
                attempt + 1, NUM_RETRIES,
            )
            last_err = type(e).__name__
            await asyncio.sleep(1)
            continue
        except ValueError as e:
            logger.warning("role-batch JSON decode error: %s", e)
            last_err = "json_decode_error"
            continue
    logger.warning("role-batch %s after %d attempts — chunk fail-closed",
                   last_err, NUM_RETRIES)
    return None


def _parse_response(content: str) -> Optional[List[Dict[str, Any]]]:
    """Extract the JSON array from an LLM response.  Tolerates code-fences
    and trailing prose."""
    if not content:
        return None
    try:
        v = json.loads(content)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", content)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        if isinstance(v, list):
            return v
    except Exception:
        return None
    return None


async def batch_check(
    leads: List[Dict[str, Any]],
    target_roles: List[str],
    api_key: Optional[str] = None,
) -> Dict[Any, bool]:
    """Run the LLM role judge over ``leads`` (the gray-zone set).

    Args:
        leads: list of ``{"id": <any hashable>, "role": <str>}``.
        target_roles: ICP's target_roles list.
        api_key: override; defaults to ``OPENROUTER_KEY`` env.

    Returns ``{id: bool}`` — True iff LLM judged role matches a target.
    Missing or failed leads default to False (fail-closed).
    """
    if not leads or not target_roles:
        return {}

    key = (
        api_key
        or os.environ.get("OPENROUTER_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not key:
        logger.warning(
            "role-batch: no OPENROUTER_KEY — fail-closed (rejecting %d gray-zone leads)",
            len(leads),
        )
        return {l["id"]: False for l in leads}

    # Chunk into BATCH_SIZE-sized groups
    chunks = [leads[i:i + BATCH_SIZE] for i in range(0, len(leads), BATCH_SIZE)]

    results: Dict[Any, bool] = {l["id"]: False for l in leads}  # default reject
    async with httpx.AsyncClient() as http:
        for chunk_idx, chunk in enumerate(chunks):
            parsed = await _judge_chunk(http, key, target_roles, chunk)
            if parsed is None:
                # Chunk failed — keep defaults (False).
                continue
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                lid = item.get("id")
                if lid is None or lid not in results:
                    continue
                results[lid] = bool(item.get("match"))

    return results
