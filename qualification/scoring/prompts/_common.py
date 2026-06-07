"""Shared prompt building blocks for the three-stage intent verifier.

The per-evidence-type modules (``social_posting``, ``default``, future
``podcast``) compose their prompts by passing extra block(s) to
``build_verification_prompt`` and ``build_final_judge_prompt``.  Every
fixed text block lives here as a module-level constant — copied
character-by-character from the original mega-prompt in
``qualification/scoring/intent_verification_three_stage.py`` to guarantee
byte-identical output through the refactor.

Snapshot equality test: ``tests/test_prompt_refactor.py``.
"""
import json
from datetime import date
from typing import Any, Dict, Iterable, List


# Mirror of MAX_SCRAPED_CHARS in intent_verification_three_stage.py:78.
# Defined here as a plain int constant to keep this module
# import-cycle-free (the verifier module imports prompts/, not the other
# way around).  Keep these two values in sync.
MAX_SCRAPED_CHARS = 60_000


def lead_profile(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: row.get(k, "")
        for k in ("company", "website", "company_linkedin", "contact_linkedin")
    }


def visible_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    # Surface BOTH the miner's free-text claim AND the buyer's ICP signal so
    # the LLM can check semantic alignment between them.  Without
    # target_icp_signal in the prompt, the verifier only judged "does the URL
    # support what the miner wrote?" — a miner could submit a true-but-
    # orthogonal claim (e.g. "Artisan offers an AI BDR product") and get
    # approved even though it doesn't satisfy the actual ICP signal ("Company
    # has active job postings for SDR/BDR roles"). Observed 2026-05-18 on
    # multiple winners. Including target_icp_signal lets the prompt rules
    # below enforce: the claim must semantically address the ICP signal AND
    # the URL must support the claim.
    return {
        "id": str(row.get("id") or "signal-1"),
        "signal_type": row.get("signal_type") or "unknown",
        "miner_claim": row.get("claim") or "",
        "miner_signal_date": row.get("signal_date") or None,
        "target_icp_signal": row.get("_target_signal_text") or "",
        "claimed_source_urls": row.get("claimed_source_urls") or [],
    }


# ──────────────────────────────────────────────────────────────────────
# PART 0 — ENTITY CHECK
# ──────────────────────────────────────────────────────────────────────
PART_0_BLOCK = """  PART 0 — ENTITY CHECK (CRITICAL, evaluate FIRST and REJECT on any doubt):

    The lead profile gives THREE identity anchors:
      • ``company`` — short / informal name (often ambiguous)
      • ``website`` — the lead's exact company URL
      • ``company_linkedin`` — the lead's exact LinkedIn URL

    Many short company names ("Hive", "Apex", "Copper", "Halcyon",
    "Risotto", "Shovels", "Augment", "Proto", "Square", "Apple")
    refer to MULTIPLE real corporate entities globally.  The
    ``company`` field alone CANNOT disambiguate.  The ``website`` and
    ``company_linkedin`` are the canonical entity identifiers.

    To return `supported`, you must affirmatively verify the article
    (URL + extracted content) is about THE SPECIFIC entity at the
    lead's ``website`` and ``company_linkedin``.  This means at least
    ONE of the following is true:
      (a) URL hostname matches the lead's website domain (case already
          handled structurally — these URLs are guaranteed same-entity).
      (b) The article EXPLICITLY mentions the lead's website domain
          (e.g. ``thehive.ai``) or LinkedIn URL.
      (c) The article's canonical full company name (as actually
          printed in headline, byline, infobox, or URL slug)
          UNAMBIGUOUSLY matches the lead — including any branding
          differentiator the lead uses (e.g. ``Hive AI`` if the
          lead's domain is ``thehive.ai``; ``Tractian Technologies``
          if the lead's website is ``tractian.com``).
      (d) The article's stated industry, headquarters, leadership,
          or product description matches the lead's known business
          (inferable from the lead's website domain).

    Return `wrong_entity` whenever ANY of the following holds:
      • The article identifies a longer / different corporate name
        sharing the lead's first word but adding distinguishing words
        not in the lead's website / LinkedIn URL.  Examples:
            lead at thehive.ai (Hive AI, content moderation)
              vs article about "HIVE Digital Technologies"
              (TSX:HIVE, Canadian crypto miner)         → wrong_entity
            lead at apcoworldwide.com (PR firm)
              vs article about "APCO Holdings"
              (auto-warranty parent)                    → wrong_entity
            lead at copper.com (crypto custody)
              vs article about "Copper Labs"
              (energy startup)                          → wrong_entity
            lead at parabola.io (data automation)
              vs article about "Parabola Labs"
              (different company on LinkedIn)           → wrong_entity
            lead at halcyon.ai (ransomware security)
              vs article about a "Halcyon AI" doing
              energy data analytics & clean energy      → wrong_entity
      • The URL or content's actual subject is in a clearly different
        INDUSTRY / DOMAIN than the lead.  Examples:
            lead at zefir.fr (French real estate AI)
              vs trendyol.com product page describing
              "Zefir Fabric Orange Large Checked Pattern"
              tablecloth                                → wrong_entity
            lead at shovels.ai (construction-permits AI)
              vs github.com/rabbitmq discussion
              about RabbitMQ's "Shovel" plugin / API    → wrong_entity
            lead at tryrisotto.com (IT help-desk AI)
              vs github.com/joeroe/risotto, an R package
              for radiocarbon dating by user "joeroe"   → wrong_entity
            lead at wearepro.to (Proto, AI customer service)
              vs an article about "Proto Corporation"
              launching travel experiences              → wrong_entity
      • The URL has the lead's name in it but the article body /
        title is about a DIFFERENT named company sharing only the
        first word.  Example:
            lead "Augment" at goaugment.io (logistics AI)
              vs augment.market article about a different
              "Augment" doing private-market trading
              ($12M Series A led by Builders VC)        → wrong_entity

    Functional equivalents (same entity, NOT wrong_entity):
      • Parent ↔ subsidiary / division
        (lead Axos Bank ↔ "Axos Financial" parent;
         lead Ara Partners ↔ "Ara Energy" platform;
         lead Penzance ↔ "Penzance Digital Infrastructure" division)
      • Formal corporate suffix
        (lead "Bluebeam" ↔ "Bluebeam, Inc.";
         lead "Tractian" ↔ "Tractian Technologies, Inc.";
         lead "AmplifAI" ↔ "AmplifAI Solutions, Inc.")
      • Verb usage in URL slug
        (lead "Omnes" at omnescapital.com → URL slug
         "omnes-partners-with-apex-group" — "partners"
         is a VERB here; same entity)
      • Aggregator profile pages keyed by the lead's slug
        (e.g. indeed.com/cmp/<Lead-Name>/, crunchbase.com/
         organization/<lead>, builtin.com/company/<lead>)"""


# ──────────────────────────────────────────────────────────────────────
# PART A — CLAIM ↔ ICP SEMANTIC ALIGNMENT
# ──────────────────────────────────────────────────────────────────────
PART_A_BLOCK = """  PART A — CLAIM ↔ ICP SEMANTIC ALIGNMENT:
    The miner asserts their evidence proves the buyer's target_icp_signal.
    Check whether the miner_claim, even if true, would actually mean
    the target_icp_signal is satisfied.
      * "Company X offers product Y" is NOT the same as "Company X hires for
        role Y". Selling an AI BDR is the opposite signal of hiring BDRs.
      * "Company X had a funding round" is NOT the same as "Company X is
        hiring sales roles" unless the URL specifically ties the funding to
        sales-team expansion.
      * "Company X has a careers page" is NOT the same as "Company X has
        open positions for {role}".
    If miner_claim does not semantically map to target_icp_signal, return
    `contradicted` (the URL is about a different topic than the ICP
    signal asks for).  Do NOT return `wrong_entity` — `wrong_entity` is
    reserved STRICTLY for entity-identity mismatch in PART 0."""


# ──────────────────────────────────────────────────────────────────────
# PART B — URL SUPPORTS THE CLAIM
# ──────────────────────────────────────────────────────────────────────
PART_B_BLOCK = """  PART B — URL SUPPORTS THE CLAIM:
    - Treat the lead profile as context, not proof.
    - Anchor company identity to company website and company LinkedIn.
    - If source URL(s) are provided, use source_grounded mode.
    - In source_grounded mode, validate only against exact supplied source URL(s).
    - Do not use alternate URLs, same-domain pages, cached pages, search results, or replacement URLs as support.
    - Independent search may only flag contradiction, stale data, or wrong-entity issues.
    - If no source URL is provided, use discovery mode and search credible sources."""


# ──────────────────────────────────────────────────────────────────────
# Signal status decision rules
# ──────────────────────────────────────────────────────────────────────
SIGNAL_STATUS_DECISION_BLOCK = """Signal status decision (mutually exclusive — pick exactly one):
- supported: PART 0 entity check PASSES AND PART A holds AND exact evidence directly supports miner_claim.
- partially_supported: PART 0 + PART A hold AND exact evidence supports only part of miner_claim.
- contradicted: PART 0 entity check PASSES BUT exact evidence contradicts miner_claim,
                OR PART A fails (miner_claim does not semantically address target_icp_signal).
- unable_to_verify: PART 0 entity check is INCONCLUSIVE (the article COULD be about
                    the lead but you can't confirm it from canonical-name / domain /
                    LinkedIn / industry signals), OR evidence is missing, inaccessible,
                    ambiguous, stale, or insufficient.
- wrong_entity: PART 0 entity check definitively FAILS — the article identifies a
                clearly DIFFERENT corporate entity than the lead's `website` and
                `company_linkedin` (e.g., HIVE Digital Technologies vs Hive AI,
                APCO Holdings vs APCO Worldwide, joeroe/risotto R package vs
                tryrisotto.com).  RESERVED STRICTLY for entity-identity mismatch;
                NEVER use wrong_entity for a claim-vs-ICP mismatch (that's
                `contradicted`).

When uncertain about PART 0 (article COULD be about the lead but proof is weak),
return `unable_to_verify` — NOT `wrong_entity`.  When confidence the article is
about a DIFFERENT entity, return `wrong_entity`.

When also setting wrong_entity, ALSO set `same_entity_check` field to "fail"
so structural callers can distinguish entity-mismatch from claim-mismatch."""


# ──────────────────────────────────────────────────────────────────────
# Final-judge-only blocks (consumed by build_final_judge_prompt)
# ──────────────────────────────────────────────────────────────────────
FINAL_JUDGE_RULES_BLOCK = """Final judge rules:
- Re-apply the PART A check from above BEFORE judging content support: if
  miner_claim does not semantically map to target_icp_signal, return
  wrong_entity regardless of what the extracted content shows. Do not let a
  factually-true but orthogonal claim pass just because the URL supports it.
- Use only the exact source extraction above as supporting evidence.
- Page titles, navigation menus, headers, and breadcrumbs are NOT evidence.
  Only specific factual claims in the page BODY count.
- If the body contains explicit negation about the claim ("0 open positions",
  "no longer open", "no longer accepting applications", "not currently hiring",
  "position has been removed", "page not found", "404",
  "the job you are looking for is no longer"),
  return contradicted regardless of titles, headers, or partial context.
- Job-posting freshness rule: when the claim is about an active/open job
  posting (hiring signal), the source MUST show that posting is still open.
  Treat any of these closed-state phrases as contradicted — even if other
  parts of the page still describe the role:
    "no longer accepting applications", "applications are closed",
    "this job is closed", "position filled", "we are no longer hiring",
    "job is no longer available", "expired".
- Job-posting timeline rule: when the claim is about active/current hiring
  (e.g. "is hiring", "actively recruiting", "open positions for X"), if the
  source body shows a posting age (e.g. "Posted 7 months ago", "Posted on
  YYYY-MM-DD") AND that age is > 6 months relative to "Today's date" above,
  return contradicted (stale_posting). Job-board sidebars and "similar
  jobs" timestamps do NOT count — only the posting age of the ACTUAL job
  being verified.
  For non-hiring claims (e.g. funding announcements, expansion signals,
  product launches, acquisitions, tech-stack inferences from job
  requirements), do NOT penalize on age — older or closed job postings can
  still validate those factual claims.  If no posting age is visible on the
  page, do NOT penalize on staleness; judge content only.
- If exact extracted content directly supports miner_claim AND PART A holds,
  return supported.
- If extracted content supports only part of miner_claim AND PART A holds,
  return partially_supported.
- If extraction failed or content is insufficient, return unable_to_verify.
- If content contradicts miner_claim, return contradicted.
- If content is about another company/person, return wrong_entity.
- evidence_urls_used must contain only exact supplied source URLs whose
  extracted content supports or contradicts the claim."""


MINER_DATE_CHECK_BLOCK = """Miner-date consistency check (set claim_matches_miner_date) — STEP BY STEP:
1. Treat miner_signal_date as a HYPOTHESIS to verify, not a fact.
2. Look for a POST TIMESTAMP in extracted_text.  Allowed forms ONLY:
   a) absolute date ("2025-03-15", "March 15, 2025", "Posted on Jan 4, 2026")
   b) anchored relative phrase ("Posted N months ago", "Published N weeks
      ago", "Shared yesterday", "Updated last week")
   c) LinkedIn timestamp badge ("4mo •", "2w •", "5d •")
   Body prose ("we launched 6 months ago", "joined 5 years ago", "I saw
   this 3 weeks ago") is NOT a post timestamp — IGNORE it.
3. If you find one, COMPUTE the implied post date relative to today (the
   page-extract date).  Example: "Posted 4 months ago" extracted today
   2026-06-04 → implied post date ≈ 2026-02-04.
4. Compare implied post date to miner_signal_date:
   - within ±15 days  → "consistent"
   - 15–30 days apart → "consistent" (benefit of the doubt)
   - >30 days apart   → "contradicted".  When contradicted, ALSO set
     signal_status="contradicted" and add "date_mismatch" to risk_notes.
5. If no post timestamp found → "no_date_in_content".  Don't penalize.
6. If miner_signal_date is null → "no_date_in_content" regardless."""


# ──────────────────────────────────────────────────────────────────────
# Assemblers
# ──────────────────────────────────────────────────────────────────────
def build_verification_prompt(
    row: Dict[str, Any],
    extra_parts: Iterable[str] = (),
) -> str:
    """Assemble the Stage 1 verification prompt.

    ``extra_parts`` is a sequence of additional PART blocks (e.g. PART D
    for SOCIAL_POSTING, or future PART E for PODCAST_APPEARANCE).  Each
    one is joined into the body with a blank line separator, matching
    the original prompt's whitespace exactly.
    """
    signal = visible_signal(row)
    parts: List[str] = [PART_0_BLOCK, PART_A_BLOCK, PART_B_BLOCK]
    parts.extend(extra_parts)
    body = "\n\n".join(parts)
    return f"""Evaluate this B2B sales lead.

Lead profile:
{json.dumps(lead_profile(row), indent=2)}

Intent signal to verify:
{json.dumps(signal, indent=2)}

Three-part verification — ALL THREE must hold for `supported`:

{body}

{SIGNAL_STATUS_DECISION_BLOCK}

Return only schema-valid JSON."""


def build_final_judge_prompt(
    row: Dict[str, Any],
    contents: Dict[str, Any],
    source_name: str = "SD/Exa Contents",
    extra_parts: Iterable[str] = (),
) -> str:
    """Assemble the Stage 3 final-judge prompt.

    Wraps the verification prompt with the extracted source content
    blocks plus the final-judge + miner-date rules.  ``extra_parts``
    is forwarded to ``build_verification_prompt`` unchanged.
    """
    blocks: List[str] = []
    for res in (contents.get("results") or []):
        url = res.get("url") or res.get("id") or ""
        text = (res.get("text") or "")[:MAX_SCRAPED_CHARS]
        title = res.get("title") or ""
        blocks.append(f"URL: {url}\nTITLE: {title}\nCONTENT:\n{text}")
    if not blocks:
        blocks = [
            "NO CONTENT. STATUSES:\n"
            + json.dumps(contents.get("statuses") or [], indent=2)
        ]
    today_str = date.today().isoformat()
    verification = build_verification_prompt(row, extra_parts=extra_parts)
    return f"""{verification}

{source_name} exact supplied source extraction:
{chr(10).join(blocks)}

Today's date: {today_str}

{FINAL_JUDGE_RULES_BLOCK}

{MINER_DATE_CHECK_BLOCK}"""
