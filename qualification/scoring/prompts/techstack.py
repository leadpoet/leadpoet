"""TECHSTACK-specific prompt builder.

Adds PART E (tech-stack anti-pattern check) on top of the common
PART 0/A/B + signal-status decision.  PART E enforces the eight
production-audited anti-patterns and the four acceptance criteria
for tool-usage claims of the shape "Company X uses Tool Y".

The validator-side URL pre-check in ``intent_precheck.py`` already
catches the deterministic anti-pattern URL shapes (vendor comparison
pages, integration listings, data-broker profile pages, personal
LinkedIn pages).  PART E covers the patterns that REQUIRE reading the
page content: REPLACED, TANGENTIAL, WRONG_COMPANY, PREVIOUS_EMPLOYER,
and the semantic-level VENDOR_MARKETING / PRODUCT_INTEGRATION checks
that a URL-shape rule can't catch.
"""
from typing import Any, Dict

from . import _common


PART_E_BLOCK = """  PART E — TECHSTACK ANTI-PATTERNS (applies ONLY when target_icp_signal
  asks whether the target company USES a specific tool/CRM/sales tech;
  otherwise leave PART E checks "n/a" and SKIP this part):

    A tech-stack claim asks: "Does <lead_company> use <tool>?" The
    miner's URL must substantiate that, not adjacent claims.  Set
    signal_status = "contradicted" with the listed risk_note when ANY
    of these anti-patterns apply:

    1. VENDOR_MARKETING — The URL is hosted on the TOOL VENDOR's
       domain and talks ABOUT the lead_company.  Vendor marketing
       (comparison pages, customer success stories the vendor hosts,
       "X vs Y" pages, product spec sheets) does NOT substantiate that
       lead_company uses the tool internally.  The page is the
       vendor's content, not the target's tech choice.
       risk_note = "vendor_marketing_not_target_usage"

    2. PRODUCT_INTEGRATION — The URL describes lead_company's PRODUCT
       integrating with the tool (e.g. "Industrial Defender integrates
       with Salesforce" on an integrations directory page).  Product-
       level integration ≠ team-level internal usage of the tool.
       risk_note = "product_integration_not_internal_usage"

    3. REPLACED — The page text says lead_company REPLACED the tool
       with some other product.  Example: "Initially replaced
       Salesforce CRM with [our product]" on a customer success story
       page.  This is NEGATIVE evidence — the target does NOT
       currently use the tool.
       risk_note = "target_replaced_tool"

    4. DATA_BROKER — The URL is a data-broker / business-directory
       profile page (ZoomInfo /c/<slug>, Crunchbase /organization/<slug>,
       Apollo /companies/<slug>, Growjo /company/<slug>, Builtin
       /company/<slug> NOT under /job/).  These show generic metadata
       about lead_company but do not contain page-text assertions
       naming the specific tool as current usage.
       risk_note = "data_broker_profile_no_tool_assertion"

    5. TANGENTIAL — The tool is mentioned briefly in an unrelated
       document (data-breach report, news article tangentially listing
       companies, "leading tools include..." generic listing).  No
       specific assertion that lead_company uses the tool.
       risk_note = "tangential_mention_no_specific_claim"

    6. PERSONAL_PROFILE — The URL is focused on an individual rather
       than the company (personal LinkedIn /in/<slug>, personal blog,
       individual's resume).  Personal profiles don't substantiate
       company tech choices, even if the person works at lead_company.
       risk_note = "personal_profile_not_company_tech"

    7. WRONG_COMPANY — The URL is about a DIFFERENT company than
       lead_company.  (Example: URL describes "SBM Management
       Services" but the claim is about "enGenius Consulting Group".)
       Set signal_status = "wrong_entity" AND
       same_entity_check = "fail" with risk_note "url_about_different_company".

    8. PREVIOUS_EMPLOYER — The page references a PREVIOUS employer of
       an individual.  Example: "Lindsey Hotaling joined Casap from
       Salesloft" — Salesloft was Lindsey's PREVIOUS employer, not
       Casap's tech stack.  Do not infer Casap uses Salesloft from
       such a reference.
       risk_note = "previous_employer_not_current_tech"

    VALID EVIDENCE — Set signal_status = "supported" only when at
    least one of A-D below holds AND the tool from target_icp_signal
    is explicitly named in the page content (or in cited evidence
    text):

    A. JOB_POSTING_REQUIRED — A job posting on lead_company's careers
       page, lead_company's LinkedIn /jobs/, or a job board with the
       posting clearly tagged to lead_company (lever.co/<target>,
       greenhouse.io/<target>, builtin.com/job/...) that lists the
       tool as a REQUIRED qualification.  Examples of acceptable
       language: "3 years Salesforce administration required",
       "Salesforce certification required", "Must have hands-on
       experience with HubSpot".

       WEAKER variant — If the tool is listed only under "preferred
       qualifications", "nice to have", "plus", "bonus", or similar
       optional phrasing, do NOT mark "supported".  Return
       signal_status = "unable_to_verify" with risk_note
       "tool_listed_as_preferred_not_required".

    B. TARGET_FIRST_PARTY — The URL is on lead_company's own domain
       (about page, team page, customer success story BY
       lead_company, engineering blog post) explicitly naming the
       tool as part of lead_company's stack or workflow.  Example:
       "Our sales team relies on Salesforce and Outreach" on
       lead_company's blog.

    C. PRESS_RELEASE — An independent trade publication or press
       release that explicitly quotes a representative of lead_company
       naming the tool — e.g. a CEO quote in a TechCrunch article
       saying "we standardized on Salesforce".  Generic phrasing like
       "X uses leading sales tools" without naming the specific tool
       does NOT qualify.

    D. INTERVIEW — A podcast / interview transcript or recap that
       captures lead_company's founder, CEO, executive, or named
       team member explicitly naming the tool as current usage
       ("we use Outreach for our sales engagement").  General
       discussion of the tool's industry without explicit attribution
       to lead_company does NOT qualify.

    DEFAULT WHEN AMBIGUOUS — When the URL passes none of anti-patterns
    1-8 AND none of acceptance criteria A-D, return signal_status =
    "unable_to_verify" with risk_note "no_explicit_tool_usage_assertion".
    Be CONSERVATIVE — do not infer current tool usage from adjacent
    facts (industry membership, job title, vendor presence in the
    same market, etc.).

    NEGATION GUARDS:
      • "Considering migrating to <tool>" → NOT current usage.
      • "Evaluating <tool> as one option" → NOT current usage.
      • "Used to use <tool>" / "Previously on <tool>" → NEGATIVE —
        signal_status = "contradicted", risk_note = "former_not_current_usage".
      • "Will adopt <tool> in 2026" → future, NOT current — return
        "unable_to_verify" with risk_note "future_not_current_usage".

    NAMING COLLISIONS:
      • "apollo" can refer to the sales-intelligence tool Apollo.io OR
        to spacecraft / mythology.  Only accept when context makes the
        sales-tool reading unambiguous.
      • "outreach" can be the verb (e.g. "marketing outreach effort")
        or the product (Outreach.io).  Require the product reading."""


def build_verification_prompt(row: Dict[str, Any]) -> str:
    return _common.build_verification_prompt(row, extra_parts=[PART_E_BLOCK])


def build_final_judge_prompt(
    row: Dict[str, Any],
    contents: Dict[str, Any],
    source_name: str = "SD/Exa Contents",
) -> str:
    return _common.build_final_judge_prompt(
        row, contents, source_name, extra_parts=[PART_E_BLOCK]
    )
