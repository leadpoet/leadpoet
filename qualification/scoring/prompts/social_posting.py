"""SOCIAL_POSTING-specific prompt builder.

Adds PART D (author-role check) on top of the common PART 0/A/B + signal-
status decision.  PART D self-gates on URL shape and ICP role spec —
when the source URL is not a social-media post or the ICP doesn't
mention a person role, the LLM sets all author_* fields to ``"n/a"`` and
skips the check.

The PART D block below is copied character-by-character from the
original mega-prompt that lived in
``intent_verification_three_stage.py`` (lines 1015–1081 at refactor
time).  Snapshot equality test: ``tests/test_prompt_refactor.py``.
"""
from typing import Any, Dict

from . import _common


PART_D_BLOCK = """  PART D — AUTHOR-ROLE CHECK (applies ONLY when BOTH conditions hold;
  otherwise leave all author_* fields = "n/a" and SKIP this part):
    Condition 1 — source_url is a social-media post URL:
      • LinkedIn:  linkedin.com/posts/<handle>_..., linkedin.com/pulse/<slug>,
                   linkedin.com/feed/update/urn:li:activity:<id>
      • X / Twitter: x.com/<handle>/status/<id>,
                     twitter.com/<handle>/status/<id>
    Condition 2 — target_icp_signal names a PERSON ROLE:
      CEO, Founder, Co-founder, Owner, President, CRO, CSO, CMO,
      Chief X Officer, VP of <X>, sales leader, revenue leader.

    When both hold:

    1. AUTHOR_TYPE — extract the handle from the URL path and decide:
       • "person"  → handle looks like a personal handle (firstname-lastname,
                     given-name, individual nickname).
       • "company" → handle matches lead.company_linkedin's /company/<slug>
                     exactly, OR is a clear corporate slug (ends in -inc,
                     -llc, -ai, -labs, -solutions, -group, contains "company"
                     or "official").

    2. If author_type == "company":
       The post was made by a company page / official brand account, NOT by a
       person. Since target_icp_signal demands a person role, this fails the
       check. Set:
         author_employer_matches_lead = "n/a"
         author_role_matches_spec    = "n/a"
         author_satisfies_role_spec  = "no"
       AND set signal_status = "contradicted" with risk_note
       "company_handle_not_person".

    3. If author_type == "person":
       You MUST perform a separate web search for the handle (regardless of
       what the scraped post body says).  Search the canonical LinkedIn
       profile URL (https://www.linkedin.com/in/<handle>) and any news /
       company bio coverage.  Look for the LinkedIn page-title pattern
       "Name - Title at Company - LinkedIn" or an explicit "is the
       <Title> at <Company>" in coverage.

         author_employer_matches_lead = "yes" iff cited evidence shows their
           current employer name OR current employer LinkedIn URL equals
           lead.company / lead.company_linkedin.
         author_role_matches_spec     = "yes" iff cited evidence shows their
           current title satisfies the person-role in target_icp_signal.
         author_employer_matches_lead = "no" iff cited evidence shows a
           DIFFERENT current employer.
         author_role_matches_spec     = "no" iff cited evidence shows a
           DIFFERENT title that doesn't satisfy the spec.
         Only return "unknown" if you genuinely can't surface any LinkedIn
         profile, news mention, or company bio for the handle after a real
         search attempt.  DO NOT default to "unknown" just because the
         scraped post content alone doesn't cover author bio — go search.

         author_satisfies_role_spec = "yes" iff BOTH match.
         author_satisfies_role_spec = "no"  iff either is definitively "no".
         author_satisfies_role_spec = "unknown" iff resolution failed.

    4. PART D verdict-routing — when author_satisfies_role_spec == "no":
       • If author_employer_matches_lead == "no" (right title, wrong company):
         set signal_status = "wrong_entity" AND same_entity_check = "fail".
       • Else (employer matches but title doesn't, OR company-handle case):
         set signal_status = "contradicted" with risk_note "wrong_author_role"
         or "company_handle_not_person" as applicable.

    5. PART D verdict-routing — when author_satisfies_role_spec == "unknown":
       Do NOT override the verdict from PART 0/A/B. Leave the other-parts
       verdict to stand and surface "author_unresolved" in risk_notes."""


def build_verification_prompt(row: Dict[str, Any]) -> str:
    return _common.build_verification_prompt(row, extra_parts=[PART_D_BLOCK])


def build_final_judge_prompt(
    row: Dict[str, Any],
    contents: Dict[str, Any],
    source_name: str = "SD/Exa Contents",
) -> str:
    return _common.build_final_judge_prompt(
        row, contents, source_name, extra_parts=[PART_D_BLOCK]
    )
