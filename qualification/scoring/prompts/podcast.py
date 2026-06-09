"""PODCAST_APPEARANCE-specific prompt builder.

Adds PART F on top of the common PART 0/A/B + signal-status decision.
The block guides the LLM through:
  1. PARSE the miner claim (role, topic, person name)
  2. IDENTIFY the speaker via video title + channel + web search
  3. VERIFY the speaker's current employer matches lead_company —
     independently, NOT echoing target_website back
  4. ROLE check (only if claimed_speaker_role is non-null)
  5. TOPIC match check (claim vs episode content)
  6. DECISION TABLE → signal_status

Mirrors the verification logic from
``scripts/podcast_test_artifacts/harness_hybrid_v2_FINAL.py:STAGE2A_PROMPT``
which produced 0/60 FPs across podcast verification testing.  Key
anti-FP rule: when entity OR topic confidence is low, return
unable_to_verify rather than supported — wrong-entity attacks use
similar-domain lookalikes, wrong-topic attacks use adjacent topics.

URL precheck (intent_precheck.py:_check_intent_url_evidence_quality)
already rejects non-YouTube URLs at zero cost; PART F handles the
substance check for accepted YouTube URLs.
"""
from typing import Any, Dict

from . import _common


PART_F_BLOCK = """  PART F — PODCAST APPEARANCE VERIFICATION (applies ONLY when
  target_icp_signal asks for a podcast / video interview appearance by
  someone from lead_company; otherwise leave all podcast_* fields = "n/a"
  and SKIP this part):

    A podcast claim asks: "Did <someone from lead_company> appear as a
    guest on this episode?"  The miner's URL must substantiate that
    a representative of lead_company actually appeared AND discussed
    what the claim says they discussed — not adjacent content.

    Step 1 — PARSE THE MINER CLAIM
      From miner_claim, extract:
        - claimed_speaker_role: e.g. "CEO", "Founder", "VP of Sales", or
          "n/a" if the claim doesn't specify a role
        - claimed_topic:        what the miner says was discussed
        - claimed_person_name:  specific name if mentioned, else "n/a"

    Step 2 — IDENTIFY THE GUEST
      From the URL's video title + channel + page text (if scraped),
      determine identified_guest_name.  The channel host is NOT
      automatically the guest — episodes typically feature a host
      interviewing an external guest.  Be explicit about which is which.
      If you cannot identify the guest at all → set entity_check =
      "unable_to_determine".

    Step 3 — COMPANY ENTITY VERIFICATION (CRITICAL anti-FP gate)

      3.1 Look up identified_guest_name's CURRENT employer via web
          search ("<name> LinkedIn").  Record actual_employer_name.

      3.2 Independently look up the CANONICAL website for
          actual_employer_name — via "'<company name>' official site"
          search OR from the LinkedIn /company/<slug>/ page.
          Record actual_employer_canonical_url.

          CRITICAL: Do NOT copy target_website into this field.
          target_website is the CLAIM being verified — it might be
          a wrong-entity attack using a lookalike domain.  You MUST
          independently find the company's real website via web search.

      3.3 Compare actual_employer_name to lead_company:
          - Same legal entity (modulo legal suffix LLC/Inc/Corp)?
          - One is documented DBA / parent / subsidiary / rebrand of
            other?

      3.4 Compare actual_employer_canonical_url to lead_company's
          website (normalize: strip http/https://, www., trailing /).

      Decide entity_check:
        "yes"                — 3.3 ✓ AND 3.4 ✓
        "no"                 — 3.3 ✗ OR 3.4 ✗ (different domain even
                               when name matches → wrong_entity)
        "unable_to_determine" — web search cannot confirm either way

    Step 4 — ROLE CHECK (only when claimed_speaker_role != "n/a")

      Compare identified_guest's actual role to claimed_speaker_role:
        Allow: acronym ↔ full form ("CEO" matches "Chief Executive
               Officer"); compound titles ("Founder & CEO" matches
               either "CEO" or "Founder"); generic seniority terms
               ("senior leader", "executive") match any C-suite /
               VP+ / Director+ role.
        Reject: different function (CFO ≠ CMO, CTO ≠ COO).

      role_check ∈ {"yes", "no", "unable_to_determine"}.
      When claimed_speaker_role == "n/a" → role_check = "n/a".

    Step 5 — TOPIC ALIGNMENT
      5a. From video title + description + page text + any third-party
          recap you find, what was actually discussed?
      5b. What does miner_claim say was discussed?
      5c. Do they match at SUBJECT-MATTER level (same noun, not just
          same general field — "AML monitoring strategy" ≠ "compliance
          generally")?

      Title-only evidence over-promises; third-party recaps are stronger.
      With only title + description, lower your confidence.

      topic_match ∈ {"yes_high", "yes_low", "no", "unable_to_determine"}

    Step 6 — FINAL VERDICT (decision table)

      | entity_check        | role_check        | topic_match        | signal_status        |
      |---------------------|-------------------|--------------------|----------------------|
      | no                  | -                 | -                  | wrong_entity         |
      | unable_to_determine | -                 | -                  | unable_to_verify     |
      | yes                 | no                | -                  | contradicted         |
      | yes                 | unable_to_determine | -                | unable_to_verify     |
      | yes                 | yes OR n/a        | no                 | contradicted         |
      | yes                 | yes OR n/a        | unable_to_determine | unable_to_verify    |
      | yes                 | yes OR n/a        | yes_high           | supported            |
      | yes                 | yes OR n/a        | yes_low            | unable_to_verify     |

    CRITICAL ANTI-FP RULE: when you cannot confirm BOTH entity AND
    topic with high confidence, return unable_to_verify (NOT supported).
    Wrong-entity attacks use lookalike domains; wrong-topic attacks
    use adjacent-but-different subject matter.  The miner has every
    incentive to push borderline claims — be conservative."""


def build_verification_prompt(row: Dict[str, Any]) -> str:
    return _common.build_verification_prompt(row, extra_parts=[PART_F_BLOCK])


def build_final_judge_prompt(
    row: Dict[str, Any],
    contents: Dict[str, Any],
    source_name: str = "SD/Exa Contents",
) -> str:
    return _common.build_final_judge_prompt(
        row, contents, source_name, extra_parts=[PART_F_BLOCK]
    )
