"""Tests for the Stage-3 company-in-scrape pre-gate.

The pre-gate (`company_in_scrape` + `_entity_plausibly_present`, consumed in
`verify_three_stage`) is a COST filter in front of the Stage-3 sonar-pro entity
judge, not the judge itself. It must:

  • proceed to Stage 3 when the exact / base name is present (unchanged),
  • DEFER to Stage 3 (not hard-reject) when only the distinctive core token is
    present — e.g. lead "OpenArt AI" vs a source that writes just "OpenArt",
  • hard-reject cheaply (no sonar-pro call) only on a confident absence —
    no distinctive core token anywhere.

The unit tests pin the two deterministic helpers; the integration tests drive
the real `verify_three_stage` with the LLM and scrape boundaries mocked and
assert the routing (Stage-3 reached vs cheap reject) for each case, including
the fraud guard: a same-first-word DIFFERENT company must DEFER to the judge
(which rejects it), never be auto-passed by the pre-gate.
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qualification.scoring import intent_verification_three_stage as ivs


# ── deterministic helpers ──────────────────────────────────────────────
class CompanyCoreTokensTest(unittest.TestCase):
    def test_drops_descriptor_and_keeps_distinctive(self):
        self.assertEqual(ivs._company_core_tokens("OpenArt AI"), {"openart"})

    def test_drops_generic_tail(self):
        self.assertEqual(ivs._company_core_tokens("Artha Capital"), {"artha"})

    def test_strips_legal_suffix_chain(self):
        self.assertEqual(
            ivs._company_core_tokens("Tractian Technologies, Inc."),
            {"tractian"},
        )

    def test_multiword_distinctive_all_kept(self):
        self.assertEqual(
            ivs._company_core_tokens("Emery Sapp & Sons"),
            {"emery", "sapp", "sons"},
        )

    def test_generic_only_name_has_no_fingerprint(self):
        self.assertEqual(ivs._company_core_tokens("Capital Group"), set())

    def test_bare_descriptor_has_no_fingerprint(self):
        self.assertEqual(ivs._company_core_tokens("AI"), set())

    def test_short_and_numeric_tokens_dropped(self):
        # "studio" is generic, "54" is numeric → no distinctive fingerprint.
        self.assertEqual(ivs._company_core_tokens("Studio 54"), set())

    def test_similar_names_stay_distinct(self):
        # Guards against the OpenArt/OpenAI collision class being conflated.
        self.assertEqual(ivs._company_core_tokens("OpenAI"), {"openai"})
        self.assertNotEqual(
            ivs._company_core_tokens("OpenAI"),
            ivs._company_core_tokens("OpenArt AI"),
        )


class EntityPlausiblyPresentTest(unittest.TestCase):
    def test_core_token_present_defers(self):
        self.assertTrue(
            ivs._entity_plausibly_present("OpenArt AI", "we love OpenArt here")
        )

    def test_core_token_absent_is_confident_absence(self):
        self.assertFalse(
            ivs._entity_plausibly_present("OpenArt AI", "we love Canva here")
        )

    def test_generic_tail_alone_is_not_presence(self):
        # "capital" is a generic tail; only "artha" is distinctive and it is
        # absent → confident absence.
        self.assertFalse(
            ivs._entity_plausibly_present("Artha Capital", "the firm raised capital")
        )

    def test_generic_only_name_always_defers(self):
        # No reliable fingerprint → never cheap-reject; defer to the judge.
        self.assertTrue(
            ivs._entity_plausibly_present("Capital Group", "unrelated text")
        )

    def test_word_boundary_not_substring(self):
        # "openart" must appear as a whole word, not inside another token.
        self.assertFalse(
            ivs._entity_plausibly_present("OpenArt AI", "reopenarted the file")
        )


class CompanyInScrapeRegressionTest(unittest.TestCase):
    """The confident-present check is unchanged by the fix."""

    def test_exact_match(self):
        self.assertTrue(ivs.company_in_scrape("OpenArt", "hello OpenArt world"))

    def test_base_suffix_match(self):
        self.assertTrue(
            ivs.company_in_scrape("Tractian Technologies, Inc.", "Tractian shipped it")
        )

    def test_word_boundary_no_false_substring(self):
        self.assertFalse(ivs.company_in_scrape("Art", "Bart Simpson"))

    def test_descriptor_variant_still_misses_here(self):
        # This function intentionally still returns False for "OpenArt AI" vs a
        # source that writes only "OpenArt" — the fix lives at the gate (defer
        # to Stage 3), NOT by loosening this exact/base matcher.
        self.assertFalse(ivs.company_in_scrape("OpenArt AI", "we love OpenArt"))


# ── integration: real verify_three_stage, mocked boundaries ────────────
_S1_REVIEW = {
    "answer": {
        "signal_evaluations": [{
            "signal_status": "unable_to_verify",
            "verification_mode": "source_grounded",
            "confidence": "medium",
        }]
    },
    "model": "sonar",
    "usage": {},
}


def _s3_approve(url):
    return {
        "answer": {
            "signal_evaluations": [{
                "signal_status": "supported",
                "confidence": "high",
                "verification_mode": "source_grounded",
                "evidence_urls_used": [url],
                "same_entity_check": "pass",
            }]
        },
        "model": "sonar-pro",
        "usage": {},
    }


def _s3_wrong_entity():
    return {
        "answer": {
            "signal_evaluations": [{
                "signal_status": "wrong_entity",
                "verification_mode": "source_grounded",
                "same_entity_check": "fail",
                "confidence": "high",
            }]
        },
        "model": "sonar-pro",
        "usage": {},
    }


class _Caller:
    """Fake _call_openrouter: Stage 1 → review, Stage 2+ → the given s3 env."""

    def __init__(self, s3_env):
        self.calls = 0
        self.s3_env = s3_env

    async def __call__(self, client, model, prompt):
        self.calls += 1
        return _S1_REVIEW if self.calls == 1 else self.s3_env


def _fetch(text, url):
    async def _f(urls, *a, **k):
        return {"results": [{"url": url, "text": text, "meta": {}}], "statuses": []}
    return _f


class GateRoutingIntegrationTest(unittest.TestCase):
    URL = "https://techblog.example.com/openart-launches-new-model"

    def _run(self, *, company_name, text, s3_env):
        caller = _Caller(s3_env)
        with mock.patch.object(ivs, "_call_openrouter", caller), \
                mock.patch.object(ivs, "_fetch_sd_then_exa", _fetch(text, self.URL)):
            result = asyncio.run(ivs.verify_three_stage(
                None,
                company_name=company_name,
                company_linkedin="",
                company_website="",
                source_url=self.URL,
                miner_claim="the company launched a new product",
                target_signal_text="product launch",
            ))
        return result, caller

    def test_descriptor_variant_defers_to_stage3_and_approves(self):
        # THE REPORTED BUG: lead "OpenArt AI", source says only "OpenArt".
        result, caller = self._run(
            company_name="OpenArt AI",
            text="OpenArt just launched a new model for creators today.",
            s3_env=_s3_approve(self.URL),
        )
        self.assertEqual(caller.calls, 2, "Stage 3 must be reached, not pre-rejected")
        self.assertIsNotNone(result["stage3"])
        self.assertNotEqual(
            result["rejection_reason"], "wrong_entity_company_not_in_fetched_content"
        )
        self.assertEqual(result["decision"], "approve")
        self.assertIsNone(result["company_check"], "deferred → company_check None")

    def test_exact_match_reaches_stage3(self):
        result, caller = self._run(
            company_name="OpenArt",
            text="OpenArt just launched a new model for creators today.",
            s3_env=_s3_approve(self.URL),
        )
        self.assertEqual(caller.calls, 2)
        self.assertIsNotNone(result["stage3"])
        self.assertTrue(result["company_check"])
        self.assertEqual(result["decision"], "approve")

    def test_confident_absence_cheap_rejects_without_stage3(self):
        # Marriott-style page: distinctive token "artha" is absent entirely.
        result, caller = self._run(
            company_name="Artha Capital",
            text="Marriott International announced a new hotel opening in Denver.",
            s3_env=_s3_approve(self.URL),
        )
        self.assertEqual(caller.calls, 1, "Stage 3 must NOT be called")
        self.assertIsNone(result["stage3"])
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(
            result["rejection_reason"], "wrong_entity_company_not_in_fetched_content"
        )
        self.assertFalse(result["company_check"])

    def test_generic_tail_present_still_cheap_rejects(self):
        # "capital" appears but is a generic tail; "artha" (distinctive) does
        # not → still a confident absence, still cheap-rejected.
        result, caller = self._run(
            company_name="Artha Capital",
            text="The private equity firm raised fresh capital for its new fund.",
            s3_env=_s3_approve(self.URL),
        )
        self.assertEqual(caller.calls, 1)
        self.assertEqual(
            result["rejection_reason"], "wrong_entity_company_not_in_fetched_content"
        )

    def test_generic_only_name_defers_never_cheap_rejects(self):
        # No distinctive fingerprint → must defer to the judge even when the
        # name is absent, rather than cheap-reject on an unreliable signal.
        result, caller = self._run(
            company_name="Capital Group",
            text="An unrelated article about renewable energy policy in Europe.",
            s3_env=_s3_approve(self.URL),
        )
        self.assertEqual(caller.calls, 2, "generic-only name must reach Stage 3")
        self.assertIsNotNone(result["stage3"])

    def test_fraud_guard_same_first_word_defers_and_judge_rejects(self):
        # Lead "Hive AI" vs a page about the DIFFERENT "HIVE Digital
        # Technologies". The pre-gate must DEFER (core token "hive" present),
        # never auto-pass; the real judge then returns wrong_entity.
        result, caller = self._run(
            company_name="Hive AI",
            text="HIVE Digital Technologies, a Canadian crypto miner, posted results.",
            s3_env=_s3_wrong_entity(),
        )
        self.assertEqual(caller.calls, 2, "must reach the judge, not auto-pass")
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["rejection_reason"], "stage3_wrong_entity")


if __name__ == "__main__":
    unittest.main()
