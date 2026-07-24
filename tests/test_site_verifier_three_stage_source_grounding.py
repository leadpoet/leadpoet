from __future__ import annotations

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from qualification.scoring.intent_verification_three_stage import (
    _rescue_medium_with_corroboration,
    _scrape_exa,
    _search_exa_corroboration,
    verify_three_stage,
)


def supported(url: str) -> dict:
    return {
        "answer": {
            "signal_evaluations": [{
                "signal_status": "supported",
                "confidence": "high",
                "same_entity_check": "pass",
                "verification_mode": "source_grounded",
                "evidence_urls_used": [url],
                "claim_matches_miner_date": "supported",
            }],
        },
        "model": "test-model",
        "usage": {},
    }


def contradicted(url: str) -> dict:
    value = supported(url)
    value["answer"]["signal_evaluations"][0]["signal_status"] = "contradicted"
    return value


def medium_supported(url: str, date_status: str = "no_date_in_content") -> dict:
    value = supported(url)
    item = value["answer"]["signal_evaluations"][0]
    item["confidence"] = "medium"
    item["claim_matches_miner_date"] = date_status
    return value


class SourceGroundingTests(unittest.IsolatedAsyncioTestCase):
    # Lab adaptation: the bounded corroboration rescue is flag-gated in this
    # repo (RESEARCH_LAB_INTENT_CORROBORATION_RESCUE, default off). These
    # tests verify the rescue behavior itself, so the flag is enabled for
    # THIS class only (setUp/tearDown, no module-level env leakage).
    def setUp(self):
        super().setUp()
        self._prev_rescue_flag = os.environ.get(
            "RESEARCH_LAB_INTENT_CORROBORATION_RESCUE"
        )
        os.environ["RESEARCH_LAB_INTENT_CORROBORATION_RESCUE"] = "1"

    def tearDown(self):
        if self._prev_rescue_flag is None:
            os.environ.pop("RESEARCH_LAB_INTENT_CORROBORATION_RESCUE", None)
        else:
            os.environ["RESEARCH_LAB_INTENT_CORROBORATION_RESCUE"] = (
                self._prev_rescue_flag
            )
        super().tearDown()

    async def test_exact_official_domain_can_establish_entity_but_not_claim(self):
        url = "https://advario.com/news/terminal-project"
        stage_one = supported(url)
        stage_three = contradicted(url)
        stage_three["answer"]["signal_evaluations"][0]["same_entity_check"] = "pass"
        call = AsyncMock(side_effect=[stage_one, stage_three])
        fetch = AsyncMock(return_value={
            "results": [{
                "url": url,
                "title": "Terminal project",
                "text": "A terminal project description whose extracted body omits the brand name.",
            }],
            "statuses": [{"source": "scrapingdog", "stage": "ok"}],
        })
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Advario",
                company_linkedin="https://www.linkedin.com/company/advario",
                company_website="advario.com",
                source_url=url,
                miner_claim="Advario launched a shore-power initiative",
                target_signal_text="Require a recent shore-power initiative",
                miner_signal_date="2026-07-01",
                stage1_soft_reject=True,
            )

        self.assertTrue(result["company_check"])
        self.assertFalse(result["client_ready"])
        self.assertEqual(result["rejection_reason"], "stage3_contradicted")
        self.assertEqual(call.await_count, 2)

    async def test_unrelated_domain_without_entity_name_still_fails_closed(self):
        url = "https://news.example/terminal-project"
        call = AsyncMock(return_value=supported(url))
        fetch = AsyncMock(return_value={
            "results": [{
                "url": url,
                "title": "Terminal project",
                "text": "A different operator announced a terminal expansion.",
            }],
            "statuses": [{"source": "scrapingdog", "stage": "ok"}],
        })
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Advario",
                company_linkedin="https://www.linkedin.com/company/advario",
                company_website="advario.com",
                source_url=url,
                miner_claim="Advario launched a shore-power initiative",
                target_signal_text="Require a recent shore-power initiative",
                miner_signal_date="2026-07-01",
                stage1_soft_reject=True,
            )

        self.assertFalse(result["client_ready"])
        self.assertFalse(result["company_check"])
        self.assertEqual(
            result["rejection_reason"],
            "wrong_entity_company_not_in_fetched_content",
        )
        self.assertEqual(call.await_count, 1)

    async def test_exa_retries_only_a_transient_fetch_without_changing_evidence(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(503, request=request)
            return httpx.Response(200, request=request, json={
                "results": [{"text": "Acme " + ("verified evidence " * 30)}],
            })

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"EXA_API_KEY": "test-key"}),
            patch(
                "qualification.scoring.intent_verification_three_stage.httpx.AsyncClient",
                side_effect=lambda *args, **kwargs: real_async_client(transport=transport),
            ),
            patch(
                "qualification.scoring.intent_verification_three_stage.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            result = await _scrape_exa("https://acme.example/evidence")

        self.assertTrue(result["ok"])
        self.assertEqual(calls, 2)
        self.assertIn("Acme", result["content"])

    async def test_exa_does_not_retry_a_deterministic_client_error(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(400, request=request)

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"EXA_API_KEY": "test-key"}),
            patch(
                "qualification.scoring.intent_verification_three_stage.httpx.AsyncClient",
                side_effect=lambda *args, **kwargs: real_async_client(transport=transport),
            ),
        ):
            result = await _scrape_exa("https://acme.example/evidence")

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "exa_http_error")
        self.assertEqual(calls, 1)

    async def test_stage_one_provider_error_is_unavailable_not_a_false_rejection(self):
        call = AsyncMock(return_value={"_error": "http_403"})
        with patch(
            "qualification.scoring.intent_verification_three_stage._call_openrouter",
            call,
        ):
            result = await verify_three_stage(
                object(),
                company_name="Acme",
                company_linkedin="https://www.linkedin.com/company/acme",
                company_website="https://acme.com",
                source_url="https://news.example/acme-funding",
                miner_claim="Acme raised a Series B",
                target_signal_text="The company recently raised funding",
                miner_signal_date="2026-07-01",
                stage1_soft_reject=False,
            )
        self.assertFalse(result["client_ready"])
        self.assertEqual(result["decision"], "unavailable")
        self.assertEqual(result["stage1"]["status"], "llm_error")

    async def test_stage_three_provider_error_is_unavailable_not_a_false_rejection(self):
        url = "https://news.example/acme-funding"
        call = AsyncMock(side_effect=[supported(url), {"_error": "http_403"}])
        fetch = AsyncMock(return_value={
            "results": [{
                "url": url,
                "title": "Acme funding",
                "text": "Acme raised a Series B on July 1, 2026.",
            }],
            "statuses": [{"source": "scrapingdog", "stage": "ok"}],
        })
        with (
            patch(
                "qualification.scoring.intent_verification_three_stage._call_openrouter",
                call,
            ),
            patch(
                "qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa",
                fetch,
            ),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Acme",
                company_linkedin="https://www.linkedin.com/company/acme",
                company_website="https://acme.com",
                source_url=url,
                miner_claim="Acme raised a Series B",
                target_signal_text="The company recently raised funding",
                miner_signal_date="2026-07-01",
                stage1_soft_reject=True,
            )
        self.assertFalse(result["client_ready"])
        self.assertEqual(result["decision"], "unavailable")
        self.assertEqual(result["stage3"]["status"], "llm_error")

    async def test_stage_one_approval_cannot_bypass_a_failed_evidence_fetch(self):
        url = "https://news.example/acme-funding"
        call = AsyncMock(return_value=supported(url))
        fetch = AsyncMock(return_value={
            "results": [],
            "statuses": [{"source": "scrapingdog", "stage": "timeout"},
                         {"source": "exa_fallback", "stage": "empty"}],
        })
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Acme",
                company_linkedin="https://www.linkedin.com/company/acme",
                company_website="https://acme.com",
                source_url=url,
                miner_claim="Acme raised a Series B",
                target_signal_text="The company recently raised funding",
                miner_signal_date="2026-07-01",
                stage1_soft_reject=True,
            )
        self.assertEqual(call.await_count, 1)
        fetch.assert_awaited_once_with([url])
        self.assertFalse(result["client_ready"])
        self.assertEqual(result["rejection_reason"], "evidence_fetch_failed")
        self.assertEqual(result["verdict"]["signal_evaluations"][0]["signal_status"], "unable_to_verify")

    async def test_stage_three_makes_the_terminal_decision_from_fetched_content(self):
        url = "https://news.example/acme-funding"
        call = AsyncMock(side_effect=[supported(url), supported(url)])
        fetch = AsyncMock(return_value={
            "results": [{"url": url, "title": "Acme funding", "text": "Acme raised a Series B on July 1, 2026."}],
            "statuses": [{"source": "exa_fallback", "stage": "ok"}],
        })
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Acme",
                company_linkedin="https://www.linkedin.com/company/acme",
                company_website="https://acme.com",
                source_url=url,
                miner_claim="Acme raised a Series B",
                target_signal_text="The company recently raised funding",
                miner_signal_date="2026-07-01",
                stage1_soft_reject=True,
            )
        self.assertEqual(call.await_count, 2)
        self.assertTrue(result["client_ready"])
        self.assertEqual(result["scrape"]["result_count"], 1)
        self.assertEqual(result["stage1"]["original_decision"], "approve")

    async def test_source_grounded_proof_can_overturn_only_a_blind_stage_one_reject(self):
        url = "https://news.example/acme-product-launch"
        call = AsyncMock(side_effect=[contradicted(url), supported(url)])
        fetch = AsyncMock(return_value={
            "results": [{
                "url": url,
                "title": "Acme launches Atlas",
                "text": "Acme launched its Atlas product on July 1, 2026.",
            }],
            "statuses": [{"source": "scrapingdog", "stage": "ok"}],
        })
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Acme",
                company_linkedin="https://www.linkedin.com/company/acme",
                company_website="https://acme.com",
                source_url=url,
                miner_claim="Acme launched Atlas",
                target_signal_text="The company recently launched a product",
                miner_signal_date="2026-07-01",
                evidence_type="PRODUCT_LAUNCH",
                stage1_soft_reject=True,
            )
        self.assertTrue(result["client_ready"])
        self.assertEqual(result["stage1"]["original_decision"], "reject")
        self.assertEqual(result["stage3"]["decision"], "approve")
        self.assertTrue(result["company_check"])

    async def test_medium_supported_claim_is_approved_after_independent_corroboration(self):
        original = "https://www.businesswire.com/news/home/barte-series-a"
        corroborating = "https://latamlist.com/barte-raises-series-a"
        call = AsyncMock(side_effect=[
            medium_supported(original),
            medium_supported(original),
            medium_supported(corroborating),
        ])
        fetch = AsyncMock(side_effect=[
            {
                "results": [{
                    "url": original,
                    "title": "Barte Series A",
                    "text": "Barte announced an $8 million Series A in October 2024.",
                }],
                "statuses": [{"source": "scrapingdog", "stage": "ok"}],
            },
            {
                "results": [{
                    "url": corroborating,
                    "title": "Barte raises $8M",
                    "text": "LatamList reports that Barte raised an $8 million Series A in October 2024.",
                }],
                "statuses": [{"source": "exa_fallback", "stage": "ok"}],
            },
        ])
        search = AsyncMock(return_value={
            "urls": [corroborating], "provider": "exa",
            "result_count": 1, "error": None,
        })
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
            patch("qualification.scoring.intent_verification_three_stage._search_exa_corroboration", search),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Barte",
                company_linkedin="https://www.linkedin.com/company/barte",
                company_website="https://barte.com",
                source_url=original,
                miner_claim="Barte raised an $8 million Series A",
                target_signal_text="The company recently raised funding",
                miner_signal_date="2024-10-01",
                stage1_soft_reject=True,
            )

        self.assertTrue(result["client_ready"])
        self.assertEqual(result["decision"], "approve")
        self.assertTrue(result["stage3"]["corroborated"])
        self.assertEqual(
            result["stage3"]["claim_matches_miner_date"],
            "no_date_in_content",
        )
        self.assertEqual(result["corroboration"]["independent_urls"], [corroborating])
        self.assertTrue(result["corroboration"]["cited_independent"])
        self.assertEqual(call.await_count, 3)

    async def test_medium_claim_without_corroboration_stays_rejected_even_with_review_override(self):
        original = "https://news.example/acme-series-b"
        call = AsyncMock(side_effect=[
            medium_supported(original), medium_supported(original),
        ])
        fetch = AsyncMock(return_value={
            "results": [{
                "url": original,
                "title": "Acme Series B",
                "text": "Acme announced a Series B in July 2026.",
            }],
            "statuses": [{"source": "scrapingdog", "stage": "ok"}],
        })
        search = AsyncMock(return_value={
            "urls": [], "provider": "exa", "result_count": 0,
            "error": "http_503",
        })
        with (
            patch.dict(os.environ, {"INTENT_VERIFIER_REVIEW_AS_ACCEPT": "on"}),
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
            patch("qualification.scoring.intent_verification_three_stage._search_exa_corroboration", search),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Acme",
                company_linkedin="https://www.linkedin.com/company/acme",
                company_website="https://acme.com",
                source_url=original,
                miner_claim="Acme raised a Series B",
                target_signal_text="The company recently raised funding",
                miner_signal_date="2026-07-01",
                stage1_soft_reject=True,
            )

        self.assertFalse(result["client_ready"])
        self.assertEqual(
            result["rejection_reason"],
            "corroboration_no_independent_corroboration",
        )
        self.assertEqual(result["corroboration"]["exa_error"], "http_503")
        self.assertEqual(call.await_count, 2)

    async def test_wire_syndication_does_not_count_as_independent(self):
        original = "https://www.businesswire.com/news/home/barte-series-a"
        mirror = "https://www.streetinsider.com/Business+Wire/barte-series-a"
        call = AsyncMock(side_effect=[
            medium_supported(original), medium_supported(original),
        ])
        fetch = AsyncMock(side_effect=[
            {
                "results": [{
                    "url": original, "title": "Barte Series A",
                    "text": "Barte announced an $8 million Series A in October 2024.",
                }],
                "statuses": [],
            },
            {
                "results": [{
                    "url": mirror, "title": "Barte Series A",
                    "text": "NEW YORK--(BUSINESS WIRE)--Barte announced an $8 million Series A.",
                }],
                "statuses": [],
            },
        ])
        search = AsyncMock(return_value={
            "urls": [mirror], "provider": "exa", "result_count": 1,
            "error": None,
        })
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
            patch("qualification.scoring.intent_verification_three_stage._search_exa_corroboration", search),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Barte",
                company_linkedin="https://www.linkedin.com/company/barte",
                company_website="https://barte.com",
                source_url=original,
                miner_claim="Barte raised an $8 million Series A",
                target_signal_text="The company recently raised funding",
                miner_signal_date="2024-10-01",
                stage1_soft_reject=True,
            )

        self.assertFalse(result["client_ready"])
        self.assertEqual(result["corroboration"]["independent_count"], 0)
        self.assertEqual(
            result["corroboration"]["excluded"][0]["reason"],
            "wire_syndication",
        )
        self.assertEqual(call.await_count, 2)

    async def test_perplexity_citations_are_fallback_when_exa_finds_nothing(self):
        original = "https://www.businesswire.com/news/home/barte-series-a"
        corroborating = "https://alleycorp.com/portfolio/barte"
        stage1 = medium_supported(original)
        stage1["citations"] = [corroborating]
        call = AsyncMock(side_effect=[
            stage1,
            medium_supported(original),
            medium_supported(corroborating, "consistent"),
        ])
        fetch = AsyncMock(side_effect=[
            {
                "results": [{
                    "url": original, "title": "Barte Series A",
                    "text": "Barte announced an $8 million Series A in October 2024.",
                }],
                "statuses": [],
            },
            {
                "results": [{
                    "url": corroborating, "title": "Barte",
                    "text": "AlleyCorp invested in Barte's $8 million Series A in October 2024.",
                }],
                "statuses": [],
            },
        ])
        search = AsyncMock(return_value={
            "urls": [], "provider": "exa", "result_count": 0,
            "error": None,
        })
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
            patch("qualification.scoring.intent_verification_three_stage._search_exa_corroboration", search),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Barte",
                company_linkedin="https://www.linkedin.com/company/barte",
                company_website="https://barte.com",
                source_url=original,
                miner_claim="Barte raised an $8 million Series A",
                target_signal_text="The company recently raised funding",
                miner_signal_date="2024-10-01",
                stage1_soft_reject=True,
            )

        self.assertTrue(result["client_ready"])
        self.assertEqual(
            result["corroboration"]["providers"],
            ["exa", "perplexity_citations"],
        )

    async def test_contradicted_stage_three_result_never_enters_corroboration(self):
        original = "https://news.example/acme-expansion"
        call = AsyncMock(side_effect=[
            medium_supported(original), contradicted(original),
        ])
        fetch = AsyncMock(return_value={
            "results": [{
                "url": original, "title": "Acme partnership",
                "text": "Acme signed a customer partnership; it did not enter a new market.",
            }],
            "statuses": [],
        })
        search = AsyncMock()
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
            patch("qualification.scoring.intent_verification_three_stage._search_exa_corroboration", search),
        ):
            result = await verify_three_stage(
                object(),
                company_name="Acme",
                company_linkedin="https://www.linkedin.com/company/acme",
                company_website="https://acme.com",
                source_url=original,
                miner_claim="Acme entered a new market",
                target_signal_text="The company recently expanded geographically",
                miner_signal_date="2026-07-01",
                stage1_soft_reject=True,
            )

        self.assertFalse(result["client_ready"])
        self.assertEqual(result["rejection_reason"], "stage3_contradicted")
        search.assert_not_awaited()

    async def test_exa_search_request_is_bounded_and_excludes_original_domain(self):
        original = "https://www.businesswire.com/news/home/barte-series-a"
        duplicate_domain = "https://businesswire.com/other-copy"
        independent = "https://latamlist.com/barte-raises-series-a"

        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://api.exa.ai/search")
            self.assertEqual(request.headers["x-api-key"], "test-exa-key")
            payload = json.loads(request.content)
            self.assertEqual(payload["numResults"], 6)
            self.assertEqual(payload["type"], "auto")
            self.assertIn("Barte", payload["query"])
            return httpx.Response(200, json={
                "results": [
                    {"url": original},
                    {"url": duplicate_domain},
                    {"url": independent},
                ],
            })

        transport = httpx.MockTransport(handler)
        row = {
            "company": "Barte",
            "claim": "Barte raised an $8 million Series A",
            "signal_date": "2024-10-01",
            "_target_signal_text": "The company recently raised funding",
            "claimed_source_urls": [original],
        }
        with patch.dict(os.environ, {"EXA_API_KEY": "test-exa-key"}):
            async with httpx.AsyncClient(transport=transport) as client:
                result = await _search_exa_corroboration(client, row)

        self.assertEqual(result["urls"], [independent])
        self.assertEqual(result["result_count"], 3)

    async def test_rescue_rejects_uncited_corroboration_and_date_contradiction(self):
        original = "https://news.example/acme-series-b"
        corroborating = "https://trade.example/acme-series-b"
        row = {
            "id": "signal-1",
            "company": "Acme",
            "website": "https://acme.com",
            "company_linkedin": "https://www.linkedin.com/company/acme",
            "contact_linkedin": "",
            "claim": "Acme raised a Series B",
            "signal_date": "2026-07-01",
            "signal_type": "intent",
            "claimed_source_urls": [original],
            "_target_signal_text": "The company recently raised funding",
            "_evidence_type": "FUNDING",
        }
        original_contents = {
            "results": [{
                "url": original, "title": "Acme funding",
                "text": "Acme announced a Series B in July 2026.",
            }],
            "statuses": [],
        }
        candidate_contents = {
            "results": [{
                "url": corroborating, "title": "Acme Series B",
                "text": "An independent report confirms Acme's Series B.",
            }],
            "statuses": [],
        }
        search_result = {
            "urls": [corroborating], "results": [], "provider": "exa",
            "result_count": 1, "error": None,
        }
        cases = [
            (medium_supported(original, "consistent"), False),
            (medium_supported(corroborating, "contradicted"), True),
        ]
        for final_verdict, cites_independent in cases:
            with self.subTest(cites_independent=cites_independent):
                with (
                    patch(
                        "qualification.scoring.intent_verification_three_stage._search_exa_corroboration",
                        AsyncMock(return_value=search_result),
                    ),
                    patch(
                        "qualification.scoring.intent_verification_three_stage._fetch_corroboration_candidates",
                        AsyncMock(return_value=candidate_contents),
                    ),
                    patch(
                        "qualification.scoring.intent_verification_three_stage._call_openrouter",
                        AsyncMock(return_value=final_verdict),
                    ),
                ):
                    result = await _rescue_medium_with_corroboration(
                        object(),
                        row=row,
                        original_contents=original_contents,
                        perplexity_citations=[],
                        stage3_model="test-model",
                    )

                self.assertFalse(result["approved"])
                self.assertEqual(
                    result["metadata"]["reason"],
                    "corroboration_not_confirmed",
                )
