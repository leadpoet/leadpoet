from __future__ import annotations

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from qualification.scoring.intent_verification_three_stage import (
    _apply_guardrails,
    _canonical_target_absence_receipt,
    _canonical_target_crawl_failure_receipt,
    _exa_target_absence_receipt,
    _exa_target_crawl_failure_receipt,
    _fetch_sd_then_exa,
    _project_contents_for_prompt,
    _scrape_exa,
    _scrape_sd_hardened,
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


class SourceGroundingTests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_context_is_exact_cited_fetched_content_only(self):
        url = "https://acme.example/news/leadership"
        body = "Acme appointed a new chief executive. Effective today, the executive leads technology and operations."
        verdict = supported(url)
        verdict["answer"]["signal_evaluations"][0]["supporting_quotes"] = [body.split(" Effective")[0]]
        uncertain = supported(url)
        uncertain["answer"]["signal_evaluations"][0]["signal_status"] = "unable_to_verify"
        call = AsyncMock(side_effect=[uncertain, verdict])
        fetch = AsyncMock(return_value={"results": [
            {"url": url, "text": body, "source_publication_date": "2026-09-14"},
            {"url": "https://unrelated.example/", "text": "UNRELATED SOURCE"},
        ], "statuses": []})
        with (
            patch("qualification.scoring.intent_verification_three_stage._call_openrouter", call),
            patch("qualification.scoring.intent_verification_three_stage._fetch_sd_then_exa", fetch),
        ):
            result = await verify_three_stage(
                object(), company_name="Acme", company_linkedin="https://linkedin.com/company/acme",
                company_website="https://acme.example", source_url=url,
                miner_claim="Acme appointed a new chief executive.",
                target_signal_text="Announced a leadership change in the past year.",
                miner_signal_date="2026-09-14", evidence_type="LEADERSHIP_CHANGE",
                declared_source="news", integrity_policy=True,
            )
        assert result["client_ready"] is True
        assert result["verified_source_context"] == [{
            "url": url, "text": body, "source_publication_date": "2026-09-14",
        }]
        assert fetch.await_count == 1

    async def _scrape_exa_outcomes(self, url: str, outcomes: list[object]):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            outcome = outcomes[calls]
            calls += 1
            if isinstance(outcome, BaseException):
                raise outcome
            status, document = outcome
            return httpx.Response(status, request=request, json=document)

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"EXA_API_KEY": "test-key"}),
            patch(
                "qualification.scoring.intent_verification_three_stage.httpx.AsyncClient",
                side_effect=lambda *args, **kwargs: real_async_client(
                    transport=transport
                ),
            ),
            patch(
                "qualification.scoring.intent_verification_three_stage.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            result = await _scrape_exa(url)
        return result, calls

    async def _scrape_exa_document(self, url: str, document: object):
        return await self._scrape_exa_outcomes(
            url, [(200, document), (200, document)]
        )

    async def _verify_empty_fetch(self, url: str, statuses: list[dict]):
        call = AsyncMock(return_value=supported(url))
        fetch = AsyncMock(
            return_value={"results": [], "statuses": statuses}
        )
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
        return result, call, fetch

    def test_guardrail_rejects_same_domain_different_evidence_path(self):
        supplied = "https://news.example/exact-article"
        verdict = supported("https://news.example/different-article")

        guarded = _apply_guardrails(
            {"claimed_source_urls": [supplied]},
            verdict["answer"],
        )

        item = guarded["signal_evaluations"][0]
        self.assertEqual(item["signal_status"], "unable_to_verify")
        self.assertEqual(item["source_urls_supplied"], [supplied])

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

    async def test_domain_absence_defers_to_stage_three_entity_authority(self):
        url = "https://news.example/terminal-project"
        stage_three = contradicted(url)
        stage_three["answer"]["signal_evaluations"][0].update(
            signal_status="wrong_entity",
            same_entity_check="fail",
        )
        call = AsyncMock(side_effect=[supported(url), stage_three])
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
        self.assertIsNone(result["company_check"])
        self.assertEqual(
            result["rejection_reason"],
            "stage3_wrong_entity",
        )
        self.assertEqual(call.await_count, 2)

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

    async def test_exa_retries_a_successful_empty_envelope(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    request=request,
                    json={"results": [], "statuses": [{"status": "pending"}]},
                )
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

    async def test_exa_preserves_exact_target_not_found_receipt(self):
        url = "https://news.example/acme-funding"
        result, calls = await self._scrape_exa_document(url, {
            "results": [],
            "statuses": [{
                "id": url,
                "status": "error",
                "error": {
                    "tag": "CRAWL_NOT_FOUND",
                    "httpStatusCode": 404,
                },
            }],
        })

        self.assertEqual(calls, 2)
        self.assertEqual(result, {
            "ok": False,
            "stage": "exa_target_not_found",
            "content": "",
            "error": "target_not_found",
            "target_absence": {
                "id_matches_requested_url": True,
                "status": "error",
                "error_tag": "CRAWL_NOT_FOUND",
                "error_http_status": 404,
                "confirmed_attempts": 2,
            },
        })

    async def test_exa_preserves_two_exact_target_crawl_failures(self):
        url = "https://news.example/acme-funding"
        documents = [
            {
                "results": [],
                "statuses": [{
                    "id": url, "status": "error",
                    "error": {
                        "tag": "CRAWL_LIVECRAWL_TIMEOUT",
                        "httpStatusCode": 504,
                    },
                }],
            },
            {
                "results": [],
                "statuses": [{
                    "id": url, "status": "error",
                    "error": {
                        "tag": "CRAWL_UNKNOWN_ERROR",
                        "httpStatusCode": 500,
                    },
                }],
            },
        ]
        result, calls = await self._scrape_exa_outcomes(
            url, [(200, document) for document in documents]
        )

        self.assertEqual(calls, 2)
        self.assertEqual(result["stage"], "exa_no_results")
        self.assertEqual(result["target_crawl_failure"], {
            "id_matches_requested_url": True,
            "confirmed_attempts": 2,
            "observations": [
                {"error_tag": "CRAWL_LIVECRAWL_TIMEOUT", "error_http_status": 504},
                {"error_tag": "CRAWL_UNKNOWN_ERROR", "error_http_status": 500},
            ],
        })
        projected = _project_contents_for_prompt({
            "results": [],
            "statuses": [{
                "url": url,
                "source": "none",
                "sd_stage": "all_tiers_exhausted:http_502",
                "exa_stage": "exa_no_results",
                "exa_error": "private provider detail",
                "exa_target_crawl_failure": result["target_crawl_failure"],
            }],
        })
        self.assertEqual(
            projected["statuses"][0]["exa_target_crawl_failure"],
            result["target_crawl_failure"],
        )
        self.assertNotIn("exa_error", projected["statuses"][0])

    async def test_exa_endpoint_500_has_no_target_crawl_failure_receipt(self):
        result, calls = await self._scrape_exa_outcomes(
            "https://news.example/acme-funding",
            [(500, {}), (500, {})],
        )
        self.assertEqual(calls, 2)
        self.assertEqual(result["stage"], "exa_transient_exhausted")
        self.assertNotIn("target_crawl_failure", result)

    def test_exa_target_crawl_failure_receipt_is_strict(self):
        url = "https://news.example/acme-funding"
        exact = {
            "results": [],
            "statuses": [{
                "id": url, "status": "error",
                "error": {
                    "tag": "CRAWL_LIVECRAWL_TIMEOUT",
                    "httpStatusCode": 504,
                },
            }],
        }
        self.assertEqual(
            _exa_target_crawl_failure_receipt(exact, url),
            {"error_tag": "CRAWL_LIVECRAWL_TIMEOUT", "error_http_status": 504},
        )
        for document in (
            {**exact, "statuses": [{**exact["statuses"][0], "id": url + "/other"}]},
            {**exact, "statuses": [{**exact["statuses"][0], "error": {
                "tag": "CRAWL_LIVECRAWL_TIMEOUT", "httpStatusCode": 429,
            }}]},
            {**exact, "statuses": [{**exact["statuses"][0], "error": {
                "tag": ["CRAWL_LIVECRAWL_TIMEOUT"], "httpStatusCode": 504,
            }}]},
            {"results": [], "statuses": [{"status": "error"}]},
        ):
            self.assertIsNone(_exa_target_crawl_failure_receipt(document, url))
        self.assertIsNone(_canonical_target_crawl_failure_receipt({
            "id_matches_requested_url": True,
            "confirmed_attempts": 2,
            "observations": [{
                "error_tag": "CRAWL_LIVECRAWL_TIMEOUT",
                "error_http_status": 504,
            }],
        }))

    def test_exa_target_not_found_receipt_is_strict(self):
        url = "https://news.example/acme-funding"
        exact_status = {
            "id": url,
            "status": "error",
            "error": {
                "tag": "CRAWL_NOT_FOUND",
                "httpStatusCode": 404,
            },
        }
        documents = (
            {"results": {}, "statuses": [exact_status]},
            {"results": [], "statuses": [{
                **exact_status,
                "id": "https://news.example/a-different-page",
            }]},
            {"results": [], "statuses": [{
                **exact_status,
                "error": {
                    "tag": "CRAWL_TIMEOUT",
                    "httpStatusCode": 504,
                },
            }]},
            {"results": [], "statuses": [{
                **exact_status,
                "error": {
                    "tag": "SOURCE_NOT_AVAILABLE",
                    "httpStatusCode": 403,
                },
            }]},
            {"results": [], "statuses": [exact_status, exact_status]},
            {"results": [], "statuses": [{
                **exact_status,
                "error": {"tag": "CRAWL_NOT_FOUND"},
            }]},
            {"results": [], "statuses": [{
                **exact_status,
                "error": {
                    "tag": "CRAWL_NOT_FOUND",
                    "httpStatusCode": 404.0,
                },
            }]},
        )
        for document in documents:
            with self.subTest(document=document):
                self.assertIsNone(
                    _exa_target_absence_receipt(document, url)
                )

        canonical = {
            "id_matches_requested_url": True,
            "status": "error",
            "error_tag": "CRAWL_NOT_FOUND",
            "error_http_status": 404,
            "confirmed_attempts": 2,
        }
        self.assertEqual(
            _canonical_target_absence_receipt(canonical), canonical
        )
        for field, value in (
            ("id_matches_requested_url", 1),
            ("error_http_status", 404.0),
            ("confirmed_attempts", 2.0),
        ):
            with self.subTest(field=field, value=value):
                self.assertIsNone(_canonical_target_absence_receipt({
                    **canonical,
                    field: value,
                }))

    async def test_exa_requires_matching_absence_on_both_attempts(self):
        url = "https://news.example/acme-funding"
        exact_absence = {
            "results": [],
            "statuses": [{
                "id": url,
                "status": "error",
                "error": {
                    "tag": "CRAWL_NOT_FOUND",
                    "httpStatusCode": 404,
                },
            }],
        }
        mixed_first_outcomes = (
            (200, {"results": [], "statuses": [{"status": "pending"}]}),
            (200, {"results": [{"text": "too short"}], "statuses": []}),
            (200, {
                "results": [],
                "statuses": [{
                    "id": "https://news.example/a-different-page",
                    "status": "error",
                    "error": {
                        "tag": "CRAWL_NOT_FOUND",
                        "httpStatusCode": 404,
                    },
                }],
            }),
            (429, {}),
            (503, {}),
            httpx.TimeoutException("timed out"),
        )
        for first in mixed_first_outcomes:
            with self.subTest(first=first):
                result, calls = await self._scrape_exa_outcomes(
                    url, [first, (200, exact_absence)]
                )

                self.assertEqual(calls, 2)
                self.assertNotIn("target_absence", result)

    async def test_sd_requires_every_attempt_to_return_target_404(self):
        async def scrape(responses):
            calls = 0

            def handler(request: httpx.Request) -> httpx.Response:
                nonlocal calls
                status, body = responses[calls]
                calls += 1
                return httpx.Response(status, request=request, text=body)

            transport = httpx.MockTransport(handler)
            real_async_client = httpx.AsyncClient
            with (
                patch.dict(os.environ, {"SCRAPINGDOG_API_KEY": "test-key"}),
                patch(
                    "qualification.scoring.intent_verification_three_stage.httpx.AsyncClient",
                    side_effect=lambda *args, **kwargs: real_async_client(
                        transport=transport
                    ),
                ),
            ):
                return await _scrape_sd_hardened(
                    "https://news.example/acme-funding"
                )

        confirmed = await scrape([(404, ""), (404, "")])
        mixed = await scrape([(200, "short"), (404, "")])

        self.assertEqual(confirmed["stage"], "genuine_404")
        self.assertNotEqual(mixed["stage"], "genuine_404")
        self.assertEqual(
            [item[1] for item in mixed["stage_history"]],
            ["body_too_short", "http_404"],
        )

    async def test_fetch_preserves_both_exact_absence_receipts(self):
        url = "https://news.example/acme-funding"
        sd = AsyncMock(return_value={
            "ok": False,
            "stage": "genuine_404",
            "content": "",
            "error": "http_404",
        })
        exa = AsyncMock(return_value={
            "ok": False,
            "stage": "exa_target_not_found",
            "content": "",
            "error": "target_not_found",
            "target_absence": {
                "id_matches_requested_url": True,
                "status": "error",
                "error_tag": "CRAWL_NOT_FOUND",
                "error_http_status": 404,
                "confirmed_attempts": 2,
            },
        })
        with (
            patch(
                "qualification.scoring.intent_verification_three_stage._scrape_sd_hardened",
                sd,
            ),
            patch(
                "qualification.scoring.intent_verification_three_stage._scrape_exa",
                exa,
            ),
        ):
            result = await _fetch_sd_then_exa([url])

        self.assertEqual(result, {
            "results": [],
            "statuses": [{
                "url": url,
                "source": "none",
                "sd_stage": "genuine_404",
                "sd_error": "http_404",
                "exa_stage": "exa_target_not_found",
                "exa_error": "target_not_found",
                "exa_target_absence": {
                    "id_matches_requested_url": True,
                    "status": "error",
                    "error_tag": "CRAWL_NOT_FOUND",
                    "error_http_status": 404,
                    "confirmed_attempts": 2,
                },
            }],
        })

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
        status_cases = (
            [{"source": "scrapingdog", "stage": "timeout"},
             {"source": "exa_fallback", "stage": "empty"}],
            [{
                "source": "none",
                "sd_stage": "all_tiers_exhausted:html_empty_body",
                "exa_stage": "exa_empty",
            }],
        )
        for statuses in status_cases:
            with self.subTest(statuses=statuses):
                call = AsyncMock(return_value=supported(url))
                fetch = AsyncMock(return_value={"results": [], "statuses": statuses})
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
                self.assertEqual(result["decision"], "unavailable")
                self.assertEqual(result["rejection_reason"], "evidence_fetch_failed")
                self.assertEqual(
                    result["verdict"]["signal_evaluations"][0]["signal_status"],
                    "unable_to_verify",
                )

    async def test_confirmed_source_absence_is_semantic_not_infrastructure(self):
        url = "https://news.example/acme-funding"
        statuses = [{
            "url": url,
            "source": "none",
            "sd_stage": "genuine_404",
            "sd_error": "http_404",
            "exa_stage": "exa_target_not_found",
            "exa_error": "target_not_found",
            "exa_target_absence": {
                "id_matches_requested_url": True,
                "status": "error",
                "error_tag": "CRAWL_NOT_FOUND",
                "error_http_status": 404,
                "confirmed_attempts": 2,
            },
        }]
        result, call, _fetch = await self._verify_empty_fetch(url, statuses)

        self.assertEqual(call.await_count, 1)
        self.assertFalse(result["client_ready"])
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["rejection_reason"], "evidence_not_found")
        self.assertEqual(result["scrape"]["result_count"], 0)
        self.assertEqual(
            result["scrape"]["statuses"][0]["exa_target_absence"],
            statuses[0]["exa_target_absence"],
        )

    async def test_source_absence_requires_both_exact_provider_receipts(self):
        url = "https://news.example/acme-funding"
        status_cases = (
            [{
                "url": url,
                "source": "none",
                "sd_stage": "genuine_404",
                "sd_error": "http_404",
                "exa_stage": "exa_no_results",
            }],
            [{
                "url": url,
                "source": "none",
                "sd_stage": "all_tiers_exhausted:client_deadline",
                "exa_stage": "exa_target_not_found",
                "exa_target_absence": {
                    "id_matches_requested_url": True,
                    "status": "error",
                    "error_tag": "CRAWL_NOT_FOUND",
                    "error_http_status": 404,
                    "confirmed_attempts": 2,
                },
            }],
            [{
                "url": url,
                "source": "none",
                "sd_stage": "genuine_404",
                "sd_error": "http_404",
                "exa_stage": "exa_target_not_found",
                "exa_target_absence": {
                    "id_matches_requested_url": False,
                    "status": "error",
                    "error_tag": "CRAWL_NOT_FOUND",
                    "error_http_status": 404,
                    "confirmed_attempts": 2,
                },
            }],
        )
        for statuses in status_cases:
            with self.subTest(statuses=statuses):
                result, _call, _fetch = await self._verify_empty_fetch(
                    url, statuses
                )

                self.assertEqual(result["decision"], "unavailable")
                self.assertEqual(
                    result["rejection_reason"], "evidence_fetch_failed"
                )

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
