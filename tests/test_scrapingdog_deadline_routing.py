import unittest
from unittest import mock

import httpx

from qualification.scoring import intent_verification_three_stage as intent


class _HttpxClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class IntentScrapingDogDeadlineTests(unittest.IsolatedAsyncioTestCase):
    def test_empty_visible_html_body_is_not_accepted(self):
        long_head = "<meta name='description' content='" + ("metadata " * 100) + "'>"
        empty_body = f"<!doctype html><html><head>{long_head}</head><body> </body></html>"
        scripts_only = (
            "<html><head>" + long_head + "</head><body>"
            "<template><script>window.payload = '" + ("data " * 200)
            + "';</script></template>"
            "<style>.hidden { display: none; }</style>"
            "<!-- metadata is not visible body text -->"
            "</body></html>"
        )

        self.assertEqual(intent._evaluate_sd_response(200, empty_body), "html_empty_body")
        self.assertEqual(intent._evaluate_sd_response(200, scripts_only), "html_empty_body")

    def test_literal_body_markup_in_head_does_not_hide_real_body(self):
        article = (
            "<!doctype html><html><head>"
            "<!-- literal <body></body> metadata -->"
            "<script>window.template = '<body></body>';</script>"
            "</head><body><article>"
            + ("The company announced generally available software. " * 100)
            + "</article></body></html>"
        )

        self.assertEqual(intent._evaluate_sd_response(200, article), "ok")

    def test_real_html_plain_text_and_json_keep_existing_acceptance(self):
        article = (
            "<!doctype html><html><head><title>News</title></head><body>"
            "<article><time>September 6, 2026</time>"
            + ("The company announced generally available software. " * 100)
            + "</article></body></html>"
        )
        plain_text = "The company announced generally available software. " * 100
        json_text = '{"article":"' + ("verified evidence " * 200) + '"}'

        self.assertEqual(intent._evaluate_sd_response(200, article), "ok")
        self.assertEqual(intent._evaluate_sd_response(200, plain_text), "ok")
        self.assertEqual(intent._evaluate_sd_response(200, json_text), "ok")

    async def test_client_deadline_stops_scrapingdog_ladder(self):
        client = _HttpxClient([httpx.ReadTimeout("client deadline")])
        with mock.patch.object(intent.httpx, "AsyncClient", return_value=client), \
                mock.patch.object(
                    intent,
                    "_try_wayback",
                    new=mock.AsyncMock(return_value={
                        "ok": False,
                        "stage": "wayback_no_snapshot",
                        "content": "",
                        "error": "no archived snapshot",
                    }),
                ), \
                mock.patch.dict("os.environ", {"SCRAPINGDOG_API_KEY": "test"}):
            result = await intent._scrape_sd_hardened("https://unreachable.example")

        self.assertFalse(result["ok"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            result["stage_history"][0],
            ("baseline", "client_deadline:baseline"),
        )

    async def test_provider_5xx_stops_scrapingdog_ladder(self):
        client = _HttpxClient([
            httpx.Response(503, text="provider unavailable", headers={}),
        ])
        with mock.patch.object(intent.httpx, "AsyncClient", return_value=client), \
                mock.patch.object(
                    intent,
                    "_try_wayback",
                    new=mock.AsyncMock(return_value={
                        "ok": False,
                        "stage": "wayback_no_snapshot",
                        "content": "",
                        "error": "no archived snapshot",
                    }),
                ), \
                mock.patch.dict("os.environ", {"SCRAPINGDOG_API_KEY": "test"}):
            result = await intent._scrape_sd_hardened("https://provider-failure.example")

        self.assertFalse(result["ok"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result["stage_history"][0], ("baseline", "http_503"))

    async def test_content_challenge_still_escalates(self):
        client = _HttpxClient([
            httpx.Response(200, text="<html>captcha" + ("x" * 700), headers={}),
            httpx.Response(200, text="<html><body>" + ("evidence " * 600), headers={}),
        ])
        with mock.patch.object(intent.httpx, "AsyncClient", return_value=client), \
                mock.patch.dict("os.environ", {"SCRAPINGDOG_API_KEY": "test"}):
            result = await intent._scrape_sd_hardened("https://challenged.example")

        self.assertTrue(result["ok"])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result["stage"], "sd:dynamic_render")

    async def test_empty_body_uses_full_native_cascade_then_exa_for_exact_url(self):
        url = "https://news.example/company-announcement"
        shell = (
            "<!doctype html><html><head>"
            + ("<meta name='description' content='metadata'>" * 30)
            + "</head><body><script>window.state = {};</script></body></html>"
        )
        extracted = "The company announced generally available software. " * 30
        client = _HttpxClient([
            httpx.Response(200, text=shell, headers={})
            for _ in intent._SD_TIERS
        ])
        wayback = mock.AsyncMock(return_value={
            "ok": False,
            "stage": "wayback_no_snapshot",
            "content": "",
            "error": "no archived snapshot",
        })
        exa = mock.AsyncMock(return_value={
            "ok": True,
            "stage": "exa_scraped",
            "content": extracted,
            "error": None,
        })
        with mock.patch.object(intent.httpx, "AsyncClient", return_value=client), \
                mock.patch.object(intent, "_try_wayback", new=wayback), \
                mock.patch.object(intent, "_scrape_exa", new=exa), \
                mock.patch.dict("os.environ", {"SCRAPINGDOG_API_KEY": "test"}):
            result = await intent._fetch_sd_then_exa([url])

        self.assertEqual(len(client.calls), len(intent._SD_TIERS))
        self.assertTrue(all(call[1]["params"]["url"] == url for call in client.calls))
        wayback.assert_awaited_once_with(url)
        exa.assert_awaited_once_with(url)
        self.assertEqual(result["results"][0]["url"], url)
        self.assertEqual(result["results"][0]["text"], extracted)
        self.assertEqual(result["statuses"][0]["source"], "exa_fallback")

    async def test_mislabeled_pdf_binary_skips_text_and_archive_paths(self):
        client = _HttpxClient([
            httpx.Response(
                200,
                content=b"%PDF-1.7\n" + (b"\x00\x01binary-stream" * 100),
                headers={"content-type": "text/csv"},
            ),
        ])
        wayback = mock.AsyncMock(
            side_effect=AssertionError("binary PDFs must use text extraction")
        )
        with mock.patch.object(intent.httpx, "AsyncClient", return_value=client), \
                mock.patch.object(intent, "_try_wayback", new=wayback), \
                mock.patch.dict("os.environ", {"SCRAPINGDOG_API_KEY": "test"}):
            result = await intent._scrape_sd_hardened(
                "https://investors.example/report.pdf"
            )

        self.assertFalse(result["ok"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result["stage_history"], [("baseline", "pdf_binary")])
        self.assertEqual(result["stage"], "all_tiers_exhausted:pdf_binary")
        self.assertEqual(result["content"], "")
        wayback.assert_not_awaited()

    async def test_plain_text_with_csv_content_type_is_unchanged(self):
        text = "Tyro announced a verified company event. " * 100
        client = _HttpxClient([
            httpx.Response(
                200,
                text=text,
                headers={"content-type": "text/csv"},
            ),
        ])
        with mock.patch.object(intent.httpx, "AsyncClient", return_value=client), \
                mock.patch.dict("os.environ", {"SCRAPINGDOG_API_KEY": "test"}):
            result = await intent._scrape_sd_hardened(
                "https://investors.example/announcement"
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "sd:baseline")
        self.assertEqual(result["content"], text)
        self.assertEqual(len(client.calls), 1)

    async def test_pdf_uses_existing_exa_text_extraction_before_stage_three(self):
        extracted = (
            "Tyro Payments announced the verified event on November 5, 2025."
        )
        with mock.patch.object(
            intent,
            "_scrape_sd_hardened",
            new=mock.AsyncMock(return_value={
                "ok": False,
                "stage": "all_tiers_exhausted:pdf_binary",
                "content": "",
                "error": "pdf_binary",
            }),
        ), mock.patch.object(
            intent,
            "_scrape_exa",
            new=mock.AsyncMock(return_value={
                "ok": True,
                "stage": "exa_scraped",
                "content": extracted,
                "error": None,
            }),
        ):
            result = await intent._fetch_sd_then_exa([
                "https://investors.example/report.pdf"
            ])

        self.assertEqual(result["results"][0]["text"], extracted)
        self.assertEqual(result["statuses"][0]["source"], "exa_fallback")
        prompt = intent._build_final_judge_prompt(
            {
                "id": "signal-1",
                "company": "tyro.com",
                "website": "tyro.com",
                "claim": "Tyro announced a verified company event.",
                "claimed_source_urls": [
                    "https://investors.example/report.pdf"
                ],
            },
            result,
        )
        self.assertNotRegex(prompt, r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
        self.assertIn(extracted, prompt)

    def test_terminal_tier_has_provider_delivery_margin(self):
        self.assertGreater(
            intent._SD_TIER_TIMEOUT["full_combined"],
            intent.SCRAPINGDOG_PROVIDER_DEADLINE_S,
        )


if __name__ == "__main__":
    unittest.main()
