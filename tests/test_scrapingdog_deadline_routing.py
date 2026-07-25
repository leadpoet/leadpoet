import asyncio
import unittest
from unittest import mock

import aiohttp
import httpx

from qualification.scoring import intent_verification_three_stage as intent
from validator_models import fulfillment_attribute_verification as attributes


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


class _AiohttpResponse:
    def __init__(self, status, body, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def text(self):
        return self._body


class _AiohttpContext:
    def __init__(self, outcome):
        self.outcome = outcome

    async def __aenter__(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    async def __aexit__(self, *_args):
        return False


class _AiohttpSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _AiohttpContext(self.outcomes.pop(0))


class IntentScrapingDogDeadlineTests(unittest.IsolatedAsyncioTestCase):
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

    def test_terminal_tier_has_provider_delivery_margin(self):
        self.assertGreater(
            intent._SD_TIER_TIMEOUT["full_combined"],
            intent.SCRAPINGDOG_PROVIDER_DEADLINE_S,
        )


class AttributeScrapingDogDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_deadline_stops_scrapingdog_ladder(self):
        session = _AiohttpSession([asyncio.TimeoutError()])
        with mock.patch.object(attributes, "SCRAPINGDOG_KEY", "test"), \
                mock.patch.object(
                    attributes,
                    "_wayback_fetch",
                    new=mock.AsyncMock(return_value=(False, "", "wayback_no_snapshot")),
                ):
            result = await attributes.fetch_url_via_scrapingdog(
                session,
                "https://unreachable.example",
            )

        self.assertFalse(result[0])
        self.assertEqual(len(session.calls), 1)
        self.assertIn("client_deadline:baseline", result[2])

    async def test_provider_5xx_stops_scrapingdog_ladder(self):
        session = _AiohttpSession([
            _AiohttpResponse(503, "provider unavailable"),
        ])
        with mock.patch.object(attributes, "SCRAPINGDOG_KEY", "test"), \
                mock.patch.object(
                    attributes,
                    "_wayback_fetch",
                    new=mock.AsyncMock(return_value=(False, "", "wayback_no_snapshot")),
                ):
            result = await attributes.fetch_url_via_scrapingdog(
                session,
                "https://provider-failure.example",
            )

        self.assertFalse(result[0])
        self.assertEqual(len(session.calls), 1)
        self.assertIn("http_503", result[2])

    async def test_transport_error_stops_scrapingdog_ladder(self):
        session = _AiohttpSession([aiohttp.ClientConnectionError("unreachable")])
        with mock.patch.object(attributes, "SCRAPINGDOG_KEY", "test"), \
                mock.patch.object(
                    attributes,
                    "_wayback_fetch",
                    new=mock.AsyncMock(return_value=(False, "", "wayback_no_snapshot")),
                ):
            result = await attributes.fetch_url_via_scrapingdog(
                session,
                "https://unreachable.example",
            )

        self.assertFalse(result[0])
        self.assertEqual(len(session.calls), 1)
        self.assertIn("transport_error:ClientConnectionError", result[2])

    def test_terminal_tier_has_provider_delivery_margin(self):
        self.assertGreater(
            attributes._SD_TIER_TIMEOUT["full_combined"],
            attributes.SCRAPINGDOG_PROVIDER_DEADLINE_S,
        )


if __name__ == "__main__":
    unittest.main()
