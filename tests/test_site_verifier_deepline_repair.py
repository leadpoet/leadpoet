from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from leadpoet_verifier.deepline_repair import (
    DeeplineEvidenceRepairClient,
    DeeplineEvidenceRepairUnavailable,
    PLAY_NAME,
)


class DeeplineEvidenceRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_http_failure_retries_once_then_succeeds(self):
        response = httpx.Response(
            503,
            request=httpx.Request("POST", "https://code.deepline.com/api/v2/plays/run"),
        )
        transport = AsyncMock(side_effect=[
            httpx.HTTPStatusError(
                "unavailable", request=response.request, response=response,
            ),
            {
                "status": "completed",
                "output": {"sources": [{"url": "https://acme.example/about"}]},
            },
        ])
        client = DeeplineEvidenceRepairClient(api_key="secret", transport=transport)
        with patch(
            "leadpoet_verifier.deepline_repair.asyncio.sleep",
            new=AsyncMock(),
        ):
            sources = await client.repair(
                company_name="Acme",
                company_domain="acme.example",
                requested_criterion="Privately held",
                evidence_kind="required_attribute",
                existing_url=None,
            )

        self.assertEqual(sources[0]["url"], "https://acme.example/about")
        self.assertEqual(transport.await_count, 2)

    async def test_deterministic_http_failure_is_sanitized_and_not_retried(self):
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://code.deepline.com/api/v2/plays/run"),
        )
        transport = AsyncMock(side_effect=httpx.HTTPStatusError(
            "bad request", request=response.request, response=response,
        ))
        client = DeeplineEvidenceRepairClient(api_key="secret", transport=transport)

        with self.assertRaises(DeeplineEvidenceRepairUnavailable) as raised:
            await client.repair(
                company_name="Acme",
                company_domain="acme.example",
                requested_criterion="Privately held",
                evidence_kind="required_attribute",
                existing_url=None,
            )

        self.assertEqual(transport.await_count, 1)
        self.assertEqual(raised.exception.code, "deepline_http_400")
        self.assertEqual(raised.exception.receipt(), {
            "reason_code": "deepline_http_400",
            "status_code": 400,
            "endpoint": "plays_run_start",
            "retryable": False,
        })

    async def test_immediate_result_uses_bounded_grounded_input(self):
        transport = AsyncMock(return_value={
            "status": "completed",
            "output": {"sources": [{"url": "https://acme.example/about", "excerpt": "x"}]},
        })
        client = DeeplineEvidenceRepairClient(
            api_key="secret",
            transport=transport,
        )

        sources = await client.repair(
            company_name="Acme",
            company_domain="acme.example",
            requested_criterion="Privately held",
            evidence_kind="required_attribute",
            existing_url="https://acme.example/search",
        )

        self.assertEqual(sources[0]["url"], "https://acme.example/about")
        method, url, body = transport.await_args.args
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://code.deepline.com/api/v2/plays/run")
        self.assertEqual(body["name"], PLAY_NAME)
        self.assertEqual(body["input"]["requested_criterion"], "Privately held")
        self.assertNotIn("candidate", body["input"])

    async def test_polling_stops_at_first_completed_source(self):
        transport = AsyncMock(side_effect=[
            {"workflowId": "run/id", "status": "running"},
            {"status": "running"},
            {
                "status": "completed",
                "data": {"sources": [{"url": "https://acme.example/about", "excerpt": "x"}]},
            },
        ])
        client = DeeplineEvidenceRepairClient(
            api_key="secret",
            timeout_seconds=10,
            poll_seconds=0.1,
            transport=transport,
        )
        with patch(
            "leadpoet_verifier.deepline_repair.asyncio.sleep",
            new=AsyncMock(),
        ):
            sources = await client.repair(
                company_name="Acme",
                company_domain="acme.example",
                requested_criterion="Privately held",
                evidence_kind="required_attribute",
                existing_url=None,
            )

        self.assertEqual(len(sources), 1)
        self.assertEqual(transport.await_count, 3)
        self.assertIn("run%2Fid?full=true", transport.await_args.args[1])

    async def test_terminal_failure_is_explicit(self):
        client = DeeplineEvidenceRepairClient(
            api_key="secret",
            transport=AsyncMock(return_value={"id": "run-1", "status": "failed"}),
        )
        with self.assertRaisesRegex(
            DeeplineEvidenceRepairUnavailable,
            "deepline_repair_failed",
        ):
            await client.repair(
                company_name="Acme",
                company_domain="acme.example",
                requested_criterion="Privately held",
                evidence_kind="required_attribute",
                existing_url=None,
            )

    async def test_polling_uses_legacy_fallback_only_after_primary_404(self):
        response = httpx.Response(
            404,
            request=httpx.Request("GET", "https://code.deepline.com/api/v2/runs/run-1"),
        )
        transport = AsyncMock(side_effect=[
            {"workflowId": "run-1", "status": "running"},
            httpx.HTTPStatusError("not found", request=response.request, response=response),
            {
                "status": "completed",
                "output": {"sources": [{"url": "https://acme.example", "excerpt": "x"}]},
            },
        ])
        client = DeeplineEvidenceRepairClient(
            api_key="secret",
            timeout_seconds=10,
            poll_seconds=0.1,
            transport=transport,
        )
        with patch(
            "leadpoet_verifier.deepline_repair.asyncio.sleep",
            new=AsyncMock(),
        ):
            sources = await client.repair(
                company_name="Acme",
                company_domain="acme.example",
                requested_criterion="Privately held",
                evidence_kind="required_attribute",
                existing_url=None,
            )

        self.assertEqual(len(sources), 1)
        self.assertIn("/api/v2/plays/run/run-1?full=true", transport.await_args.args[1])

    def test_environment_flag_is_strictly_gated_by_key(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(DeeplineEvidenceRepairClient.from_env())
        with patch.dict(
            os.environ,
            {"VERIFIER_DEEPLINE_EVIDENCE_REPAIR_ENABLED": "true"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires DEEPLINE_API_KEY"):
                DeeplineEvidenceRepairClient.from_env()


if __name__ == "__main__":
    unittest.main()
