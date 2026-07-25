from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

from leadpoet_verifier.exa_repair import (
    MAX_RESULTS,
    ExaEvidenceRepairClient,
    ExaEvidenceRepairUnavailable,
)


class ExaEvidenceRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_bounded_search_returns_only_public_unique_urls(self):
        transport = AsyncMock(return_value={
            "_leadpoet_request_id": "req_123",
            "costDollars": {"total": 0.0123},
            "results": [
                {"url": "https://acme.example/about"},
                {"url": "https://acme.example/about"},
                {"url": "http://localhost/private"},
                {"url": "http://10.0.0.1/private"},
                {"url": "https://regulator.example/acme"},
            ],
        })
        client = ExaEvidenceRepairClient(api_key="secret", transport=transport)

        sources = await client.repair(
            company_name="Acme",
            company_domain="acme.example",
            requested_criterion="Operates a regulated digital asset exchange",
            evidence_kind="industry",
            existing_url=None,
        )

        self.assertEqual(
            [source["url"] for source in sources],
            [
                "https://acme.example/about",
                "https://regulator.example/acme",
            ],
        )
        self.assertEqual(sources[0]["provider_request_id"], "req_123")
        self.assertEqual(sources[0]["provider_cost_usd"], 0.0123)
        payload = transport.await_args.args[0]
        self.assertEqual(payload["numResults"], MAX_RESULTS)
        self.assertLessEqual(len(payload["query"]), 2_000)
        self.assertNotIn("contents", payload)
        self.assertNotIn("candidate", payload)

    async def test_invalid_response_fails_closed(self):
        client = ExaEvidenceRepairClient(
            api_key="secret",
            transport=AsyncMock(return_value=[]),
        )
        with self.assertRaisesRegex(
            ExaEvidenceRepairUnavailable,
            "invalid_exa_response",
        ):
            await client.repair(
                company_name="Acme",
                company_domain="acme.example",
                requested_criterion="Privately held",
                evidence_kind="required_attribute",
                existing_url=None,
            )

    async def test_empty_result_retains_request_and_cost_receipt(self):
        client = ExaEvidenceRepairClient(
            api_key="secret",
            transport=AsyncMock(return_value={
                "_leadpoet_request_id": "req_empty",
                "costDollars": {"total": 0.004},
                "results": [],
            }),
        )
        receipt = await client.repair(
            company_name="Acme",
            company_domain="acme.example",
            requested_criterion="Privately held",
            evidence_kind="required_attribute",
            existing_url=None,
        )
        self.assertIsNone(receipt[0]["url"])
        self.assertEqual(receipt[0]["result_count"], 0)
        self.assertEqual(receipt[0]["provider_request_id"], "req_empty")
        self.assertEqual(receipt[0]["provider_cost_usd"], 0.004)

    def test_legacy_flag_migrates_to_direct_exa_without_deepline_key(self):
        with patch.dict(
            os.environ,
            {
                "VERIFIER_DEEPLINE_EVIDENCE_REPAIR_ENABLED": "true",
                "EXA_API_KEY": "exa-secret",
            },
            clear=True,
        ):
            self.assertIsNotNone(ExaEvidenceRepairClient.from_env())

    def test_direct_flag_requires_exa_key(self):
        with patch.dict(
            os.environ,
            {"VERIFIER_EXA_EVIDENCE_REPAIR_ENABLED": "true"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires EXA_API_KEY"):
                ExaEvidenceRepairClient.from_env()


if __name__ == "__main__":
    unittest.main()
