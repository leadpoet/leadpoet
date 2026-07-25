from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from gateway.qualification.models import (
    CompanyOutput,
    ICPPrompt,
    IntentSignal,
    IntentSignalSource,
)
from qualification.scoring.lead_scorer import score_company_autoresearch_intent_v2


class FreshnessSourceOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_signal_is_fetched_and_judged_before_terminal_rejection(self):
        company = CompanyOutput(
            company_name="Acme",
            company_website="https://acme.com",
            company_linkedin="https://www.linkedin.com/company/acme",
            industry="Financial Services",
            sub_industry="Payments",
            employee_count="201-500 employees",
            company_stage="",
            country="United States",
            state="NY",
            description="Payments infrastructure.",
            intent_signals=[IntentSignal(
                source=IntentSignalSource.NEWS,
                description="Acme raised a Series B",
                url="https://news.example/acme-funding",
                date="2020-01-01",
                snippet="The article names Acme and the financing.",
                matched_icp_signal=0,
            )],
        )
        icp = ICPPrompt(
            icp_id="request-1",
            prompt="Find US fintechs",
            industry="Financial Services",
            sub_industry="",
            employee_count="201-500",
            company_stage="",
            geography="United States",
            country="United States",
            product_service="Pipeline intelligence",
            intent_signals=["The company recently raised funding"],
            intent_signal_evidence_types=["FUNDING"],
            intent_max_age_days=90,
        )
        source_verifier = AsyncMock(return_value=(
            60.0,
            95,
            "verified",
            None,
            0,
        ))
        with patch(
            "qualification.scoring.lead_scorer._score_single_intent_signal",
            source_verifier,
        ):
            result = await score_company_autoresearch_intent_v2(
                company,
                icp,
                run_cost_usd=0,
                run_time_seconds=0,
                seen_companies=set(),
            )
        source_verifier.assert_awaited_once()
        self.assertEqual(result.final_score, 0)
        detail = result.intent_signals_detail[0]
        self.assertEqual(detail["date_status"], "out_of_window")
        self.assertEqual(detail["judge_verdict"]["rejection_reason"], "signal_out_of_window")


if __name__ == "__main__":
    unittest.main()
