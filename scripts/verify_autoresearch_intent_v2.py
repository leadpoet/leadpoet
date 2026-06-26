"""Verify the opt-in autoresearch intent_v2 scorer without live API calls."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring.lead_scorer import (
    aggregate_autoresearch_intent_scores,
    score_company_autoresearch_intent_v2,
)
from research_lab.eval.evaluator import prepare_autoresearch_scoring_payload


class AutoresearchIntentAggregationTests(unittest.TestCase):
    def test_single_signal_keeps_score(self):
        self.assertEqual(aggregate_autoresearch_intent_scores([45]), 45)

    def test_two_signals_use_capped_sum_not_average(self):
        self.assertEqual(aggregate_autoresearch_intent_scores([30, 50]), 80)

    def test_equal_strong_signals_increase_until_six_signal_cap(self):
        scores = [
            aggregate_autoresearch_intent_scores([60] * count)
            for count in (3, 4, 5, 6)
        ]
        self.assertEqual(scores, [88, 92, 96, 100])
        self.assertTrue(scores[0] < scores[1] < scores[2] < scores[3])

    def test_only_top_six_signals_count(self):
        self.assertEqual(aggregate_autoresearch_intent_scores([60] * 7), 100)
        self.assertEqual(
            aggregate_autoresearch_intent_scores([1, 2, 3, 4, 5, 6, 100]),
            100,
        )

    def test_zero_duplicate_style_scores_do_not_raise_cap(self):
        self.assertEqual(aggregate_autoresearch_intent_scores([50, 0]), 50)


class AutoresearchAdapterTests(unittest.TestCase):
    def test_private_output_additional_intents_become_scored_signals(self):
        company = {
            "company_name": "Acme",
            "company_website": "https://acme.com",
            "industry": "Financial Services",
            "subindustry": "Fintech",
            "employee_count": "51-200",
            "hq_country": "United States",
            "intent": {
                "source": "news",
                "signal": "The company recently raised a funding round",
                "url": "https://news.example/acme-funding",
                "date": "2026-01-01",
            },
            "additional_intents": [
                {
                    "signal": "The company is actively hiring engineers",
                    "category": "HIRING",
                    "source": "job_listing",
                    "url": "https://acme.com/jobs",
                    "date": "2026-01-02",
                    "points": 999,
                },
                {
                    "signal": "Not requested",
                    "category": "OTHER",
                    "url": "https://acme.com/other",
                },
            ],
            "score": 999,
        }
        icp = {
            "icp_id": "icp-1",
            "industry": "Financial Services",
            "sub_industry": "Fintech",
            "geography": "United States",
            "employee_count": "51-200",
            "company_stage": "Any",
            "product_service": "Treasury automation",
            "intent_signal": "The company recently raised a funding round",
            "intent_category": "FUNDING",
            "bonus_intents": [
                {
                    "intent_signal": "The company is actively hiring engineers",
                    "intent_category": "HIRING",
                }
            ],
        }

        normalized_company, normalized_icp = prepare_autoresearch_scoring_payload(company, icp)

        self.assertEqual(normalized_icp["intent_signals"], [
            "The company recently raised a funding round",
            "The company is actively hiring engineers",
        ])
        self.assertEqual(len(normalized_company["intent_signals"]), 2)
        self.assertEqual(normalized_company["intent_signals"][0]["matched_icp_signal"], 0)
        self.assertEqual(normalized_company["intent_signals"][1]["matched_icp_signal"], 1)
        self.assertNotIn("score", normalized_company)


class AutoresearchScorerTests(unittest.TestCase):
    def _company(self, **overrides):
        data = {
            "company_name": "Acme",
            "company_website": "https://acme.com",
            "industry": "Financial Services",
            "sub_industry": "Fintech",
            "employee_count": "51-200",
            "company_stage": "Any",
            "country": "United States",
            "intent_signals": [
                {
                    "source": "news",
                    "description": "Acme raised a funding round.",
                    "url": "https://news.example/acme",
                    "date": "2026-01-01",
                    "snippet": "Acme raised capital.",
                    "matched_icp_signal": 0,
                }
            ],
        }
        data.update(overrides)
        return CompanyOutput(**data)

    def _icp(self, **overrides):
        data = {
            "icp_id": "icp-1",
            "prompt": "",
            "industry": "Financial Services",
            "sub_industry": "Fintech",
            "employee_count": "51-200",
            "company_stage": "Any",
            "geography": "United States",
            "country": "United States",
            "product_service": "Treasury automation",
            "intent_signals": ["The company recently raised a funding round"],
        }
        data.update(overrides)
        return ICPPrompt(**data)

    def test_binary_gate_failure_returns_zero(self):
        result = asyncio.run(score_company_autoresearch_intent_v2(
            company=self._company(employee_count="201-500"),
            icp=self._icp(),
            run_cost_usd=0,
            run_time_seconds=0,
            seen_companies=set(),
        ))
        self.assertEqual(result.final_score, 0)
        self.assertIn("Employee count mismatch", result.failure_reason)

    def test_passing_gates_use_capped_sum_and_zero_icp_fit(self):
        async def fake_signal_score(*_args, **_kwargs):
            return 50.0, 90, "verified", None, 0

        company = self._company(
            intent_signals=[
                {
                    "source": "news",
                    "description": "Acme raised a funding round.",
                    "url": "https://acmefundingnews.com/acme-1",
                    "date": "2026-06-01",
                    "snippet": "Acme raised capital.",
                    "matched_icp_signal": 0,
                },
                {
                    "source": "news",
                    "description": "Acme is hiring engineers.",
                    "url": "https://acmejobsboard.com/acme-2",
                    "date": "2026-06-02",
                    "snippet": "Acme hiring engineers.",
                    "matched_icp_signal": 1,
                },
            ]
        )
        icp = self._icp(
            intent_signals=[
                "The company recently raised a funding round",
                "The company is actively hiring engineers",
            ]
        )
        with (
            mock.patch(
                "qualification.scoring.lead_scorer.verify_company_exists",
                return_value=(True, "ok"),
            ),
            mock.patch(
                "qualification.scoring.lead_scorer._score_single_intent_signal",
                side_effect=fake_signal_score,
            ),
        ):
            result = asyncio.run(score_company_autoresearch_intent_v2(
                company=company,
                icp=icp,
                run_cost_usd=0,
                run_time_seconds=0,
                seen_companies=set(),
            ))
        self.assertEqual(result.icp_fit, 0)
        self.assertEqual(result.intent_signal_final, 80)
        self.assertEqual(result.final_score, 80)


if __name__ == "__main__":
    unittest.main()
