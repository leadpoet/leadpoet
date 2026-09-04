from __future__ import annotations

import asyncio
import os
import unittest
from unittest import mock

from qualification.scoring.lead_scorer import (
    _decision_from_observed_employee_size,
    _llm_reverify_company,
    _normalize_icp_employee_buckets,
    _normalize_linkedin_employee_bucket,
)
from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
)
from research_lab.employee_buckets import (
    LINKEDIN_EMPLOYEE_BUCKETS,
    normalize_observed_employee_count_bucket,
)


class EmployeeSizeBucketTests(unittest.TestCase):
    def test_every_linkedin_bucket_round_trips_exactly(self):
        for bucket in LINKEDIN_EMPLOYEE_BUCKETS:
            with self.subTest(bucket=bucket):
                self.assertEqual(_normalize_linkedin_employee_bucket(bucket), bucket)
                self.assertEqual(_normalize_icp_employee_buckets([bucket]), ({bucket}, True))

    def test_comma_ranges_are_not_split_as_lists(self):
        self.assertEqual(
            _normalize_icp_employee_buckets("501-1,000 | 1,001-5,000"),
            ({"501-1,000", "1,001-5,000"}, True),
        )

    def test_mixed_legacy_and_canonical_default_bands_are_deduplicated(self):
        production_default = [
            "11-50", "51-200", "201-500", "501-1000", "1001-5000",
            "5001-10000", "10000+", "1-10", "501-1,000",
            "1,001-5,000", "5,001-10,000", "10,001+",
        ]
        self.assertEqual(
            _normalize_icp_employee_buckets(production_default),
            ({
                "2-10", "11-50", "51-200", "201-500", "501-1,000",
                "1,001-5,000", "5,001-10,000", "10,001+",
            }, True),
        )

    def test_known_legacy_bands_map_to_exact_linkedin_buckets(self):
        self.assertEqual(
            _normalize_icp_employee_buckets(["1-10", "501-1000", "10000+"]),
            ({"2-10", "501-1,000", "10,001+"}, True),
        )

    def test_unknown_and_malformed_icp_sizes_fail_closed(self):
        for value in (None, "", "any", "about 500", "500ish", "eleven to fifty"):
            with self.subTest(value=value):
                self.assertEqual(_normalize_icp_employee_buckets(value), (set(), False))

    def test_exact_observed_counts_project_at_every_bucket_boundary(self):
        cases = {
            0: "0-1",
            1: "0-1",
            2: "2-10",
            10: "2-10",
            11: "11-50",
            50: "11-50",
            51: "51-200",
            200: "51-200",
            201: "201-500",
            500: "201-500",
            501: "501-1,000",
            1_000: "501-1,000",
            1_001: "1,001-5,000",
            5_000: "1,001-5,000",
            5_001: "5,001-10,000",
            10_000: "5,001-10,000",
            10_001: "10,001+",
        }
        for count, bucket in cases.items():
            with self.subTest(count=count, representation="integer"):
                self.assertEqual(normalize_observed_employee_count_bucket(count), bucket)
            with self.subTest(count=count, representation="string"):
                self.assertEqual(
                    normalize_observed_employee_count_bucket(str(count)),
                    bucket,
                )

    def test_nonexact_observed_counts_remain_unverified(self):
        for value in (
            None,
            "",
            True,
            -1,
            "-1",
            1.5,
            "1.5",
            "+1",
            "01",
            " 1 ",
            "1,000",
            "1-10",
            "about 50",
            "50 employees",
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_observed_employee_count_bucket(value), "")

    def test_company_fit_projects_numeric_observation_before_exact_comparison(self):
        icp = ICPPrompt(
            icp_id="numeric-observation",
            prompt="test",
            industry="Software",
            sub_industry="SaaS",
            employee_count="11-50",
            company_stage="",
            geography="United States",
            product_service="test",
        )
        self.assertEqual(
            _decision_from_observed_employee_size(
                {
                    "observed_employee_count": "11-50",
                    "employee_size_matches": True,
                },
                icp,
            ),
            COMPANY_FIT_MATCH,
        )
        self.assertEqual(
            _decision_from_observed_employee_size(
                {
                    "observed_employee_count": "50",
                    "employee_size_matches": True,
                },
                icp,
            ),
            COMPANY_FIT_MATCH,
        )
        self.assertEqual(
            _decision_from_observed_employee_size(
                {
                    "observed_employee_count": "51",
                    "employee_size_matches": True,
                },
                icp,
            ),
            COMPANY_FIT_UNAVAILABLE,
        )
        self.assertEqual(
            _decision_from_observed_employee_size(
                {
                    "observed_employee_count": "50.0",
                    "employee_size_matches": True,
                },
                icp,
            ),
            COMPANY_FIT_UNAVAILABLE,
        )
        for value in ("1-10", "about 50", "50 employees"):
            with self.subTest(value=value):
                self.assertEqual(
                    _decision_from_observed_employee_size(
                        {
                            "observed_employee_count": value,
                            "employee_size_matches": True,
                        },
                        icp,
                    ),
                    COMPANY_FIT_UNAVAILABLE,
                )

    def test_reverify_prompt_requires_canonical_bucket_or_strict_integer(self):
        observed = {}

        class FakeResponse:
            status = 503

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        class FakeSession:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def post(self, _url, **kwargs):
                observed.update(kwargs)
                return FakeResponse()

        company = CompanyOutput(
            company_name="Acme",
            company_website="https://acme.example",
            industry="Software",
            employee_count="11-50",
            country="United States",
            intent_signals=[
                {
                    "description": "Acme announced a new project.",
                    "source": "news",
                    "url": "https://news.example/acme",
                    "date": "2026-08-01",
                    "snippet": "Acme announced a new project.",
                }
            ],
        )
        icp = ICPPrompt(
            icp_id="prompt-enum",
            prompt="test",
            industry="Software",
            sub_industry="SaaS",
            employee_count="11-50",
            company_stage="",
            geography="United States",
            product_service="test",
        )
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "fixture"}), mock.patch(
            "qualification.scoring.lead_scorer.aiohttp.ClientSession",
            FakeSession,
        ):
            result = asyncio.run(
                _llm_reverify_company(
                    company,
                    icp,
                    require_company_fit_dimensions=True,
                )
            )
        self.assertEqual(result.decision, COMPANY_FIT_UNAVAILABLE)
        prompt = next(
            message["content"]
            for message in observed["json"]["messages"]
            if message["role"] == "user"
        )
        for bucket in LINKEDIN_EMPLOYEE_BUCKETS:
            with self.subTest(bucket=bucket):
                self.assertIn(bucket, prompt)
        self.assertIn("return that as a JSON integer", prompt)
        self.assertIn(
            "Never return an approximate, qualified, decimal, or custom range",
            prompt,
        )

    def test_employee_count_projector_fault_is_not_silenced_as_unavailable(self):
        icp = ICPPrompt(
            icp_id="projector-fault",
            prompt="test",
            industry="Software",
            sub_industry="SaaS",
            employee_count="11-50",
            company_stage="",
            geography="United States",
            product_service="test",
        )
        with mock.patch(
            "qualification.employee_buckets.normalize_observed_employee_count_bucket",
            side_effect=RuntimeError("test-only projector fault"),
        ):
            with self.assertRaisesRegex(RuntimeError, "projector fault"):
                _decision_from_observed_employee_size(
                    {
                        "observed_employee_count": 50,
                        "employee_size_matches": True,
                    },
                    icp,
                )


if __name__ == "__main__":
    unittest.main()
