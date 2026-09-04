from __future__ import annotations

import json
import os
import subprocess
import unittest
from unittest.mock import patch

from gateway.research_lab.public_baseline_runner import (
    PublicBaselineDockerRunner,
    PublicBaselineRunError,
    RESULT_SENTINEL,
)


def _line(value: dict) -> str:
    return RESULT_SENTINEL + json.dumps(value)


def _company(url: str = "https://example.com") -> dict:
    return {
        "company_name": "Example",
        "company_website": url,
        "company_linkedin": "",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "state": "California",
        "fit_summary": "The company fits.",
        "fit_evidence_urls": ["https://example.com/about"],
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": "A current launch.",
                "date": "2026-09-01",
                "why_now": "The launch makes outreach timely.",
                "url": "https://example.com/news",
                "snippet": "Example launched a product.",
            }
        ],
    }


def _execution_result(company: dict) -> dict:
    return {
        "ok": True,
        "companies": [company],
        "usage": {"input_tokens": 10},
        "provider_calls": [{"tool": "search_companies"}],
        "provider_call_count": 1,
        "estimated_provider_cost_usd": 0.01,
        "model_cost_usd": 0.02,
        "estimated_combined_cost_usd": 0.03,
        "cost_limit_status": "within_limit",
        "token_limit_status": "within_limit",
        "latency_seconds": 1.2,
    }


class PublicBaselineRunnerTests(unittest.TestCase):
    def test_builds_once_and_runs_with_env_names_only(self) -> None:
        commands: list[tuple[list[str], str]] = []

        def execute(argv: list[str], stdin: str, _timeout: float):
            commands.append((list(argv), stdin))
            if argv[:2] == ["docker", "build"]:
                return 0, "", ""
            if argv[-1] == "preflight":
                return 0, _line(
                    {
                        "ok": True,
                        "selected_model": "openai/test",
                        "model_pricing": {"prompt": "0.1"},
                        "deepline": {"connected": True},
                    }
                ), ""
            return 0, _line(_execution_result(_company())), ""

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "secret-openrouter",
                "DEEPLINE_API_KEY": "secret-deepline",
                "SCRAPINGDOG_API_KEY": "secret-scrapingdog",
            },
            clear=False,
        ):
            runner = PublicBaselineDockerRunner(execute=execute)
            self.assertEqual(runner.preflight()["model"], "openai/test")
            result = runner.run_icp(
                {"icp_id": "icp_today"},
                evaluation_date="2026-09-03",
                max_companies=3,
            )

        self.assertEqual(len(result.companies), 1)
        self.assertEqual(sum(1 for argv, _ in commands if argv[:2] == ["docker", "build"]), 1)
        run_argv = commands[-1][0]
        self.assertIn("OPENROUTER_API_KEY", run_argv)
        self.assertNotIn("secret-openrouter", run_argv)
        self.assertIn("--max-companies", run_argv)
        self.assertEqual(json.loads(commands[-1][1]), {"icp_id": "icp_today"})

    def test_rejects_non_public_evidence_url(self) -> None:
        call = 0

        def execute(argv: list[str], _stdin: str, _timeout: float):
            nonlocal call
            call += 1
            if call == 1:
                return 0, "", ""
            if argv[-1] == "preflight":
                return 0, _line(
                    {
                        "ok": True,
                        "selected_model": "openai/test",
                        "model_pricing": {},
                        "deepline": {"connected": True},
                    }
                ), ""
            return 0, _line(
                _execution_result(_company("http://127.0.0.1/private"))
            ), ""

        runner = PublicBaselineDockerRunner(execute=execute)
        runner.preflight()
        with self.assertRaisesRegex(PublicBaselineRunError, "non-public"):
            runner.run_icp({}, evaluation_date="2026-09-03")

    def test_rejects_reserved_or_internal_hostnames(self) -> None:
        for url in (
            "https://internal/news",
            "https://source.localhost/news",
            "https://source.invalid/news",
            "https://source.test/news",
            "http://127.1/private",
        ):
            with self.subTest(url=url):
                call = 0

                def execute(argv: list[str], _stdin: str, _timeout: float):
                    nonlocal call
                    call += 1
                    if call == 1:
                        return 0, "", ""
                    if argv[-1] == "preflight":
                        return 0, _line(
                            {
                                "ok": True,
                                "selected_model": "openai/test",
                                "model_pricing": {},
                                "deepline": {"connected": True},
                            }
                        ), ""
                    return 0, _line(_execution_result(_company(url))), ""

                runner = PublicBaselineDockerRunner(execute=execute)
                runner.preflight()
                with self.assertRaisesRegex(PublicBaselineRunError, "non-public"):
                    runner.run_icp({}, evaluation_date="2026-09-03")

    def test_allows_empty_optional_stage_and_state(self) -> None:
        call = 0

        def execute(argv: list[str], _stdin: str, _timeout: float):
            nonlocal call
            call += 1
            if call == 1:
                return 0, "", ""
            if argv[-1] == "preflight":
                return 0, _line(
                    {
                        "ok": True,
                        "selected_model": "openai/test",
                        "model_pricing": {},
                        "deepline": {"connected": True},
                    }
                ), ""
            company = _company()
            company["company_stage"] = ""
            company["state"] = ""
            return 0, _line(_execution_result(company)), ""

        runner = PublicBaselineDockerRunner(execute=execute)
        runner.preflight()
        result = runner.run_icp({}, evaluation_date="2026-09-03")

        self.assertEqual(result.companies[0]["company_stage"], "")
        self.assertEqual(result.companies[0]["state"], "")

    def test_timeout_is_a_bounded_error(self) -> None:
        def execute(argv: list[str], _stdin: str, timeout: float):
            if argv[:2] == ["docker", "build"]:
                return 0, "", ""
            raise subprocess.TimeoutExpired(argv, timeout)

        runner = PublicBaselineDockerRunner(execute=execute)
        with self.assertRaisesRegex(PublicBaselineRunError, "timed out"):
            runner.preflight()

    def test_rejects_reported_limit_failure(self) -> None:
        call = 0

        def execute(argv: list[str], _stdin: str, _timeout: float):
            nonlocal call
            call += 1
            if call == 1:
                return 0, "", ""
            if argv[-1] == "preflight":
                return 0, _line(
                    {
                        "ok": True,
                        "selected_model": "openai/test",
                        "model_pricing": {},
                        "deepline": {"connected": True},
                    }
                ), ""
            result = _execution_result(_company())
            result["provider_call_count"] = 31
            result["provider_calls"] = [
                {"tool": "search_companies"} for _ in range(31)
            ]
            return 0, _line(result), ""

        runner = PublicBaselineDockerRunner(execute=execute)
        runner.preflight()
        with self.assertRaisesRegex(PublicBaselineRunError, "provider-call"):
            runner.run_icp({}, evaluation_date="2026-09-03")

    def test_rejects_more_companies_than_the_requested_limit(self) -> None:
        call = 0

        def execute(argv: list[str], _stdin: str, _timeout: float):
            nonlocal call
            call += 1
            if call == 1:
                return 0, "", ""
            if argv[-1] == "preflight":
                return 0, _line(
                    {
                        "ok": True,
                        "selected_model": "openai/test",
                        "model_pricing": {},
                        "deepline": {"connected": True},
                    }
                ), ""
            result = _execution_result(_company())
            result["companies"].append(_company("https://second.example"))
            return 0, _line(result), ""

        runner = PublicBaselineDockerRunner(execute=execute)
        runner.preflight()
        with self.assertRaisesRegex(PublicBaselineRunError, "company list"):
            runner.run_icp(
                {},
                evaluation_date="2026-09-03",
                max_companies=1,
            )

    def test_rejects_incomplete_intent_evidence(self) -> None:
        call = 0

        def execute(argv: list[str], _stdin: str, _timeout: float):
            nonlocal call
            call += 1
            if call == 1:
                return 0, "", ""
            if argv[-1] == "preflight":
                return 0, _line(
                    {
                        "ok": True,
                        "selected_model": "openai/test",
                        "model_pricing": {},
                        "deepline": {"connected": True},
                    }
                ), ""
            company = _company()
            company["intent_signals"][0]["why_now"] = ""
            return 0, _line(_execution_result(company)), ""

        runner = PublicBaselineDockerRunner(execute=execute)
        runner.preflight()
        with self.assertRaisesRegex(PublicBaselineRunError, "invalid field"):
            runner.run_icp({}, evaluation_date="2026-09-03")


if __name__ == "__main__":
    unittest.main()
