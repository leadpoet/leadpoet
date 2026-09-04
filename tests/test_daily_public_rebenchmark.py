from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from gateway.research_lab.daily_rebenchmark import run_daily_public_rebenchmark
from gateway.research_lab.icp_window import ResearchLabRollingIcpWindow
from gateway.research_lab.public_baseline_runner import PublicBaselineExecution


def _item(index: int) -> dict:
    return {
        "icp": {
            "icp_id": f"icp_20260903_{index:03d}",
            "industry": "Software",
            "sub_industry": "Vertical SaaS",
            "country": "United States",
            "employee_count": ["51-200"],
            "company_stage": ["Series A"],
            "intent_signals": ["Launched a product in the last 30 days"],
            "max_companies": 5,
        },
        "icp_ref": f"qualification_private_icp_sets:20260903:icp_20260903_{index:03d}",
        "icp_hash": "sha256:" + str(index) * 64,
        "set_id": 20260903,
        "day_index": 1,
        "day_rank": index,
        "intent_signal_signature": f"signal-{index}",
        "cohort": "daily",
    }


def _window() -> ResearchLabRollingIcpWindow:
    items = (_item(1), _item(2))
    return ResearchLabRollingIcpWindow(
        window_hash="sha256:" + "a" * 64,
        benchmark_id="daily",
        split_ref="daily",
        public_doc={"rolling_window_hash": "sha256:" + "a" * 64},
        benchmark_items=items,
        item_refs=tuple(item["icp_ref"] for item in items),
        set_ids=(20260903,),
    )


class MemoryStore:
    def __init__(self, existing_results=None):
        self.row = {
            "run_id": "run-1",
            "status": "running",
            "expected_icp_count": 2,
            "completed_icp_count": len(existing_results or []),
            "per_icp_results": list(existing_results or []),
        }
        self.progress: list[list[dict]] = []
        self.failed = []

    async def get_or_create_run(self, **_kwargs):
        return dict(self.row)

    async def save_progress(self, **kwargs):
        self.progress.append(list(kwargs["per_icp_results"]))
        self.row.update(
            status="running",
            per_icp_results=list(kwargs["per_icp_results"]),
            completed_icp_count=len(kwargs["per_icp_results"]),
        )
        return dict(self.row)

    async def complete_run(self, **kwargs):
        self.row.update(
            status="completed",
            aggregate_score=kwargs["aggregate_score"],
            per_icp_results=list(kwargs["per_icp_results"]),
            completed_icp_count=len(kwargs["per_icp_results"]),
        )
        return dict(self.row)

    async def fail_run(self, **kwargs):
        self.failed.append(kwargs)
        self.row["status"] = "failed"
        return dict(self.row)


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.closed = False

    def preflight(self):
        return {"model": "openai/test", "provider_status": {"connected": True}}

    def run_icp(self, icp, *, evaluation_date, max_companies):
        self.calls.append((icp["icp_id"], evaluation_date, max_companies))
        return PublicBaselineExecution(
            companies=[
                {
                    "company_name": "Example",
                    "company_website": "https://example.com",
                    "company_linkedin": "",
                    "industry": "Software",
                    "employee_count": "51-200",
                    "company_stage": "Series A",
                    "country": "United States",
                    "state": "California",
                    "fit_summary": "Matches the ICP.",
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
            ],
            usage={"input_tokens": 10, "output_tokens": 2},
            provider_calls=[{"tool": "search_companies"}],
            provider_cost_usd=0.01,
            model_cost_usd=0.02,
            combined_cost_usd=0.03,
            latency_seconds=1.0,
        )

    def close(self):
        self.closed = True


class FakeScorer:
    async def score_with_breakdowns(self, companies, _icp, _is_reference):
        return [
            {
                "final_score": 80.0,
                "icp_fit": 40.0,
                "intent_signal_final": 40.0,
                "failure_reason": "",
                "intent_signals_detail": [],
            }
            for _ in companies
        ]


def _config():
    return SimpleNamespace(
        baseline_start_utc_offset_seconds=0,
        public_benchmark_public_total_icps=1,
        public_benchmark_public_weak_total=0,
    )


class DailyPublicRebenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_every_current_set_icp_and_checkpoints_each_result(self):
        store = MemoryStore()
        runner = FakeRunner()
        fetch_calls = []

        async def fetch_window(**kwargs):
            fetch_calls.append(kwargs)
            return _window()

        result = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=12,
            now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
            runner_factory=lambda: runner,
            scorer_factory=FakeScorer,
            fetch_window=fetch_window,
            store=store,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(fetch_calls[0]["set_id"], 20260903)
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual([len(rows) for rows in store.progress], [1, 2])
        self.assertEqual(store.row["completed_icp_count"], 2)
        self.assertTrue(runner.closed)

    async def test_resume_skips_a_completed_icp(self):
        previous = {
            "icp_ref": _item(1)["icp_ref"],
            "icp_hash": _item(1)["icp_hash"],
            "status": "completed",
            "companies": [],
            "score_breakdowns": [],
            "summary": {
                "icp_ref": _item(1)["icp_ref"],
                "icp_hash": _item(1)["icp_hash"],
                "score": 0.0,
                "company_count": 0,
                "industry": "Software",
                "sub_industry": "Vertical SaaS",
                "country": "United States",
                "company_size_bucket": "51-200",
                "intent_category_bucket": "other",
                "diagnostics": {"failure_categories": [], "sourcing_failed": False},
            },
            "usage": {
                "model": "openai/test",
                "provider_call_count": 0,
                "provider_cost_usd": 0.0,
                "combined_cost_usd": 0.0,
            },
        }
        store = MemoryStore([previous])
        runner = FakeRunner()

        async def fetch_window(**_kwargs):
            return _window()

        result = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=12,
            now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
            runner_factory=lambda: runner,
            scorer_factory=FakeScorer,
            fetch_window=fetch_window,
            store=store,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual([call[0] for call in runner.calls], ["icp_20260903_002"])


if __name__ == "__main__":
    unittest.main()
