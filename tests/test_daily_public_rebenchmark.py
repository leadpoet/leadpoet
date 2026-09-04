from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from gateway.research_lab.daily_icp_set import (
    DailyIcpSet,
    daily_icp_set_from_input_doc,
)
from gateway.research_lab.daily_rebenchmark import run_daily_public_rebenchmark
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
            "intent_category": "PRODUCT_LAUNCH",
            "intent_max_age_days": 30,
            "max_companies": 5,
        },
        "icp_ref": f"qualification_private_icp_sets:20260903:icp_20260903_{index:03d}",
        "set_id": 20260903,
        "position": index,
    }


def _window() -> DailyIcpSet:
    items = tuple(_item(index) for index in range(1, 21))
    refs = tuple(item["icp_ref"] for item in items)
    input_doc = {
        "schema_version": "research_lab_daily_icp_inputs.v1",
        "set_id": 20260903,
        "icps": [dict(item["icp"]) for item in items],
        "icp_refs": list(refs),
    }
    return daily_icp_set_from_input_doc(
        input_doc,
        required_set_id=20260903,
    )


class MemoryStore:
    def __init__(self, existing_results=None, *, existing=False):
        self.exists = bool(existing or existing_results is not None)
        self.row = {
            "run_id": "run-1",
            "status": "running",
            "attempt_count": 1,
            "evaluation_epoch": 7,
            "expected_icp_count": 20,
            "completed_icp_count": len(existing_results or []),
            "per_icp_results": list(existing_results or []),
        }
        self.progress: list[list[dict]] = []
        self.failed = []
        self.renewals = 0
        self.claims = 0
        self.resets = 0
        self.retries = 0
        self.recoveries = 0

    async def load_run(self, **_kwargs):
        return dict(self.row) if self.exists else None

    async def get_or_create_run(self, **kwargs):
        self.exists = True
        self.row.update(
            benchmark_date=kwargs["benchmark_date"],
            baseline_id=kwargs["baseline_id"],
            baseline_repository=kwargs["baseline_repository"],
            baseline_entrypoint=kwargs["baseline_entrypoint"],
            window_doc=dict(kwargs["window_doc"]),
            benchmark_input_doc=dict(kwargs["benchmark_input_doc"]),
            evaluation_epoch=kwargs["evaluation_epoch"],
            expected_icp_count=kwargs["expected_icp_count"],
        )
        return dict(self.row)

    async def claim_run(self, _row, **_kwargs):
        self.claims += 1
        return dict(self.row)

    async def retry_failed_run(self, _row, **_kwargs):
        if int(self.row.get("attempt_count") or 1) >= 2:
            return None
        self.retries += 1
        self.row.update(
            status="running",
            attempt_count=2,
            completed_icp_count=0,
            per_icp_results=[],
            error_doc={},
        )
        return dict(self.row)

    async def recover_invalid_completed_run(self, _row, **_kwargs):
        self.recoveries += 1
        if int(self.row.get("attempt_count") or 1) >= 2:
            self.row.update(
                status="failed",
                error_doc={"code": "invalid_completed_run"},
            )
        else:
            self.row.update(
                status="running",
                attempt_count=2,
                completed_icp_count=0,
                per_icp_results=[],
                score_summary_doc={},
                public_report_doc={},
                error_doc={},
            )
        return dict(self.row)

    async def reset_progress(self, **_kwargs):
        self.resets += 1
        self.row.update(completed_icp_count=0, per_icp_results=[])
        return dict(self.row)

    async def save_progress(self, **kwargs):
        self.progress.append(list(kwargs["per_icp_results"]))
        self.row.update(
            status="running",
            per_icp_results=list(kwargs["per_icp_results"]),
            completed_icp_count=len(kwargs["per_icp_results"]),
        )
        return dict(self.row)

    async def renew_claim(self, **_kwargs):
        self.renewals += 1
        return dict(self.row)

    async def complete_run(self, **kwargs):
        self.row.update(
            status="completed",
            aggregate_score=kwargs["aggregate_score"],
            per_icp_results=list(kwargs["per_icp_results"]),
            completed_icp_count=len(kwargs["per_icp_results"]),
            usage_doc=dict(kwargs["usage_doc"]),
            score_summary_doc=dict(kwargs["score_summary_doc"]),
            public_report_doc=dict(kwargs["public_report_doc"]),
        )
        return dict(self.row)

    async def fail_run(self, **kwargs):
        self.failed.append(kwargs)
        self.row.update(
            status="failed",
            error_doc={"code": kwargs["error_code"]},
        )
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


class InfrastructureFailureScorer:
    async def score_with_breakdowns(self, companies, _icp, _is_reference):
        return [
            {
                "final_score": 0.0,
                "failure_reason": "LLM scoring error: provider timeout",
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
    async def test_runner_construction_failure_is_persisted(self):
        store = MemoryStore()

        async def fetch_window(**_kwargs):
            return _window()

        def fail_runner():
            raise RuntimeError("runner failed")

        result = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=12,
            now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
            runner_factory=fail_runner,
            scorer_factory=FakeScorer,
            fetch_window=fetch_window,
            store=store,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(store.failed), 1)

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
        self.assertEqual(len(runner.calls), 20)
        self.assertEqual([len(rows) for rows in store.progress], list(range(1, 21)))
        self.assertEqual(store.row["completed_icp_count"], 20)
        self.assertEqual(store.renewals, 42)
        self.assertTrue(runner.closed)

    async def test_valid_completed_row_is_reused_without_live_source(self):
        store = MemoryStore()

        async def fetch_window(**_kwargs):
            return _window()

        first = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=12,
            now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
            runner_factory=FakeRunner,
            scorer_factory=FakeScorer,
            fetch_window=fetch_window,
            store=store,
        )
        self.assertEqual(first["status"], "completed")

        async def forbidden_fetch(**_kwargs):
            raise AssertionError("completed recovery must use frozen inputs")

        second = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=99,
            now=datetime(2026, 9, 3, 13, tzinfo=timezone.utc),
            runner_factory=lambda: (_ for _ in ()).throw(
                AssertionError("valid completed row must not start a runner")
            ),
            scorer_factory=FakeScorer,
            fetch_window=forbidden_fetch,
            store=store,
        )

        self.assertEqual(second["status"], "already_benchmarked")
        self.assertEqual(store.recoveries, 0)

    async def test_invalid_completed_row_uses_one_same_day_full_recovery(self):
        store = MemoryStore()

        async def fetch_window(**_kwargs):
            return _window()

        first = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=12,
            now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
            runner_factory=FakeRunner,
            scorer_factory=FakeScorer,
            fetch_window=fetch_window,
            store=store,
        )
        self.assertEqual(first["status"], "completed")
        store.row["score_summary_doc"]["per_icp_summaries"] = []
        recovery_runner = FakeRunner()

        async def forbidden_fetch(**_kwargs):
            raise AssertionError("same-day recovery must use frozen inputs")

        second = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=99,
            now=datetime(2026, 9, 3, 13, tzinfo=timezone.utc),
            runner_factory=lambda: recovery_runner,
            scorer_factory=FakeScorer,
            fetch_window=forbidden_fetch,
            store=store,
        )

        self.assertEqual(second["status"], "completed")
        self.assertEqual(store.recoveries, 1)
        self.assertEqual(store.row["attempt_count"], 2)
        self.assertEqual(len(recovery_runner.calls), 20)

    async def test_claimed_stored_input_validation_failure_is_persisted(self):
        store = MemoryStore(existing=True)
        store.row.update(
            baseline_repository="https://github.com/leadpoet/pydantic-harness.git",
            baseline_entrypoint="harness.run_icp",
            window_doc=_window().public_doc,
            benchmark_input_doc={"schema_version": "invalid"},
        )

        async def forbidden_fetch(**_kwargs):
            raise AssertionError("recovery must not read the live source")

        result = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=12,
            now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
            runner_factory=lambda: (_ for _ in ()).throw(
                AssertionError("invalid stored input must not start a runner")
            ),
            scorer_factory=FakeScorer,
            fetch_window=forbidden_fetch,
            store=store,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(store.claims, 1)
        self.assertEqual(len(store.failed), 1)
        self.assertEqual(store.row["status"], "failed")

    async def test_recovery_resets_partial_progress_and_reruns_every_icp(self):
        previous = {
            "icp_ref": _item(1)["icp_ref"],
            "status": "completed",
            "companies": [],
            "score_breakdowns": [],
            "summary": {
                "icp_ref": _item(1)["icp_ref"],
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
            raise AssertionError("recovery must use its stored frozen ICP input")

        store.row.update(
            baseline_repository="https://github.com/leadpoet/pydantic-harness.git",
            baseline_entrypoint="harness.run_icp",
            window_doc=_window().public_doc,
            benchmark_input_doc=_window().input_doc,
        )

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
        self.assertEqual(
            [call[0] for call in runner.calls],
            [f"icp_20260903_{index:03d}" for index in range(1, 21)],
        )
        self.assertEqual(store.claims, 1)
        self.assertEqual(store.resets, 1)

    async def test_fully_checkpointed_run_finalizes_without_provider_preflight(self):
        previous_results = []
        for index in range(1, 21):
            item = _item(index)
            previous_results.append(
                {
                    "icp_ref": item["icp_ref"],
                    "status": "completed",
                    "companies": [],
                    "score_breakdowns": [],
                    "summary": {
                        "icp_ref": item["icp_ref"],
                        "score": float(index),
                        "company_count": 0,
                    },
                    "usage": {
                        "model": "openai/test",
                        "provider_call_count": 0,
                        "provider_cost_usd": 0.0,
                        "combined_cost_usd": 0.0,
                    },
                }
            )
        store = MemoryStore(previous_results)

        store.row.update(
            baseline_repository="https://github.com/leadpoet/pydantic-harness.git",
            baseline_entrypoint="harness.run_icp",
            window_doc=_window().public_doc,
            benchmark_input_doc=_window().input_doc,
        )

        async def fetch_window(**_kwargs):
            raise AssertionError("recovery must use its stored frozen ICP input")

        def fail_runner():
            raise AssertionError("provider preflight must not run")

        def fail_scorer():
            raise AssertionError("scorer must not be created")

        result = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=12,
            now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
            runner_factory=fail_runner,
            scorer_factory=fail_scorer,
            fetch_window=fetch_window,
            store=store,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["aggregate_score"], 10.5)
        self.assertEqual(
            store.row["score_summary_doc"]["evaluation_epoch"],
            7,
        )

    async def test_retryable_scorer_infrastructure_failure_is_not_published(self):
        store = MemoryStore()
        runner = FakeRunner()

        async def fetch_window(**_kwargs):
            return _window()

        result = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=12,
            now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
            runner_factory=lambda: runner,
            scorer_factory=InfrastructureFailureScorer,
            fetch_window=fetch_window,
            store=store,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "baseline_scoring_failed")
        self.assertEqual(store.row["status"], "failed")
        self.assertNotIn("aggregate_score", store.row)
        self.assertEqual(store.progress[0][0]["status"], "failed")
        self.assertEqual(len(runner.calls), 1)

    async def test_failed_daily_run_gets_one_whole_run_retry(self):
        store = MemoryStore(existing=True)
        store.row.update(
            status="failed",
            error_doc={"code": "baseline_run_failed"},
            baseline_repository="https://github.com/leadpoet/pydantic-harness.git",
            baseline_entrypoint="harness.run_icp",
            window_doc=_window().public_doc,
            benchmark_input_doc=_window().input_doc,
        )

        async def fetch_window(**_kwargs):
            raise AssertionError("retry must use its stored frozen ICP input")

        runner = FakeRunner()

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
        self.assertEqual(store.retries, 1)
        self.assertEqual(len(runner.calls), 20)

    async def test_failed_second_attempt_is_not_retried(self):
        store = MemoryStore(existing=True)
        store.row.update(
            status="failed",
            attempt_count=2,
            error_doc={"code": "baseline_run_failed"},
            baseline_repository="https://github.com/leadpoet/pydantic-harness.git",
            baseline_entrypoint="harness.run_icp",
            window_doc=_window().public_doc,
            benchmark_input_doc=_window().input_doc,
        )

        async def forbidden_fetch(**_kwargs):
            raise AssertionError("terminal state must not read the ICP source")

        result = await run_daily_public_rebenchmark(
            config=_config(),
            worker_ref="worker-0",
            evaluation_epoch=12,
            now=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
            runner_factory=lambda: (_ for _ in ()).throw(
                AssertionError("terminal state must not start a runner")
            ),
            scorer_factory=FakeScorer,
            fetch_window=forbidden_fetch,
            store=store,
        )

        self.assertEqual(result["status"], "already_failed")
        self.assertEqual(result["error_code"], "baseline_run_failed")
        self.assertEqual(store.retries, 0)


if __name__ == "__main__":
    unittest.main()
