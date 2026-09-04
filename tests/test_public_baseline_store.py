from __future__ import annotations

import pytest

from gateway.research_lab import public_baseline_store as store


@pytest.mark.asyncio
async def test_save_progress_counts_only_completed_icps(monkeypatch):
    captured = {}

    async def update_row(_table, values, *, filters):
        captured["values"] = dict(values)
        captured["filters"] = tuple(filters)
        return dict(values, run_id=filters[0][1])

    monkeypatch.setattr(store, "update_row", update_row)
    result = await store.save_progress(
        run_id="run-1",
        expected_icp_count=2,
        per_icp_results=[
            {"icp_ref": "icp-1", "status": "completed"},
            {"icp_ref": "icp-2", "status": "failed"},
        ],
        expected_icp_refs=("icp-1", "icp-2"),
        usage_doc={},
        worker_ref="worker-0",
        claim_token="claim-1",
    )

    assert result["completed_icp_count"] == 1
    assert ("claim_token", "claim-1") in captured["filters"]


@pytest.mark.asyncio
async def test_stale_progress_cannot_reopen_completed_run(monkeypatch):
    completed = {"run_id": "run-1", "status": "completed"}

    async def update_row(*_args, **_kwargs):
        raise RuntimeError("no rows")

    async def select_one(*_args, **_kwargs):
        return completed

    monkeypatch.setattr(store, "update_row", update_row)
    monkeypatch.setattr(store, "select_one", select_one)
    result = await store.save_progress(
        run_id="run-1",
        expected_icp_count=1,
        per_icp_results=[{"icp_ref": "icp-1", "status": "completed"}],
        expected_icp_refs=("icp-1",),
        usage_doc={},
        worker_ref="stale-worker",
        claim_token="stale-claim",
    )

    assert result is completed
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_stale_failure_cannot_downgrade_completed_run(monkeypatch):
    completed = {"run_id": "run-1", "status": "completed"}

    async def update_row(*_args, **_kwargs):
        raise RuntimeError("no rows")

    async def select_one(*_args, **_kwargs):
        return completed

    monkeypatch.setattr(store, "update_row", update_row)
    monkeypatch.setattr(store, "select_one", select_one)
    result = await store.fail_run(
        run_id="run-1",
        error_code="stale_failure",
        error_message="late worker",
        worker_ref="stale-worker",
        claim_token="stale-claim",
    )

    assert result is completed
    assert result["status"] == "completed"


def test_progress_rejects_duplicate_icp_refs():
    with pytest.raises(ValueError, match="unique refs"):
        store.validate_progress_results(
            [
                {"icp_ref": "icp-1", "status": "completed"},
                {"icp_ref": "icp-1", "status": "failed"},
            ],
            expected_icp_count=2,
            expected_icp_refs=("icp-1", "icp-2"),
        )


@pytest.mark.asyncio
async def test_running_daily_run_cannot_be_claimed_twice(monkeypatch):
    async def select_one(*_args, **_kwargs):
        return {"run_id": "run-1", "status": "running"}

    async def call_rpc(_name, _params):
        return {"claim_status": "busy"}

    monkeypatch.setattr(store, "select_one", select_one)
    monkeypatch.setattr(store, "call_rpc", call_rpc)

    with pytest.raises(store.DailyRebenchmarkBusy):
        await store.get_or_create_run(
            benchmark_date="2026-09-04",
            baseline_id="leadpoet/pydantic-harness",
            baseline_repository="https://github.com/leadpoet/pydantic-harness.git",
            baseline_entrypoint="harness.run_icp",
            window_doc={},
            benchmark_input_doc={},
            evaluation_epoch=1,
            expected_icp_count=20,
            worker_ref="worker-2",
            claim_token="claim-2",
        )


@pytest.mark.asyncio
async def test_expired_second_attempt_claim_returns_terminal_run(monkeypatch):
    terminal = {
        "run_id": "run-1",
        "status": "failed",
        "attempt_count": 2,
        "error_doc": {"code": "daily_rebenchmark_lease_exhausted"},
    }

    async def call_rpc(_name, _params):
        return {"claim_status": "exhausted", "run": terminal}

    monkeypatch.setattr(store, "call_rpc", call_rpc)
    result = await store.claim_run(
        {"run_id": "run-1", "status": "running", "attempt_count": 2},
        worker_ref="worker-2",
        claim_token="claim-2",
    )

    assert result == terminal


@pytest.mark.asyncio
async def test_failed_run_retries_only_from_the_expected_first_attempt(monkeypatch):
    captured = {}

    async def call_rpc(name, params):
        captured["name"] = name
        captured["params"] = dict(params)
        return {
            "retry_status": "retried",
            "run": {"run_id": "run-1", "status": "running", "attempt_count": 2},
        }

    monkeypatch.setattr(store, "call_rpc", call_rpc)
    result = await store.retry_failed_run(
        {"run_id": "run-1", "status": "failed", "attempt_count": 1},
        worker_ref="worker-2",
        claim_token="claim-2",
    )

    assert result["attempt_count"] == 2
    assert captured["name"] == "research_lab_retry_daily_rebenchmark"
    assert captured["params"]["p_expected_attempt"] == 1
    assert captured["params"]["p_claim_token"] == "claim-2"


@pytest.mark.asyncio
async def test_second_failed_attempt_does_not_call_retry_rpc(monkeypatch):
    async def forbidden_rpc(*_args, **_kwargs):
        raise AssertionError("the second failed attempt must be terminal")

    monkeypatch.setattr(store, "call_rpc", forbidden_rpc)
    result = await store.retry_failed_run(
        {"run_id": "run-1", "status": "failed", "attempt_count": 2},
        worker_ref="worker-2",
        claim_token="claim-2",
    )

    assert result is None
