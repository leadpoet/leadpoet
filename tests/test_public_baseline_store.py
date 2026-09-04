from __future__ import annotations

import pytest

from gateway.research_lab import public_baseline_store as store


@pytest.mark.asyncio
async def test_save_progress_counts_only_completed_icps(monkeypatch):
    captured = {}

    async def update_row(_table, values, *, filters):
        captured.update(values)
        return dict(values, run_id=filters[0][1])

    monkeypatch.setattr(store, "update_row", update_row)
    result = await store.save_progress(
        run_id="run-1",
        expected_icp_count=2,
        per_icp_results=[
            {"icp_ref": "icp-1", "status": "completed"},
            {"icp_ref": "icp-2", "status": "failed"},
        ],
        usage_doc={},
        worker_ref="worker-0",
        claim_token="claim-1",
    )

    assert result["completed_icp_count"] == 1


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
        )
