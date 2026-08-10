from __future__ import annotations

import asyncio
import concurrent.futures
import inspect

import pytest

from gateway.research_lab import scoring_worker as sw
from gateway.research_lab.model_authority_v2 import AttestedPrivateModelRunnerV2
from research_lab.eval import private_runtime


def _receipt_hash(index: int) -> str:
    return f"sha256:{index:064x}"


@pytest.mark.asyncio
async def test_attested_receipt_collectors_are_task_local() -> None:
    async def collect(index: int) -> set[str]:
        receipt_hashes, token = (
            private_runtime.begin_attested_receipt_hash_collection()
        )
        try:
            private_runtime.publish_attested_receipt_hash(_receipt_hash(index))
            await asyncio.sleep(0)
            private_runtime.publish_attested_receipt_hash(_receipt_hash(index + 100))
            await asyncio.sleep(0)
            return set(receipt_hashes)
        finally:
            private_runtime.end_attested_receipt_hash_collection(token)

    observed = await asyncio.gather(*(collect(index) for index in range(20)))

    assert observed == [
        {_receipt_hash(index), _receipt_hash(index + 100)}
        for index in range(20)
    ]


def test_concurrency_one_uses_async_aware_batch_scheduler() -> None:
    config = sw.ResearchLabGatewayConfig(private_baseline_concurrency=1)
    assert inspect.iscoroutinefunction(AttestedPrivateModelRunnerV2.__call__)
    assert sw._private_baseline_uses_batch_execution(config)
    assert sw._private_baseline_uses_batch_execution(
        sw.ResearchLabGatewayConfig(private_baseline_concurrency=2)
    )


def test_v2_baseline_receipt_capacity_is_bounded_before_scoring() -> None:
    sw._require_v2_baseline_receipt_capacity(40)
    sw._require_v2_baseline_receipt_capacity(64)

    with pytest.raises(RuntimeError, match="receipt authority capacity"):
        sw._require_v2_baseline_receipt_capacity(65)


@pytest.mark.asyncio
async def test_40_icp_attempt_receipts_remain_exact_after_concurrent_scoring() -> None:
    worker = sw.ResearchLabGatewayScoringWorker(
        sw.ResearchLabGatewayConfig(private_baseline_concurrency=40)
    )

    class Runner:
        async def __call__(self, icp, _context):  # noqa: ANN001
            index = int(icp["index"])
            private_runtime.publish_attested_receipt_hash(_receipt_hash(index))
            await asyncio.sleep(0)
            return [
                {
                    "company_name": f"company-{index}",
                    "employee_count": "11-50",
                }
            ]

    class Scorer:
        async def score_with_breakdowns(self, _outputs, icp, _is_reference):  # noqa: ANN001
            index = int(icp["index"])
            await asyncio.sleep(0)
            private_runtime.publish_attested_receipt_hash(
                _receipt_hash(index + 1000)
            )
            return [{"final_score": float(index + 1)}]

    items = [
        {
            "icp": {"index": index, "employee_count": ["11-50"]},
            "icp_ref": f"icp:{index}",
            "icp_hash": f"icp-hash-{index}",
            "set_id": 1,
            "day_index": index,
            "day_rank": 1,
        }
        for index in range(40)
    ]
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        rows = await asyncio.gather(
            *(
                worker._run_baseline_icp(
                    runner=Runner(),
                    scorer=Scorer(),
                    item=item,
                    item_index=index + 1,
                    total_icps=40,
                    run_start=0.0,
                    executor=executor,
                )
                for index, item in enumerate(items)
            )
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    frontier: set[str] = set()
    for index, row in enumerate(rows):
        expected = [_receipt_hash(index), _receipt_hash(index + 1000)]
        assert row[sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD] == expected
        assert sw._record_baseline_attempt_parent_receipts(frontier, row) == tuple(
            expected
        )

    assert len(frontier) == 80
    assert len(
        sw._attested_receipts_with_persisted_roots(
            persisted_receipt_hashes=sorted(frontier)
        )
    ) == 80


@pytest.mark.asyncio
async def test_40_icp_two_round_recovery_excludes_superseded_receipts(
    monkeypatch,
) -> None:
    worker = sw.ResearchLabGatewayScoringWorker(
        sw.ResearchLabGatewayConfig(
            private_baseline_concurrency=8,
            private_baseline_retry_concurrency=8,
            private_baseline_provider_retry_rounds=2,
        )
    )

    class Runner:
        def __init__(self, round_no: int) -> None:
            self.round_no = round_no

        async def __call__(self, icp, _context):  # noqa: ANN001
            index = int(icp["index"])
            private_runtime.publish_attested_receipt_hash(
                _receipt_hash(10_000 + self.round_no * 100 + index)
            )
            return [
                {
                    "company_name": f"company-{index}",
                    "employee_count": "11-50",
                }
            ]

    class Scorer:
        def __init__(self) -> None:
            self.calls: dict[int, int] = {}

        async def score_with_breakdowns(self, _outputs, icp, _is_reference):  # noqa: ANN001
            index = int(icp["index"])
            call_no = self.calls.get(index, 0) + 1
            self.calls[index] = call_no
            if call_no <= 4:
                raise RuntimeError("HTTP Error 503: provider unavailable")
            private_runtime.publish_attested_receipt_hash(
                _receipt_hash(20_000 + index)
            )
            return [{"final_score": float(index + 1)}]

    async def maintenance_state():
        return {"paused": False}

    async def no_sleep(_seconds):  # noqa: ANN001
        return None

    monkeypatch.setattr(sw, "get_scoring_maintenance_state", maintenance_state)
    monkeypatch.setattr(sw.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        sw,
        "_retry_runner_with_provider_cost_scope",
        lambda _runner, *, retry_round, **_kwargs: Runner(retry_round),
    )

    window = type(
        "Window",
        (),
        {
            "benchmark_items": [
                {
                    "icp": {"index": index, "employee_count": ["11-50"]},
                    "icp_ref": f"icp:{index}",
                    "icp_hash": f"icp-hash-{index}",
                    "set_id": 1,
                    "day_index": index,
                    "day_rank": 1,
                }
                for index in range(40)
            ]
        },
    )()
    frontier: set[str] = set()

    async def checkpoint(row):  # noqa: ANN001
        sw._record_baseline_attempt_parent_receipts(frontier, row)
        return True

    rows, stats = await worker._run_baseline_batch_inner(
        runner=Runner(0),
        retry_runner=Runner(0),
        scorer=Scorer(),
        window=window,
        run_start=0.0,
        icp_checkpoint=checkpoint,
    )

    assert stats == {"retried": 80, "recovered": 40, "unresolved": 0}
    assert len(frontier) == 80
    for index, row in enumerate(rows):
        expected = {
            _receipt_hash(10_000 + 2 * 100 + index),
            _receipt_hash(20_000 + index),
        }
        assert set(row[sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD]) == expected
        assert expected.issubset(frontier)
        assert _receipt_hash(10_000 + index) not in frontier
        assert _receipt_hash(10_000 + 100 + index) not in frontier


@pytest.mark.parametrize(
    "raw_hashes",
    (
        None,
        [],
        [_receipt_hash(1)],
        [_receipt_hash(1), _receipt_hash(1)],
        [_receipt_hash(1), "invalid"],
        [_receipt_hash(1), _receipt_hash(2), _receipt_hash(3)],
    ),
)
def test_completed_v2_icp_requires_exact_model_and_scorer_roots(raw_hashes) -> None:
    row = {"icp_ref": "icp:1"}
    if raw_hashes is not None:
        row[sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD] = raw_hashes

    with pytest.raises(RuntimeError, match="causal receipt roots"):
        sw._record_baseline_attempt_parent_receipts(set(), row)


def test_unresolved_final_attempt_keeps_only_its_available_receipt_roots() -> None:
    final_model_receipt = _receipt_hash(500)
    frontier: set[str] = set()

    observed = sw._record_baseline_attempt_parent_receipts(
        frontier,
        {
            "icp_ref": "icp:unresolved",
            sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD: [final_model_receipt],
        },
        require_complete=False,
    )
    empty = sw._record_baseline_attempt_parent_receipts(
        frontier,
        {
            "icp_ref": "icp:failed-before-receipt",
            sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD: [],
        },
        require_complete=False,
    )

    assert observed == (final_model_receipt,)
    assert empty == ()
    assert frontier == {final_model_receipt}
