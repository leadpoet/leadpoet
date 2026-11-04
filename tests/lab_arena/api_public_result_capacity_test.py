"""A stalled object store must not let public result reads starve the paid lane.

On 2026-11-04 an unreachable S3 leg made GET
/arena/v1/rounds/{round_id}/results/{submission_id} run for up to 22 minutes a
call. Those reads were unlimited inside the non-provider reserve, so the shared
pool sat pinned at its 64-worker ceiling for 45 minutes and run claims,
completions and training capture queued behind anonymous public reads.
"""
from __future__ import annotations

import asyncio
import threading

import anyio
import httpx
import pytest

from lab_arena.api import PUBLIC_OBJECT_READ_WORKER_LIMIT, create_app
from lab_arena.service import ServiceError
from tests.lab_arena.api_provider_capacity_test import (
    BlockingService, POOL, wait_until,
)


def get_results(client, submission_id, *, code=False):
    return asyncio.create_task(
        client.get(
            "/arena/v1/submissions/" + submission_id + "/code"
            if code else "/arena/v1/rounds/round-1/results/" + submission_id
        )
    )


class StalledObjectStoreService(BlockingService):
    """public_results blocks the way a hung S3 GET blocks it: holding a worker."""

    def __init__(self, default, fail_read=False):
        super().__init__(default)
        self.fail_read = fail_read
        self.results_release = threading.Event()
        self.results_active = 0
        self.results_maximum = 0

    def public_results(self, round_id, submission_id):
        with self.lock:
            self.results_active += 1
            self.results_maximum = max(self.results_maximum, self.results_active)
        try:
            if not self.results_release.wait(5):
                raise AssertionError("test did not release the stalled read")
            if self.fail_read and submission_id == "sub-0":
                raise ServiceError("public_result_unavailable", 503)
            return {"round_id": round_id, "submission_id": submission_id}
        finally:
            with self.lock:
                self.results_active -= 1

    def public_submission_code(self, submission_id):
        return self.public_results("round-1", submission_id)


@pytest.mark.parametrize("route_mode", ("results", "code", "mixed"))
@pytest.mark.parametrize("fail_read", (False, True))
def test_stalled_public_results_cannot_starve_claims_and_completions(route_mode, fail_read):
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = StalledObjectStoreService(default, fail_read)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Exceed the entire shared pool: 24 reads cannot reproduce
                # starvation on the old code because the pool has 64 workers.
                count = POOL + 16
                reads = [get_results(
                    client, "sub-" + str(i),
                    code=route_mode == "code" or (route_mode == "mixed" and i % 2 == 1),
                ) for i in range(count)]
                try:
                    await wait_until(
                        lambda: service.results_active >= PUBLIC_OBJECT_READ_WORKER_LIMIT
                    )
                    # The paid lane must still answer while the reads are stuck.
                    paid = await asyncio.wait_for(asyncio.gather(
                        client.post("/arena/v1/runs/claim", json={}),
                        client.post(
                            "/arena/v1/runs/done/complete",
                            json={"body": {"run_id": "done"}},
                        ),
                    ), 1)
                    assert all(response.status_code == 200 for response in paid)
                    # Only the bulkhead's share is consumed; the rest queue
                    # without holding a worker.
                    assert service.results_active == PUBLIC_OBJECT_READ_WORKER_LIMIT
                    assert default.borrowed_tokens <= PUBLIC_OBJECT_READ_WORKER_LIMIT
                    assert default.total_tokens == POOL == 64
                finally:
                    service.results_release.set()
                    responses = await asyncio.wait_for(asyncio.gather(*reads), 5)
                # A service failure releases its permit and preserves the
                # existing error response; all queued reads still finish.
                expected = [200] * count
                if fail_read:
                    expected[0] = 503
                    assert responses[0].json() == {
                        "status": "rejected", "code": "public_result_unavailable",
                    }
                assert [r.status_code for r in responses] == expected
                assert all(r.headers["cache-control"] == "no-store" for r in responses)
                assert all(r.headers["x-content-type-options"] == "nosniff" for r in responses)
                assert service.results_maximum == PUBLIC_OBJECT_READ_WORKER_LIMIT

    anyio.run(scenario)
