"""The gateway process must populate its own dynamic validator PCR0 cache."""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.parametrize("historical_warm, expected_count", [(False, 1), (True, 20)])
def test_runtime_startup_populates_cache_before_periodic_wait(
    monkeypatch, historical_warm, expected_count
):
    from gateway.utils import pcr0_builder

    builds = []

    async def build(count):
        builds.append(count)
        raise asyncio.CancelledError

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(pcr0_builder, "PCR0_STARTUP_HISTORICAL_WARM_ENABLED", historical_warm)
    monkeypatch.setattr(pcr0_builder, "PCR0_CACHE_SIZE", 20)
    monkeypatch.setattr(pcr0_builder, "check_prerequisites", lambda: _true())
    monkeypatch.setattr(pcr0_builder, "build_pcr0_for_recent_commits", build)
    monkeypatch.setattr(pcr0_builder.asyncio, "sleep", no_delay)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pcr0_builder.pcr0_builder_task())

    assert builds == [expected_count]


def test_start_returns_the_owned_task_for_gateway_shutdown(monkeypatch):
    from gateway.utils import pcr0_builder

    async def run():
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def builder():
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        monkeypatch.setattr(pcr0_builder, "pcr0_builder_task", builder)
        task = pcr0_builder.start_pcr0_builder()
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        assert stopped.is_set()

    asyncio.run(run())


async def _true():
    return True
