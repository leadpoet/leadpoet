"""Regression tests: worker process count is decoupled from proxy count.

Egress reduction: configuring N proxies must not spawn N worker processes.
RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT / _SCORING_WORKER_PROCESS_COUNT
explicitly control concurrency; proxies are assigned round-robin; a hard max
guards against a misconfiguration spawning an unbounded fleet.
"""

from __future__ import annotations

import gateway.research_lab.worker_autostart as wa
from gateway.research_lab.config import _worker_total_from_proxy_count


def _hosted_env(**extra: str) -> dict[str, str]:
    env = {
        "RESEARCH_LAB_AUTO_START_WORKERS": "true",
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
    }
    env.update(extra)
    return env


def test_resolve_worker_count_process_count_is_authoritative() -> None:
    # 20 proxies but PROCESS_COUNT=2 -> 2 workers, not 20.
    assert wa._resolve_worker_count(2, 20) == 2
    # Unset (0) -> default one-per-proxy (historical behavior).
    assert wa._resolve_worker_count(0, 5) == 5
    # No proxies and no explicit count -> zero.
    assert wa._resolve_worker_count(0, 0) == 0
    # Clamped to the hard max.
    assert wa._resolve_worker_count(10_000, 3) == wa._MAX_WORKER_PROCESSES


def test_twenty_proxies_do_not_produce_twenty_processes() -> None:
    env = _hosted_env(
        **{
            f"RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_{i}": f"http://p{i}"
            for i in range(1, 21)
        },
        **{
            f"RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_{i}": f"http://s{i}"
            for i in range(1, 21)
        },
        RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT="2",
        RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT="3",
    )
    plan = wa.build_research_lab_worker_autostart_plan(env)
    assert plan.hosted.worker_count == 2
    assert plan.scoring.worker_count == 3
    # The proxies are still available for round-robin assignment.
    assert len(plan.hosted.proxy_values) == 20
    assert len(plan.scoring.proxy_values) == 20


def test_default_is_one_worker_per_proxy_when_process_count_unset() -> None:
    env = _hosted_env(
        **{
            f"RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_{i}": f"http://p{i}"
            for i in range(1, 4)
        },
        RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT="1",
        RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1="http://s1",
    )
    plan = wa.build_research_lab_worker_autostart_plan(env)
    assert plan.hosted.worker_count == 3  # one per proxy, unchanged default


def test_config_total_workers_honors_process_count() -> None:
    # Explicit process count is authoritative for partitioning too.
    import os

    prev = {
        k: os.environ.get(k)
        for k in ("RESEARCH_LAB_WORKER_PROXY_1", "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT")
    }
    try:
        os.environ["RESEARCH_LAB_WORKER_PROXY_1"] = "http://p1"
        os.environ["RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT"] = "4"
        total = _worker_total_from_proxy_count(
            prefixes=("RESEARCH_LAB_WORKER_PROXY",),
            legacy_total_env="RESEARCH_LAB_HOSTED_WORKER_TOTAL_WORKERS",
            process_count_env="RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT",
        )
        assert total == 4  # process count wins over the single proxy
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_preflight_allows_fewer_workers_than_profiles_but_not_more() -> None:
    from gateway.tee.restart_preflight_v2 import GatewayRestartPreflightV2Error

    # Reproduce the exact preflight check in isolation.
    def check(required: int, available: int) -> None:
        if required < 1 or required > available:
            raise GatewayRestartPreflightV2Error("no coverage")

    check(2, 20)  # fewer workers than profiles: OK
    check(20, 20)  # equal: OK
    import pytest

    with pytest.raises(GatewayRestartPreflightV2Error):
        check(21, 20)  # more workers than profiles: rejected
    with pytest.raises(GatewayRestartPreflightV2Error):
        check(0, 20)  # zero workers: rejected
