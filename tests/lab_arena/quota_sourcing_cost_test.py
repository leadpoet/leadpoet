"""Opt-in cost readout across the existing API/socket; v1 remains unchanged."""

import copy
import tempfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import lab_arena_checkpoint as sdk, runner
from lab_arena.api import create_app
from tests.lab_arena.quota_snapshot_test import LEASE_TOKEN, snapshot


def costs():
    return {
        "successful_microusd": 200_000,
        "settled_microusd": 250_000,
        "reserved_or_uncertain_microusd": 100_000,
        "success_unresolved_microusd": 100_000,
        "inflight_calls": 1,
        "success_unresolved_calls": 1,
        "admission_cap_microusd": 4_000_000,
        "per_qualified_pair_cap_microusd": 800_000,
    }


def cost_snapshot():
    return {
        **snapshot(),
        "schema_version": sdk.QUOTA_COST_SNAPSHOT_SCHEMA_VERSION,
        "sourcing_cost": costs(),
    }


class CostApi:
    def __init__(self):
        self.calls = []

    def quota_usage(
        self, run_id, lease_token, *, include_sourcing_cost=False
    ):
        self.calls.append((run_id, lease_token, include_sourcing_cost))
        return cost_snapshot() if include_sourcing_cost else snapshot()

    handle_quota_snapshot = quota_usage


def test_opt_in_http_costs_and_unchanged_default():
    service = CostApi()
    with TestClient(create_app(service)) as client:
        for opt_in in (False, True):
            response = client.get(
                "/arena/v1/runs/run-1/quota",
                params={"include_sourcing_cost": str(opt_in).lower()},
                headers={"x-lab-arena-lease": LEASE_TOKEN},
            )
            assert response.status_code == 200
            assert response.json() == (
                cost_snapshot() if opt_in else snapshot()
            )
            assert response.headers["cache-control"] == "no-store"
        denied = client.get(
            "/arena/v1/runs/run-1/quota?include_sourcing_cost=true"
        )
        assert denied.status_code == 401
    assert service.calls == [
        ("run-1", LEASE_TOKEN, False),
        ("run-1", LEASE_TOKEN, True),
    ]


def test_cost_snapshot_uses_native_socket_and_v1_projection(monkeypatch):
    api = CostApi()
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    with tempfile.TemporaryDirectory(
        prefix="quota-cost-", dir="/tmp"
    ) as directory:
        socket = Path(directory) / "worker.sock"
        server = runner.WorkerSocketServer(socket, api, state)
        server.start()
        monkeypatch.setenv(sdk.WORKER_SOCKET_ENV, str(socket))
        try:
            assert sdk.quota_usage(include_sourcing_cost=True) == cost_snapshot()
            assert sdk.quota_usage() == snapshot()
            assert sdk.quota_usage(include_sourcing_cost=True) == cost_snapshot()
        finally:
            server.stop()
    assert api.calls == [("run-1", LEASE_TOKEN, True)]
    assert state.action_sequence == state.refusals == 0
    assert state.calls == []


def test_cache_upgrades_to_costs_and_drops_stale_costs_on_failure():
    api = CostApi()
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    now = [10.0]
    server = runner.WorkerSocketServer(
        Path("/tmp/not-opened-quota-cost.sock"),
        api,
        state,
        monotonic=lambda: now[0],
    )
    assert server._quota_snapshot() == snapshot()
    assert (
        server._quota_snapshot(include_sourcing_cost=True) == cost_snapshot()
    )
    assert server._quota_snapshot() == snapshot()
    assert [call[2] for call in api.calls] == [False, True]
    now[0] += runner.QUOTA_SNAPSHOT_CACHE_SECONDS + 1
    api.quota_usage = lambda *args, **kwargs: {"invalid": True}
    assert server._quota_snapshot(include_sourcing_cost=True) is None
    assert not state.trusted_quota_failure
    assert state.quota_snapshot == snapshot()
    assert state.action_sequence == state.refusals == 0
    assert state.calls == []
    # The preserved counter projection is also expired. A v1 caller performs
    # its normal authoritative refresh and retains the existing failure path.
    assert server._quota_snapshot() is None
    assert state.trusted_quota_failure
    assert state.quota_snapshot is None


def test_failed_cost_upgrade_preserves_fresh_v1_cache_and_health():
    api = CostApi()
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    now = [10.0]
    server = runner.WorkerSocketServer(
        Path("/tmp/not-opened-quota-cost-v1.sock"),
        api,
        state,
        monotonic=lambda: now[0],
    )
    assert server._quota_snapshot() == snapshot()
    api.quota_usage = lambda *args, **kwargs: {"invalid": True}
    assert server._quota_snapshot(include_sourcing_cost=True) is None
    assert not state.trusted_quota_failure
    assert state.quota_snapshot == snapshot()
    assert state.quota_snapshot_at == 10.0
    assert server._quota_snapshot() == snapshot()


def test_valid_v2_repairs_v1_failure_but_failed_v2_does_not():
    api = CostApi()
    calls = []

    def staged_read(*args, **kwargs):
        calls.append(bool(kwargs.get("include_sourcing_cost")))
        if len(calls) == 1:
            raise RuntimeError("counter read failed")
        if len(calls) == 2:
            return {"invalid": True}
        return cost_snapshot()

    api.quota_usage = staged_read
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    server = runner.WorkerSocketServer(
        Path("/tmp/not-opened-quota-cost-recovery.sock"), api, state
    )
    assert server._quota_snapshot() is None
    assert state.trusted_quota_failure
    assert server._quota_snapshot(include_sourcing_cost=True) is None
    assert state.trusted_quota_failure
    assert server._quota_snapshot(include_sourcing_cost=True) == cost_snapshot()
    assert not state.trusted_quota_failure
    assert server._quota_snapshot() == snapshot()
    assert calls == [False, True, True]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["sourcing_cost"].update(
            successful_microusd=-1
        ),
        lambda value: value["sourcing_cost"].update(
            successful_microusd=True
        ),
        lambda value: value["sourcing_cost"].update(
            successful_microusd=300_000
        ),
        lambda value: value["sourcing_cost"].update(inflight_calls=2),
        lambda value: value["sourcing_cost"].update(
            admission_cap_microusd=0
        ),
        lambda value: value["sourcing_cost"].update(
            private_account="unexpected"
        ),
        lambda value: value.update(
            schema_version=sdk.QUOTA_SNAPSHOT_SCHEMA_VERSION
        ),
    ],
)
def test_cost_snapshot_rejects_corrupt_values(mutation):
    value = copy.deepcopy(cost_snapshot())
    mutation(value)
    with pytest.raises(sdk.QuotaUnavailable):
        sdk.validate_quota_cost_snapshot(value)


def test_cost_api_client_negotiates_and_never_downgrades():
    def handle(request):
        assert request.url.params["include_sourcing_cost"] == "true"
        return httpx.Response(200, json=cost_snapshot())

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = runner.HttpArenaApiClient(
            "https://arena.example", client=http
        )
        assert client.quota_usage(
            "run-1", LEASE_TOKEN, include_sourcing_cost=True
        ) == cost_snapshot()
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=snapshot())
        )
    ) as http:
        client = runner.HttpArenaApiClient(
            "https://arena.example", client=http
        )
        with pytest.raises(runner.RunnerError):
            client.quota_usage(
                "run-1", LEASE_TOKEN, include_sourcing_cost=True
            )
