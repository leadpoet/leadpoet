"""Active work must not block bounded pickup after temporary claim failures."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import threading
import time

import httpx
import pytest

from lab_arena import contracts, runner as rn


@pytest.mark.parametrize(
    "failure,refills",
    [
        ("no_pending", True),
        ("no_free_slot", True),
        ("transport", True),
        ("availability", True),
        ("server", True),
        ("stage_closed", False),
        ("no_open_round", False),
        ("authorization", False),
        ("contract_server", False),
        ("invalid_json", False),
    ],
)
def test_active_refill_preserves_claim_retries_and_denials(
    tmp_path, monkeypatch, failure, refills
):
    _check_active_refill(tmp_path, monkeypatch, failure, refills=refills)


def test_stop_during_active_claim_wait_prevents_refill(tmp_path, monkeypatch):
    _check_active_refill(
        tmp_path, monkeypatch, "no_free_slot", refills=False, stop_at_wait=True
    )


def _check_active_refill(
    tmp_path, monkeypatch, failure, *, refills, stop_at_wait=False
):
    release_first = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()
    failure_seen = threading.Event()
    stop = threading.Event()
    calls = []
    waits = []
    retry_sleeps = []
    failures = 3 if failure in {
        "transport", "availability", "server", "contract_server", "invalid_json"
    } else 1
    poll_seconds = 0.08

    def transport(request):
        if request.url.path.endswith("/complete"):
            return httpx.Response(200, json={"status": "accepted"})
        assert request.url.path.endswith("/claim")
        calls.append((time.monotonic(), json.loads(request.content)))
        position = len(calls)
        if position == 1 or position > 1 + failures:
            return httpx.Response(200, json={
                "status": "leased", "run_id": "first" if position == 1 else "second",
                "lease_token": "test-lease", "icp": {},
            })
        if position == 1 + failures:
            failure_seen.set()
        if failure == "transport":
            raise httpx.ConnectError("test transport outage", request=request)
        if failure == "invalid_json":
            return httpx.Response(200, text="invalid JSON")
        if failure in {"availability", "server", "contract_server"}:
            code = {
                "availability": "arena_store_unavailable",
                "server": None,
                "contract_server": "signature_invalid",
            }[failure]
            return httpx.Response(503, json={"code": code})
        if failure == "authorization":
            return httpx.Response(403, json={
                "status": "rejected", "code": "runner_validator_required",
            })
        return httpx.Response(200, json={"status": failure})

    client = httpx.Client(transport=httpx.MockTransport(transport))
    api = rn.HttpArenaApiClient("http://localhost", client=client)
    config = rn.RunnerConfig(
        round_id="arena-2099-01-01",
        identity=rn.RunnerIdentity(
            hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            sign=lambda _message: "sig",
        ),
        api=api, sandbox_runtime=object(), image_cache=object(), source_cache=object(),
        work_dir=tmp_path, socket_root=tmp_path / "sockets", max_parallel_runs=2,
        clock=lambda: datetime(2099, 1, 1, tzinfo=timezone.utc),
        claim_retry_seconds=(0.0, 0.0), claim_poll_seconds=poll_seconds,
    )
    worker = rn.Runner(config)

    def execute(lease, _token, _icp):
        if lease["run_id"] == "first":
            first_started.set()
            assert release_first.wait(3)
        else:
            second_started.set()
        return contracts.build_signed_request(
            scope=contracts.SCOPE_COMPLETE, round_id=config.round_id,
            hotkey=config.identity.hotkey, body={"run_id": lease["run_id"]},
            timestamp=int(config.clock().timestamp()), sign_message=config.identity.sign,
        )

    worker._executor.execute = execute
    original_wait = rn.wait

    def observe_wait(*args, **kwargs):
        waits.append(kwargs.get("timeout"))
        if stop_at_wait:
            stop.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(rn, "wait", observe_wait)
    monkeypatch.setattr(rn.time, "sleep", retry_sleeps.append)
    caller = ThreadPoolExecutor(max_workers=1)
    try:
        result = caller.submit(worker.run_once, max_claims=2, stop_event=stop)
        assert first_started.wait(1)
        assert failure_seen.wait(1)
        if refills:
            assert second_started.wait(1)
            assert not release_first.is_set()
            assert calls[-1][0] - calls[failures][0] >= poll_seconds * 0.8
            assert waits == [poll_seconds]
            assert len(calls) == failures + 2
        else:
            assert not second_started.wait(poll_seconds * 2)
            assert len(calls) == failures + 1
        assert not result.done()
        release_first.set()
        assert result.result(1) == (2 if refills else 1)
        assert len(calls) == failures + (2 if refills else 1)
        assert retry_sleeps == ([0.0, 0.0] if failures == 3 else [])
        # The existing same-envelope retries recover an uncertain claim. Only
        # the later bounded scan creates a fresh request for new assignment work.
        if failures == 3:
            assert calls[1][1] == calls[2][1] == calls[3][1]
        if refills:
            assert calls[-1][1]["request_id"] != calls[1][1]["request_id"]
        assert all(item[1]["body"]["declared_parallelism"] == 2 for item in calls)
        assert worker.abandoned == 0
    finally:
        release_first.set()
        caller.shutdown(wait=True)
        worker.close()
        api.close()
