"""Safe diagnostics for runner round discovery and assignment pickup."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from lab_arena import runner as rn


ROUND_ID = "arena-2099-01-01"
RUNNER_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
SECRET_CANARY = "sk-secret-canary?signature=private"


def _config(tmp_path, api, *, pinned: bool, parallel: int = 2) -> rn.RunnerConfig:
    return rn.RunnerConfig(
        round_id=ROUND_ID if pinned else None,
        identity=rn.RunnerIdentity(hotkey=RUNNER_HOTKEY, sign=lambda _message: "sig"),
        api=api,
        sandbox_runtime=object(),
        image_cache=object(),
        source_cache=object(),
        work_dir=tmp_path,
        max_parallel_runs=parallel,
        clock=lambda: datetime(2099, 1, 1, tzinfo=timezone.utc),
        claim_retry_seconds=(2.0, 5.0),
    )


def test_round_discovery_failure_is_distinct_from_idle_and_redacted(tmp_path, capsys):
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(503, text=SECRET_CANARY)
        )
    )
    api = rn.HttpArenaApiClient("http://localhost", client=client)
    runner = rn.Runner(_config(tmp_path, api, pinned=False))

    try:
        assert runner.run_once() == 0
    finally:
        runner.close()
        api.close()

    assert capsys.readouterr().err == (
        "Lab Arena pickup failure: phase=round_discovery reason=request_failed "
        "http_status=503 denial_code=-\n"
    )


def test_claim_failure_keeps_retries_and_releases_slots(tmp_path, monkeypatch, capsys):
    class ClaimFailureApi:
        def __init__(self):
            self.claims = 0

        def claim(self, _envelope):
            self.claims += 1
            raise rn.RunnerError(
                "unsafe claim detail " + SECRET_CANARY,
                http_status=502,
            )

    api = ClaimFailureApi()
    sleeps = []
    monkeypatch.setattr(rn.time, "sleep", sleeps.append)
    runner = rn.Runner(_config(tmp_path, api, pinned=True))

    assert runner.run_once() == 0

    assert api.claims == 3
    assert sleeps == [2.0, 5.0]
    assert capsys.readouterr().err == (
        "Lab Arena pickup failure: phase=claim reason=request_failed "
        "http_status=502 denial_code=-\n"
    )
    assert runner._slots.acquire(blocking=False)
    assert runner._slots.acquire(blocking=False)
    runner._slots.release()
    runner._slots.release()


def test_runner_error_keeps_zero_argument_compatibility():
    assert str(rn.RunnerError()) == ""


def test_claim_diagnostic_write_failure_does_not_change_retry_or_slot_behavior(
    tmp_path, monkeypatch
):
    class ClaimFailureApi:
        def __init__(self):
            self.claims = 0

        def claim(self, _envelope):
            self.claims += 1
            raise rn.RunnerError("request failed")

    class BrokenLogSink:
        def write(self, _value):
            raise OSError("log unavailable")

        def flush(self):
            raise OSError("log unavailable")

    api = ClaimFailureApi()
    monkeypatch.setattr(rn.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(rn.sys, "stderr", BrokenLogSink())
    runner = rn.Runner(_config(tmp_path, api, pinned=True))

    assert runner.run_once() == 0
    assert api.claims == 3
    assert runner._slots.acquire(blocking=False)
    assert runner._slots.acquire(blocking=False)
    runner._slots.release()
    runner._slots.release()


def test_http_claim_denial_logs_only_known_code_and_status(tmp_path, capsys):
    def deny(_request):
        return httpx.Response(
            403,
            json={
                "status": "rejected",
                "code": "runner_validator_required",
                "detail": SECRET_CANARY,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(deny))
    api = rn.HttpArenaApiClient("http://localhost", client=client)
    runner = rn.Runner(_config(tmp_path, api, pinned=True))
    try:
        assert runner.run_once() == 0
    finally:
        runner.close()
        api.close()

    diagnostic = capsys.readouterr().err
    assert diagnostic == (
        "Lab Arena pickup failure: phase=claim reason=claim_denied "
        "http_status=403 denial_code=runner_validator_required\n"
    )
    assert SECRET_CANARY not in diagnostic


def test_deep_500_claim_body_cannot_escape_runner_error_handling(
    tmp_path, monkeypatch, capsys
):
    calls = 0

    def fail(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(500, content=b"[" * 1100 + b"]" * 1100)

    client = httpx.Client(transport=httpx.MockTransport(fail))
    api = rn.HttpArenaApiClient("http://localhost", client=client)
    monkeypatch.setattr(rn.time, "sleep", lambda _delay: None)
    runner = rn.Runner(_config(tmp_path, api, pinned=True))
    try:
        assert runner.run_once() == 0
    finally:
        runner.close()
        api.close()

    assert calls == 3
    assert capsys.readouterr().err == (
        "Lab Arena pickup failure: phase=claim reason=request_failed "
        "http_status=500 denial_code=-\n"
    )


@pytest.mark.parametrize("status", ("no_pending", "no_open_round", "stage_closed"))
def test_normal_idle_claim_responses_do_not_log(tmp_path, capsys, status):
    class IdleApi:
        def claim(self, _envelope):
            return {"status": status}

    runner = rn.Runner(_config(tmp_path, IdleApi(), pinned=True))

    assert runner.run_once() == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("status", ([], {}))
def test_unhashable_claim_status_keeps_the_nonleased_path(tmp_path, capsys, status):
    class InvalidStatusApi:
        def claim(self, _envelope):
            return {"status": status}

    runner = rn.Runner(_config(tmp_path, InvalidStatusApi(), pinned=True))

    assert runner.run_once() == 0
    assert capsys.readouterr().err == (
        "Lab Arena pickup failure: phase=claim reason=claim_denied "
        "http_status=- denial_code=-\n"
    )
