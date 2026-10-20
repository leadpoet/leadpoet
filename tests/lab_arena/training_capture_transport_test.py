"""Training evidence has its own bounded transport budget and safe failures."""

from types import SimpleNamespace

import httpx
import pytest

from lab_arena import runner


TOKEN = "private-lease-value-do-not-log"
RUN = "capture:test"


def config(api):
    return SimpleNamespace(api=api)


def test_capture_accepts_slow_transport_with_separate_bounded_timeout():
    observed = []

    def respond(request):
        timeout = request.extensions["timeout"]
        observed.append((request.url.path, timeout))
        # Model a three-second input and a fifteen-second aggregate validation
        # using httpx's actual timeout extension. Both old budgets fail here.
        required_seconds = 15.0 if request.url.path.endswith("/training-observations") else 3.0
        if min(timeout.values()) < required_seconds:
            raise httpx.ReadTimeout("private response text", request=request)
        return httpx.Response(200, json={"status": "stored", "payload_sha256": "sha256:" + "a" * 64})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        api = runner.HttpArenaApiClient("https://arena.invalid", client=client)
        state = runner.RunState(lease={"run_id": RUN}, lease_token=TOKEN)
        runner._record_training_input(config(api), state.lease, TOKEN, {"input": "captured"})
        result = runner._record_training_observations(config(api), state.lease, TOKEN, state)
        assert result["status"] == "stored"
        api.claim({})
        api.provider(RUN, TOKEN, {})
    assert len(observed) == 4
    for path, timeout in observed[:2]:
        assert path.endswith(("/training-input", "/training-observations"))
        expected = 60.0 if path.endswith("/training-observations") else 10.0
        assert set(timeout.values()) == {expected}
    assert all(set(timeout.values()) == {runner.API_TIMEOUT_SECONDS}
               for _, timeout in observed[2:])


@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_capture_failure_stays_observational_and_logs_only_safe_fields(failure, capsys):
    attempts = []
    secret = "private-body-token-do-not-log"

    def respond(request):
        attempts.append(request.url.path)
        if failure == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        return httpx.Response(403, json={"status": "rejected", "detail": secret})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        api = runner.HttpArenaApiClient("https://arena.invalid", client=client)
        state = runner.RunState(lease={"run_id": RUN}, lease_token=TOKEN)
        before = (state.action_sequence, list(state.calls), state.local_refusals)
        assert runner._record_training_input(config(api), state.lease, TOKEN, {}) is None
        summary = runner._record_training_observations(config(api), state.lease, TOKEN, state)
        assert summary["status"] == "unavailable"
        assert (state.action_sequence, state.calls, state.local_refusals) == before
    logs = capsys.readouterr().err
    assert secret not in logs and TOKEN not in logs
    assert "exception_class=RunnerError" in logs
    if failure == "timeout":
        assert "transport_cause=ReadTimeout" in logs
        assert len(attempts) == 4
    else:
        assert "http_status=403" in logs and "transport_cause=-" in logs
