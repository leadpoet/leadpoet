"""Training evidence has its own bounded transport budget and safe failures."""

import json
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import contracts, runner, training_input


TOKEN = "private-lease-value-do-not-log"
RUN = "capture:test"
CLAIM_HASH = "sha256:" + "a" * 64


def active_lease():
    return {"run_id": RUN, "kind": "execute", "lease_generation": 1,
            "claim_request_hash": CLAIM_HASH}


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
        response = {"status": "stored"}
        if request.url.path.endswith("/training-observations"):
            response["payload_sha256"] = contracts.document_hash(json.loads(request.content))
        elif request.url.path.endswith("/training-input"):
            snapshot, _ = training_input.build_snapshot(
                RUN, "execute", json.loads(request.content), lease_generation=1,
                claim_request_hash=CLAIM_HASH, secrets=(TOKEN,),
            )
            response.update({key: snapshot[key] for key in
                             ("schema_version", "lease_generation", "claim_request_hash",
                              "input_sha256", "privacy")})
            response["ref"] = training_input.object_ref(RUN, 1)
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        api = runner.HttpArenaApiClient("https://arena.invalid", client=client)
        state = runner.RunState(lease=active_lease(), lease_token=TOKEN)
        assert runner._record_training_input(config(api), state.lease, TOKEN, {"input": "captured"}) is True
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
@pytest.mark.parametrize("max_attempts", [1, 2])
def test_capture_failure_stays_observational_and_logs_only_safe_fields(failure, max_attempts, capsys):
    attempts = []
    secret = "private-body-token-do-not-log"

    def respond(request):
        attempts.append((request.method, request.url.path))
        if failure == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        return httpx.Response(403, json={"status": "rejected", "detail": secret})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        api = runner.HttpArenaApiClient("https://arena.invalid", client=client)
        state = runner.RunState(lease=active_lease(), lease_token=TOKEN)
        before = (state.action_sequence, list(state.calls), state.local_refusals)
        assert runner._record_training_input(config(api), state.lease, TOKEN, {}, max_attempts=max_attempts) is False
        summary = runner._record_training_observations(config(api), state.lease, TOKEN, state)
        assert summary["status"] == "unavailable"
        assert (state.action_sequence, state.calls, state.local_refusals) == before
    logs = capsys.readouterr().err
    assert secret not in logs and TOKEN not in logs
    assert "exception_class=RunnerError" in logs
    if failure == "timeout":
        assert "transport_cause=ReadTimeout" in logs
    else:
        assert "http_status=403" in logs and "transport_cause=-" in logs
    input_path = f"/arena/v1/runs/{RUN}/training-input"
    observations_path = f"/arena/v1/runs/{RUN}/training-observations"
    assert attempts == (
        [("POST", input_path)] * (max_attempts if failure == "timeout" else 1)
        + [("POST", observations_path)] * 2
        + [("HEAD", observations_path)]
    )
