"""Dependency diagnostics cross the private trajectory without changing execution."""

import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import contracts, runner, trajectory
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PostgrestTransport, hash_lease_token
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, _source_archive_with_requirements, lease, make_config,
)
from tests.lab_arena.trajectory_runner_test import LoggingApi


@pytest.mark.parametrize(
    ("failure", "terminal", "kind", "pip_attempts", "expected_detail"),
    [
        ("invalid_requirement", "model_error", "source_error", 0,
         "requirements.txt may contain only package names and version constraints"),
        ("unknown_pip", "model_error", "source_error", 1, "No matching distribution"),
        ("network", "provider_error", "network_error", 2, "NewConnectionError"),
        ("host", "provider_error", "installer_error", 1, "No module named pip"),
        ("timeout", "provider_error", "timeout", 2, "ReadTimeoutError"),
    ],
)
@pytest.mark.parametrize("upload_unavailable", [False, True])
def test_dependency_failure_private_trajectory_round_trip(
    tmp_path, monkeypatch, capsys, failure, terminal, kind, pip_attempts,
    expected_detail, upload_unavailable,
):
    claimed = lease()
    persisted = []
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "GET" and request.url.path == "/rest/v1/lab_arena_runs":
            assert request.url.params["run_id"] == "eq." + claimed["run_id"]
            return httpx.Response(200, json=[{**claimed, "runner_hotkey": "host-runner"}])
        if request.method == "GET" and request.url.path == "/rest/v1/lab_arena_trajectory_events":
            return httpx.Response(200, json=persisted)
        assert request.method == "POST"
        assert request.url.path == "/rest/v1/rpc/lab_arena_append_trajectory_events_v1"
        body = json.loads(request.content)
        assert body["p_run_id"] == claimed["run_id"]
        assert body["p_lease_token_hash"] == hash_lease_token(claimed["lease_token"])
        events = trajectory.validate_batch({"events": body["p_events"]})
        if upload_unavailable:
            return httpx.Response(503, json={"code": "unavailable"})
        # The controlled database boundary retains the exact JSON wire payload.
        persisted.extend(json.loads(json.dumps(events)))
        return httpx.Response(200, json={"status": "accepted", "accepted": len(events),
                                        "inserted": len(events), "existing": 0})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    store = ArenaStore(PostgrestTransport(
        "https://database.example", service_key="sb_secret_test", http_client=client,
    ))
    gateway = object.__new__(ArenaService)
    gateway._store = store
    gateway._require_round_ownership = lambda _round_id: None

    class GatewayApi(LoggingApi):
        def trajectory(self, run_id, lease_token, events):
            super().trajectory(run_id, lease_token, events)
            return gateway.handle_trajectory(run_id, lease_token, {"events": list(events)})

    api = GatewayApi()
    requirement = "-r other.txt\n" if failure == "invalid_requirement" else "missing-package==1.0.0\n"
    api.source_payload = _source_archive_with_requirements(requirement)
    claimed["source_size_bytes"] = len(api.source_payload)
    api.leases = [claimed]
    pip_calls = []
    message = {
        "unknown_pip": "No matching distribution",
        "network": "NewConnectionError('connection failed')",
        "host": "/usr/bin/python3: No module named pip",
        "timeout": "ReadTimeoutError",
    }.get(failure, "")

    def fake_install_command(command, **kwargs):
        if "pip" in command:
            pip_calls.append(command)
            kwargs["stderr"].write((
                "resolver progress\n" * 1000 + message
                + " Bearer secret-bearer https://user:pass@private.test/pip?token=query-secret"
                + " api_key=plain-secret sk-proj-secret " + claimed["lease_token"] + "\n"
            ).encode())
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_install_command)
    sandbox = BridgingRuntime(calls=0)
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, sandbox)
    config.source_cache = runner.SourceCache(
        tmp_path / "failing-sources", api.source,
        dependency_installer=runner.install_binary_requirements,
    )
    worker = runner.Runner(config)
    try:
        assert worker.run_once() == 1
        read_back = store.list_trajectory_events(claimed["run_id"])
    finally:
        worker.close()
        client.close()
    assert len(pip_calls) == pip_attempts
    assert sandbox.specs == [] and api.provider_frames == []
    assert worker.abandoned == 0
    completion = api.completions[0]["body"]
    result = contracts.validate_run_result(completion["result"])
    assert result["terminal_status"] == terminal
    assert result["resource_summary"]["provider_call_count"] == 0
    assert completion["output"] is None
    assert "diagnostic_detail" not in json.dumps(completion)
    errors = [event for batch in api.batches for event in batch["events"]
              if event["kind"] == "runtime.error"]
    assert len(errors) == (2 if upload_unavailable else 1)  # same bounded replay
    error = errors[0]
    assert error["content"]["failure_kind"] == kind
    assert expected_detail in error["content"]["diagnostic_detail"]
    assert len(error["content"]["diagnostic_detail"]) <= runner.MAX_JUDGE_DIAGNOSTIC_CHARS
    assert read_back == persisted
    serialized = json.dumps(read_back)
    for secret in ("secret-bearer", "user:pass", "query-secret", "plain-secret",
                   "sk-proj-secret", claimed["lease_token"]):
        assert secret not in serialized
    if upload_unavailable:
        assert persisted == []
        assert errors[0]["event_id"] == errors[1]["event_id"]
    else:
        stored_error = next(event for event in persisted if event["kind"] == "runtime.error")
        assert stored_error["content"]["failure_kind"] == kind
        assert expected_detail in stored_error["content"]["diagnostic_detail"]
    if upload_unavailable:
        assert "Arena trajectory upload failed" in capsys.readouterr().err
