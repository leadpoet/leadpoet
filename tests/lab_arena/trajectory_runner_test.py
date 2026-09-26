"""Runtime log delivery preserves provider, completion and secret boundaries."""

import json
from dataclasses import replace

import httpx
import pytest

from lab_arena import runner, runtime, trajectory
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, lease, make_config, valid_company,
)


class LoggingApi(FakeApi):
    def __init__(self, *, failures=0):
        super().__init__([lease()])
        self.batches = []
        self.failures = failures
        self.order = []

    def trajectory(self, run_id, lease_token, events):
        self.batches.append({"run_id": run_id, "events": list(events)})
        self.order.append("trajectory")
        trajectory.validate_batch({"events": list(events)})
        if self.failures:
            self.failures -= 1
            raise runner.RunnerError("transport failure")
        return {"accepted": len(events)}

    def complete(self, envelope):
        self.order.append("complete")
        return super().complete(envelope)


def run_model(tmp_path, api, sandbox):
    (tmp_path / "work").mkdir()
    worker = runner.Runner(make_config(tmp_path, api, sandbox))
    try:
        assert worker.run_once() == 1
    finally:
        worker.close()
    return worker


def test_runtime_events_precede_accepted_completion_and_retry_same_ids(tmp_path):
    api = LoggingApi(failures=1)
    run_model(tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]}))
    assert api.batches[0] == api.batches[1]
    events = [event for batch in api.batches[1:] for event in batch["events"]]
    assert [event["kind"] for event in events] == [
        "runtime.started", "runtime.finished", "runtime.stdout",
    ]
    finished = events[1]["content"]
    assert finished["status"] == "accepted"
    assert finished["company_count"] == 1
    assert finished["resource_summary"]["provider_call_count"] == 1
    assert events[2]["content"]["text"] == "model log line\n"
    assert api.order[-1] == "complete"
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    assert all(set(event) == {"kind", "event_id", "occurred_at", "content"} for event in events)


def test_logging_outage_does_not_lose_model_completion(tmp_path, capsys):
    api = LoggingApi(failures=100)
    worker = run_model(tmp_path, api, BridgingRuntime(output={"companies": []}))
    assert worker.abandoned == 0
    assert len(api.completions) == 1
    assert "Arena trajectory upload failed: RunnerError" in capsys.readouterr().err


def test_runtime_crash_retains_error_without_exposing_exception_text(tmp_path):
    class BrokenRuntime:
        def run_icp(self, spec):
            raise ValueError("secret-private-exception-payload")

    api = LoggingApi()
    worker = run_model(tmp_path, api, BrokenRuntime())
    assert worker.abandoned == 1
    events = [event for batch in api.batches for event in batch["events"]]
    assert events[-1]["kind"] == "runtime.error"
    assert events[-1]["content"] == {"status": "abandoned", "error_class": "ValueError"}
    assert "secret-private-exception-payload" not in json.dumps(events)


@pytest.mark.parametrize("text", ["\x00" * 65536, "\U0001f600" * 16384, "line\n" * 13108], ids=["control", "unicode", "lines"])
def test_full_bounded_stream_survives_event_and_batch_limits(text, tmp_path):
    raw = text.encode()[:runtime.MAX_LOG_BYTES]
    result = replace(runtime.fake_result(stdout=raw, stderr=raw), stdout_truncated=True)
    events = runner._runtime_log_events(result, "private-lease-token", "2026-09-25T12:00:00Z")
    for stream in ("stdout", "stderr"):
        selected = [event for event in events if event["kind"] == "runtime." + stream]
        expected = trajectory.redact_text(raw.decode()).encode()[:runtime.MAX_LOG_BYTES].decode(errors="replace")
        assert "".join(event["content"]["text"] for event in selected) == expected
        assert "\x00" not in json.dumps(selected)
        assert [event["content"]["sequence"] for event in selected] == list(range(len(selected)))
    api = LoggingApi()
    config = make_config(tmp_path, api, None)
    runner._record_trajectory(config, lease(), "private-lease-token", events)
    assert sum(len(batch["events"]) for batch in api.batches) == len(events)


def test_redaction_happens_before_runtime_chunking():
    secret = "lease-secret-" + "x" * 80
    text = "a" * 1000 + secret + "\nAuthorization: Bearer abcdefghijklmnop\n"
    events = runner._runtime_log_events(
        runtime.fake_result(stdout=text.encode()), secret, "2026-09-25T12:00:00Z",
    )
    encoded = json.dumps(events)
    assert secret not in encoded and "abcdefghijklmnop" not in encoded
    assert "[REDACTED]" in encoded


def test_external_client_upload_uses_only_existing_gateway_lease(monkeypatch):
    for key in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY"):
        monkeypatch.delenv(key, raising=False)
    event = trajectory.event("runtime.started", {"status": "starting"})

    def respond(request):
        assert request.url.path == "/arena/v1/runs/run-123/trajectory"
        assert request.headers["x-lab-arena-lease"] == "existing-lease"
        assert "authorization" not in request.headers and "apikey" not in request.headers
        assert json.loads(request.content) == {"events": [event]}
        return httpx.Response(200, json={"accepted": 1})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        api = runner.HttpArenaApiClient("https://gateway.example", client=client)
        assert api.trajectory("run-123", "existing-lease", [event]) == {"accepted": 1}
