from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import urlopen

import pytest
from fastapi import FastAPI

from lab_arena import telemetry
from scripts import run_lab_arena_service as entrypoint
from gateway.observability.otel_bootstrap import ArenaTelemetry
import leadpoet_observability
import lab_arena.wiring


def _capture_lifecycle(monkeypatch: pytest.MonkeyPatch) -> tuple[list[tuple], list[dict]]:
    spans: list[tuple] = []
    stages: list[dict] = []
    monkeypatch.setattr(
        telemetry,
        "record",
        lambda *args, **kwargs: spans.append((*args, kwargs)),
    )
    monkeypatch.setattr(
        leadpoet_observability,
        "record_stage",
        lambda **kwargs: stages.append(kwargs),
    )
    monkeypatch.setattr(entrypoint, "_install_arena_telemetry", lambda app: None)
    monkeypatch.setattr(leadpoet_observability, "init_sentry", lambda **kwargs: False)
    monkeypatch.setitem(
        entrypoint.SOURCE_METADATA, "validator_source_commit", "a" * 40
    )
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("GATEWAY_RESTART_INVOCATION_ID", "gateway-1792000000-1234")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "do-not-export-secret")
    return spans, stages


def test_arena_startup_failure_emits_version_without_started_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spans, stages = _capture_lifecycle(monkeypatch)

    def fail_startup(mode: str) -> tuple:
        raise NameError("contracts is not defined")

    monkeypatch.setattr(lab_arena.wiring, "build_service_from_environment", fail_startup)
    with pytest.raises(NameError, match="contracts"):
        entrypoint.main(["--check-only"])
    assert spans == [("service_preflight", "failed", {"source_sha": "a" * 40})]
    assert [stage["status"] for stage in stages] == ["failed"]
    assert stages[0]["candidate_sha"] == "a" * 40
    assert stages[0]["physical_role"] == "arena-combined"
    output = capsys.readouterr().out
    assert "event=preflight outcome=failed" in output
    assert "source_sha=" + "a" * 40 in output
    assert "do-not-export-secret" not in output


def test_arena_clean_exit_emits_start_and_stop_with_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spans, stages = _capture_lifecycle(monkeypatch)
    flushes = []
    monkeypatch.setattr(telemetry, "flush", lambda: flushes.append(True))
    app = FastAPI()
    service = SimpleNamespace(startup_checks=lambda: {"database_identity": {"current_user": "service_role"}})
    monkeypatch.setattr(
        lab_arena.wiring,
        "build_service_from_environment",
        lambda mode: (service, app),
    )
    def run(*args, **kwargs):
        app.router.on_shutdown[0]()

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    assert entrypoint.main(["--no-driver"]) == 0
    assert spans == [
        ("service_startup", "ok", {"source_sha": "a" * 40}),
        ("service_shutdown", "ok", {"source_sha": "a" * 40}),
    ]
    assert [stage["status"] for stage in stages] == ["ready", "completed"]
    assert {stage["candidate_sha"] for stage in stages} == {"a" * 40}
    assert {stage["physical_role"] for stage in stages} == {"arena-api-8792"}
    assert {stage["restart_invocation_id"] for stage in stages} == {"gateway-1792000000-1234"}
    assert flushes == [True]
    output = capsys.readouterr().out
    assert "event=startup outcome=ok" in output
    assert "event=shutdown outcome=ok" in output
    assert "do-not-export-secret" not in output


def test_arena_server_error_emits_one_failed_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans, _ = _capture_lifecycle(monkeypatch)
    service = SimpleNamespace(startup_checks=lambda: {"database_identity": {"current_user": "service_role"}})
    app = FastAPI()
    monkeypatch.setattr(
        lab_arena.wiring, "build_service_from_environment", lambda mode: (service, app)
    )
    def fail_run(*args, **kwargs):
        raise RuntimeError("server failed")

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fail_run))
    with pytest.raises(RuntimeError, match="server failed"):
        entrypoint.main(["--no-driver"])
    assert [(stage, outcome) for stage, outcome, _ in spans] == [
        ("service_startup", "ok"), ("service_shutdown", "failed")
    ]


@pytest.mark.parametrize("no_driver", [False, True])
def test_arena_sigterm_emits_shutdown_from_real_uvicorn(no_driver: bool) -> None:
    code = """
import os
from fastapi import FastAPI
from types import SimpleNamespace
import lab_arena.wiring
from scripts import run_lab_arena_service as entrypoint
service = SimpleNamespace(
    startup_checks=lambda: {"database_identity": {"current_user": "service_role"}},
    _store=SimpleNamespace(_transport=SimpleNamespace(rpc=lambda *_: {
        "schema_version": "leadpoet.lab_arena.restart_guard_state.v1", "guard_present": False,
    })),
    _config=SimpleNamespace(defaults=SimpleNamespace(daily_cutoff_hour_utc=0)),
    review_pending_submissions=lambda **_: print("ARENA_TEST_REVIEW", flush=True),
)
app = FastAPI()
app.get("/health")(lambda: {"ok": True})
lab_arena.wiring.build_service_from_environment = lambda mode: (service, app)
entrypoint.drive_once = lambda _: print("ARENA_TEST_DRIVE", flush=True) or "idle"
arguments = ["--port", os.environ["ARENA_TEST_PORT"]]
if os.environ["ARENA_TEST_NO_DRIVER"] == "1":
    arguments.append("--no-driver")
raise SystemExit(entrypoint.main(arguments))
"""
    received = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    collector = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    collector_thread = threading.Thread(target=collector.serve_forever, daemon=True)
    collector_thread.start()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {name: value for name, value in os.environ.items() if not name.startswith("OTEL_")}
    env.update(
        LAB_ARENA_MODE="live", ARENA_TEST_PORT=str(port), PYTHONDONTWRITEBYTECODE="1",
        ARENA_TEST_NO_DRIVER="1" if no_driver else "0",
        GATEWAY_OTEL_ENABLED="1",
        GATEWAY_OTEL_ENDPOINT=f"http://127.0.0.1:{collector.server_port}/v1/traces",
        GATEWAY_OTEL_TOKEN="fixture-ingest-token",
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2) as response:
                    assert response.status == 200
                    assert response.read() == b'{"ok":true}'
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("Arena test server did not start")
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        collector.shutdown()
        collector.server_close()
        collector_thread.join(timeout=5)
    assert process.returncode == -signal.SIGTERM
    assert output.count("ARENA_SERVICE_LIFECYCLE event=startup outcome=ok") == 1
    assert output.count("ARENA_SERVICE_LIFECYCLE event=shutdown outcome=ok") == 1
    assert ("ARENA_TEST_DRIVE" in output) is not no_driver
    assert ("ARENA_TEST_REVIEW" in output) is not no_driver
    assert any(b"arena.service_shutdown" in body for body in received)


def test_arena_check_only_does_not_claim_a_service_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans, stages = _capture_lifecycle(monkeypatch)
    service = SimpleNamespace(startup_checks=lambda: {"database_identity": {"current_user": "service_role"}})
    monkeypatch.setattr(
        lab_arena.wiring,
        "build_service_from_environment",
        lambda mode: (service, object()),
    )
    assert entrypoint.main(["--check-only"]) == 0
    assert spans == []
    assert stages == []


def test_driver_only_sigterm_stops_worker_without_serving_api() -> None:
    code = """
import signal
import threading
from fastapi import FastAPI
from types import SimpleNamespace
import lab_arena.wiring
from scripts import run_lab_arena_service as entrypoint
service = SimpleNamespace(
    startup_checks=lambda: {"database_identity": {"current_user": "service_role"}},
    _store=SimpleNamespace(_transport=SimpleNamespace(rpc=lambda *_: {
        "schema_version": "leadpoet.lab_arena.restart_guard_state.v1", "guard_present": False,
    })),
    _config=SimpleNamespace(defaults=SimpleNamespace(daily_cutoff_hour_utc=0)),
    review_pending_submissions=lambda **_: None,
)
lab_arena.wiring.build_service_from_environment = lambda mode: (service, FastAPI())
entrypoint.drive_once = lambda _: signal.raise_signal(signal.SIGTERM) or "idle"
import uvicorn
def refuse_api(*args, **kwargs):
    raise AssertionError("driver-only process attempted to start an API")
uvicorn.run = refuse_api
assert entrypoint.main(["--driver-only"]) == 0
assert not any(t.name in {"lab-arena-driver", "lab-arena-code-review"} for t in threading.enumerate())
print("ARENA_TEST_WORKERS_STOPPED")
"""
    env = dict(os.environ, LAB_ARENA_MODE="live", GATEWAY_OTEL_ENABLED="0", LEADPOET_SENTRY_ENABLED="0")
    result = subprocess.run(
        [sys.executable, "-u", "-c", code], capture_output=True,
        text=True, env=env, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("ARENA_SERVICE_LIFECYCLE event=startup outcome=ok") == 1
    assert result.stdout.count("ARENA_SERVICE_LIFECYCLE event=shutdown outcome=ok") == 1
    assert "physical_role=arena-driver" in result.stdout
    assert "ARENA_TEST_WORKERS_STOPPED" in result.stdout


@pytest.mark.parametrize(
    ("arguments", "role"),
    [(["--no-driver", "--port", "8794"], "arena-api-8794"),
     (["--driver-only"], "arena-driver")],
)
def test_split_process_failure_identifies_exact_role_and_invocation(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], role: str,
) -> None:
    spans, stages = _capture_lifecycle(monkeypatch)
    contexts = []
    monkeypatch.setattr(
        leadpoet_observability, "configure_sentry_context",
        lambda **kwargs: contexts.append(kwargs),
    )
    monkeypatch.setattr(
        lab_arena.wiring, "build_service_from_environment",
        lambda mode: (_ for _ in ()).throw(RuntimeError("startup unavailable")),
    )
    with pytest.raises(RuntimeError, match="startup unavailable"):
        entrypoint.main(arguments)
    assert contexts[0]["physical_role"] == role
    assert contexts[0]["restart_invocation_id"] == "gateway-1792000000-1234"
    assert spans[0][2]["source_sha"] == "a" * 40
    assert stages[0]["physical_role"] == role


def test_broken_telemetry_does_not_mask_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_lifecycle(monkeypatch)
    monkeypatch.setattr(
        telemetry,
        "record",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("telemetry down")),
    )
    monkeypatch.setattr(
        lab_arena.wiring,
        "build_service_from_environment",
        lambda mode: (_ for _ in ()).throw(NameError("contracts is not defined")),
    )
    with pytest.raises(NameError, match="contracts"):
        entrypoint.main(["--check-only"])


def test_arena_flush_uses_bounded_provider_flush_and_is_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        telemetry, "_recorder",
        ArenaTelemetry(None, flush_spans=lambda timeout: calls.append(timeout) or True),
    )
    telemetry.flush(timeout_millis=750)
    assert calls == [750]
    monkeypatch.setattr(
        telemetry, "_recorder",
        ArenaTelemetry(None, flush_spans=lambda timeout: (_ for _ in ()).throw(RuntimeError("export failed"))),
    )
    telemetry.flush(timeout_millis=750)
