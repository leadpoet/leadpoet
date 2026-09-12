from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMITTER = ROOT / "gateway" / "observability" / "emit_restart_marker.py"


def _load_emitter():
    spec = importlib.util.spec_from_file_location("emit_restart_marker", EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _span(payload: dict) -> dict:
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]


def _attrs(span: dict) -> dict:
    return {
        attribute["key"]: next(iter(attribute["value"].values()))
        for attribute in span["attributes"]
    }


def test_marker_pair_shares_a_trace_and_carries_the_outcome() -> None:
    emitter = _load_emitter()

    started = emitter.build_payload(
        event="started",
        status="",
        stage="bootstrap",
        invocation_id="gateway-1757000000-4242",
        candidate_sha="c0ffee",
        elapsed_seconds=0.0,
        now_ns=1_757_000_000_000_000_000,
    )
    finished = emitter.build_payload(
        event="finished",
        status="failed",
        stage="v2_pre_shutdown_preflight",
        invocation_id="gateway-1757000000-4242",
        candidate_sha="c0ffee",
        elapsed_seconds=91.5,
        now_ns=1_757_000_091_500_000_000,
    )

    started_span, finished_span = _span(started), _span(finished)
    assert started_span["traceId"] == finished_span["traceId"]
    assert started_span["spanId"] != finished_span["spanId"]
    assert len(started_span["traceId"]) == 32
    assert len(started_span["spanId"]) == 16

    resource = started["resourceSpans"][0]["resource"]["attributes"]
    assert resource == [
        {
            "key": "service.name",
            "value": {"stringValue": "leadpoet-gateway-restart"},
        }
    ]

    # A failed restart is an error span, so the outcome is readable without
    # having to inspect the attributes.
    assert started_span["status"]["code"] == 0
    assert finished_span["status"]["code"] == 2

    attributes = _attrs(finished_span)
    assert attributes["restart.event"] == "finished"
    assert attributes["restart.status"] == "failed"
    assert attributes["restart.stage"] == "v2_pre_shutdown_preflight"
    assert attributes["restart.component"] == "gateway"
    assert attributes["restart.candidate_sha"] == "c0ffee"
    assert attributes["restart.elapsed_seconds"] == 91.5
    assert attributes["schema.version"] == emitter.SCHEMA_VERSION

    # Empty fields are dropped rather than published as empty strings.
    assert "restart.status" not in _attrs(started_span)

    json.dumps(finished)


def test_a_successful_restart_is_not_an_error_span() -> None:
    emitter = _load_emitter()
    span = _span(
        emitter.build_payload(
            event="finished",
            status="passed",
            stage="complete",
            invocation_id="gateway-1757000000-4242",
            candidate_sha="",
            elapsed_seconds=12.0,
            now_ns=1_757_000_012_000_000_000,
        )
    )
    assert span["status"]["code"] == 0
    assert "restart.candidate_sha" not in _attrs(span)


def test_emitter_never_fails_the_restart_when_telemetry_is_unconfigured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    emitter = _load_emitter()
    monkeypatch.delenv("GATEWAY_OTEL_ENDPOINT", raising=False)
    monkeypatch.delenv("GATEWAY_OTEL_TOKEN", raising=False)

    argv = [
        "--event",
        "started",
        "--env-file",
        str(tmp_path / "absent.env"),
    ]
    assert emitter.main(argv) == 0
    assert emitter.main(argv + ["--strict"]) == 1


def test_restart_script_publishes_both_boundaries() -> None:
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert restart.count('emit_gateway_restart_marker "started"') == 1
    assert restart.count('emit_gateway_restart_marker "finished" "passed"') == 1
    # There is deliberately no "finished" "failed" call.  gw_restart.sh has no
    # EXIT trap today -- `trap on_gateway_restart_exit EXIT` was removed in
    # 1a9d6055 -- so there is no place a failure marker could be hooked without
    # reintroducing one.  An aborted restart is therefore a "started" marker
    # with no "finished" partner, which is itself the signature to alert on.
    assert 'emit_gateway_restart_marker "finished" "failed"' not in restart
    # The marker must be bounded and unable to break the restart.
    assert "timeout 3 \"$GATEWAY_PYTHON_BIN\" \\" in restart


def test_started_marker_precedes_any_shutdown_work() -> None:
    """The boundary must be published before the gateway stops serving."""
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    started = restart.index('emit_gateway_restart_marker "started"')
    # The first stage recorded after the runtime is selected; everything that
    # can take the gateway down happens after it.
    assert started < restart.index('record_gateway_restart_timing "dependency_preflight_complete"')
    assert started < restart.index('record_gateway_restart_timing "pre_shutdown_checks_complete"')
