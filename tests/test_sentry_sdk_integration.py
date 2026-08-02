"""Small real-SDK contract test; no network traffic is permitted."""

from __future__ import annotations

import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from leadpoet_observability import sentry_bootstrap
from leadpoet_observability import sentry_operations


sentry_sdk = pytest.importorskip("sentry_sdk")
from sentry_sdk.transport import Transport


class _CaptureTransport(Transport):
    def __init__(self, events):
        super().__init__()
        self._events = events

    def capture_envelope(self, envelope):
        for item in envelope.items:
            event = item.get_event() or item.get_transaction_event()
            if event is not None:
                self._events.append(event)


def test_real_sdk_captures_only_coalesced_scrubbed_errors(monkeypatch):
    captured = []
    real_init = sentry_sdk.init

    def init_without_network(**options):
        options["transport"] = _CaptureTransport(captured)
        return real_init(**options)

    monkeypatch.setattr(sentry_sdk, "init", init_without_network)
    monkeypatch.setenv(sentry_bootstrap.ENABLED_ENV, "1")
    monkeypatch.setenv(
        sentry_bootstrap.DSN_ENV,
        "https://public@example.ingest.sentry.io/1",
    )
    monkeypatch.setenv("GITHUB_SHA", "ab" * 20)
    sentry_bootstrap._reset_for_tests()
    try:
        assert sentry_bootstrap.init_sentry("sdk-contract") is True
        logger = logging.getLogger("leadpoet.sdk.contract")
        logger.setLevel(logging.INFO)
        hotkey = "5" + "D" * 47
        sentry_sdk.set_extra("person@example.com", "sk-secretvalue123456789")
        sentry_sdk.set_extra("wallet_hotkey", hotkey)
        logger.info("not an event and not a breadcrumb for person@example.com")
        logger.error("provider retry 100 for person@example.com")
        logger.error("provider retry 101 for person@example.com")
        try:
            local_secret = "sk-or-v1-this-must-not-leave"
            raise RuntimeError(f"wallet_hotkey={hotkey} {local_secret}")
        except RuntimeError:
            logger.exception("unexpected provider crash for person@example.com")
        sentry_sdk.capture_event(
            {
                "message": "manual boundary contract",
                "future_payload": "raw-customer-payload-must-not-leave",
            }
        )
        sentry_sdk.flush(timeout=2)
        assert len(captured) == 3
        encoded = repr(captured)
        assert "person@example.com" not in encoded
        assert "not an event" not in encoded
        assert "sk-secretvalue" not in encoded
        assert "sk-or-v1-this-must-not-leave" not in encoded
        assert hotkey not in encoded
        assert "local_secret" not in encoded
        assert "raw-customer-payload-must-not-leave" not in encoded
        assert captured[0]["fingerprint"][0] == "leadpoet-error"
        assert not captured[0].get("breadcrumbs", {}).get("values")
    finally:
        real_init(dsn=None, default_integrations=False)
        sentry_bootstrap._reset_for_tests()


def test_real_sdk_captures_correlated_semantic_event_and_manual_transaction(
    monkeypatch,
):
    captured = []
    real_init = sentry_sdk.init

    def init_without_network(**options):
        options["transport"] = _CaptureTransport(captured)
        return real_init(**options)

    monkeypatch.setattr(sentry_sdk, "init", init_without_network)
    monkeypatch.setattr(sentry_bootstrap, "_MAX_TRACES_SAMPLE_RATE", 1.0)
    monkeypatch.setenv(sentry_bootstrap.ENABLED_ENV, "1")
    monkeypatch.setenv(
        sentry_bootstrap.DSN_ENV,
        "https://public@example.ingest.sentry.io/1",
    )
    monkeypatch.setenv(sentry_bootstrap.TRACES_SAMPLE_RATE_ENV, "1")
    monkeypatch.setenv("GITHUB_SHA", "ab" * 20)
    sentry_bootstrap._reset_for_tests()
    sentry_operations._reset_for_tests()
    try:
        assert sentry_bootstrap.init_sentry("sdk-contract") is True
        correlation = sentry_operations.weight_correlation_id(
            runtime_sha="ab" * 20,
            netuid=71,
            epoch_id=24307,
            bundle_hash="sha256:" + "cd" * 32,
        )
        with sentry_operations.sentry_stage(
            component="validator",
            operation="weight_submission",
            stage="broadcast",
            weight_correlation_id=correlation,
            runtime_sha="ab" * 20,
            netuid=71,
            epoch_id=24307,
            bundle_hash="sha256:" + "cd" * 32,
        ):
            pass
        assert sentry_operations.capture_failure(
            "weight.finalization_missing",
            component="validator",
            stage="last_update_readback",
            weight_correlation_id=correlation,
            runtime_sha="ab" * 20,
            netuid=71,
            epoch_id=24307,
            bundle_hash="sha256:" + "cd" * 32,
        )
        sentry_sdk.flush(timeout=2)
        terminal = next(
            event
            for event in captured
            if event.get("tags", {}).get("leadpoet.failure_code")
            == "weight.finalization_missing"
        )
        transaction = next(
            event for event in captured if event.get("type") == "transaction"
        )
        assert terminal["fingerprint"] == [
            "leadpoet-semantic",
            "validator",
            "last_update_readback",
            "weight.finalization_missing",
        ]
        assert terminal["tags"]["leadpoet.weight_correlation_id"] == correlation
        assert transaction["transaction"] == "validator.broadcast"
        assert (
            transaction["contexts"]["trace"]["trace_id"]
            == sentry_operations._trace_id(
                {"weight_correlation_id": correlation}
            )
        )
        assert "cd" * 32 in repr(transaction)
    finally:
        real_init(dsn=None, default_integrations=False)
        sentry_bootstrap._reset_for_tests()
        sentry_operations._reset_for_tests()


class _SlowCollector(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - stdlib server callback
        time.sleep(0.4)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args):
        return


def test_real_http_transport_does_not_block_capture_path(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowCollector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    real_init = sentry_sdk.init
    port = server.server_address[1]
    monkeypatch.setenv(sentry_bootstrap.ENABLED_ENV, "1")
    monkeypatch.setenv(
        sentry_bootstrap.DSN_ENV,
        f"http://public@127.0.0.1:{port}/1",
    )
    monkeypatch.setenv("GITHUB_SHA", "ab" * 20)
    sentry_bootstrap._reset_for_tests()
    sentry_operations._reset_for_tests()
    try:
        assert sentry_bootstrap.init_sentry("sdk-slow-collector") is True
        started = time.monotonic()
        assert sentry_operations.capture_failure(
            "restart.terminal_failure",
            component="gateway",
            stage="startup",
            restart_invocation_id="restart:slow-collector",
        )
        assert time.monotonic() - started < 0.2
        started = time.monotonic()
        sentry_bootstrap.flush_sentry(timeout=0.05)
        assert time.monotonic() - started < 0.25
    finally:
        real_init(dsn=None, default_integrations=False)
        sentry_bootstrap._reset_for_tests()
        sentry_operations._reset_for_tests()
        server.shutdown()
        server.server_close()


def test_unreachable_http_collector_does_not_block_or_raise(monkeypatch):
    reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
    reservation.close()
    real_init = sentry_sdk.init
    monkeypatch.setenv(sentry_bootstrap.ENABLED_ENV, "1")
    monkeypatch.setenv(
        sentry_bootstrap.DSN_ENV,
        f"http://public@127.0.0.1:{port}/1",
    )
    monkeypatch.setenv("GITHUB_SHA", "ab" * 20)
    sentry_bootstrap._reset_for_tests()
    sentry_operations._reset_for_tests()
    try:
        assert sentry_bootstrap.init_sentry("sdk-unreachable-collector") is True
        started = time.monotonic()
        assert sentry_operations.capture_failure(
            "restart.terminal_failure",
            component="gateway",
            stage="startup",
            restart_invocation_id="restart:unreachable-collector",
        )
        assert time.monotonic() - started < 0.2
        started = time.monotonic()
        sentry_bootstrap.flush_sentry(timeout=0.05)
        assert time.monotonic() - started < 0.25
    finally:
        real_init(dsn=None, default_integrations=False)
        sentry_bootstrap._reset_for_tests()
        sentry_operations._reset_for_tests()
