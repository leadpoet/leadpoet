"""Small real-SDK contract test; no network traffic is permitted."""

from __future__ import annotations

import logging

import pytest

from leadpoet_observability import sentry_bootstrap


sentry_sdk = pytest.importorskip("sentry_sdk")
from sentry_sdk.transport import Transport


class _CaptureTransport(Transport):
    def __init__(self, events):
        super().__init__()
        self._events = events

    def capture_envelope(self, envelope):
        for item in envelope.items:
            event = item.get_event()
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
