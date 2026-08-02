"""Contract tests for the opt-in, fail-closed Sentry bootstrap.

A stub ``sentry_sdk`` module is injected into ``sys.modules`` so these
tests run identically whether or not the real (optional) dependency is
installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from leadpoet_observability import sentry_bootstrap
from leadpoet_observability.sentry_scrubbing import REDACTED_PROTECTED


class _StubSentrySdk(types.ModuleType):
    def __init__(self):
        super().__init__("sentry_sdk")
        self.init_calls = []
        self.tags = {}

    def init(self, **kwargs):
        self.init_calls.append(kwargs)

    def set_tag(self, key, value):
        self.tags[key] = value


@pytest.fixture(autouse=True)
def _reset_bootstrap_state():
    sentry_bootstrap._reset_for_tests()
    yield
    sentry_bootstrap._reset_for_tests()


@pytest.fixture
def stub_sdk(monkeypatch):
    stub = _StubSentrySdk()
    monkeypatch.setitem(sys.modules, "sentry_sdk", stub)
    return stub


@pytest.fixture
def enabled_env(monkeypatch):
    monkeypatch.setenv(sentry_bootstrap.ENABLED_ENV, "1")
    monkeypatch.setenv(sentry_bootstrap.DSN_ENV, "https://key@example.ingest.sentry.io/1")
    monkeypatch.delenv(sentry_bootstrap.ENVIRONMENT_ENV, raising=False)
    monkeypatch.delenv(sentry_bootstrap.RELEASE_ENV, raising=False)
    monkeypatch.delenv(sentry_bootstrap.EXTRA_PROTECTED_ENV, raising=False)
    monkeypatch.delenv(sentry_bootstrap.MESSAGE_MODE_ENV, raising=False)


def test_complete_noop_without_env_gate(monkeypatch, stub_sdk):
    monkeypatch.delenv(sentry_bootstrap.ENABLED_ENV, raising=False)
    monkeypatch.delenv(sentry_bootstrap.DSN_ENV, raising=False)
    assert sentry_bootstrap.init_sentry("test") is False
    assert stub_sdk.init_calls == []


def test_enabled_flag_alone_is_not_enough(monkeypatch, stub_sdk):
    monkeypatch.setenv(sentry_bootstrap.ENABLED_ENV, "true")
    monkeypatch.delenv(sentry_bootstrap.DSN_ENV, raising=False)
    assert sentry_bootstrap.init_sentry("test") is False
    assert stub_sdk.init_calls == []


def test_dsn_alone_is_not_enough(monkeypatch, stub_sdk):
    monkeypatch.delenv(sentry_bootstrap.ENABLED_ENV, raising=False)
    monkeypatch.setenv(sentry_bootstrap.DSN_ENV, "https://key@example/1")
    assert sentry_bootstrap.init_sentry("test") is False
    assert stub_sdk.init_calls == []


def test_initializes_with_hard_off_capture_options(enabled_env, stub_sdk):
    assert sentry_bootstrap.init_sentry("gateway") is True
    assert len(stub_sdk.init_calls) == 1
    kwargs = stub_sdk.init_calls[0]
    assert kwargs["dsn"] == "https://key@example.ingest.sentry.io/1"
    assert kwargs["environment"] == "production"
    # Errors only; payload capture hard-off.
    assert kwargs["traces_sample_rate"] is None
    assert kwargs["auto_session_tracking"] is False
    assert kwargs["send_default_pii"] is False
    assert kwargs["include_local_variables"] is False
    assert kwargs["max_request_body_size"] == "never"
    assert kwargs["auto_enabling_integrations"] is False
    assert kwargs["debug"] is False
    assert callable(kwargs["before_send"])
    assert callable(kwargs["before_breadcrumb"])
    assert stub_sdk.tags["leadpoet.component"] == "gateway"


def test_first_call_wins_and_later_calls_do_not_reinit(enabled_env, stub_sdk):
    assert sentry_bootstrap.init_sentry("gateway") is True
    assert sentry_bootstrap.init_sentry("validator") is True
    assert len(stub_sdk.init_calls) == 1
    assert stub_sdk.tags["leadpoet.component"] == "gateway"


def test_missing_sdk_is_swallowed(monkeypatch, enabled_env):
    # None in sys.modules makes ``import sentry_sdk`` raise ImportError.
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    assert sentry_bootstrap.init_sentry("gateway") is False


def test_sdk_init_failure_is_swallowed(enabled_env, stub_sdk):
    def _boom(**kwargs):
        raise RuntimeError("unknown option")

    stub_sdk.init = _boom
    assert sentry_bootstrap.init_sentry("gateway") is False


def test_before_send_scrubs_and_fails_closed(enabled_env, stub_sdk, monkeypatch):
    sentry_bootstrap.init_sentry("gateway")
    before_send = stub_sdk.init_calls[0]["before_send"]

    event = {"message": "user a@b.co failed", "request": {"data": "body"}}
    scrubbed = before_send(event, None)
    assert "request" not in scrubbed
    assert "a@b.co" not in scrubbed["message"]

    def _raise(*args, **kwargs):
        raise ValueError("scrubber broke")

    monkeypatch.setattr(
        sentry_bootstrap.sentry_scrubbing, "scrub_event", _raise
    )
    assert before_send({"message": "x"}, None) is None


def test_before_breadcrumb_drops_protected_categories(enabled_env, stub_sdk):
    sentry_bootstrap.init_sentry("gateway")
    before_breadcrumb = stub_sdk.init_calls[0]["before_breadcrumb"]
    assert before_breadcrumb({"category": "research_lab.engine_v1", "message": "m"}, None) is None
    kept = before_breadcrumb({"category": "gateway.db", "message": "ok"}, None)
    assert kept["message"] == "ok"


def test_extra_protected_modules_env_widens_redaction(monkeypatch, enabled_env, stub_sdk):
    monkeypatch.setenv(sentry_bootstrap.EXTRA_PROTECTED_ENV, "myco.private, other.pkg")
    sentry_bootstrap.init_sentry("gateway")
    before_send = stub_sdk.init_calls[0]["before_send"]
    event = {
        "exception": {
            "values": [
                {
                    "type": "ValueError",
                    "value": "secret sauce",
                    "stacktrace": {
                        "frames": [{"module": "myco.private.engine", "filename": "x.py"}]
                    },
                }
            ]
        }
    }
    scrubbed = before_send(event, None)
    assert scrubbed["exception"]["values"][0]["value"] == REDACTED_PROTECTED


def test_redact_all_message_mode(monkeypatch, enabled_env, stub_sdk):
    monkeypatch.setenv(sentry_bootstrap.MESSAGE_MODE_ENV, "redact-all")
    sentry_bootstrap.init_sentry("gateway")
    before_send = stub_sdk.init_calls[0]["before_send"]
    scrubbed = before_send({"message": "ordinary infra message"}, None)
    assert scrubbed["message"] == REDACTED_PROTECTED


def test_release_env_override_wins(monkeypatch, enabled_env, stub_sdk):
    monkeypatch.setenv(sentry_bootstrap.RELEASE_ENV, "ab" * 20)
    sentry_bootstrap.init_sentry("gateway")
    assert stub_sdk.init_calls[0]["release"] == "ab" * 20


def test_set_sentry_tag_is_inert_when_inactive(stub_sdk):
    sentry_bootstrap.set_sentry_tag("validator.mode", "worker")
    assert stub_sdk.tags == {}


def test_set_sentry_tag_scrubs_values_when_active(enabled_env, stub_sdk):
    sentry_bootstrap.init_sentry("gateway")
    sentry_bootstrap.set_sentry_tag("note", "reach me at a@b.co")
    assert "a@b.co" not in stub_sdk.tags["leadpoet.note"]


def test_init_never_raises_even_on_weird_environment(monkeypatch, stub_sdk):
    monkeypatch.setenv(sentry_bootstrap.ENABLED_ENV, "1")
    monkeypatch.setenv(sentry_bootstrap.DSN_ENV, "   ")
    # Whitespace DSN fails the gate: still a clean no-op, never a raise.
    assert sentry_bootstrap.init_sentry("gateway") is False
    assert stub_sdk.init_calls == []
