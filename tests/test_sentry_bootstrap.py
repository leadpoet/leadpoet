"""Contract tests for the opt-in, fail-closed Sentry bootstrap.

A stub ``sentry_sdk`` module is injected into ``sys.modules`` so these
tests run identically whether or not the real (optional) dependency is
installed.
"""

from __future__ import annotations

import logging
import sys
import threading
import types

import pytest

from leadpoet_observability import sentry_bootstrap
from leadpoet_observability.sentry_scrubbing import REDACTED_PROTECTED


class _StubScope:
    def __init__(self):
        self.tags = {}
        self.contexts = {}

    def set_tag(self, key, value):
        self.tags[key] = value

    def set_context(self, key, value):
        self.contexts[key] = value


class _StubContextManager:
    def __init__(self, value):
        self.value = value
        self.exits = []

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, traceback):
        self.exits.append((exc_type, exc, traceback))
        return False


class _StubSpan:
    def __init__(self):
        self.data = {}

    def set_data(self, key, value):
        self.data[key] = value


class _StubSentrySdk(types.ModuleType):
    def __init__(self):
        super().__init__("sentry_sdk")
        self.init_calls = []
        self.tags = {}
        self.breadcrumbs = []
        self.events = []
        self.exceptions = []
        self.scopes = []
        self.transactions = []
        self.flush_calls = []
        self.distributions = []
        self.metrics = types.SimpleNamespace(distribution=self._distribution)

    def init(self, **kwargs):
        self.init_calls.append(kwargs)

    def set_tag(self, key, value):
        self.tags[key] = value

    def add_breadcrumb(self, **kwargs):
        self.breadcrumbs.append(kwargs)

    def capture_exception(self, exception):
        self.exceptions.append(exception)

    def capture_event(self, event):
        self.events.append(event)

    def new_scope(self):
        scope = _StubScope()
        self.scopes.append(scope)
        return _StubContextManager(scope)

    def start_transaction(self, **kwargs):
        manager = _StubContextManager(_StubSpan())
        self.transactions.append((kwargs, manager))
        return manager

    def flush(self, timeout):
        self.flush_calls.append(timeout)

    def _distribution(self, name, value, **kwargs):
        self.distributions.append((name, value, kwargs))


@pytest.fixture(autouse=True)
def _reset_bootstrap_state():
    sentry_bootstrap._reset_for_tests()
    yield
    sentry_bootstrap._reset_for_tests()


@pytest.fixture
def stub_sdk(monkeypatch):
    stub = _StubSentrySdk()
    monkeypatch.setitem(sys.modules, "sentry_sdk", stub)
    monkeypatch.setattr(sentry_bootstrap, "_core_integrations", lambda: ["core-errors"])
    return stub


@pytest.fixture
def enabled_env(monkeypatch):
    monkeypatch.setenv(sentry_bootstrap.ENABLED_ENV, "1")
    monkeypatch.setenv(sentry_bootstrap.DSN_ENV, "https://key@example.ingest.sentry.io/1")
    monkeypatch.delenv(sentry_bootstrap.ENVIRONMENT_ENV, raising=False)
    monkeypatch.delenv(sentry_bootstrap.RELEASE_ENV, raising=False)
    monkeypatch.delenv(sentry_bootstrap.EXTRA_PROTECTED_ENV, raising=False)
    monkeypatch.delenv(sentry_bootstrap.MESSAGE_MODE_ENV, raising=False)
    monkeypatch.delenv(sentry_bootstrap.TRACES_SAMPLE_RATE_ENV, raising=False)


def test_complete_noop_without_env_gate(monkeypatch, stub_sdk):
    monkeypatch.delenv(sentry_bootstrap.ENABLED_ENV, raising=False)
    monkeypatch.delenv(sentry_bootstrap.DSN_ENV, raising=False)
    assert sentry_bootstrap.init_sentry("test") is False
    assert stub_sdk.init_calls == []


def test_fallback_diagnostics_use_stderr(capsys):
    sentry_bootstrap._safe_print("leadpoet_sentry_disabled sdk_import=TestError")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "leadpoet_sentry_disabled sdk_import=TestError\n"


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
    # Terminal errors are complete; only explicit operational traces are sampled.
    assert kwargs["traces_sample_rate"] == 0.01
    assert kwargs["propagate_traces"] is False
    assert kwargs["enable_logs"] is False
    assert kwargs["spotlight"] is False
    assert kwargs["keep_alive"] is False
    assert kwargs["auto_session_tracking"] is False
    assert kwargs["send_default_pii"] is False
    assert kwargs["include_local_variables"] is False
    assert kwargs["max_request_body_size"] == "never"
    assert kwargs["auto_enabling_integrations"] is False
    assert kwargs["default_integrations"] is False
    assert kwargs["integrations"] == ["core-errors"]
    assert kwargs["max_breadcrumbs"] == 50
    assert kwargs["shutdown_timeout"] == 1
    assert kwargs["debug"] is False
    assert callable(kwargs["before_send"])
    assert callable(kwargs["before_breadcrumb"])
    assert callable(kwargs["before_send_transaction"])
    assert stub_sdk.tags["leadpoet.component"] == "gateway"


@pytest.mark.parametrize(
    ("configured", "expected"),
    (("0", 0.0), ("0.05", 0.05), ("1", 0.10), ("invalid", 0.01)),
)
def test_trace_sampling_is_bounded(
    monkeypatch, enabled_env, stub_sdk, configured, expected
):
    monkeypatch.setenv(sentry_bootstrap.TRACES_SAMPLE_RATE_ENV, configured)
    assert sentry_bootstrap.init_sentry("gateway") is True
    assert stub_sdk.init_calls[0]["traces_sample_rate"] == expected


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


def test_before_send_transaction_drops_unknown_and_sensitive_payloads(
    enabled_env, stub_sdk
):
    sentry_bootstrap.init_sentry("gateway")
    scrub = stub_sdk.init_calls[0]["before_send_transaction"]
    transaction = scrub(
        {
            "type": "transaction",
            "transaction": "gateway.weight.bundle_generation",
            "request": {"data": "sk-or-v1-never-send"},
            "future_payload": "customer contents",
            "spans": [
                {
                    "op": "leadpoet.weight",
                    "description": "gateway.bundle_generation",
                    "data": {
                        "bundle_hash": "sha256:" + "ab" * 32,
                        "authorization": "Bearer secret-value",
                    },
                }
            ],
        },
        None,
    )
    encoded = repr(transaction)
    assert "request" not in transaction
    assert "future_payload" not in transaction
    assert "sk-or-v1" not in encoded
    assert "Bearer secret-value" not in encoded
    assert transaction["transaction"] == "gateway.weight.bundle_generation"


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


def test_exact_deploy_sha_is_release_fallback(monkeypatch, enabled_env, stub_sdk):
    monkeypatch.setenv("GITHUB_SHA", "cd" * 20)
    sentry_bootstrap.init_sentry("gateway")
    assert stub_sdk.init_calls[0]["release"] == "cd" * 20


def test_git_worktree_release_is_resolved(tmp_path):
    common = tmp_path / "common"
    worktree = tmp_path / "checkout"
    git_dir = common / "worktrees" / "task"
    git_dir.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (git_dir / "HEAD").write_text("ef" * 20 + "\n", encoding="utf-8")
    assert sentry_bootstrap._read_git_release(worktree) == "ef" * 20


def test_event_coalescer_bounds_duplicates_and_memory():
    now = [10.0]
    limiter = sentry_bootstrap._EventCoalescer(
        window_seconds=5,
        max_signatures=2,
        clock=lambda: now[0],
    )
    assert limiter.admit("a") is True
    assert limiter.admit("a") is False
    now[0] += 6
    assert limiter.admit("a") is True
    assert limiter.admit("b") is True
    assert limiter.admit("c") is True
    assert len(limiter._seen) == 2


def test_log_filter_coalesces_before_event_construction():
    record = logging.LogRecord(
        name="gateway.retry",
        level=logging.ERROR,
        pathname="gateway/retry.py",
        lineno=10,
        msg="provider retry %s",
        args=(100,),
        exc_info=None,
    )
    limiter = sentry_bootstrap._SentryLogRecordFilter()
    assert limiter.filter(record) is True
    assert limiter.filter(record) is False


def test_before_send_groups_dynamic_retries_without_hiding_distinct_errors(
    enabled_env, stub_sdk
):
    sentry_bootstrap.init_sentry("gateway")
    before_send = stub_sdk.init_calls[0]["before_send"]
    first = before_send({"logger": "gateway.retry", "message": "retry 100"}, None)
    duplicate = before_send(
        {"logger": "gateway.retry", "message": "retry 101"}, None
    )
    distinct = before_send(
        {"logger": "gateway.retry", "message": "permanent failure"}, None
    )
    assert first is not None
    assert duplicate is None
    assert distinct is not None
    assert first["fingerprint"][0] == "leadpoet-error"


def test_semantic_failure_uses_stable_fingerprint(enabled_env, stub_sdk):
    sentry_bootstrap.init_sentry("gateway")
    before_send = stub_sdk.init_calls[0]["before_send"]
    event = before_send(
        {
            "level": "error",
            "tags": {
                "leadpoet.component": "gateway",
                "leadpoet.stage": "bundle_generation",
                "leadpoet.failure_code": "weight.bundle_divergence",
            },
            "message": "Leadpoet terminal failure",
        },
        None,
    )
    assert event["fingerprint"] == [
        "leadpoet-semantic",
        "gateway",
        "bundle_generation",
        "weight.bundle_divergence",
    ]


def test_manual_span_preserves_original_application_exception(enabled_env, stub_sdk):
    sentry_bootstrap.init_sentry("gateway")
    original = RuntimeError("application failure")
    with pytest.raises(RuntimeError) as raised:
        with sentry_bootstrap.start_sentry_span(
            operation="leadpoet.weight",
            description="gateway.bundle_generation",
            trace_id="ab" * 16,
            data={"epoch_id": 24307},
        ):
            raise original
    assert raised.value is original
    kwargs, manager = stub_sdk.transactions[0]
    assert kwargs["trace_id"] == "ab" * 16
    assert manager.value.data == {"epoch_id": 24307}
    assert manager.exits[0][1] is original


def test_manual_span_sdk_failure_is_a_noop(enabled_env, stub_sdk):
    sentry_bootstrap.init_sentry("gateway")

    def _boom(**kwargs):
        raise RuntimeError("telemetry unavailable")

    stub_sdk.start_transaction = _boom
    with sentry_bootstrap.start_sentry_span(
        operation="leadpoet.weight",
        description="gateway.bundle_generation",
    ) as span:
        assert span is None


def test_initialization_is_singleton_under_concurrency(enabled_env, stub_sdk):
    outcomes = []
    threads = [
        threading.Thread(
            target=lambda: outcomes.append(sentry_bootstrap.init_sentry("gateway"))
        )
        for _ in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == [True] * 12
    assert len(stub_sdk.init_calls) == 1


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
