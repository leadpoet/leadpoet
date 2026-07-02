"""Regression tests for the Langfuse client wrapper (plan §28.2).

Pins the two invariants the scoring path depends on: (1) with
``LANGFUSE_ENABLED`` unset/false every wrapper call is an inert no-op, and
(2) with a client present, a Langfuse or redaction failure is contained —
it can never raise into (or delay) an evaluation. Also pins the metadata
whitelist's miner-identity discipline.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import research_lab.observability.langfuse_client as lc
import research_lab.observability.score_export as se
from research_lab.observability.schema import SAFE_TRACE_METADATA_KEYS
from research_lab.observability.tracing import finish_score_bundle_observation


class FakeSpan:
    def __init__(self) -> None:
        self.trace_id = "trace-fake-1"
        self.updates: list[dict] = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class FakeClient:
    """Captures wrapper calls; can be armed to explode."""

    def __init__(self, *, explode: bool = False) -> None:
        self.explode = explode
        self.observations: list[dict] = []
        self.scores: list[dict] = []
        self.flushed = 0
        self.span = FakeSpan()

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        if self.explode:
            raise RuntimeError("langfuse exploded")
        self.observations.append(kwargs)
        yield self.span

    def score(self, **kwargs) -> None:
        if self.explode:
            raise RuntimeError("langfuse exploded")
        self.scores.append(kwargs)

    def flush(self) -> None:
        if self.explode:
            raise RuntimeError("langfuse exploded")
        self.flushed += 1


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("LANGFUSE_REDACTION_LEVEL", raising=False)


# ---------------------------------------------------------------------------
# Disabled = inert
# ---------------------------------------------------------------------------


def test_disabled_by_default() -> None:
    assert lc.langfuse_enabled() is False


def test_observation_is_noop_when_disabled() -> None:
    with lc.observation("research_lab.private_eval_pair", metadata={"run_id": "r"}) as obs:
        assert obs is None


def test_update_and_flush_are_noops_when_disabled() -> None:
    lc.update_observation(None, output={"x": 1})
    lc.flush_langfuse()


def test_explicit_false_stays_disabled(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert lc.get_langfuse_client() is None


# ---------------------------------------------------------------------------
# Failures are contained, payloads are redacted
# ---------------------------------------------------------------------------


def test_exploding_client_never_raises(monkeypatch) -> None:
    fake = FakeClient(explode=True)
    monkeypatch.setattr(lc, "get_langfuse_client", lambda: fake)
    with lc.observation("research_lab.private_eval_pair", metadata={"run_id": "r"}) as obs:
        assert obs is None
    lc.flush_langfuse()  # must not raise


def test_redaction_blocked_yields_none_and_client_untouched(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(lc, "get_langfuse_client", lambda: fake)
    with lc.observation("t", metadata={"prompt": "raw llm text"}) as obs:
        assert obs is None
    assert fake.observations == []


def test_metadata_is_redacted_before_reaching_client(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(lc, "get_langfuse_client", lambda: fake)
    with lc.observation(
        "t",
        metadata={"run_id": "r1", "api_key": "raw-key-value", "miner_hotkey": "5Fraw"},
    ) as obs:
        assert obs is fake.span
    sent = fake.observations[0]["metadata"]
    assert sent["run_id"] == "r1"
    assert "raw-key-value" not in str(sent)
    assert "5Fraw" not in str(sent)


def test_update_observation_failure_contained(monkeypatch) -> None:
    class ExplodingSpan:
        def update(self, **kwargs):
            raise RuntimeError("update failed")

    lc.update_observation(ExplodingSpan(), output={"ok": 1})  # must not raise


def test_finish_score_bundle_observation_none_obs_is_noop() -> None:
    assert finish_score_bundle_observation(None, {"aggregates": {}}) == ""


def test_finish_score_bundle_exports_via_span_trace_id(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(se, "get_langfuse_client", lambda: fake)
    span = FakeSpan()
    bundle = {
        "score_bundle_hash": "sha256:" + "b" * 64,
        "aggregates": {"mean_delta": 4.0, "base_score": 28.0, "candidate_score": 32.0},
    }
    trace_id = finish_score_bundle_observation(span, bundle)
    assert trace_id == "trace-fake-1"
    assert span.updates, "output must be attached to the span"
    assert {s["name"] for s in fake.scores} >= {"mean_delta", "base_score", "candidate_score"}


# ---------------------------------------------------------------------------
# Metadata whitelist discipline
# ---------------------------------------------------------------------------


def test_whitelist_carries_hash_never_raw_miner_identity() -> None:
    assert "miner_hotkey_hash" in SAFE_TRACE_METADATA_KEYS
    assert "miner_hotkey" not in SAFE_TRACE_METADATA_KEYS


def test_whitelist_keys_survive_redaction() -> None:
    from research_lab.observability.redaction import redact_for_langfuse

    payload = {key: f"value-{key}" for key in sorted(SAFE_TRACE_METADATA_KEYS)}
    assert redact_for_langfuse(payload) == payload
