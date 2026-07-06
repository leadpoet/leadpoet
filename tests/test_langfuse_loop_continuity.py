"""End-to-end Langfuse trace continuity for the auto-research loop.

Pins the invariants the loop-engine instrumentation depends on:

1. ``run_trace_id`` is a pure deterministic function of ``run_id`` — the loop
   engine and the scoring worker derive the SAME trace id independently, so
   continuity needs no id plumbing through candidate records.
2. Sampling seeded by ``run_id`` is all-or-nothing per run: a sampled run
   keeps every span (stages, builds, private eval); an unsampled run keeps
   none. Never fragments.
3. ``observation(trace_id=...)`` attaches to the existing trace via
   ``trace_context`` and degrades to a fresh trace on clients without it.
4. A caller exception raised inside an ``observation`` block propagates
   unchanged (regression: the old wrapper caught it and re-raised as
   ``RuntimeError: generator didn't stop after throw()``, mangling e.g.
   StaleParentDuringScoring whenever Langfuse was live).
"""

from __future__ import annotations

from contextlib import contextmanager
import re

import pytest

import research_lab.observability.langfuse_client as lc


class FakeSpan:
    def __init__(self) -> None:
        self.trace_id = "trace-fake-1"
        self.updates: list[dict] = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class FakeClient:
    def __init__(self, *, accept_trace_context: bool = True) -> None:
        self.accept_trace_context = accept_trace_context
        self.observations: list[dict] = []
        self.span = FakeSpan()

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        if "trace_context" in kwargs and not self.accept_trace_context:
            raise TypeError("unexpected keyword argument 'trace_context'")
        self.observations.append(kwargs)
        yield self.span


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("LANGFUSE_REDACTION_LEVEL", raising=False)


# ---------------------------------------------------------------------------
# Deterministic per-run trace id
# ---------------------------------------------------------------------------


def test_run_trace_id_is_deterministic_and_w3c_shaped() -> None:
    a1 = lc.run_trace_id("run-123")
    a2 = lc.run_trace_id("run-123")
    b = lc.run_trace_id("run-456")
    assert a1 == a2
    assert a1 != b
    assert re.fullmatch(r"[0-9a-f]{32}", a1)


def test_run_trace_id_empty_for_empty_run_id() -> None:
    assert lc.run_trace_id("") == ""
    assert lc.run_trace_id("   ") == ""


# ---------------------------------------------------------------------------
# Deterministic per-run sampling (all-or-nothing)
# ---------------------------------------------------------------------------


def test_deterministic_sampling_is_stable_per_seed(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.5")
    decisions = {lc.deterministic_sampled("run-abc") for _ in range(50)}
    assert len(decisions) == 1  # never flip-flops within a run


def test_deterministic_sampling_respects_rate_bounds(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1.0")
    assert lc.deterministic_sampled("any-run") is True
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.0")
    assert lc.deterministic_sampled("any-run") is False


# ---------------------------------------------------------------------------
# trace_context attachment
# ---------------------------------------------------------------------------


def test_observation_attaches_to_existing_trace(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(lc, "get_langfuse_client", lambda *a, **k: fake)
    trace_id = lc.run_trace_id("run-123")
    with lc.observation("research_lab.loop_stage.planner", metadata={"run_id": "run-123"}, trace_id=trace_id) as obs:
        assert obs is fake.span
    assert fake.observations[0]["trace_context"] == {"trace_id": trace_id}


def test_observation_degrades_without_trace_context_support(monkeypatch) -> None:
    fake = FakeClient(accept_trace_context=False)
    monkeypatch.setattr(lc, "get_langfuse_client", lambda *a, **k: fake)
    with lc.observation("t", metadata={"run_id": "r"}, trace_id="a" * 32) as obs:
        assert obs is fake.span
    assert len(fake.observations) == 1
    assert "trace_context" not in fake.observations[0]


def test_engine_and_scoring_worker_derive_same_trace_id() -> None:
    # Continuity contract: both sides call run_trace_id(run_id); no plumbing.
    assert lc.run_trace_id("run-xyz") == lc.run_trace_id("run-xyz")


# ---------------------------------------------------------------------------
# Exception transparency (regression)
# ---------------------------------------------------------------------------


class _SentinelError(Exception):
    pass


def test_caller_exception_propagates_unchanged(monkeypatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(lc, "get_langfuse_client", lambda *a, **k: fake)
    with pytest.raises(_SentinelError):
        with lc.observation("t", metadata={"run_id": "r"}):
            raise _SentinelError("must not be re-typed by the wrapper")


def test_caller_exception_propagates_when_disabled() -> None:
    with pytest.raises(_SentinelError):
        with lc.observation("t", metadata={"run_id": "r"}):
            raise _SentinelError("no-op path must also be transparent")


def test_span_end_failure_never_masks_caller_exception(monkeypatch) -> None:
    class ExplodingExitClient(FakeClient):
        @contextmanager
        def start_as_current_observation(self, **kwargs):
            try:
                yield self.span
            finally:
                raise RuntimeError("langfuse exit exploded")

    monkeypatch.setattr(lc, "get_langfuse_client", lambda *a, **k: ExplodingExitClient())
    with pytest.raises(_SentinelError):
        with lc.observation("t", metadata={"run_id": "r"}):
            raise _SentinelError("caller error wins over langfuse exit error")
    # And a clean body survives an exploding exit too.
    with lc.observation("t", metadata={"run_id": "r"}):
        pass


# ---------------------------------------------------------------------------
# Sampling wiring in observation()
# ---------------------------------------------------------------------------


def test_observation_sample_seed_reaches_client_factory(monkeypatch) -> None:
    seen: list[dict] = []

    def _capture(*args, **kwargs):
        seen.append({"args": args, "kwargs": kwargs})
        return None

    monkeypatch.setattr(lc, "get_langfuse_client", _capture)
    with lc.observation("t", sample_seed="run-123") as obs:
        assert obs is None
    assert seen[0]["kwargs"].get("sample_seed") == "run-123"


def test_unsampled_run_drops_all_spans(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.0")
    assert lc.get_langfuse_client(sample_seed="run-123") is None


def test_flush_bypasses_sampling(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.0")
    flushed: list[bool] = []

    class FlushClient:
        def flush(self) -> None:
            flushed.append(True)

    def _factory(*args, **kwargs):
        if kwargs.get("skip_sampling"):
            return FlushClient()
        # Sampling path: rate 0.0 must return None.
        return lc.deterministic_sampled(kwargs.get("sample_seed") or "x") and FlushClient() or None

    monkeypatch.setattr(lc, "get_langfuse_client", _factory)
    lc.flush_langfuse()
    assert flushed == [True]
