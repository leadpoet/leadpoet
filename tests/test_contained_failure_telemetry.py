"""P3 (trajectoryimprovements.md): contained stage failures keep the failed
call's provider usage / cost instead of dropping them on the containment
floor. The raw trace is already captured at the HTTP layer — the containment
boundary must propagate the pointer, not lose it."""

from __future__ import annotations

import asyncio

import pytest

import gateway.research_lab.code_loop_engine as engine


class _UsageError(RuntimeError):
    def __init__(self, message, provider_usage=None, cost_microusd=0):
        super().__init__(message)
        self.provider_usage = provider_usage
        self.cost_microusd = cost_microusd


def _engine_with_caller(caller):
    return engine.CodeEditLoopEngine(
        settings=object(),
        call_openrouter=caller,
        event_sink=None,
        builder=None,
    )


def test_contained_failure_is_str_and_carries_usage(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT", "true")
    usage = {
        "model": "test/model",
        "raw_trace_ref": {"s3_ref": "s3://b/t/0001.json.enc", "sha256": "sha256:aa"},
    }

    async def failing_caller(messages, timeout, max_tokens, stage):
        raise _UsageError("boom", provider_usage=usage, cost_microusd=1234)

    loop_engine = _engine_with_caller(failing_caller)
    result, failure = asyncio.run(
        loop_engine._call_stage_contained([], 10, 100, "code_edit_draft")
    )
    assert result is None
    assert isinstance(failure, engine.ContainedStageFailure)
    assert isinstance(failure, str)  # legacy error-text plumbing still works
    assert "boom" in failure
    assert failure.cost_microusd == 1234
    assert failure.provider_usage["raw_trace_ref"]["s3_ref"].startswith("s3://")
    entries = failure.failure_usage_entries()
    assert entries[0]["call_stage"] == "code_edit_draft"
    assert entries[0]["call_outcome"] == "contained_failure"


def test_contained_failure_without_usage_yields_empty_entries(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT", "true")

    async def failing_caller(messages, timeout, max_tokens, stage):
        raise RuntimeError("plain failure")

    loop_engine = _engine_with_caller(failing_caller)
    result, failure = asyncio.run(
        loop_engine._call_stage_contained([], 10, 100, "source_inspection")
    )
    assert result is None
    assert failure.provider_usage is None
    assert failure.failure_usage_entries() == []
    assert failure.cost_microusd == 0


def test_success_path_unchanged(monkeypatch):
    async def ok_caller(messages, timeout, max_tokens, stage):
        return "raw text result"

    loop_engine = _engine_with_caller(ok_caller)
    result, failure = asyncio.run(
        loop_engine._call_stage_contained([], 10, 100, "code_edit_draft")
    )
    assert failure is None
    assert result.content == "raw text result"
