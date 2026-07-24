from __future__ import annotations

import asyncio

from qualification.scoring import intent_verification_three_stage as verifier


class _Response:
    status_code = 200
    text = ""

    def __init__(self, doc):
        self._doc = doc

    def json(self):
        if isinstance(self._doc, BaseException):
            raise self._doc
        return self._doc


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def post(self, *args, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _completion(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _install_call_fakes(monkeypatch):
    traces = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-secret")
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(
        "research_lab.openrouter_telemetry.include_reasoning_default",
        lambda: False,
    )
    monkeypatch.setattr(
        "research_lab.openrouter_telemetry.record_openrouter_trace",
        lambda **kwargs: traces.append(kwargs) or {"trace_ref": "trace:test"},
    )
    return traces


async def _no_sleep(_seconds):
    return None


def test_malformed_content_retries_then_returns_valid_json(monkeypatch):
    traces = _install_call_fakes(monkeypatch)
    client = _Client(
        [
            _Response(_completion('{"signal_evaluations":[{"signal_status":"supported"}')),
            _Response(_completion('{"signal_evaluations":[]}')),
        ]
    )

    result = asyncio.run(verifier._call_openrouter(client, "test/model", "prompt"))

    assert client.calls == 2
    assert result["answer"] == {"signal_evaluations": []}
    assert "_error" not in result
    assert [trace["outcome"] for trace in traces] == ["response", "response"]


def test_malformed_content_exhaustion_returns_structured_error(monkeypatch):
    _install_call_fakes(monkeypatch)
    client = _Client([_Response(_completion("{")) for _ in range(3)])

    result = asyncio.run(verifier._call_openrouter(client, "test/model", "prompt"))

    assert client.calls == 3
    assert result["_error"] == "invalid_json_content"
    assert result["provider_usage"] == {"trace_ref": "trace:test"}


def test_malformed_response_envelope_retries_without_leaking_body(monkeypatch):
    import json

    traces = _install_call_fakes(monkeypatch)
    client = _Client(
        [
            _Response(json.JSONDecodeError("bad envelope", "private body", 0)),
            _Response(_completion('{"signal_evaluations":[]}')),
        ]
    )

    result = asyncio.run(verifier._call_openrouter(client, "test/model", "prompt"))

    assert client.calls == 2
    assert result["answer"] == {"signal_evaluations": []}
    assert traces[0]["outcome"] == "invalid_json_envelope"
    assert "private body" not in str(traces[0]["response_doc"])


def test_verifier_fails_closed_without_raising_after_malformed_json(monkeypatch):
    _install_call_fakes(monkeypatch)
    client = _Client([_Response(_completion("{")) for _ in range(3)])

    result = asyncio.run(
        verifier.verify_three_stage(
            client,
            company_name="Example Co",
            company_linkedin="",
            company_website="https://example.com",
            source_url="https://example.com/news",
            miner_claim="Example Co announced a new product.",
            target_signal_text="New product announcement",
        )
    )

    assert result["client_ready"] is False
    # Provider failures are UNAVAILABLE, not content rejections (source-
    # grounding taxonomy): still fails closed for publication, but the label
    # lets the evaluator distinguish infrastructure from falsified intent.
    assert result["decision"] == "unavailable"
    assert result["rejection_reason"] == "stage1_llm_error:invalid_json_content"
    assert result["stage1"]["status"] == "llm_error"
