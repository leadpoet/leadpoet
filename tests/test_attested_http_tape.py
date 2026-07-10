from __future__ import annotations

import asyncio
import copy
import json

import aiohttp
import httpx
import pytest

from research_lab.eval.http_tape import (
    HttpTapeRecorder,
    HttpTapeError,
    record_provider_http_tape,
    replay_provider_http_tape,
    validate_http_tape,
)


@pytest.mark.asyncio
async def test_httpx_record_replay_is_byte_exact_and_never_calls_live_transport():
    live_calls = []

    async def live_handler(request):
        live_calls.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "set-cookie": "do-not-record"},
            content=b'{"score":17}',
        )

    with record_provider_http_tape() as recorder:
        async with httpx.AsyncClient(transport=httpx.MockTransport(live_handler)) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": "Bearer raw-secret"},
                json={"model": "test/model", "messages": [{"content": "private prompt"}]},
            )
            assert response.content == b'{"score":17}'
    tape = recorder.document()

    async def forbidden_handler(_request):
        raise AssertionError("replay must never call a live transport")

    with replay_provider_http_tape(tape):
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden_handler)) as client:
            replayed = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": "Bearer another-secret"},
                json={"model": "test/model", "messages": [{"content": "private prompt"}]},
            )
            assert replayed.status_code == 200
            assert replayed.content == b'{"score":17}'

    encoded = json.dumps(tape)
    assert "raw-secret" not in encoded
    assert "another-secret" not in encoded
    assert "set-cookie" not in encoded
    assert live_calls == ["https://openrouter.ai/api/v1/chat/completions"]


@pytest.mark.asyncio
async def test_aiohttp_partial_body_consumption_records_only_bytes_scoring_consumed():
    response_body = b"company-page-body"

    async def handler(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: "
            + str(len(response_body)).encode("ascii")
            + b"\r\nConnection: close\r\n\r\n"
            + response_body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    url = "http://127.0.0.1:%s/company" % port
    try:
        with record_provider_http_tape() as recorder:
            async with aiohttp.ClientSession() as client:
                async with client.get(url) as response:
                    consumed = await response.content.read(7)
                    assert consumed == b"company"
        tape = recorder.document()
    finally:
        server.close()
        await server.wait_closed()

    assert tape["total_response_bytes"] == 7
    with replay_provider_http_tape(tape):
        async with aiohttp.ClientSession() as client:
            async with client.get(url) as response:
                assert response.status == 200
                assert await response.content.read(7) == b"company"


@pytest.mark.asyncio
async def test_replay_miss_fails_closed_without_live_fallback():
    calls = 0

    async def live_handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"ok")

    with record_provider_http_tape() as recorder:
        async with httpx.AsyncClient(transport=httpx.MockTransport(live_handler)) as client:
            assert (await client.get("https://api.exa.ai/search?q=one")).status_code == 200
    tape = recorder.document()
    assert calls == 1

    with pytest.raises(HttpTapeError, match="replay miss"):
        with replay_provider_http_tape(tape):
            async with httpx.AsyncClient(transport=httpx.MockTransport(live_handler)) as client:
                await client.get("https://api.exa.ai/search?q=two")
    assert calls == 1


def test_tape_validation_rejects_body_and_summary_tampering():
    recorder = HttpTapeRecorder()
    index = recorder.begin(
        "GET",
        "https://api.scrapingdog.com/scrape?api_key=secret&url=x",
        status=200,
        headers={"content-type": "text/html"},
    )
    recorder.set_body(index, b"evidence")
    tape = recorder.document()
    assert validate_http_tape(tape) == tape

    tampered = copy.deepcopy(tape)
    tampered["entries"][0]["response"]["body_b64"] = "dGFtcGVyZWQ="
    with pytest.raises(HttpTapeError, match="body hash"):
        validate_http_tape(tampered)

    tampered = copy.deepcopy(tape)
    tampered["entry_count"] = 2
    with pytest.raises(HttpTapeError, match="summary"):
        validate_http_tape(tampered)


@pytest.mark.asyncio
async def test_recording_limit_failure_does_not_change_authoritative_http_response(monkeypatch):
    from research_lab.eval import http_tape

    monkeypatch.setattr(http_tape, "MAX_RESPONSE_BYTES", 2)

    async def live_handler(_request):
        return httpx.Response(200, content=b"authoritative-body")

    with record_provider_http_tape() as recorder:
        async with httpx.AsyncClient(transport=httpx.MockTransport(live_handler)) as client:
            response = await client.get("https://api.exa.ai/search")
            assert response.content == b"authoritative-body"

    with pytest.raises(HttpTapeError, match="recording failed"):
        recorder.document()


@pytest.mark.asyncio
async def test_qualification_executor_replays_tape_without_second_provider_call(monkeypatch):
    from gateway.tee.scoring_executor import (
        OP_QUALIFICATION_COMPANY_SCORES,
        execute_scoring_operation,
    )
    from research_lab.eval import evaluator

    url = "https://openrouter.ai/api/v1/chat/completions"
    body = {"model": "test/model", "messages": [{"content": "same prompt"}]}
    live_calls = 0

    async def live_handler(_request):
        nonlocal live_calls
        live_calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"score": 42},
        )

    with record_provider_http_tape() as recorder:
        async with httpx.AsyncClient(transport=httpx.MockTransport(live_handler)) as client:
            assert (await client.post(url, json=body)).json() == {"score": 42}
    tape = recorder.document()

    class FakeScorer:
        async def score_with_breakdowns(self, companies, icp, is_reference_model):
            async def forbidden_handler(_request):
                raise AssertionError("enclave replay attempted a second provider call")

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(forbidden_handler)
            ) as client:
                score = (await client.post(url, json=body)).json()["score"]
            return [{"final_score": score}]

    monkeypatch.setattr(evaluator, "QualificationStyleCompanyScorer", FakeScorer)
    result = await execute_scoring_operation(
        OP_QUALIFICATION_COMPANY_SCORES,
        {
            "companies": [{"company_name": "Example"}],
            "icp": {"industry": "Software"},
            "is_reference_model": False,
            "provider_tape": tape,
        },
    )

    assert result == {"breakdowns": [{"final_score": 42}], "scores": [42.0]}
    assert live_calls == 1


@pytest.mark.asyncio
async def test_live_qualification_executor_keeps_raw_evidence_inside_enclave(monkeypatch):
    from gateway.tee.scoring_executor import (
        OP_QUALIFICATION_COMPANY_SCORES,
        ScoringExecutionResult,
        execute_scoring_operation,
    )
    from research_lab.eval import evaluator

    calls = 0

    async def live_handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"private_provider_value":73}',
        )

    class FakeScorer:
        async def score_with_breakdowns(self, companies, icp, is_reference_model):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(live_handler)
            ) as client:
                value = (
                    await client.get("https://api.exa.ai/search?q=committed")
                ).json()["private_provider_value"]
            return [{"final_score": value}]

    monkeypatch.setattr(evaluator, "QualificationStyleCompanyScorer", FakeScorer)
    execution = await execute_scoring_operation(
        OP_QUALIFICATION_COMPANY_SCORES,
        {
            "companies": [{"company_name": "Example"}],
            "icp": {"industry": "Software"},
            "is_reference_model": False,
            "provider_execution_mode": "live_enclave",
        },
    )

    assert isinstance(execution, ScoringExecutionResult)
    assert execution.result == {
        "breakdowns": [{"final_score": 73}],
        "scores": [73.0],
    }
    assert execution.evidence_roots["provider_http_tape"].startswith("sha256:")
    assert "private_provider_value" not in json.dumps(execution.evidence_roots)
    assert calls == 1
