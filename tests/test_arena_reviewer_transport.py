from __future__ import annotations

import asyncio
from types import MappingProxyType

import aiohttp
import pytest

from arena_reviewer_runtime.transport import post_json


class _Response:
    status = 503

    def __init__(self, body):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class _Session:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.body)


def test_transport_preserves_provider_response_and_copies_request_mappings():
    body = {"provider": "unavailable"}
    session = _Session(body)
    headers = MappingProxyType({"Authorization": "test-placeholder"})
    payload = MappingProxyType({"model": "frozen-model", "messages": []})
    assert asyncio.run(post_json(
        session, "https://provider.example/chat", headers=headers, payload=payload,
    )) == (503, body)
    assert session.calls == [("https://provider.example/chat", {
        "headers": dict(headers), "json": dict(payload),
    })]
    assert session.calls[0][1]["headers"] is not headers
    assert session.calls[0][1]["json"] is not payload


@pytest.mark.parametrize("error", [ValueError(), aiohttp.ContentTypeError(None, ())])
def test_malformed_json_retains_status_without_a_retry(error):
    session = _Session(error)
    assert asyncio.run(post_json(
        session, "https://provider.example/chat", headers={}, payload={},
    )) == (503, None)
    assert len(session.calls) == 1


def test_network_failure_reaches_existing_frozen_retry_handler():
    class FailedSession:
        def post(self, *_args, **_kwargs):
            raise aiohttp.ClientConnectionError("test transport failure")

    with pytest.raises(aiohttp.ClientConnectionError):
        asyncio.run(post_json(
            FailedSession(), "https://provider.example/chat", headers={}, payload={},
        ))
