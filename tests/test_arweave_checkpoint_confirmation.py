from __future__ import annotations

import base64

import pytest
import requests

from gateway.utils import arweave_client


class _Response:
    def __init__(self, *, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_checkpoint_payload_is_canonical():
    first = arweave_client.checkpoint_payload_bytes(
        header={"b": 2, "a": 1},
        signature="sig",
        events=b"events",
        tree_levels=[["root"]],
    )
    second = arweave_client.checkpoint_payload_bytes(
        header={"a": 1, "b": 2},
        signature="sig",
        events=b"events",
        tree_levels=[["root"]],
    )

    assert first == second


@pytest.mark.asyncio
async def test_confirmation_requires_exact_confirmed_readback(monkeypatch):
    expected = b'{"checkpoint":"exact"}'
    encoded = base64.urlsafe_b64encode(expected).rstrip(b"=")

    def get(url, timeout):
        assert timeout == 30
        if url.endswith("/status"):
            return _Response(
                payload={
                    "block_height": 100,
                    "block_indep_hash": "block",
                    "number_of_confirmations": 2,
                }
            )
        assert url.endswith("/data")
        return _Response(content=encoded)

    monkeypatch.setattr(requests, "get", get)

    assert await arweave_client.wait_for_confirmation(
        "A" * 43,
        expected_payload=expected,
        timeout=1,
    )


@pytest.mark.asyncio
async def test_confirmation_rejects_different_readback(monkeypatch):
    def get(url, timeout):
        if url.endswith("/status"):
            return _Response(
                payload={
                    "block_height": 100,
                    "block_indep_hash": "block",
                    "number_of_confirmations": 2,
                }
            )
        return _Response(
            content=base64.urlsafe_b64encode(b"different").rstrip(b"=")
        )

    monkeypatch.setattr(requests, "get", get)

    with pytest.raises(RuntimeError, match="differs"):
        await arweave_client.wait_for_confirmation(
            "A" * 43,
            expected_payload=b"expected",
            timeout=1,
        )
