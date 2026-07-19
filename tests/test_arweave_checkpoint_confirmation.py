from __future__ import annotations

import base64

import pytest
import requests

from gateway.utils import arweave_client


class _Response:
    def __init__(
        self,
        *,
        status_code=200,
        payload=None,
        content=b"",
        text="",
    ):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = text

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
async def test_checkpoint_upload_requires_network_acceptance(monkeypatch):
    class FakeTransaction:
        api_url = "https://arweave.example"
        id = "T" * 43
        json_data = '{"signed":true}'

        def __init__(self, wallet, data):
            assert wallet is arweave_client._wallet
            assert data

        def add_tag(self, key, value):
            assert key
            assert value

        def sign(self):
            return None

    attempts = []

    def post(url, *, data, headers, timeout):
        attempts.append(
            {
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _Response(status_code=400, text="invalid transaction")

    monkeypatch.setattr(arweave_client, "_wallet", object())
    monkeypatch.setattr(arweave_client, "Transaction", FakeTransaction)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(arweave_client.asyncio, "sleep", lambda _: _no_op())

    with pytest.raises(RuntimeError, match=r"HTTP 400.*invalid transaction"):
        await arweave_client.upload_checkpoint(
            header={
                "checkpoint_number": 7,
                "event_count": 1,
                "merkle_root": "a" * 64,
                "time_range": {"start": "start", "end": "end"},
            },
            signature="signature",
            events=b"events",
            tree_levels=[["root"]],
        )

    assert len(attempts) == arweave_client.MAX_RETRIES
    assert all(item["url"] == "https://arweave.example/tx" for item in attempts)
    assert all(item["timeout"] == 60 for item in attempts)


async def _no_op():
    return None


@pytest.mark.asyncio
async def test_checkpoint_upload_returns_id_after_http_200(monkeypatch):
    class FakeTransaction:
        api_url = "https://arweave.example"
        id = "A" * 43
        json_data = '{"signed":true}'

        def __init__(self, wallet, data):
            assert wallet is arweave_client._wallet

        def add_tag(self, key, value):
            return None

        def sign(self):
            return None

    monkeypatch.setattr(arweave_client, "_wallet", object())
    monkeypatch.setattr(arweave_client, "Transaction", FakeTransaction)
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(status_code=200),
    )

    tx_id = await arweave_client.upload_checkpoint(
        header={
            "checkpoint_number": 8,
            "event_count": 1,
            "merkle_root": "b" * 64,
            "time_range": {"start": "start", "end": "end"},
        },
        signature="signature",
        events=b"events",
        tree_levels=[["root"]],
    )

    assert tx_id == "A" * 43


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
