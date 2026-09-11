import json
import sys
import types
import urllib.error

import pytest

from lab_arena.chain import ArenaChainConfig, load_arena_cutover
from lab_arena.local_weight_signer import (
    BittensorDrandBackend,
    HttpsJsonRpcTransport,
    LocalWeightSignerError,
    build_local_weight_signer,
    load_public_chain_signing_profile,
)
from lab_arena.signing import LocalSigner, signing_key_document
from validator_tee.enclave.chain_source_v2 import ValidatorChainSourceV2Error


class _Response:
    def __init__(self, body, *, status=200, declared=None):
        self._body = body
        self._status = status
        self.headers = {}
        if declared is not None:
            self.headers["Content-Length"] = str(declared)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self._status

    def read(self, limit):
        return self._body[:limit]

    def close(self):
        self.closed = True


class _Transport:
    def __init__(self):
        self.closed = 0

    def call(self, **_kwargs):
        raise AssertionError("builder must not read the chain")

    def close(self):
        self.closed += 1


def _arena_key():
    signer = LocalSigner.generate()
    document = signing_key_document(signer.public_key_der)
    return document, document["public_key_hash"]


def test_https_transport_converts_wss_and_binds_strict_json_rpc():
    observed = {}

    def open_request(request, *, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        observed["request"] = json.loads(request.data)
        return _Response(
            json.dumps({"jsonrpc": "2.0", "id": 7, "result": "0xabc"}).encode()
        )

    transport = HttpsJsonRpcTransport(
        "wss://chain.example:443", timeout_seconds=9, opener=open_request
    )
    assert transport.call(method="chain_getBlockHash", params=[3], request_id=7) == "0xabc"
    assert observed == {
        "url": "https://chain.example:443/",
        "timeout": pytest.approx(9),
        "request": {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "chain_getBlockHash",
            "params": [3],
        },
    }
    transport.close()
    with pytest.raises(ValidatorChainSourceV2Error, match="closed"):
        transport.call(method="chain_getBlockHash", params=[3], request_id=7)


def test_https_transport_rejects_plaintext_wan_and_oversized_reply():
    with pytest.raises(LocalWeightSignerError, match="loopback"):
        HttpsJsonRpcTransport("ws://chain.example", timeout_seconds=1)

    transport = HttpsJsonRpcTransport(
        "https://chain.example",
        timeout_seconds=1,
        max_response_bytes=8,
        opener=lambda *_args, **_kwargs: _Response(b"{}", declared=9),
    )
    with pytest.raises(ValidatorChainSourceV2Error, match="exceeds"):
        transport.call(method="chain_getBlockHash", params=[3], request_id=1)


def test_https_transport_retries_transient_http_within_original_deadline():
    now = [100.0]
    sleeps = []
    observed = []

    def open_request(request, *, timeout):
        observed.append((bytes(request.data), timeout))
        now[0] += 2.0
        if len(observed) == 1:
            response = _Response(b"")
            error_responses.append(response)
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "Thu, 01 Jan 1970 00:01:43 GMT"},
                response,
            )
        return _Response(
            json.dumps({"jsonrpc": "2.0", "id": 7, "result": "0xabc"}).encode()
        )

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    error_responses = []
    transport = HttpsJsonRpcTransport(
        "https://chain.example",
        timeout_seconds=9,
        opener=open_request,
        clock=lambda: now[0],
        wall_clock=lambda: 100.0,
        sleep=sleep,
    )
    assert transport.call(method="chain_getBlockHash", params=[3], request_id=7) == "0xabc"
    assert sleeps == [3.0]
    assert [timeout for _body, timeout in observed] == [9, 4.0]
    assert observed[0][0] == observed[1][0]
    assert error_responses[0].closed is True


def test_https_transport_exhausts_bounded_transient_http_retries():
    calls = []
    sleeps = []

    def open_request(request, *, timeout):
        calls.append((bytes(request.data), timeout))
        raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, None)

    transport = HttpsJsonRpcTransport(
        "https://chain.example",
        timeout_seconds=30,
        opener=open_request,
        clock=lambda: 100.0,
        sleep=sleeps.append,
    )
    with pytest.raises(ValidatorChainSourceV2Error, match="exhausted transient"):
        transport.call(method="chain_getBlockHash", params=[3], request_id=7)
    assert len(calls) == 3
    assert sleeps == [1.0, 3.0]
    assert calls[0][0] == calls[1][0] == calls[2][0]


def test_https_transport_does_not_retry_invalid_or_authenticated_errors():
    responses = [
        _Response(b"not-json"),
        _Response(json.dumps({"jsonrpc": "2.0", "id": 7, "error": {}}).encode()),
        _Response(b"{}", status=401),
    ]
    for response in responses:
        calls = []
        transport = HttpsJsonRpcTransport(
            "https://chain.example",
            timeout_seconds=9,
            opener=lambda *_args, **_kwargs: calls.append(1) or response,
            sleep=lambda _delay: pytest.fail("non-transient responses must not sleep"),
        )
        with pytest.raises(ValidatorChainSourceV2Error):
            transport.call(method="chain_getBlockHash", params=[3], request_id=7)
        assert calls == [1]


def test_https_transport_closes_a_nontransient_http_error_response():
    response = _Response(b"")
    calls = []

    def open_request(request, *, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, response)

    transport = HttpsJsonRpcTransport(
        "https://chain.example",
        timeout_seconds=9,
        opener=open_request,
    )
    with pytest.raises(ValidatorChainSourceV2Error, match="HTTP error"):
        transport.call(method="chain_getBlockHash", params=[3], request_id=7)
    assert len(calls) == 1
    assert response.closed is True


def test_https_transport_does_not_cross_deadline_for_retry_after():
    calls = []
    sleeps = []

    def open_request(request, *, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "rate limited",
            {"Retry-After": "9" * 10_000},
            None,
        )

    transport = HttpsJsonRpcTransport(
        "https://chain.example",
        timeout_seconds=9,
        opener=open_request,
        clock=lambda: 100.0,
        sleep=sleeps.append,
    )
    with pytest.raises(ValidatorChainSourceV2Error, match="deadline"):
        transport.call(method="chain_getBlockHash", params=[3], request_id=7)
    assert calls == [9]
    assert sleeps == []


def test_https_transport_stops_retrying_when_closed():
    calls = []
    transport = None

    def open_request(request, *, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(request.full_url, 502, "bad gateway", {}, None)

    def close_during_backoff(_delay):
        transport.close()

    transport = HttpsJsonRpcTransport(
        "https://chain.example",
        timeout_seconds=9,
        opener=open_request,
        clock=lambda: 100.0,
        sleep=close_during_backoff,
    )
    with pytest.raises(ValidatorChainSourceV2Error, match="closed"):
        transport.call(method="chain_getBlockHash", params=[3], request_id=7)
    assert calls == [9]


def test_python_drand_backend_passes_stateful_sdk_arguments_exactly():
    observed = []

    def generate(*args):
        observed.extend(args)
        return b"commitment", 91

    backend = BittensorDrandBackend(generate)
    assert backend.generate_commit(
        uids=[1, 4],
        weights_u16=[5, 6],
        version_key=7,
        last_epoch_block=8,
        pending_epoch_at=9,
        subnet_epoch_index=10,
        tempo=11,
        blocks_since_last_step=12,
        current_block=13,
        subnet_reveal_period_epochs=14,
        block_time=15.0,
        hotkey_public_key=b"h" * 32,
    ) == (b"commitment", 91)
    assert observed == [
        [1, 4], [5, 6], 7, 8, 9, 10, 11, 12, 13, 14, 15.0, b"h" * 32
    ]


def test_python_drand_backend_rejects_an_installed_package_without_v2_api(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "bittensor_drand", types.ModuleType("bittensor_drand"))
    with pytest.raises(
        LocalWeightSignerError,
        match=(
            "bittensor-drand 2.x stateful API required; "
            "install repository requirements"
        ),
    ):
        BittensorDrandBackend()


def test_builder_uses_public_profile_local_wallet_and_compatible_client_methods():
    from bittensor_wallet import Keypair

    keypair = Keypair.create_from_seed("0x" + ("01" * 32))
    signing_key, signing_key_hash = _arena_key()
    live = _Transport()
    archive = _Transport()
    config = ArenaChainConfig(
        endpoint="wss://entrypoint-finney.opentensor.ai:443",
        netuid=71,
        network_name="finney",
        request_timeout_seconds=4,
    )
    client = build_local_weight_signer(
        keypair,
        config,
        signing_key,
        signing_key_hash,
        load_arena_cutover({}),
        burn_hotkey="5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
        live_transport=live,
        archive_transport=archive,
        drand_backend=object(),
        finalization_sleep=lambda _seconds: None,
    )
    assert client.extrinsic_period == 8

    calls = []
    client._signer.prepare = lambda request: calls.append(("prepare", request)) or {"ok": 1}
    client._signer.confirm = lambda request: calls.append(("confirm", request)) or {"ok": 2}
    client._signer.recover = lambda request: calls.append(("recover", request)) or {"ok": 3}
    client._signer.sign_chain_outcome = lambda request: calls.append(("outcome", request)) or {"ok": 4}
    assert client.prepare_arena_weight_extrinsic_v1({"a": 1}) == {"ok": 1}
    assert client.confirm_arena_weight_extrinsic_v1({"b": 2}) == {"ok": 2}
    assert client.recover_arena_weight_extrinsic_v1({"c": 3}) == {"ok": 3}
    assert client.sign_arena_chain_outcome_v1({"d": 4}) == {"ok": 4}
    assert calls == [
        ("prepare", {"a": 1}),
        ("confirm", {"b": 2}),
        ("recover", {"c": 3}),
        ("outcome", {"d": 4}),
    ]

    client._chain_source.read_finalized_snapshot = lambda **_kwargs: {
        "finalized_block_hash": "2" * 64,
        "header": {"block": 104},
        "metagraph": {"hotkeys": [keypair.ss58_address]},
        "epoch_authority": {"settlement_epoch_id": 32001},
    }
    client._chain_source.read_chain_signing_runtime = lambda **_kwargs: {
        "spec_version": 438,
        "transaction_version": 1,
        "genesis_hash": "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03",
    }
    assert client.readiness(32001) == {
        "schema_version": "leadpoet.arena.local_weight_signer_readiness.v1",
        "network": "finney",
        "netuid": 71,
        "epoch": 32001,
        "validator_hotkey": keypair.ss58_address,
        "validator_uid": 0,
        "finalized_block": 104,
        "finalized_block_hash": "2" * 64,
        "runtime_spec_version": 438,
        "transaction_version": 1,
    }

    client.close()
    client.close()
    assert live.closed == archive.closed == 1
    with pytest.raises(LocalWeightSignerError, match="closed"):
        client.prepare({})


def test_public_profile_selection_is_network_specific_and_config_bound():
    assert load_public_chain_signing_profile("finney")["network"] == "finney"
    assert load_public_chain_signing_profile("test")["network"] == "test"
    with pytest.raises(LocalWeightSignerError, match="no public"):
        load_public_chain_signing_profile("local")

    from bittensor_wallet import Keypair

    signing_key, signing_key_hash = _arena_key()
    wrong_endpoint = ArenaChainConfig(
        endpoint="wss://other.example:443",
        netuid=71,
        network_name="finney",
        request_timeout_seconds=4,
    )
    with pytest.raises(LocalWeightSignerError, match="differs"):
        build_local_weight_signer(
            Keypair.create_from_seed("0x" + ("02" * 32)),
            wrong_endpoint,
            signing_key,
            signing_key_hash,
            load_arena_cutover({}),
            burn_hotkey="5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
        )


def test_builder_rejects_an_unpinned_finney_archive():
    from bittensor_wallet import Keypair

    signing_key, signing_key_hash = _arena_key()
    config = ArenaChainConfig(
        endpoint="wss://entrypoint-finney.opentensor.ai:443",
        netuid=71,
        network_name="finney",
        request_timeout_seconds=4,
    )
    with pytest.raises(LocalWeightSignerError, match="archive chain RPC endpoint"):
        build_local_weight_signer(
            Keypair.create_from_seed("0x" + ("03" * 32)),
            config,
            signing_key,
            signing_key_hash,
            load_arena_cutover({}),
            burn_hotkey="5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
            archive_endpoint="wss://untrusted.example:443",
        )
