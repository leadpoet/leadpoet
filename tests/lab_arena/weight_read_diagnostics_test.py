import json
import urllib.error

import pytest

from lab_arena.chain_source import ValidatorChainSourceV2, ValidatorChainSourceV2Error
from lab_arena.local_weight_signer import HttpsJsonRpcTransport
from lab_arena.validator_logging import ValidatorOperationalLogger
from lab_arena.weight_read_diagnostics import chain_error_fields, chain_read
from lab_arena.weight_signer import ArenaWeightSignerError
from lab_arena import validator_events
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_RPC_RETRY_BACKOFF_SECONDS, ChainSourceV2Error,
)
from leadpoet_canonical import chain_source_v2 as canonical
from leadpoet_canonical.hotkey_authority_v2 import signed_extrinsic_hash_v2


HEAD = "0x" + "11" * 32
HEADER = {"number": "0x1", "stateRoot": HEAD,
          "parentHash": HEAD, "extrinsicsRoot": HEAD}


class Response:
    headers = {}

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return 200

    def read(self, limit):
        return self.body[:limit]


def source(rpc):
    return ValidatorChainSourceV2(rpc_call=rpc, archive_rpc_call=rpc,
                                 epoch_authority_supplier=lambda: None,
                                 finalization_sleep=lambda _: None)


def logged(error, *, reason=None):
    log = ValidatorOperationalLogger("https://gateway.invalid", keypair=None,
                                     network="finney", netuid=71)
    log.error("weight_recovery" if reason else "weights", error, reason=reason)
    queued = list(log._queue)[0]
    assert validator_events.validate_event(queued) == queued
    return queued["content"]


@pytest.mark.parametrize("body,code", [
    (b"not JSON: password=PRIVATE", "rpc_invalid_json"),
    (b'{"jsonrpc":"2.0","id":1,"result":null,"secret":"PRIVATE"}', "rpc_invalid_fields"),
    (b'{"jsonrpc":"2.0","id":999,"result":null}', "rpc_invalid_binding"),
    (b'{"jsonrpc":"2.0","id":1,"error":{"message":"PRIVATE"}}', "rpc_error_response"),
    (b'{"jsonrpc":"2.0","id":1}', "rpc_result_missing"),
])
def test_actual_transport_parser_error_survives_source_and_signer_wrapping(body, code):
    calls = []

    def opener(request, **_kwargs):
        calls.append(request)
        return Response(body)

    transport = HttpsJsonRpcTransport("https://rpc.invalid", timeout_seconds=30,
                                     opener=opener)
    with pytest.raises(ValidatorChainSourceV2Error) as caught:
        source(transport.call).read_finalized_head()
    original = caught.value
    assert type(original.__cause__) is ChainSourceV2Error
    try:
        raise ArenaWeightSignerError("PRIVATE wrapper") from original
    except ArenaWeightSignerError as wrapped:
        content = logged(wrapped)
        assert wrapped.__cause__ is original
    assert content["reason"] == code
    assert content["operation"] == "finalized_head:live_chain_getFinalizedHead:" + code
    assert len(calls) == 1  # Malformed/error envelopes still do not gain retries.
    assert "PRIVATE" not in str(content) and "rpc.invalid" not in str(content)


@pytest.mark.parametrize("stage", ["snapshot", "signing_runtime", "confirm_inclusion", "confirm_reveal"])
def test_actual_prepare_and_confirm_read_stages_keep_direct_decoder_failure(stage):
    def rpc(*, method, **_kwargs):
        return HEAD if method == "chain_getFinalizedHead" else None

    chain = source(rpc)
    if stage == "snapshot":
        invoke = lambda: chain.read_finalized_snapshot(netuid=71, epoch_id=123)
    elif stage == "signing_runtime":
        invoke = lambda: chain.read_chain_signing_runtime(runtime_block_hash=HEAD, max_block_drift=16)
    elif stage == "confirm_inclusion":
        digest = signed_extrinsic_hash_v2(b"\x01\x02")
        invoke = lambda: chain.find_finalized_extrinsic_inclusion(
            expected_extrinsics={digest: "0102"}, expected_commitments={digest: {}},
            minimum_block=0, maximum_block=1)
    else:
        invoke = lambda: chain.prove_timelocked_reveal_transition(
            netuid=71, validator_hotkey="unused", hotkey_public_key_hex="11" * 32,
            subnet_epoch_index=1, commitment_hex="11", reveal_round=1,
            inclusion_block=0, reveal_deadline_block=2, expected_weights=[],
            expected_recipient_uid_hotkeys=[], chain_profile={})
    with pytest.raises(ChainSourceV2Error) as caught:
        invoke()
    assert chain_error_fields(caught.value) == {
        "reason": "header_invalid", "operation": stage + ":header_invalid"}
    content = logged(caught.value, reason="prior_epoch_25813")
    assert content["reason"] == "prior_epoch_25813"
    assert content["operation"] == stage + ":header_invalid"


def test_provider_http_error_diagnostic_keeps_same_exception_and_no_new_retry():
    calls = []

    def opener(request, **_kwargs):
        calls.append(request)
        raise urllib.error.HTTPError("https://user:PRIVATE@rpc.invalid", 403,
                                     "PRIVATE", {}, None)

    transport = HttpsJsonRpcTransport("https://rpc.invalid", timeout_seconds=30,
                                     opener=opener)
    with pytest.raises(ValidatorChainSourceV2Error) as caught:
        source(transport.call).read_finalized_head()
    assert caught.value.__cause__.code == 403
    assert logged(caught.value)["reason"] == "rpc_http_error"
    assert logged(caught.value)["operation"] == (
        "finalized_head:live_chain_getFinalizedHead:rpc_http_error:http_403")
    assert len(calls) == 1
    assert "PRIVATE" not in str(logged(caught.value))


def test_successful_provider_retry_and_normal_head_are_unchanged():
    calls, sleeps, clock = [], [], [0.0]

    def opener(request, **_kwargs):
        body = json.loads(request.data)
        calls.append(body["method"])
        if len(calls) == 1:
            raise urllib.error.HTTPError("https://rpc.invalid", 503, "PRIVATE", {}, None)
        result = HEAD if body["method"] == "chain_getFinalizedHead" else HEADER
        return Response(json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode())

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    transport = HttpsJsonRpcTransport("https://rpc.invalid", timeout_seconds=30,
                                     opener=opener, clock=lambda: clock[0], sleep=sleep)
    assert source(transport.call).read_finalized_head() == {"block": 1, "block_hash": HEAD[2:]}
    assert calls == ["chain_getFinalizedHead", "chain_getFinalizedHead", "chain_getHeader"]
    assert sleeps == [CHAIN_RPC_RETRY_BACKOFF_SECONDS[0]]


def test_canonical_hash_prepare_read_retains_stage_for_unknown_hash_failure():
    with pytest.raises(ChainSourceV2Error) as caught:
        source(lambda **_kwargs: None).read_canonical_block_hash(block=1)
    assert chain_error_fields(caught.value) == {"operation": "canonical_block_hash"}


def test_actual_archive_failure_identifies_reveal_stage_and_rpc():
    def live(*, method, **_kwargs):
        return HEAD if method == "chain_getFinalizedHead" else HEADER

    transport = HttpsJsonRpcTransport(
        "https://archive.invalid", timeout_seconds=30,
        opener=lambda *_args, **_kwargs: Response(b"PRIVATE invalid JSON"))
    chain = ValidatorChainSourceV2(rpc_call=live, archive_rpc_call=transport.call,
                                   epoch_authority_supplier=lambda: None)
    with pytest.raises(ValidatorChainSourceV2Error) as caught:
        chain.prove_timelocked_reveal_transition(
            netuid=71, validator_hotkey="unused", hotkey_public_key_hex="11" * 32,
            subnet_epoch_index=1, commitment_hex="11", reveal_round=1,
            inclusion_block=0, reveal_deadline_block=2, expected_weights=[],
            expected_recipient_uid_hotkeys=[], chain_profile={})
    assert chain_error_fields(caught.value) == {
        "reason": "rpc_invalid_json",
        "operation": "confirm_reveal:archive_chain_getBlockHash:rpc_invalid_json"}


@pytest.mark.parametrize("parser,value,code", [
    (canonical.parse_finalized_header, {}, "header_number_invalid"),
    (canonical.parse_finalized_header, {"number": "-1"}, "header_number_negative"),
    (canonical.decode_runtime_metadata_commitment, "PRIVATE", "metadata_invalid"),
    (canonical.decode_runtime_metadata_commitment, "0xgg", "metadata_invalid_hex"),
    (canonical.decode_runtime_metadata_commitment, "0x00", "metadata_exceeds_policy"),
    (canonical.decode_last_update_storage, "PRIVATE", "last_update_invalid"),
    (canonical.decode_last_update_storage, "0xgg", "last_update_invalid_hex"),
    (canonical.decode_last_update_storage, "0x", "last_update_empty"),
    (canonical.decode_last_update_storage, "0x0540", "last_update_exceeds_policy"),
    (canonical.decode_last_update_storage, "0x04", "last_update_truncated"),
    (canonical.decode_last_update_storage, "0x0000", "last_update_trailing_bytes"),
    (canonical.decode_weights_storage, "PRIVATE", "weights_invalid"),
    (canonical.decode_weights_storage, "0xgg", "weights_invalid_hex"),
    (canonical.decode_weights_storage, "0x", "weights_empty"),
    (canonical.decode_weights_storage, "0x0540", "weights_exceed_policy"),
    (canonical.decode_weights_storage, "0x04", "weights_truncated"),
    (canonical.decode_weights_storage, "0x0000", "weights_trailing_bytes"),
    (canonical.decode_weights_storage, "0x080000010000000100", "weights_duplicate_uids"),
    (canonical.decode_timelocked_weight_commits, "PRIVATE", "commits_invalid"),
    (canonical.decode_timelocked_weight_commits, "0xgg", "commits_invalid_hex"),
    (canonical.decode_timelocked_weight_commits, "0x", "commits_empty"),
    (canonical.decode_timelocked_weight_commits, "0x0540", "commits_exceed_policy"),
    (canonical.decode_timelocked_weight_commits, "0x0000", "commits_trailing_bytes"),
    (canonical.decode_timelocked_weight_commits, "0x04", "commit_account_truncated"),
    (canonical.decode_timelocked_weight_commits, "0x04" + "00" * 32, "commit_block_truncated"),
    (canonical.decode_timelocked_weight_commits, "0x04" + "00" * 41, "commit_size_invalid"),
    (canonical.decode_timelocked_weight_commits, "0x04" + "00" * 40 + "04", "commit_truncated"),
    (canonical.decode_timelocked_weight_commits, "0x04" + "00" * 40 + "0400", "commit_round_truncated"),
    (canonical.decode_selective_metagraph_result, None, "metagraph_invalid"),
    (canonical.decode_selective_metagraph_result, "gg", "metagraph_invalid_hex"),
    (canonical.decode_selective_metagraph_result, b"", "metagraph_absent"),
    (canonical.decode_selective_metagraph_result, b"\x01\x00" + b"\x00" * 4, "metagraph_owner_absent"),
    (canonical.decode_selective_metagraph_result, b"\x01\x00" + b"\x00" * 4 + b"\x01", "metagraph_owner_truncated"),
])
def test_actual_canonical_decoder_failures_use_exact_finite_codes(parser, value, code):
    with pytest.raises(ChainSourceV2Error) as caught:
        parser(value)
    assert chain_error_fields(caught.value) == {"reason": code}
    assert "PRIVATE" not in str(logged(caught.value))


@pytest.mark.parametrize("value,code", [
    (None, "block_fields_invalid"),
    ({}, "block_body_invalid"),
    ({"block": {"header": HEADER, "extrinsics": []}}, "block_number_mismatch"),
    ({"block": {"header": HEADER, "extrinsics": None}}, "block_extrinsics_invalid"),
    ({"block": {"header": HEADER, "extrinsics": ["PRIVATE"]}}, "block_extrinsic_invalid_hex"),
])
def test_actual_block_decoder_failure(value, code):
    expected_block = 2 if code == "block_number_mismatch" else 1
    with pytest.raises(ChainSourceV2Error) as caught:
        canonical.parse_finalized_block_extrinsics(value, expected_block=expected_block)
    assert chain_error_fields(caught.value) == {"reason": code}


def test_unknown_or_spoofed_exception_values_are_never_projected():
    class ForeignError(RuntimeError):
        def __str__(self):
            raise AssertionError("must not inspect arbitrary exception text")

    foreign = ForeignError("chain RPC result is missing")
    foreign._arena_weight_read_operation = "PRIVATE"
    foreign._arena_weight_rpc_operation = "live_chain_getHeader"
    assert chain_error_fields(foreign) == {}
    assert "operation" not in logged(foreign)

    @chain_read("confirm_inclusion")
    def fail():
        raise ChainSourceV2Error("chain RPC result is missing PRIVATE")

    with pytest.raises(ChainSourceV2Error) as caught:
        fail()
    assert chain_error_fields(caught.value) == {"operation": "confirm_inclusion"}
    assert "reason" not in logged(caught.value)
    assert "PRIVATE" not in str(logged(caught.value))

    class PrivateValue:
        def __str__(self):
            raise AssertionError("must not stringify unknown error arguments")

    assert chain_error_fields(ChainSourceV2Error(PrivateValue())) == {}


def test_suppressed_chain_context_stays_suppressed():
    outer = ArenaWeightSignerError("PRIVATE")
    outer.__context__ = ChainSourceV2Error("chain RPC result is missing")
    outer.__suppress_context__ = True
    assert chain_error_fields(outer) == {}
    assert "reason" not in logged(outer) and "operation" not in logged(outer)


def test_untrusted_subclass_and_http_status_attributes_are_ignored():
    class ForeignChainError(ChainSourceV2Error):
        pass

    foreign = ForeignChainError("chain RPC result is missing")
    foreign._arena_weight_read_operation = "snapshot"
    foreign._arena_weight_rpc_operation = "live_chain_getHeader"
    assert chain_error_fields(foreign) == {}

    class ForeignHttpError(urllib.error.HTTPError):
        pass

    error = ChainSourceV2Error("chain RPC result is missing")
    error._arena_weight_read_operation = "snapshot"
    for cause in (ForeignHttpError("PRIVATE", 429, "PRIVATE", {}, None),
                  urllib.error.HTTPError("PRIVATE", "PRIVATE", "PRIVATE", {}, None)):
        error.__cause__ = cause
        assert chain_error_fields(error) == {
            "reason": "rpc_result_missing", "operation": "snapshot:rpc_result_missing"}


def test_broken_diagnostic_helper_never_replaces_chain_error(monkeypatch):
    import lab_arena.weight_read_diagnostics as diagnostics

    def broken(_exc):
        raise RuntimeError("PRIVATE diagnostic failure")

    monkeypatch.setattr(diagnostics, "_trusted_errors", broken)
    original = ChainSourceV2Error("chain RPC result is missing")

    @chain_read("snapshot")
    def fail():
        diagnostics.note_rpc_failure(original, "live", "chain_getHeader")
        raise original

    with pytest.raises(ChainSourceV2Error) as caught:
        fail()
    assert caught.value is original and caught.value.__cause__ is None
    assert chain_error_fields(original) == {}
    assert "reason" not in logged(original) and "operation" not in logged(original)


def test_injected_annotation_failure_keeps_original_source_error(monkeypatch):
    import lab_arena.chain_source as chain_module
    import lab_arena.weight_read_diagnostics as diagnostics

    def broken(*_args, **_kwargs):
        raise RuntimeError("PRIVATE annotation failure")

    monkeypatch.setattr(diagnostics, "_set_context", broken)
    monkeypatch.setattr(chain_module, "note_rpc_failure", broken)
    original = ChainSourceV2Error("chain RPC result is missing")

    def fail(**_kwargs):
        raise original

    with pytest.raises(ChainSourceV2Error) as caught:
        source(fail).read_finalized_head()
    assert caught.value is original and caught.value.__cause__ is None


def test_all_finite_operations_fit_existing_event_contract():
    from lab_arena.weight_read_diagnostics import _OPERATIONS, _RPC_OPERATIONS, _REASONS

    longest = max(len(":".join((stage, rpc, reason, "http_599")))
                  for stage in _OPERATIONS for rpc in _RPC_OPERATIONS
                  for reason in _REASONS.values())
    assert longest <= 128
    original = ChainSourceV2Error("host chain RPC exhausted transient HTTP retries")
    original._arena_weight_read_operation = "confirm_inclusion"
    original._arena_weight_rpc_operation = "archive_chain_getFinalizedHead"
    original.__cause__ = urllib.error.HTTPError("PRIVATE", 599, "PRIVATE", {}, None)
    assert logged(original, reason="prior_epoch_25813")["operation"].endswith(":http_599")


def test_annotation_keeps_original_exception_and_bounds_cyclic_causes():
    original = ChainSourceV2Error("weight storage value is empty")
    original.__cause__ = original

    @chain_read("confirm_reveal")
    def fail():
        raise original

    with pytest.raises(ChainSourceV2Error) as caught:
        fail()
    assert caught.value is original and original.__cause__ is original
    assert chain_error_fields(original) == {
        "reason": "weights_empty", "operation": "confirm_reveal:weights_empty"}
