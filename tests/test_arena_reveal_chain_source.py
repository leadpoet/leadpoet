"""Direct historical-chain tests for the protected Arena reveal proof."""

from __future__ import annotations

import hashlib

import pytest

from leadpoet_canonical.chain_source_v2 import (
    last_update_storage_key,
    system_event_count_storage_key,
    system_events_storage_key,
    timelocked_weight_commits_storage_key,
    weights_storage_key,
    ss58_encode_account_id,
)
from validator_tee.enclave import chain_source_v2 as module
from validator_tee.enclave.chain_source_v2 import (
    ValidatorChainSourceV2,
    ValidatorChainSourceV2Error,
)

NETUID = 71
EPOCH = 101
VALIDATOR_UID = 1
VALIDATOR_ACCOUNT = bytes.fromhex("31" * 32)
RECIPIENT_A_ACCOUNT = bytes.fromhex("41" * 32)
RECIPIENT_B_ACCOUNT = bytes.fromhex("42" * 32)
RECIPIENT_A = ss58_encode_account_id(RECIPIENT_A_ACCOUNT)
RECIPIENT_B = ss58_encode_account_id(RECIPIENT_B_ACCOUNT)
VALIDATOR_HOTKEY = ss58_encode_account_id(VALIDATOR_ACCOUNT)
COMMITMENT = b"arena-commitment"
ROUND = 998_877
WEIGHTS = [(0, 12_345), (2, 54_321)]


def _compact(value: int) -> bytes:
    assert 0 <= value < (1 << 14)
    if value < 64:
        return bytes((value << 2,))
    return ((value << 2) | 1).to_bytes(2, "little")


def _commits(present: bool) -> str:
    if not present:
        return "0x00"
    raw = bytearray(b"\x04")
    raw.extend(VALIDATOR_ACCOUNT)
    raw.extend((900).to_bytes(8, "little"))
    raw.extend(_compact(len(COMMITMENT)))
    raw.extend(COMMITMENT)
    raw.extend(ROUND.to_bytes(8, "little"))
    return "0x" + raw.hex()


def _weights() -> str:
    raw = bytearray(_compact(len(WEIGHTS)))
    for uid, weight in WEIGHTS:
        raw.extend(uid.to_bytes(2, "little"))
        raw.extend(weight.to_bytes(2, "little"))
    return "0x" + raw.hex()


def _last_update(block: int) -> str:
    values = [0, block, 0]
    return "0x" + (_compact(len(values)) + b"".join(
        value.to_bytes(8, "little") for value in values
    )).hex()


def _metagraph(block: int, *, recycle_uid: bool) -> str:
    # Production SelectiveMetagraph SCALE shape, reduced to three hotkeys.
    accounts = [RECIPIENT_A_ACCOUNT, VALIDATOR_ACCOUNT, RECIPIENT_B_ACCOUNT]
    if recycle_uid:
        accounts[2] = bytes.fromhex("43" * 32)
    raw = bytearray((1,))
    raw.extend(_compact(NETUID))
    raw.extend(b"\x00" * 4)
    raw.extend(b"\x01" + bytes.fromhex("44" * 32))
    raw.extend(b"\x00")
    raw.extend(b"\x01" + _compact(block))
    raw.extend(b"\x00" * 44)
    raw.extend(b"\x01" + _compact(len(accounts)))
    raw.extend(b"".join(accounts))
    raw.extend(b"\x00" * 24)
    return "0x" + raw.hex()


class ArchiveFixture:
    def __init__(self, *, reveal_block: int = 105, event_success: bool = True,
                 recycle_uid: bool = False,
                 last_update_block: int = 100) -> None:
        self.reveal_block = reveal_block
        self.event_success = event_success
        self.recycle_uid = recycle_uid
        self.last_update_block = last_update_block
        self.hashes = {
            block: hashlib.sha256(("block:%d" % block).encode()).hexdigest()
            for block in range(90, 131)
        }

    def block_for_hash(self, value: str) -> int:
        digest = value[2:] if value.startswith("0x") else value
        return next(block for block, observed in self.hashes.items() if observed == digest)

    def result(self, method, params):
        if method == "chain_getBlockHash":
            return "0x" + self.hashes[int(params[0])]
        if method == "state_getRuntimeVersion":
            return {"specVersion": 452, "transactionVersion": 1}
        if method == "state_getMetadata":
            return "0x0102"
        if method == "state_getStorageHash":
            return "0x" + "55" * 32
        if method == "state_call":
            block = self.block_for_hash(params[2])
            return _metagraph(block, recycle_uid=self.recycle_uid)
        assert method == "state_getStorage"
        key, digest = params
        block = self.block_for_hash(digest)
        if key == timelocked_weight_commits_storage_key(
                netuid=NETUID, subnet_epoch_index=EPOCH):
            return _commits(block < self.reveal_block)
        if key == weights_storage_key(netuid=NETUID, validator_uid=VALIDATOR_UID):
            return _weights()
        if key == last_update_storage_key(netuid=NETUID):
            return _last_update(self.last_update_block)
        if key == system_events_storage_key():
            return "0x01" if self.event_success else "0x00"
        if key == system_event_count_storage_key():
            return "0x01000000" if self.event_success else "0x00000000"
        raise AssertionError("unexpected storage key %s" % key)


def _source(monkeypatch, fixture: ArchiveFixture, *, finalized_head: int = 130):
    source = ValidatorChainSourceV2(
        rpc_call=lambda **_: None, archive_rpc_call=lambda **_: None,
        epoch_authority_supplier=lambda: None,
    )
    source._call = lambda **kwargs: {"result": (
        "0x" + fixture.hashes[finalized_head]
        if kwargs["method"] == "chain_getFinalizedHead"
        else {"number": hex(finalized_head), "stateRoot": "0x" + "11" * 32,
              "extrinsicsRoot": "0x" + "22" * 32,
              "parentHash": "0x" + fixture.hashes[finalized_head - 1],
              "digest": {"logs": []}}
    )}
    source._archive_call = lambda **kwargs: {
        "result": fixture.result(kwargs["method"], kwargs["params"]),
        "attempts": [{}], "artifacts": [],
    }
    # Metadata/profile decoding has its own production-fixture suite. These
    # tests retain the event-presence boundary while focusing on archive search.
    monkeypatch.setattr(module, "decode_runtime_metadata_commitment", lambda _: {})
    monkeypatch.setattr(module, "load_subtensor_events_profile_v2", lambda: {})
    monkeypatch.setattr(module, "validate_subtensor_events_profile_v2", lambda *_, **__: {})

    def prove(events, **_kwargs):
        if events != b"\x01":
            raise ValidatorChainSourceV2Error("successful initialization reveal event is absent")
        return {"event": "TimelockedWeightsRevealed"}

    monkeypatch.setattr(module, "prove_timelocked_weights_reveal_v2", prove)
    return source


def _prove(source):
    return source.prove_timelocked_reveal_transition(
        netuid=NETUID, validator_hotkey=VALIDATOR_HOTKEY,
        hotkey_public_key_hex=VALIDATOR_ACCOUNT.hex(),
        subnet_epoch_index=EPOCH, commitment_hex=COMMITMENT.hex(),
        reveal_round=ROUND, inclusion_block=100, reveal_deadline_block=110,
        expected_weights=WEIGHTS,
        expected_recipient_uid_hotkeys=[
            {"uid": 0, "hotkey": RECIPIENT_A},
            {"uid": 2, "hotkey": RECIPIENT_B},
        ],
        chain_profile={"genesis_hash": "11" * 32},
    )


def test_historical_reveal_succeeds_after_latest_head_passed_deadline(monkeypatch):
    result = _prove(_source(monkeypatch, ArchiveFixture(), finalized_head=130))
    assert result["reveal_block"] == 105
    assert result["weights"] == WEIGHTS
    assert result["event_witness"] == {"event": "TimelockedWeightsRevealed"}


def test_removed_commit_without_successful_reveal_event_fails(monkeypatch):
    source = _source(monkeypatch, ArchiveFixture(event_success=False))
    with pytest.raises(ValidatorChainSourceV2Error, match="reveal event"):
        _prove(source)


def test_rewarded_uid_recycling_at_reveal_fails(monkeypatch):
    source = _source(monkeypatch, ArchiveFixture(recycle_uid=True))
    with pytest.raises(ValidatorChainSourceV2Error, match="UID ownership changed"):
        _prove(source)


def test_reveal_rejects_last_update_changed_by_another_commit(monkeypatch):
    source = _source(
        monkeypatch, ArchiveFixture(last_update_block=104)
    )
    with pytest.raises(
        ValidatorChainSourceV2Error,
        match="differs from the proved Arena commitment",
    ):
        _prove(source)
