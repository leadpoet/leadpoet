from __future__ import annotations

import json

import pytest

from leadpoet_canonical.attested_v2 import sha256_bytes
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_RPC_METHOD,
    ChainSourceV2Error,
    chain_source_policy_hash,
    decode_timelocked_weight_commits,
    decode_selective_metagraph_result,
    encode_selective_metagraph_params,
    json_rpc_request,
    parse_finalized_header,
    parse_json_rpc_response,
    ss58_encode_account_id,
    timelocked_weight_commits_storage_key,
)


OWNER_ACCOUNT = bytes.fromhex(
    "924620afb270acb1ee27bd034aa9e97108ef276da5079db982883cd70294741a"
)
SECOND_ACCOUNT = bytes.fromhex(
    "74adb27b7edd7126a81f5bac79e9bda1a4c8ec94d2c4f2ce795e0c56932a5383"
)


def _selective_fixture() -> bytes:
    # Option<SelectiveMetagraphInfo>::Some, compact netuid 71.
    encoded = bytearray((1, 0x1D, 0x01))
    encoded.extend(b"\x00" * 4)  # fields 1..4 omitted
    encoded.extend(b"\x01" + OWNER_ACCOUNT)  # owner hotkey
    encoded.extend(b"\x00")  # owner coldkey omitted
    encoded.extend(b"\x01" + ((8_597_161 << 2) | 2).to_bytes(4, "little"))
    encoded.extend(b"\x00" * 44)  # fields 8..51 omitted
    encoded.extend(b"\x01\x08" + OWNER_ACCOUNT + SECOND_ACCOUNT)
    encoded.extend(b"\x00" * 21)  # fields 53..73 omitted
    return bytes(encoded)


def test_selective_metagraph_codec_matches_live_sdk_shape():
    assert encode_selective_metagraph_params(netuid=71) == (
        "0x470000100000050007003400"
    )
    decoded = decode_selective_metagraph_result(_selective_fixture())
    assert decoded == {
        "netuid": 71,
        "block": 8_597_161,
        "owner_hotkey": "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
        "hotkeys": [
            "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
            ss58_encode_account_id(SECOND_ACCOUNT),
        ],
    }


def test_selective_metagraph_codec_rejects_unrequested_or_trailing_fields():
    fixture = bytearray(_selective_fixture())
    fixture[3] = 1
    with pytest.raises(ChainSourceV2Error, match="unexpected selective"):
        decode_selective_metagraph_result(bytes(fixture))
    with pytest.raises(ChainSourceV2Error, match="trailing"):
        decode_selective_metagraph_result(_selective_fixture() + b"\x00")


def test_finalized_header_commits_real_chain_root_without_mislabeling_algorithm():
    state_root = "12" * 32
    parsed = parse_finalized_header(
        {
            "number": "0x832729",
            "stateRoot": "0x" + state_root,
            "parentHash": "0x" + "34" * 32,
            "extrinsicsRoot": "0x" + "56" * 32,
        }
    )
    assert parsed["block"] == int("832729", 16)
    assert parsed["state_root"] == state_root
    assert parsed["state_root_commitment"] == sha256_bytes(bytes.fromhex(state_root))


def test_json_rpc_contract_is_exact_and_method_limited():
    request = json_rpc_request(
        "state_call",
        [CHAIN_RPC_METHOD, encode_selective_metagraph_params(netuid=71), "0x" + "ab" * 32],
        3,
    )
    assert json.loads(request) == {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "state_call",
        "params": [
            CHAIN_RPC_METHOD,
            "0x470000100000050007003400",
            "0x" + "ab" * 32,
        ],
    }
    response = json.dumps(
        {"jsonrpc": "2.0", "id": 3, "result": "0x01"},
        separators=(",", ":"),
    ).encode()
    assert parse_json_rpc_response(response, 3) == "0x01"
    with pytest.raises(ChainSourceV2Error, match="outside policy"):
        json_rpc_request("author_submitExtrinsic", [], 4)
    with pytest.raises(ChainSourceV2Error, match="authenticated error"):
        parse_json_rpc_response(
            b'{"jsonrpc":"2.0","id":3,"error":{"code":-1}}', 3
        )


def test_chain_policy_hash_is_stable_and_typed():
    assert chain_source_policy_hash().startswith("sha256:")
    assert len(chain_source_policy_hash()) == 71


def test_timelocked_commit_storage_key_matches_live_finney_metadata():
    assert timelocked_weight_commits_storage_key(netuid=71, mechid=0) == (
        "0x658faa385070e074c85bf6b568cf0555"
        "119e1c428ff5c7c6dac4c888a10d1443"
        "8510facd6ced86974700"
        "bb1bdbcacd6ac9340000000000000000"
    )


def test_timelocked_commit_storage_decoder_is_exact():
    commitment = b"measured-commitment"
    encoded = b"".join(
        (
            b"\x04",  # Vec length 1
            OWNER_ACCOUNT,
            (123).to_bytes(8, "little"),
            bytes((len(commitment) << 2,)),
            commitment,
            (998877).to_bytes(8, "little"),
        )
    )
    assert decode_timelocked_weight_commits("0x" + encoded.hex()) == [
        {
            "hotkey_public_key": OWNER_ACCOUNT.hex(),
            "submitted_at": 123,
            "commitment_hex": commitment.hex(),
            "reveal_round": 998877,
        }
    ]
    with pytest.raises(ChainSourceV2Error, match="trailing"):
        decode_timelocked_weight_commits("0x" + (encoded + b"\x00").hex())
