import copy
import hashlib
import json
from pathlib import Path

import pytest

from leadpoet_canonical.hotkey_authority_v2 import (
    APPLICATION_SIGNATURE_SCHEMA_VERSION,
    CHAIN_SIGNING_PROFILE_SCHEMA_VERSION,
    HotkeyAuthorityV2Error,
    build_application_signature_request_v2,
    build_serve_axon_extrinsic_authorization_v2,
    build_weight_inputs_request_v2,
    build_weight_extrinsic_authorization_v2,
    classify_application_message_v2,
    compact_scale_uint,
    encode_commit_timelocked_call,
    encode_mortal_era,
    encode_serve_axon_call,
    encode_weight_signature_payload,
    validate_application_signature_request_v2,
    validate_chain_signing_profile,
    validate_serve_axon_extrinsic_authorization_v2,
    validate_weight_extrinsic_authorization_v2,
    validate_weight_inputs_request_v2,
    weight_inputs_request_message_v2,
)


HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
TARGET = "5GcFM97at7gaatFieL1qBHXs6fCD8Xqui3nwmdaZUaUoYAAE"


def _profile():
    return {
        "schema_version": CHAIN_SIGNING_PROFILE_SCHEMA_VERSION,
        "network": "finney",
        "chain_endpoint": "wss://entrypoint-finney.opentensor.ai:443",
        "genesis_hash": "0" * 64,
        "spec_version": 424,
        "transaction_version": 1,
        "version_key": 9012000,
        "commit_call_index": "0776",
        "serve_axon_call_index": "0704",
        "commit_reveal_version": 4,
        "mechid": 0,
        "tempo": 360,
        "subnet_reveal_period_epochs": 1,
        "block_time_millis": 12000,
        "max_snapshot_block_drift": 64,
        "extrinsic_period": 8,
        "signed_extensions": [
            "CheckMortality",
            "CheckNonce",
            "ChargeTransactionPayment",
            "CheckMetadataHash",
            "CheckSpecVersion",
            "CheckTxVersion",
            "CheckGenesis",
            "CheckMortalityAdditionalSigned",
            "CheckMetadataHashAdditionalSigned",
        ],
    }


def test_scale_encoding_matches_observed_finney_call_shape():
    assert compact_scale_uint(3).hex() == "0c"
    assert compact_scale_uint(64).hex() == "0101"
    call = encode_commit_timelocked_call(
        profile=_profile(),
        netuid=71,
        commitment=b"abc",
        reveal_round=123,
    )
    assert call.hex() == "07764700000c6162637b000000000000000400"


def test_serve_axon_encoding_matches_finalized_finney_metadata():
    call = encode_serve_axon_call(
        profile=_profile(),
        netuid=71,
        version=9012000,
        ip=2130706433,
        port=8093,
        ip_type=4,
        protocol=4,
        placeholder1=0,
        placeholder2=0,
    )
    assert call.hex() == (
        "07044700208389000100007f0000000000000000000000009d1f04040000"
    )


def test_checked_in_finney_profile_matches_observed_runtime_payload_fixture():
    root = Path(__file__).resolve().parents[1]
    profile = validate_chain_signing_profile(
        json.loads(
            (root / "validator_tee/enclave/chain_signing_profile_v2.json").read_text()
        )
    )
    call = encode_commit_timelocked_call(
        profile=profile,
        netuid=71,
        commitment=b"abc",
        reveal_round=123,
    )
    preimage, signed = encode_weight_signature_payload(
        profile=profile,
        call_bytes=call,
        era_current=8596708,
        nonce=5,
        block_hash="a23c4fd9e3650f154d48200853e6c44271dd325d9363b7c5aefa582c7a708b23",
    )
    assert preimage.hex() == (
        "07764700000c6162637b0000000000000004004200140000a801000001000000"
        "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
        "a23c4fd9e3650f154d48200853e6c44271dd325d9363b7c5aefa582c7a708b2300"
    )
    assert signed == preimage


def test_mortal_era_encoding_is_stable_for_period_eight():
    assert encode_mortal_era(period=8, current=8).hex() == "0200"
    assert encode_mortal_era(period=8, current=9).hex() == "1200"
    assert encode_mortal_era(period=8, current=15).hex() == "7200"


def test_weight_signature_payload_uses_blake2b_only_above_256_bytes():
    small_call = encode_commit_timelocked_call(
        profile=_profile(),
        netuid=71,
        commitment=b"abc",
        reveal_round=123,
    )
    small_preimage, small_signed = encode_weight_signature_payload(
        profile=_profile(),
        call_bytes=small_call,
        era_current=100,
        nonce=3,
        block_hash="1" * 64,
    )
    assert len(small_preimage) < 256
    assert small_signed == small_preimage

    large_call = encode_commit_timelocked_call(
        profile=_profile(),
        netuid=71,
        commitment=b"x" * 512,
        reveal_round=123,
    )
    large_preimage, large_signed = encode_weight_signature_payload(
        profile=_profile(),
        call_bytes=large_call,
        era_current=100,
        nonce=3,
        block_hash="1" * 64,
    )
    assert len(large_preimage) > 256
    assert large_signed == hashlib.blake2b(
        large_preimage, digest_size=32
    ).digest()


def _authorization():
    return build_weight_extrinsic_authorization_v2(
        profile=_profile(),
        validator_hotkey=HOTKEY,
        hotkey_public_key_hex="2" * 64,
        epoch_id=23860,
        netuid=71,
        weight_receipt_hash="sha256:" + "3" * 64,
        weight_submission_event_hash="sha256:" + "4" * 64,
        weights_hash="5" * 64,
        sparse_uids=[0, 14, 213],
        sparse_weights_u16=[65535, 3210, 2600],
        commitment=b"encrypted-commitment" * 20,
        reveal_round=987654,
        era_current=8596708,
        nonce=12,
        block_hash="6" * 64,
    )


def test_weight_extrinsic_authorization_binds_exact_call_and_payload():
    value = _authorization()
    assert value["schema_version"] == "leadpoet.weight_extrinsic_authorization.v2"
    assert validate_weight_extrinsic_authorization_v2(
        value, profile=_profile()
    ) == value
    assert value["call_data_hex"].startswith("0776470000")
    assert value["signed_message_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    "field,replacement,error",
    [
        ("netuid", 72, "not canonical"),
        ("reveal_round", 987655, "not canonical"),
        ("nonce", 13, "not canonical"),
        ("signed_message_hex", "00" * 32, "not canonical"),
        ("call_data_hex", "00", "not canonical"),
    ],
)
def test_weight_extrinsic_authorization_rejects_tampering(field, replacement, error):
    value = _authorization()
    value[field] = replacement
    with pytest.raises(HotkeyAuthorityV2Error, match=error):
        validate_weight_extrinsic_authorization_v2(value, profile=_profile())


def test_serve_axon_authorization_binds_only_the_measured_call():
    value = build_serve_axon_extrinsic_authorization_v2(
        profile=_profile(),
        validator_hotkey=HOTKEY,
        hotkey_public_key_hex="2" * 64,
        netuid=71,
        version=9012000,
        ip=2130706433,
        port=8093,
        ip_type=4,
        protocol=4,
        placeholder1=0,
        placeholder2=0,
        era_current=8597910,
        nonce=7,
        block_hash="6" * 64,
    )
    assert value["call_data_hex"].startswith("07044700")
    assert validate_serve_axon_extrinsic_authorization_v2(
        value, profile=_profile()
    ) == value
    tampered = dict(value)
    tampered["port"] = 8094
    with pytest.raises(HotkeyAuthorityV2Error, match="not canonical"):
        validate_serve_axon_extrinsic_authorization_v2(
            tampered, profile=_profile()
        )


def test_chain_profile_rejects_plaintext_endpoint_and_extension_drift():
    profile = _profile()
    profile["chain_endpoint"] = "ws://entrypoint-finney.opentensor.ai:80"
    with pytest.raises(HotkeyAuthorityV2Error, match="endpoint"):
        validate_chain_signing_profile(profile)
    profile = _profile()
    profile["signed_extensions"] = profile["signed_extensions"][:-1]
    with pytest.raises(HotkeyAuthorityV2Error, match="extension"):
        validate_chain_signing_profile(profile)


@pytest.mark.parametrize(
    "message,purpose",
    [
        (
            "GET_EPOCH_LEADS:23860:%s" % HOTKEY,
            "validator.gateway_epoch_leads.v2",
        ),
        (
            "VALIDATION_RESULT_BATCH:%s:550e8400-e29b-41d4-a716-446655440000:"
            "2026-07-10T20:00:00+00:00:%s:build-1" % (HOTKEY, "a" * 64),
            "validator.gateway_validation_batch.v2",
        ),
        (
            "FULFILLMENT_SCORING:%s::550e8400-e29b-41d4-a716-446655440000:1234567890"
            % HOTKEY,
            "validator.fulfillment_scoring_fetch.v2",
        ),
        (
            "FULFILLMENT_SCORE:%s:req-1:550e8400-e29b-41d4-a716-446655440000:1234567890"
            % HOTKEY,
            "validator.fulfillment_score_submit.v2",
        ),
        ("1234567890", "validator.qualification_registration.v2"),
        (
            "1234567890"
            + json.dumps(
                {
                    "business_desc": "B2B software",
                    "num_leads": 10,
                    "target_uid": 7,
                },
                sort_keys=True,
                default=str,
            ),
            "validator.curation_request.v2",
        ),
        (
            "1234567890" + json.dumps({}, sort_keys=True, default=str),
            "validator.curation_result_fetch.v2",
        ),
        (
            "1234567890123456789.%s.%s.550e8400-e29b-41d4-a716-446655440000.%s"
            % (HOTKEY, TARGET, "b" * 64),
            "validator.bittensor_dendrite.v2",
        ),
        (
            "1234567890123456789.%s.%s.550e8400-e29b-41d4-a716-446655440000"
            % (TARGET, HOTKEY),
            "validator.bittensor_axon.v2",
        ),
    ],
)
def test_application_message_domains_are_explicit(message, purpose):
    assert (
        classify_application_message_v2(
            message.encode("utf-8"), validator_hotkey=HOTKEY
        )
        == purpose
    )


def test_binding_requires_full_commit_and_application_request_round_trips():
    message = (
        "LEADPOET_VALIDATOR_BINDING|netuid=71|"
        "chain=wss://entrypoint-finney.opentensor.ai:443|"
        "enclave_pubkey=%s|validator_code_hash=sha256:%s|version=%s"
        % ("1" * 64, "2" * 64, "3" * 40)
    ).encode("utf-8")
    request = build_application_signature_request_v2(
        message=message,
        validator_hotkey=HOTKEY,
        boot_identity_hash="sha256:" + "4" * 64,
    )
    assert request["schema_version"] == APPLICATION_SIGNATURE_SCHEMA_VERSION
    assert validate_application_signature_request_v2(
        request, validator_hotkey=HOTKEY
    ) == request

    short = message.replace(("3" * 40).encode(), ("3" * 8).encode())
    with pytest.raises(HotkeyAuthorityV2Error, match="full commit"):
        classify_application_message_v2(short, validator_hotkey=HOTKEY)


def test_weight_input_request_is_canonical_and_has_a_closed_signing_domain():
    request = build_weight_inputs_request_v2(
        validator_hotkey=HOTKEY,
        netuid=71,
        epoch_id=23860,
        block=8589630,
        calculation_snapshot_hash="sha256:" + "1" * 64,
        allocation_hash="sha256:" + "2" * 64,
        leaderboard_window_start="2026-07-03T20:00:00Z",
        leaderboard_window_end="2026-07-10T20:00:00Z",
    )
    assert validate_weight_inputs_request_v2(request) == request
    message = weight_inputs_request_message_v2(request).encode("utf-8")
    assert (
        classify_application_message_v2(message, validator_hotkey=HOTKEY)
        == "validator.gateway_weight_inputs.v2"
    )
    modified = dict(request)
    modified["block"] += 1
    with pytest.raises(HotkeyAuthorityV2Error, match="canonical"):
        weight_inputs_request_message_v2(modified)


def test_weight_input_request_rejects_other_hotkeys_and_inverted_windows():
    request = build_weight_inputs_request_v2(
        validator_hotkey=TARGET,
        netuid=71,
        epoch_id=23860,
        block=8589630,
        calculation_snapshot_hash="sha256:" + "1" * 64,
        allocation_hash="sha256:" + "2" * 64,
        leaderboard_window_start="2026-07-03T20:00:00Z",
        leaderboard_window_end="2026-07-10T20:00:00Z",
    )
    with pytest.raises(HotkeyAuthorityV2Error, match="hotkey differs"):
        classify_application_message_v2(
            weight_inputs_request_message_v2(request).encode("utf-8"),
            validator_hotkey=HOTKEY,
        )


def test_unknown_and_binary_application_messages_are_rejected():
    for message in (b"arbitrary signing oracle", b"\xff\x00", b"GET_EPOCH_LEADS:1:attacker"):
        with pytest.raises(HotkeyAuthorityV2Error):
            classify_application_message_v2(message, validator_hotkey=HOTKEY)


def test_application_signature_request_rejects_purpose_substitution():
    request = build_application_signature_request_v2(
        message=("GET_EPOCH_LEADS:23860:%s" % HOTKEY).encode(),
        validator_hotkey=HOTKEY,
        boot_identity_hash="sha256:" + "4" * 64,
    )
    tampered = copy.deepcopy(request)
    tampered["purpose"] = "validator.bittensor_axon.v2"
    with pytest.raises(HotkeyAuthorityV2Error, match="not canonical"):
        validate_application_signature_request_v2(
            tampered, validator_hotkey=HOTKEY
        )
