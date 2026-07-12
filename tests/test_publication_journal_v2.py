from __future__ import annotations

import json
import os

import pytest

from leadpoet_canonical.hotkey_authority_v2 import (
    build_weight_extrinsic_authorization_v2,
    encode_signed_extrinsic_v2,
    signed_extrinsic_hash_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    validate_published_weight_bundle_v2,
)
from tests.test_validator_hotkey_authority_v2 import _profile
from tests.test_weight_authority_v2 import _bundle
from validator_tee.host.publication_journal_v2 import (
    AuthoritativeWeightPublicationJournalV2,
    WeightPublicationJournalV2Error,
)


EVENT = "sha256:" + "e" * 64
AUTHORIZATION = "sha256:" + "a" * 64


def _publication(bundle):
    verified = validate_published_weight_bundle_v2(bundle)
    return {
        "success": True,
        "epoch_id": verified["epoch_id"],
        "weights_count": len(verified["uids"]),
        "weights_hash": verified["weights_hash"],
        "weight_receipt_hash": verified["weight_receipt_hash"],
        "weight_submission_event_hash": EVENT,
        "message": "published",
    }


def _signature_result(bundle):
    verified = validate_published_weight_bundle_v2(bundle)
    result = bundle["weight_result"]
    authorization = build_weight_extrinsic_authorization_v2(
        profile=_profile(),
        validator_hotkey=bundle["validator_hotkey"],
        hotkey_public_key_hex="11" * 32,
        epoch_id=verified["epoch_id"],
        netuid=verified["netuid"],
        weight_receipt_hash=verified["weight_receipt_hash"],
        weight_submission_event_hash=EVENT,
        weights_hash=verified["weights_hash"],
        sparse_uids=result["sparse_uids"],
        sparse_weights_u16=result["sparse_weights_u16"],
        commitment=b"measured-commitment",
        reveal_round=1234,
        era_current=36099,
        nonce=7,
        block_hash="22" * 32,
    )
    signature = "33" * 64
    extrinsic = encode_signed_extrinsic_v2(
        hotkey_public_key_hex="11" * 32,
        signature_hex=signature,
        era_period=authorization["era_period"],
        era_current=authorization["era_current"],
        nonce=authorization["nonce"],
        call_data_hex=authorization["call_data_hex"],
    )
    return {
        "schema_version": "leadpoet.weight_extrinsic_signature.v2",
        "authorization_hash": authorization["authorization_hash"],
        "validator_hotkey": bundle["validator_hotkey"],
        "signature": signature,
        "extrinsic_hash": signed_extrinsic_hash_v2(extrinsic),
        "authorization": authorization,
        "receipt": {"receipt_hash": "sha256:" + "4" * 64},
    }


def test_journal_fsyncs_before_publication_and_survives_restart(tmp_path):
    bundle = _bundle()
    path = tmp_path / "weight-publication.json"
    journal = AuthoritativeWeightPublicationJournalV2(
        path, chain_profile=_profile()
    )
    prepared = journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    assert prepared["state"] == "prepared"
    assert prepared["publication"] is None
    assert path.exists()
    assert os.stat(path).st_mode & 0o777 == 0o600

    restarted = AuthoritativeWeightPublicationJournalV2(
        path, chain_profile=_profile()
    )
    published = restarted.record_published(_publication(bundle))
    assert published["state"] == "published"
    signed = restarted.record_signed(_signature_result(bundle))
    assert signed["state"] == "signed"
    assert len(signed["extrinsic_signature_results"]) == 1
    assert restarted.record_signed(_signature_result(bundle)) == signed

    restarted.clear(expected_event_hash=EVENT)
    assert not path.exists()


def test_journal_rejects_corruption_and_conflicting_clear(tmp_path):
    bundle = _bundle()
    path = tmp_path / "weight-publication.json"
    journal = AuthoritativeWeightPublicationJournalV2(
        path, chain_profile=_profile()
    )
    journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    journal.record_published(_publication(bundle))
    with pytest.raises(WeightPublicationJournalV2Error, match="another"):
        journal.clear(expected_event_hash="sha256:" + "f" * 64)

    value = json.loads(path.read_text(encoding="utf-8"))
    value["weight_authorization_id"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(WeightPublicationJournalV2Error, match="hash"):
        journal.load()


def test_journal_will_not_replace_an_unfinished_epoch(tmp_path):
    journal = AuthoritativeWeightPublicationJournalV2(
        tmp_path / "weight-publication.json", chain_profile=_profile()
    )
    bundle = _bundle()
    journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    with pytest.raises(WeightPublicationJournalV2Error, match="unfinished"):
        journal.record_prepared(
            {
                "weight_authorization_id": "sha256:" + "b" * 64,
                "published_bundle": bundle,
            }
        )
