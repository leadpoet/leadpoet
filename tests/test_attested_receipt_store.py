import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab import attested_receipt_store
from leadpoet_canonical.attested_receipts import build_receipt_body, create_signed_receipt


def _receipt(*, purpose="research_lab.allocation.v1", job_id="allocation:1", parents=()):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    body = build_receipt_body(
        role="gateway_scoring",
        purpose=purpose,
        job_id=job_id,
        epoch_id=1,
        commit_sha="1" * 40,
        build_manifest_hash="sha256:" + "2" * 64,
        config_hash="sha256:" + "3" * 64,
        input_root="sha256:" + "4" * 64,
        output_root="sha256:" + "5" * 64,
        evidence_roots={},
        parent_receipt_hashes=list(parents),
        status="succeeded",
        issued_at="2026-07-10T00:00:00Z",
    )
    return create_signed_receipt(
        body=body,
        enclave_pubkey=public_key,
        attestation_document_b64=base64.b64encode(b"nitro-attestation").decode(),
        sign_digest=private_key.sign,
    )


def test_receipt_storage_row_is_complete_and_redacted():
    receipt = _receipt()
    row = attested_receipt_store.receipt_storage_row(receipt=receipt, pcr0="a" * 96)

    assert row["receipt_hash"] == receipt["receipt_hash"]
    assert row["receipt_doc"] == receipt
    assert row["pcr0"] == "a" * 96
    assert row["attestation_document_ref"].startswith("inline:sha256:")


@pytest.mark.asyncio
async def test_persist_receipt_and_link(monkeypatch):
    writes = []

    async def _insert(table, row):
        writes.append((table, dict(row)))
        return dict(row)

    monkeypatch.setattr(attested_receipt_store, "insert_row", _insert)
    receipt = _receipt()
    stored = await attested_receipt_store.persist_attested_receipt(
        receipt=receipt,
        pcr0="b" * 96,
        artifact_links=[
            {
                "artifact_kind": "allocation",
                "artifact_ref": "epoch:1",
                "artifact_hash": "sha256:" + "6" * 64,
            }
        ],
    )

    assert stored["receipt_hash"] == receipt["receipt_hash"]
    assert [item[0] for item in writes] == [
        "research_lab_attested_execution_receipts",
        "research_lab_attested_artifact_links",
    ]


@pytest.mark.asyncio
async def test_duplicate_receipt_must_match_existing_row(monkeypatch):
    receipt = _receipt()
    expected = attested_receipt_store.receipt_storage_row(receipt=receipt, pcr0="c" * 96)

    async def _duplicate(_table, _row):
        raise RuntimeError("duplicate key 23505")

    async def _select(_table, *, filters):
        assert filters == (("receipt_hash", receipt["receipt_hash"]),)
        return dict(expected)

    monkeypatch.setattr(attested_receipt_store, "insert_row", _duplicate)
    monkeypatch.setattr(attested_receipt_store, "select_one", _select)

    stored = await attested_receipt_store.persist_attested_receipt(
        receipt=receipt,
        pcr0="c" * 96,
    )
    assert stored == expected


def test_zero_pcr0_is_rejected():
    with pytest.raises(attested_receipt_store.AttestedReceiptStoreError, match="PCR0"):
        attested_receipt_store.receipt_storage_row(receipt=_receipt(), pcr0="0" * 96)


@pytest.mark.asyncio
async def test_load_receipt_for_artifact_validates_stored_receipt(monkeypatch):
    receipt = _receipt()

    async def _select_many(table, *, filters, order_by, limit):
        assert table == "research_lab_attested_artifact_links"
        assert filters == (
            ("artifact_kind", "score_bundle"),
            ("artifact_ref", "score_bundle:abc"),
            ("artifact_hash", "sha256:" + "7" * 64),
        )
        assert order_by == (("created_at", True),)
        assert limit == 20
        return [{"receipt_hash": receipt["receipt_hash"]}]

    async def _select_one(table, *, filters):
        assert table == "research_lab_attested_execution_receipts"
        assert filters == (("receipt_hash", receipt["receipt_hash"]),)
        return {"receipt_doc": receipt}

    monkeypatch.setattr(attested_receipt_store, "select_many", _select_many)
    monkeypatch.setattr(attested_receipt_store, "select_one", _select_one)

    loaded = await attested_receipt_store.load_receipt_for_artifact(
        artifact_kind="score_bundle",
        artifact_ref="score_bundle:abc",
        artifact_hash="sha256:" + "7" * 64,
    )
    assert loaded == receipt


@pytest.mark.asyncio
async def test_load_receipt_lineage_returns_validated_parent_first_order(monkeypatch):
    score = _receipt(
        purpose="research_lab.candidate_score.v1",
        job_id="score:1",
    )
    promotion = _receipt(
        purpose="research_lab.promotion_decision.v1",
        job_id="promotion:1",
        parents=[score["receipt_hash"]],
    )
    allocation = _receipt(
        purpose="research_lab.allocation.v1",
        job_id="allocation:2",
        parents=[promotion["receipt_hash"]],
    )
    receipts = {
        score["receipt_hash"]: score,
        promotion["receipt_hash"]: promotion,
    }

    async def _select_one(_table, *, filters):
        receipt_hash = filters[0][1]
        receipt = receipts.get(receipt_hash)
        return {"receipt_doc": receipt} if receipt else None

    monkeypatch.setattr(attested_receipt_store, "select_one", _select_one)

    lineage = await attested_receipt_store.load_attested_receipt_lineage(allocation)
    assert [item["receipt_hash"] for item in lineage] == [
        score["receipt_hash"],
        promotion["receipt_hash"],
    ]


@pytest.mark.asyncio
async def test_persist_v2_weight_bundle_uses_only_additive_sidecars(monkeypatch):
    from leadpoet_canonical import weight_bundle_v2

    receipt = _receipt()
    bundle = {
        "validator_hotkey": "validator",
        "weight_receipt": receipt,
        "opaque": "canonical-v2-bundle",
    }
    writes = []
    receipt_writes = []

    monkeypatch.setattr(
        weight_bundle_v2,
        "validate_weight_bundle_v2",
        lambda *_args, **_kwargs: {
            "netuid": 71,
            "epoch_id": 9,
            "block": 3333,
            "weights_hash": "d" * 64,
        },
    )

    async def _persist_receipt(**kwargs):
        receipt_writes.append(kwargs)
        return {}

    async def _insert(table, row):
        writes.append((table, dict(row)))
        return dict(row)

    monkeypatch.setattr(attested_receipt_store, "persist_attested_receipt", _persist_receipt)
    monkeypatch.setattr(attested_receipt_store, "insert_row", _insert)

    stored = await attested_receipt_store.persist_attested_weight_bundle(
        bundle=bundle,
        validator_pcr0="e" * 96,
        verification_mode="shadow",
    )

    assert receipt_writes[0]["receipt"] == receipt
    assert receipt_writes[0]["artifact_links"][0]["artifact_hash"] == "sha256:" + "d" * 64
    assert writes[0][0] == "research_lab_attested_weight_bundles"
    assert stored["verification_mode"] == "shadow"
