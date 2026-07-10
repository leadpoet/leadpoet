import pytest
from fastapi import HTTPException

from gateway.api import weights as weights_api
from leadpoet_canonical.weight_bundle_v2 import WEIGHT_BUNDLE_V2_SCHEMA_VERSION
from leadpoet_canonical.weight_computation import WEIGHT_SNAPSHOT_SCHEMA_VERSION, weight_config_hash
from validator_tee.enclave import tee_service


def _submission():
    snapshot = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 300,
        "block": 108350,
        "commit_sha": "a" * 40,
        "config_hash": "",
        "parent_receipt_hashes": [],
        "research_lab_allocation_receipt_hash": "",
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": "burn",
        "metagraph_hotkeys": ["burn"],
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
        "ff_enabled": False,
        "base_burn_share": 0.0,
        "champion_share": 0.0,
        "champion_uid": None,
        "effective_champion_share": 0.0,
        "research_lab_fallback_share": 0.2,
        "research_lab_allocation_doc": {},
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.0,
        "fulfillment_rows": [],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125000,
        "min_total_rep_for_distribution": 100,
    }
    snapshot["config_hash"] = weight_config_hash(snapshot)
    response = tee_service.handle_request({"command": "compute_weights_v2", "snapshot": snapshot})
    assert response["status"] == "ok"
    return weights_api.WeightSubmissionV2(
        schema_version=WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
        validator_hotkey="validator-hotkey",
        binding_message="binding",
        validator_hotkey_signature="hotkey-signature",
        weight_snapshot=snapshot,
        weight_result=response["weight_result"],
        weights_signature=response["weights_signature"],
        weight_receipt=response["receipt"],
        parent_receipts=[],
    )


class _Subtensor:
    def get_current_block(self):
        return 108350


@pytest.mark.asyncio
async def test_v2_route_defaults_off_without_verification(monkeypatch):
    monkeypatch.delenv("GATEWAY_WEIGHT_SUBMISSION_V2_MODE", raising=False)
    with pytest.raises(HTTPException) as exc:
        await weights_api.submit_weights_v2(_submission())
    assert exc.value.status_code == 503
    assert "disabled" in exc.value.detail


@pytest.mark.asyncio
async def test_v2_shadow_verifies_without_database_or_v1_publication(monkeypatch):
    monkeypatch.setenv("GATEWAY_WEIGHT_SUBMISSION_V2_MODE", "shadow")
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {"validator-hotkey"})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: _Subtensor())
    monkeypatch.setattr(weights_api, "verify_binding_message", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        weights_api,
        "verify_validator_attestation_v2",
        lambda **kwargs: (
            True,
            {
                "pcr0_verification_mode": "dynamic_github",
                "pcr0_commit": "a" * 40,
                "code_hash": "b" * 64,
            },
        ),
    )
    response = await weights_api.submit_weights_v2(_submission())
    assert response.success is True
    assert response.mode == "shadow"
    assert response.weights_count == 1
    assert response.persistence_status == "disabled"
    assert "v1 remains authoritative" in response.message


@pytest.mark.asyncio
async def test_v2_shadow_can_persist_only_the_additive_sidecar(monkeypatch):
    from gateway.research_lab import attested_receipt_store

    persisted = {}

    async def _persist(**kwargs):
        persisted.update(kwargs)
        return {}

    monkeypatch.setenv("GATEWAY_WEIGHT_SUBMISSION_V2_MODE", "shadow")
    monkeypatch.setenv("GATEWAY_WEIGHT_SUBMISSION_V2_PERSIST_ENABLED", "true")
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {"validator-hotkey"})
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {71})
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: _Subtensor())
    monkeypatch.setattr(weights_api, "verify_binding_message", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        weights_api,
        "verify_validator_attestation_v2",
        lambda **kwargs: (
            True,
            {
                "pcr0_verification_mode": "dynamic_github",
                "pcr0_commit": "a" * 40,
                "pcr0": "c" * 96,
                "code_hash": "b" * 64,
            },
        ),
    )
    monkeypatch.setattr(attested_receipt_store, "persist_attested_weight_bundle", _persist)

    response = await weights_api.submit_weights_v2(_submission())

    assert response.persistence_status == "persisted"
    assert persisted["verification_mode"] == "shadow"
    assert persisted["validator_pcr0"] == "c" * 96


@pytest.mark.asyncio
async def test_v2_latest_returns_only_validated_sidecar_bundle(monkeypatch):
    from gateway.research_lab import attested_receipt_store

    expected = _submission().model_dump(mode="python")

    async def _load(**kwargs):
        assert kwargs == {"netuid": 71, "epoch_id": 300}
        return expected

    monkeypatch.setattr(attested_receipt_store, "load_attested_weight_bundle", _load)
    assert await weights_api.get_attested_weights_v2(71, 300) == expected


def test_v2_attestation_rejects_static_allowlist_even_when_nitro_is_valid(monkeypatch):
    submission = _submission()
    monkeypatch.setattr(
        "leadpoet_canonical.nitro.verify_nitro_attestation_full",
        lambda **kwargs: (
            True,
            {
                "purpose": "validator.weights.computed.v2",
                "epoch_id": 300,
                "pcr0_verification_mode": "static_allowlist",
                "pcr0_commit": "a" * 40,
            },
        ),
    )
    valid, data = weights_api.verify_validator_attestation_v2(
        receipt=submission.weight_receipt,
        expected_epoch_id=300,
    )
    assert valid is False
    assert "dynamic Git-derived" in data["error"]


def test_v2_scoring_receipt_requires_exact_aws_nitro_bindings(monkeypatch):
    captured = {}
    receipt = {
        "purpose": "research_lab.allocation.v1",
        "epoch_id": 300,
        "enclave_pubkey": "a" * 64,
        "attestation_document_b64": "attestation",
    }

    def _verify(**kwargs):
        captured.update(kwargs)
        return True, {
            "purpose": receipt["purpose"],
            "epoch_id": receipt["epoch_id"],
            "enclave_pubkey": receipt["enclave_pubkey"],
            "pcr0": "b" * 96,
        }

    monkeypatch.setattr(
        "leadpoet_canonical.nitro.verify_nitro_attestation_full",
        _verify,
    )

    valid, data = weights_api.verify_scoring_receipt_attestation_v2(receipt)

    assert valid is True
    assert data["pcr0"] == "b" * 96
    assert captured["expected_purpose"] == receipt["purpose"]
    assert captured["expected_epoch_id"] == 300
    assert captured["expected_pubkey"] == receipt["enclave_pubkey"]
    assert captured["skip_pcr0_verification"] is True


def test_v2_scoring_receipt_rejects_missing_attested_purpose(monkeypatch):
    receipt = {
        "purpose": "research_lab.allocation.v1",
        "epoch_id": 300,
        "enclave_pubkey": "a" * 64,
        "attestation_document_b64": "attestation",
    }
    monkeypatch.setattr(
        "leadpoet_canonical.nitro.verify_nitro_attestation_full",
        lambda **_kwargs: (
            True,
            {
                "purpose": None,
                "epoch_id": 300,
                "enclave_pubkey": "a" * 64,
                "pcr0": "b" * 96,
            },
        ),
    )

    valid, data = weights_api.verify_scoring_receipt_attestation_v2(receipt)

    assert valid is False
    assert "purpose" in data["error"]


@pytest.mark.asyncio
async def test_v2_required_mode_is_locked_until_sidecar_authority_exists(monkeypatch):
    submission = _submission()
    monkeypatch.setenv("GATEWAY_WEIGHT_SUBMISSION_V2_MODE", "required")
    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {"validator-hotkey"})
    monkeypatch.setattr(weights_api, "get_subtensor", lambda: _Subtensor())
    monkeypatch.setattr(weights_api, "verify_binding_message", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        weights_api,
        "verify_validator_attestation_v2",
        lambda **kwargs: (True, {"pcr0_commit": "a" * 40, "code_hash": "b" * 64}),
    )
    with pytest.raises(HTTPException) as exc:
        await weights_api.submit_weights_v2(submission)
    # Required mode first rejects the intentionally absent Research Lab receipt.
    assert exc.value.status_code == 400
    assert "allocation receipt is required" in exc.value.detail
