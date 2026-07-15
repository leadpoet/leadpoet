from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from gateway.research_lab import api


def _config():
    return SimpleNamespace(
        api_enabled=True,
        reports_enabled=True,
        shadow_bundles_enabled=True,
        reimbursements_enabled=False,
        weight_mutation_enabled=False,
    )


@pytest.mark.asyncio
async def test_attested_allocation_wraps_unchanged_bundle(monkeypatch):
    expected_bundle = {"bundle_type": "research_lab_live_allocation_bundle", "epoch": 7}
    receipt = {"receipt_hash": "sha256:" + "1" * 64}
    graph = {"root_receipt_hash": receipt["receipt_hash"], "receipts": [receipt]}
    persistence = {"root_receipt_hash": receipt["receipt_hash"]}

    async def _build(**kwargs):
        assert kwargs["persist_snapshot"] is False
        kwargs["attestation_out"].update(
            {
                "status": "matched",
                "receipt": receipt,
                "receipt_graph": graph,
                "lineage_bindings": [],
                "lineage_complete": True,
                "persistence": persistence,
            }
        )
        return expected_bundle

    def _handoff(**kwargs):
        assert kwargs == {
            "bundle": expected_bundle,
            "receipt_graph": graph,
            "lineage_bindings": [],
            "lineage_complete": True,
            "persistence": persistence,
        }
        return {"schema_version": "leadpoet.attested_allocation_handoff.v2"}

    monkeypatch.setattr(api.ResearchLabGatewayConfig, "from_env", _config)
    monkeypatch.setattr(api, "build_research_lab_allocation_bundle", _build)
    monkeypatch.setattr(
        "leadpoet_canonical.allocation_handoff_v2.build_allocation_handoff_v2",
        _handoff,
    )

    result = await api.get_research_lab_attested_allocation(7)

    assert result == {
        "schema_version": "leadpoet.attested_allocation_handoff.v2"
    }


@pytest.mark.asyncio
async def test_attested_allocation_is_unavailable_when_v2_authority_did_not_match(monkeypatch):
    async def _build(**kwargs):
        kwargs["attestation_out"].update({"status": "shadow_mismatch"})
        return {"epoch": 8}

    monkeypatch.setattr(api.ResearchLabGatewayConfig, "from_env", _config)
    monkeypatch.setattr(api, "build_research_lab_allocation_bundle", _build)

    with pytest.raises(HTTPException) as exc_info:
        await api.get_research_lab_attested_allocation(8)
    assert exc_info.value.status_code == 503
