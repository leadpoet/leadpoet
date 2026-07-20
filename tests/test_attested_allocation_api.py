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
async def test_attested_allocation_uses_execution_root_when_artifact_receipt_wraps_it(
    monkeypatch,
):
    expected_bundle = {
        "bundle_type": "research_lab_live_allocation_bundle",
        "epoch": 7,
    }
    execution_receipt = {"receipt_hash": "sha256:" + "1" * 64}
    artifact_receipt = {"receipt_hash": "sha256:" + "2" * 64}
    artifact_graph = {
        "root_receipt_hash": artifact_receipt["receipt_hash"],
        "receipts": [execution_receipt, artifact_receipt],
    }
    execution_graph = {
        "root_receipt_hash": execution_receipt["receipt_hash"],
        "boot_identities": [{"boot_identity_hash": "sha256:" + "3" * 64}],
        "receipts": [execution_receipt],
        "transport_attempts": [],
        "host_operations": [],
    }

    async def _build(**kwargs):
        kwargs["attestation_out"].update(
            {
                "status": "matched",
                "receipt": artifact_receipt,
                "execution_receipt": execution_receipt,
                "receipt_graph": artifact_graph,
                "lineage_bindings": [],
                "lineage_complete": True,
                "persistence": {
                    "root_receipt_hash": artifact_receipt["receipt_hash"]
                },
            }
        )
        return expected_bundle

    async def _load_graph(root_receipt_hash):
        assert root_receipt_hash == execution_receipt["receipt_hash"]
        return execution_graph

    def _handoff(**kwargs):
        assert kwargs["bundle"] == expected_bundle
        assert kwargs["receipt_graph"] == execution_graph
        assert kwargs["persistence"]["root_receipt_hash"] == execution_receipt[
            "receipt_hash"
        ]
        assert kwargs["persistence"]["receipt_count"] == 1
        return {"schema_version": "leadpoet.attested_allocation_handoff.v2"}

    monkeypatch.setattr(api.ResearchLabGatewayConfig, "from_env", _config)
    monkeypatch.setattr(api, "build_research_lab_allocation_bundle", _build)
    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store.load_receipt_graph_v2",
        _load_graph,
    )
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
