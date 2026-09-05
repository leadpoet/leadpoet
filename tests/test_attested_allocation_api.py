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
async def test_attested_allocation_warm_starts_from_exact_release_disk_cache(
    monkeypatch,
):
    epoch = 24103
    commit = "a" * 40
    handoff = {
        "schema_version": "leadpoet.attested_allocation_handoff.v2",
        "epoch": epoch,
    }
    observed = {}

    async def guard(_config_value, requested_epoch, internal_key):
        assert requested_epoch == epoch
        assert internal_key == "internal"
        return True

    def load(netuid, requested_epoch, persist_snapshot, release_commit):
        observed.update(
            {
                "netuid": netuid,
                "epoch": requested_epoch,
                "persist_snapshot": persist_snapshot,
                "release_commit": release_commit,
            }
        )
        return handoff

    def unexpected_build(**_kwargs):
        raise AssertionError("cold allocation build should not run")

    api._ALLOCATION_HANDOFF_CACHE.clear()
    api._ALLOCATION_BUILD_TASKS.clear()
    monkeypatch.setattr(api.ResearchLabGatewayConfig, "from_env", _config)
    monkeypatch.setattr(
        api,
        "_allocation_epoch_guard_and_persistence",
        guard,
    )
    monkeypatch.setattr(api, "_allocation_cache_release_commit", lambda: commit)
    monkeypatch.setattr(
        api.allocation_handoff_disk_cache,
        "load_handoff",
        load,
    )
    monkeypatch.setattr(api, "_allocation_build_task", unexpected_build)

    result = await api.get_research_lab_attested_allocation(
        epoch,
        x_leadpoet_internal_key="internal",
    )

    assert result == handoff
    assert observed == {
        "netuid": 71,
        "epoch": epoch,
        "persist_snapshot": True,
        "release_commit": commit,
    }
    api._ALLOCATION_HANDOFF_CACHE.clear()
    api._ALLOCATION_BUILD_TASKS.clear()


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


@pytest.mark.parametrize(
    ("raised", "expected_status"),
    [
        (TimeoutError("bundle build exceeded its budget"), 504),
        (RuntimeError("durable allocation retry generations are exhausted"), 503),
        (ValueError("champion lifetime credit is duplicated within epoch"), 500),
        (KeyError("score_bundle_id"), 500),
    ],
)
@pytest.mark.asyncio
async def test_attested_allocation_failure_class_is_visible_in_the_status_code(
    monkeypatch, raised, expected_status
):
    """A failed build must be separable by status code alone.

    Only the status code reaches request telemetry, so a timeout, a refusal to
    attempt the build, and a build that genuinely broke have to leave the
    endpoint as three different codes or an operator cannot tell them apart.
    """

    async def _build(**kwargs):
        raise raised

    monkeypatch.setattr(api.ResearchLabGatewayConfig, "from_env", _config)
    monkeypatch.setattr(api, "build_research_lab_allocation_bundle", _build)

    with pytest.raises(HTTPException) as exc_info:
        await api.get_research_lab_attested_allocation(11)
    assert exc_info.value.status_code == expected_status


@pytest.mark.asyncio
async def test_attested_allocation_timeout_is_not_reported_as_a_refusal(monkeypatch):
    """asyncio.TimeoutError subclasses nothing useful here — pin it explicitly."""

    import asyncio as _asyncio

    async def _build(**kwargs):
        raise _asyncio.TimeoutError()

    monkeypatch.setattr(api.ResearchLabGatewayConfig, "from_env", _config)
    monkeypatch.setattr(api, "build_research_lab_allocation_bundle", _build)

    with pytest.raises(HTTPException) as exc_info:
        await api.get_research_lab_attested_allocation(12)
    assert exc_info.value.status_code == 504
