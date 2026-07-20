import pytest

from gateway.tee.coordinator_executor_v2 import (
    OP_ATTEST_ARTIFACT_PERSISTENCE,
    OP_ATTEST_LEGACY_FINALIZED_ALLOCATION_V2,
    OP_ATTEST_WEIGHT_INPUT,
    OP_ATTEST_WEIGHT_PUBLICATION,
    OP_PROVIDER_OUTCOME_SNAPSHOT_V2,
    OP_RESEARCH_LAB_ALLOCATION,
    CoordinatorExecutorV2,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.provider_outcome_v2 import ProviderOutcomeLedgerV2
from gateway.tee.scoring_executor import ScoringExecutionResult
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    weight_config_hash,
)
from leadpoet_canonical.weight_authority_v2 import (
    gateway_weight_input_value_documents_v2,
)
from research_lab.eval.promotion_metric import promotion_improvement_metric


def _weight_snapshot():
    value = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 100,
        "block": 36099,
        "commit_sha": "a" * 40,
        "config_hash": "",
        "parent_receipt_hashes": [],
        "research_lab_allocation_receipt_hash": "",
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": "burn",
        "metagraph_hotkeys": ["burn", "miner"],
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
        "ff_enabled": True,
        "base_burn_share": 0.0,
        "champion_share": 0.0,
        "champion_uid": None,
        "effective_champion_share": 0.0,
        "research_lab_fallback_share": 0.2,
        "research_lab_allocation_doc": {
            "lab_cap_percent": 20.0,
            "unallocated_percent": 20.0,
            "source_add_allocations": [],
            "reimbursement_allocations": [],
            "champion_allocations": [],
            "queued_champion_allocations": [],
        },
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.705,
        "fulfillment_rows": [{"hotkey": "miner", "share": 0.705}],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125000,
        "min_total_rep_for_distribution": 100,
    }
    value["config_hash"] = weight_config_hash(value)
    return value


@pytest.mark.asyncio
async def test_coordinator_calls_unchanged_promotion_metric():
    score_bundle = {"aggregates": {"mean_delta": 1.75}}
    expected = promotion_improvement_metric(score_bundle)
    result = await CoordinatorExecutorV2()(
        "promotion_improvement",
        {"score_bundle": score_bundle},
        ExecutionContextV2(
            job_id="promotion:test",
            purpose="research_lab.ranking.v2",
            epoch_id=1,
        ),
    )
    assert result.output == {
        "improvement_points": expected.improvement_points,
        "event_doc": expected.event_doc(),
    }


@pytest.mark.asyncio
async def test_coordinator_rejects_operation_outside_measured_authority():
    with pytest.raises(ValueError, match="unsupported"):
        await CoordinatorExecutorV2()(
            "qualification_company_scores",
            {},
            ExecutionContextV2(
                job_id="score:test",
                purpose="research_lab.ranking.v2",
                epoch_id=1,
            ),
        )


@pytest.mark.asyncio
async def test_coordinator_allocation_binds_projected_receipt_output(monkeypatch):
    allocation = {"epoch_id": 100, "champion_allocations": []}
    source_state_hash = "sha256:" + "1" * 64
    kernel_evidence_hash = "sha256:" + "2" * 64
    authority = {
        "allocation": allocation,
        "allocation_inputs": {"epoch_id": 100},
        "source_state": {"epoch_id": 100},
        "source_state_hash": source_state_hash,
    }

    async def execute_allocation(operation, payload):
        assert operation == OP_RESEARCH_LAB_ALLOCATION
        assert payload == authority["allocation_inputs"]
        return ScoringExecutionResult(
            {"allocation": allocation},
            {"allocation_kernel": kernel_evidence_hash},
        )

    monkeypatch.setattr(
        "gateway.tee.coordinator_executor_v2.execute_scoring_operation",
        execute_allocation,
    )
    result = await CoordinatorExecutorV2(
        allocation_source_resolver=lambda _payload, _context: authority
    )(
        OP_RESEARCH_LAB_ALLOCATION,
        {"epoch_id": 100},
        ExecutionContextV2(
            job_id="allocation:test",
            purpose="research_lab.allocation.v2",
            epoch_id=100,
        ),
    )

    assert result.output == authority
    assert result.receipt_output == {"allocation": allocation}
    assert set(result.artifact_hashes) == {
        source_state_hash,
        kernel_evidence_hash,
    }


@pytest.mark.asyncio
async def test_coordinator_attests_legacy_settlement_only_from_measured_source():
    document = {
        "settlement_hash": "sha256:" + "1" * 64,
        "allocation_hash": "sha256:" + "2" * 64,
        "chain_compare_hash": "sha256:" + "3" * 64,
        "audit_event_hash": "sha256:" + "4" * 64,
        "checkpoint_merkle_root": "sha256:" + "5" * 64,
    }
    calls = []

    def resolver(payload, context):
        calls.append((dict(payload), context.job_id))
        return document

    result = await CoordinatorExecutorV2(
        legacy_settlement_source_resolver=resolver
    )(
        OP_ATTEST_LEGACY_FINALIZED_ALLOCATION_V2,
        {
            "schema_version": "leadpoet.legacy_finalized_allocation_request.v2",
            "netuid": 71,
            "epoch_id": 100,
        },
        ExecutionContextV2(
            job_id="legacy-settlement:100",
            purpose="research_lab.legacy_finalized_allocation.v2",
            epoch_id=101,
        ),
    )
    assert result.output == document
    assert set(result.artifact_hashes) == set(document.values())
    assert calls[0][1] == "legacy-settlement:100"

    with pytest.raises(ValueError, match="source is unavailable"):
        await CoordinatorExecutorV2()(
            OP_ATTEST_LEGACY_FINALIZED_ALLOCATION_V2,
            {
                "schema_version": "leadpoet.legacy_finalized_allocation_request.v2",
                "netuid": 71,
                "epoch_id": 100,
            },
            ExecutionContextV2(
                job_id="legacy-settlement:missing",
                purpose="research_lab.legacy_finalized_allocation.v2",
                epoch_id=101,
            ),
        )


@pytest.mark.asyncio
async def test_coordinator_attests_measured_provider_outcome_snapshot():
    snapshot = ProviderOutcomeLedgerV2(
        clock=lambda: "2026-07-10T12:00:00Z"
    ).snapshot()
    checkpoint_hash = "sha256:" + "9" * 64
    result = await CoordinatorExecutorV2(
        provider_outcome_supplier=lambda: {
            "snapshot": snapshot,
            "transport_attempts": [],
            "evidence_artifact_hashes": [checkpoint_hash],
        }
    )(
        OP_PROVIDER_OUTCOME_SNAPSHOT_V2,
        {"schema_version": "leadpoet.provider_outcome_snapshot_request.v2"},
        ExecutionContextV2(
            job_id="provider-outcome:1",
            purpose="research_lab.provider_outcome_snapshot.v2",
            epoch_id=1,
        ),
    )
    assert result.output == snapshot
    assert set(result.artifact_hashes) == {
        snapshot["provider_outcome_digest_hash"],
        snapshot["source_state_hash"],
        checkpoint_hash,
    }


@pytest.mark.asyncio
async def test_coordinator_rejects_tampered_provider_outcome_snapshot():
    snapshot = ProviderOutcomeLedgerV2(
        clock=lambda: "2026-07-10T12:00:00Z"
    ).snapshot()
    snapshot["source_state_hash"] = "sha256:" + "f" * 64
    with pytest.raises(Exception, match="commitments differ"):
        await CoordinatorExecutorV2(
            provider_outcome_supplier=lambda: {
                "snapshot": snapshot,
                "transport_attempts": [],
                "evidence_artifact_hashes": [],
            }
        )(
            OP_PROVIDER_OUTCOME_SNAPSHOT_V2,
            {"schema_version": "leadpoet.provider_outcome_snapshot_request.v2"},
            ExecutionContextV2(
                job_id="provider-outcome:1",
                purpose="research_lab.provider_outcome_snapshot.v2",
                epoch_id=1,
            ),
        )


@pytest.mark.asyncio
async def test_coordinator_attests_only_complete_persisted_artifact_evidence():
    artifact_id = "sha256:" + "a" * 64
    plaintext_hash = "sha256:" + "b" * 64
    attempts = [
        {"attempt_hash": "sha256:" + "c" * 64},
        {"attempt_hash": "sha256:" + "d" * 64},
    ]
    evidence = {
        "artifact_id": artifact_id,
        "plaintext_hash": plaintext_hash,
        "ciphertext_hash": "sha256:" + "e" * 64,
        "artifact_ref": "s3://immutable/artifact.json",
        "storage_document_hash": "sha256:" + "f" * 64,
        "encryption_context_hash": "sha256:" + "3" * 64,
        "object_lock_mode": "COMPLIANCE",
        "retain_until": "2027-07-10T12:00:00Z",
        "transport_root": "sha256:" + "1" * 64,
        "transport_attempts": attempts,
        "persisted": True,
    }
    executor = CoordinatorExecutorV2(
        artifact_evidence_supplier=lambda _ids, _context: [evidence]
    )
    result = await executor(
        OP_ATTEST_ARTIFACT_PERSISTENCE,
        {
            "source_receipt_hash": "sha256:" + "2" * 64,
            "artifact_ids": [artifact_id],
            "artifact_plaintext_hashes": [plaintext_hash],
        },
        ExecutionContextV2(
            job_id="artifact:test",
            purpose="leadpoet.artifact_persistence.v2",
            epoch_id=1,
        ),
    )
    assert result.output["source_receipt_hash"] == "sha256:" + "2" * 64
    assert result.output["artifacts"][0]["artifact_id"] == artifact_id
    assert list(result.transport_attempts) == attempts


@pytest.mark.asyncio
async def test_coordinator_rejects_artifact_plaintext_mismatch():
    executor = CoordinatorExecutorV2(
        artifact_evidence_supplier=lambda _ids, _context: [
            {
                "artifact_id": "sha256:" + "a" * 64,
                "plaintext_hash": "sha256:" + "b" * 64,
                "persisted": True,
            }
        ]
    )
    with pytest.raises(ValueError, match="plaintext commitments differ"):
        await executor(
            OP_ATTEST_ARTIFACT_PERSISTENCE,
            {
                "source_receipt_hash": "sha256:" + "2" * 64,
                "artifact_ids": ["sha256:" + "a" * 64],
                "artifact_plaintext_hashes": ["sha256:" + "f" * 64],
            },
            ExecutionContextV2(
                job_id="artifact:test",
                purpose="leadpoet.artifact_persistence.v2",
                epoch_id=1,
            ),
        )


@pytest.mark.asyncio
async def test_coordinator_derives_exact_weight_input_document():
    def resolver(payload, _context):
        return gateway_weight_input_value_documents_v2(
            calculation_snapshot=payload["calculation_snapshot"],
            gateway_authority_event_hash=payload["gateway_authority_event_hash"],
        )[payload["category"]]

    result = await CoordinatorExecutorV2(weight_source_resolver=resolver)(
        OP_ATTEST_WEIGHT_INPUT,
        {
            "category": "bans",
            "calculation_snapshot": _weight_snapshot(),
            "gateway_authority_event_hash": "sha256:" + "2" * 64,
        },
        ExecutionContextV2(
            job_id="weight-input:bans:100",
            purpose="research_lab.ban_input.v2",
            epoch_id=100,
        ),
    )
    assert result.output["category"] == "bans"
    assert result.output["value"] == {
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
    }


@pytest.mark.asyncio
async def test_coordinator_rejects_weight_category_purpose_substitution():
    with pytest.raises(ValueError, match="purpose is incorrect"):
        await CoordinatorExecutorV2(
            weight_source_resolver=lambda _payload, _context: {}
        )(
            OP_ATTEST_WEIGHT_INPUT,
            {
                "category": "bans",
                "calculation_snapshot": _weight_snapshot(),
                "gateway_authority_event_hash": "sha256:" + "2" * 64,
            },
            ExecutionContextV2(
                job_id="weight-input:bans:100",
                purpose="research_lab.fulfillment_input.v2",
                epoch_id=100,
            ),
        )


@pytest.mark.asyncio
async def test_coordinator_rejects_host_snapshot_without_measured_source():
    with pytest.raises(ValueError, match="measured weight input source"):
        await CoordinatorExecutorV2()(
            OP_ATTEST_WEIGHT_INPUT,
            {
                "category": "bans",
                "calculation_snapshot": _weight_snapshot(),
                "gateway_authority_event_hash": "sha256:" + "2" * 64,
            },
            ExecutionContextV2(
                job_id="weight-input:bans:100",
                purpose="research_lab.ban_input.v2",
                epoch_id=100,
            ),
        )


@pytest.mark.asyncio
async def test_coordinator_attests_only_complete_weight_publication_commitments():
    payload = {
        "bundle_hash": "sha256:" + "1" * 64,
        "root_receipt_hash": "sha256:" + "2" * 64,
        "durable_readback_hash": "sha256:" + "3" * 64,
        "transparency_event_hash": "sha256:" + "4" * 64,
    }
    result = await CoordinatorExecutorV2()(
        OP_ATTEST_WEIGHT_PUBLICATION,
        payload,
        ExecutionContextV2(
            job_id="weight-publication:100",
            purpose="gateway.weights.publication.v2",
            epoch_id=100,
        ),
    )
    assert result.output == {
        "schema_version": "leadpoet.weight_publication.v2",
        **payload,
    }


@pytest.mark.asyncio
async def test_coordinator_rejects_incomplete_weight_publication():
    with pytest.raises(ValueError, match="payload fields"):
        await CoordinatorExecutorV2()(
            OP_ATTEST_WEIGHT_PUBLICATION,
            {"bundle_hash": "sha256:" + "1" * 64},
            ExecutionContextV2(
                job_id="weight-publication:100",
                purpose="gateway.weights.publication.v2",
                epoch_id=100,
            ),
        )
