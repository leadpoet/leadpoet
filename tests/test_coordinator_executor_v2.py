import pytest

from gateway.tee.artifact_vault_v2 import (
    ARTIFACT_PERSISTENCE_MAX_ATTEMPTS_PER_METHOD,
)
from gateway.tee.coordinator_executor_v2 import (
    OP_ANCESTRY_CHECKPOINT_BOOTSTRAP_V2,
    OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
    OP_ATTEST_ARTIFACT_PERSISTENCE,
    OP_ATTEST_LEGACY_FINALIZED_ALLOCATION_V2,
    OP_ATTEST_WEIGHT_INPUT,
    OP_ATTEST_WEIGHT_PUBLICATION,
    OP_OBSERVE_CHAIN_REALIZED_WEIGHTS_V1,
    OP_PROVIDER_OUTCOME_SNAPSHOT_V2,
    OP_RESEARCH_LAB_ALLOCATION,
    CoordinatorExecutorV2,
    coordinator_receipt_output_v2,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.provider_outcome_v2 import ProviderOutcomeLedgerV2
from gateway.tee.scoring_executor import ScoringExecutionResult
from leadpoet_canonical.attested_v2 import (
    RECEIPT_GRAPH_SCHEMA_VERSION,
    build_transport_attempt,
    sha256_json,
    transport_root,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
    build_allocation_settlement_frontier_bootstrap_v2,
    frontier_bootstrap_artifact_hashes_v2,
)
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION,
)
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    weight_config_hash,
)
from leadpoet_canonical.weight_authority_v2 import (
    gateway_weight_input_value_documents_v2,
)
def test_reward_ancestry_accepts_checkpointed_source_add_parents() -> None:
    provenance_parent_hash = "sha256:" + "1" * 64
    precheck_doc = {
        "precheck_status": "provenance_precheck_passed",
        "reasons": ["provenance_reference_backed"],
    }
    provenance_result = {
        "schema_version": "leadpoet.source_add_provenance_result.v2",
        "submission_id": "source_add_submission:1234567890abcdef",
        "precheck_status": "provenance_precheck_passed",
        "reasons": list(precheck_doc["reasons"]),
        "precheck_doc": precheck_doc,
    }
    context = ExecutionContextV2(
        job_id="reward:test",
        purpose="research_lab.reward_decision.v2",
        epoch_id=100,
        parent_receipt_hashes=(provenance_parent_hash,),
        external_ancestry_proofs=[
            {
                "certificate": {
                    "claim": {
                        "lineage_id": "gateway:test",
                        "output_root_receipt_hash": provenance_parent_hash,
                    }
                },
                "disclosed_boot_identities": [],
                "disclosed_receipts": [
                    {
                        "receipt_hash": provenance_parent_hash,
                        "purpose": "research_lab.source_add_provenance.v2",
                        "output_root": sha256_json(provenance_result),
                    }
                ],
            },
        ],
    )

    CoordinatorExecutorV2._validate_reward_ancestry(
        {
            "decision_kind": "source_add_leg1",
            "decision_payload": {
                "provenance_result": provenance_result,
            },
        },
        context,
    )


def test_reward_ancestry_rejects_checkpointed_parent_output_mismatch() -> None:
    provenance_parent_hash = "sha256:" + "1" * 64
    precheck_doc = {
        "precheck_status": "provenance_precheck_passed",
        "reasons": ["provenance_reference_backed"],
    }
    provenance_result = {
        "schema_version": "leadpoet.source_add_provenance_result.v2",
        "submission_id": "source_add_submission:1234567890abcdef",
        "precheck_status": "provenance_precheck_passed",
        "reasons": list(precheck_doc["reasons"]),
        "precheck_doc": precheck_doc,
    }
    context = ExecutionContextV2(
        job_id="reward:test",
        purpose="research_lab.reward_decision.v2",
        epoch_id=100,
        parent_receipt_hashes=(provenance_parent_hash,),
        external_ancestry_proofs=[
            {
                "certificate": {
                    "claim": {
                        "lineage_id": "gateway:test",
                        "output_root_receipt_hash": provenance_parent_hash,
                    }
                },
                "disclosed_boot_identities": [],
                "disclosed_receipts": [
                    {
                        "receipt_hash": provenance_parent_hash,
                        "purpose": "research_lab.source_add_provenance.v2",
                        "output_root": "sha256:" + "2" * 64,
                    }
                ],
            },
        ],
    )

    with pytest.raises(ValueError, match="parent output differs"):
        CoordinatorExecutorV2._validate_reward_ancestry(
            {
                "decision_kind": "source_add_leg1",
                "decision_payload": {
                    "provenance_result": provenance_result,
                },
            },
            context,
        )

def _artifact_transport_attempts(
    artifact_id,
    job_id,
    sequence=(("GET", "transport_failure"), ("GET", "ok"), ("HEAD", "ok")),
):
    attempts = []
    for ordinal, (method, outcome) in enumerate(sequence):
        successful = outcome == "ok"
        attempts.append(
            build_transport_attempt(
                request_id=("%032x" % (ordinal + 1)),
                logical_operation_id="%s:%s" % (artifact_id, method.lower()),
                job_id=job_id,
                purpose="leadpoet.artifact_persistence.v2",
                provider_id="aws_s3_object_lock",
                attempt_number=ordinal,
                method=method,
                destination_host="immutable.example.s3.us-east-1.amazonaws.com",
                destination_port=443,
                path_hash="sha256:" + "1" * 64,
                nonsecret_headers_hash="sha256:" + "2" * 64,
                body_hash="sha256:" + "3" * 64,
                credential_ref_hash="sha256:" + "4" * 64,
                retry_policy_hash="sha256:" + "5" * 64,
                timeout_ms=30000,
                started_at="2026-07-10T12:00:00Z",
                terminal_status=(
                    "authenticated_response"
                    if successful
                    else "transport_failure"
                ),
                http_status=200 if successful else None,
                response_hash="sha256:" + "6" * 64 if successful else None,
                request_artifact_hash="sha256:" + "8" * 64,
                response_artifact_hash=(
                    "sha256:" + "6" * 64 if successful else None
                ),
                tls_peer_chain_hash=(
                    "sha256:" + "7" * 64 if successful else None
                ),
                tls_protocol="TLSv1.3" if successful else None,
                failure_code=None if successful else "unexpected_eof",
                completed_at="2026-07-10T12:00:01Z",
            )
        )
    return attempts


def _artifact_evidence(artifact_id, plaintext_hash, attempts):
    return {
        "artifact_id": artifact_id,
        "plaintext_hash": plaintext_hash,
        "ciphertext_hash": "sha256:" + "e" * 64,
        "artifact_ref": "s3://immutable/artifact.json",
        "storage_document_hash": "sha256:" + "f" * 64,
        "encryption_context_hash": "sha256:" + "3" * 64,
        "object_lock_mode": "COMPLIANCE",
        "retain_until": "2027-07-10T12:00:00Z",
        "transport_root": transport_root(attempts),
        "transport_attempts": attempts,
        "persisted": True,
    }


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
async def test_coordinator_ancestry_bootstrap_requires_canonical_legacy_roots():
    root_hash = "sha256:" + "1" * 64
    context = ExecutionContextV2(
        job_id="ancestry-bootstrap:test",
        purpose="research_lab.ancestry_checkpoint_bootstrap.v2",
        epoch_id=100,
        external_receipt_graphs=[
            {
                "schema_version": RECEIPT_GRAPH_SCHEMA_VERSION,
                "root_receipt_hash": root_hash,
                "receipts": [{"receipt_hash": root_hash}],
            }
        ],
    )
    result = await CoordinatorExecutorV2()(
        OP_ANCESTRY_CHECKPOINT_BOOTSTRAP_V2,
        {
            "schema_version": (
                ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION
            ),
            "selected_root_receipt_hashes": [root_hash],
        },
        context,
    )
    assert result.ancestry_checkpoint_bootstrap is True
    assert result.output["selected_root_receipt_hashes"] == [root_hash]

    context.external_receipt_graphs[0]["schema_version"] = (
        "leadpoet.attested_checkpointed_receipt_graph.v3"
    )
    with pytest.raises(ValueError, match="requires legacy full graphs"):
        await CoordinatorExecutorV2()(
            OP_ANCESTRY_CHECKPOINT_BOOTSTRAP_V2,
            {
                "schema_version": (
                    ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION
                ),
                "selected_root_receipt_hashes": [root_hash],
            },
            context,
        )


@pytest.mark.asyncio
async def test_coordinator_allocation_frontier_bootstrap_is_measured_and_bound():
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    document = build_allocation_settlement_frontier_bootstrap_v2(
        netuid=71,
        bootstrap_epoch=101,
        allocation_source_receipt_hash="sha256:" + "1" * 64,
        source_state_hash="sha256:" + "2" * 64,
        frontier=frontier,
    )
    observed = []

    def resolve(payload, context):
        observed.append((payload, context.purpose))
        return document

    result = await CoordinatorExecutorV2(
        allocation_frontier_bootstrap_resolver=resolve,
    )(
        ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
        {"request": "measured"},
        ExecutionContextV2(
            job_id="allocation-frontier-bootstrap:101",
            purpose=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
            epoch_id=101,
        ),
    )

    assert observed == [
        (
            {"request": "measured"},
            ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
        )
    ]
    assert result.output == document
    assert result.artifact_hashes == frontier_bootstrap_artifact_hashes_v2(
        document
    )


@pytest.mark.asyncio
async def test_coordinator_chain_realized_authorities_are_measured_and_bound():
    observation = {
        "schema_version": "leadpoet.chain_realized_weight_observation.v1",
        "epoch_id": 100,
    }
    settlement = {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_epoch_settlement.v1"
        ),
        "netuid": 71,
        "epoch_id": 100,
    }
    settlement_hash = "sha256:" + "1" * 64
    credit_hash = "sha256:" + "2" * 64
    calls = []

    def observe(payload, context):
        calls.append(("observe", dict(payload), context.job_id))
        return observation

    def settle(payload, context):
        calls.append(("settle", dict(payload), context.job_id))
        return {
            "settlement_doc": settlement,
            "settlement_hash": settlement_hash,
            "credits": [{"credit_hash": credit_hash}],
        }

    executor = CoordinatorExecutorV2(
        chain_weight_observation_resolver=observe,
        chain_realized_settlement_resolver=settle,
    )
    context = ExecutionContextV2(
        job_id="chain-realized:test",
        purpose="research_lab.chain_weight_observation.v1",
        epoch_id=100,
    )
    observation_result = await executor(
        OP_OBSERVE_CHAIN_REALIZED_WEIGHTS_V1,
        {"epoch_id": 100},
        context,
    )
    settlement_result = await executor(
        OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
        {"epoch_id": 100},
        ExecutionContextV2(
            job_id="chain-realized:settlement",
            purpose="research_lab.chain_realized_epoch_settlement.v1",
            epoch_id=100,
        ),
    )

    assert observation_result.output == observation
    assert observation_result.artifact_hashes == (sha256_json(observation),)
    assert settlement_result.output["settlement_hash"] == settlement_hash
    assert coordinator_receipt_output_v2(
        OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
        settlement_result.output,
    ) == settlement
    assert settlement_result.receipt_output == settlement
    assert sha256_json(settlement_result.receipt_output) == sha256_json(
        settlement
    )
    assert settlement_result.artifact_hashes == (
        settlement_hash,
        credit_hash,
    )
    assert [item[0] for item in calls] == ["observe", "settle"]


@pytest.mark.asyncio
async def test_coordinator_rejects_operation_outside_measured_authority():
    with pytest.raises(ValueError, match="unsupported"):
        await CoordinatorExecutorV2()(
            "promotion_improvement",
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
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    authority = {
        "allocation": allocation,
        "allocation_inputs": {"epoch_id": 100},
        "source_state": {
            "epoch_id": 100,
            "settlement_frontier": frontier,
        },
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
        frontier["frontier_hash"],
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
    job_id = "artifact:test"
    attempts = _artifact_transport_attempts(artifact_id, job_id)
    evidence = _artifact_evidence(artifact_id, plaintext_hash, attempts)
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
            job_id=job_id,
            purpose="leadpoet.artifact_persistence.v2",
            epoch_id=1,
        ),
    )
    assert result.output["source_receipt_hash"] == "sha256:" + "2" * 64
    assert result.output["artifacts"][0]["artifact_id"] == artifact_id
    assert list(result.transport_attempts) == attempts


@pytest.mark.asyncio
async def test_coordinator_rejects_artifact_transport_root_mismatch():
    artifact_id = "sha256:" + "a" * 64
    plaintext_hash = "sha256:" + "b" * 64
    attempts = _artifact_transport_attempts(artifact_id, "artifact:test")
    evidence = {
        **_artifact_evidence(artifact_id, plaintext_hash, attempts),
        "transport_root": "sha256:" + "9" * 64,
    }
    executor = CoordinatorExecutorV2(
        artifact_evidence_supplier=lambda _ids, _context: [evidence]
    )
    with pytest.raises(ValueError, match="transport root differs"):
        await executor(
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


@pytest.mark.asyncio
async def test_coordinator_rejects_artifact_retry_evidence_over_budget():
    artifact_id = "sha256:" + "a" * 64
    plaintext_hash = "sha256:" + "b" * 64
    sequence = tuple(
        ("GET", "transport_failure")
        for _ in range(ARTIFACT_PERSISTENCE_MAX_ATTEMPTS_PER_METHOD)
    ) + (("GET", "ok"), ("HEAD", "ok"))
    attempts = _artifact_transport_attempts(
        artifact_id,
        "artifact:test",
        sequence=sequence,
    )
    evidence = _artifact_evidence(artifact_id, plaintext_hash, attempts)
    executor = CoordinatorExecutorV2(
        artifact_evidence_supplier=lambda _ids, _context: [evidence]
    )
    with pytest.raises(ValueError, match="transport evidence is incomplete"):
        await executor(
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
