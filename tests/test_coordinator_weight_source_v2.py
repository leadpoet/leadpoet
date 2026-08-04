from __future__ import annotations

import base64
import copy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.coordinator_weight_source_v2 import (
    CoordinatorWeightSourceV2,
    CoordinatorWeightSourceV2Error,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_receipt_graph,
    create_boot_identity,
    create_signed_execution_receipt,
    sha256_json,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    DISABLED_LEADERBOARD_WINDOW_V1,
    GATEWAY_MEASURED_SNAPSHOT_AUTHORITY_MODE_V2,
)
from leadpoet_canonical.sourcing_history_v2 import (
    build_sourcing_decision_v2,
    build_sourcing_epoch_v2,
)
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    weight_config_hash,
)
from leadpoet_canonical.weight_authority_v2 import (
    WEIGHT_INPUT_PURPOSES,
    gateway_weight_input_value_documents_v2,
)


HASH = "sha256:" + "a" * 64
FULFILLMENT_POOL = 1.0 - 0.2 - 0.0 - 0.095


class FakeReader:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def read(self, *, policy_id, parameters, **_kwargs):
        self.calls.append((policy_id, dict(parameters)))
        return [dict(row) for row in self.rows.get(policy_id, [])]


def _snapshot(**overrides):
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
            "allocation_hash": HASH,
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
        "fulfillment_share": FULFILLMENT_POOL,
        "fulfillment_rows": [
            {"hotkey": "miner", "share": FULFILLMENT_POOL}
        ],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125000,
        "min_total_rep_for_distribution": 100,
    }
    value.update(overrides)
    value["config_hash"] = weight_config_hash(value)
    return value


def _payload(category, snapshot=None, **overrides):
    value = {
        "category": category,
        "calculation_snapshot": snapshot or _snapshot(),
        "gateway_authority_event_hash": "sha256:" + "2" * 64,
        "allocation_receipt": None,
        "leaderboard_window_start": "2026-07-03T20:00:00Z",
        "leaderboard_window_end": "2026-07-10T20:00:00Z",
    }
    value.update(overrides)
    return value


def _context(purpose, parents=()):
    return ExecutionContextV2(
        job_id="weight-input:test:100",
        purpose=purpose,
        epoch_id=100,
        parent_receipt_hashes=tuple(parents),
    )


def _sourcing_epoch_receipt(source_doc):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    body = build_execution_receipt_body(
        role="gateway_scoring",
        purpose="qualification.sourcing_epoch.v2",
        job_id="qualification-sourcing-epoch:%d" % source_doc["epoch_id"],
        epoch_id=source_doc["epoch_id"],
        sequence=0,
        commit_sha="b" * 40,
        pcr0="c" * 96,
        build_manifest_hash="sha256:" + "d" * 64,
        dependency_lock_hash="sha256:" + "e" * 64,
        config_hash="sha256:" + "f" * 64,
        boot_identity_hash="sha256:" + "1" * 64,
        input_root=source_doc["decision_root"],
        output_root=sha256_json(source_doc),
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=EMPTY_ARTIFACT_ROOT,
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at="2026-07-10T20:00:00Z",
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )


def _weight_input_receipt(category, document):
    """Build one valid direct coordinator receipt for a measured document."""

    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    role, purpose = WEIGHT_INPUT_PURPOSES[category]
    body = build_execution_receipt_body(
        role=role,
        purpose=purpose,
        job_id="weight-input:%s:%d" % (category, document["epoch_id"]),
        epoch_id=document["epoch_id"],
        sequence=0,
        commit_sha="b" * 40,
        pcr0="c" * 96,
        build_manifest_hash="sha256:" + "d" * 64,
        dependency_lock_hash="sha256:" + "e" * 64,
        config_hash="sha256:" + "f" * 64,
        boot_identity_hash="sha256:" + "1" * 64,
        input_root=sha256_json(document),
        output_root=sha256_json(document),
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=EMPTY_ARTIFACT_ROOT,
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at="2026-07-10T20:00:00Z",
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )


def _durable_bans_graph(
    bans_document,
    *,
    source_purpose="research_lab.ban_input.v2",
    source_epoch=100,
    terminal_epoch=None,
    source_count=1,
    source_output=None,
    persistence_output=None,
    persistence_output_mutator=None,
    extra_parent=False,
):
    """Build canonical durable ban ancestry without bypassing graph checks."""

    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_coordinator",
            physical_role="gateway_coordinator",
            commit_sha="b" * 40,
            pcr0="c" * 96,
            build_manifest_hash="sha256:" + "d" * 64,
            dependency_lock_hash="sha256:" + "e" * 64,
            config_hash="sha256:" + "f" * 64,
            boot_nonce="1" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="2" * 64,
            transport_certificate_hash="sha256:" + "3" * 64,
            attestation_user_data_hash="sha256:" + "4" * 64,
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"bans-graph").decode(),
    )

    def receipt(*, purpose, job_id, epoch_id, output_root, parents=()):
        return create_signed_execution_receipt(
            body=build_execution_receipt_body(
                role="gateway_coordinator",
                purpose=purpose,
                job_id=job_id,
                epoch_id=epoch_id,
                sequence=0,
                commit_sha="b" * 40,
                pcr0="c" * 96,
                build_manifest_hash="sha256:" + "d" * 64,
                dependency_lock_hash="sha256:" + "e" * 64,
                config_hash="sha256:" + "f" * 64,
                boot_identity_hash=boot["boot_identity_hash"],
                input_root=sha256_json({"job_id": job_id}),
                output_root=output_root,
                transport_root_hash=EMPTY_TRANSPORT_ROOT,
                host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
                artifact_root=EMPTY_ARTIFACT_ROOT,
                parent_receipt_hashes=parents,
                status="succeeded",
                failure_code=None,
                issued_at="2026-07-10T20:00:00Z",
            ),
            enclave_pubkey=pubkey,
            sign_digest=key.sign,
        )

    sources = [
        receipt(
            purpose=source_purpose,
            job_id="weight-input-bans:%d" % index,
            epoch_id=source_epoch,
            output_root=(
                sha256_json(source_output)
                if source_output is not None
                else sha256_json(bans_document)
            ),
        )
        for index in range(source_count)
    ]
    if persistence_output is None:
        artifacts = [
            {
                "artifact_id": "sha256:" + "7" * 64,
                "plaintext_hash": sha256_json(bans_document["value"]),
                "ciphertext_hash": "sha256:" + "8" * 64,
                "artifact_ref": "s3://immutable/bans.json",
                "storage_document_hash": "sha256:" + "9" * 64,
                "encryption_context_hash": "sha256:" + "a" * 64,
                "object_lock_mode": "COMPLIANCE",
                "retain_until": "2027-07-10T20:00:00Z",
                "transport_root": "sha256:" + "b" * 64,
            }
        ]
        persistence_output = {
            "source_receipt_hash": sources[0]["receipt_hash"],
            "artifacts": artifacts,
            "artifact_set_root": sha256_json(artifacts),
        }
    if persistence_output_mutator is not None:
        persistence_output = persistence_output_mutator(
            copy.deepcopy(persistence_output)
        )
    extra = (
        receipt(
            purpose="research_lab.leaderboard_input.v2",
            job_id="weight-input-unrelated",
            epoch_id=source_epoch,
            output_root="sha256:" + "c" * 64,
        )
        if extra_parent
        else None
    )
    terminal = receipt(
        purpose="leadpoet.artifact_persistence.v2",
        job_id="weight-input-bans-persistence",
        epoch_id=(source_epoch if terminal_epoch is None else terminal_epoch),
        output_root=sha256_json(persistence_output),
        parents=[
            *[item["receipt_hash"] for item in sources],
            *([extra["receipt_hash"]] if extra is not None else []),
        ],
    )
    return (
        build_receipt_graph(
            root_receipt_hash=terminal["receipt_hash"],
            boot_identities=[boot],
            receipts=[*sources, *([extra] if extra is not None else []), terminal],
            transport_attempts=[],
        ),
        sources,
        terminal,
        persistence_output,
    )


def test_bans_are_reconstructed_from_authenticated_rows_not_host_snapshot():
    reader = FakeReader({"banned_hotkeys": [{"hotkey": "5B"}, {"hotkey": "5A"}]})
    source = CoordinatorWeightSourceV2(reader)
    snapshot = _snapshot(
        banned_hotkeys=["5A", "5B"],
        rolling_scores=[],
    )
    document = source.resolve(
        payload=_payload("bans", snapshot=snapshot),
        context=_context("research_lab.ban_input.v2"),
    )
    assert document["value"] == {
        "banned_hotkeys": ["5A", "5B"],
        "banned_lookup_ok": True,
    }

    forged = copy.deepcopy(snapshot)
    forged["banned_hotkeys"] = []
    with pytest.raises(CoordinatorWeightSourceV2Error, match="differs"):
        source.resolve(
            payload=_payload("bans", snapshot=forged),
            context=_context("research_lab.ban_input.v2"),
        )


def test_fulfillment_rows_and_pool_cap_match_existing_formula_exactly():
    reader = FakeReader(
        {
            "fulfillment_active_rewards": [
                {
                    "miner_hotkey": "miner",
                    "reward_pct": 0.8,
                    "reward_expires_epoch": 101,
                }
            ]
        }
    )
    source = CoordinatorWeightSourceV2(reader)
    document = source.resolve(
        payload=_payload("fulfillment_rewards"),
        context=_context("research_lab.fulfillment_input.v2"),
    )
    assert document["value"]["fulfillment_share"] == FULFILLMENT_POOL
    assert document["value"]["fulfillment_rows"] == [
        {"hotkey": "miner", "share": FULFILLMENT_POOL}
    ]


def test_leaderboard_reconstructs_wins_tiebreak_and_ban_filter():
    reader = FakeReader(
        {
            "fulfillment_leaderboard_winners": [
                {"miner_hotkey": "miner", "reward_pct": 0.5},
                {"miner_hotkey": "miner", "reward_pct": 0.2},
                {"miner_hotkey": "banned", "reward_pct": 1.0},
            ],
            "banned_hotkeys": [{"hotkey": "banned"}],
        }
    )
    snapshot = _snapshot(
        leaderboard_entries=[{"miner_hotkey": "miner", "wins": 2}]
    )
    source = CoordinatorWeightSourceV2(reader)
    document = source.resolve(
        payload=_payload("leaderboard", snapshot=snapshot),
        context=_context("research_lab.leaderboard_input.v2"),
    )
    assert document["value"]["leaderboard_entries"] == [
        {"miner_hotkey": "miner", "wins": 2}
    ]


def test_disabled_leaderboard_window_commits_empty_input_without_db_read():
    reader = FakeReader(
        {
            "fulfillment_leaderboard_winners": [
                {"miner_hotkey": "miner", "reward_pct": 0.5},
            ],
            "banned_hotkeys": [],
        }
    )
    source = CoordinatorWeightSourceV2(reader)
    document = source.resolve(
        payload=_payload(
            "leaderboard",
            leaderboard_window_start=DISABLED_LEADERBOARD_WINDOW_V1,
            leaderboard_window_end=DISABLED_LEADERBOARD_WINDOW_V1,
        ),
        context=_context("research_lab.leaderboard_input.v2"),
    )
    assert document["value"] == {
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [],
        "leaderboard_fetch_ok": True,
    }
    assert reader.calls == []


def test_disabled_leaderboard_window_must_be_complete():
    source = CoordinatorWeightSourceV2(FakeReader({}))
    with pytest.raises(
        CoordinatorWeightSourceV2Error,
        match="disabled leaderboard window is incomplete",
    ):
        source.resolve(
            payload=_payload(
                "leaderboard",
                leaderboard_window_start=DISABLED_LEADERBOARD_WINDOW_V1,
            ),
            context=_context("research_lab.leaderboard_input.v2"),
        )


def test_disabled_fulfillment_commits_empty_leaderboard_without_paying_rows():
    reader = FakeReader(
        {
            "fulfillment_leaderboard_winners": [
                {"miner_hotkey": "miner", "reward_pct": 0.5},
            ],
            "banned_hotkeys": [],
        }
    )
    snapshot = _snapshot(
        ff_enabled=False,
        fulfillment_share=0.0,
        fulfillment_rows=[],
        leaderboard_entries=[],
    )
    source = CoordinatorWeightSourceV2(reader)
    document = source.resolve(
        payload=_payload("leaderboard", snapshot=snapshot),
        context=_context("research_lab.leaderboard_input.v2"),
    )
    assert document["value"]["leaderboard_entries"] == []
    assert reader.calls == []


def test_sourcing_history_is_rebuilt_only_from_signed_epoch_receipts():
    decisions = [
        build_sourcing_decision_v2(
            epoch_id=99,
            sequence=sequence,
            lead_id_hash=sha256_json({"lead": sequence}),
            miner_hotkey="miner",
            decision="approve",
            rep_score=score,
            is_icp_multiplier=0,
        )
        for sequence, score in ((0, 4), (1, 6))
    ]
    source_doc = build_sourcing_epoch_v2(epoch_id=99, decisions=decisions)
    receipt = _sourcing_epoch_receipt(source_doc)
    reader = FakeReader(
        {
            "sourcing_epoch_inputs": [
                {
                    "epoch_id": 99,
                    "epoch_hash": source_doc["epoch_hash"],
                    "receipt_hash": receipt["receipt_hash"],
                    "source_doc": source_doc,
                    "receipt_doc": receipt,
                },
            ],
            "attested_receipt_by_hash": [{"receipt_doc": receipt}],
        }
    )
    snapshot = _snapshot(
        rolling_lead_count=2,
        rolling_scores=[{"hotkey": "miner", "score": 10}],
    )
    document = CoordinatorWeightSourceV2(reader).resolve(
        payload=_payload("sourcing_history", snapshot=snapshot),
        context=_context(
            "research_lab.sourcing_input.v2",
            parents=(receipt["receipt_hash"],),
        ),
    )
    assert document["value"] == {
        "rolling_lead_count": 2,
        "rolling_scores": [{"hotkey": "miner", "score": 10}],
    }


def test_signed_sourcing_replays_bans_only_from_one_bound_bans_parent():
    decision = build_sourcing_decision_v2(
        epoch_id=99,
        sequence=0,
        lead_id_hash=sha256_json({"lead": "banned"}),
        miner_hotkey="banned",
        decision="approve",
        rep_score=4,
        is_icp_multiplier=0,
    )
    source_doc = build_sourcing_epoch_v2(epoch_id=99, decisions=[decision])
    source_receipt = _sourcing_epoch_receipt(source_doc)
    snapshot = _snapshot(
        rolling_lead_count=1,
        rolling_scores=[{"hotkey": "banned", "score": 4}],
    )
    documents = gateway_weight_input_value_documents_v2(
        calculation_snapshot=snapshot,
        gateway_authority_event_hash="sha256:" + "2" * 64,
    )
    bans_document = {
        **documents["bans"],
        "value": {
            "banned_hotkeys": ["banned"],
            "banned_lookup_ok": True,
        },
    }
    reader = FakeReader(
        {
            "sourcing_epoch_inputs": [
                {
                    "epoch_id": 99,
                    "epoch_hash": source_doc["epoch_hash"],
                    "receipt_hash": source_receipt["receipt_hash"],
                    "source_doc": source_doc,
                    "receipt_doc": source_receipt,
                },
            ],
            "attested_receipt_by_hash": [{"receipt_doc": source_receipt}],
        }
    )
    def resolve_with_bans_graph(graph, parent_hash, persistence_output):
        context = _context(
            "research_lab.sourcing_input.v2",
            parents=(source_receipt["receipt_hash"], parent_hash),
        )
        context.external_receipt_graphs = [graph]
        return CoordinatorWeightSourceV2(reader).resolve(
            payload=_payload(
                "sourcing_history",
                snapshot=snapshot,
                snapshot_authority_mode=(
                    GATEWAY_MEASURED_SNAPSHOT_AUTHORITY_MODE_V2
                ),
                bans_document=bans_document,
                bans_persistence_output=persistence_output,
            ),
            context=context,
        )

    (
        persisted_graph,
        source_receipts,
        terminal_receipt,
        persistence_output,
    ) = _durable_bans_graph(bans_document)

    document = resolve_with_bans_graph(
        persisted_graph,
        terminal_receipt["receipt_hash"],
        persistence_output,
    )

    assert document["value"] == {
        "rolling_lead_count": 1,
        "rolling_scores": [{"hotkey": "banned", "score": -100000}],
    }

    direct_graph = build_receipt_graph(
        root_receipt_hash=source_receipts[0]["receipt_hash"],
        boot_identities=persisted_graph["boot_identities"],
        receipts=[source_receipts[0]],
        transport_attempts=[],
    )
    duplicated_graph, _, duplicated_terminal, duplicated_output = _durable_bans_graph(
        bans_document, source_count=2
    )
    forged_graph = copy.deepcopy(persisted_graph)
    forged_graph["receipts"][0]["enclave_signature"] = "0" * 128
    wrong_purpose_graph, _, wrong_purpose_terminal, wrong_purpose_output = (
        _durable_bans_graph(
            bans_document,
            source_purpose="research_lab.leaderboard_input.v2",
        )
    )
    wrong_epoch_graph, _, wrong_epoch_terminal, wrong_epoch_output = (
        _durable_bans_graph(bans_document, source_epoch=99)
    )
    wrong_output_graph, _, wrong_output_terminal, wrong_output = _durable_bans_graph(
        bans_document,
        source_output={"not": "the-bans-document"},
    )

    def tamper_persisted_bans(output):
        output["artifacts"][0]["plaintext_hash"] = sha256_json(
            {"not": "the-bans-value"}
        )
        output["artifact_set_root"] = sha256_json(output["artifacts"])
        return output

    terminal_output_graph, _, terminal_output, terminal_output_evidence = (
        _durable_bans_graph(
            bans_document,
            persistence_output_mutator=tamper_persisted_bans,
        )
    )
    extra_parent_graph, _, extra_parent_terminal, extra_parent_output = (
        _durable_bans_graph(bans_document, extra_parent=True)
    )

    cases = (
        ("missing", None, None, persistence_output),
        (
            "direct ephemeral",
            direct_graph,
            source_receipts[0]["receipt_hash"],
            persistence_output,
        ),
        (
            "duplicated",
            duplicated_graph,
            duplicated_terminal["receipt_hash"],
            duplicated_output,
        ),
        (
            "forged",
            forged_graph,
            terminal_receipt["receipt_hash"],
            persistence_output,
        ),
        (
            "wrong purpose",
            wrong_purpose_graph,
            wrong_purpose_terminal["receipt_hash"],
            wrong_purpose_output,
        ),
        (
            "wrong epoch",
            wrong_epoch_graph,
            wrong_epoch_terminal["receipt_hash"],
            wrong_epoch_output,
        ),
        (
            "output root mismatch",
            wrong_output_graph,
            wrong_output_terminal["receipt_hash"],
            wrong_output,
        ),
        (
            "terminal output mismatch",
            terminal_output_graph,
            terminal_output["receipt_hash"],
            terminal_output_evidence,
        ),
        (
            "extra terminal parent",
            extra_parent_graph,
            extra_parent_terminal["receipt_hash"],
            extra_parent_output,
        ),
    )
    for _label, graph, parent_hash, case_persistence_output in cases:
        if graph is None:
            context = _context(
                "research_lab.sourcing_input.v2",
                parents=(source_receipt["receipt_hash"],),
            )
            context.external_receipt_graphs = []
            action = lambda: CoordinatorWeightSourceV2(reader).resolve(
                payload=_payload(
                    "sourcing_history",
                    snapshot=snapshot,
                    snapshot_authority_mode=(
                        GATEWAY_MEASURED_SNAPSHOT_AUTHORITY_MODE_V2
                    ),
                    bans_document=bans_document,
                    bans_persistence_output=case_persistence_output,
                ),
                context=context,
            )
        else:
            action = lambda graph=graph, parent_hash=parent_hash, case_persistence_output=case_persistence_output: resolve_with_bans_graph(
                graph,
                parent_hash,
                case_persistence_output,
            )
        with pytest.raises(CoordinatorWeightSourceV2Error):
            action()


def test_sourcing_history_rejects_undeclared_or_modified_epoch_receipt():
    source_doc = build_sourcing_epoch_v2(epoch_id=99, decisions=[])
    receipt = _sourcing_epoch_receipt(source_doc)
    row = {
        "epoch_id": 99,
        "epoch_hash": source_doc["epoch_hash"],
        "receipt_hash": receipt["receipt_hash"],
        "source_doc": source_doc,
        "receipt_doc": receipt,
    }
    source = CoordinatorWeightSourceV2(
        FakeReader(
            {
                "sourcing_epoch_inputs": [row],
                "attested_receipt_by_hash": [{"receipt_doc": receipt}],
            }
        )
    )
    with pytest.raises(CoordinatorWeightSourceV2Error, match="declared source"):
        source.resolve(
            payload=_payload("sourcing_history"),
            context=_context("research_lab.sourcing_input.v2"),
        )

    tampered = copy.deepcopy(row)
    tampered["source_doc"]["approved_lead_count"] = 1
    with pytest.raises(CoordinatorWeightSourceV2Error, match="document is invalid"):
        CoordinatorWeightSourceV2(
            FakeReader(
                {
                    "sourcing_epoch_inputs": [tampered],
                    "attested_receipt_by_hash": [{"receipt_doc": receipt}],
                }
            )
        ).resolve(
            payload=_payload("sourcing_history"),
            context=_context(
                "research_lab.sourcing_input.v2",
                parents=(receipt["receipt_hash"],),
            ),
        )


def test_allocation_projection_requires_a_signed_declared_parent():
    source = CoordinatorWeightSourceV2(FakeReader({}))
    with pytest.raises(CoordinatorWeightSourceV2Error, match="receipt is missing"):
        source.resolve(
            payload=_payload("research_lab_allocation"),
            context=_context("research_lab.allocation.v2"),
        )


def test_anomaly_hash_is_derived_only_from_signed_upstream_documents():
    snapshot = _snapshot()
    documents = gateway_weight_input_value_documents_v2(
        calculation_snapshot=snapshot,
        gateway_authority_event_hash="sha256:" + "2" * 64,
    )
    categories = (
        "research_lab_allocation",
        "fulfillment_rewards",
        "leaderboard",
        "bans",
        "sourcing_history",
    )
    upstream = {category: documents[category] for category in categories}
    graphs = []
    parent_hashes = []
    for index, category in enumerate(categories, start=1):
        receipt_hash = "sha256:" + ("%x" % index) * 64
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        graphs.append(
            {
                "root_receipt_hash": receipt_hash,
                "receipts": [
                    {
                        "receipt_hash": receipt_hash,
                        "role": role,
                        "purpose": purpose,
                        "output_root": sha256_json(upstream[category]),
                    }
                ],
            }
        )
        parent_hashes.append(receipt_hash)
    context = _context(
        "research_lab.anomaly_adjustment_input.v2",
        parents=parent_hashes,
    )
    context.external_receipt_graphs = graphs
    source = CoordinatorWeightSourceV2(FakeReader({}))

    result = source.resolve(
        payload=_payload(
            "anomaly_adjustments",
            snapshot=snapshot,
            upstream_documents=upstream,
        ),
        context=context,
    )
    assert result == documents["anomaly_adjustments"]

    persistence_graphs = []
    persistence_parent_hashes = []
    for index, graph in enumerate(graphs, start=10):
        source_receipt = graph["receipts"][0]
        persistence_hash = "sha256:" + ("%x" % index) * 64
        persistence_graphs.append(
            {
                "root_receipt_hash": persistence_hash,
                "receipts": [
                    source_receipt,
                    {
                        "receipt_hash": persistence_hash,
                        "role": "gateway_coordinator",
                        "purpose": "leadpoet.artifact_persistence.v2",
                        "parent_receipt_hashes": [
                            source_receipt["receipt_hash"]
                        ],
                        "output_root": HASH,
                    },
                ],
            }
        )
        persistence_parent_hashes.append(persistence_hash)
    persistence_context = _context(
        "research_lab.anomaly_adjustment_input.v2",
        parents=persistence_parent_hashes,
    )
    persistence_context.external_receipt_graphs = persistence_graphs
    assert source.resolve(
        payload=_payload(
            "anomaly_adjustments",
            snapshot=snapshot,
            upstream_documents=upstream,
        ),
        context=persistence_context,
    ) == documents["anomaly_adjustments"]

    detached = copy.deepcopy(persistence_graphs)
    detached[0]["receipts"][1]["parent_receipt_hashes"] = []
    detached_context = _context(
        "research_lab.anomaly_adjustment_input.v2",
        parents=persistence_parent_hashes,
    )
    detached_context.external_receipt_graphs = detached
    with pytest.raises(
        CoordinatorWeightSourceV2Error,
        match="source receipt set is incomplete",
    ):
        source.resolve(
            payload=_payload(
                "anomaly_adjustments",
                snapshot=snapshot,
                upstream_documents=upstream,
            ),
            context=detached_context,
        )

    tampered = copy.deepcopy(upstream)
    tampered["bans"]["value"]["banned_hotkeys"] = ["forged"]
    with pytest.raises(CoordinatorWeightSourceV2Error, match="differs from its receipt"):
        source.resolve(
            payload=_payload(
                "anomaly_adjustments",
                snapshot=snapshot,
                upstream_documents=tampered,
            ),
            context=context,
        )
