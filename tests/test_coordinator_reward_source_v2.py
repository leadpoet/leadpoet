from types import SimpleNamespace

import pytest

from gateway.tee.coordinator_reward_source_v2 import (
    CoordinatorRewardSourceV2,
    CoordinatorRewardSourceV2Error,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import sha256_json


SUBMISSION_ID = "source_add_submission:1234567890abcdef"
HASH = "sha256:" + "a" * 64
PROVENANCE_RECEIPT_HASH = "sha256:" + "b" * 64


class FakeReader:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def read(self, *, policy_id, parameters, **_kwargs):
        self.calls.append((policy_id, dict(parameters)))
        return [dict(item) for item in self.rows.get(policy_id, ())]


class FakeChain:
    def read_finalized_metagraph(self, *, netuid, context):
        assert netuid == 71
        return {
            "header": {"block": context.epoch_id * 360},
            "workflow_epoch_id": context.epoch_id,
        }


class MissingWorkflowEpochChain(FakeChain):
    def read_finalized_metagraph(self, *, netuid, context):
        result = super().read_finalized_metagraph(
            netuid=netuid,
            context=context,
        )
        result.pop("workflow_epoch_id")
        return result


def _config():
    return SimpleNamespace(
        netuid=71,
        source_add_leg1_alpha_percent=0.2,
        source_add_leg1_max_per_utc_day=50,
        lab_reward_epochs=20,
    )


def _context(
    *,
    with_leg1_parent=False,
    with_leg1_proof=False,
):
    provenance_receipt = {
        "receipt_hash": PROVENANCE_RECEIPT_HASH,
        "role": "gateway_coordinator",
        "purpose": "research_lab.source_add_provenance.v2",
        "status": "succeeded",
        "output_root": sha256_json(_provenance_result()),
    }
    return ExecutionContextV2(
        job_id="reward:test",
        purpose="research_lab.reward_decision.v2",
        epoch_id=100,
        parent_receipt_hashes=(PROVENANCE_RECEIPT_HASH,) if with_leg1_parent or with_leg1_proof else (),
        external_receipt_graphs=[{
            "root_receipt_hash": PROVENANCE_RECEIPT_HASH,
            "receipts": [provenance_receipt],
        }] if with_leg1_parent else [],
        external_ancestry_proofs=(
            [
                {
                    "certificate": {
                        "claim": {
                            "lineage_id": "gateway:test",
                            "output_root_receipt_hash": PROVENANCE_RECEIPT_HASH,
                        }
                    },
                    "disclosed_boot_identities": [],
                    "disclosed_receipts": [provenance_receipt],
                }
            ]
            if with_leg1_proof
            else []
        ),
    )


def _precheck_doc():
    return {
        "precheck_status": "provenance_precheck_passed",
        "reasons": ["provenance_reference_backed"],
        "docs_completeness": {"score": 5},
    }


def _provenance_result():
    return {
        "schema_version": "leadpoet.source_add_provenance_result.v2",
        "submission_id": SUBMISSION_ID,
        "precheck_status": "provenance_precheck_passed",
        "reasons": ["provenance_reference_backed"],
        "precheck_doc": _precheck_doc(),
    }


def _leg1_trigger():
    provenance_hash = sha256_json(_provenance_result())
    return {
        "provenance_precheck_passed": True,
        "submission_id": SUBMISSION_ID,
        "precheck_status": "provenance_precheck_passed",
        "provenance_receipt_hash": PROVENANCE_RECEIPT_HASH,
        "provenance_artifact_hash": provenance_hash,
        "provenance_result_hash": provenance_hash,
    }


def _leg1_authority_rows(*, miner_hotkey: str = "miner"):
    return {
        "source_add_submission_by_id": [
            {
                "submission_id": SUBMISSION_ID,
                "adapter_id": "adapter:test",
                "miner_hotkey": miner_hotkey,
                "precheck_status": "provenance_precheck_passed",
                "precheck_doc": _precheck_doc(),
                "submission_doc": {
                    "provenance_receipt_hash": PROVENANCE_RECEIPT_HASH,
                    "provenance_artifact_hash": sha256_json(
                        _provenance_result()
                    ),
                },
            }
        ],
    }


def _leg1_payload():
    return {
        "decision_kind": "source_add_leg1",
        "decision_payload": {
            "adapter_id": "adapter:test",
            "miner_ref": "miner",
            "start_epoch": 101,
            "existing_rewards": [],
            "alpha_percent": 0.2,
            "reward_epochs": 20,
            "provenance_result": _provenance_result(),
            "trigger_evidence": _leg1_trigger(),
        },
    }


def test_leg1_replaces_host_reward_rows_with_authenticated_rows():
    authenticated = [
        {
            "reward_ref": "source_add_reward:old",
            "adapter_id": "adapter:test",
            "leg": 2,
            "current_reward_status": "active",
        }
    ]
    reader = FakeReader(
        {
            "source_add_rewards_by_adapter": authenticated,
            **_leg1_authority_rows(),
        }
    )
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )
    payload = {
        "decision_kind": "source_add_leg1",
        "decision_payload": {
            "adapter_id": "adapter:test",
            "miner_ref": "miner",
            "start_epoch": 101,
            "existing_rewards": [{"adapter_id": "forged", "leg": 1}],
            "alpha_percent": 0.2,
            "reward_epochs": 20,
            "provenance_result": _provenance_result(),
            "trigger_evidence": _leg1_trigger(),
        },
    }

    resolved = resolver.resolve(
        payload=payload,
        context=_context(with_leg1_parent=True),
    )

    assert resolved["decision_payload"]["existing_rewards"] == authenticated
    assert reader.calls == [
        ("source_add_rewards_by_adapter", {"adapter_id": "adapter:test"}),
        (
            "source_add_submission_by_id",
            {"submission_id": SUBMISSION_ID},
        ),
    ]


def test_leg1_accepts_checkpointed_provenance_parent_proof():
    reader = FakeReader(
        {
            "source_add_rewards_by_adapter": [],
            **_leg1_authority_rows(),
        }
    )
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )

    resolved = resolver.resolve(
        payload={
            "decision_kind": "source_add_leg1",
            "decision_payload": {
                "adapter_id": "adapter:test",
                "miner_ref": "miner",
                "start_epoch": 101,
                "existing_rewards": [],
                "alpha_percent": 0.2,
                "reward_epochs": 20,
                "provenance_result": _provenance_result(),
                "trigger_evidence": _leg1_trigger(),
            },
        },
        context=_context(with_leg1_proof=True),
    )

    assert resolved["decision_kind"] == "source_add_leg1"


def test_leg1_rejects_nonpassing_provenance_authority():
    authority = _leg1_authority_rows()
    authority["source_add_submission_by_id"][0]["precheck_status"] = (
        "needs_manual_review"
    )
    reader = FakeReader(
        {
            "source_add_rewards_by_adapter": [],
            **authority,
        }
    )
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )

    with pytest.raises(
        CoordinatorRewardSourceV2Error,
        match="durable provenance result is invalid",
    ):
        resolver.resolve(
            payload=_leg1_payload(),
            context=_context(with_leg1_parent=True),
        )


def test_leg1_rejects_missing_provenance_authority():
    authority = _leg1_authority_rows()
    authority["source_add_submission_by_id"] = []
    reader = FakeReader({"source_add_rewards_by_adapter": [], **authority})
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )

    with pytest.raises(
        CoordinatorRewardSourceV2Error,
        match="submission owner",
    ):
        resolver.resolve(
            payload=_leg1_payload(),
            context=_context(with_leg1_parent=True),
        )


def test_leg1_rejects_mismatched_provenance_receipt():
    authority = _leg1_authority_rows()
    authority["source_add_submission_by_id"][0]["submission_doc"] = (
        {"provenance_receipt_hash": "sha256:" + "f" * 64}
    )
    reader = FakeReader({"source_add_rewards_by_adapter": [], **authority})
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )

    with pytest.raises(
        CoordinatorRewardSourceV2Error,
        match="provenance",
    ):
        resolver.resolve(
            payload=_leg1_payload(),
            context=_context(with_leg1_parent=True),
        )


def test_leg1_rejects_mismatched_durable_provenance_artifact_hash():
    authority = _leg1_authority_rows()
    authority["source_add_submission_by_id"][0]["submission_doc"][
        "provenance_artifact_hash"
    ] = "sha256:" + "f" * 64
    reader = FakeReader({"source_add_rewards_by_adapter": [], **authority})
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )

    with pytest.raises(
        CoordinatorRewardSourceV2Error,
        match="provenance receipt differs",
    ):
        resolver.resolve(
            payload=_leg1_payload(),
            context=_context(with_leg1_parent=True),
        )


def test_reward_never_falls_back_to_finalized_block_modulo():
    resolver = CoordinatorRewardSourceV2(
        reader=FakeReader({}),
        chain_source=MissingWorkflowEpochChain(),
        config_supplier=_config,
    )

    with pytest.raises(
        CoordinatorRewardSourceV2Error,
        match="execution epoch differs",
    ):
        resolver.resolve(
            payload={
                "decision_kind": "source_add_leg1",
                "decision_payload": {
                    "adapter_id": "adapter:test",
                    "start_epoch": 101,
                },
            },
            context=_context(),
        )


def test_champion_migration_reconstructs_exact_measured_reward_and_bundle():
    reward_id = "champion_reward:sha256:" + "1" * 64
    bundle_id = "score_bundle:" + "2" * 64
    reward = {
        "champion_reward_id": reward_id,
        "score_bundle_id": bundle_id,
        "desired_alpha_percent": 7.45,
        "current_reward_status": "active",
    }
    score_bundle = {
        "score_bundle_id": bundle_id,
        "score_bundle_hash": "sha256:" + "3" * 64,
        "score_bundle_doc": {"schema_version": "1.0"},
    }
    reader = FakeReader(
        {
            "champion_reward_by_id": [reward],
            "score_bundle_by_id": [score_bundle],
        }
    )
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )

    resolved = resolver.resolve(
        payload={
            "decision_kind": "champion_migration",
            "decision_payload": {"champion_reward_id": reward_id},
        },
        context=_context(),
    )

    assert resolved == {
        "decision_kind": "champion_migration",
        "decision_payload": {
            "reward_row": reward,
            "score_bundle": score_bundle,
        },
    }
    assert reader.calls == [
        ("champion_reward_by_id", {"champion_reward_id": reward_id}),
        ("score_bundle_by_id", {"score_bundle_id": bundle_id}),
    ]

    with pytest.raises(
        CoordinatorRewardSourceV2Error,
        match="champion migration request fields are invalid",
    ):
        resolver.resolve(
            payload={
                "decision_kind": "champion_migration",
                "decision_payload": {
                    "champion_reward_id": reward_id,
                    "desired_alpha_percent": 99.0,
                },
            },
            context=_context(),
        )


def test_source_add_migration_reconstructs_reward_and_measured_submission():
    reward_ref = "source_add_reward:201a08f0d2b503bf"
    submission_id = "source_add_submission:a3d8f3e562dca636"
    reward = {
        "reward_ref": reward_ref,
        "trigger_evidence_doc": {"submission_id": submission_id},
    }
    submission = {
        "submission_id": submission_id,
        "adapter_id": "adapter:test",
        "miner_hotkey": "miner",
        "precheck_status": "provenance_precheck_passed",
    }
    reader = FakeReader(
        {
            "source_add_reward_by_ref": [reward],
            "source_add_submission_by_id": [submission],
        }
    )
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )

    resolved = resolver.resolve(
        payload={
            "decision_kind": "source_add_migration",
            "decision_payload": {"reward_ref": reward_ref},
        },
        context=_context(),
    )

    assert resolved == {
        "decision_kind": "source_add_migration",
        "decision_payload": {
            "reward_row": reward,
            "source_submission": submission,
        },
    }
    assert reader.calls == [
        ("source_add_reward_by_ref", {"reward_ref": reward_ref}),
        ("source_add_submission_by_id", {"submission_id": submission_id}),
    ]


def test_leg1_daily_cap_is_not_rechecked_outside_atomic_slot_transaction():
    reader = FakeReader(
        {
            "source_add_rewards_by_adapter": [],
            **_leg1_authority_rows(),
            "source_add_leg1_events_since": [
                {"reward_ref": "reward-%d" % index} for index in range(10)
            ],
        }
    )
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )
    payload = {
        "decision_kind": "source_add_leg1",
        "decision_payload": {
            "adapter_id": "adapter:test",
            "miner_ref": "miner",
            "start_epoch": 101,
            "existing_rewards": [],
            "alpha_percent": 0.2,
            "reward_epochs": 20,
            "provenance_result": _provenance_result(),
            "trigger_evidence": _leg1_trigger(),
        },
    }

    resolved = resolver.resolve(
        payload=payload,
        context=_context(with_leg1_parent=True),
    )

    assert resolved["decision_payload"]["provenance_result"] == (
        _provenance_result()
    )
    assert all(call[0] != "source_add_leg1_events_since" for call in reader.calls)


def test_leg1_rejects_host_substituted_miner():
    reader = FakeReader(
        {
            "source_add_rewards_by_adapter": [],
            "source_add_submission_by_id": [
                {
                    "submission_id": SUBMISSION_ID,
                    "adapter_id": "adapter:test",
                    "miner_hotkey": "real-owner",
                    "precheck_status": "provenance_precheck_passed",
                    "precheck_doc": _precheck_doc(),
                    "submission_doc": {
                        "provenance_receipt_hash": PROVENANCE_RECEIPT_HASH,
                    },
                }
            ],
        }
    )
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )
    payload = {
        "decision_kind": "source_add_leg1",
        "decision_payload": {
            "adapter_id": "adapter:test",
            "miner_ref": "forged-owner",
            "start_epoch": 101,
            "existing_rewards": [],
            "alpha_percent": 0.2,
            "reward_epochs": 20,
            "provenance_result": _provenance_result(),
            "trigger_evidence": _leg1_trigger(),
        },
    }
    with pytest.raises(CoordinatorRewardSourceV2Error, match="owner or status"):
        resolver.resolve(
            payload=payload,
            context=_context(with_leg1_parent=True),
        )


@pytest.mark.parametrize("kind", ("champion", "reimbursement", "source_add_leg2"))
def test_retired_loop_rewards_cannot_read_or_create_business_state(kind):
    reader = FakeReader({})
    authority = CoordinatorRewardSourceV2(
        reader=reader, chain_source=FakeChain(), config_supplier=_config,
    )
    with pytest.raises(CoordinatorRewardSourceV2Error, match="kind is unsupported"):
        authority.resolve(
            payload={"decision_kind": kind, "decision_payload": {}},
            context=_context(),
        )
    assert reader.calls == []
