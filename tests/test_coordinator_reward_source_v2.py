from datetime import datetime, timezone
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
JUDGE_RECEIPT_HASH = "sha256:" + "c" * 64


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
        source_add_leg2_alpha_percent=5.0,
        source_add_leg1_max_per_utc_day=50,
        lab_reward_epochs=20,
        reimbursements_enabled=True,
        shadow_reimbursements_enabled=False,
        reimbursement_default_island="generalist",
        default_compute_budget_usd=25.0,
        loop_start_fee_usd=5.0,
        clamp_compute_budget_usd=lambda value: float(value),
        reimbursement_policy_doc=lambda enabled: {
            "policy_id": "policy:v2",
            "enabled": bool(enabled),
        },
    )


def _context(
    *,
    with_leg1_parent=False,
    with_leg1_proof=False,
    with_judge_parent=False,
):
    provenance_receipt = {
        "receipt_hash": PROVENANCE_RECEIPT_HASH,
        "role": "gateway_coordinator",
        "purpose": "research_lab.source_add_provenance.v2",
        "status": "succeeded",
        "output_root": sha256_json(_provenance_result()),
    }
    judge_receipt = {
        "receipt_hash": JUDGE_RECEIPT_HASH,
        "role": "gateway_scoring",
        "purpose": "research_lab.source_add_judge.v2",
        "status": "succeeded",
        "output_root": sha256_json(_judge_result()),
    }
    return ExecutionContextV2(
        job_id="reward:test",
        purpose="research_lab.reward_decision.v2",
        epoch_id=100,
        parent_receipt_hashes=(
            (JUDGE_RECEIPT_HASH,)
            if with_judge_parent
            else (
                (PROVENANCE_RECEIPT_HASH,)
                if with_leg1_parent or with_leg1_proof
                else ()
            )
        ),
        external_receipt_graphs=(
            [
                {
                    "root_receipt_hash": JUDGE_RECEIPT_HASH,
                    "receipts": [judge_receipt],
                }
            ]
            if with_judge_parent
            else (
                [
                    {
                        "root_receipt_hash": PROVENANCE_RECEIPT_HASH,
                        "receipts": [provenance_receipt],
                    }
                ]
                if with_leg1_parent
                else []
            )
        ),
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


def _judge_result():
    return {
        "schema_version": "leadpoet.source_add_judge_result.v2",
        "candidate_id": "candidate:test",
        "score_bundle_hash": HASH,
        "provisioned_sources_hash": HASH,
        "verdict": {
            "verdict": "helped",
            "confidence": 0.9,
            "source_used": True,
            "adapter_id": "adapter:test",
            "registry_provider_id": "provider:test",
            "evidence_summary": "The measured source materially helped.",
            "reason_codes": ["material_source_use"],
            "model_id": "openai/gpt-test",
            "provider_usage": {},
            "judge_doc_hash": HASH,
        },
    }


def _trigger():
    verdict = _judge_result()["verdict"]
    return {
        "llm_judge_passed": True,
        "llm_verdict": "helped",
        "llm_confidence": 0.9,
        "source_used": True,
        "adapter_id": "adapter:test",
        "registry_provider_id": "provider:test",
        "evidence_summary": verdict["evidence_summary"],
        "reason_codes": ["material_source_use"],
        "judge_model": "openai/gpt-test",
        "judge_doc_hash": HASH,
        "provider_usage": {},
    }


def _leg2_reader():
    return FakeReader(
        {
            "source_add_rewards_by_adapter": [],
            "source_add_provisioning_by_adapter": [
                {
                    "adapter_id": "adapter:test",
                    "miner_hotkey": "real-owner",
                    "registry_provider_id": "provider:test",
                    "provision_status": "provisioned_autoresearch_eligible",
                }
            ],
        }
    )


def _leg2_payload(*, judge_result=None):
    return {
        "decision_kind": "source_add_leg2",
        "decision_payload": {
            "adapter_id": "adapter:test",
            "miner_ref": "real-owner",
            "start_epoch": 101,
            "trigger_evidence": _trigger(),
            "judge_result": judge_result or _judge_result(),
            "existing_rewards": [],
            "alpha_percent": 5.0,
            "reward_epochs": 20,
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
        clock=lambda: datetime(2026, 7, 10, 12, tzinfo=timezone.utc),
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


def test_leg2_requires_exact_signed_judge_parent():
    resolver = CoordinatorRewardSourceV2(
        reader=_leg2_reader(),
        chain_source=FakeChain(),
        config_supplier=_config,
    )

    resolved = resolver.resolve(
        payload=_leg2_payload(),
        context=_context(with_judge_parent=True),
    )

    assert resolved["decision_kind"] == "source_add_leg2"
    assert resolved["decision_payload"]["judge_result"] == _judge_result()


def test_leg2_zero_alpha_fails_closed_at_reward_authority():
    config = _config()
    config.source_add_leg2_alpha_percent = 0.0
    resolver = CoordinatorRewardSourceV2(
        reader=_leg2_reader(),
        chain_source=FakeChain(),
        config_supplier=lambda: config,
    )

    with pytest.raises(CoordinatorRewardSourceV2Error, match="leg is disabled"):
        resolver.resolve(
            payload=_leg2_payload(),
            context=_context(with_judge_parent=True),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "zero_parents",
        "multiple_parents",
        "wrong_role",
        "wrong_purpose",
        "wrong_status",
        "wrong_root",
        "host_mutated_result",
    ],
)
def test_leg2_rejects_unbound_or_mismatched_judge_parent(mutation):
    resolver = CoordinatorRewardSourceV2(
        reader=_leg2_reader(),
        chain_source=FakeChain(),
        config_supplier=_config,
    )
    context = _context(with_judge_parent=True)
    payload = _leg2_payload()
    if mutation == "zero_parents":
        context = _context()
    elif mutation == "multiple_parents":
        second_hash = "sha256:" + "d" * 64
        context.parent_receipt_hashes = (JUDGE_RECEIPT_HASH, second_hash)
        context.external_receipt_graphs.append(
            {
                "root_receipt_hash": second_hash,
                "receipts": [
                    {
                        "receipt_hash": second_hash,
                        "role": "gateway_scoring",
                        "purpose": "research_lab.source_add_judge.v2",
                        "status": "succeeded",
                        "output_root": sha256_json(_judge_result()),
                    }
                ],
            }
        )
    elif mutation == "wrong_root":
        context.external_receipt_graphs[0]["root_receipt_hash"] = HASH
    elif mutation == "host_mutated_result":
        mutated = _judge_result()
        mutated["verdict"] = {**mutated["verdict"], "confidence": 0.8}
        payload = _leg2_payload(judge_result=mutated)
    else:
        field, value = {
            "wrong_role": ("role", "gateway_coordinator"),
            "wrong_purpose": ("purpose", "research_lab.promotion_decision.v2"),
            "wrong_status": ("status", "failed"),
        }[mutation]
        context.external_receipt_graphs[0]["receipts"][0][field] = value

    with pytest.raises(CoordinatorRewardSourceV2Error):
        resolver.resolve(payload=payload, context=context)


def test_leg2_requires_authenticated_adapter_owner():
    reader = FakeReader(
        {
            "source_add_rewards_by_adapter": [],
            "source_add_provisioning_by_adapter": [
                {
                    "adapter_id": "adapter:test",
                    "miner_hotkey": "real-owner",
                    "registry_provider_id": "provider:test",
                    "provision_status": "provisioned_autoresearch_eligible",
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
        "decision_kind": "source_add_leg2",
        "decision_payload": {
            "adapter_id": "adapter:test",
            "miner_ref": "forged-owner",
            "start_epoch": 101,
            "trigger_evidence": _trigger(),
            "judge_result": _judge_result(),
            "existing_rewards": [],
            "alpha_percent": 5.0,
            "reward_epochs": 20,
        },
    }

    with pytest.raises(CoordinatorRewardSourceV2Error, match="owner"):
        resolver.resolve(payload=payload, context=_context(with_judge_parent=True))


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


def test_leg2_rejects_trigger_that_differs_from_signed_judge():
    reader = FakeReader(
        {
            "source_add_rewards_by_adapter": [],
            "source_add_provisioning_by_adapter": [
                {
                    "adapter_id": "adapter:test",
                    "miner_hotkey": "real-owner",
                    "registry_provider_id": "provider:test",
                    "provision_status": "provisioned_autoresearch_eligible",
                }
            ],
        }
    )
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )
    trigger = _trigger()
    trigger["llm_confidence"] = 1.0
    payload = {
        "decision_kind": "source_add_leg2",
        "decision_payload": {
            "adapter_id": "adapter:test",
            "miner_ref": "real-owner",
            "start_epoch": 101,
            "trigger_evidence": trigger,
            "judge_result": _judge_result(),
            "existing_rewards": [],
            "alpha_percent": 5.0,
            "reward_epochs": 20,
        },
    }
    with pytest.raises(CoordinatorRewardSourceV2Error, match="signed judge"):
        resolver.resolve(payload=payload, context=_context(with_judge_parent=True))


def test_reimbursement_reconstructs_formula_inputs_from_measured_rows():
    run_id = "11111111-1111-4111-8111-111111111111"
    ticket_id = "22222222-2222-4222-8222-222222222222"
    receipt_id = "33333333-3333-4333-8333-333333333333"
    payment_id = "44444444-4444-4444-8444-444444444444"
    reader = FakeReader(
        {
            "reimbursement_ticket_by_id": [
                {
                    "ticket_id": ticket_id,
                    "miner_hotkey": "miner",
                    "island": "generalist",
                    "brief_sanitized_ref": "brief:1",
                    "miner_openrouter_key_ref": "encrypted_ref:openrouter:abc",
                    "ticket_doc": {"requested_compute_budget_usd": 25.0},
                    "created_at": "2026-07-09T12:00:00Z",
                    "current_status_at": "2026-07-09T12:00:00Z",
                }
            ],
            "reimbursement_receipt_by_id": [
                {
                    "receipt_id": receipt_id,
                    "run_id": run_id,
                    "ticket_id": ticket_id,
                    "loop_start_payment_id": payment_id,
                    "loop_start_credit_id": None,
                    "current_receipt_status": "completed",
                }
            ],
            "reimbursement_payment_by_id": [
                {
                    "payment_id": payment_id,
                    "ticket_id": ticket_id,
                    "payment_status": "verified",
                    "verification_doc": {"compute_budget_usd": 25.0},
                }
            ],
            "reimbursement_queue_events_by_run": [],
            "reimbursement_participation_tickets": [
                {
                    "ticket_id": ticket_id,
                    "miner_hotkey": "miner",
                    "island": "generalist",
                    "brief_sanitized_ref": "brief:1",
                    "created_at": "2026-07-09T12:00:00Z",
                    "current_status_at": "2026-07-09T12:00:00Z",
                }
            ],
            "reimbursement_queue_by_ticket": [
                {
                    "run_id": run_id,
                    "ticket_id": ticket_id,
                    "current_queue_status": "completed",
                    "current_status_at": "2026-07-10T19:59:00Z",
                }
            ],
            "reimbursement_cap_awards_by_day": [],
        }
    )
    resolver = CoordinatorRewardSourceV2(
        reader=reader,
        chain_source=FakeChain(),
        config_supplier=_config,
    )
    autoresearch_result = {
        "actual_openrouter_cost_microusd": 1_250_000,
        "status": "completed",
    }
    parent_hash = "sha256:" + "c" * 64
    context = _context()
    context.parent_receipt_hashes = (parent_hash,)
    context.external_receipt_graphs = [
        {
            "root_receipt_hash": parent_hash,
            "receipts": [
                {
                    "receipt_hash": parent_hash,
                    "purpose": "research_lab.candidate_decision.v2",
                    "issued_at": "2026-07-10T20:00:00Z",
                }
            ],
        }
    ]
    payload = {
        "decision_kind": "reimbursement",
        "decision_payload": {
            "source_request": {
                "run_id": run_id,
                "ticket_id": ticket_id,
                "receipt_id": receipt_id,
            },
            "autoresearch_result": autoresearch_result,
        },
    }

    resolved = resolver.resolve(payload=payload, context=context)

    decision = resolved["decision_payload"]
    assert decision["run_cost"]["actual_openrouter_cost_usd"] == 1.25
    assert decision["run_cost"]["verified_loop_start_payment"] is True
    assert decision["participation_snapshot"]["paid_loop_count"] == 1
    assert decision["participation_snapshot"]["lookback_end"] == (
        "2026-07-10T20:00:00+00:00"
    )
    assert decision["start_epoch"] == 101
    assert decision["autoresearch_result"] == autoresearch_result
