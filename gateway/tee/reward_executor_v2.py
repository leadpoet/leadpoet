"""Measured adapters for the unchanged Research Lab reward kernels."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from leadpoet_verifier.economics import build_champion_reward_obligation
from research_lab.reimbursements import (
    ReimbursementCapUsage,
    build_reimbursement_schedule,
    compute_reimbursement_award,
)
from research_lab.source_add_rewards import create_leg1_reward, create_leg2_reward


OP_RESEARCH_LAB_REWARD_DECISION = "research_lab_reward_decision"
REWARD_DECISION_KINDS = frozenset(
    {
        "champion",
        "reimbursement",
        "source_add_leg1",
        "source_add_leg2",
    }
)


class RewardExecutorV2Error(ValueError):
    """A reward decision request is not canonical for its existing kernel."""


def reward_receipt_projection_v2(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a reward decision onto fields already persisted by business rows."""

    if not isinstance(result, Mapping):
        raise RewardExecutorV2Error("reward result is invalid")
    kind = str(result.get("decision_kind") or "")
    if kind == "champion":
        reward = _mapping(result.get("reward"), "champion reward")
        return champion_reward_row_projection_v2(reward)
    if kind in {"source_add_leg1", "source_add_leg2"}:
        reward = _mapping(result.get("reward"), "SOURCE_ADD reward")
        return source_add_reward_row_projection_v2(kind, reward)
    if kind == "reimbursement":
        award = _mapping(result.get("award"), "reimbursement award")
        schedule = _mapping(result.get("schedule"), "reimbursement schedule")
        return reimbursement_reward_row_projection_v2(award, schedule)
    raise RewardExecutorV2Error("reward result kind is unsupported")


def champion_reward_row_projection_v2(reward: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "leadpoet.reward_row_projection.v2",
        "decision_kind": "champion",
        "reward_row": {
            "champion_reward_id": str(reward["champion_reward_id"]),
            "score_bundle_id": str(reward.get("score_bundle_id") or ""),
            "candidate_id": str(reward.get("candidate_id") or ""),
            "run_id": str(reward["run_id"]),
            "miner_hotkey": str(reward["miner_hotkey"]),
            "miner_uid": int(reward.get("miner_uid", reward.get("uid", -1))),
            "island": str(reward["island"]),
            "evaluation_epoch": int(reward["evaluation_epoch"]),
            "start_epoch": int(reward["start_epoch"]),
            "epoch_count": int(reward["epoch_count"]),
            "improvement_points": float(reward["improvement_points"]),
            "threshold_points": float(reward["threshold_points"]),
            "desired_alpha_percent": float(reward["desired_alpha_percent"]),
            "input_hash": str(reward["input_hash"]),
            "anchored_hash": str(reward["anchored_hash"]),
        },
    }


def source_add_reward_row_projection_v2(
    kind: str,
    reward: Mapping[str, Any],
) -> Dict[str, Any]:
    if kind not in {"source_add_leg1", "source_add_leg2"}:
        raise RewardExecutorV2Error("SOURCE_ADD reward projection kind is invalid")
    return {
        "schema_version": "leadpoet.reward_row_projection.v2",
        "decision_kind": str(kind),
        "reward_row": {
            "reward_ref": str(reward["reward_ref"]),
            "adapter_id": str(reward["adapter_id"]),
            "miner_hotkey": str(
                reward.get("miner_hotkey", reward.get("miner_ref", ""))
            ),
            "leg": int(reward["leg"]),
            "reward_kind": str(reward["reward_kind"]),
            "alpha_percent": float(reward["alpha_percent"]),
            "reward_epochs": int(reward["reward_epochs"]),
            "start_epoch": int(reward["start_epoch"]),
            "initial_reward_status": str(
                reward.get("initial_reward_status", reward.get("state", ""))
            ),
            "trigger_evidence_doc": dict(
                reward.get("trigger_evidence_doc", reward.get("trigger_evidence", {}))
                or {}
            ),
            "public_label": str(reward.get("public_label") or ""),
        },
    }


def reimbursement_reward_row_projection_v2(
    award: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "leadpoet.reward_row_projection.v2",
        "decision_kind": "reimbursement",
        "award_row": {
            "award_id": str(award["award_id"]),
            "run_id": str(award["run_id"]),
            "miner_hotkey": str(award["miner_hotkey"]),
            "island": str(award["island"]),
            "run_day": str(award["run_day"]),
            "award_status": str(award.get("award_status", award.get("status", ""))),
            "participation_score": float(award["participation_score"]),
            "participation_fraction": float(award["participation_fraction"]),
            "rebate_rate": float(award["rebate_rate"]),
            "eligible_cost_microusd": int(award["eligible_cost_microusd"]),
            "target_reimbursement_microusd": int(
                award["target_reimbursement_microusd"]
            ),
            "reimbursement_epochs": int(award["reimbursement_epochs"]),
            "loop_start_fee_included": bool(award["loop_start_fee_included"]),
            "input_hash": str(award["input_hash"]),
        },
        "schedule_row": {
            "schedule_id": str(schedule["schedule_id"]),
            "award_id": str(schedule["award_id"]),
            "schedule_status": str(
                schedule.get("schedule_status", schedule.get("status", ""))
            ),
            "start_epoch": int(schedule["start_epoch"]),
            "epoch_count": int(schedule["epoch_count"]),
            "total_microusd": int(schedule["total_microusd"]),
            "entries": [dict(item) for item in schedule.get("entries") or ()],
        },
    }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RewardExecutorV2Error("%s is invalid" % field)
    return value


def execute_reward_decision_v2(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "decision_kind",
        "decision_payload",
    }:
        raise RewardExecutorV2Error("reward decision payload fields are invalid")
    kind = str(payload.get("decision_kind") or "")
    if kind not in REWARD_DECISION_KINDS:
        raise RewardExecutorV2Error("reward decision kind is unsupported")
    value = payload.get("decision_payload")
    if not isinstance(value, Mapping):
        raise RewardExecutorV2Error("reward decision input is invalid")
    if kind == "champion":
        return _champion(value)
    if kind == "reimbursement":
        return _reimbursement(value)
    return _source_add(kind, value)


def _champion(value: Mapping[str, Any]) -> Dict[str, Any]:
    if set(value) != {"obligation_input", "policy", "promotion_decision"}:
        raise RewardExecutorV2Error("champion reward fields are invalid")
    obligation_input = value.get("obligation_input")
    policy = value.get("policy")
    promotion_decision = value.get("promotion_decision")
    if (
        not isinstance(obligation_input, Mapping)
        or not isinstance(policy, Mapping)
        or not isinstance(promotion_decision, Mapping)
    ):
        raise RewardExecutorV2Error("champion reward input is invalid")
    if promotion_decision.get("status") != "promotion_passed":
        raise RewardExecutorV2Error("champion reward promotion decision is invalid")
    return {
        "decision_kind": "champion",
        "reward": build_champion_reward_obligation(obligation_input, policy),
    }


def _source_add(kind: str, value: Mapping[str, Any]) -> Dict[str, Any]:
    common = {
        "adapter_id",
        "miner_ref",
        "start_epoch",
        "existing_rewards",
        "alpha_percent",
        "reward_epochs",
    }
    expected = common | (
        {"trigger_evidence", "judge_result"}
        if kind == "source_add_leg2"
        else {"provenance_result"}
    )
    if set(value) != expected:
        raise RewardExecutorV2Error("SOURCE_ADD reward fields are invalid")
    existing = value.get("existing_rewards")
    if not isinstance(existing, list) or any(
        not isinstance(item, Mapping) for item in existing
    ):
        raise RewardExecutorV2Error("SOURCE_ADD existing rewards are invalid")
    kwargs = {
        "adapter_id": str(value.get("adapter_id") or ""),
        "start_epoch": int(value.get("start_epoch") or 0),
        "existing_rewards": [dict(item) for item in existing],
        "alpha_percent": float(value.get("alpha_percent") or 0.0),
        "reward_epochs": int(value.get("reward_epochs") or 0),
    }
    if kind == "source_add_leg1":
        provenance = value.get("provenance_result")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("precheck_status") != "provenance_precheck_passed"
        ):
            raise RewardExecutorV2Error(
                "SOURCE_ADD Leg 1 provenance result is invalid"
            )
        reward = create_leg1_reward(
            miner_ref=str(value.get("miner_ref") or ""),
            **kwargs,
        )
    else:
        trigger = value.get("trigger_evidence")
        judge_result = value.get("judge_result")
        if not isinstance(trigger, Mapping) or not isinstance(judge_result, Mapping):
            raise RewardExecutorV2Error("SOURCE_ADD trigger evidence is invalid")
        verdict = judge_result.get("verdict")
        if (
            not isinstance(verdict, Mapping)
            or verdict.get("verdict") != "helped"
            or verdict.get("source_used") is not True
            or trigger.get("llm_judge_passed") is not True
        ):
            raise RewardExecutorV2Error(
                "SOURCE_ADD Leg 2 signed judge did not approve the reward"
            )
        reward = create_leg2_reward(
            adapter_owner_miner_ref=str(value.get("miner_ref") or ""),
            trigger_evidence=dict(trigger),
            **kwargs,
        )
    return {
        "decision_kind": kind,
        "reward": reward.to_dict() if reward is not None else None,
    }


def _reimbursement(value: Mapping[str, Any]) -> Dict[str, Any]:
    if set(value) != {
        "run_cost",
        "participation_snapshot",
        "policy",
        "cap_usage",
        "start_epoch",
        "autoresearch_result",
        "source_state",
    }:
        raise RewardExecutorV2Error("reimbursement decision fields are invalid")
    for field in ("run_cost", "participation_snapshot", "policy", "cap_usage"):
        if not isinstance(value.get(field), Mapping):
            raise RewardExecutorV2Error("reimbursement %s is invalid" % field)
    for field in ("autoresearch_result", "source_state"):
        if not isinstance(value.get(field), Mapping):
            raise RewardExecutorV2Error("reimbursement %s is invalid" % field)
    award = compute_reimbursement_award(
        value["run_cost"],
        value["participation_snapshot"],
        value["policy"],
        ReimbursementCapUsage.from_mapping(value["cap_usage"]),
    ).to_dict()
    schedule = build_reimbursement_schedule(
        award,
        start_epoch=int(value.get("start_epoch") or 0),
    ).to_dict()
    return {
        "decision_kind": "reimbursement",
        "award": award,
        "schedule": schedule,
        "source_state": dict(value["source_state"]),
    }
