"""Measured adapters for the unchanged Research Lab reward kernels."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping

from leadpoet_canonical.attested_v2 import sha256_json


OP_RESEARCH_LAB_REWARD_DECISION = "research_lab_reward_decision"
REWARD_DECISION_KINDS = frozenset(
    {
        "champion_migration",
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
    if kind == "champion_migration":
        return _champion_migration(value)
    return _champion_migration(value)


def _champion_migration(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Attest an immutable pre-V2 obligation without replaying a new policy.

    Historical reward policies changed over time. Recomputing an old row with
    today's policy would rewrite history. Instead, reconstruct the exact
    anchored payload from the stored obligation and its immutable score bundle,
    then require its original hashes to match byte-for-byte.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "reward_row",
        "score_bundle",
    }:
        raise RewardExecutorV2Error(
            "champion migration fields are invalid"
        )
    reward_row = _mapping(value.get("reward_row"), "champion migration reward")
    score_bundle = _mapping(
        value.get("score_bundle"), "champion migration score bundle"
    )
    bundle_id = str(score_bundle.get("score_bundle_id") or "")
    bundle_hash = str(score_bundle.get("score_bundle_hash") or "")
    bundle_doc = _mapping(
        score_bundle.get("score_bundle_doc"), "champion migration score document"
    )
    if (
        bundle_id != str(reward_row.get("score_bundle_id") or "")
        or bundle_hash != str(bundle_doc.get("score_bundle_hash") or "")
    ):
        raise RewardExecutorV2Error(
            "champion migration score bundle differs from reward"
        )
    stored_source_hash = str(reward_row.get("source_score_bundle_hash") or "")
    if stored_source_hash and stored_source_hash != bundle_hash:
        raise RewardExecutorV2Error(
            "champion migration source score bundle hash differs"
        )
    daily_counts: Dict[str, int] = {}
    aggregates = bundle_doc.get("aggregates")
    per_icp = (
        aggregates.get("per_icp_results")
        if isinstance(aggregates, Mapping)
        else None
    )
    if not isinstance(per_icp, list):
        raise RewardExecutorV2Error(
            "champion migration score bundle has no per-ICP results"
        )
    for item in per_icp:
        if not isinstance(item, Mapping):
            raise RewardExecutorV2Error(
                "champion migration per-ICP result is invalid"
            )
        ref = str(item.get("icp_ref") or "")
        match = re.search(r"qualification_private_icp_sets:(\d+):", ref)
        day = match.group(1) if match else ref.split(":")[0]
        if not day:
            raise RewardExecutorV2Error(
                "champion migration per-ICP day is missing"
            )
        daily_counts[day] = daily_counts.get(day, 0) + 1

    candidate_id = str(reward_row.get("candidate_id") or "")
    run_id = str(reward_row.get("run_id") or "")
    reconstructed = {
        "champion_reward_id": "",
        "status": "active",
        "reasons": [],
        "uid": int(reward_row.get("miner_uid", -1)),
        "miner_hotkey": str(reward_row.get("miner_hotkey") or ""),
        "island": str(reward_row.get("island") or "generalist"),
        "source_id": candidate_id or bundle_id or run_id or "unknown",
        "score_bundle_id": bundle_id,
        "candidate_id": candidate_id,
        "run_id": run_id,
        "evaluation_epoch": int(reward_row.get("evaluation_epoch") or 0),
        "start_epoch": int(reward_row.get("start_epoch") or 0),
        "epoch_count": int(reward_row.get("epoch_count") or 0),
        "improvement_points": float(
            reward_row.get("improvement_points") or 0.0
        ),
        "threshold_points": float(reward_row.get("threshold_points") or 0.0),
        "desired_alpha_percent": float(
            reward_row.get("desired_alpha_percent") or 0.0
        ),
        "daily_icp_counts": dict(sorted(daily_counts.items())),
        "required_icp_count": sum(daily_counts.values()),
        "input_hash": str(reward_row.get("input_hash") or ""),
    }
    anchored_hash = sha256_json(reconstructed)
    reward_id = "champion_reward:" + anchored_hash
    if (
        reward_id != str(reward_row.get("champion_reward_id") or "")
        or anchored_hash != str(reward_row.get("anchored_hash") or "")
    ):
        raise RewardExecutorV2Error(
            "champion migration anchored payload differs from stored obligation"
        )
    return {
        "decision_kind": "champion",
        "reward": {
            **reconstructed,
            "champion_reward_id": reward_id,
            "anchored_hash": anchored_hash,
        },
    }
