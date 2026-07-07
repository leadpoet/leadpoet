"""Adapter-ablation attribution harness (W6 §3.2 gate 4).

Leg 2 pays only when the implementing merge's improvement is attributable to
the adapter: re-run the implementing candidate on a holdout set with the
adapter ENABLED vs DISABLED; the on-minus-off delta must reach the threshold
(launch default 0.5 points — half the 1.0-point merge bar).

"Disabled" is mechanical, not cooperative: the adapter-off evaluation runs
against an evidence-proxy registry with the adapter's provider entry removed,
so every call the candidate routes to that source 404s at the proxy. A patch
cannot fake attribution by ignoring a disable flag — the source is simply
unreachable. ``disabled_registry`` builds that registry; the injectable
``evaluator`` runs the existing benchmark eval for one candidate against a
given registry and returns its score.

This gates the REWARD only — promotion stays score-only (§4 invariant).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence

from gateway.research_lab.provider_evidence_proxy import ProviderRegistryEntry
from research_lab.canonical import sha256_json
from research_lab.source_add_rewards import DEFAULT_ABLATION_THRESHOLD_POINTS

logger = logging.getLogger(__name__)

# evaluator contract: (candidate_ref, registry_entries, holdout_ref) -> score.
# The caller wires the existing benchmark eval runner; the harness only
# controls which registry the eval's proxy serves.
AblationEvaluator = Callable[
    [str, Sequence[ProviderRegistryEntry], str],
    Awaitable[float],
]


@dataclass(frozen=True)
class AdapterAblationResult:
    adapter_id: str
    registry_provider_id: str
    candidate_ref: str
    holdout_ref: str
    adapter_on_score: float
    adapter_off_score: float
    delta_points: float
    threshold_points: float
    passed: bool
    evaluated_at: str
    result_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["result_hash"]:
            data["result_hash"] = sha256_json({key: value for key, value in data.items() if key != "result_hash"})
        return data


def disabled_registry(
    entries: Sequence[ProviderRegistryEntry],
    registry_provider_id: str,
) -> list[ProviderRegistryEntry]:
    """The eval registry with the adapter's provider removed (adapter off)."""

    remaining = [entry for entry in entries if entry.id != str(registry_provider_id)]
    if len(remaining) == len(entries):
        raise ValueError(
            f"registry_provider_id {registry_provider_id!r} is not in the eval registry — "
            "ablation would compare two identical runs"
        )
    return remaining


async def run_adapter_ablation(
    *,
    adapter_id: str,
    registry_provider_id: str,
    candidate_ref: str,
    registry_entries: Sequence[ProviderRegistryEntry],
    evaluator: AblationEvaluator,
    holdout_ref: str = "",
    threshold_points: float = DEFAULT_ABLATION_THRESHOLD_POINTS,
) -> AdapterAblationResult:
    """Run the on/off pair and score attribution.

    The holdout set (``holdout_ref``) must be a lab-scheduled set, never the
    private benchmark window — same rule as SOURCE_ADD trials.
    """

    off_entries = disabled_registry(registry_entries, registry_provider_id)
    adapter_on_score = float(await evaluator(candidate_ref, list(registry_entries), holdout_ref))
    adapter_off_score = float(await evaluator(candidate_ref, off_entries, holdout_ref))
    delta = adapter_on_score - adapter_off_score
    result = AdapterAblationResult(
        adapter_id=str(adapter_id),
        registry_provider_id=str(registry_provider_id),
        candidate_ref=str(candidate_ref),
        holdout_ref=str(holdout_ref),
        adapter_on_score=adapter_on_score,
        adapter_off_score=adapter_off_score,
        delta_points=delta,
        threshold_points=float(threshold_points),
        passed=delta >= float(threshold_points),
        evaluated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    logger.info(
        "research_lab_source_add_ablation adapter=%s candidate=%s on=%.4f off=%.4f delta=%.4f passed=%s",
        adapter_id,
        candidate_ref,
        adapter_on_score,
        adapter_off_score,
        delta,
        result.passed,
    )
    return result


async def arm_leg2_for_merge(
    *,
    adapter_id: str,
    adapter_owner_miner_ref: str,
    catalog_id: str,
    catalog_registry_ids: Sequence[str],
    merged: bool,
    merged_diff_routed_registry_ids: Sequence[str],
    merge_cleared_score_bar: bool,
    shadow_monitor_live: bool,
    shadow_window_days_elapsed: float,
    shadow_window_survived: bool,
    ablation: AdapterAblationResult | None,
    start_epoch: int,
    accepted_at: str,
    market_open_at: str = "",
    existing_rewards: Sequence[Mapping[str, Any]] = (),
    shadow_window_days_required: float | None = None,
    expiry_months: int | None = None,
    persist: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Evaluate the leg-2 trigger and (optionally) persist the reward rows.

    Returns (reward_doc, blockers). ``reward_doc`` is None whenever any gate
    blocks — including a missing/failed ablation, which is exactly the §3.2
    hard-blocker behavior.
    """

    from research_lab.source_add_rewards import (
        DEFAULT_LEG2_EXPIRY_MONTHS,
        DEFAULT_SHADOW_WINDOW_DAYS,
        create_leg2_reward,
        evaluate_leg2_trigger,
    )

    armed, blockers, evidence = evaluate_leg2_trigger(
        adapter_id=adapter_id,
        catalog_registry_ids=catalog_registry_ids,
        merged=merged,
        merged_diff_routed_registry_ids=merged_diff_routed_registry_ids,
        merge_cleared_score_bar=merge_cleared_score_bar,
        shadow_monitor_live=shadow_monitor_live,
        shadow_window_days_elapsed=shadow_window_days_elapsed,
        shadow_window_survived=shadow_window_survived,
        ablation_adapter_on_score=ablation.adapter_on_score if ablation else None,
        ablation_adapter_off_score=ablation.adapter_off_score if ablation else None,
        now=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        accepted_at=accepted_at,
        market_open_at=market_open_at,
        existing_rewards=existing_rewards,
        shadow_window_days_required=(
            float(shadow_window_days_required) if shadow_window_days_required is not None else DEFAULT_SHADOW_WINDOW_DAYS
        ),
        ablation_threshold_points=ablation.threshold_points if ablation else DEFAULT_ABLATION_THRESHOLD_POINTS,
        expiry_months=int(expiry_months) if expiry_months is not None else DEFAULT_LEG2_EXPIRY_MONTHS,
    )
    if not armed:
        return None, blockers
    if ablation is not None:
        evidence["ablation_result"] = ablation.to_dict()
    record = create_leg2_reward(
        adapter_id=adapter_id,
        adapter_owner_miner_ref=adapter_owner_miner_ref,
        start_epoch=int(start_epoch),
        trigger_evidence=evidence,
        existing_rewards=existing_rewards,
    )
    if record is None:
        return None, ["leg2_already_created"]
    if persist:
        from gateway.research_lab.store import insert_row

        await insert_row(
            "research_lab_source_add_reward_obligations",
            {
                "reward_ref": record.reward_ref,
                "adapter_id": record.adapter_id,
                "catalog_id": catalog_id or None,
                "miner_hotkey": record.miner_ref,
                "leg": record.leg,
                "reward_kind": record.reward_kind,
                "alpha_percent": record.alpha_percent,
                "reward_epochs": record.reward_epochs,
                "start_epoch": record.start_epoch,
                "trigger_evidence_doc": record.trigger_evidence or {},
                "public_label": record.public_label,
            },
        )
        await insert_row(
            "research_lab_source_add_reward_events",
            {
                "reward_ref": record.reward_ref,
                "seq": 0,
                "reward_status": record.state,
                "reason": "leg2_implementation_rider",
            },
        )
    return record.to_dict(), []
