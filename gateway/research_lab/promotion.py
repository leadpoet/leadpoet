"""Gateway-owned Research Lab promotion and private model lineage."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Mapping

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.store import (
    canonical_hash,
    create_candidate_promotion_event,
    create_champion_reward_obligation,
    create_private_model_version,
    create_private_model_version_event,
    select_many,
)
from leadpoet_verifier.economics import build_champion_reward_obligation
from research_lab.eval import (
    PrivateModelArtifactManifest,
    load_private_artifact_manifest,
    validate_private_model_artifact_manifest,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivePrivateModel:
    artifact: PrivateModelArtifactManifest
    version_row: dict[str, Any] | None = None


async def load_active_private_model(
    config: ResearchLabGatewayConfig,
    *,
    register_bootstrap: bool = False,
) -> ActivePrivateModel:
    """Load the current active private model.

    The lineage table is authoritative when present. If it has not been
    initialized yet, the configured manifest URI is used as a bootstrap source.
    """

    try:
        rows = await select_many(
            "research_lab_private_model_version_current",
            filters=(("current_version_status", "active"),),
            order_by=(("current_status_at", True),),
            limit=1,
        )
    except Exception as exc:
        logger.warning("research_lab_active_model_lineage_unavailable: %s", str(exc)[:200])
        rows = []

    stale_active_rows: list[tuple[dict[str, Any], str, dict[str, str]]] = []
    for row in rows:
        try:
            artifact = _load_valid_artifact(str(row["private_model_manifest_uri"]))
        except Exception as exc:
            stale_active_rows.append((row, "manifest_load_failed", {"error": _safe_text(str(exc))}))
            logger.warning(
                "research_lab_active_model_lineage_row_load_failed: version=%s error=%s",
                _short_ref(row.get("private_model_version_id")),
                _safe_text(str(exc))[:200],
            )
            continue

        row_artifact_hash = str(row["model_artifact_hash"])
        row_manifest_hash = str(row["private_model_manifest_hash"])
        if artifact.model_artifact_hash == row_artifact_hash and artifact.manifest_hash == row_manifest_hash:
            return ActivePrivateModel(artifact=artifact, version_row=row)

        stale_active_rows.append(
            (
                row,
                "mutable_manifest_hash_mismatch",
                {
                    "row_model_artifact_hash": row_artifact_hash,
                    "loaded_model_artifact_hash": artifact.model_artifact_hash,
                    "row_private_model_manifest_hash": row_manifest_hash,
                    "loaded_private_model_manifest_hash": artifact.manifest_hash,
                },
            )
        )
        logger.warning(
            "research_lab_active_model_lineage_stale: version=%s row_artifact=%s loaded_artifact=%s",
            _short_ref(row.get("private_model_version_id")),
            _short_ref(row_artifact_hash),
            _short_ref(artifact.model_artifact_hash),
        )

    artifact = _load_valid_artifact(config.private_model_manifest_uri)
    version_row = None
    if register_bootstrap:
        try:
            version_row, _event = await create_private_model_version(
                artifact_manifest=artifact.to_dict(),
                manifest_uri=config.private_model_manifest_uri,
                redacted_version_doc={
                    "source": "bootstrap_private_model_manifest_uri",
                    "model_artifact_hash": artifact.model_artifact_hash,
                    "private_model_manifest_hash": artifact.manifest_hash,
                    "git_commit_sha": artifact.git_commit_sha,
                    "component_registry_version": artifact.component_registry_version,
                    "scoring_adapter_version": artifact.scoring_adapter_version,
                },
                version_status="active",
                reason="bootstrap_private_model_manifest_uri",
            )
            for stale_row, stale_reason, stale_doc in stale_active_rows:
                stale_version_id = str(stale_row.get("private_model_version_id") or "")
                if not stale_version_id or stale_version_id == str(version_row.get("private_model_version_id") or ""):
                    continue
                try:
                    await create_private_model_version_event(
                        private_model_version_id=stale_version_id,
                        event_type="superseded",
                        version_status="superseded",
                        reason="superseded_by_current_private_model_manifest",
                        event_doc={
                            "reason": stale_reason,
                            "replacement_private_model_version_id": str(version_row["private_model_version_id"]),
                            "replacement_model_artifact_hash": artifact.model_artifact_hash,
                            "replacement_private_model_manifest_hash": artifact.manifest_hash,
                            **stale_doc,
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "research_lab_stale_active_model_supersede_failed: version=%s error=%s",
                        _short_ref(stale_version_id),
                        _safe_text(str(exc))[:200],
                    )
        except Exception as exc:
            logger.warning("research_lab_active_model_bootstrap_write_failed: %s", str(exc)[:200])
    return ActivePrivateModel(artifact=artifact, version_row=version_row)


async def latest_public_benchmark_summary() -> dict[str, Any]:
    """Return the latest sanitized public benchmark report, or a safe fallback."""

    try:
        rows = await select_many(
            "research_lab_public_benchmark_report_current",
            filters=(("current_report_status", "published"),),
            order_by=(("benchmark_date", True), ("created_at", True)),
            limit=1,
        )
    except Exception as exc:
        logger.warning("research_lab_public_benchmark_summary_unavailable: %s", str(exc)[:200])
        rows = []
    if rows:
        report_doc = rows[0].get("report_doc")
        if isinstance(report_doc, Mapping):
            return dict(report_doc)
    return {
        "schema_version": "1.0",
        "report_type": "research_lab_public_daily_benchmark",
        "status": "unavailable",
        "guidance": "No sanitized daily benchmark report has been published yet.",
    }


class ResearchLabPromotionController:
    """Process scored candidates into active private model versions."""

    def __init__(self, config: ResearchLabGatewayConfig, *, worker_ref: str):
        self.config = config
        self.worker_ref = worker_ref

    async def process_scored_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidate_parent = str(candidate.get("parent_artifact_hash") or score_bundle.get("parent_artifact_hash") or "")
        candidate_kind = str(candidate.get("candidate_kind") or "patch")
        improvement_points = float((score_bundle.get("aggregates") or {}).get("mean_delta") or 0.0)
        delta_lcb = float((score_bundle.get("aggregates") or {}).get("delta_lcb") or 0.0)
        threshold = float(self.config.improvement_threshold_points)
        rolling_window_hash = str(score_bundle.get("icp_set_hash") or "")
        score_bundle_id = str(score_bundle_row.get("score_bundle_id") or "")

        if not self.config.auto_promotion_enabled:
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="promotion_checked",
                promotion_status="checked",
                active_parent_artifact_hash=candidate_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "delta_lcb": round(delta_lcb, 6),
                    "auto_commit_enabled": self.config.auto_commit_enabled,
                    "candidate_kind": candidate_kind,
                    "auto_promotion_enabled": False,
                },
            )
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="promotion_disabled",
                promotion_status="disabled",
                active_parent_artifact_hash=candidate_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "auto_promotion_enabled": False,
                    "auto_commit_enabled": self.config.auto_commit_enabled,
                    "candidate_kind": candidate_kind,
                    "delta_lcb": round(delta_lcb, 6),
                },
            )
            return {"status": "disabled"}

        active = await load_active_private_model(self.config, register_bootstrap=True)
        active_parent = active.artifact.model_artifact_hash

        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=score_bundle_id,
            event_type="promotion_checked",
            promotion_status="checked",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={
                "delta_lcb": round(delta_lcb, 6),
                "auto_commit_enabled": self.config.auto_commit_enabled,
                "candidate_kind": candidate_kind,
            },
        )

        if candidate_kind != "image_build":
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="unsupported_candidate_kind",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "candidate_kind": candidate_kind,
                    "reason": "patch_candidates_are_legacy_read_only",
                },
            )
            return {"status": "rejected_legacy_patch_candidate"}

        if improvement_points < threshold or delta_lcb < float(self.config.improvement_min_delta_lcb):
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="below_threshold",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "mean_delta": round(improvement_points, 6),
                    "delta_lcb": round(delta_lcb, 6),
                    "min_delta_lcb": self.config.improvement_min_delta_lcb,
                },
            )
            return {"status": "rejected_below_threshold"}

        if candidate_parent != active_parent:
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="stale_parent_detected",
                promotion_status="rebase_required",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "candidate_kind": "image_build",
                    "action": "rescore_candidate_image_against_current_parent",
                },
            )
            return {"status": "stale_parent_needs_rescore"}

        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=score_bundle_id,
            event_type="promotion_passed",
            promotion_status="passed",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={"auto_commit_enabled": self.config.auto_commit_enabled},
        )

        return await self._promote_built_image_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            active=active,
            active_parent=active_parent,
            candidate_parent=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold=threshold,
        )

    async def _promote_built_image_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        active: ActivePrivateModel,
        active_parent: str,
        candidate_parent: str,
        rolling_window_hash: str,
        improvement_points: float,
        threshold: float,
    ) -> dict[str, Any]:
        manifest_doc = candidate.get("candidate_model_manifest_doc")
        if not isinstance(manifest_doc, Mapping):
            raise RuntimeError("image_build candidate missing candidate_model_manifest_doc")
        new_artifact = PrivateModelArtifactManifest.from_mapping(manifest_doc)
        errors = validate_private_model_artifact_manifest(new_artifact)
        if errors:
            raise RuntimeError("candidate image manifest failed validation: " + "; ".join(errors))
        if str(score_bundle.get("candidate_artifact_hash") or "") != new_artifact.model_artifact_hash:
            raise RuntimeError("score bundle candidate artifact does not match built image manifest")
        if active.version_row:
            await create_private_model_version_event(
                private_model_version_id=str(active.version_row["private_model_version_id"]),
                event_type="superseded",
                version_status="superseded",
                reason="superseded_by_research_lab_image_build_promotion",
                event_doc={"source_candidate_id": str(candidate["candidate_id"])},
            )
        version_row, _version_event = await create_private_model_version(
            artifact_manifest=new_artifact.to_dict(),
            manifest_uri=new_artifact.manifest_uri,
            source_candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row["score_bundle_id"]),
            redacted_version_doc={
                "source": "gateway_code_edit_image_build",
                "model_artifact_hash": new_artifact.model_artifact_hash,
                "private_model_manifest_hash": new_artifact.manifest_hash,
                "git_commit_sha": new_artifact.git_commit_sha,
                "component_registry_version": new_artifact.component_registry_version,
                "scoring_adapter_version": new_artifact.scoring_adapter_version,
                "candidate_source_diff_hash": candidate.get("candidate_source_diff_hash"),
            },
            version_status="active",
            reason="research_lab_image_build_candidate_promoted",
        )
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row["score_bundle_id"]),
            private_model_version_id=str(version_row["private_model_version_id"]),
            event_type="active_version_created",
            promotion_status="merged",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={
                "new_model_artifact_hash": new_artifact.model_artifact_hash,
                "candidate_kind": "image_build",
            },
        )
        reward_status = await self._maybe_create_champion_reward(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            improvement_points=improvement_points,
            threshold=threshold,
        )
        return {
            "status": "merged",
            "private_model_version_id": str(version_row["private_model_version_id"]),
            **reward_status,
        }

    async def _maybe_create_champion_reward(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        improvement_points: float,
        threshold: float,
    ) -> dict[str, Any]:
        uid = await _resolve_miner_uid(str(candidate["miner_hotkey"]))
        if uid is None:
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=str(score_bundle_row.get("score_bundle_id") or ""),
                event_type="champion_reward_pending_uid",
                promotion_status="reward_pending_uid",
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={"miner_hotkey_ref": canonical_hash(str(candidate["miner_hotkey"]))},
            )
            return {"champion_reward_status": "uid_resolution_pending"}
        policy = {
            **self.config.reimbursement_policy_doc(enabled=True),
            "champion_threshold_points": threshold,
            "reward_epochs": self.config.lab_reward_epochs,
        }
        obligation_input = {
            "uid": uid,
            "miner_uid": uid,
            "miner_hotkey": str(candidate["miner_hotkey"]),
            "island": str(candidate.get("island") or "generalist"),
            "candidate_id": str(candidate["candidate_id"]),
            "score_bundle_id": str(score_bundle_row["score_bundle_id"]),
            "run_id": str(candidate["run_id"]),
            "evaluation_epoch": int(score_bundle.get("evaluation_epoch") or self.config.evaluation_epoch or 0),
            "start_epoch": int(score_bundle.get("evaluation_epoch") or self.config.evaluation_epoch or 0) + 1,
            "improvement_points": improvement_points,
            "threshold_points": threshold,
            "daily_icp_counts": _daily_counts_from_score_bundle(score_bundle),
        }
        obligation = build_champion_reward_obligation(obligation_input, policy)
        if obligation["status"] != "active":
            return {"champion_reward_status": obligation["status"]}
        row, _event = await create_champion_reward_obligation(
            obligation=obligation,
            ticket_id=str(candidate["ticket_id"]),
            obligation_doc={
                "policy_id": str(policy["policy_id"]),
                "source": "gateway_promotion_event",
                "source_score_bundle_hash": str(score_bundle.get("score_bundle_hash") or ""),
            },
        )
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row["score_bundle_id"]),
            event_type="champion_reward_created",
            promotion_status="reward_created",
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={"champion_reward_id": str(row["champion_reward_id"])},
        )
        return {"champion_reward_status": "created", "champion_reward_id": str(row["champion_reward_id"])}


def _load_valid_artifact(uri: str) -> PrivateModelArtifactManifest:
    artifact = PrivateModelArtifactManifest.from_mapping(load_private_artifact_manifest(uri))
    errors = validate_private_model_artifact_manifest(artifact)
    if errors:
        raise RuntimeError("private artifact manifest failed validation: " + "; ".join(errors))
    return artifact


async def _resolve_miner_uid(hotkey: str) -> int | None:
    try:
        from gateway.qualification.utils.chain import get_metagraph

        metagraph = await get_metagraph()
        hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
        return hotkeys.index(hotkey) if hotkey in hotkeys else None
    except Exception as exc:
        logger.warning("research_lab_miner_uid_resolution_failed: %s", str(exc)[:200])
        return None


def _daily_counts_from_score_bundle(score_bundle: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
    for item in aggregates.get("per_icp_results", []) or []:
        ref = str(item.get("icp_ref") or "")
        match = re.search(r"qualification_private_icp_sets:(\d+):", ref)
        day = match.group(1) if match else ref.split(":")[0]
        if day:
            counts[day] = counts.get(day, 0) + 1
    return counts


def _safe_text(value: str) -> str:
    text = value or ""
    for marker in ("sk-or-", "service_role", "openrouter_api_key"):
        text = text.replace(marker, "[redacted]")
    return text[:500]


def _short_ref(value: Any) -> str:
    text = str(value or "")
    if len(text) <= 24:
        return text
    return f"{text[:14]}...{text[-6:]}"
