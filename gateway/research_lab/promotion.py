"""Gateway-owned Research Lab promotion and private model lineage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.store import (
    canonical_hash,
    create_candidate_artifact,
    create_candidate_promotion_event,
    create_champion_reward_obligation,
    create_private_model_version,
    create_private_model_version_event,
    create_private_repo_commit_event,
    select_many,
)
from leadpoet_verifier.economics import build_champion_reward_obligation
from research_lab.auto_research_prompt import AutoResearchCandidateDraft, build_validated_candidate_manifest
from research_lab.engine_v1 import ComponentRegistry
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

    if rows:
        row = rows[0]
        artifact = _load_valid_artifact(str(row["private_model_manifest_uri"]))
        if artifact.model_artifact_hash != str(row["model_artifact_hash"]):
            raise RuntimeError("active private model artifact hash does not match lineage row")
        if artifact.manifest_hash != str(row["private_model_manifest_hash"]):
            raise RuntimeError("active private model manifest hash does not match lineage row")
        return ActivePrivateModel(artifact=artifact, version_row=row)

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
        active_component_registry: ComponentRegistry | None = None,
    ) -> dict[str, Any]:
        if not self.config.auto_promotion_enabled:
            return {"status": "disabled"}

        active = await load_active_private_model(self.config, register_bootstrap=True)
        active_parent = active.artifact.model_artifact_hash
        candidate_parent = str(candidate.get("parent_artifact_hash") or score_bundle.get("parent_artifact_hash") or "")
        improvement_points = float((score_bundle.get("aggregates") or {}).get("mean_delta") or 0.0)
        delta_lcb = float((score_bundle.get("aggregates") or {}).get("delta_lcb") or 0.0)
        threshold = float(self.config.improvement_threshold_points)
        rolling_window_hash = str(score_bundle.get("icp_set_hash") or "")
        score_bundle_id = str(score_bundle_row.get("score_bundle_id") or "")

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
            },
        )

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
            return await self._queue_rebased_candidate(
                candidate=candidate,
                score_bundle_id=score_bundle_id,
                active=active,
                active_component_registry=active_component_registry,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold=threshold,
            )

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

        if not self.config.auto_commit_enabled:
            return {"status": "promotion_passed_auto_commit_disabled"}

        try:
            new_artifact = await asyncio.to_thread(
                _auto_commit_private_repo,
                self.config,
                candidate,
                score_bundle_row,
                score_bundle,
                active.artifact,
            )
            if active.version_row:
                await create_private_model_version_event(
                    private_model_version_id=str(active.version_row["private_model_version_id"]),
                    event_type="superseded",
                    version_status="superseded",
                    reason="superseded_by_research_lab_promotion",
                    event_doc={"source_candidate_id": str(candidate["candidate_id"])},
                )
            version_row, _version_event = await create_private_model_version(
                artifact_manifest=new_artifact.to_dict(),
                manifest_uri=new_artifact.manifest_uri,
                source_candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                redacted_version_doc={
                    "source": "gateway_auto_commit",
                    "model_artifact_hash": new_artifact.model_artifact_hash,
                    "private_model_manifest_hash": new_artifact.manifest_hash,
                    "git_commit_sha": new_artifact.git_commit_sha,
                    "component_registry_version": new_artifact.component_registry_version,
                    "scoring_adapter_version": new_artifact.scoring_adapter_version,
                },
                version_status="active",
                reason="research_lab_candidate_promoted",
            )
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                private_model_version_id=str(version_row["private_model_version_id"]),
                event_type="active_version_created",
                promotion_status="merged",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={"new_model_artifact_hash": new_artifact.model_artifact_hash},
            )
            reward_status = await self._maybe_create_champion_reward(
                candidate=candidate,
                score_bundle_row=score_bundle_row,
                score_bundle=score_bundle,
                improvement_points=improvement_points,
                threshold=threshold,
            )
            return {"status": "merged", "private_model_version_id": str(version_row["private_model_version_id"]), **reward_status}
        except Exception as exc:
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="promotion_failed",
                promotion_status="failed",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={"error": _safe_text(str(exc))},
            )
            logger.exception("Research Lab promotion failed for candidate %s", candidate.get("candidate_id"))
            return {"status": "promotion_failed", "error": str(exc)[:200]}

    async def _queue_rebased_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_id: str,
        active: ActivePrivateModel,
        active_component_registry: ComponentRegistry | None,
        rolling_window_hash: str,
        improvement_points: float,
        threshold: float,
    ) -> dict[str, Any]:
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=score_bundle_id,
            event_type="stale_parent_detected",
            promotion_status="rebase_required",
            active_parent_artifact_hash=active.artifact.model_artifact_hash,
            candidate_parent_artifact_hash=str(candidate.get("parent_artifact_hash") or ""),
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={"action": "queue_derived_candidate_against_active_model"},
        )
        if active_component_registry is None:
            return {"status": "stale_parent_rebase_required"}
        request = _rebased_candidate_request(candidate, active.artifact, active_component_registry)
        derived_row, _event = await create_candidate_artifact(request)
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            derived_candidate_id=str(derived_row["candidate_id"]),
            source_score_bundle_id=score_bundle_id,
            event_type="rebase_queued",
            promotion_status="rebenchmarking",
            active_parent_artifact_hash=active.artifact.model_artifact_hash,
            candidate_parent_artifact_hash=str(candidate.get("parent_artifact_hash") or ""),
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={
                "derived_candidate_artifact_hash": str(derived_row["candidate_artifact_hash"]),
                "derived_parent_artifact_hash": active.artifact.model_artifact_hash,
            },
        )
        return {"status": "stale_parent_rebased_candidate_queued", "derived_candidate_id": str(derived_row["candidate_id"])}

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


def _rebased_candidate_request(
    candidate: Mapping[str, Any],
    artifact: PrivateModelArtifactManifest,
    registry: ComponentRegistry,
) -> Any:
    from gateway.research_lab.models import ResearchLabCandidateArtifactCreateRequest

    patch_manifest = dict(candidate["candidate_patch_manifest"])
    hypothesis = candidate.get("hypothesis_doc") if isinstance(candidate.get("hypothesis_doc"), Mapping) else {}
    draft = AutoResearchCandidateDraft(
        failure_mode=str(hypothesis.get("failure_mode") or "Rebase typed patch onto current active model.")[:600],
        mechanism=str(hypothesis.get("mechanism") or "Same typed patch payload, latest active parent.")[:900],
        expected_improvement=str(hypothesis.get("expected_improvement") or "Expected to improve over current active model.")[:900],
        risk=str(hypothesis.get("risk") or "Patch may not transfer to latest parent.")[:600],
        predicted_delta=float(hypothesis.get("predicted_delta") or 1.0),
        falsifier=str(hypothesis.get("falsifier") or "proxy_score"),
        patch_type=str(patch_manifest["patch_type"]),
        target_component_id=str(patch_manifest["target_component_id"]),
        patch_doc=dict(patch_manifest.get("patch_doc") or {}),
        redacted_summary=str(patch_manifest.get("redacted_summary") or candidate.get("redacted_public_summary") or "")[:900],
    )
    sequence = int(canonical_hash({"candidate_id": candidate["candidate_id"], "parent": artifact.model_artifact_hash}).split(":", 1)[1][:8], 16)
    rebased_patch, rebased_hypothesis, _patch = build_validated_candidate_manifest(
        draft=draft,
        artifact_manifest=artifact,
        component_registry=registry,
        run_id=str(candidate["run_id"]),
        sequence=sequence,
        miner_brief_ref=f"rebased_from:{candidate['candidate_id']}",
    )
    return ResearchLabCandidateArtifactCreateRequest(
        run_id=candidate["run_id"],
        ticket_id=candidate["ticket_id"],
        receipt_id=candidate.get("receipt_id") or None,
        miner_hotkey=str(candidate["miner_hotkey"]),
        island=str(candidate.get("island") or "generalist"),
        private_model_manifest=artifact.to_dict(),
        candidate_patch_manifest=rebased_patch.to_dict(),
        hypothesis_doc={
            **rebased_hypothesis.to_dict(),
            "rebased_from_candidate_id": str(candidate["candidate_id"]),
        },
        redacted_public_summary=rebased_patch.redacted_summary,
    )


def _auto_commit_private_repo(
    config: ResearchLabGatewayConfig,
    candidate: Mapping[str, Any],
    score_bundle_row: Mapping[str, Any],
    score_bundle: Mapping[str, Any],
    active_artifact: PrivateModelArtifactManifest,
) -> PrivateModelArtifactManifest:
    required = {
        "RESEARCH_LAB_PRIVATE_REPO_URL": config.private_repo_url,
        "RESEARCH_LAB_PRIVATE_PATCH_APPLIER_CMD": config.private_patch_applier_cmd,
        "RESEARCH_LAB_PRIVATE_TEST_CMD": config.private_test_cmd,
        "RESEARCH_LAB_PRIVATE_BUILD_CMD": config.private_build_cmd,
        "RESEARCH_LAB_PRIVATE_ARTIFACT_MANIFEST_OUTPUT": config.private_artifact_manifest_output,
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        raise RuntimeError("private repo auto-commit is missing config: " + ", ".join(missing))

    branch = config.private_repo_branch or "main"
    repo_hash = canonical_hash({"private_repo_url": config.private_repo_url})
    with tempfile.TemporaryDirectory(prefix="research-lab-private-repo-") as tmp:
        repo_dir = Path(tmp) / "repo"
        _run(["git", "clone", "--branch", branch, "--single-branch", config.private_repo_url, str(repo_dir)], cwd=Path(tmp))
        _run(["git", "config", "user.name", "Leadpoet Research Lab"], cwd=repo_dir)
        _run(["git", "config", "user.email", "research-lab@leadpoet.local"], cwd=repo_dir)
        _run(["git", "checkout", "-B", branch], cwd=repo_dir)
        patch_path = Path(tmp) / "candidate_patch.json"
        active_manifest_path = Path(tmp) / "active_manifest.json"
        score_bundle_path = Path(tmp) / "score_bundle.json"
        patch_path.write_text(json.dumps(candidate["candidate_patch_manifest"], sort_keys=True), encoding="utf-8")
        active_manifest_path.write_text(json.dumps(active_artifact.to_dict(), sort_keys=True), encoding="utf-8")
        score_bundle_path.write_text(json.dumps(score_bundle, sort_keys=True), encoding="utf-8")
        env = {
            **os.environ,
            "RESEARCH_LAB_PATCH_MANIFEST_PATH": str(patch_path),
            "RESEARCH_LAB_ACTIVE_MANIFEST_PATH": str(active_manifest_path),
            "RESEARCH_LAB_SCORE_BUNDLE_PATH": str(score_bundle_path),
            "RESEARCH_LAB_CANDIDATE_ID": str(candidate["candidate_id"]),
            "RESEARCH_LAB_SCORE_BUNDLE_ID": str(score_bundle_row["score_bundle_id"]),
        }
        asyncio.run(
            create_private_repo_commit_event(
                commit_status="started",
                branch_name=branch,
                candidate_id=str(candidate["candidate_id"]),
                score_bundle_id=str(score_bundle_row["score_bundle_id"]),
                private_repo_ref_hash=repo_hash,
                event_doc={"step": "clone_complete"},
            )
        )
        _run_shell(config.private_patch_applier_cmd, cwd=repo_dir, env=env)
        asyncio.run(
            create_private_repo_commit_event(
                commit_status="patch_applied",
                branch_name=branch,
                candidate_id=str(candidate["candidate_id"]),
                score_bundle_id=str(score_bundle_row["score_bundle_id"]),
                private_repo_ref_hash=repo_hash,
                event_doc={"step": "patch_applied"},
            )
        )
        _run_shell(config.private_test_cmd, cwd=repo_dir, env=env)
        asyncio.run(
            create_private_repo_commit_event(
                commit_status="tests_passed",
                branch_name=branch,
                candidate_id=str(candidate["candidate_id"]),
                score_bundle_id=str(score_bundle_row["score_bundle_id"]),
                private_repo_ref_hash=repo_hash,
                event_doc={"step": "tests_passed"},
            )
        )
        _run(["git", "add", "-A"], cwd=repo_dir)
        _run(
            [
                "git",
                "commit",
                "-m",
                (
                    "Research Lab promotion "
                    f"{str(candidate['candidate_id']).split(':')[-1][:12]} "
                    f"score_bundle={str(score_bundle_row['score_bundle_id']).split(':')[-1][:12]}"
                ),
            ],
            cwd=repo_dir,
        )
        git_commit_sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip()
        env["RESEARCH_LAB_PRIVATE_COMMIT_SHA"] = git_commit_sha
        asyncio.run(
            create_private_repo_commit_event(
                commit_status="committed",
                branch_name=branch,
                candidate_id=str(candidate["candidate_id"]),
                score_bundle_id=str(score_bundle_row["score_bundle_id"]),
                git_commit_sha=git_commit_sha,
                private_repo_ref_hash=repo_hash,
                event_doc={"step": "committed"},
            )
        )
        _run_shell(config.private_build_cmd, cwd=repo_dir, env=env)
        manifest_path = Path(config.private_artifact_manifest_output)
        if not manifest_path.is_absolute():
            manifest_path = repo_dir / manifest_path
        if not manifest_path.exists():
            raise RuntimeError("private build did not produce artifact manifest output")
        artifact = PrivateModelArtifactManifest.from_mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
        errors = validate_private_model_artifact_manifest(artifact)
        if errors:
            raise RuntimeError("built private artifact manifest failed validation: " + "; ".join(errors))
        asyncio.run(
            create_private_repo_commit_event(
                commit_status="build_passed",
                branch_name=branch,
                candidate_id=str(candidate["candidate_id"]),
                score_bundle_id=str(score_bundle_row["score_bundle_id"]),
                git_commit_sha=git_commit_sha,
                private_repo_ref_hash=repo_hash,
                event_doc={"step": "build_passed", "model_artifact_hash": artifact.model_artifact_hash},
            )
        )
        _run(["git", "push", "origin", f"HEAD:{branch}"], cwd=repo_dir)
        asyncio.run(
            create_private_repo_commit_event(
                commit_status="pushed",
                branch_name=branch,
                candidate_id=str(candidate["candidate_id"]),
                score_bundle_id=str(score_bundle_row["score_bundle_id"]),
                git_commit_sha=git_commit_sha,
                private_repo_ref_hash=repo_hash,
                event_doc={"step": "pushed", "model_artifact_hash": artifact.model_artifact_hash},
            )
        )
        return artifact


def _run(cmd: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=True, timeout=1800)
    return completed.stdout.strip()


def _run_shell(cmd: str, *, cwd: Path, env: Mapping[str, str]) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=dict(env),
        shell=True,
        check=True,
        text=True,
        capture_output=True,
        timeout=3600,
    )
    return completed.stdout.strip()


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
