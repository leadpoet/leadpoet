"""Authenticated source reconstruction for measured Research Lab rewards."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from gateway.tee.coordinator_chain_source_v2 import CoordinatorChainSourceV2
from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionJobV2Error,
)
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_runtime_catalog_v2,
)
from gateway.research_lab.source_add_provenance import (
    rebuild_attested_provenance_result_v2,
)
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.attested_v2 import sha256_json


class CoordinatorRewardSourceV2Error(RuntimeError):
    """A reward proposal differs from authenticated database or chain state."""


class CoordinatorRewardSourceV2:
    """Replace host-selected SOURCE_ADD inputs with measured source values."""

    def __init__(
        self,
        *,
        reader: SupabaseSourceReaderV2,
        chain_source: CoordinatorChainSourceV2,
        config_supplier: Callable[[], Any],
    ) -> None:
        self._reader = reader
        self._chain_source = chain_source
        self._config_supplier = config_supplier

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {
            "decision_kind",
            "decision_payload",
        }:
            raise CoordinatorRewardSourceV2Error(
                "reward authority payload fields are invalid"
            )
        kind = str(payload.get("decision_kind") or "")
        if kind == "champion_migration":
            return self._resolve_champion_migration(
                payload=payload,
                context=context,
            )
        if kind == "source_add_migration":
            return self._resolve_source_add_migration(
                payload=payload,
                context=context,
            )
        if kind not in {"source_add_leg1", "source_add_leg2"}:
            raise CoordinatorRewardSourceV2Error(
                "reward source kind is unsupported"
            )
        proposed = payload.get("decision_payload")
        if not isinstance(proposed, Mapping):
            raise CoordinatorRewardSourceV2Error("reward decision input is invalid")
        decision = dict(proposed)
        adapter_id = str(decision.get("adapter_id") or "")
        if not adapter_id:
            raise CoordinatorRewardSourceV2Error("SOURCE_ADD adapter is missing")

        config = self._config_supplier()
        chain_state = self._chain_source.read_finalized_metagraph(
            netuid=int(getattr(config, "netuid", 71) or 71),
            context=context,
        )
        chain_epoch = int(chain_state.get("workflow_epoch_id", -1))
        if chain_epoch != int(context.epoch_id):
            raise CoordinatorRewardSourceV2Error(
                "reward execution epoch differs from finalized chain state"
            )
        expected_start_epoch = chain_epoch + 1
        if int(decision.get("start_epoch") or -1) != expected_start_epoch:
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD start epoch differs from finalized chain state"
            )

        alpha_attr = (
            "source_add_leg1_alpha_percent"
            if kind == "source_add_leg1"
            else "source_add_leg2_alpha_percent"
        )
        expected_alpha = float(
            getattr(config, alpha_attr, 0.2 if kind == "source_add_leg1" else 5.0)
        )
        if expected_alpha <= 0.0:
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD reward leg is disabled"
            )
        expected_epochs = int(getattr(config, "lab_reward_epochs", 20) or 20)
        if (
            float(decision.get("alpha_percent") or 0.0) != expected_alpha
            or int(decision.get("reward_epochs") or 0) != expected_epochs
        ):
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD reward policy differs from measured configuration"
            )

        decision["existing_rewards"] = self._read(
            "source_add_rewards_by_adapter",
            {"adapter_id": adapter_id},
            context,
        )
        if kind == "source_add_leg1":
            provenance = decision.get("provenance_result")
            if (
                not isinstance(provenance, Mapping)
                or set(provenance)
                != {
                    "schema_version",
                    "submission_id",
                    "precheck_status",
                    "reasons",
                    "precheck_doc",
                }
                or provenance.get("schema_version")
                != "leadpoet.source_add_provenance_result.v2"
                or provenance.get("precheck_status")
                != "provenance_precheck_passed"
                or not isinstance(provenance.get("precheck_doc"), Mapping)
                or not isinstance(provenance.get("reasons"), list)
            ):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 credible provenance result is invalid"
                )
            submission_id = str(provenance.get("submission_id") or "")
            submission_rows = self._read(
                "source_add_submission_by_id",
                {"submission_id": submission_id},
                context,
            )
            if len(submission_rows) != 1:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD submission owner is missing or ambiguous"
                )
            submission = submission_rows[0]
            precheck_doc = submission.get("precheck_doc")
            submission_doc = submission.get("submission_doc")
            if not isinstance(precheck_doc, Mapping) or not isinstance(
                submission_doc, Mapping
            ):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 durable provenance is missing"
                )
            try:
                measured_provenance = rebuild_attested_provenance_result_v2(
                    submission_id=str(submission.get("submission_id") or ""),
                    precheck_status=str(submission.get("precheck_status") or ""),
                    precheck_doc=precheck_doc,
                    submission_doc=submission_doc,
                )
            except ValueError as exc:
                raise CoordinatorRewardSourceV2Error(str(exc)) from exc
            if (
                str(submission.get("adapter_id") or "") != adapter_id
                or str(submission.get("miner_hotkey") or "")
                != str(decision.get("miner_ref") or "")
                or str(submission.get("precheck_status") or "")
                != "provenance_precheck_passed"
                or measured_provenance != dict(provenance)
            ):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 owner or status differs from measured submission"
                )
            try:
                graphs = list(context.external_receipt_authority_graphs())
            except ExecutionJobV2Error as exc:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 provenance parent is invalid"
                ) from exc
            if len(graphs) != 1:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 requires one provenance parent"
                )
            graph = graphs[0]
            root_hash = str(graph.get("root_receipt_hash") or "")
            root = next(
                (
                    item
                    for item in graph.get("receipts") or ()
                    if isinstance(item, Mapping)
                    and item.get("receipt_hash") == root_hash
                ),
                None,
            )
            provenance_hash = sha256_json(dict(provenance))
            if (
                not isinstance(root, Mapping)
                or root.get("role") != "gateway_coordinator"
                or root.get("purpose") != "research_lab.source_add_provenance.v2"
                or root.get("status") != "succeeded"
                or root.get("output_root") != provenance_hash
                or tuple(context.parent_receipt_hashes) != (root_hash,)
                or not isinstance(submission_doc, Mapping)
                or str(submission_doc.get("provenance_receipt_hash") or "")
                != root_hash
                or (
                    bool(submission_doc.get("provenance_artifact_hash"))
                    and str(submission_doc["provenance_artifact_hash"])
                    != provenance_hash
                )
            ):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 provenance receipt differs from measured state"
                )
            trigger = decision.get("trigger_evidence")
            expected_trigger = {
                "provenance_precheck_passed": True,
                "submission_id": submission_id,
                "precheck_status": "provenance_precheck_passed",
                "provenance_receipt_hash": root_hash,
                "provenance_artifact_hash": provenance_hash,
                "provenance_result_hash": provenance_hash,
            }
            if not isinstance(trigger, Mapping) or dict(trigger) != expected_trigger:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 trigger differs from measured provenance"
                )
        else:
            judge_result = decision.get("judge_result")
            if not isinstance(judge_result, Mapping):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 judge result is invalid"
                )
            try:
                graphs = list(context.external_receipt_authority_graphs())
            except ExecutionJobV2Error as exc:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 judge parent is invalid"
                ) from exc
            if len(graphs) != 1:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 requires one judge parent"
                )
            graph = graphs[0]
            root_hash = str(graph.get("root_receipt_hash") or "")
            root = next(
                (
                    item
                    for item in graph.get("receipts") or ()
                    if isinstance(item, Mapping)
                    and item.get("receipt_hash") == root_hash
                ),
                None,
            )
            if (
                not isinstance(root, Mapping)
                or tuple(context.parent_receipt_hashes) != (root_hash,)
                or root.get("role") != "gateway_scoring"
                or root.get("purpose") != "research_lab.source_add_judge.v2"
                or root.get("status") != "succeeded"
                or root.get("output_root") != sha256_json(dict(judge_result))
            ):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 judge receipt differs"
                )
            verdict = (
                judge_result.get("verdict")
                if isinstance(judge_result, Mapping)
                else None
            )
            if (
                not isinstance(verdict, Mapping)
                or verdict.get("verdict") != "helped"
                or verdict.get("source_used") is not True
            ):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 signed judge did not approve the reward"
                )
            rows = self._read(
                "source_add_provisioning_by_adapter",
                {"adapter_id": adapter_id},
                context,
            )
            if len(rows) != 1:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD provisioned owner is missing or ambiguous"
                )
            owner = str(rows[0].get("miner_hotkey") or "")
            if not owner or owner != str(decision.get("miner_ref") or ""):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD reward owner differs from measured provisioning"
                )
            matched_adapter = str(verdict.get("adapter_id") or "") == adapter_id
            matched_registry = str(verdict.get("registry_provider_id") or "") == str(
                rows[0].get("registry_provider_id") or ""
            )
            if not matched_adapter and not matched_registry:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 judge differs from measured provisioning"
                )
            trigger = decision.get("trigger_evidence")
            if not isinstance(trigger, Mapping):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 trigger evidence is invalid"
                )
            expected_trigger = {
                "llm_judge_passed": True,
                "llm_verdict": "helped",
                "llm_confidence": float(verdict.get("confidence") or 0.0),
                "source_used": True,
                "adapter_id": str(verdict.get("adapter_id") or ""),
                "registry_provider_id": str(
                    verdict.get("registry_provider_id") or ""
                ),
                "evidence_summary": str(verdict.get("evidence_summary") or "")[:1000],
                "reason_codes": [
                    str(item) for item in (verdict.get("reason_codes") or ())
                ][:20],
                "judge_model": str(verdict.get("model_id") or ""),
                "judge_doc_hash": str(verdict.get("judge_doc_hash") or ""),
                "provider_usage": dict(verdict.get("provider_usage") or {}),
            }
            for field, expected in expected_trigger.items():
                if trigger.get(field) != expected:
                    raise CoordinatorRewardSourceV2Error(
                        "SOURCE_ADD Leg 2 trigger differs from signed judge"
                    )
        return {
            "decision_kind": kind,
            "decision_payload": decision,
        }

    def _resolve_champion_migration(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        proposed = payload.get("decision_payload")
        if not isinstance(proposed, Mapping) or set(proposed) != {
            "champion_reward_id"
        }:
            raise CoordinatorRewardSourceV2Error(
                "champion migration request fields are invalid"
            )
        reward_id = str(proposed.get("champion_reward_id") or "")
        reward = self._one(
            "champion_reward_by_id",
            {"champion_reward_id": reward_id},
            context,
        )
        if str(reward.get("champion_reward_id") or "") != reward_id:
            raise CoordinatorRewardSourceV2Error(
                "champion migration reward differs from measured state"
            )
        bundle_id = str(reward.get("score_bundle_id") or "")
        score_bundle = self._one(
            "score_bundle_by_id",
            {"score_bundle_id": bundle_id},
            context,
        )
        bundle_doc = score_bundle.get("score_bundle_doc")
        if (
            str(score_bundle.get("score_bundle_id") or "") != bundle_id
            or not isinstance(bundle_doc, Mapping)
        ):
            raise CoordinatorRewardSourceV2Error(
                "champion migration score bundle differs from measured state"
            )
        return {
            "decision_kind": "champion_migration",
            "decision_payload": {
                "reward_row": dict(reward),
                "score_bundle": {
                    "score_bundle_id": bundle_id,
                    "score_bundle_hash": str(
                        score_bundle.get("score_bundle_hash") or ""
                    ),
                    "score_bundle_doc": dict(bundle_doc),
                },
            },
        }

    def _resolve_source_add_migration(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        proposed = payload.get("decision_payload")
        if not isinstance(proposed, Mapping) or set(proposed) != {"reward_ref"}:
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD migration request fields are invalid"
            )
        reward_ref = str(proposed.get("reward_ref") or "")
        reward = self._one(
            "source_add_reward_by_ref",
            {"reward_ref": reward_ref},
            context,
        )
        if str(reward.get("reward_ref") or "") != reward_ref:
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD migration reward differs from measured state"
            )
        trigger = reward.get("trigger_evidence_doc")
        if not isinstance(trigger, Mapping):
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD migration trigger is invalid"
            )
        submission_id = str(trigger.get("submission_id") or "")
        submission = self._one(
            "source_add_submission_by_id",
            {"submission_id": submission_id},
            context,
        )
        return {
            "decision_kind": "source_add_migration",
            "decision_payload": {
                "reward_row": dict(reward),
                "source_submission": dict(submission),
            },
        }


    def _one(
        self,
        policy_id: str,
        parameters: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        rows = self._read(policy_id, parameters, context)
        if len(rows) != 1:
            raise CoordinatorRewardSourceV2Error(
                "%s source row is missing or ambiguous" % policy_id
            )
        return rows[0]

    def catalog_snapshot(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {"limit"}:
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD catalog snapshot payload is invalid"
            )
        if int(payload.get("limit") or 0) != 200:
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD catalog snapshot limit is invalid"
            )
        rows = self._read("source_add_provisioning_eligible", {}, context)
        private_registry_rows = self._read(
            "provider_registry_recent",
            {},
            context,
        )
        runtime_catalog = build_source_add_runtime_catalog_v2(rows)
        result = {
            "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
            "provisioned_sources": rows,
            "provisioned_sources_hash": sha256_json(rows),
            "private_registry_rows": private_registry_rows,
            "private_registry_rows_hash": sha256_json(private_registry_rows),
            "runtime_catalog": runtime_catalog,
            "runtime_catalog_hash": str(runtime_catalog["catalog_hash"]),
        }
        return result

    def _read(
        self,
        policy_id: str,
        parameters: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> list[Dict[str, Any]]:
        return self._reader.read(
            policy_id=policy_id,
            parameters=parameters,
            job_id=context.job_id,
            purpose=context.purpose,
            record_transport=context.record_transport,
            record_artifact=context.record_artifact,
        )
