"""Coordinator-owned reconstruction of database-backed V2 weight inputs."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping

from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_json,
    validate_signed_execution_receipt,
)
from leadpoet_canonical.weight_authority_v2 import (
    WEIGHT_INPUT_PURPOSES,
    gateway_weight_input_value_documents_v2,
)
from leadpoet_canonical.sourcing_history_v2 import (
    SourcingHistoryV2Error,
    rolling_sourcing_history_v2,
    validate_sourcing_epoch_v2,
)


_ALLOCATION_CATEGORIES = frozenset(
    {
        "research_lab_allocation",
        "champions",
        "reimbursements",
        "source_add_rewards",
    }
)


class CoordinatorWeightSourceV2Error(RuntimeError):
    """An authenticated source does not reproduce the proposed weight input."""


def _same(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _allocation_share(calculation: Mapping[str, Any]) -> float:
    allocation = calculation["research_lab_allocation_doc"]
    if isinstance(allocation, Mapping) and allocation.get("lab_cap_percent") not in (
        None,
        "",
    ):
        try:
            return max(0.0, min(1.0, float(allocation["lab_cap_percent"]) / 100.0))
        except (TypeError, ValueError):
            pass
    return float(calculation["research_lab_fallback_share"])


class CoordinatorWeightSourceV2:
    def __init__(self, reader: SupabaseSourceReaderV2) -> None:
        self._reader = reader

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        required = {
            "category",
            "calculation_snapshot",
            "gateway_authority_event_hash",
            "allocation_receipt",
            "leaderboard_window_start",
            "leaderboard_window_end",
        }
        if not isinstance(payload, Mapping):
            raise CoordinatorWeightSourceV2Error(
                "weight source payload fields are invalid"
            )
        category = str(payload.get("category") or "")
        expected_fields = (
            required | {"upstream_documents"}
            if category == "anomaly_adjustments"
            else required
        )
        if set(payload) != expected_fields:
            raise CoordinatorWeightSourceV2Error(
                "weight source payload fields are invalid"
            )
        expected = WEIGHT_INPUT_PURPOSES.get(category)
        if expected is None or expected[0] != "gateway_coordinator":
            raise CoordinatorWeightSourceV2Error(
                "weight source category is not coordinator-owned"
            )
        if context.purpose != expected[1]:
            raise CoordinatorWeightSourceV2Error(
                "weight source purpose differs from category"
            )
        calculation = payload.get("calculation_snapshot")
        if not isinstance(calculation, Mapping):
            raise CoordinatorWeightSourceV2Error(
                "weight calculation snapshot is invalid"
            )
        documents = gateway_weight_input_value_documents_v2(
            calculation_snapshot=calculation,
            gateway_authority_event_hash=str(
                payload.get("gateway_authority_event_hash") or ""
            ),
        )
        proposed = documents[category]
        if int(proposed["epoch_id"]) != int(context.epoch_id):
            raise CoordinatorWeightSourceV2Error(
                "weight source epoch differs from execution scope"
            )

        if category in _ALLOCATION_CATEGORIES:
            reconstructed = self._allocation_document(
                category=category,
                proposed=proposed,
                calculation=calculation,
                payload=payload,
                context=context,
            )
        elif category == "bans":
            reconstructed = self._ban_document(proposed, context)
        elif category == "fulfillment_rewards":
            reconstructed = self._fulfillment_document(
                proposed, calculation, context
            )
        elif category == "leaderboard":
            reconstructed = self._leaderboard_document(
                proposed=proposed,
                calculation=calculation,
                window_start=str(payload.get("leaderboard_window_start") or ""),
                window_end=str(payload.get("leaderboard_window_end") or ""),
                context=context,
            )
        elif category == "sourcing_history":
            reconstructed = self._sourcing_document(
                proposed=proposed,
                calculation=calculation,
                context=context,
            )
        elif category == "anomaly_adjustments":
            reconstructed = self._anomaly_document(
                proposed=proposed,
                upstream_documents=payload.get("upstream_documents"),
                context=context,
            )
        else:
            raise CoordinatorWeightSourceV2Error(
                "weight source category has no measured producer"
            )
        if not _same(reconstructed, proposed):
            raise CoordinatorWeightSourceV2Error(
                "%s authenticated source differs from calculation snapshot"
                % category
            )
        return reconstructed

    def _anomaly_document(
        self,
        *,
        proposed: Mapping[str, Any],
        upstream_documents: Any,
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        categories = (
            "research_lab_allocation",
            "fulfillment_rewards",
            "leaderboard",
            "bans",
            "sourcing_history",
        )
        if not isinstance(upstream_documents, Mapping) or set(
            upstream_documents
        ) != set(categories):
            raise CoordinatorWeightSourceV2Error(
                "anomaly upstream documents are incomplete"
            )
        roots = {}
        graph_roots = {}
        for graph in context.external_receipt_graphs:
            root_hash = str(graph.get("root_receipt_hash") or "")
            receipts = {
                str(item.get("receipt_hash") or ""): item
                for item in graph.get("receipts") or ()
                if isinstance(item, Mapping)
            }
            root = receipts.get(root_hash)
            if not isinstance(root, Mapping):
                continue
            for category in categories:
                role, purpose = WEIGHT_INPUT_PURPOSES[category]
                source_receipt = None
                if root.get("role") == role and root.get("purpose") == purpose:
                    source_receipt = root
                elif (
                    root.get("role") == "gateway_coordinator"
                    and root.get("purpose")
                    == "leadpoet.artifact_persistence.v2"
                ):
                    direct_parents = {
                        str(value)
                        for value in root.get("parent_receipt_hashes") or ()
                    }
                    matches = [
                        receipts[parent_hash]
                        for parent_hash in direct_parents
                        if parent_hash in receipts
                        and receipts[parent_hash].get("role") == role
                        and receipts[parent_hash].get("purpose") == purpose
                    ]
                    if len(matches) > 1:
                        raise CoordinatorWeightSourceV2Error(
                            "anomaly source receipt is duplicated"
                        )
                    if matches:
                        source_receipt = matches[0]
                if source_receipt is not None:
                    if category in roots:
                        raise CoordinatorWeightSourceV2Error(
                            "anomaly source receipt is duplicated"
                        )
                    roots[category] = source_receipt
                    graph_roots[category] = root_hash
        if set(roots) != set(categories):
            raise CoordinatorWeightSourceV2Error(
                "anomaly source receipt set is incomplete"
            )
        observed_parent_hashes = set(graph_roots.values())
        if observed_parent_hashes != set(context.parent_receipt_hashes):
            raise CoordinatorWeightSourceV2Error(
                "anomaly direct parents differ from measured sources"
            )
        normalized = {}
        for category in categories:
            document = upstream_documents[category]
            if not isinstance(document, Mapping):
                raise CoordinatorWeightSourceV2Error(
                    "anomaly upstream document is invalid"
                )
            document = dict(document)
            if (
                document.get("category") != category
                or int(document.get("epoch_id", -1)) != int(context.epoch_id)
                or roots[category].get("output_root") != sha256_json(document)
            ):
                raise CoordinatorWeightSourceV2Error(
                    "%s anomaly source document differs from its receipt" % category
                )
            normalized[category] = document
        post_adjustment_values = {
            "research_lab_allocation_doc": dict(
                normalized["research_lab_allocation"]["value"]["allocation_doc"]
            ),
            "fulfillment_rows": list(
                normalized["fulfillment_rewards"]["value"]["fulfillment_rows"]
            ),
            "leaderboard_entries": list(
                normalized["leaderboard"]["value"]["leaderboard_entries"]
            ),
            "banned_hotkeys": list(
                normalized["bans"]["value"]["banned_hotkeys"]
            ),
            "rolling_scores": list(
                normalized["sourcing_history"]["value"]["rolling_scores"]
            ),
        }
        return {
            **{key: proposed[key] for key in proposed if key != "value"},
            "value": {
                "post_adjustment_values_hash": sha256_json(
                    post_adjustment_values
                )
            },
        }

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

    def _allocation_document(
        self,
        *,
        category: str,
        proposed: Mapping[str, Any],
        calculation: Mapping[str, Any],
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        receipt = payload.get("allocation_receipt")
        if not isinstance(receipt, Mapping):
            raise CoordinatorWeightSourceV2Error(
                "allocation source receipt is missing"
            )
        validate_signed_execution_receipt(receipt)
        if (
            receipt.get("role") != "gateway_coordinator"
            or receipt.get("purpose") != "research_lab.allocation.v2"
            or int(receipt.get("epoch_id", -1)) != int(context.epoch_id)
            or receipt.get("receipt_hash") not in context.parent_receipt_hashes
        ):
            raise CoordinatorWeightSourceV2Error(
                "allocation source receipt is not a declared parent"
            )
        allocation = dict(calculation["research_lab_allocation_doc"])
        if receipt.get("output_root") != sha256_json({"allocation": allocation}):
            raise CoordinatorWeightSourceV2Error(
                "allocation source receipt does not bind allocation output"
            )
        if payload.get("gateway_authority_event_hash") != receipt.get("receipt_hash"):
            raise CoordinatorWeightSourceV2Error(
                "gateway allocation authority hash differs from receipt"
            )
        persisted = self._read(
            "attested_receipt_by_hash",
            {"receipt_hash": receipt["receipt_hash"]},
            context,
        )
        if len(persisted) != 1 or not _same(
            persisted[0].get("receipt_doc"), receipt
        ):
            raise CoordinatorWeightSourceV2Error(
                "allocation source receipt is not durably persisted"
            )
        rows = self._read(
            "research_lab_allocation_current",
            {"epoch_id": context.epoch_id, "netuid": proposed["netuid"]},
            context,
        )
        if len(rows) != 1 or not _same(rows[0].get("allocation_doc"), allocation):
            raise CoordinatorWeightSourceV2Error(
                "persisted Research Lab allocation differs from enclave output"
            )
        return dict(proposed)

    def _ban_document(
        self,
        proposed: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        rows = self._read("banned_hotkeys", {}, context)
        hotkeys = [str(row.get("hotkey") or "") for row in rows]
        if any(not hotkey for hotkey in hotkeys) or len(hotkeys) != len(set(hotkeys)):
            raise CoordinatorWeightSourceV2Error("banned hotkey rows are invalid")
        return {
            **{key: proposed[key] for key in proposed if key != "value"},
            "value": {
                "banned_hotkeys": sorted(hotkeys),
                "banned_lookup_ok": True,
            },
        }

    def _fulfillment_document(
        self,
        proposed: Mapping[str, Any],
        calculation: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        rows = self._read(
            "fulfillment_active_rewards",
            {"epoch_id": context.epoch_id},
            context,
        )
        per_miner: "OrderedDict[str, float]" = OrderedDict()
        for row in rows:
            hotkey = str(row.get("miner_hotkey") or "")
            try:
                reward = float(row["reward_pct"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CoordinatorWeightSourceV2Error(
                    "active fulfillment reward row is invalid"
                ) from exc
            if not hotkey:
                raise CoordinatorWeightSourceV2Error(
                    "active fulfillment reward hotkey is empty"
                )
            per_miner[hotkey] = per_miner.get(hotkey, 0.0) + reward
        allocation_share = _allocation_share(calculation)
        pool = max(
            0.0,
            1.0
            - allocation_share
            - float(calculation["champion_share"])
            - float(calculation["leaderboard_bonus_share"]),
        )
        raw_total = sum(per_miner.values())
        effective = 0.0
        effective_rows = OrderedDict()
        if bool(calculation["ff_enabled"]) and raw_total > 0:
            if raw_total <= pool:
                effective = raw_total
                effective_rows = per_miner
            else:
                effective = pool
                scale = pool / raw_total
                effective_rows = OrderedDict(
                    (hotkey, reward * scale)
                    for hotkey, reward in per_miner.items()
                )
        return {
            **{key: proposed[key] for key in proposed if key != "value"},
            "value": {
                "fulfillment_share": effective,
                "fulfillment_rows": [
                    {"hotkey": hotkey, "share": share}
                    for hotkey, share in effective_rows.items()
                ],
                "fulfillment_fetch_ok": True,
            },
        }

    def _leaderboard_document(
        self,
        *,
        proposed: Mapping[str, Any],
        calculation: Mapping[str, Any],
        window_start: str,
        window_end: str,
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        winner_rows = self._read(
            "fulfillment_leaderboard_winners",
            {"window_start": window_start, "window_end": window_end},
            context,
        )
        banned_rows = self._read("banned_hotkeys", {}, context)
        banned = {str(row.get("hotkey") or "") for row in banned_rows}
        per_miner: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for row in winner_rows:
            hotkey = str(row.get("miner_hotkey") or "")
            if not hotkey or hotkey in banned:
                continue
            record = per_miner.setdefault(
                hotkey, {"wins": 0, "total_reward_pct": 0.0}
            )
            record["wins"] += 1
            try:
                record["total_reward_pct"] += float(row.get("reward_pct") or 0.0)
            except (TypeError, ValueError):
                pass
        ranked = sorted(
            per_miner.items(),
            key=lambda item: (
                -item[1]["wins"],
                -item[1]["total_reward_pct"],
            ),
        )[:3]
        entries = (
            [
                {"miner_hotkey": hotkey, "wins": record["wins"]}
                for hotkey, record in ranked
            ]
            if bool(calculation["ff_enabled"])
            else []
        )
        return {
            **{key: proposed[key] for key in proposed if key != "value"},
            "value": {
                "leaderboard_bonus_share": calculation[
                    "leaderboard_bonus_share"
                ],
                "leaderboard_rank_shares": list(
                    calculation["leaderboard_rank_shares"]
                ),
                "leaderboard_entries": entries,
                "leaderboard_fetch_ok": True,
            },
        }

    def _sourcing_document(
        self,
        *,
        proposed: Mapping[str, Any],
        calculation: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        current_epoch = int(context.epoch_id)
        start_epoch = max(0, current_epoch - 30)
        if current_epoch == 0:
            rows = self._read("sourcing_epoch_inputs_empty", {}, context)
        else:
            rows = self._read(
                "sourcing_epoch_inputs",
                {"start_epoch": start_epoch, "end_epoch": current_epoch - 1},
                context,
            )
        epochs = []
        source_receipt_hashes = []
        for row in rows:
            try:
                source_doc = validate_sourcing_epoch_v2(row.get("source_doc"))
            except (SourcingHistoryV2Error, TypeError) as exc:
                raise CoordinatorWeightSourceV2Error(
                    "sourcing epoch document is invalid"
                ) from exc
            if (
                int(row.get("epoch_id", -1)) != int(source_doc["epoch_id"])
                or row.get("epoch_hash") != source_doc["epoch_hash"]
            ):
                raise CoordinatorWeightSourceV2Error(
                    "sourcing epoch row differs from its canonical document"
                )
            receipt = row.get("receipt_doc")
            if not isinstance(receipt, Mapping):
                raise CoordinatorWeightSourceV2Error(
                    "sourcing epoch receipt is missing"
                )
            validate_signed_execution_receipt(receipt)
            receipt_hash = str(receipt.get("receipt_hash") or "")
            if (
                receipt_hash != row.get("receipt_hash")
                or receipt.get("role") != "gateway_scoring"
                or receipt.get("purpose") != "qualification.sourcing_epoch.v2"
                or int(receipt.get("epoch_id", -1)) != int(source_doc["epoch_id"])
                or receipt.get("output_root") != sha256_json(source_doc)
                or receipt_hash not in context.parent_receipt_hashes
            ):
                raise CoordinatorWeightSourceV2Error(
                    "sourcing epoch receipt does not bind a declared source"
                )
            persisted = self._read(
                "attested_receipt_by_hash",
                {"receipt_hash": receipt_hash},
                context,
            )
            if len(persisted) != 1 or not _same(
                persisted[0].get("receipt_doc"), receipt
            ):
                raise CoordinatorWeightSourceV2Error(
                    "sourcing epoch receipt is not durably persisted"
                )
            epochs.append(source_doc)
            source_receipt_hashes.append(receipt_hash)
        if sorted(context.parent_receipt_hashes) != sorted(source_receipt_hashes):
            raise CoordinatorWeightSourceV2Error(
                "sourcing input parents differ from authenticated epoch receipts"
            )
        try:
            scores, lead_count = rolling_sourcing_history_v2(
                current_epoch=current_epoch,
                epochs=epochs,
            )
        except SourcingHistoryV2Error as exc:
            raise CoordinatorWeightSourceV2Error(
                "sourcing rolling history is invalid"
            ) from exc
        return {
            **{key: proposed[key] for key in proposed if key != "value"},
            "value": {
                "rolling_lead_count": lead_count,
                "rolling_scores": [
                    {"hotkey": hotkey, "score": score}
                    for hotkey, score in scores.items()
                ],
            },
        }
