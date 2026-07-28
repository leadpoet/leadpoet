"""Measured authority for epoch-scoped chain-realized Lab settlement."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Set

from gateway.research_lab.champion_settlement_v2 import (
    CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1,
    CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1,
    CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1,
    CHAIN_WEIGHT_OBSERVATION_RECEIPT_PURPOSE_V1,
    CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
    ChampionSettlementV2Error,
    _preliminary_finalized_bundle_authority_v1,
    build_chain_realized_settlement_package_v1,
    select_chain_realized_bundle_candidate_v1,
    validate_chain_weight_observation_v1,
    validate_finalized_allocation_authorities_v2,
)
from gateway.tee.coordinator_chain_source_v2 import CoordinatorChainSourceV2
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.attested_v2 import sha256_json, validate_receipt_graph


OP_OBSERVE_CHAIN_REALIZED_WEIGHTS_V1 = "observe_chain_realized_weights_v1"
OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1 = (
    "attest_chain_realized_settlement_v1"
)
CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1 = (
    CHAIN_WEIGHT_OBSERVATION_RECEIPT_PURPOSE_V1
)
CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1 = (
    CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1
)


class CoordinatorChainRealizedSettlementV1Error(RuntimeError):
    """A chain-realized settlement could not be proven inside the coordinator."""


def _receipt_by_root(
    graphs: Sequence[Mapping[str, Any]],
    *,
    receipt_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = []
    for graph in graphs:
        validate_receipt_graph(graph)
        receipts = {
            str(receipt.get("receipt_hash") or ""): dict(receipt)
            for receipt in graph.get("receipts") or ()
            if isinstance(receipt, Mapping)
        }
        if receipt_hash in receipts:
            matches.append((dict(graph), receipts[receipt_hash]))
    if len(matches) != 1:
        raise CoordinatorChainRealizedSettlementV1Error(
            "chain settlement parent receipt is missing or ambiguous"
        )
    return matches[0]


class CoordinatorChainRealizedSettlementV1:
    def __init__(
        self,
        *,
        reader: SupabaseSourceReaderV2,
        chain_source: CoordinatorChainSourceV2,
    ) -> None:
        self._reader = reader
        self._chain_source = chain_source

    def observe(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        netuid, epoch_id = self._request_scope(
            payload,
            context=context,
            schema_version=CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1,
        )
        rows = self._read(
            "latest_finalized_allocation_authority",
            {"netuid": netuid},
            context,
        )
        preliminary = [
            _preliminary_finalized_bundle_authority_v1(row) for row in rows
        ]
        if not preliminary:
            raise CoordinatorChainRealizedSettlementV1Error(
                "no finalized canonical bundle identifies the primary validator"
            )
        latest_block = max(int(item["finalized_block"]) for item in preliminary)
        latest_hotkeys = {
            str(item["validator_hotkey"])
            for item in preliminary
            if int(item["finalized_block"]) == latest_block
        }
        if len(latest_hotkeys) != 1:
            raise CoordinatorChainRealizedSettlementV1Error(
                "primary validator identity is ambiguous"
            )
        chain_state = self._chain_source.read_stateful_epoch_close_weights(
            netuid=netuid,
            epoch_id=epoch_id,
            validator_hotkey=next(iter(latest_hotkeys)),
            context=context,
        )
        observation = {
            "schema_version": CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
            "netuid": netuid,
            "epoch_id": epoch_id,
            "official_subnet_epoch_id": int(
                chain_state["official_subnet_epoch_id"]
            ),
            "cutover_mapping_hash": str(
                chain_state["cutover_mapping_hash"]
            ),
            "close_block": int(chain_state["close_block"]),
            "close_block_hash": str(chain_state["close_block_hash"]),
            "close_state_root": str(
                chain_state["close_header"]["state_root"]
            ),
            "next_epoch_block": int(chain_state["next_epoch_block"]),
            "next_epoch_block_hash": str(
                chain_state["next_epoch_block_hash"]
            ),
            "validator_hotkey": str(chain_state["validator_hotkey"]),
            "validator_uid": int(chain_state["validator_uid"]),
            "metagraph_hotkeys": list(chain_state["metagraph_hotkeys"]),
            "weights": [list(item) for item in chain_state["weights"]],
            "weights_storage_key": str(chain_state["weights_storage_key"]),
            "last_update_storage_key": str(
                chain_state["last_update_storage_key"]
            ),
            "last_update_block": int(chain_state["last_update_block"]),
            "last_update_block_hash": str(
                chain_state["last_update_block_hash"]
            ),
            "last_update_official_subnet_epoch_id": int(
                chain_state["last_update_official_subnet_epoch_id"]
            ),
            "active_source_epoch_id": int(
                chain_state["active_source_epoch_id"]
            ),
            "weights_vector_hash": sha256_json(
                {
                    "uids": [int(item[0]) for item in chain_state["weights"]],
                    "weights_u16": [
                        int(item[1]) for item in chain_state["weights"]
                    ],
                }
            ),
        }
        return validate_chain_weight_observation_v1(observation)

    def settle(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "netuid",
            "epoch_id",
            "observation",
            "observation_receipt_hash",
            "bundle_hash",
        }:
            raise CoordinatorChainRealizedSettlementV1Error(
                "chain settlement request fields are invalid"
            )
        netuid, epoch_id = self._request_scope(
            payload,
            context=context,
            schema_version=CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1,
            additional_fields={
                "observation",
                "observation_receipt_hash",
                "bundle_hash",
            },
        )
        observation = validate_chain_weight_observation_v1(
            payload["observation"]
        )
        if (
            int(observation["netuid"]) != netuid
            or int(observation["epoch_id"]) != epoch_id
        ):
            raise CoordinatorChainRealizedSettlementV1Error(
                "chain settlement observation scope differs"
            )
        observation_hash = sha256_json(observation)
        observation_receipt_hash = str(
            payload.get("observation_receipt_hash") or ""
        )
        _observation_graph, observation_receipt = _receipt_by_root(
            context.external_receipt_graphs,
            receipt_hash=observation_receipt_hash,
        )
        if (
            observation_receipt.get("role") != "gateway_coordinator"
            or observation_receipt.get("purpose")
            != CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1
            or observation_receipt.get("status") != "succeeded"
            or int(observation_receipt.get("epoch_id", -1)) != epoch_id
            or observation_receipt.get("output_root") != observation_hash
        ):
            raise CoordinatorChainRealizedSettlementV1Error(
                "chain settlement observation receipt differs"
            )

        rows = self._read(
            "finalized_authority_by_chain_vector",
            {
                "netuid": netuid,
                "uids": [int(item[0]) for item in observation["weights"]],
                "weights_u16": [
                    int(item[1]) for item in observation["weights"]
                ],
                "source_epoch_id": int(
                    observation["active_source_epoch_id"]
                ),
                "validator_hotkey": str(
                    observation["validator_hotkey"]
                ),
                "finalized_block": int(
                    observation["last_update_block"]
                ),
                "finalized_block_hash": str(
                    observation["last_update_block_hash"]
                ),
            },
            context,
        )
        try:
            selected = select_chain_realized_bundle_candidate_v1(
                rows,
                observation=observation,
            )
        except ChampionSettlementV2Error as exc:
            raise CoordinatorChainRealizedSettlementV1Error(str(exc)) from exc
        if selected["bundle_hash"] != str(payload.get("bundle_hash") or ""):
            raise CoordinatorChainRealizedSettlementV1Error(
                "host-selected chain settlement bundle is not authoritative"
            )
        selected_rows = [
            row
            for row in rows
            if str(row.get("bundle_hash") or "") == selected["bundle_hash"]
        ]
        if len(selected_rows) != 1:
            raise CoordinatorChainRealizedSettlementV1Error(
                "selected chain settlement bundle row is ambiguous"
            )
        finalization_receipt_hash = str(
            selected["finalization_receipt_hash"]
        )
        finalization_graph, _finalization_receipt = _receipt_by_root(
            context.external_receipt_graphs,
            receipt_hash=finalization_receipt_hash,
        )
        validated = validate_finalized_allocation_authorities_v2(
            selected_rows,
            finalization_graphs={
                finalization_receipt_hash: finalization_graph,
            },
        )
        if len(validated) != 1:
            raise CoordinatorChainRealizedSettlementV1Error(
                "selected chain settlement authority is invalid"
            )
        return build_chain_realized_settlement_package_v1(
            observation=observation,
            authority=selected,
        )

    def _read(
        self,
        policy_id: str,
        parameters: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> list[dict[str, Any]]:
        return self._reader.read(
            policy_id=policy_id,
            parameters=parameters,
            job_id=context.job_id,
            purpose=context.purpose,
            record_transport=context.record_transport,
            record_artifact=context.record_artifact,
        )

    @staticmethod
    def _request_scope(
        payload: Mapping[str, Any],
        *,
        context: ExecutionContextV2,
        schema_version: str,
        additional_fields: Optional[Set[str]] = None,
    ) -> tuple[int, int]:
        expected_fields = {"schema_version", "netuid", "epoch_id"} | set(
            additional_fields or ()
        )
        if (
            not isinstance(payload, Mapping)
            or set(payload) != expected_fields
            or payload.get("schema_version") != schema_version
        ):
            raise CoordinatorChainRealizedSettlementV1Error(
                "chain settlement request is invalid"
            )
        try:
            netuid = int(payload["netuid"])
            epoch_id = int(payload["epoch_id"])
        except (TypeError, ValueError) as exc:
            raise CoordinatorChainRealizedSettlementV1Error(
                "chain settlement request scope is invalid"
            ) from exc
        if (
            isinstance(payload["netuid"], bool)
            or isinstance(payload["epoch_id"], bool)
            or netuid <= 0
            or epoch_id < 0
            or epoch_id != int(context.epoch_id)
        ):
            raise CoordinatorChainRealizedSettlementV1Error(
                "chain settlement request scope is invalid"
            )
        return netuid, epoch_id
