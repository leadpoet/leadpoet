"""Measured authority for epoch-scoped chain-realized Lab settlement."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set

from gateway.research_lab.champion_settlement_v2 import (
    CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1,
    CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1,
    CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1,
    CHAIN_WEIGHT_OBSERVATION_RECEIPT_PURPOSE_V1,
    CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
    CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V2,
    ChampionSettlementV2Error,
    _preliminary_compact_finalized_bundle_authority_v2,
    _preliminary_finalized_bundle_authority_v1,
    build_compact_chain_realized_settlement_package_v2,
    build_chain_realized_settlement_package_v1,
    build_unattributed_chain_realized_settlement_package_v2,
    select_compact_chain_realized_bundle_candidate_v2,
    select_chain_realized_bundle_candidate_v1,
    validate_chain_weight_observation_v1,
    validate_finalized_allocation_authorities_v2,
)
from gateway.tee.coordinator_chain_source_v2 import CoordinatorChainSourceV2
from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionJobV2Error,
)
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.attested_v2 import sha256_json, validate_receipt_graph
from leadpoet_canonical.compact_auditor_authority_v2 import (
    verify_compact_published_weight_authority_v2,
)


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


def _finalized_authority_summary_v1(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the bounded fields used only to locate one full authority."""

    required = {
        "bundle_hash",
        "netuid",
        "epoch_id",
        "validator_hotkey",
        "finalized_block",
        "finalized_block_hash",
        "finalization_receipt_hash",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise CoordinatorChainRealizedSettlementV1Error(
            "finalized allocation authority summary fields are invalid"
        )
    try:
        netuid = int(row["netuid"])
        epoch_id = int(row["epoch_id"])
        finalized_block = int(row["finalized_block"])
    except (TypeError, ValueError) as exc:
        raise CoordinatorChainRealizedSettlementV1Error(
            "finalized allocation authority summary scope is invalid"
        ) from exc
    bundle_hash = str(row["bundle_hash"] or "").lower()
    finalization_receipt_hash = str(
        row["finalization_receipt_hash"] or ""
    ).lower()
    finalized_block_hash = str(row["finalized_block_hash"] or "").lower()
    validator_hotkey = str(row["validator_hotkey"] or "")
    if (
        isinstance(row["netuid"], bool)
        or isinstance(row["epoch_id"], bool)
        or isinstance(row["finalized_block"], bool)
        or netuid <= 0
        or epoch_id < 0
        or finalized_block < 0
        or not validator_hotkey
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", bundle_hash)
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", finalization_receipt_hash
        )
        or not re.fullmatch(r"[0-9a-f]{64}", finalized_block_hash)
    ):
        raise CoordinatorChainRealizedSettlementV1Error(
            "finalized allocation authority summary is invalid"
        )
    return {
        "bundle_hash": bundle_hash,
        "netuid": netuid,
        "epoch_id": epoch_id,
        "validator_hotkey": validator_hotkey,
        "finalized_block": finalized_block,
        "finalized_block_hash": finalized_block_hash,
        "finalization_receipt_hash": finalization_receipt_hash,
    }


def _compact_finalized_authority_summary_v2(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "bundle_hash",
        "compact_submission_hash",
        "netuid",
        "epoch_id",
        "validator_hotkey",
        "authority_hash",
        "lineage_id",
        "finalization_receipt_hash",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise CoordinatorChainRealizedSettlementV1Error(
            "compact finalized authority summary fields are invalid"
        )
    try:
        netuid = int(row["netuid"])
        epoch_id = int(row["epoch_id"])
    except (TypeError, ValueError) as exc:
        raise CoordinatorChainRealizedSettlementV1Error(
            "compact finalized authority summary scope is invalid"
        ) from exc
    hashes = {
        field: str(row.get(field) or "").lower()
        for field in (
            "bundle_hash",
            "compact_submission_hash",
            "authority_hash",
            "lineage_id",
            "finalization_receipt_hash",
        )
    }
    validator_hotkey = str(row.get("validator_hotkey") or "")
    if (
        isinstance(row["netuid"], bool)
        or isinstance(row["epoch_id"], bool)
        or netuid <= 0
        or epoch_id < 0
        or not validator_hotkey
        or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", value)
            for value in hashes.values()
        )
    ):
        raise CoordinatorChainRealizedSettlementV1Error(
            "compact finalized authority summary is invalid"
        )
    return {
        **hashes,
        "netuid": netuid,
        "epoch_id": epoch_id,
        "validator_hotkey": validator_hotkey,
    }


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
        expected_lineage_id: Optional[str] = None,
        expected_chain: Optional[str] = None,
        chain_signing_profile: Optional[Mapping[str, Any]] = None,
        boot_verifier: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._reader = reader
        self._chain_source = chain_source
        self._expected_lineage_id = str(expected_lineage_id or "")
        self._expected_chain = str(expected_chain or "")
        self._chain_signing_profile = (
            dict(chain_signing_profile)
            if isinstance(chain_signing_profile, Mapping)
            else None
        )
        self._boot_verifier = boot_verifier

    def _verify_compact_authority(
        self, row: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            not self._expected_lineage_id
            or not self._expected_chain
            or self._chain_signing_profile is None
            or self._boot_verifier is None
        ):
            raise CoordinatorChainRealizedSettlementV1Error(
                "compact chain settlement verifier is unavailable"
            )
        try:
            preliminary = _preliminary_compact_finalized_bundle_authority_v2(
                row
            )
            verified = verify_compact_published_weight_authority_v2(
                preliminary["authority_doc"],
                identity_cache=None,
                chain_signing_profile=self._chain_signing_profile,
                expected_lineage_id=self._expected_lineage_id,
                expected_chain=self._expected_chain,
                boot_verifier=self._boot_verifier,
            )
        except Exception as exc:
            raise CoordinatorChainRealizedSettlementV1Error(
                "compact chain settlement authority verification failed"
            ) from exc
        return preliminary, verified

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
        compact_summary_rows = self._read(
            "latest_compact_finalized_authority_summaries",
            {"netuid": netuid},
            context,
        )
        if compact_summary_rows:
            summaries = [
                _compact_finalized_authority_summary_v2(row)
                for row in compact_summary_rows
            ]
            latest_epoch = max(int(item["epoch_id"]) for item in summaries)
            latest = [
                item
                for item in summaries
                if int(item["epoch_id"]) == latest_epoch
            ]
            latest_hotkeys = {str(item["validator_hotkey"]) for item in latest}
            if len(latest_hotkeys) != 1:
                raise CoordinatorChainRealizedSettlementV1Error(
                    "primary validator identity is ambiguous"
                )
            selected_summary = min(latest, key=lambda item: item["bundle_hash"])
            authority_rows = self._read(
                "compact_finalized_authority_by_bundle_hash",
                {
                    "netuid": netuid,
                    "bundle_hash": selected_summary["bundle_hash"],
                },
                context,
            )
            if len(authority_rows) != 1:
                raise CoordinatorChainRealizedSettlementV1Error(
                    "latest compact finalized authority is unavailable"
                )
            authority, verified = self._verify_compact_authority(
                authority_rows[0]
            )
            for field in selected_summary:
                if authority.get(field) != selected_summary[field]:
                    raise CoordinatorChainRealizedSettlementV1Error(
                        "compact finalized authority summary differs at %s"
                        % field
                    )
            for field in (
                "bundle_hash",
                "netuid",
                "epoch_id",
                "validator_hotkey",
                "finalized_block",
                "finalized_block_hash",
                "finalization_receipt_hash",
            ):
                if authority.get(field) != verified.get(field):
                    raise CoordinatorChainRealizedSettlementV1Error(
                        "verified compact authority differs at %s" % field
                    )
        else:
            summary_rows = self._read(
                "latest_finalized_allocation_authority_summaries",
                {"netuid": netuid},
                context,
            )
            summaries = [
                _finalized_authority_summary_v1(row) for row in summary_rows
            ]
            if not summaries:
                raise CoordinatorChainRealizedSettlementV1Error(
                    "no finalized canonical bundle identifies the primary validator"
                )
            latest_block = max(
                int(item["finalized_block"]) for item in summaries
            )
            latest = [
                item
                for item in summaries
                if int(item["finalized_block"]) == latest_block
            ]
            latest_hotkeys = {
                str(item["validator_hotkey"]) for item in latest
            }
            if len(latest_hotkeys) != 1:
                raise CoordinatorChainRealizedSettlementV1Error(
                    "primary validator identity is ambiguous"
                )
            selected_summary = min(latest, key=lambda item: item["bundle_hash"])
            authority_rows = self._read(
                "finalized_allocation_authority_by_bundle_hash",
                {
                    "netuid": netuid,
                    "bundle_hash": selected_summary["bundle_hash"],
                },
                context,
            )
            if len(authority_rows) != 1:
                raise CoordinatorChainRealizedSettlementV1Error(
                    "latest finalized allocation authority is unavailable"
                )
            authority = _preliminary_finalized_bundle_authority_v1(
                authority_rows[0]
            )
            for field in (
                "bundle_hash",
                "netuid",
                "epoch_id",
                "validator_hotkey",
                "finalized_block",
                "finalized_block_hash",
                "finalization_receipt_hash",
            ):
                if authority.get(field) != selected_summary[field]:
                    raise CoordinatorChainRealizedSettlementV1Error(
                        "finalized allocation authority summary differs at %s"
                        % field
                    )
        chain_state = self._chain_source.read_stateful_epoch_close_weights(
            netuid=netuid,
            epoch_id=epoch_id,
            validator_hotkey=str(authority["validator_hotkey"]),
            context=context,
        )
        reveal_proof = None
        scheduled_source_epoch = chain_state[
            "scheduled_reveal_source_epoch_id"
        ]
        scheduled_subnet_epoch = chain_state[
            "scheduled_reveal_subnet_epoch_id"
        ]
        if (
            scheduled_source_epoch is not None
            and scheduled_subnet_epoch is not None
        ):
            reveal_rows = self._read(
                "compact_finalized_authority_by_identity",
                {
                    "netuid": netuid,
                    "source_epoch_id": int(scheduled_source_epoch),
                    "validator_hotkey": str(authority["validator_hotkey"]),
                },
                context,
            )
            candidate_proofs = []
            if len(reveal_rows) <= 10:
                for reveal_row in reveal_rows:
                    try:
                        reveal_authority, reveal_verified = (
                            self._verify_compact_authority(reveal_row)
                        )
                        if any(
                            reveal_authority.get(field)
                            != reveal_verified.get(field)
                            for field in (
                                "bundle_hash",
                                "netuid",
                                "epoch_id",
                                "validator_hotkey",
                                "finalized_block",
                                "finalized_block_hash",
                                "finalization_receipt_hash",
                            )
                        ):
                            raise CoordinatorChainRealizedSettlementV1Error(
                                "verified reveal authority differs"
                            )
                        candidate_proof = (
                            self._chain_source.read_timelocked_reveal_proof(
                                chain_state=chain_state,
                                authority=reveal_authority,
                                context=context,
                            )
                        )
                        if candidate_proof is not None:
                            candidate_proofs.append(candidate_proof)
                    except CoordinatorChainRealizedSettlementV1Error:
                        continue
            if len(candidate_proofs) == 1:
                reveal_proof = candidate_proofs[0]
        observation = {
            "schema_version": CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V2,
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
            "latest_commit_source_epoch_id": int(
                chain_state["latest_commit_source_epoch_id"]
            ),
            "epoch_start_block": int(chain_state["epoch_start_block"]),
            "epoch_start_block_hash": str(
                chain_state["epoch_start_block_hash"]
            ),
            "reveal_window_start_block": int(
                chain_state["reveal_window_start_block"]
            ),
            "reveal_window_start_block_hash": str(
                chain_state["reveal_window_start_block_hash"]
            ),
            "scheduled_reveal_subnet_epoch_id": chain_state[
                "scheduled_reveal_subnet_epoch_id"
            ],
            "scheduled_reveal_source_epoch_id": chain_state[
                "scheduled_reveal_source_epoch_id"
            ],
            "revealed_bundle_hash": (
                str(reveal_proof["bundle_hash"])
                if reveal_proof is not None
                else None
            ),
            "reveal_proof": (
                dict(reveal_proof) if reveal_proof is not None else None
            ),
            "subnet_reveal_period_epochs": int(
                chain_state["subnet_reveal_period_epochs"]
            ),
            "reveal_period_storage_key": str(
                chain_state["reveal_period_storage_key"]
            ),
            "reveal_period_storage_override": chain_state[
                "reveal_period_storage_override"
            ],
            "reveal_period_metadata_hash": chain_state[
                "reveal_period_metadata_hash"
            ],
            "reveal_period_runtime_spec_version": int(
                chain_state["reveal_period_runtime_spec_version"]
            ),
            "chain_signing_profile": dict(
                chain_state["chain_signing_profile"]
            ),
            "chain_signing_profile_hash": str(
                chain_state["chain_signing_profile_hash"]
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
            "authority_mode",
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
                "authority_mode",
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
        try:
            parent_authority_graphs = (
                context.external_receipt_authority_graphs()
            )
        except ExecutionJobV2Error as exc:
            raise CoordinatorChainRealizedSettlementV1Error(
                "chain settlement parent authority is invalid"
            ) from exc
        _observation_graph, observation_receipt = _receipt_by_root(
            parent_authority_graphs,
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

        authority_mode = str(payload.get("authority_mode") or "")
        if authority_mode not in {"finalized_bundle", "unattributed"}:
            raise CoordinatorChainRealizedSettlementV1Error(
                "chain settlement authority mode is invalid"
            )
        observation_v2 = (
            observation["schema_version"]
            == CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V2
        )
        source_epoch_id = int(
            observation[
                "latest_commit_source_epoch_id"
                if observation_v2
                else "active_source_epoch_id"
            ]
        )
        if observation_v2:
            revealed_bundle_hash = observation["revealed_bundle_hash"]
            use_compact = revealed_bundle_hash is not None
            rows = (
                self._read(
                    "compact_finalized_authority_by_bundle_hash",
                    {
                        "netuid": netuid,
                        "bundle_hash": str(revealed_bundle_hash),
                    },
                    context,
                )
                if revealed_bundle_hash is not None
                else []
            )
            try:
                selected = (
                    select_compact_chain_realized_bundle_candidate_v2(
                        rows,
                        observation=observation,
                    )
                    if rows
                    else None
                )
            except ChampionSettlementV2Error as exc:
                raise CoordinatorChainRealizedSettlementV1Error(
                    str(exc)
                ) from exc
        else:
            compact_cutover_rows = self._read(
                "compact_finalized_authority_cutover",
                {"netuid": netuid},
                context,
            )
            compact_cutover_epoch: int | None = None
            if compact_cutover_rows:
                if len(compact_cutover_rows) != 1 or set(
                    compact_cutover_rows[0]
                ) != {"epoch_id"}:
                    raise CoordinatorChainRealizedSettlementV1Error(
                        "compact authority cutover is invalid"
                    )
                raw_cutover_epoch = compact_cutover_rows[0]["epoch_id"]
                if isinstance(raw_cutover_epoch, bool):
                    raise CoordinatorChainRealizedSettlementV1Error(
                        "compact authority cutover is invalid"
                    )
                try:
                    compact_cutover_epoch = int(raw_cutover_epoch)
                except (TypeError, ValueError) as exc:
                    raise CoordinatorChainRealizedSettlementV1Error(
                        "compact authority cutover is invalid"
                    ) from exc
            use_compact = (
                compact_cutover_epoch is not None
                and source_epoch_id >= compact_cutover_epoch
            )
            policy_id = (
                "compact_finalized_authority_by_identity"
                if use_compact
                else "finalized_authority_by_chain_vector"
            )
            parameters = {
                "netuid": netuid,
                "source_epoch_id": source_epoch_id,
                "validator_hotkey": str(observation["validator_hotkey"]),
            }
            if not use_compact:
                parameters.update(
                    {
                        "uids": [
                            int(item[0]) for item in observation["weights"]
                        ],
                        "weights_u16": [
                            int(item[1]) for item in observation["weights"]
                        ],
                        "finalized_block": int(
                            observation["last_update_block"]
                        ),
                        "finalized_block_hash": str(
                            observation["last_update_block_hash"]
                        ),
                    }
                )
            rows = self._read(policy_id, parameters, context)
            try:
                selected = (
                    select_compact_chain_realized_bundle_candidate_v2(
                        rows,
                        observation=observation,
                    )
                    if use_compact
                    else (
                        select_chain_realized_bundle_candidate_v1(
                            rows, observation=observation
                        )
                        if rows
                        else None
                    )
                )
            except ChampionSettlementV2Error as exc:
                raise CoordinatorChainRealizedSettlementV1Error(str(exc)) from exc
        if authority_mode == "unattributed":
            if (
                selected is not None
                or payload.get("bundle_hash") is not None
                or (
                    observation_v2
                    and observation["revealed_bundle_hash"] is not None
                )
            ):
                raise CoordinatorChainRealizedSettlementV1Error(
                    "unattributed settlement has finalized bundle authority"
                )
            return build_unattributed_chain_realized_settlement_package_v2(
                observation=observation,
            )
        if selected is None:
            raise CoordinatorChainRealizedSettlementV1Error(
                "no finalized canonical bundle matches the active chain vector"
            )
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
            parent_authority_graphs,
            receipt_hash=finalization_receipt_hash,
        )
        if use_compact:
            compact_authority, verified = self._verify_compact_authority(
                selected_rows[0]
            )
            if compact_authority != selected:
                raise CoordinatorChainRealizedSettlementV1Error(
                    "selected compact chain settlement authority changed"
                )
            return build_compact_chain_realized_settlement_package_v2(
                observation=observation,
                authority=compact_authority,
                verified=verified,
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
