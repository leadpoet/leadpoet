"""Independent source reconstruction for authoritative Research Lab allocation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Set, Tuple

from gateway.research_lab.allocations import (
    ACTIVE_CHAMPION_STATUSES,
    ACTIVE_REIMBURSEMENT_STATUSES,
    ACTIVE_SCHEDULE_STATUSES,
    _champion_paid_alpha_to_date_from_snapshots,
    _champion_obligation_caps,
    _champion_replay_obligation,
    _epoch_active,
    _source_add_paid_alpha_to_date_from_snapshots,
)
from gateway.research_lab.champion_settlement_v2 import (
    merge_finalized_allocation_histories_v2,
    validate_finalized_allocation_authorities_v2,
    validate_legacy_settlement_migrations_v2,
)
from gateway.research_lab.alpha_pricing import (
    compute_alpha_price_valuation,
    inject_alpha_price_valuation,
    static_alpha_price_fallback,
)
from gateway.research_lab.bundles import contains_secret_material
from gateway.tee.coordinator_chain_source_v2 import CoordinatorChainSourceV2
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from gateway.tee.reward_executor_v2 import (
    champion_reward_row_projection_v2,
    reimbursement_reward_row_projection_v2,
    source_add_reward_row_projection_v2,
)
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_json,
    validate_receipt_graph,
    validate_signed_execution_receipt,
)
from leadpoet_verifier.economics import allocate_research_lab_epoch


class CoordinatorAllocationSourceV2Error(RuntimeError):
    """Authenticated allocation sources are incomplete or inconsistent."""


def _same(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _receipt_subgraph(
    graph: Mapping[str, Any],
    *,
    root_receipt_hash: str,
) -> dict[str, Any]:
    validate_receipt_graph(graph)
    receipts_by_hash = {
        str(receipt.get("receipt_hash") or ""): receipt
        for receipt in graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
    }
    if root_receipt_hash not in receipts_by_hash:
        raise CoordinatorAllocationSourceV2Error(
            "declared allocation parent is absent from receipt graphs"
        )
    selected_hashes: set[str] = set()

    def select(receipt_hash: str) -> None:
        if receipt_hash in selected_hashes:
            return
        receipt = receipts_by_hash.get(receipt_hash)
        if not isinstance(receipt, Mapping):
            raise CoordinatorAllocationSourceV2Error(
                "declared allocation parent ancestry is incomplete"
            )
        selected_hashes.add(receipt_hash)
        for parent_hash in receipt.get("parent_receipt_hashes") or ():
            select(str(parent_hash))

    select(root_receipt_hash)
    selected_receipts = [
        dict(receipt)
        for receipt in graph["receipts"]
        if str(receipt["receipt_hash"]) in selected_hashes
    ]
    selected_boot_hashes = {
        str(receipt["boot_identity_hash"]) for receipt in selected_receipts
    }
    selected_scopes = {
        (str(receipt["job_id"]), str(receipt["purpose"]))
        for receipt in selected_receipts
    }
    subgraph = {
        "schema_version": graph["schema_version"],
        "root_receipt_hash": root_receipt_hash,
        "boot_identities": [
            dict(identity)
            for identity in graph["boot_identities"]
            if str(identity["boot_identity_hash"]) in selected_boot_hashes
        ],
        "receipts": selected_receipts,
        "transport_attempts": [
            dict(attempt)
            for attempt in graph["transport_attempts"]
            if (str(attempt["job_id"]), str(attempt["purpose"]))
            in selected_scopes
        ],
        "host_operations": [
            dict(record)
            for record in graph["host_operations"]
            if (
                str(record["request"]["job_id"]),
                str(record["request"]["purpose"]),
            )
            in selected_scopes
        ],
    }
    validate_receipt_graph(subgraph)
    return subgraph


def _receipt_graphs_by_declared_root(
    graphs: Sequence[Mapping[str, Any]],
    declared_roots: Sequence[str],
) -> dict[str, dict[str, Any]]:
    by_root: dict[str, dict[str, Any]] = {}
    for root in declared_roots:
        matches = [
            graph
            for graph in graphs
            if any(
                isinstance(receipt, Mapping)
                and str(receipt.get("receipt_hash") or "") == root
                for receipt in graph.get("receipts") or ()
            )
        ]
        if not matches:
            raise CoordinatorAllocationSourceV2Error(
                "declared allocation parent is absent from receipt graphs"
            )
        derived = _receipt_subgraph(
            matches[0],
            root_receipt_hash=str(root),
        )
        for graph in matches[1:]:
            candidate = _receipt_subgraph(
                graph,
                root_receipt_hash=str(root),
            )
            if not _same(derived, candidate):
                raise CoordinatorAllocationSourceV2Error(
                    "declared allocation parent graphs conflict"
                )
        by_root[str(root)] = derived
    return by_root


class CoordinatorAllocationSourceV2:
    """Rebuild allocation inputs from measured database and chain reads."""

    def __init__(
        self,
        *,
        reader: SupabaseSourceReaderV2,
        chain_source: CoordinatorChainSourceV2,
        config_supplier: Callable[[], Any],
        network_supplier: Callable[[], str],
    ) -> None:
        self._reader = reader
        self._chain_source = chain_source
        self._config_supplier = config_supplier
        self._network_supplier = network_supplier

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {"epoch", "netuid"}:
            raise CoordinatorAllocationSourceV2Error(
                "allocation authority payload fields are invalid"
            )
        epoch = self._non_negative_int(payload.get("epoch"), "epoch")
        netuid = self._non_negative_int(payload.get("netuid"), "netuid")
        if epoch != int(context.epoch_id):
            raise CoordinatorAllocationSourceV2Error(
                "allocation epoch differs from execution scope"
            )
        config = self._config_supplier()
        required_parent_hashes: Set[str] = set()
        policy, chain_state = self._policy_and_chain_state(
            config=config,
            epoch=epoch,
            netuid=netuid,
            context=context,
        )
        hotkey_uids = {
            str(hotkey): uid
            for uid, hotkey in enumerate(chain_state["metagraph"]["hotkeys"])
        }
        reimbursement_rows, reimbursement_skipped = self._reimbursements(
            epoch=epoch,
            policy=policy,
            hotkey_uids=hotkey_uids,
            context=context,
            required_parents=required_parent_hashes,
        )
        champion_source_rows = self._read(
            "allocation_champion_rewards", {"epoch_id": epoch}, context
        )
        source_add_rows = self._read(
            "allocation_source_add_rewards", {"epoch_id": epoch}, context
        )
        finalized_reward_history = self._finalized_champion_history(
            epoch=epoch,
            netuid=netuid,
            champion_rows=tuple(champion_source_rows) + tuple(source_add_rows),
            context=context,
            required_parents=required_parent_hashes,
        )
        champion_rows, champion_skipped = self._champions(
            epoch=epoch,
            rows=champion_source_rows,
            history=finalized_reward_history,
            hotkey_uids=hotkey_uids,
            context=context,
            required_parents=required_parent_hashes,
        )
        source_add_obligations, source_add_skipped = self._source_add(
            epoch=epoch,
            rows=source_add_rows,
            history=finalized_reward_history,
            hotkey_uids=hotkey_uids,
            context=context,
            required_parents=required_parent_hashes,
        )
        required_parent_hash_list = sorted(required_parent_hashes)
        observed_parent_hashes = sorted(set(context.parent_receipt_hashes))
        if required_parent_hash_list != observed_parent_hashes:
            raise CoordinatorAllocationSourceV2Error(
                "allocation parent receipt set differs from authenticated sources"
            )

        source_add_present = bool(source_add_obligations or source_add_skipped)
        allocation_inputs: Dict[str, Any] = {
            "epoch": epoch,
            "policy": policy,
            "active_reimbursement_obligations": reimbursement_rows,
            "active_champion_obligations": champion_rows,
        }
        if source_add_present:
            allocation_inputs["active_source_add_obligations"] = source_add_obligations
        allocation = allocate_research_lab_epoch(
            epoch,
            policy,
            reimbursement_rows,
            champion_rows,
            active_source_add_obligations=source_add_obligations,
        )
        source_state: Dict[str, Any] = {
            "epoch": epoch,
            "netuid": netuid,
            "policy_id": str(policy["policy_id"]),
            "policy": policy,
            "reimbursement_obligation_count": len(reimbursement_rows),
            "champion_obligation_count": len(champion_rows),
            "reimbursement_obligations": reimbursement_rows,
            "champion_obligations": champion_rows,
            "skipped": {
                "reimbursements": reimbursement_skipped,
                "champions": champion_skipped,
            },
        }
        if source_add_present:
            source_state["source_add_obligation_count"] = len(
                source_add_obligations
            )
            source_state["source_add_obligations"] = source_add_obligations
            source_state["skipped"]["source_add"] = source_add_skipped
        if contains_secret_material(source_state) or contains_secret_material(allocation):
            raise CoordinatorAllocationSourceV2Error(
                "allocation authority output contains secret material"
            )
        return {
            "allocation": allocation,
            "allocation_inputs": allocation_inputs,
            "source_state": source_state,
            "source_state_hash": sha256_json(source_state),
        }

    def _policy_and_chain_state(
        self,
        *,
        config: Any,
        epoch: int,
        netuid: int,
        context: ExecutionContextV2,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        policy = dict(config.reimbursement_policy_doc(enabled=True))
        network = str(self._network_supplier() or "finney")
        dynamic_enabled = bool(config.reimbursement_dynamic_alpha_price_enabled)
        require_live = bool(config.reimbursement_require_live_alpha_price)
        if dynamic_enabled:
            try:
                chain_state = self._chain_source.resolve_live_prices(
                    netuid=netuid,
                    context=context,
                )
                valuation = compute_alpha_price_valuation(
                    network=network,
                    netuid=netuid,
                    epoch=epoch,
                    tao_per_alpha=chain_state["tao_per_alpha"],
                    tao_usd=chain_state["tao_usd"],
                    miner_alpha_per_epoch=config.reimbursement_miner_alpha_per_epoch,
                    pricing_status="live",
                    fetched_at=str(chain_state["fetched_at"]),
                )
            except Exception as exc:
                if require_live:
                    raise CoordinatorAllocationSourceV2Error(
                        "required live alpha price is unavailable"
                    ) from exc
                chain_state = self._chain_source.read_finalized_metagraph(
                    netuid=netuid,
                    context=context,
                    attempt_number=3,
                )
                valuation = static_alpha_price_fallback(
                    network=network,
                    netuid=netuid,
                    epoch=epoch,
                    static_usd_per_0_1_percent_epoch=(
                        config.reimbursement_usd_per_0_1_percent_epoch
                    ),
                    miner_alpha_per_epoch=config.reimbursement_miner_alpha_per_epoch,
                    reason="%s: %s" % (type(exc).__name__, str(exc)),
                )
        else:
            chain_state = self._chain_source.read_finalized_metagraph(
                netuid=netuid,
                context=context,
            )
            valuation = static_alpha_price_fallback(
                network=network,
                netuid=netuid,
                epoch=epoch,
                static_usd_per_0_1_percent_epoch=(
                    config.reimbursement_usd_per_0_1_percent_epoch
                ),
                miner_alpha_per_epoch=config.reimbursement_miner_alpha_per_epoch,
                reason="dynamic_alpha_price_disabled",
            )
        if int(chain_state.get("workflow_epoch_id", -1)) != epoch:
            raise CoordinatorAllocationSourceV2Error(
                "finalized chain state differs from allocation epoch"
            )
        return inject_alpha_price_valuation(policy, valuation), chain_state

    def _reimbursements(
        self,
        *,
        epoch: int,
        policy: Mapping[str, Any],
        hotkey_uids: Mapping[str, int],
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        try:
            epoch_span = max(1, int(policy.get("reimbursement_epochs") or 20))
        except (TypeError, ValueError):
            epoch_span = 20
        schedules = self._read(
            "allocation_reimbursement_schedules",
            {"epoch_id": epoch, "start_epoch": max(0, epoch - epoch_span)},
            context,
        )
        schedules = [
            row
            for row in schedules
            if str(row.get("schedule_status") or "") in ACTIVE_SCHEDULE_STATUSES
            and _epoch_active(row, epoch)
        ]
        award_ids = sorted(
            {str(row.get("award_id") or "") for row in schedules if row.get("award_id")}
        )
        award_rows = (
            self._read(
                "allocation_reimbursement_awards",
                {"award_ids": award_ids},
                context,
            )
            if award_ids
            else []
        )
        awards = {
            str(row.get("award_id") or ""): row
            for row in award_rows
            if str(row.get("current_award_status") or row.get("award_status") or "")
            in ACTIVE_REIMBURSEMENT_STATUSES
        }
        obligations = []
        skipped = []
        for schedule in schedules:
            award = awards.get(str(schedule.get("award_id") or ""))
            if not award:
                continue
            award_id = str(award.get("award_id") or "")
            self._require_reward_receipt(
                artifact_kind="reimbursement_decision",
                artifact_ref=award_id,
                expected_output_root=sha256_json(
                    reimbursement_reward_row_projection_v2(award, schedule)
                ),
                context=context,
                required_parents=required_parents,
            )
            hotkey = str(award.get("miner_hotkey") or "")
            uid = hotkey_uids.get(hotkey)
            if uid is None:
                skipped.append(
                    {"award_id": award_id, "reason": "miner_hotkey_not_registered"}
                )
                continue
            obligations.append(
                {
                    "uid": uid,
                    "miner_uid": uid,
                    "miner_hotkey": hotkey,
                    "source_id": str(
                        schedule.get("schedule_id") or award_id
                    ),
                    "schedule_id": str(schedule.get("schedule_id") or ""),
                    "award_id": award_id,
                    "run_id": str(award.get("run_id") or ""),
                    "island": str(award.get("island") or "generalist"),
                    "status": "active",
                    "start_epoch": int(schedule.get("start_epoch") or 0),
                    "epoch_count": int(schedule.get("epoch_count") or 0),
                    "target_reimbursement_microusd": int(
                        award.get("target_reimbursement_microusd") or 0
                    ),
                    "total_microusd": int(
                        schedule.get("total_microusd")
                        or award.get("target_reimbursement_microusd")
                        or 0
                    ),
                    "participation_score": float(
                        award.get("participation_score") or 0.0
                    ),
                }
            )
        return obligations, skipped

    def _champions(
        self,
        *,
        epoch: int,
        rows: Sequence[Mapping[str, Any]],
        history: Sequence[Mapping[str, Any]],
        hotkey_uids: Mapping[str, int],
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        paid = _champion_paid_alpha_to_date_from_snapshots(
            list(history),
            obligation_caps=_champion_obligation_caps(rows),
        )
        obligations = []
        skipped = []
        for row in rows:
            status = str(row.get("current_reward_status") or row.get("reward_status") or "")
            if status not in ACTIVE_CHAMPION_STATUSES:
                continue
            reward_id = str(row.get("champion_reward_id") or "")
            replay = _champion_replay_obligation(row, paid_by_reward=paid, epoch=epoch)
            if replay is None:
                continue
            self._require_reward_receipt(
                artifact_kind="champion_reward_decision",
                artifact_ref=reward_id,
                expected_output_root=sha256_json(
                    champion_reward_row_projection_v2(row)
                ),
                context=context,
                required_parents=required_parents,
            )
            hotkey = str(row.get("miner_hotkey") or "")
            uid = hotkey_uids.get(hotkey)
            if uid is None:
                skipped.append(
                    {
                        "champion_reward_id": reward_id,
                        "reason": "miner_hotkey_not_registered",
                    }
                )
                continue
            obligations.append(
                {
                    "uid": uid,
                    "miner_uid": uid,
                    "miner_hotkey": hotkey,
                    "source_id": reward_id,
                    "champion_reward_id": reward_id,
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "score_bundle_id": str(row.get("score_bundle_id") or ""),
                    "run_id": str(row.get("run_id") or ""),
                    "island": str(row.get("island") or "generalist"),
                    "status": "active",
                    "reward_kind": str(row.get("reward_kind") or "champion"),
                    **replay,
                }
            )
        return obligations, skipped

    def _finalized_champion_history(
        self,
        *,
        epoch: int,
        netuid: int,
        champion_rows: Sequence[Mapping[str, Any]],
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> list[Dict[str, Any]]:
        starts = [
            int(row.get("start_epoch") or 0)
            for row in champion_rows
            if int(row.get("start_epoch") or 0) <= epoch
        ]
        if not starts or epoch <= 0:
            return []
        native_rows = self._read(
            "finalized_allocation_authorities",
            {
                "netuid": netuid,
                "start_epoch": min(starts),
                "end_epoch": epoch - 1,
            },
            context,
        )
        legacy_rows = self._read(
            "legacy_finalized_allocation_migrations",
            {
                "netuid": netuid,
                "start_epoch": min(starts),
                "end_epoch": epoch - 1,
            },
            context,
        )
        graph_by_root = _receipt_graphs_by_declared_root(
            context.external_receipt_graphs,
            context.parent_receipt_hashes,
        )
        native = validate_finalized_allocation_authorities_v2(
            native_rows,
            finalization_graphs=graph_by_root,
        )
        migrated = validate_legacy_settlement_migrations_v2(
            legacy_rows,
            receipt_graphs=graph_by_root,
        )
        finalized = merge_finalized_allocation_histories_v2(native, migrated)
        for row in finalized:
            authority_types = set(row.get("authority_types") or ())
            if "native_v2_finalization" in authority_types:
                receipt_hash = self._require_allocation_receipt(
                    epoch=int(row["epoch"]),
                    allocation=dict(row["allocation_doc"]),
                    allocation_hash=str(row["allocation_hash"]),
                    context=context,
                    required_parents=required_parents,
                )
                if receipt_hash != str(
                    row.get("allocation_authority_receipt_hash") or ""
                ):
                    raise CoordinatorAllocationSourceV2Error(
                        "finalized weight bundle used another allocation receipt"
                    )
            if "legacy_finalized_chain_migration_v2" in authority_types:
                receipt_hash = str(
                    row.get("legacy_settlement_receipt_hash") or ""
                )
                if (
                    not receipt_hash
                    or receipt_hash not in graph_by_root
                    or receipt_hash not in context.parent_receipt_hashes
                ):
                    raise CoordinatorAllocationSourceV2Error(
                        "legacy finalized allocation receipt is not a declared source"
                    )
                required_parents.add(receipt_hash)
        used_finalization_roots = {
            str(row.get("finalization_receipt_hash") or "")
            for row in native_rows
        }
        for root in used_finalization_roots:
            if not root or root not in graph_by_root:
                raise CoordinatorAllocationSourceV2Error(
                    "finalized allocation graph is not a declared source"
                )
            required_parents.add(root)
        return finalized

    def _source_add(
        self,
        *,
        epoch: int,
        rows: Sequence[Mapping[str, Any]],
        history: Sequence[Mapping[str, Any]],
        hotkey_uids: Mapping[str, int],
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        paid = _source_add_paid_alpha_to_date_from_snapshots(list(history))
        obligations = []
        skipped = []
        for row in rows:
            status = str(row.get("current_reward_status") or "")
            if status not in ACTIVE_CHAMPION_STATUSES:
                continue
            reward_ref = str(row.get("reward_ref") or "")
            replay = _champion_replay_obligation(
                {
                    "champion_reward_id": reward_ref,
                    "start_epoch": int(row.get("start_epoch") or 0),
                    "epoch_count": int(
                        row.get("epoch_count") or row.get("reward_epochs") or 0
                    ),
                    "desired_alpha_percent": float(
                        row.get("desired_alpha_percent")
                        or row.get("alpha_percent")
                        or 0.0
                    ),
                },
                paid_by_reward=paid,
                epoch=epoch,
            )
            if replay is None:
                continue
            self._require_reward_receipt(
                artifact_kind="source_add_reward_decision",
                artifact_ref=reward_ref,
                expected_output_root=sha256_json(
                    source_add_reward_row_projection_v2(
                        "source_add_leg%d" % int(row.get("leg") or 0),
                        {
                            **dict(row),
                            "initial_reward_status": "active",
                        },
                    )
                ),
                context=context,
                required_parents=required_parents,
            )
            hotkey = str(row.get("miner_hotkey") or "")
            uid = hotkey_uids.get(hotkey)
            if uid is None:
                skipped.append(
                    {
                        "source_add_reward_id": reward_ref,
                        "reason": "miner_hotkey_not_registered",
                    }
                )
                continue
            obligations.append(
                {
                    "uid": uid,
                    "miner_uid": uid,
                    "miner_hotkey": hotkey,
                    "source_id": reward_ref,
                    "source_add_reward_id": reward_ref,
                    "adapter_id": str(row.get("adapter_id") or ""),
                    "leg": int(row.get("leg") or 0),
                    "reward_kind": str(row.get("reward_kind") or ""),
                    "status": "active",
                    **replay,
                }
            )
        return obligations, skipped

    def _allocation_history(
        self,
        *,
        epoch: int,
        netuid: int,
        champion_rows: Sequence[Mapping[str, Any]],
        source_add_rows: Sequence[Mapping[str, Any]],
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> list[Dict[str, Any]]:
        starts = [
            int(row.get("start_epoch") or 0)
            for row in tuple(champion_rows) + tuple(source_add_rows)
            if int(row.get("start_epoch") or 0) <= epoch
        ]
        if not starts or epoch <= 0:
            return []
        rows = self._read(
            "allocation_history",
            {
                "netuid": netuid,
                "start_epoch": min(starts),
                "end_epoch": epoch - 1,
            },
            context,
        )
        for row in rows:
            allocation = row.get("allocation_doc")
            allocation_hash = str(row.get("allocation_hash") or "")
            if (
                not isinstance(allocation, Mapping)
                or allocation.get("allocation_hash") != allocation_hash
                or sha256_json(
                    {
                        key: value
                        for key, value in allocation.items()
                        if key != "allocation_hash"
                    }
                )
                != allocation_hash
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "historical allocation row is invalid"
                )
            self._require_allocation_receipt(
                epoch=int(row.get("epoch") or -1),
                allocation=dict(allocation),
                allocation_hash=allocation_hash,
                context=context,
                required_parents=required_parents,
            )
        return [dict(row) for row in rows]

    def _require_reward_receipt(
        self,
        *,
        artifact_kind: str,
        artifact_ref: str,
        expected_output_root: str,
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> None:
        link, receipt = self._business_receipt(
            artifact_kind=artifact_kind,
            artifact_ref=artifact_ref,
            artifact_hash=expected_output_root,
            context=context,
        )
        if (
            receipt.get("role") != "gateway_coordinator"
            or receipt.get("purpose") != "research_lab.reward_decision.v2"
            or receipt.get("output_root") != expected_output_root
            or link.get("artifact_hash") != expected_output_root
        ):
            raise CoordinatorAllocationSourceV2Error(
                "%s receipt does not bind its decision" % artifact_kind
            )
        required_parents.add(str(receipt["receipt_hash"]))

    def _require_allocation_receipt(
        self,
        *,
        epoch: int,
        allocation: Mapping[str, Any],
        allocation_hash: str,
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> str:
        link, receipt = self._business_receipt(
            artifact_kind="allocation",
            artifact_ref="epoch:%d" % epoch,
            artifact_hash=allocation_hash,
            context=context,
        )
        if (
            link.get("artifact_hash") != allocation_hash
            or receipt.get("role") != "gateway_coordinator"
            or receipt.get("purpose") != "research_lab.allocation.v2"
            or int(receipt.get("epoch_id", -1)) != epoch
            or receipt.get("output_root")
            != sha256_json({"allocation": dict(allocation)})
        ):
            raise CoordinatorAllocationSourceV2Error(
                "historical allocation receipt does not bind its row"
            )
        required_parents.add(str(receipt["receipt_hash"]))
        return str(receipt["receipt_hash"])

    def _business_receipt(
        self,
        *,
        artifact_kind: str,
        artifact_ref: str,
        artifact_hash: str,
        context: ExecutionContextV2,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        links = self._read(
            "attested_business_artifact_by_ref",
            {
                "artifact_kind": artifact_kind,
                "artifact_ref": artifact_ref,
                "artifact_hash": artifact_hash,
            },
            context,
        )
        if len(links) != 1:
            raise CoordinatorAllocationSourceV2Error(
                "%s V2 business receipt link is missing or ambiguous" % artifact_kind
            )
        link = links[0]
        receipt_hash = str(link.get("receipt_hash") or "")
        rows = self._read(
            "attested_receipt_by_hash",
            {"receipt_hash": receipt_hash},
            context,
        )
        if len(rows) != 1 or not isinstance(rows[0].get("receipt_doc"), Mapping):
            raise CoordinatorAllocationSourceV2Error(
                "%s V2 receipt is not persisted" % artifact_kind
            )
        receipt = dict(rows[0]["receipt_doc"])
        validate_signed_execution_receipt(receipt)
        if (
            link.get("artifact_hash") != artifact_hash
            or receipt.get("receipt_hash") != receipt_hash
            or not _same(
                {
                    key: rows[0].get(key)
                    for key in (
                        "receipt_hash",
                        "role",
                        "purpose",
                        "epoch_id",
                        "output_root",
                        "boot_identity_hash",
                    )
                },
                {
                    key: receipt.get(key)
                    for key in (
                        "receipt_hash",
                        "role",
                        "purpose",
                        "epoch_id",
                        "output_root",
                        "boot_identity_hash",
                    )
                },
            )
            or receipt_hash not in context.parent_receipt_hashes
        ):
            raise CoordinatorAllocationSourceV2Error(
                "%s V2 receipt is not a declared source" % artifact_kind
            )
        return dict(link), receipt

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

    @staticmethod
    def _non_negative_int(value: Any, field: str) -> int:
        if isinstance(value, bool):
            raise CoordinatorAllocationSourceV2Error("%s must be an integer" % field)
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise CoordinatorAllocationSourceV2Error(
                "%s must be an integer" % field
            ) from exc
        if result < 0:
            raise CoordinatorAllocationSourceV2Error(
                "%s must be non-negative" % field
            )
        return result
