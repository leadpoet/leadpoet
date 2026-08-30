"""Independent source reconstruction for authoritative Research Lab allocation."""

from __future__ import annotations

from decimal import Decimal
import re
from typing import Any, Callable, Dict, Mapping, Sequence, Set, Tuple

from gateway.research_lab.allocations import (
    ACTIVE_CHAMPION_STATUSES,
    ACTIVE_REIMBURSEMENT_STATUSES,
    ACTIVE_SCHEDULE_STATUSES,
    _champion_lifetime_credit_ledger_from_snapshots,
    _champion_replay_obligation,
    _epoch_active,
    _historical_compute_fallback_from_snapshot,
    _source_add_paid_alpha_to_date_from_snapshots,
    champion_reward_requires_allocation_history_v2,
)
from gateway.research_lab.champion_settlement_v2 import (
    CHAIN_REALIZED_AUTHORITY_TYPE_V1,
    merge_settled_allocation_histories_v2,
    merge_finalized_allocation_histories_v2,
    validate_chain_realized_epoch_settlements_v1,
    validate_chain_realized_obligation_credits_v1,
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
    RewardExecutorV2Error,
    champion_reward_row_projection_v2,
    reimbursement_reward_row_projection_v2,
    source_add_reward_row_projection_v2,
)
from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS,
    COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    canonical_json,
    merkle_root,
    sha256_json,
    validate_receipt_graph,
    validate_receipt_graphs,
    validate_signed_execution_receipt,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
    build_reward_settlement_checkpoint_v2,
    frontier_artifact_hashes_v2,
    frontier_paid_maps_v2,
    reward_checkpoint_index_v2,
    validate_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
    frontier_bootstrap_artifact_hashes_v2,
    validate_allocation_settlement_frontier_bootstrap_v2,
)
from leadpoet_verifier.economics import allocate_research_lab_epoch


class CoordinatorAllocationSourceV2Error(RuntimeError):
    """Authenticated allocation sources are incomplete or inconsistent."""


SETTLEMENT_FRONTIER_RETIREMENT_SCHEMA_VERSION = (
    "leadpoet.allocation_settlement_frontier_retirement.v1"
)
_TERMINAL_SETTLEMENT_REWARD_STATUSES = {
    "champion": frozenset({"paid", "voided", "tombstoned"}),
    "source_add": frozenset({"stopped_forward"}),
}


def _same(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _receipt_subgraph(
    graph: Mapping[str, Any],
    *,
    root_receipt_hash: str,
) -> dict[str, Any]:
    validate_receipt_graph(graph)
    subgraph = _receipt_subgraph_from_validated(
        graph,
        root_receipt_hash=root_receipt_hash,
    )
    validate_receipt_graph(subgraph)
    return subgraph


def _receipt_subgraph_from_validated(
    graph: Mapping[str, Any],
    *,
    root_receipt_hash: str,
) -> dict[str, Any]:
    if (
        graph.get("schema_version")
        in CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS
    ):
        if root_receipt_hash != str(graph.get("root_receipt_hash") or ""):
            raise CoordinatorAllocationSourceV2Error(
                "checkpointed allocation parent differs from certified root"
            )
        # A checkpoint certificate binds omitted parent edges to this exact
        # root. Preserve the complete bounded graph and proof; deriving a
        # different root would either strip that authority or misstate it.
        return dict(graph)
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
        receipt
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
            identity
            for identity in graph["boot_identities"]
            if str(identity["boot_identity_hash"]) in selected_boot_hashes
        ],
        "receipts": selected_receipts,
        "transport_attempts": [
            attempt
            for attempt in graph["transport_attempts"]
            if (str(attempt["job_id"]), str(attempt["purpose"]))
            in selected_scopes
        ],
        "host_operations": [
            record
            for record in graph["host_operations"]
            if (
                str(record["request"]["job_id"]),
                str(record["request"]["purpose"]),
            )
            in selected_scopes
        ],
    }
    return subgraph


def _compact_checkpoint_graph_from_proof(
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose one already-verified compact parent through the graph contract."""

    try:
        claim = proof["certificate"]["claim"]
        graph = {
            "schema_version": COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
            "root_receipt_hash": claim["output_root_receipt_hash"],
            "boot_identities": [
                dict(item) for item in proof["disclosed_boot_identities"]
            ],
            "receipts": [dict(item) for item in proof["disclosed_receipts"]],
            "transport_attempts": [],
            "host_operations": [],
            "ancestry_lineage_id": claim["lineage_id"],
            "ancestry_proof": dict(proof),
        }
        validate_receipt_graph(graph)
    except (KeyError, TypeError, ValueError) as exc:
        raise CoordinatorAllocationSourceV2Error(
            "compact allocation parent authority is invalid"
        ) from exc
    return graph


def _receipt_authority_graphs_from_context(
    context: ExecutionContextV2,
) -> list[Mapping[str, Any]]:
    """Combine full graphs with canonical views of verified compact proofs."""

    graphs = list(context.external_receipt_graphs)
    graph_roots = {
        str(graph.get("root_receipt_hash") or "")
        for graph in graphs
        if isinstance(graph, Mapping)
    }
    proof_roots: set[str] = set()
    for proof in context.external_ancestry_proofs:
        graph = _compact_checkpoint_graph_from_proof(proof)
        root = str(graph["root_receipt_hash"])
        if root in graph_roots or root in proof_roots:
            raise CoordinatorAllocationSourceV2Error(
                "allocation parent is supplied as both graph and proof"
            )
        proof_roots.add(root)
        graphs.append(graph)
    return graphs


def _receipt_graphs_by_declared_root(
    graphs: Sequence[Mapping[str, Any]],
    declared_roots: Sequence[str],
) -> dict[str, dict[str, Any]]:
    validate_receipt_graphs(graphs)
    roots = tuple(dict.fromkeys(declared_roots))
    requested_roots = set(roots)
    matches_by_root: dict[str, list[Mapping[str, Any]]] = {
        root: [] for root in roots
    }
    for graph in graphs:
        graph_roots: set[str] = set()
        for receipt in graph.get("receipts") or ():
            if not isinstance(receipt, Mapping):
                continue
            receipt_hash = str(receipt.get("receipt_hash") or "")
            if receipt_hash in requested_roots:
                graph_roots.add(receipt_hash)
        for root in graph_roots:
            matches_by_root[root].append(graph)

    by_root: dict[str, dict[str, Any]] = {}
    for root in roots:
        matches = matches_by_root[root]
        if not matches:
            raise CoordinatorAllocationSourceV2Error(
                "declared allocation parent is absent from receipt graphs"
            )
        derived = _receipt_subgraph_from_validated(
            matches[0],
            root_receipt_hash=str(root),
        )
        for graph in matches[1:]:
            candidate = _receipt_subgraph_from_validated(
                graph,
                root_receipt_hash=str(root),
            )
            if not _same(derived, candidate):
                raise CoordinatorAllocationSourceV2Error(
                    "declared allocation parent graphs conflict"
                )
        by_root[str(root)] = derived
    validate_receipt_graphs(list(by_root.values()))
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
            "allocation_champion_rewards",
            {
                "epoch_id": epoch,
                "include_paid": not bool(
                    policy.get("enable_champ_cap", True)
                ),
            },
            context,
        )
        try:
            champion_source_rows = [
                row
                for row in champion_source_rows
                if champion_reward_requires_allocation_history_v2(
                    row,
                    epoch=epoch,
                    enable_champ_cap=bool(
                        policy.get("enable_champ_cap", True)
                    ),
                )
            ]
        except ValueError as exc:
            raise CoordinatorAllocationSourceV2Error(str(exc)) from exc
        source_add_rows = self._read(
            "allocation_source_add_rewards", {"epoch_id": epoch}, context
        )
        prior_frontier_context = self._load_prior_settlement_frontier(
            epoch=epoch,
            netuid=netuid,
            context=context,
            required_parents=required_parent_hashes,
        )
        prior_frontier = (
            prior_frontier_context["frontier"]
            if prior_frontier_context is not None
            else None
        )
        finalized_reward_history = self._finalized_champion_history(
            epoch=epoch,
            netuid=netuid,
            champion_rows=tuple(champion_source_rows) + tuple(source_add_rows),
            history_start=(
                int(prior_frontier["settled_through_epoch"]) + 1
                if prior_frontier is not None
                else None
            ),
            context=context,
            required_parents=required_parent_hashes,
        )
        settlement_frontier_retirements = (
            self._resolve_settlement_frontier_retirements(
                predecessor=prior_frontier,
                champion_rows=champion_source_rows,
                source_add_rows=source_add_rows,
                context=context,
            )
        )
        settlement_frontier = self._build_settlement_frontier(
            epoch=epoch,
            netuid=netuid,
            champion_rows=champion_source_rows,
            source_add_rows=source_add_rows,
            history=finalized_reward_history,
            predecessor=prior_frontier,
            terminal_retirements=settlement_frontier_retirements,
        )
        paid_maps = frontier_paid_maps_v2(settlement_frontier)
        champion_rows, champion_skipped = self._champions(
            epoch=epoch,
            rows=champion_source_rows,
            paid_by_reward=paid_maps["champion"],
            hotkey_uids=hotkey_uids,
            enable_champ_cap=bool(policy.get("enable_champ_cap", True)),
            context=context,
            required_parents=required_parent_hashes,
        )
        source_add_obligations, source_add_skipped = self._source_add(
            epoch=epoch,
            rows=source_add_rows,
            paid_by_reward=paid_maps["source_add"],
            hotkey_uids=hotkey_uids,
            context=context,
            required_parents=required_parent_hashes,
        )
        fallback_reimbursement_rows: list[Dict[str, Any]] = []
        fallback_reimbursement_skipped: list[Dict[str, Any]] = []
        fallback_source: Dict[str, Any] = {}
        if policy.get("enable_conservative", True) is False:
            native_compute = self._read(
                "latest_native_compute_allocation_authority",
                {"epoch_id": epoch, "netuid": netuid},
                context,
            )
            legacy_compute = self._read(
                "latest_legacy_compute_allocation_authority",
                {"epoch_id": epoch, "netuid": netuid},
                context,
            )
            if len(native_compute) > 1 or len(legacy_compute) > 1:
                raise CoordinatorAllocationSourceV2Error(
                    "historical compute fallback authority is ambiguous"
                )
            identities = []
            if native_compute:
                identities.append(
                    (
                        self._non_negative_int(
                            native_compute[0].get("epoch_id"),
                            "native compute authority epoch",
                        ),
                        "native",
                        native_compute[0],
                    )
                )
            if legacy_compute:
                identities.append(
                    (
                        self._non_negative_int(
                            legacy_compute[0].get("epoch_id"),
                            "legacy compute authority epoch",
                        ),
                        "legacy",
                        legacy_compute[0],
                    )
                )
            if identities:
                source_epoch, source_kind, source_identity = max(
                    identities,
                    key=lambda item: (item[0], item[1] == "native"),
                )
                fallback_snapshot = self._load_historical_compute_authority(
                    source_epoch=source_epoch,
                    source_kind=source_kind,
                    source_identity=source_identity,
                    netuid=netuid,
                    context=context,
                    required_parents=required_parent_hashes,
                )
                (
                    fallback_reimbursement_rows,
                    fallback_reimbursement_skipped,
                    fallback_source,
                ) = _historical_compute_fallback_from_snapshot(
                    fallback_snapshot,
                    hotkey_uids=hotkey_uids,
                    reward_epochs=max(
                        1,
                        int(
                            policy.get("reimbursement_epochs")
                            or policy.get("reward_epochs")
                            or 20
                        ),
                    ),
                    expected_netuid=netuid,
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
        if fallback_reimbursement_rows:
            allocation_inputs["fallback_reimbursement_obligations"] = (
                fallback_reimbursement_rows
            )
        allocation = allocate_research_lab_epoch(
            epoch,
            policy,
            reimbursement_rows,
            champion_rows,
            active_source_add_obligations=source_add_obligations,
            fallback_reimbursement_obligations=fallback_reimbursement_rows,
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
            "settlement_frontier": settlement_frontier,
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
        if settlement_frontier_retirements:
            source_state["settlement_frontier_retirements"] = (
                settlement_frontier_retirements
            )
        if fallback_reimbursement_rows or fallback_reimbursement_skipped:
            source_state.update(
                {
                    "fallback_reimbursement_obligation_count": len(
                        fallback_reimbursement_rows
                    ),
                    "fallback_reimbursement_obligations": (
                        fallback_reimbursement_rows
                    ),
                    "historical_compute_fallback_source": fallback_source,
                }
            )
            source_state["skipped"]["fallback_reimbursements"] = (
                fallback_reimbursement_skipped
            )
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

    def _load_prior_settlement_frontier(
        self,
        *,
        epoch: int,
        netuid: int,
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> Any:
        activation_rows = self._read(
            "allocation_settlement_frontier_activation",
            {"netuid": netuid},
            context,
        )
        frontier_rows = self._read(
            "allocation_settlement_frontiers",
            {"netuid": netuid, "before_epoch": epoch},
            context,
        )
        if not activation_rows:
            if frontier_rows:
                raise CoordinatorAllocationSourceV2Error(
                    "allocation settlement frontier exists without activation"
                )
            return None
        if len(activation_rows) != 1:
            raise CoordinatorAllocationSourceV2Error(
                "allocation settlement frontier is unavailable or ambiguous"
            )
        activation = activation_rows[0]
        try:
            first_epoch = int(activation["first_allocation_epoch"])
        except (KeyError, TypeError, ValueError, RewardExecutorV2Error) as exc:
            raise CoordinatorAllocationSourceV2Error(
                "allocation settlement frontier activation is invalid"
            ) from exc
        if (
            activation.get("schema_version")
            != "leadpoet.research_lab_allocation_settlement_frontier_activation.v2"
            or int(activation.get("netuid", -1)) != netuid
            or first_epoch < 1
            or first_epoch > epoch
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(activation.get("first_frontier_hash") or ""),
            )
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(activation.get("source_receipt_hash") or ""),
            )
        ):
            raise CoordinatorAllocationSourceV2Error(
                "allocation settlement frontier activation is invalid"
            )
        # The activation and first frontier are committed atomically after the
        # first allocation execution. Replaying that same execution has no
        # prior frontier by definition, so rebuild it from its original parent
        # graphs. Every later epoch must have exactly one prior frontier.
        if not frontier_rows:
            if first_epoch == epoch:
                return None
            raise CoordinatorAllocationSourceV2Error(
                "allocation settlement frontier is unavailable or ambiguous"
            )
        if len(frontier_rows) != 1:
            raise CoordinatorAllocationSourceV2Error(
                "allocation settlement frontier is unavailable or ambiguous"
            )
        row = frontier_rows[0]
        try:
            frontier = validate_allocation_settlement_frontier_v2(
                row.get("frontier_doc")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoordinatorAllocationSourceV2Error(
                "allocation settlement frontier is invalid"
            ) from exc
        source_receipt_hash = str(row.get("source_receipt_hash") or "")
        source_state_hash = str(row.get("source_state_hash") or "")
        if (
            first_epoch > int(frontier["allocation_epoch"])
            or int(row.get("netuid", -1)) != netuid
            or int(row.get("allocation_epoch", -1))
            != int(frontier["allocation_epoch"])
            or int(row.get("settled_through_epoch", -2))
            != int(frontier["settled_through_epoch"])
            or row.get("schema_version") != frontier["schema_version"]
            or row.get("frontier_hash") != frontier["frontier_hash"]
            or row.get("predecessor_frontier_hash")
            != frontier.get("predecessor_frontier_hash")
            or int(frontier["netuid"]) != netuid
            or int(frontier["allocation_epoch"]) >= epoch
        ):
            raise CoordinatorAllocationSourceV2Error(
                "allocation settlement frontier scope differs"
            )
        if frontier["mode"] == "legacy_full_history_bootstrap":
            if (
                int(frontier["allocation_epoch"]) != first_epoch
                or frontier["frontier_hash"]
                != activation.get("first_frontier_hash")
                or source_receipt_hash
                != activation.get("source_receipt_hash")
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "allocation settlement frontier activation differs"
                )
        elif int(frontier["allocation_epoch"]) <= first_epoch:
            raise CoordinatorAllocationSourceV2Error(
                "allocation settlement frontier successor is invalid"
            )

        graphs = _receipt_graphs_by_declared_root(
            _receipt_authority_graphs_from_context(context),
            context.parent_receipt_hashes,
        )

        def validate_frontier_authority(
            *,
            authority_row: Mapping[str, Any],
            authority_frontier: Mapping[str, Any],
        ) -> str:
            authority_receipt_hash = str(
                authority_row.get("source_receipt_hash") or ""
            )
            authority_state_hash = str(
                authority_row.get("source_state_hash") or ""
            )
            execution_rows = self._read(
                "attested_execution_result_by_receipt",
                {"receipt_hash": authority_receipt_hash},
                context,
            )
            receipt_rows = self._read(
                "attested_receipt_by_hash",
                {"receipt_hash": authority_receipt_hash},
                context,
            )
            if len(execution_rows) != 1 or len(receipt_rows) != 1:
                raise CoordinatorAllocationSourceV2Error(
                    "allocation frontier execution authority is unavailable"
                )
            execution = execution_rows[0]
            result = execution.get("result_doc")
            receipt = receipt_rows[0].get("receipt_doc")
            artifact_hashes = execution.get("artifact_hashes")
            if (
                not isinstance(result, Mapping)
                or not isinstance(receipt, Mapping)
                or not isinstance(artifact_hashes, list)
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "allocation frontier execution authority is incomplete"
                )
            validate_signed_execution_receipt(receipt)
            normalized_artifacts = sorted(
                {str(item or "").lower() for item in artifact_hashes}
            )
            operation = str(execution.get("operation") or "")
            purpose = str(execution.get("purpose") or "")
            expected_epoch = int(authority_frontier["allocation_epoch"])
            expected_output_root = ""
            required_artifacts: Set[str]
            if operation == "research_lab_allocation":
                if purpose != "research_lab.allocation.v2":
                    raise CoordinatorAllocationSourceV2Error(
                        "allocation frontier execution purpose differs"
                    )
                source_state = result.get("source_state")
                allocation = result.get("allocation")
                required_artifacts = set(
                    frontier_artifact_hashes_v2(authority_frontier)
                ) | {authority_state_hash}
                if (
                    not isinstance(source_state, Mapping)
                    or not isinstance(allocation, Mapping)
                    or source_state.get("settlement_frontier")
                    != authority_frontier
                    or sha256_json(dict(source_state))
                    != authority_state_hash
                    or result.get("source_state_hash")
                    != authority_state_hash
                ):
                    raise CoordinatorAllocationSourceV2Error(
                        "allocation frontier source state differs"
                    )
                expected_output_root = sha256_json(
                    {"allocation": dict(allocation)}
                )
            elif (
                operation
                == ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION
                and purpose
                == ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE
            ):
                try:
                    bootstrap = (
                        validate_allocation_settlement_frontier_bootstrap_v2(
                            result
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise CoordinatorAllocationSourceV2Error(
                        "allocation frontier bootstrap authority is invalid"
                    ) from exc
                if (
                    bootstrap["frontier"] != authority_frontier
                    or bootstrap["source_state_hash"]
                    != authority_state_hash
                    or int(bootstrap["netuid"])
                    != int(authority_frontier["netuid"])
                    or int(bootstrap["allocation_epoch"])
                    != int(authority_frontier["allocation_epoch"])
                ):
                    raise CoordinatorAllocationSourceV2Error(
                        "allocation frontier bootstrap authority differs"
                    )
                expected_epoch = int(bootstrap["bootstrap_epoch"])
                expected_output_root = sha256_json(dict(bootstrap))
                required_artifacts = set(
                    frontier_bootstrap_artifact_hashes_v2(bootstrap)
                )
                allocation_receipt_hash = str(
                    bootstrap["allocation_source_receipt_hash"]
                )
                parent_hashes = receipt.get("parent_receipt_hashes")
                if (
                    not isinstance(parent_hashes, list)
                    or allocation_receipt_hash not in parent_hashes
                ):
                    raise CoordinatorAllocationSourceV2Error(
                        "allocation frontier bootstrap source is undeclared"
                    )
                allocation_execution_rows = self._read(
                    "attested_execution_result_by_receipt",
                    {"receipt_hash": allocation_receipt_hash},
                    context,
                )
                allocation_receipt_rows = self._read(
                    "attested_receipt_by_hash",
                    {"receipt_hash": allocation_receipt_hash},
                    context,
                )
                if (
                    len(allocation_execution_rows) != 1
                    or len(allocation_receipt_rows) != 1
                ):
                    raise CoordinatorAllocationSourceV2Error(
                        "allocation frontier bootstrap source is unavailable"
                    )
                allocation_execution = allocation_execution_rows[0]
                allocation_result = allocation_execution.get("result_doc")
                allocation_receipt = allocation_receipt_rows[0].get(
                    "receipt_doc"
                )
                allocation_artifacts = allocation_execution.get(
                    "artifact_hashes"
                )
                if (
                    not isinstance(allocation_result, Mapping)
                    or not isinstance(allocation_receipt, Mapping)
                    or not isinstance(allocation_artifacts, list)
                ):
                    raise CoordinatorAllocationSourceV2Error(
                        "allocation frontier bootstrap source is incomplete"
                    )
                validate_signed_execution_receipt(allocation_receipt)
                allocation_state = allocation_result.get("source_state")
                allocation = allocation_result.get("allocation")
                normalized_allocation_artifacts = sorted(
                    {str(item or "").lower() for item in allocation_artifacts}
                )
                allocation_output_root = (
                    sha256_json({"allocation": dict(allocation)})
                    if isinstance(allocation, Mapping)
                    else ""
                )
                if (
                    len(normalized_allocation_artifacts)
                    != len(allocation_artifacts)
                    or any(
                        not re.fullmatch(r"sha256:[0-9a-f]{64}", item)
                        for item in normalized_allocation_artifacts
                    )
                    or not isinstance(allocation_state, Mapping)
                    or not isinstance(allocation, Mapping)
                    or allocation_state.get("settlement_frontier") is not None
                    or sha256_json(dict(allocation_state))
                    != authority_state_hash
                    or allocation_result.get("source_state_hash")
                    != authority_state_hash
                    or int(allocation_state.get("netuid", -1))
                    != int(authority_frontier["netuid"])
                    or int(allocation_state.get("epoch", -1))
                    != int(authority_frontier["allocation_epoch"])
                    or authority_state_hash
                    not in normalized_allocation_artifacts
                    or allocation_execution.get("schema_version")
                    != "leadpoet.attested_execution_result.v2"
                    or allocation_execution.get("receipt_hash")
                    != allocation_receipt_hash
                    or allocation_execution.get("role")
                    != "gateway_coordinator"
                    or allocation_execution.get("operation")
                    != "research_lab_allocation"
                    or allocation_execution.get("purpose")
                    != "research_lab.allocation.v2"
                    or int(allocation_execution.get("epoch_id", -1))
                    != int(authority_frontier["allocation_epoch"])
                    or allocation_execution.get("result_hash")
                    != sha256_json(dict(allocation_result))
                    or allocation_execution.get("artifact_root")
                    != merkle_root(
                        normalized_allocation_artifacts,
                        domain="leadpoet-artifact-v2",
                    )
                    or allocation_execution.get("output_root")
                    != allocation_output_root
                    or allocation_receipt.get("receipt_hash")
                    != allocation_receipt_hash
                    or allocation_receipt.get("role")
                    != "gateway_coordinator"
                    or allocation_receipt.get("purpose")
                    != "research_lab.allocation.v2"
                    or allocation_receipt.get("status") != "succeeded"
                    or allocation_receipt.get("job_id")
                    != allocation_execution.get("job_id")
                    or allocation_receipt.get("sequence")
                    != allocation_execution.get("sequence")
                    or not re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(allocation_execution.get("release_hash") or ""),
                    )
                    or int(allocation_receipt.get("epoch_id", -1))
                    != int(authority_frontier["allocation_epoch"])
                    or allocation_receipt.get("input_root")
                    != allocation_execution.get("input_root")
                    or allocation_receipt.get("output_root")
                    != allocation_output_root
                    or allocation_receipt.get("artifact_root")
                    != allocation_execution.get("artifact_root")
                    or allocation_receipt_rows[0].get("receipt_hash")
                    != allocation_receipt_hash
                ):
                    raise CoordinatorAllocationSourceV2Error(
                        "allocation frontier bootstrap source differs"
                    )
            else:
                raise CoordinatorAllocationSourceV2Error(
                    "allocation frontier execution operation differs"
                )
            if (
                len(normalized_artifacts) != len(artifact_hashes)
                or any(
                    not re.fullmatch(r"sha256:[0-9a-f]{64}", item)
                    for item in normalized_artifacts
                )
                or not required_artifacts.issubset(set(normalized_artifacts))
                or execution.get("schema_version")
                != "leadpoet.attested_execution_result.v2"
                or execution.get("receipt_hash") != authority_receipt_hash
                or execution.get("role") != "gateway_coordinator"
                or execution.get("operation") != operation
                or execution.get("purpose") != purpose
                or int(execution.get("epoch_id", -1)) != expected_epoch
                or execution.get("result_hash") != sha256_json(dict(result))
                or execution.get("artifact_root")
                != merkle_root(
                    normalized_artifacts,
                    domain="leadpoet-artifact-v2",
                )
                or receipt.get("receipt_hash") != authority_receipt_hash
                or receipt.get("role") != "gateway_coordinator"
                or receipt.get("purpose") != purpose
                or receipt.get("status") != "succeeded"
                or receipt.get("job_id") != execution.get("job_id")
                or receipt.get("sequence") != execution.get("sequence")
                or not re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(execution.get("release_hash") or ""),
                )
                or int(receipt.get("epoch_id", -1)) != expected_epoch
                or receipt.get("output_root") != expected_output_root
                or receipt.get("artifact_root")
                != execution.get("artifact_root")
                or receipt.get("input_root") != execution.get("input_root")
                or receipt.get("output_root") != execution.get("output_root")
                or receipt_rows[0].get("receipt_hash")
                != authority_receipt_hash
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "allocation frontier execution authority differs"
                )
            if (
                authority_receipt_hash not in graphs
                or authority_receipt_hash
                not in context.parent_receipt_hashes
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "allocation frontier receipt is not a declared source"
                )
            required_parents.add(authority_receipt_hash)
            return authority_receipt_hash

        first_rows = self._read(
            "allocation_settlement_frontier_by_epoch",
            {"netuid": netuid, "allocation_epoch": first_epoch},
            context,
        )
        if len(first_rows) != 1:
            raise CoordinatorAllocationSourceV2Error(
                "allocation frontier activation authority is unavailable"
            )
        first_row = first_rows[0]
        try:
            first_frontier = validate_allocation_settlement_frontier_v2(
                first_row.get("frontier_doc")
            )
        except (TypeError, ValueError) as exc:
            raise CoordinatorAllocationSourceV2Error(
                "allocation frontier activation authority is invalid"
            ) from exc
        if (
            first_frontier.get("mode") != "legacy_full_history_bootstrap"
            or int(first_frontier.get("netuid", -1)) != netuid
            or int(first_frontier.get("allocation_epoch", -1)) != first_epoch
            or first_frontier.get("predecessor_frontier_hash") is not None
            or first_frontier.get("frontier_hash")
            != activation.get("first_frontier_hash")
            or int(first_row.get("netuid", -1)) != netuid
            or int(first_row.get("allocation_epoch", -1)) != first_epoch
            or int(first_row.get("settled_through_epoch", -2))
            != int(first_frontier["settled_through_epoch"])
            or first_row.get("schema_version")
            != first_frontier.get("schema_version")
            or first_row.get("frontier_hash")
            != first_frontier.get("frontier_hash")
            or first_row.get("predecessor_frontier_hash") is not None
            or first_row.get("source_receipt_hash")
            != activation.get("source_receipt_hash")
        ):
            raise CoordinatorAllocationSourceV2Error(
                "allocation frontier activation authority differs"
            )
        first_receipt_hash = validate_frontier_authority(
            authority_row=first_row,
            authority_frontier=first_frontier,
        )
        if source_receipt_hash != first_receipt_hash:
            validate_frontier_authority(
                authority_row=row,
                authority_frontier=frontier,
            )
        return {"frontier": frontier, "receipt_hash": source_receipt_hash}

    @staticmethod
    def _settlement_retirement_evidence(
        *,
        reward_kind: str,
        checkpoint: Mapping[str, Any],
        terminal_row: Mapping[str, Any],
    ) -> Dict[str, Any]:
        source_id = str(checkpoint.get("source_id") or "")
        terminal_status = str(terminal_row.get("current_reward_status") or "")
        accepted_statuses = _TERMINAL_SETTLEMENT_REWARD_STATUSES.get(
            reward_kind
        )
        if accepted_statuses is None or terminal_status not in accepted_statuses:
            raise CoordinatorAllocationSourceV2Error(
                "settlement frontier reward is not terminal"
            )
        try:
            if reward_kind == "champion":
                observed_source_id = str(
                    terminal_row.get("champion_reward_id") or ""
                )
                projection = champion_reward_row_projection_v2(terminal_row)
            else:
                observed_source_id = str(terminal_row.get("reward_ref") or "")
                projection = source_add_reward_row_projection_v2(
                    "source_add_leg%d" % int(terminal_row.get("leg") or 0),
                    {
                        **dict(terminal_row),
                        "initial_reward_status": "active",
                    },
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoordinatorAllocationSourceV2Error(
                "settlement frontier terminal reward is invalid"
            ) from exc
        obligation_hash = sha256_json(projection)
        if (
            observed_source_id != source_id
            or obligation_hash != checkpoint.get("obligation_hash")
        ):
            raise CoordinatorAllocationSourceV2Error(
                "settlement frontier terminal reward identity changed"
            )
        evidence = {
            "schema_version": SETTLEMENT_FRONTIER_RETIREMENT_SCHEMA_VERSION,
            "reward_kind": reward_kind,
            "source_id": source_id,
            "terminal_status": terminal_status,
            "obligation_hash": obligation_hash,
            "predecessor_checkpoint_hash": str(
                checkpoint.get("checkpoint_hash") or ""
            ),
        }
        evidence["retirement_hash"] = sha256_json(evidence)
        return evidence

    def _resolve_settlement_frontier_retirements(
        self,
        *,
        predecessor: Any,
        champion_rows: Sequence[Mapping[str, Any]],
        source_add_rows: Sequence[Mapping[str, Any]],
        context: ExecutionContextV2,
    ) -> list[Dict[str, Any]]:
        if predecessor is None:
            return []
        previous = validate_allocation_settlement_frontier_v2(predecessor)
        active_keys = {
            ("champion", str(row.get("champion_reward_id") or ""))
            for row in champion_rows
        }
        active_keys.update(
            ("source_add", str(row.get("reward_ref") or ""))
            for row in source_add_rows
        )
        retirements: list[Dict[str, Any]] = []
        for checkpoint in previous["reward_checkpoints"]:
            reward_kind = str(checkpoint["reward_kind"])
            source_id = str(checkpoint["source_id"])
            if (reward_kind, source_id) in active_keys:
                continue
            if reward_kind == "champion":
                rows = self._read(
                    "champion_reward_by_id",
                    {"champion_reward_id": source_id},
                    context,
                )
            elif reward_kind == "source_add":
                rows = self._read(
                    "source_add_reward_by_ref",
                    {"reward_ref": source_id},
                    context,
                )
            else:
                raise CoordinatorAllocationSourceV2Error(
                    "settlement frontier reward kind is unsupported"
                )
            if len(rows) != 1:
                raise CoordinatorAllocationSourceV2Error(
                    "settlement frontier terminal reward is unavailable or ambiguous"
                )
            retirements.append(
                self._settlement_retirement_evidence(
                    reward_kind=reward_kind,
                    checkpoint=checkpoint,
                    terminal_row=rows[0],
                )
            )
        return sorted(
            retirements,
            key=lambda item: (item["reward_kind"], item["source_id"]),
        )

    def _build_settlement_frontier(
        self,
        *,
        epoch: int,
        netuid: int,
        champion_rows: Sequence[Mapping[str, Any]],
        source_add_rows: Sequence[Mapping[str, Any]],
        history: Sequence[Mapping[str, Any]],
        predecessor: Any,
        terminal_retirements: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        previous = (
            validate_allocation_settlement_frontier_v2(predecessor)
            if predecessor is not None
            else None
        )
        previous_index = (
            reward_checkpoint_index_v2(previous["reward_checkpoints"])
            if previous is not None
            else {}
        )
        champion_delta = _champion_lifetime_credit_ledger_from_snapshots(
            list(history),
            obligation_caps=None,
        )["realized_by_reward"]
        source_add_delta = _source_add_paid_alpha_to_date_from_snapshots(
            list(history)
        )
        checkpoints: list[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        retirement_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for raw_retirement in terminal_retirements:
            if not isinstance(raw_retirement, Mapping):
                raise CoordinatorAllocationSourceV2Error(
                    "settlement frontier retirement evidence is invalid"
                )
            retirement = dict(raw_retirement)
            body = {
                key: retirement.get(key)
                for key in (
                    "schema_version",
                    "reward_kind",
                    "source_id",
                    "terminal_status",
                    "obligation_hash",
                    "predecessor_checkpoint_hash",
                )
            }
            reward_kind = str(body["reward_kind"] or "")
            source_id = str(body["source_id"] or "")
            key = (reward_kind, source_id)
            if (
                set(retirement) != set(body) | {"retirement_hash"}
                or body["schema_version"]
                != SETTLEMENT_FRONTIER_RETIREMENT_SCHEMA_VERSION
                or body["terminal_status"]
                not in _TERMINAL_SETTLEMENT_REWARD_STATUSES.get(
                    reward_kind, frozenset()
                )
                or retirement.get("retirement_hash") != sha256_json(body)
                or key in retirement_index
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "settlement frontier retirement evidence is invalid"
                )
            retirement_index[key] = retirement

        def append_checkpoint(
            *,
            reward_kind: str,
            source_id: str,
            obligation_hash: str,
            start_epoch: int,
            epoch_count: int,
            desired_alpha_percent: Any,
            delta_realized: Any,
        ) -> None:
            key = (reward_kind, source_id)
            if key in seen:
                raise CoordinatorAllocationSourceV2Error(
                    "allocation settlement reward is duplicated"
                )
            seen.add(key)
            if key in retirement_index:
                raise CoordinatorAllocationSourceV2Error(
                    "active reward has terminal settlement evidence"
                )
            prior = previous_index.get(key)
            if (
                prior is None
                and previous is not None
                and start_epoch <= int(previous["settled_through_epoch"])
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "historical reward is absent from prior settlement frontier"
                )
            prior_applied = Decimal(
                str(prior["applied_alpha_percent"] if prior else "0")
            )
            prior_realized = Decimal(
                str(prior["realized_alpha_percent"] if prior else "0")
            )
            try:
                delta = Decimal(str(delta_realized or 0))
                desired = Decimal(str(desired_alpha_percent or 0))
            except Exception as exc:
                raise CoordinatorAllocationSourceV2Error(
                    "allocation settlement reward amount is invalid"
                ) from exc
            if (
                not delta.is_finite()
                or delta < 0
                or not desired.is_finite()
                or desired < 0
                or epoch_count <= 0
                or start_epoch < 0
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "allocation settlement reward amount is invalid"
                )
            total_due = desired * Decimal(epoch_count)
            realized = prior_realized + delta
            applied = min(total_due, prior_applied + delta)
            checkpoint = build_reward_settlement_checkpoint_v2(
                reward_kind=reward_kind,
                source_id=source_id,
                obligation_hash=obligation_hash,
                start_epoch=start_epoch,
                epoch_count=epoch_count,
                desired_alpha_percent=desired,
                applied_alpha_percent=applied,
                realized_alpha_percent=realized,
                excess_alpha_percent=realized - applied,
            )
            if prior is not None and any(
                checkpoint[field] != prior[field]
                for field in (
                    "obligation_hash",
                    "start_epoch",
                    "epoch_count",
                    "desired_alpha_percent",
                    "total_due_alpha_percent",
                )
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "allocation settlement reward identity changed"
                )
            checkpoints.append(checkpoint)

        for row in champion_rows:
            source_id = str(row.get("champion_reward_id") or "")
            append_checkpoint(
                reward_kind="champion",
                source_id=source_id,
                obligation_hash=sha256_json(
                    champion_reward_row_projection_v2(row)
                ),
                start_epoch=int(row.get("start_epoch") or 0),
                epoch_count=int(row.get("epoch_count") or 0),
                desired_alpha_percent=row.get("desired_alpha_percent") or 0,
                delta_realized=champion_delta.get(source_id, 0),
            )
        for row in source_add_rows:
            source_id = str(row.get("reward_ref") or "")
            append_checkpoint(
                reward_kind="source_add",
                source_id=source_id,
                obligation_hash=sha256_json(
                    source_add_reward_row_projection_v2(
                        "source_add_leg%d" % int(row.get("leg") or 0),
                        {
                            **dict(row),
                            "initial_reward_status": "active",
                        },
                    )
                ),
                start_epoch=int(row.get("start_epoch") or 0),
                epoch_count=int(
                    row.get("epoch_count")
                    or row.get("reward_epochs")
                    or 0
                ),
                desired_alpha_percent=(
                    row.get("desired_alpha_percent")
                    or row.get("alpha_percent")
                    or 0
                ),
                delta_realized=source_add_delta.get(source_id, 0),
            )
        for key, prior in previous_index.items():
            if key in seen:
                continue
            reward_kind, source_id = key
            raw_delta = (
                champion_delta.get(source_id, 0)
                if reward_kind == "champion"
                else source_add_delta.get(source_id, 0)
            )
            try:
                delta = Decimal(str(raw_delta or 0))
                prior_applied = Decimal(str(prior["applied_alpha_percent"]))
                total_due = Decimal(str(prior["total_due_alpha_percent"]))
            except Exception as exc:
                raise CoordinatorAllocationSourceV2Error(
                    "retired allocation settlement reward is invalid"
                ) from exc
            if not delta.is_finite() or delta < 0:
                raise CoordinatorAllocationSourceV2Error(
                    "retired allocation settlement reward is invalid"
                )
            retirement = retirement_index.pop(key, None)
            if (
                retirement is None
                and min(total_due, prior_applied + delta) != total_due
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "unsettled reward disappeared from the active frontier"
                )
            if retirement is not None and (
                retirement.get("obligation_hash")
                != prior.get("obligation_hash")
                or retirement.get("predecessor_checkpoint_hash")
                != prior.get("checkpoint_hash")
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "settlement frontier retirement differs from predecessor"
                )
        if retirement_index:
            raise CoordinatorAllocationSourceV2Error(
                "settlement frontier retirement lacks a predecessor reward"
            )
        return build_allocation_settlement_frontier_v2(
            mode=(
                "legacy_full_history_bootstrap"
                if previous is None
                else "bounded_delta_v1"
            ),
            netuid=netuid,
            allocation_epoch=epoch,
            predecessor_frontier_hash=(
                str(previous["frontier_hash"])
                if previous is not None
                else None
            ),
            reward_checkpoints=checkpoints,
        )

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
                    "eligible_compute_microusd": int(
                        award.get("eligible_cost_microusd")
                        or award.get("target_reimbursement_microusd")
                        or 0
                    ),
                    "participation_score": float(
                        award.get("participation_score") or 0.0
                    ),
                }
            )
        return obligations, skipped

    def _load_historical_compute_authority(
        self,
        *,
        source_epoch: int,
        source_kind: str,
        source_identity: Mapping[str, Any],
        netuid: int,
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> Dict[str, Any]:
        if (
            source_kind not in {"native", "legacy"}
            or int(source_identity.get("epoch_id", -1)) != int(source_epoch)
            or int(source_identity.get("netuid", -1)) != int(netuid)
        ):
            raise CoordinatorAllocationSourceV2Error(
                "historical compute fallback identity is invalid"
            )
        native_rows = self._read(
            "finalized_allocation_authorities",
            {
                "netuid": netuid,
                "start_epoch": source_epoch,
                "end_epoch": source_epoch,
            },
            context,
        )
        legacy_rows = self._read(
            "legacy_finalized_allocation_migrations",
            {
                "netuid": netuid,
                "start_epoch": source_epoch,
                "end_epoch": source_epoch,
            },
            context,
        )
        graph_by_root = _receipt_graphs_by_declared_root(
            _receipt_authority_graphs_from_context(context),
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
        authorities = merge_finalized_allocation_histories_v2(
            native,
            migrated,
        )
        matches = [
            authority
            for authority in authorities
            if int(authority.get("epoch", -1)) == source_epoch
            and int(authority.get("netuid", -1)) == netuid
            and isinstance(authority.get("allocation_doc"), Mapping)
            and bool(
                authority["allocation_doc"].get(
                    "reimbursement_allocations"
                )
            )
            and authority["allocation_doc"].get(
                "historical_compute_fallback_source_epoch"
            )
            is None
            and (
                source_kind != "native"
                or "native_v2_finalization"
                in set(authority.get("authority_types") or ())
            )
            and (
                source_kind != "legacy"
                or (
                    "legacy_finalized_chain_migration_v2"
                    in set(authority.get("authority_types") or ())
                    and str(authority.get("allocation_hash") or "")
                    == str(source_identity.get("allocation_hash") or "")
                    and _same(
                        authority.get("allocation_doc"),
                        source_identity.get("allocation_doc"),
                    )
                )
            )
        ]
        if len(matches) != 1:
            raise CoordinatorAllocationSourceV2Error(
                "historical compute fallback lacks finalized allocation authority"
            )
        authority = matches[0]
        allocation = authority["allocation_doc"]
        allocation_hash = str(authority.get("allocation_hash") or "")
        authority_types = set(authority.get("authority_types") or ())
        if "native_v2_finalization" in authority_types:
            receipt_hash = self._require_allocation_receipt(
                epoch=source_epoch,
                allocation=dict(allocation),
                allocation_hash=allocation_hash,
                context=context,
                required_parents=required_parents,
            )
            if receipt_hash != str(
                authority.get("allocation_authority_receipt_hash") or ""
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "historical compute fallback used another allocation receipt"
                )
            for native_row in native_rows:
                root = str(native_row.get("finalization_receipt_hash") or "")
                if not root or root not in graph_by_root:
                    raise CoordinatorAllocationSourceV2Error(
                        "historical compute finalization graph is not declared"
                    )
                required_parents.add(root)
        if "legacy_finalized_chain_migration_v2" in authority_types:
            root = str(
                authority.get("legacy_settlement_receipt_hash") or ""
            )
            if (
                not root
                or root not in graph_by_root
                or root not in context.parent_receipt_hashes
            ):
                raise CoordinatorAllocationSourceV2Error(
                    "historical compute legacy settlement is not declared"
                )
            required_parents.add(root)
        if not authority_types.intersection(
            {
                "native_v2_finalization",
                "legacy_finalized_chain_migration_v2",
            }
        ):
            raise CoordinatorAllocationSourceV2Error(
                "historical compute fallback authority type is unsupported"
            )
        return {
            "epoch": int(source_epoch),
            "netuid": int(netuid),
            "allocation_hash": allocation_hash,
            "allocation_doc": dict(allocation),
        }

    def _champions(
        self,
        *,
        epoch: int,
        rows: Sequence[Mapping[str, Any]],
        paid_by_reward: Mapping[str, float],
        hotkey_uids: Mapping[str, int],
        enable_champ_cap: bool,
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        obligations = []
        skipped = []
        accepted_statuses = (
            ACTIVE_CHAMPION_STATUSES
            if enable_champ_cap
            else ACTIVE_CHAMPION_STATUSES | {"paid"}
        )
        for row in rows:
            status = str(row.get("current_reward_status") or row.get("reward_status") or "")
            if status not in accepted_statuses:
                continue
            reward_id = str(row.get("champion_reward_id") or "")
            self._require_reward_receipt(
                artifact_kind="champion_reward_decision",
                artifact_ref=reward_id,
                expected_output_root=sha256_json(
                    champion_reward_row_projection_v2(row)
                ),
                context=context,
                required_parents=required_parents,
            )
            replay = _champion_replay_obligation(
                row,
                paid_by_reward=paid_by_reward,
                epoch=epoch,
                enable_champ_cap=bool(enable_champ_cap),
            )
            if replay is None:
                continue
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
        history_start: Any = None,
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> list[Dict[str, Any]]:
        starts = [
            int(row.get("start_epoch") or 0)
            for row in champion_rows
            if int(row.get("start_epoch") or 0) <= epoch
        ]
        if epoch <= 0 or (history_start is None and not starts):
            return []
        normalized_history_start = (
            min(starts)
            if history_start is None
            else self._non_negative_int(history_start, "history_start")
        )
        history_end = epoch - 1
        activation_rows = self._read(
            "chain_realized_settlement_activation",
            {"netuid": netuid},
            context,
        )
        if len(activation_rows) != 1:
            raise CoordinatorAllocationSourceV2Error(
                "chain realized settlement activation is unavailable or ambiguous"
            )
        activation = activation_rows[0]
        try:
            activation_epoch = int(activation["first_epoch_id"])
            source_epoch = int(activation["source_bundle_epoch_id"])
            activation_netuid = int(activation["netuid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CoordinatorAllocationSourceV2Error(
                "chain realized settlement activation is invalid"
            ) from exc
        if (
            activation.get("schema_version")
            != "leadpoet.research_lab_chain_realized_settlement_activation.v1"
            or activation_netuid != netuid
            or source_epoch != activation_epoch
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(activation.get("source_bundle_hash") or "")
            )
        ):
            raise CoordinatorAllocationSourceV2Error(
                "chain realized settlement activation is invalid"
            )
        # Chain-realized settlement supersedes submitted-weight intent at the
        # activation epoch. Only request graphs for authorities the enclave
        # will actually consume.
        finalized_end = min(history_end, activation_epoch - 1)
        native_rows = (
            self._read(
                "finalized_allocation_authorities",
                {
                    "netuid": netuid,
                    "start_epoch": normalized_history_start,
                    "end_epoch": finalized_end,
                },
                context,
            )
            if finalized_end >= normalized_history_start
            else []
        )
        legacy_rows = (
            self._read(
                "legacy_finalized_allocation_migrations",
                {
                    "netuid": netuid,
                    "start_epoch": normalized_history_start,
                    "end_epoch": finalized_end,
                },
                context,
            )
            if finalized_end >= normalized_history_start
            else []
        )
        chain_start = max(normalized_history_start, activation_epoch)
        chain_settlement_rows = (
            self._read(
                "chain_realized_epoch_settlements",
                {
                    "netuid": netuid,
                    "start_epoch": chain_start,
                    "end_epoch": history_end,
                },
                context,
            )
            if chain_start <= history_end
            else []
        )
        expected_epochs = set(
            range(chain_start, epoch)
        )
        observed_epochs = {
            int(row["epoch_id"]) for row in chain_settlement_rows
        }
        if observed_epochs != expected_epochs:
            raise CoordinatorAllocationSourceV2Error(
                "chain realized settlement history is incomplete"
            )
        chain_credit_rows = (
            self._read(
                "chain_realized_obligation_credits",
                {
                    "netuid": netuid,
                    "start_epoch": chain_start,
                    "end_epoch": history_end,
                },
                context,
            )
            if chain_start <= history_end
            else []
        )
        graph_by_root = _receipt_graphs_by_declared_root(
            _receipt_authority_graphs_from_context(context),
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
        chain_settlements = validate_chain_realized_epoch_settlements_v1(
            chain_settlement_rows,
            receipt_graphs=graph_by_root,
            _receipt_graphs_prevalidated=True,
        )
        chain_realized = validate_chain_realized_obligation_credits_v1(
            chain_credit_rows,
            settlement_rows=chain_settlements,
            receipt_graphs=graph_by_root,
            _receipt_graphs_prevalidated=True,
        )
        finalized = merge_settled_allocation_histories_v2(
            finalized,
            chain_realized,
        )
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
            if CHAIN_REALIZED_AUTHORITY_TYPE_V1 in authority_types:
                settlement_receipt = str(
                    row.get("chain_realized_settlement_receipt_hash") or ""
                )
                if (
                    not settlement_receipt
                    or settlement_receipt not in graph_by_root
                    or settlement_receipt not in context.parent_receipt_hashes
                ):
                    raise CoordinatorAllocationSourceV2Error(
                        "chain realized settlement receipt is not a declared source"
                    )
                required_parents.add(settlement_receipt)
                for receipt_hash in row.get("chain_realized_credit_receipt_hashes") or ():
                    credit_receipt = str(receipt_hash or "")
                    if (
                        not credit_receipt
                        or credit_receipt not in graph_by_root
                        or credit_receipt not in context.parent_receipt_hashes
                    ):
                        raise CoordinatorAllocationSourceV2Error(
                            "chain realized credit receipt is not a declared source"
                        )
                    required_parents.add(credit_receipt)
        used_finalization_roots = {
            str(row.get("finalization_receipt_hash") or "")
            for row in native_rows
        }
        used_legacy_roots = {
            str(row.get("settlement_receipt_hash") or "")
            for row in legacy_rows
        }
        used_chain_settlement_roots = {
            str(row.get("settlement_receipt_hash") or "")
            for row in chain_settlement_rows
        }
        used_chain_credit_roots = {
            str(row.get("credit_receipt_hash") or "")
            for row in chain_credit_rows
        }
        for root in used_finalization_roots:
            if not root or root not in graph_by_root:
                raise CoordinatorAllocationSourceV2Error(
                    "finalized allocation graph is not a declared source"
                )
            required_parents.add(root)
        for root in (
            used_legacy_roots
            | used_chain_settlement_roots
            | used_chain_credit_roots
        ):
            if not root or root not in graph_by_root:
                raise CoordinatorAllocationSourceV2Error(
                    "settlement authority graph is not a declared source"
                )
            required_parents.add(root)
        return finalized

    def _source_add(
        self,
        *,
        epoch: int,
        rows: Sequence[Mapping[str, Any]],
        paid_by_reward: Mapping[str, float],
        hotkey_uids: Mapping[str, int],
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        obligations = []
        skipped = []
        for row in rows:
            status = str(row.get("current_reward_status") or "")
            if status not in ACTIVE_CHAMPION_STATUSES:
                continue
            reward_ref = str(row.get("reward_ref") or "")
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
                paid_by_reward=paid_by_reward,
                epoch=epoch,
            )
            if replay is None:
                continue
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
                    "created_at": str(row.get("created_at") or ""),
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
