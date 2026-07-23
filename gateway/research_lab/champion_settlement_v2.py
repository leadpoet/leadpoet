"""Finalized-chain settlement authority for champion obligations.

Allocation snapshots describe what the gateway intended to pay.  They are not
payment evidence.  This module accepts an allocation epoch only when the exact
allocation receipt was consumed by a canonical V2 weight bundle and that
bundle has a canonical finalized-chain submission.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import logging
import re
from typing import Any, Mapping, Sequence

from leadpoet_canonical.attested_v2 import (
    sha256_json,
    validate_receipt_graph,
)
from leadpoet_canonical.legacy_settlement_v2 import (
    validate_legacy_nonfinalization_document_v2,
    validate_legacy_settlement_document_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    validate_published_weight_bundle_v2,
    validate_weight_finalization_submission_v2,
)


FINALIZED_ALLOCATION_VIEW_V2 = "research_lab_finalized_allocation_epochs_v2"
LEGACY_SETTLEMENT_TABLE_V2 = (
    "research_lab_legacy_finalized_allocation_migrations_v2"
)
LEGACY_NONFINALIZATION_TABLE_V2 = (
    "research_lab_legacy_allocation_nonfinalizations_v2"
)
logger = logging.getLogger(__name__)


class ChampionSettlementV2Error(RuntimeError):
    """Finalized weight evidence is missing, inconsistent, or tampered."""


def _allocation_authority_receipt_hash_v2(
    *,
    bundle_doc: Mapping[str, Any],
    allocation_input_receipt_hash: str,
    allocation: Mapping[str, Any],
    epoch_id: int,
) -> str:
    graph = bundle_doc.get("receipt_graph")
    if not isinstance(graph, Mapping):
        raise ChampionSettlementV2Error(
            "finalized weight bundle receipt graph is missing"
        )
    receipts = {
        str(receipt.get("receipt_hash") or ""): receipt
        for receipt in graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
    }
    input_receipt = receipts.get(allocation_input_receipt_hash)
    if not isinstance(input_receipt, Mapping):
        raise ChampionSettlementV2Error(
            "finalized weight bundle allocation input receipt is missing"
        )
    parent_hashes = input_receipt.get("parent_receipt_hashes")
    if not isinstance(parent_hashes, list) or len(parent_hashes) != 1:
        raise ChampionSettlementV2Error(
            "finalized weight bundle allocation input ancestry is invalid"
        )
    authority_hash = str(parent_hashes[0] or "")
    authority_receipt = receipts.get(authority_hash)
    if (
        not isinstance(authority_receipt, Mapping)
        or authority_receipt.get("role") != "gateway_coordinator"
        or authority_receipt.get("purpose") != "research_lab.allocation.v2"
        or authority_receipt.get("epoch_id") != int(epoch_id)
        or authority_receipt.get("status") != "succeeded"
        or authority_receipt.get("output_root")
        != sha256_json({"allocation": dict(allocation)})
    ):
        raise ChampionSettlementV2Error(
            "finalized weight bundle allocation authority receipt is invalid"
        )
    return authority_hash


def validate_finalized_allocation_authorities_v2(
    rows: Sequence[Mapping[str, Any]],
    *,
    finalization_graphs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one allocation document per independently finalized epoch.

    Multiple validators may finalize the same epoch.  They never create extra
    payment credit: every accepted authority for an epoch must bind the same
    allocation document and the result is emitted once.
    """

    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_bundles: set[str] = set()
    for raw_row in rows:
        row = dict(raw_row)
        bundle_doc = row.get("bundle_doc")
        finalization_doc = row.get("finalization_doc")
        if not isinstance(bundle_doc, Mapping) or not isinstance(
            finalization_doc, Mapping
        ):
            raise ChampionSettlementV2Error(
                "finalized allocation authority documents are missing"
            )
        bundle = validate_published_weight_bundle_v2(bundle_doc)
        bundle_hash = str(row.get("bundle_hash") or "")
        if bundle_hash in seen_bundles:
            raise ChampionSettlementV2Error(
                "finalized allocation authority bundle is duplicated"
            )
        seen_bundles.add(bundle_hash)

        expected_bundle_row = {
            "bundle_hash": bundle["bundle_hash"],
            "schema_version": str(bundle_doc["schema_version"]),
            "netuid": bundle["netuid"],
            "epoch_id": bundle["epoch_id"],
            "block": bundle["block"],
            "validator_hotkey": bundle["validator_hotkey"],
            "root_receipt_hash": bundle["root_receipt_hash"],
            "weights_hash": bundle["weights_hash"],
            "snapshot_hash": bundle["snapshot_hash"],
            "bundle_doc": dict(bundle_doc),
        }
        for field, expected in expected_bundle_row.items():
            if row.get(field) != expected:
                raise ChampionSettlementV2Error(
                    "finalized allocation bundle row differs at %s" % field
                )

        publication_doc = row.get("publication_doc")
        if not isinstance(publication_doc, Mapping):
            raise ChampionSettlementV2Error(
                "finalized allocation publication document is missing"
            )
        durable_hash = sha256_json(expected_bundle_row)
        expected_publication = {
            "schema_version": "leadpoet.weight_publication.v2",
            "bundle_hash": bundle_hash,
            "root_receipt_hash": bundle["root_receipt_hash"],
            "durable_readback_hash": durable_hash,
            "transparency_event_hash": str(
                row.get("transparency_event_hash") or ""
            ),
        }
        if dict(publication_doc) != expected_publication:
            raise ChampionSettlementV2Error(
                "finalized allocation publication differs from its bundle"
            )
        submission_event_hash = sha256_json(
            {
                "bundle_hash": bundle_hash,
                "publication_receipt_hash": str(
                    row.get("publication_receipt_hash") or ""
                ),
                "transparency_event_hash": expected_publication[
                    "transparency_event_hash"
                ],
                "durable_readback_hash": durable_hash,
            }
        )
        if row.get("weight_submission_event_hash") != submission_event_hash:
            raise ChampionSettlementV2Error(
                "finalized allocation publication event hash differs"
            )

        finalization_receipt_hash = str(
            row.get("finalization_receipt_hash") or ""
        )
        graph = finalization_graphs.get(finalization_receipt_hash)
        if not isinstance(graph, Mapping) or str(
            graph.get("root_receipt_hash") or ""
        ) != finalization_receipt_hash:
            raise ChampionSettlementV2Error(
                "finalized allocation receipt graph is missing"
            )
        submission = {
            "schema_version": "leadpoet.weight_finalization_submission.v2",
            "validator_hotkey": bundle["validator_hotkey"],
            "weight_submission_event_hash": submission_event_hash,
            "finalization": dict(finalization_doc),
            "receipt_graph": dict(graph),
        }
        finalization = validate_weight_finalization_submission_v2(submission)
        for field in (
            "validator_hotkey",
            "netuid",
            "epoch_id",
            "weights_hash",
            "weight_receipt_hash",
        ):
            if finalization[field] != bundle[field]:
                raise ChampionSettlementV2Error(
                    "finalized allocation differs from bundle at %s" % field
                )
        expected_finalization_event = sha256_json(
            {
                "weight_submission_event_hash": submission_event_hash,
                "bundle_hash": bundle_hash,
                "finalization_receipt_hash": finalization_receipt_hash,
                "extrinsic_authorization_hash": finalization[
                    "extrinsic_authorization_hash"
                ],
                "extrinsic_hash": finalization["extrinsic_hash"],
                "finalized_block": finalization["finalized_block"],
                "finalized_block_hash": finalization["finalized_block_hash"],
                "state_transition_hash": finalization[
                    "state_transition_hash"
                ],
            }
        )
        if row.get("weight_finalization_event_hash") != expected_finalization_event:
            raise ChampionSettlementV2Error(
                "finalized allocation event hash differs"
            )

        snapshot = bundle_doc.get("weight_snapshot")
        calculation = (
            snapshot.get("calculation_snapshot")
            if isinstance(snapshot, Mapping)
            else None
        )
        input_receipts = (
            snapshot.get("input_receipt_hashes")
            if isinstance(snapshot, Mapping)
            else None
        )
        allocation_doc = (
            calculation.get("research_lab_allocation_doc")
            if isinstance(calculation, Mapping)
            else None
        )
        allocation_receipt_hash = (
            str(input_receipts.get("research_lab_allocation") or "")
            if isinstance(input_receipts, Mapping)
            else ""
        )
        if not isinstance(allocation_doc, Mapping) or not allocation_receipt_hash:
            raise ChampionSettlementV2Error(
                "finalized weight bundle has no Research Lab allocation input"
            )
        allocation = dict(allocation_doc)
        allocation_hash = str(allocation.get("allocation_hash") or "")
        if allocation_hash != sha256_json(
            {key: value for key, value in allocation.items() if key != "allocation_hash"}
        ):
            raise ChampionSettlementV2Error(
                "finalized Research Lab allocation hash is invalid"
            )
        epoch_id = int(bundle["epoch_id"])
        allocation_authority_receipt_hash = (
            _allocation_authority_receipt_hash_v2(
                bundle_doc=bundle_doc,
                allocation_input_receipt_hash=allocation_receipt_hash,
                allocation=allocation,
                epoch_id=epoch_id,
            )
        )
        by_epoch[epoch_id].append(
            {
                "epoch": epoch_id,
                "netuid": int(bundle["netuid"]),
                "allocation_hash": allocation_hash,
                "allocation_doc": allocation,
                "allocation_receipt_hash": allocation_receipt_hash,
                "allocation_authority_receipt_hash": (
                    allocation_authority_receipt_hash
                ),
                "bundle_hash": bundle_hash,
                "validator_hotkey": bundle["validator_hotkey"],
                "finalization_receipt_hash": finalization_receipt_hash,
            }
        )

    settled: list[dict[str, Any]] = []
    for epoch_id in sorted(by_epoch):
        authorities = by_epoch[epoch_id]
        commitments = {
            (
                str(item["allocation_hash"]),
                str(item["allocation_receipt_hash"]),
                str(item["allocation_authority_receipt_hash"]),
            )
            for item in authorities
        }
        if len(commitments) != 1:
            raise ChampionSettlementV2Error(
                "finalized validators disagree on epoch %d allocation" % epoch_id
            )
        first = authorities[0]
        settled.append(
            {
                "epoch": epoch_id,
                "netuid": int(first["netuid"]),
                "allocation_hash": str(first["allocation_hash"]),
                "allocation_doc": dict(first["allocation_doc"]),
                "allocation_receipt_hash": str(first["allocation_receipt_hash"]),
                "allocation_authority_receipt_hash": str(
                    first["allocation_authority_receipt_hash"]
                ),
                "finalized_authority_count": len(authorities),
                "finalized_bundle_hashes": sorted(
                    str(item["bundle_hash"]) for item in authorities
                ),
                "finalization_receipt_hashes": sorted(
                    str(item["finalization_receipt_hash"])
                    for item in authorities
                ),
            }
        )
    return settled


def validate_legacy_settlement_migrations_v2(
    rows: Sequence[Mapping[str, Any]],
    *,
    receipt_graphs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate append-only pre-V2 settlement rows and their V2 receipts."""

    settled: list[dict[str, Any]] = []
    seen_epochs: set[tuple[int, int]] = set()
    for raw_row in rows:
        row = dict(raw_row)
        document_value = row.get("settlement_doc")
        if not isinstance(document_value, Mapping):
            raise ChampionSettlementV2Error(
                "legacy settlement document is missing"
            )
        document = validate_legacy_settlement_document_v2(document_value)
        expected = {
            "netuid": int(document["netuid"]),
            "epoch_id": int(document["epoch_id"]),
            "schema_version": str(document["schema_version"]),
            "allocation_hash": str(document["allocation_hash"]),
            "settlement_hash": str(document["settlement_hash"]),
            "allocation_doc": dict(document["allocation_doc"]),
            "settlement_doc": dict(document),
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ChampionSettlementV2Error(
                    "legacy settlement row differs at %s" % field
                )
        key = (expected["netuid"], expected["epoch_id"])
        if key in seen_epochs:
            raise ChampionSettlementV2Error(
                "legacy settlement epoch is duplicated"
            )
        seen_epochs.add(key)
        receipt_hash = str(row.get("settlement_receipt_hash") or "")
        graph = receipt_graphs.get(receipt_hash)
        if not isinstance(graph, Mapping):
            raise ChampionSettlementV2Error(
                "legacy settlement receipt graph is missing"
            )
        validate_receipt_graph(graph)
        if graph.get("root_receipt_hash") != receipt_hash:
            raise ChampionSettlementV2Error(
                "legacy settlement receipt graph root differs"
            )
        root = next(
            (
                receipt
                for receipt in graph.get("receipts") or ()
                if isinstance(receipt, Mapping)
                and receipt.get("receipt_hash") == receipt_hash
            ),
            None,
        )
        if (
            not isinstance(root, Mapping)
            or root.get("role") != "gateway_coordinator"
            or root.get("purpose")
            != "research_lab.legacy_finalized_allocation.v2"
            or root.get("status") != "succeeded"
            or root.get("output_root") != sha256_json(document)
        ):
            raise ChampionSettlementV2Error(
                "legacy settlement receipt differs"
            )
        settled.append(
            {
                "epoch": expected["epoch_id"],
                "netuid": expected["netuid"],
                "allocation_hash": expected["allocation_hash"],
                "allocation_doc": expected["allocation_doc"],
                "allocation_receipt_hash": receipt_hash,
                "finalized_authority_count": 1,
                "authority_types": ["legacy_finalized_chain_migration_v2"],
                "legacy_settlement_receipt_hash": receipt_hash,
                "legacy_settlement_hash": expected["settlement_hash"],
                "finalized_bundle_hashes": [],
                "finalization_receipt_hashes": [],
            }
        )
    return sorted(settled, key=lambda item: int(item["epoch"]))


def validate_legacy_allocation_nonfinalizations_v2(
    rows: Sequence[Mapping[str, Any]],
    *,
    receipt_graphs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate append-only proof that legacy allocations were not paid."""

    findings: list[dict[str, Any]] = []
    seen_epochs: set[tuple[int, int]] = set()
    for raw_row in rows:
        row = dict(raw_row)
        document_value = row.get("finding_doc")
        if not isinstance(document_value, Mapping):
            raise ChampionSettlementV2Error(
                "legacy nonfinalization document is missing"
            )
        document = validate_legacy_nonfinalization_document_v2(
            document_value
        )
        expected = {
            "netuid": int(document["netuid"]),
            "epoch_id": int(document["epoch_id"]),
            "schema_version": str(document["schema_version"]),
            "allocation_hash": str(document["allocation_hash"]),
            "finding_hash": str(document["finding_hash"]),
            "allocation_doc": dict(document["allocation_doc"]),
            "finding_doc": dict(document),
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ChampionSettlementV2Error(
                    "legacy nonfinalization row differs at %s" % field
                )
        key = (expected["netuid"], expected["epoch_id"])
        if key in seen_epochs:
            raise ChampionSettlementV2Error(
                "legacy nonfinalization epoch is duplicated"
            )
        seen_epochs.add(key)
        receipt_hash = str(row.get("finding_receipt_hash") or "")
        graph = receipt_graphs.get(receipt_hash)
        if not isinstance(graph, Mapping):
            raise ChampionSettlementV2Error(
                "legacy nonfinalization receipt graph is missing"
            )
        validate_receipt_graph(graph)
        if graph.get("root_receipt_hash") != receipt_hash:
            raise ChampionSettlementV2Error(
                "legacy nonfinalization receipt graph root differs"
            )
        root = next(
            (
                receipt
                for receipt in graph.get("receipts") or ()
                if isinstance(receipt, Mapping)
                and receipt.get("receipt_hash") == receipt_hash
            ),
            None,
        )
        if (
            not isinstance(root, Mapping)
            or root.get("role") != "gateway_coordinator"
            or root.get("purpose")
            != "research_lab.legacy_finalized_allocation.v2"
            or root.get("status") != "succeeded"
            or root.get("output_root") != sha256_json(document)
        ):
            raise ChampionSettlementV2Error(
                "legacy nonfinalization receipt differs"
            )
        findings.append(
            {
                "epoch": expected["epoch_id"],
                "netuid": expected["netuid"],
                "allocation_hash": expected["allocation_hash"],
                "allocation_doc": expected["allocation_doc"],
                "finding_hash": expected["finding_hash"],
                "finding_receipt_hash": receipt_hash,
                "finding_doc": expected["finding_doc"],
            }
        )
    return sorted(findings, key=lambda item: int(item["epoch"]))


def merge_finalized_allocation_histories_v2(
    native_rows: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse all authorities to one allocation credit per netuid/epoch."""

    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for source, rows in (
        ("native_v2_finalization", native_rows),
        ("legacy_finalized_chain_migration_v2", legacy_rows),
    ):
        for raw_row in rows:
            row = dict(raw_row)
            key = (int(row["netuid"]), int(row["epoch"]))
            existing = merged.get(key)
            if existing is None:
                row["authority_types"] = sorted(
                    set(row.get("authority_types") or ()) | {source}
                )
                merged[key] = row
                continue
            if (
                existing.get("allocation_hash") != row.get("allocation_hash")
                or existing.get("allocation_doc") != row.get("allocation_doc")
            ):
                raise ChampionSettlementV2Error(
                    "finalized allocation authorities conflict for epoch %d"
                    % key[1]
                )
            existing["authority_types"] = sorted(
                set(existing.get("authority_types") or ())
                | set(row.get("authority_types") or ())
                | {source}
            )
            existing["finalized_authority_count"] = max(
                int(existing.get("finalized_authority_count") or 0),
                int(row.get("finalized_authority_count") or 0),
            )
            for field in (
                "finalized_bundle_hashes",
                "finalization_receipt_hashes",
            ):
                existing[field] = sorted(
                    set(existing.get(field) or ()) | set(row.get(field) or ())
                )
            for field in (
                "legacy_settlement_receipt_hash",
                "legacy_settlement_hash",
            ):
                if row.get(field):
                    existing[field] = row[field]
    return [merged[key] for key in sorted(merged)]


async def load_finalized_allocation_history_v2(
    *,
    netuid: int,
    start_epoch: int,
    end_epoch: int,
) -> list[dict[str, Any]]:
    """Load and verify finalized allocation epochs from the durable V2 store."""

    if int(end_epoch) < int(start_epoch):
        return []
    from gateway.research_lab.attested_v2_store import load_receipt_graphs_v2
    from gateway.research_lab.store import select_all

    native_rows = await select_all(
        FINALIZED_ALLOCATION_VIEW_V2,
        filters=(
            ("netuid", int(netuid)),
            ("epoch_id", "gte", int(start_epoch)),
            ("epoch_id", "lte", int(end_epoch)),
        ),
        order_by=(("epoch_id", False), ("validator_hotkey", False)),
        max_rows=max(1000, (int(end_epoch) - int(start_epoch) + 1) * 100),
        allow_partial=False,
    )
    legacy_rows = await select_all(
        LEGACY_SETTLEMENT_TABLE_V2,
        filters=(
            ("netuid", int(netuid)),
            ("epoch_id", "gte", int(start_epoch)),
            ("epoch_id", "lte", int(end_epoch)),
        ),
        order_by=(("epoch_id", False),),
        max_rows=max(1000, int(end_epoch) - int(start_epoch) + 1),
        allow_partial=False,
    )
    native_roots = {
        str(row.get("finalization_receipt_hash") or "")
        for row in native_rows
        if row.get("finalization_receipt_hash")
    }
    migration_roots = {
        str(row.get("settlement_receipt_hash") or "")
        for row in legacy_rows
        if row.get("settlement_receipt_hash")
    }
    loaded_graphs = await load_receipt_graphs_v2(
        native_roots | migration_roots
    )
    graphs = {
        root: loaded_graphs[root]
        for root in native_roots
        if root in loaded_graphs
    }
    migration_graphs = {
        root: loaded_graphs[root]
        for root in migration_roots
        if root in loaded_graphs
    }
    native = validate_finalized_allocation_authorities_v2(
        native_rows,
        finalization_graphs=graphs,
    )
    migrated = validate_legacy_settlement_migrations_v2(
        legacy_rows,
        receipt_graphs=migration_graphs,
    )
    return merge_finalized_allocation_histories_v2(native, migrated)


async def load_legacy_allocation_nonfinalizations_v2(
    *,
    netuid: int,
    start_epoch: int,
    end_epoch: int,
) -> list[dict[str, Any]]:
    """Load measured findings that create no historical payment credit."""

    if int(end_epoch) < int(start_epoch):
        return []
    from gateway.research_lab.attested_v2_store import load_receipt_graphs_v2
    from gateway.research_lab.store import select_all

    rows = await select_all(
        LEGACY_NONFINALIZATION_TABLE_V2,
        filters=(
            ("netuid", int(netuid)),
            ("epoch_id", "gte", int(start_epoch)),
            ("epoch_id", "lte", int(end_epoch)),
        ),
        order_by=(("epoch_id", False),),
        max_rows=max(1000, int(end_epoch) - int(start_epoch) + 1),
        allow_partial=False,
    )
    roots = {
        str(row.get("finding_receipt_hash") or "")
        for row in rows
        if row.get("finding_receipt_hash")
    }
    loaded_graphs = await load_receipt_graphs_v2(roots)
    graphs = {
        root: loaded_graphs[root]
        for root in roots
        if root in loaded_graphs
    }
    return validate_legacy_allocation_nonfinalizations_v2(
        rows,
        receipt_graphs=graphs,
    )


def _legacy_allocation_active_champion_payment_v2(
    raw_row: Mapping[str, Any],
    *,
    netuid: int,
    active_reward_ids: set[str],
    active_source_reward_ids: set[str] | None = None,
) -> tuple[int, str, bool]:
    allocation = raw_row.get("allocation_doc")
    allocation_hash = str(raw_row.get("allocation_hash") or "")
    try:
        row_epoch = int(raw_row.get("epoch"))
        row_netuid = int(raw_row.get("netuid"))
    except (TypeError, ValueError) as exc:
        raise ChampionSettlementV2Error(
            "historical allocation scope is invalid"
        ) from exc
    if (
        not isinstance(allocation, Mapping)
        or row_netuid != int(netuid)
        or int(allocation.get("epoch")) != row_epoch
        or (
            "netuid" in allocation
            and int(allocation.get("netuid")) != int(netuid)
        )
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
        raise ChampionSettlementV2Error("historical allocation hash differs")

    pays_active = False
    for section in (
        "champion_allocations",
        "queued_champion_allocations",
    ):
        values = allocation.get(section) or []
        if not isinstance(values, list):
            raise ChampionSettlementV2Error(
                "historical champion allocation list is invalid"
            )
        for item in values:
            if not isinstance(item, Mapping):
                raise ChampionSettlementV2Error(
                    "historical champion allocation is invalid"
                )
            reward_id = str(
                item.get("source_id")
                or item.get("champion_reward_id")
                or ""
            )
            if (
                reward_id in active_reward_ids
                and Decimal(str(item.get("paid_alpha_percent") or 0)) > 0
            ):
                pays_active = True
    source_values = allocation.get("source_add_allocations") or []
    if not isinstance(source_values, list):
        raise ChampionSettlementV2Error(
            "historical SOURCE_ADD allocation list is invalid"
        )
    source_reward_ids = set(active_source_reward_ids or ())
    for item in source_values:
        if not isinstance(item, Mapping):
            raise ChampionSettlementV2Error(
                "historical SOURCE_ADD allocation is invalid"
            )
        reward_id = str(
            item.get("source_add_reward_id")
            or item.get("source_id")
            or ""
        )
        if (
            reward_id in source_reward_ids
            and Decimal(str(item.get("paid_alpha_percent") or 0)) > 0
        ):
            pays_active = True
    return row_epoch, allocation_hash, pays_active


async def champion_v2_cutover_readiness(
    *,
    epoch: int,
    netuid: int,
    _finalized_history_out: list[dict[str, Any]] | None = None,
    _business_graphs_out: dict[
        tuple[str, str], dict[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    """Prove every positive-balance champion has one exact V2 receipt."""

    from gateway.research_lab.allocations import (
        SETTLEMENT_TRACKED_CHAMPION_STATUSES,
        _champion_obligation_caps,
        _champion_paid_alpha_to_date_from_snapshots,
    )
    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graph_by_ref_v2,
        load_business_artifact_graphs_by_ref_v2,
    )
    from gateway.research_lab.store import select_all
    from gateway.tee.reward_executor_v2 import champion_reward_row_projection_v2

    rows: list[dict[str, Any]] = []
    for status in sorted(SETTLEMENT_TRACKED_CHAMPION_STATUSES):
        rows.extend(
            await select_all(
                "research_lab_champion_reward_current",
                filters=(("current_reward_status", status),),
                order_by=(("start_epoch", False), ("champion_reward_id", False)),
                max_rows=10000,
                allow_partial=False,
            )
        )
    source_rows: list[dict[str, Any]] = []
    for status in sorted(SETTLEMENT_TRACKED_CHAMPION_STATUSES):
        source_rows.extend(
            await select_all(
                "research_lab_source_add_reward_current",
                filters=(("current_reward_status", status),),
                order_by=(("start_epoch", False), ("reward_ref", False)),
                max_rows=10000,
                allow_partial=False,
            )
        )
    starts = [
        int(row.get("start_epoch") or 0)
        for row in rows + source_rows
        if int(row.get("start_epoch") or 0) <= int(epoch)
    ]
    finalized = (
        await load_finalized_allocation_history_v2(
            netuid=int(netuid),
            start_epoch=min(starts),
            end_epoch=int(epoch) - 1,
        )
        if starts and int(epoch) > 0
        else []
    )
    nonfinalized = (
        await load_legacy_allocation_nonfinalizations_v2(
            netuid=int(netuid),
            start_epoch=min(starts),
            end_epoch=int(epoch) - 1,
        )
        if starts and int(epoch) > 0
        else []
    )
    if _finalized_history_out is not None:
        _finalized_history_out.clear()
        _finalized_history_out.extend(dict(item) for item in finalized)
    legacy_allocations = (
        await select_all(
            "research_lab_emission_allocation_current",
            filters=(
                ("netuid", int(netuid)),
                ("epoch", "gte", min(starts)),
                ("epoch", "lt", int(epoch)),
            ),
            order_by=(("epoch", False),),
            max_rows=max(10000, int(epoch) - min(starts) + 100),
            allow_partial=False,
        )
        if starts and int(epoch) > 0
        else []
    )
    caps = _champion_obligation_caps(rows)
    paid = _champion_paid_alpha_to_date_from_snapshots(
        finalized,
        obligation_caps=caps,
    )
    positive: list[dict[str, Any]] = []
    settled: list[dict[str, Any]] = []
    for row in rows:
        reward_id = str(row.get("champion_reward_id") or "")
        total_due = caps.get(reward_id, Decimal("0"))
        credited = min(total_due, Decimal(str(paid.get(reward_id, 0))))
        remaining = max(Decimal("0"), total_due - credited)
        summary = {
            "champion_reward_id": reward_id,
            "current_reward_status": str(
                row.get("current_reward_status") or ""
            ),
            "total_due_alpha_percent": float(total_due),
            "paid_alpha_percent_to_date": float(credited),
            "remaining_alpha_percent": float(remaining),
        }
        if remaining > 0:
            positive.append({**summary, "row": row})
        else:
            settled.append(summary)

    covered: list[str] = []
    missing: list[dict[str, Any]] = []
    decision_graphs: dict[tuple[str, str], dict[str, Any]] = {}
    decision_refs = {
        (
            "champion_reward_decision",
            str(item["champion_reward_id"]),
        )
        for item in positive
    }
    if decision_refs:
        try:
            decision_graphs = await load_business_artifact_graphs_by_ref_v2(
                decision_refs
            )
        except Exception as exc:
            logger.warning(
                "research_lab_champion_v2_cutover_batch_receipt_fallback "
                "count=%d error=%s",
                len(decision_refs),
                str(exc)[:240],
            )
    for item in positive:
        row = item["row"]
        reward_id = str(item["champion_reward_id"])
        try:
            graph = decision_graphs.get(
                ("champion_reward_decision", reward_id)
            )
            if not isinstance(graph, Mapping):
                graph = await load_business_artifact_graph_by_ref_v2(
                    artifact_kind="champion_reward_decision",
                    artifact_ref=reward_id,
                )
            root_hash = str(graph.get("root_receipt_hash") or "")
            root = next(
                (
                    receipt
                    for receipt in graph.get("receipts") or ()
                    if isinstance(receipt, Mapping)
                    and receipt.get("receipt_hash") == root_hash
                ),
                None,
            )
            expected_output = sha256_json(champion_reward_row_projection_v2(row))
            if (
                not isinstance(root, Mapping)
                or root.get("role") != "gateway_coordinator"
                or root.get("purpose") != "research_lab.reward_decision.v2"
                or root.get("output_root") != expected_output
            ):
                raise ChampionSettlementV2Error(
                    "champion reward receipt projection differs"
                )
            covered.append(reward_id)
            if _business_graphs_out is not None:
                _business_graphs_out[
                    ("champion_reward_decision", reward_id)
                ] = dict(graph)
        except Exception as exc:
            logger.warning(
                "research_lab_champion_v2_cutover_receipt_uncovered "
                "reward_id=%s error=%s",
                reward_id,
                str(exc)[:240],
            )
            missing.append(
                {
                    "champion_reward_id": reward_id,
                    "remaining_alpha_percent": item[
                        "remaining_alpha_percent"
                    ],
                    "reason": "missing_or_invalid_v2_reward_receipt",
                }
            )
    required_count = len(positive)
    covered_count = len(covered)
    coverage = 1.0 if required_count == 0 else covered_count / required_count
    active_reward_ids = {
        str(row.get("champion_reward_id") or "") for row in rows
    }
    active_source_reward_ids = {
        str(row.get("reward_ref") or "") for row in source_rows
    }
    finalized_by_epoch = {
        int(item["epoch"]): item for item in finalized
    }
    nonfinalized_by_epoch = {
        int(item["epoch"]): item for item in nonfinalized
    }
    conflicting_classification_epochs = sorted(
        set(finalized_by_epoch) & set(nonfinalized_by_epoch)
    )
    invalid_settlements: list[dict[str, Any]] = []
    for conflict_epoch in conflicting_classification_epochs:
        invalid_settlements.append(
            {
                "epoch": conflict_epoch,
                "reason": "conflicting_historical_chain_classifications",
            }
        )
    current_payment_allocations: dict[int, str] = {}
    for row in legacy_allocations:
        try:
            row_epoch, allocation_hash, pays_active = (
                _legacy_allocation_active_champion_payment_v2(
                    row,
                    netuid=int(netuid),
                    active_reward_ids=active_reward_ids,
                    active_source_reward_ids=active_source_reward_ids,
                )
            )
            if pays_active:
                existing_hash = current_payment_allocations.get(row_epoch)
                if existing_hash and existing_hash != allocation_hash:
                    raise ChampionSettlementV2Error(
                        "historical allocation epoch is ambiguous"
                    )
                current_payment_allocations[row_epoch] = allocation_hash
        except Exception as exc:
            invalid_settlements.append(
                {
                    "epoch": row.get("epoch"),
                    "reason": "invalid_historical_allocation",
                    "error": str(exc)[:240],
                }
            )

    candidate_anchors: list[dict[str, Any]] = []
    candidate_bundles: list[dict[str, Any]] = []
    historical_snapshots: list[dict[str, Any]] = []
    if starts and int(epoch) > 0:
        candidate_start = min(starts)
        candidate_end = int(epoch) - 1
        candidate_max_rows = max(
            10000,
            (candidate_end - candidate_start + 1) * 100,
        )
        candidate_anchors = await select_all(
            "research_lab_arweave_epoch_audit_anchor_current",
            filters=(
                ("netuid", int(netuid)),
                ("epoch", "gte", candidate_start),
                ("epoch", "lte", candidate_end),
                ("audit_kind", "active"),
                ("current_anchor_status", "checkpointed"),
            ),
            order_by=(("epoch", False), ("current_status_at", False)),
            max_rows=candidate_max_rows,
            allow_partial=False,
        )
        candidate_bundles = await select_all(
            "published_weight_bundles",
            filters=(
                ("netuid", int(netuid)),
                ("epoch_id", "gte", candidate_start),
                ("epoch_id", "lte", candidate_end),
            ),
            order_by=(("epoch_id", False), ("created_at", False)),
            max_rows=candidate_max_rows,
            allow_partial=False,
        )
        historical_snapshots = await select_all(
            "research_lab_emission_allocation_snapshots",
            filters=(
                ("netuid", int(netuid)),
                ("epoch", "gte", candidate_start),
                ("epoch", "lte", candidate_end),
            ),
            order_by=(("epoch", False), ("created_at", False)),
            max_rows=candidate_max_rows,
            allow_partial=False,
        )

    anchors_by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_anchors:
        try:
            anchors_by_epoch[int(row.get("epoch"))].append(dict(row))
        except (TypeError, ValueError):
            invalid_settlements.append(
                {
                    "epoch": row.get("epoch"),
                    "reason": "invalid_historical_settlement_candidate",
                    "error": "checkpointed audit anchor epoch is invalid",
                }
            )
    bundles_by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_bundles:
        try:
            bundles_by_epoch[int(row.get("epoch_id"))].append(dict(row))
        except (TypeError, ValueError):
            invalid_settlements.append(
                {
                    "epoch": row.get("epoch_id"),
                    "reason": "invalid_historical_settlement_candidate",
                    "error": "published weight bundle epoch is invalid",
                }
            )
    snapshots_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in historical_snapshots:
        snapshots_by_hash[str(row.get("allocation_hash") or "")].append(
            dict(row)
        )

    historical_payment_allocations: dict[int, str] = {}
    required_settlements: dict[int, str] = {}
    unproven_allocations: list[dict[str, Any]] = []
    unproven_keys: set[tuple[int, str, str]] = set()

    def add_unproven(epoch_id: int, allocation_hash: str, reason: str) -> None:
        key = (int(epoch_id), str(allocation_hash), str(reason))
        if key in unproven_keys:
            return
        unproven_keys.add(key)
        unproven_allocations.append(
            {
                "epoch": key[0],
                "allocation_hash": key[1],
                "reason": key[2],
            }
        )

    for settlement_epoch, authority in sorted(finalized_by_epoch.items()):
        try:
            authority_epoch, allocation_hash, pays_active = (
                _legacy_allocation_active_champion_payment_v2(
                    {
                        "epoch": settlement_epoch,
                        "netuid": authority.get("netuid"),
                        "allocation_hash": authority.get("allocation_hash"),
                        "allocation_doc": authority.get("allocation_doc"),
                    },
                    netuid=int(netuid),
                    active_reward_ids=active_reward_ids,
                )
            )
            if pays_active:
                historical_payment_allocations[authority_epoch] = (
                    allocation_hash
                )
        except Exception as exc:
            invalid_settlements.append(
                {
                    "epoch": settlement_epoch,
                    "reason": "invalid_finalized_historical_allocation",
                    "error": str(exc)[:240],
                }
            )

    candidate_epochs = sorted(
        set(current_payment_allocations) | set(anchors_by_epoch)
    )
    for settlement_epoch in candidate_epochs:
        current_hash = current_payment_allocations.get(settlement_epoch)
        epoch_anchors = anchors_by_epoch.get(settlement_epoch, [])
        if not epoch_anchors:
            if (
                current_hash
                and settlement_epoch not in historical_payment_allocations
            ):
                add_unproven(
                    settlement_epoch,
                    current_hash,
                    "no_checkpointed_audit_anchor",
                )
            continue
        if len(epoch_anchors) != 1:
            relevant = bool(current_hash)
            for candidate in epoch_anchors:
                candidate_hash = str(candidate.get("allocation_hash") or "")
                candidate_rows = snapshots_by_hash.get(candidate_hash, [])
                if len(candidate_rows) != 1:
                    relevant = True
                    continue
                try:
                    _, _, candidate_pays = (
                        _legacy_allocation_active_champion_payment_v2(
                            candidate_rows[0],
                            netuid=int(netuid),
                            active_reward_ids=active_reward_ids,
                        )
                    )
                except Exception:
                    relevant = True
                else:
                    relevant = relevant or candidate_pays
            if relevant:
                invalid_settlements.append(
                    {
                        "epoch": settlement_epoch,
                        "allocation_hash": current_hash or "",
                        "reason": "ambiguous_checkpointed_audit_anchor",
                    }
                )
            continue
        anchor = epoch_anchors[0]
        allocation_hash = str(anchor.get("allocation_hash") or "")
        if not allocation_hash:
            add_unproven(
                settlement_epoch,
                current_hash or "",
                "checkpointed_anchor_has_no_allocation",
            )
            continue
        exact_snapshots = snapshots_by_hash.get(allocation_hash, [])
        if len(exact_snapshots) != 1:
            if current_hash:
                invalid_settlements.append(
                    {
                        "epoch": settlement_epoch,
                        "allocation_hash": current_hash,
                        "anchor_allocation_hash": allocation_hash,
                        "reason": "anchor_bound_allocation_snapshot_missing_or_ambiguous",
                    }
                )
            else:
                add_unproven(
                    settlement_epoch,
                    allocation_hash,
                    "anchor_bound_allocation_snapshot_missing_or_ambiguous",
                )
            continue
        try:
            snapshot_epoch, snapshot_hash, anchor_pays_active = (
                _legacy_allocation_active_champion_payment_v2(
                    exact_snapshots[0],
                    netuid=int(netuid),
                    active_reward_ids=active_reward_ids,
                )
            )
            if (
                snapshot_epoch != settlement_epoch
                or snapshot_hash != allocation_hash
            ):
                raise ChampionSettlementV2Error(
                    "anchor-bound allocation scope differs"
                )
        except Exception as exc:
            invalid_settlements.append(
                {
                    "epoch": settlement_epoch,
                    "allocation_hash": allocation_hash,
                    "reason": "invalid_anchor_bound_historical_allocation",
                    "error": str(exc)[:240],
                }
            )
            continue

        existing_hash = historical_payment_allocations.get(settlement_epoch)
        if existing_hash:
            if anchor_pays_active and existing_hash != allocation_hash:
                invalid_settlements.append(
                    {
                        "epoch": settlement_epoch,
                        "allocation_hash": allocation_hash,
                        "finalized_allocation_hash": existing_hash,
                        "reason": "finalized_chain_allocation_hash_mismatch",
                    }
                )
            if current_hash and current_hash != existing_hash:
                add_unproven(
                    settlement_epoch,
                    current_hash,
                    "current_allocation_not_finalized",
                )
            continue
        if not anchor_pays_active:
            if current_hash:
                add_unproven(
                    settlement_epoch,
                    current_hash,
                    "current_allocation_not_checkpointed",
                )
            continue

        anchor_weights_hash = str(anchor.get("weights_hash") or "").removeprefix(
            "sha256:"
        )
        arweave_tx_id = str(anchor.get("current_arweave_tx_id") or "")
        transparency_event_hash = str(
            anchor.get("current_transparency_event_hash") or ""
        ).removeprefix("sha256:")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", anchor_weights_hash)
            or not re.fullmatch(r"[A-Za-z0-9_-]{43}", arweave_tx_id)
            or not re.fullmatch(r"[0-9a-f]{64}", transparency_event_hash)
        ):
            invalid_settlements.append(
                {
                    "epoch": settlement_epoch,
                    "allocation_hash": allocation_hash,
                    "reason": "invalid_checkpointed_audit_anchor",
                }
            )
            continue
        epoch_bundles = bundles_by_epoch.get(settlement_epoch, [])
        if not epoch_bundles:
            add_unproven(
                settlement_epoch,
                allocation_hash,
                "no_published_weight_bundle",
            )
            continue
        matching_bundles = [
            row
            for row in epoch_bundles
            if str(row.get("weights_hash") or "").removeprefix("sha256:")
            == anchor_weights_hash
        ]
        if not matching_bundles:
            invalid_settlements.append(
                {
                    "epoch": settlement_epoch,
                    "allocation_hash": allocation_hash,
                    "reason": "published_weight_bundle_hash_mismatch",
                }
            )
            continue
        historical_payment_allocations[settlement_epoch] = allocation_hash
        required_settlements[settlement_epoch] = allocation_hash
        if current_hash and current_hash != allocation_hash:
            add_unproven(
                settlement_epoch,
                current_hash,
                "current_allocation_not_checkpointed",
            )

    missing_classifications: list[dict[str, Any]] = list(
        invalid_settlements
    )
    covered_settlement_epochs: list[int] = []
    covered_nonfinalization_epochs: list[int] = []
    for settlement_epoch, allocation_hash in sorted(
        historical_payment_allocations.items()
    ):
        authority = finalized_by_epoch.get(settlement_epoch)
        finding = nonfinalized_by_epoch.get(settlement_epoch)
        if (
            authority is not None
            and finding is not None
        ):
            missing_classifications.append(
                {
                    "epoch": settlement_epoch,
                    "allocation_hash": allocation_hash,
                    "reason": "conflicting_historical_chain_classifications",
                }
            )
        elif (
            authority is not None
            and authority.get("allocation_hash") == allocation_hash
        ):
            covered_settlement_epochs.append(settlement_epoch)
        elif authority is not None:
            missing_classifications.append(
                {
                    "epoch": settlement_epoch,
                    "allocation_hash": allocation_hash,
                    "finalized_allocation_hash": authority.get("allocation_hash"),
                    "reason": "finalized_chain_allocation_hash_mismatch",
                }
            )
        elif (
            finding is not None
            and finding.get("allocation_hash") == allocation_hash
        ):
            covered_nonfinalization_epochs.append(settlement_epoch)
            add_unproven(
                settlement_epoch,
                allocation_hash,
                "finalized_chain_vector_mismatch",
            )
        elif finding is not None:
            missing_classifications.append(
                {
                    "epoch": settlement_epoch,
                    "allocation_hash": allocation_hash,
                    "nonfinalized_allocation_hash": finding.get(
                        "allocation_hash"
                    ),
                    "reason": "nonfinalized_chain_allocation_hash_mismatch",
                }
            )
        elif settlement_epoch in required_settlements:
            missing_classifications.append(
                {
                    "epoch": settlement_epoch,
                    "allocation_hash": allocation_hash,
                    "reason": "missing_finalized_chain_classification_authority",
                }
            )
    classification_required_count = (
        len(covered_settlement_epochs)
        + len(covered_nonfinalization_epochs)
        + len(missing_classifications)
    )
    classification_covered_count = (
        len(covered_settlement_epochs)
        + len(covered_nonfinalization_epochs)
    )
    return {
        "schema_version": "leadpoet.champion_v2_cutover_readiness.v1",
        "epoch": int(epoch),
        "netuid": int(netuid),
        "ready": (
            required_count == covered_count
            and classification_required_count
            == classification_covered_count
        ),
        "required_positive_balance_count": required_count,
        "covered_positive_balance_count": covered_count,
        "receipt_coverage": coverage,
        "covered_champion_reward_ids": sorted(covered),
        "missing": missing,
        "required_historical_classification_count": (
            classification_required_count
        ),
        "covered_historical_classification_count": (
            classification_covered_count
        ),
        "historical_classification_coverage": (
            1.0
            if classification_required_count == 0
            else classification_covered_count
            / classification_required_count
        ),
        "covered_historical_classification_epochs": sorted(
            covered_settlement_epochs + covered_nonfinalization_epochs
        ),
        "covered_historical_nonfinalization_epochs": (
            covered_nonfinalization_epochs
        ),
        "missing_historical_classifications": missing_classifications,
        # Compatibility aliases for the existing operator API. The gate now
        # requires a measured finalized/nonfinalized classification, not an
        # assumption that every published legacy bundle reached chain state.
        "required_historical_settlement_count": (
            classification_required_count
        ),
        "covered_historical_settlement_count": (
            classification_covered_count
        ),
        "historical_settlement_coverage": (
            1.0
            if classification_required_count == 0
            else classification_covered_count
            / classification_required_count
        ),
        "covered_historical_settlement_epochs": covered_settlement_epochs,
        "missing_historical_settlements": missing_classifications,
        "unproven_historical_allocation_count": len(unproven_allocations),
        "unproven_historical_allocations": unproven_allocations,
        "zero_balance_active_rows": settled,
        "finalized_allocation_epoch_count": len(finalized),
        "native_finalized_allocation_epoch_count": sum(
            1
            for item in finalized
            if "native_v2_finalization" in (item.get("authority_types") or ())
        ),
        "migrated_finalized_allocation_epoch_count": sum(
            1
            for item in finalized
            if "legacy_finalized_chain_migration_v2"
            in (item.get("authority_types") or ())
        ),
        "measured_nonfinalized_allocation_epoch_count": len(nonfinalized),
    }
