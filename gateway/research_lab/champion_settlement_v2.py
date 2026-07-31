"""Finalized-chain settlement authority for champion obligations.

Allocation snapshots describe what the gateway intended to pay.  They are not
payment evidence.  This module accepts an allocation epoch only when the exact
allocation receipt was consumed by a canonical V2 weight bundle and that
bundle has a canonical finalized-chain submission.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_EVEN
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
from leadpoet_verifier.economics import (
    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1,
)


FINALIZED_ALLOCATION_VIEW_V2 = "research_lab_finalized_allocation_epochs_v2"
LEGACY_SETTLEMENT_TABLE_V2 = (
    "research_lab_legacy_finalized_allocation_migrations_v2"
)
LEGACY_NONFINALIZATION_TABLE_V2 = (
    "research_lab_legacy_allocation_nonfinalizations_v2"
)
CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1 = (
    "research_lab_chain_realized_epoch_settlements_v1"
)
CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1 = (
    "research_lab_chain_realized_settlement_activation_v1"
)
CHAIN_REALIZED_OBLIGATION_CREDIT_TABLE_V1 = (
    "research_lab_chain_realized_obligation_credits_v1"
)
CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1 = (
    "leadpoet.research_lab_chain_realized_epoch_settlement.v1"
)
CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2 = (
    "leadpoet.research_lab_chain_realized_epoch_settlement.v2"
)
CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3 = (
    "leadpoet.research_lab_chain_realized_epoch_settlement.v3"
)
CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1 = (
    "leadpoet.research_lab_chain_realized_obligation_credit.v1"
)
CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2 = (
    "leadpoet.research_lab_chain_realized_obligation_credit.v2"
)
CHAIN_REALIZED_CHAMPION_CREDIT_POLICY_LEGACY_V1 = "scheduled_bonus_v1"
CHAIN_REALIZED_AUTHORITY_TYPE_V1 = "chain_realized_emission_v1"
CHAIN_REALIZED_UNATTRIBUTED_AUTHORITY_TYPE_V1 = (
    "chain_realized_unattributed_v1"
)
CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1 = (
    "leadpoet.chain_realized_weight_observation.v1"
)
CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1 = (
    "leadpoet.chain_realized_weight_observation_request.v1"
)
CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1 = (
    "leadpoet.chain_realized_settlement_request.v1"
)
CHAIN_WEIGHT_OBSERVATION_RECEIPT_PURPOSE_V1 = (
    "research_lab.chain_weight_observation.v1"
)
CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1 = (
    "research_lab.chain_realized_epoch_settlement.v1"
)
_CHAIN_CREDIT_SECTION_BY_KIND_V1 = {
    "champion": "champion_allocations",
    "queued_champion": "queued_champion_allocations",
    "source_add": "source_add_allocations",
    "reimbursement": "reimbursement_allocations",
}
logger = logging.getLogger(__name__)
_CHAIN_DECIMAL_QUANTUM_V1 = Decimal("0.000000000001")


class ChampionSettlementV2Error(RuntimeError):
    """Finalized weight evidence is missing, inconsistent, or tampered."""


def _chain_decimal_text_v1(value: Any, field: str) -> str:
    normalized = _non_negative_decimal_v1(value, field).quantize(
        _CHAIN_DECIMAL_QUANTUM_V1,
        rounding=ROUND_HALF_EVEN,
    )
    return format(normalized, "f")


def validate_chain_weight_observation_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "netuid",
        "epoch_id",
        "official_subnet_epoch_id",
        "cutover_mapping_hash",
        "close_block",
        "close_block_hash",
        "close_state_root",
        "next_epoch_block",
        "next_epoch_block_hash",
        "validator_hotkey",
        "validator_uid",
        "metagraph_hotkeys",
        "weights",
        "weights_storage_key",
        "last_update_storage_key",
        "last_update_block",
        "last_update_block_hash",
        "last_update_official_subnet_epoch_id",
        "active_source_epoch_id",
        "weights_vector_hash",
    }:
        raise ChampionSettlementV2Error(
            "chain weight observation fields are invalid"
        )
    if value.get("schema_version") != CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1:
        raise ChampionSettlementV2Error(
            "chain weight observation schema is invalid"
        )
    try:
        netuid = int(value["netuid"])
        epoch_id = int(value["epoch_id"])
        official_epoch = int(value["official_subnet_epoch_id"])
        close_block = int(value["close_block"])
        next_epoch_block = int(value["next_epoch_block"])
        validator_uid = int(value["validator_uid"])
        last_update_block = int(value["last_update_block"])
        last_update_official_epoch = int(
            value["last_update_official_subnet_epoch_id"]
        )
        active_source_epoch_id = int(value["active_source_epoch_id"])
    except (TypeError, ValueError) as exc:
        raise ChampionSettlementV2Error(
            "chain weight observation scope is invalid"
        ) from exc
    if (
        any(
            isinstance(value[field], bool)
            for field in (
                "netuid",
                "epoch_id",
                "official_subnet_epoch_id",
                "close_block",
                "next_epoch_block",
                "validator_uid",
                "last_update_block",
                "last_update_official_subnet_epoch_id",
                "active_source_epoch_id",
            )
        )
        or netuid <= 0
        or min(
            epoch_id,
            official_epoch,
            close_block,
            validator_uid,
            last_update_block,
            last_update_official_epoch,
            active_source_epoch_id,
        )
        < 0
        or next_epoch_block != close_block + 1
        or last_update_block > close_block
        or last_update_official_epoch > official_epoch
        or active_source_epoch_id > epoch_id
    ):
        raise ChampionSettlementV2Error(
            "chain weight observation scope is invalid"
        )
    for field in (
        "cutover_mapping_hash",
        "weights_vector_hash",
    ):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get(field) or "")):
            raise ChampionSettlementV2Error(
                "chain weight observation %s is invalid" % field
            )
    for field in (
        "close_block_hash",
        "close_state_root",
        "next_epoch_block_hash",
        "last_update_block_hash",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(field) or "")):
            raise ChampionSettlementV2Error(
                "chain weight observation %s is invalid" % field
            )
    validator_hotkey = str(value.get("validator_hotkey") or "")
    storage_key = str(value.get("weights_storage_key") or "")
    last_update_key = str(value.get("last_update_storage_key") or "")
    hotkeys = value.get("metagraph_hotkeys")
    weights = value.get("weights")
    if (
        not validator_hotkey
        or not storage_key.startswith("0x")
        or not last_update_key.startswith("0x")
        or not isinstance(hotkeys, list)
        or validator_uid >= len(hotkeys)
        or hotkeys[validator_uid] != validator_hotkey
        or any(not isinstance(item, str) or not item for item in hotkeys)
        or not isinstance(weights, list)
    ):
        raise ChampionSettlementV2Error(
            "chain weight observation identities are invalid"
        )
    normalized_weights: list[list[int]] = []
    seen_uids: set[int] = set()
    for pair in weights:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ChampionSettlementV2Error(
                "chain weight observation vector is invalid"
            )
        try:
            uid, weight = int(pair[0]), int(pair[1])
        except (TypeError, ValueError) as exc:
            raise ChampionSettlementV2Error(
                "chain weight observation vector is invalid"
            ) from exc
        if (
            isinstance(pair[0], bool)
            or isinstance(pair[1], bool)
            or uid < 0
            or uid >= len(hotkeys)
            or not 1 <= weight <= 65535
            or uid in seen_uids
        ):
            raise ChampionSettlementV2Error(
                "chain weight observation vector is invalid"
            )
        seen_uids.add(uid)
        normalized_weights.append([uid, weight])
    normalized_weights.sort()
    if normalized_weights != weights:
        raise ChampionSettlementV2Error(
            "chain weight observation vector is not canonical"
        )
    expected_vector_hash = sha256_json(
        {"uids": [item[0] for item in weights], "weights_u16": [item[1] for item in weights]}
    )
    if value.get("weights_vector_hash") != expected_vector_hash:
        raise ChampionSettlementV2Error(
            "chain weight observation vector hash differs"
        )
    return dict(value)


def _preliminary_finalized_bundle_authority_v1(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_doc = row.get("bundle_doc")
    finalization_doc = row.get("finalization_doc")
    if not isinstance(bundle_doc, Mapping) or not isinstance(
        finalization_doc, Mapping
    ):
        raise ChampionSettlementV2Error(
            "chain settlement bundle authority documents are missing"
        )
    bundle = validate_published_weight_bundle_v2(bundle_doc)
    expected_bundle_fields = {
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
    for field, expected in expected_bundle_fields.items():
        if row.get(field) != expected:
            raise ChampionSettlementV2Error(
                "chain settlement bundle row differs at %s" % field
            )
    try:
        finalized_block = int(finalization_doc["finalized_block"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChampionSettlementV2Error(
            "chain settlement finalization block is invalid"
        ) from exc
    for field, expected in (
        ("validator_hotkey", bundle["validator_hotkey"]),
        ("netuid", bundle["netuid"]),
        ("epoch_id", bundle["epoch_id"]),
        ("weights_hash", bundle["weights_hash"]),
        ("finalized_block", finalized_block),
    ):
        if finalization_doc.get(field) != expected or row.get(field) != expected:
            raise ChampionSettlementV2Error(
                "chain settlement finalization differs at %s" % field
            )
    if (
        finalization_doc.get("weight_receipt_hash")
        != bundle["weight_receipt_hash"]
    ):
        raise ChampionSettlementV2Error(
            "chain settlement finalization differs at weight_receipt_hash"
        )
    return {
        **bundle,
        "bundle_doc": dict(bundle_doc),
        "finalized_block": finalized_block,
        "finalized_block_hash": str(
            finalization_doc.get("finalized_block_hash") or ""
        ),
        "finalization_receipt_hash": str(
            row.get("finalization_receipt_hash") or ""
        ),
    }


def select_chain_realized_bundle_candidate_v1(
    rows: Sequence[Mapping[str, Any]],
    *,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    observed = validate_chain_weight_observation_v1(observation)
    observed_pairs = [list(item) for item in observed["weights"]]
    candidates = []
    for row in rows:
        authority = _preliminary_finalized_bundle_authority_v1(row)
        if (
            int(authority["netuid"]) == int(observed["netuid"])
            and int(authority["epoch_id"])
            == int(observed["active_source_epoch_id"])
            and authority["validator_hotkey"]
            == observed["validator_hotkey"]
            and int(authority["finalized_block"])
            == int(observed["last_update_block"])
            and authority["finalized_block_hash"]
            == observed["last_update_block_hash"]
            and [
                [int(uid), int(weight)]
                for uid, weight in zip(
                    authority["uids"],
                    authority["weights_u16"],
                )
            ]
            == observed_pairs
        ):
            candidates.append(authority)
    if not candidates:
        raise ChampionSettlementV2Error(
            "no finalized canonical bundle matches the active chain vector"
        )
    latest_block = max(int(item["finalized_block"]) for item in candidates)
    latest = [
        item for item in candidates if int(item["finalized_block"]) == latest_block
    ]
    identities = {
        (
            str(item["bundle_hash"]),
            str(item["finalization_receipt_hash"]),
        )
        for item in latest
    }
    if len(identities) != 1:
        raise ChampionSettlementV2Error(
            "active chain vector has ambiguous finalized bundle authority"
        )
    return dict(latest[0])


def build_chain_realized_settlement_package_v1(
    *,
    observation: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    observed = validate_chain_weight_observation_v1(observation)
    bundle_doc = authority.get("bundle_doc")
    if not isinstance(bundle_doc, Mapping):
        raise ChampionSettlementV2Error(
            "chain settlement bundle document is missing"
        )
    bundle = validate_published_weight_bundle_v2(bundle_doc)
    if (
        str(authority.get("bundle_hash") or "") != bundle["bundle_hash"]
        or int(bundle["netuid"]) != int(observed["netuid"])
        or [
            [int(uid), int(weight)]
            for uid, weight in zip(bundle["uids"], bundle["weights_u16"])
        ]
        != observed["weights"]
    ):
        raise ChampionSettlementV2Error(
            "chain settlement bundle differs from active weights"
        )
    weight_result = bundle_doc["weight_result"]
    planned_uid_weight_percent = {
        int(uid): Decimal(str(weight)) * Decimal("100")
        for uid, weight in zip(
            weight_result["uids"],
            weight_result["weights"],
        )
    }
    total_observed_weight = sum(
        int(weight) for _uid, weight in observed["weights"]
    )
    if total_observed_weight <= 0:
        raise ChampionSettlementV2Error(
            "chain settlement active weight vector is empty"
        )
    observed_uid_weight_percent = {
        int(uid): (
            Decimal(int(weight))
            * Decimal("100")
            / Decimal(total_observed_weight)
        ).quantize(_CHAIN_DECIMAL_QUANTUM_V1, rounding=ROUND_HALF_EVEN)
        for uid, weight in observed["weights"]
    }
    snapshot = bundle_doc["weight_snapshot"]
    calculation = snapshot["calculation_snapshot"]
    allocation = calculation.get("research_lab_allocation_doc")
    input_receipts = snapshot.get("input_receipt_hashes")
    if not isinstance(allocation, Mapping) or not isinstance(
        input_receipts, Mapping
    ):
        raise ChampionSettlementV2Error(
            "chain settlement allocation evidence is missing"
        )
    allocation_hash = str(allocation.get("allocation_hash") or "")
    if allocation_hash != sha256_json(
        {key: value for key, value in allocation.items() if key != "allocation_hash"}
    ):
        raise ChampionSettlementV2Error(
            "chain settlement allocation hash is invalid"
        )
    champion_credit_policy = allocation.get("champion_credit_policy")
    if champion_credit_policy not in (
        None,
        CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1,
    ):
        raise ChampionSettlementV2Error(
            "chain settlement champion credit policy is invalid"
        )
    lifetime_cap_policy = (
        champion_credit_policy
        == CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
    )
    allocation_receipt_hash = str(
        input_receipts.get("research_lab_allocation") or ""
    )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", allocation_receipt_hash):
        raise ChampionSettlementV2Error(
            "chain settlement allocation receipt is invalid"
        )

    section_specs = (
        ("champion_allocations", "champion", ("champion_reward_id", "source_id")),
        (
            "queued_champion_allocations",
            "queued_champion",
            ("champion_reward_id", "source_id"),
        ),
        ("source_add_allocations", "source_add", ("source_add_reward_id", "source_id")),
        ("reimbursement_allocations", "reimbursement", ("schedule_id", "source_id")),
    )
    close_hotkeys = list(observed["metagraph_hotkeys"])
    credits: list[dict[str, Any]] = []
    eligible_allocations: list[dict[str, Any]] = []
    planned_lab_by_uid: dict[int, Decimal] = defaultdict(Decimal)
    seen_obligations: set[tuple[str, str]] = set()
    observation_hash = sha256_json(observed)
    for section, kind, source_fields in section_specs:
        rows = allocation.get(section) or []
        if not isinstance(rows, list):
            raise ChampionSettlementV2Error(
                "chain settlement allocation section is invalid"
            )
        for raw_item in rows:
            if not isinstance(raw_item, Mapping):
                raise ChampionSettlementV2Error(
                    "chain settlement allocation item is invalid"
                )
            item = dict(raw_item)
            source_id = next(
                (
                    str(item.get(field) or "")
                    for field in source_fields
                    if str(item.get(field) or "")
                ),
                "",
            )
            try:
                uid = int(item.get("uid", item.get("miner_uid")))
            except (TypeError, ValueError) as exc:
                raise ChampionSettlementV2Error(
                    "chain settlement allocation UID is invalid"
                ) from exc
            hotkey = str(item.get("miner_hotkey") or "")
            paid = _non_negative_decimal_v1(
                item.get("paid_alpha_percent"),
                "paid_alpha_percent",
            )
            if paid <= 0:
                continue
            obligation_identity_kind = (
                "champion"
                if kind in {"champion", "queued_champion"}
                else kind
            )
            obligation_key = (obligation_identity_kind, source_id)
            if (
                not source_id
                or obligation_key in seen_obligations
                or uid < 0
                or uid >= len(close_hotkeys)
                or not hotkey
            ):
                raise ChampionSettlementV2Error(
                    "chain settlement allocation identity is invalid"
                )
            seen_obligations.add(obligation_key)
            if close_hotkeys[uid] != hotkey:
                continue
            planned_lab_by_uid[uid] += paid
            scheduled_value = item.get("base_desired_alpha_percent")
            if lifetime_cap_policy and kind in {
                "champion",
                "queued_champion",
            }:
                required_lifetime_fields = {
                    "total_due_alpha_percent",
                    "paid_alpha_percent_to_date",
                    "remaining_alpha_percent_before_epoch",
                    "remaining_alpha_percent_after_epoch",
                }
                if not required_lifetime_fields.issubset(item):
                    raise ChampionSettlementV2Error(
                        "chain settlement champion lifetime evidence is incomplete"
                    )
                total_due = _non_negative_decimal_v1(
                    item["total_due_alpha_percent"],
                    "total_due_alpha_percent",
                )
                paid_to_date = _non_negative_decimal_v1(
                    item["paid_alpha_percent_to_date"],
                    "paid_alpha_percent_to_date",
                )
                remaining_before = _non_negative_decimal_v1(
                    item["remaining_alpha_percent_before_epoch"],
                    "remaining_alpha_percent_before_epoch",
                )
                remaining_after = _non_negative_decimal_v1(
                    item["remaining_alpha_percent_after_epoch"],
                    "remaining_alpha_percent_after_epoch",
                )
                if (
                    paid_to_date + remaining_before != total_due
                    or paid > remaining_before
                    or remaining_after != remaining_before - paid
                ):
                    raise ChampionSettlementV2Error(
                        "chain settlement champion lifetime evidence is inconsistent"
                    )
            eligible_allocations.append(
                {
                    "section": section,
                    "kind": kind,
                    "source_id": source_id,
                    "uid": uid,
                    "hotkey": hotkey,
                    "paid": paid,
                    "scheduled": _non_negative_decimal_v1(
                        (
                            paid
                            if scheduled_value in (None, "")
                            else scheduled_value
                        ),
                        "scheduled_alpha_percent",
                    ),
                }
            )

    for uid, planned_lab in planned_lab_by_uid.items():
        planned_percent = planned_uid_weight_percent.get(uid, Decimal("0"))
        if (
            planned_percent <= 0
            or planned_lab > planned_percent + Decimal("0.000000001")
        ):
            raise ChampionSettlementV2Error(
                "chain settlement Lab attribution exceeds canonical weight"
            )

    allocations_by_uid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible_allocations:
        allocations_by_uid[int(item["uid"])].append(item)
    for uid in sorted(allocations_by_uid):
        items = sorted(
            allocations_by_uid[uid],
            key=lambda item: (str(item["kind"]), str(item["source_id"])),
        )
        planned_lab = planned_lab_by_uid[uid]
        observed_percent = observed_uid_weight_percent.get(
            uid, Decimal("0")
        )
        realized_total = min(planned_lab, observed_percent).quantize(
            _CHAIN_DECIMAL_QUANTUM_V1,
            rounding=ROUND_HALF_EVEN,
        )
        realized_values = [
            min(
                Decimal(
                    _chain_decimal_text_v1(
                        item["paid"],
                        "paid_alpha_percent",
                    )
                ),
                (
                    realized_total * Decimal(item["paid"]) / planned_lab
                ).quantize(
                    _CHAIN_DECIMAL_QUANTUM_V1,
                    rounding=ROUND_HALF_EVEN,
                ),
            )
            for item in items
        ]
        delta = realized_total - sum(realized_values)
        for ordinal, item in enumerate(items):
            if delta == 0:
                break
            if delta > 0:
                maximum = Decimal(
                    _chain_decimal_text_v1(
                        item["paid"],
                        "paid_alpha_percent",
                    )
                )
                adjustment = min(delta, maximum - realized_values[ordinal])
            else:
                adjustment = -min(-delta, realized_values[ordinal])
            realized_values[ordinal] += adjustment
            delta -= adjustment
        if delta != 0 or sum(realized_values) != realized_total:
            raise ChampionSettlementV2Error(
                "chain settlement realized allocation is inconsistent"
            )

        for item, realized in zip(items, realized_values):
            scheduled = Decimal(item["scheduled"])
            credited = (
                realized
                if lifetime_cap_policy
                and item["kind"] in {"champion", "queued_champion"}
                else min(realized, scheduled)
                if scheduled > 0
                else realized
            )
            credit_doc = {
                "schema_version": (
                    CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2
                    if lifetime_cap_policy
                    else CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1
                ),
                "netuid": int(observed["netuid"]),
                "epoch_id": int(observed["epoch_id"]),
                "obligation_kind": str(item["kind"]),
                "obligation_source_id": str(item["source_id"]),
                "miner_hotkey": str(item["hotkey"]),
                "miner_uid": uid,
                "observed_chain_alpha_percent": _chain_decimal_text_v1(
                    observed_percent,
                    "observed_chain_alpha_percent",
                ),
                "lab_attributed_alpha_percent": _chain_decimal_text_v1(
                    realized,
                    "lab_attributed_alpha_percent",
                ),
                "scheduled_alpha_percent": _chain_decimal_text_v1(
                    scheduled,
                    "scheduled_alpha_percent",
                ),
                "credited_alpha_percent": _chain_decimal_text_v1(
                    credited,
                    "credited_alpha_percent",
                ),
                "attribution_doc": {
                    "schema_version": "leadpoet.chain_realized_lab_attribution.v1",
                    "source_bundle_hash": bundle["bundle_hash"],
                    "source_bundle_epoch_id": int(bundle["epoch_id"]),
                    "source_allocation_hash": allocation_hash,
                    "source_allocation_receipt_hash": allocation_receipt_hash,
                    "allocation_section": str(item["section"]),
                },
                "observation_doc": {
                    "schema_version": "leadpoet.chain_realized_weight_observation_ref.v1",
                    "observation_hash": observation_hash,
                    "close_block": int(observed["close_block"]),
                    "close_block_hash": str(observed["close_block_hash"]),
                    "weights_vector_hash": str(observed["weights_vector_hash"]),
                },
            }
            if lifetime_cap_policy:
                credit_doc["champion_credit_policy"] = (
                    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
                )
            credits.append(
                {
                    "credit_hash": sha256_json(credit_doc),
                    "credit_doc": credit_doc,
                }
            )
    credits.sort(key=lambda item: str(item["credit_hash"]))
    settlement_doc = {
        "schema_version": (
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3
            if lifetime_cap_policy
            else CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1
        ),
        "netuid": int(observed["netuid"]),
        "epoch_id": int(observed["epoch_id"]),
        "credit_hashes": [str(item["credit_hash"]) for item in credits],
        "observation_summary": {
            "schema_version": "leadpoet.chain_realized_observation_summary.v1",
            "observation_hash": observation_hash,
            "weights_vector_hash": str(observed["weights_vector_hash"]),
            "close_block": int(observed["close_block"]),
            "close_block_hash": str(observed["close_block_hash"]),
            "official_subnet_epoch_id": int(
                observed["official_subnet_epoch_id"]
            ),
            "validator_hotkey": str(observed["validator_hotkey"]),
            "validator_uid": int(observed["validator_uid"]),
            "source_bundle_hash": bundle["bundle_hash"],
            "source_bundle_epoch_id": int(bundle["epoch_id"]),
            "source_bundle_finalized_block": int(authority["finalized_block"]),
            "source_bundle_finalized_block_hash": str(
                authority["finalized_block_hash"]
            ),
            "last_update_block": int(observed["last_update_block"]),
            "last_update_block_hash": str(
                observed["last_update_block_hash"]
            ),
            "active_source_epoch_id": int(
                observed["active_source_epoch_id"]
            ),
            "complete": True,
        },
    }
    if lifetime_cap_policy:
        settlement_doc["champion_credit_policy"] = (
            CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
        )
    return {
        "settlement_doc": settlement_doc,
        "settlement_hash": sha256_json(settlement_doc),
        "credits": credits,
    }


def build_unattributed_chain_realized_settlement_package_v2(
    *,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Record an exact chain observation without granting Lab credit.

    This is the fail-closed path for a realized primary vector that has no
    canonical finalized V2 bundle.  It advances the immutable epoch history
    while deliberately leaving every obligation unpaid.
    """

    observed = validate_chain_weight_observation_v1(observation)
    settlement_doc = {
        "schema_version": CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2,
        "netuid": int(observed["netuid"]),
        "epoch_id": int(observed["epoch_id"]),
        "credit_hashes": [],
        "observation_summary": {
            "schema_version": (
                "leadpoet.chain_realized_unattributed_observation_summary.v1"
            ),
            "authority_mode": "unattributed_chain_observation",
            "observation_hash": sha256_json(observed),
            "weights_vector_hash": str(observed["weights_vector_hash"]),
            "close_block": int(observed["close_block"]),
            "close_block_hash": str(observed["close_block_hash"]),
            "official_subnet_epoch_id": int(
                observed["official_subnet_epoch_id"]
            ),
            "validator_hotkey": str(observed["validator_hotkey"]),
            "validator_uid": int(observed["validator_uid"]),
            "last_update_block": int(observed["last_update_block"]),
            "last_update_block_hash": str(
                observed["last_update_block_hash"]
            ),
            "active_source_epoch_id": int(
                observed["active_source_epoch_id"]
            ),
            "complete": True,
        },
    }
    return {
        "settlement_doc": settlement_doc,
        "settlement_hash": sha256_json(settlement_doc),
        "credits": [],
    }


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


def _non_negative_decimal_v1(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ChampionSettlementV2Error(
            "chain realized %s is invalid" % field
        ) from exc
    if not result.is_finite() or result < 0:
        raise ChampionSettlementV2Error(
            "chain realized %s is invalid" % field
        )
    return result


def _chain_realized_receipt_root_v1(
    *,
    receipt_hash: str,
    receipt_graphs: Mapping[str, Mapping[str, Any]],
    purpose: str,
    output_root: str,
) -> str:
    graph = receipt_graphs.get(str(receipt_hash or ""))
    if not isinstance(graph, Mapping):
        raise ChampionSettlementV2Error(
            "chain realized receipt graph is missing"
        )
    validate_receipt_graph(graph)
    if graph.get("root_receipt_hash") != receipt_hash:
        raise ChampionSettlementV2Error(
            "chain realized receipt graph root differs"
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
        or root.get("purpose") != purpose
        or root.get("status") != "succeeded"
        or root.get("output_root") != output_root
    ):
        raise ChampionSettlementV2Error(
            "chain realized receipt differs"
        )
    return receipt_hash


def validate_chain_realized_epoch_settlements_v1(
    rows: Sequence[Mapping[str, Any]],
    *,
    receipt_graphs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate complete epoch-level chain-realized settlement markers.

    A per-obligation chain credit is safe only when the epoch settlement marker
    declares the complete credit-hash set for that netuid/epoch.  This prevents
    a partial observation from replacing a finalized allocation snapshot and
    accidentally under-crediting other Lab obligations in the same epoch.
    """

    settled: list[dict[str, Any]] = []
    seen_epochs: set[tuple[int, int]] = set()
    for raw_row in rows:
        row = dict(raw_row)
        document = row.get("settlement_doc")
        if not isinstance(document, Mapping):
            raise ChampionSettlementV2Error(
                "chain realized settlement document is missing"
            )
        schema_version = str(document.get("schema_version") or "")
        base_settlement_fields = {
            "schema_version",
            "netuid",
            "epoch_id",
            "credit_hashes",
            "observation_summary",
        }
        expected_settlement_fields = (
            base_settlement_fields | {"champion_credit_policy"}
            if schema_version
            == CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3
            else base_settlement_fields
        )
        if set(document) != expected_settlement_fields or schema_version not in {
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1,
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V2,
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3,
        }:
            raise ChampionSettlementV2Error(
                "chain realized settlement document is invalid"
            )
        if (
            schema_version
            == CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3
            and document.get("champion_credit_policy")
            != CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
        ):
            raise ChampionSettlementV2Error(
                "chain realized settlement champion credit policy is invalid"
            )
        try:
            netuid = int(document["netuid"])
            epoch_id = int(document["epoch_id"])
        except (TypeError, ValueError) as exc:
            raise ChampionSettlementV2Error(
                "chain realized settlement scope is invalid"
            ) from exc
        if netuid <= 0 or epoch_id < 0:
            raise ChampionSettlementV2Error(
                "chain realized settlement scope is invalid"
            )
        credit_hashes = document.get("credit_hashes")
        if (
            not isinstance(credit_hashes, list)
            or len(credit_hashes) != len(set(str(item) for item in credit_hashes))
            or any(
                not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item or ""))
                for item in credit_hashes
            )
        ):
            raise ChampionSettlementV2Error(
                "chain realized settlement credit hashes are invalid"
            )
        observation_summary = document.get("observation_summary")
        finalized_summary_fields = {
                "schema_version",
                "observation_hash",
                "weights_vector_hash",
                "close_block",
                "close_block_hash",
                "official_subnet_epoch_id",
                "validator_hotkey",
                "validator_uid",
                "source_bundle_hash",
                "source_bundle_epoch_id",
                "source_bundle_finalized_block",
                "source_bundle_finalized_block_hash",
                "last_update_block",
                "last_update_block_hash",
                "active_source_epoch_id",
                "complete",
        }
        unattributed_summary_fields = {
            "schema_version",
            "authority_mode",
            "observation_hash",
            "weights_vector_hash",
            "close_block",
            "close_block_hash",
            "official_subnet_epoch_id",
            "validator_hotkey",
            "validator_uid",
            "last_update_block",
            "last_update_block_hash",
            "active_source_epoch_id",
            "complete",
        }
        finalized_authority = schema_version in {
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V1,
            CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3,
        }
        if not isinstance(observation_summary, Mapping) or (
            finalized_authority
            and (
                set(observation_summary) != finalized_summary_fields
                or observation_summary.get("schema_version")
                != "leadpoet.chain_realized_observation_summary.v1"
            )
        ) or (
            not finalized_authority
            and (
                set(observation_summary) != unattributed_summary_fields
                or observation_summary.get("schema_version")
                != (
                    "leadpoet.chain_realized_unattributed_"
                    "observation_summary.v1"
                )
                or observation_summary.get("authority_mode")
                != "unattributed_chain_observation"
                or credit_hashes
            )
        ) or observation_summary.get("complete") is not True:
            raise ChampionSettlementV2Error(
                "chain realized settlement observation summary is invalid"
            )
        hash_fields = ["observation_hash", "weights_vector_hash"]
        if finalized_authority:
            hash_fields.append("source_bundle_hash")
        for field in hash_fields:
            if not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(observation_summary.get(field) or ""),
            ):
                raise ChampionSettlementV2Error(
                    "chain realized settlement observation summary is invalid"
                )
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            str(observation_summary.get("close_block_hash") or ""),
        ):
            raise ChampionSettlementV2Error(
                "chain realized settlement observation summary is invalid"
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            str(observation_summary.get("last_update_block_hash") or ""),
        ):
            raise ChampionSettlementV2Error(
                "chain realized settlement observation summary is invalid"
            )
        try:
            summary_epoch = int(
                observation_summary["official_subnet_epoch_id"]
            )
            active_source_epoch = int(
                observation_summary["active_source_epoch_id"]
            )
            close_block = int(observation_summary["close_block"])
            last_update_block = int(
                observation_summary["last_update_block"]
            )
            validator_uid = int(observation_summary["validator_uid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ChampionSettlementV2Error(
                "chain realized settlement observation summary is invalid"
            ) from exc
        if min(
            summary_epoch,
            active_source_epoch,
            close_block,
            last_update_block,
            validator_uid,
        ) < 0 or last_update_block > close_block or active_source_epoch > epoch_id:
            raise ChampionSettlementV2Error(
                "chain realized settlement observation summary is invalid"
            )
        if finalized_authority:
            if not re.fullmatch(
                r"[0-9a-f]{64}",
                str(
                    observation_summary.get(
                        "source_bundle_finalized_block_hash"
                    )
                    or ""
                ),
            ):
                raise ChampionSettlementV2Error(
                    "chain realized settlement observation summary is invalid"
                )
            try:
                source_epoch = int(
                    observation_summary["source_bundle_epoch_id"]
                )
                finalized_block = int(
                    observation_summary["source_bundle_finalized_block"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ChampionSettlementV2Error(
                    "chain realized settlement observation summary is invalid"
                ) from exc
            if (
                source_epoch < 0
                or finalized_block < 0
                or source_epoch != active_source_epoch
                or finalized_block != last_update_block
                or observation_summary[
                    "source_bundle_finalized_block_hash"
                ]
                != observation_summary["last_update_block_hash"]
            ):
                raise ChampionSettlementV2Error(
                    "chain realized settlement observation summary is invalid"
                )
        settlement_hash = sha256_json(dict(document))
        expected = {
            "netuid": netuid,
            "epoch_id": epoch_id,
            "schema_version": schema_version,
            "settlement_hash": settlement_hash,
            "settlement_doc": dict(document),
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ChampionSettlementV2Error(
                    "chain realized settlement row differs at %s" % field
                )
        key = (netuid, epoch_id)
        if key in seen_epochs:
            raise ChampionSettlementV2Error(
                "chain realized settlement epoch is duplicated"
            )
        seen_epochs.add(key)
        receipt_hash = str(row.get("settlement_receipt_hash") or "")
        _chain_realized_receipt_root_v1(
            receipt_hash=receipt_hash,
            receipt_graphs=receipt_graphs,
            purpose=CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1,
            output_root=settlement_hash,
        )
        settled.append(
            {
                "epoch": epoch_id,
                "netuid": netuid,
                "settlement_hash": settlement_hash,
                "settlement_doc": dict(document),
                "settlement_receipt_hash": receipt_hash,
                "credit_hashes": sorted(str(item) for item in credit_hashes),
                "authority_types": [
                    (
                        CHAIN_REALIZED_AUTHORITY_TYPE_V1
                        if finalized_authority
                        else CHAIN_REALIZED_UNATTRIBUTED_AUTHORITY_TYPE_V1
                    )
                ],
            }
        )
    return sorted(settled, key=lambda item: int(item["epoch"]))


def validate_chain_realized_obligation_credits_v1(
    credit_rows: Sequence[Mapping[str, Any]],
    *,
    settlement_rows: Sequence[Mapping[str, Any]],
    receipt_graphs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and normalize complete chain-realized obligation credits."""

    settlements = {
        (
            int(row["netuid"]),
            int(row["epoch"]),
            str(row["settlement_hash"]),
        ): dict(row)
        for row in settlement_rows
    }
    credits_by_settlement: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    seen_credit_hashes: set[str] = set()
    seen_obligations: set[tuple[int, int, str, str]] = set()
    attributed_by_uid: dict[
        tuple[tuple[int, int, str], int], Decimal
    ] = defaultdict(Decimal)
    observed_by_uid: dict[
        tuple[tuple[int, int, str], int], tuple[str, Decimal]
    ] = {}
    for raw_row in credit_rows:
        row = dict(raw_row)
        document = row.get("credit_doc")
        if not isinstance(document, Mapping):
            raise ChampionSettlementV2Error(
                "chain realized credit document is missing"
            )
        credit_schema_version = str(
            document.get("schema_version") or ""
        )
        base_credit_fields = {
            "schema_version",
            "netuid",
            "epoch_id",
            "obligation_kind",
            "obligation_source_id",
            "miner_hotkey",
            "miner_uid",
            "observed_chain_alpha_percent",
            "lab_attributed_alpha_percent",
            "scheduled_alpha_percent",
            "credited_alpha_percent",
            "attribution_doc",
            "observation_doc",
        }
        expected_credit_fields = (
            base_credit_fields | {"champion_credit_policy"}
            if credit_schema_version
            == CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2
            else base_credit_fields
        )
        if (
            set(document) != expected_credit_fields
            or credit_schema_version
            not in {
                CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1,
                CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2,
            }
        ):
            raise ChampionSettlementV2Error(
                "chain realized credit document is invalid"
            )
        champion_credit_policy = (
            str(document.get("champion_credit_policy") or "")
            if credit_schema_version
            == CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2
            else CHAIN_REALIZED_CHAMPION_CREDIT_POLICY_LEGACY_V1
        )
        if (
            credit_schema_version
            == CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2
            and champion_credit_policy
            != CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
        ):
            raise ChampionSettlementV2Error(
                "chain realized credit champion policy is invalid"
            )
        try:
            netuid = int(document["netuid"])
            epoch_id = int(document["epoch_id"])
            miner_uid = int(document["miner_uid"])
        except (TypeError, ValueError) as exc:
            raise ChampionSettlementV2Error(
                "chain realized credit scope is invalid"
            ) from exc
        if netuid <= 0 or epoch_id < 0 or miner_uid < 0:
            raise ChampionSettlementV2Error(
                "chain realized credit scope is invalid"
            )
        kind = str(document.get("obligation_kind") or "")
        if kind not in _CHAIN_CREDIT_SECTION_BY_KIND_V1:
            raise ChampionSettlementV2Error(
                "chain realized credit obligation kind is invalid"
            )
        source_id = str(document.get("obligation_source_id") or "")
        hotkey = str(document.get("miner_hotkey") or "")
        if not source_id or not hotkey:
            raise ChampionSettlementV2Error(
                "chain realized credit identity is invalid"
            )
        observed = _non_negative_decimal_v1(
            document.get("observed_chain_alpha_percent"),
            "observed_chain_alpha_percent",
        )
        attributed = _non_negative_decimal_v1(
            document.get("lab_attributed_alpha_percent"),
            "lab_attributed_alpha_percent",
        )
        scheduled = _non_negative_decimal_v1(
            document.get("scheduled_alpha_percent"),
            "scheduled_alpha_percent",
        )
        credited = _non_negative_decimal_v1(
            document.get("credited_alpha_percent"),
            "credited_alpha_percent",
        )
        if credited > attributed or attributed > observed:
            raise ChampionSettlementV2Error(
                "chain realized credit exceeds observed attribution"
            )
        lifetime_champion_credit = (
            credit_schema_version
            == CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2
            and kind in {"champion", "queued_champion"}
        )
        if lifetime_champion_credit and credited != attributed:
            raise ChampionSettlementV2Error(
                "chain realized lifetime champion credit differs from attribution"
            )
        if (
            not lifetime_champion_credit
            and scheduled > 0
            and credited > scheduled
        ):
            raise ChampionSettlementV2Error(
                "chain realized credit exceeds scheduled epoch amount"
            )
        attribution_doc = document.get("attribution_doc")
        observation_doc = document.get("observation_doc")
        if (
            not isinstance(attribution_doc, Mapping)
            or set(attribution_doc)
            != {
                "schema_version",
                "source_bundle_hash",
                "source_bundle_epoch_id",
                "source_allocation_hash",
                "source_allocation_receipt_hash",
                "allocation_section",
            }
            or attribution_doc.get("schema_version")
            != "leadpoet.chain_realized_lab_attribution.v1"
            or attribution_doc.get("allocation_section")
            != _CHAIN_CREDIT_SECTION_BY_KIND_V1[kind]
            or not isinstance(observation_doc, Mapping)
            or set(observation_doc)
            != {
                "schema_version",
                "observation_hash",
                "close_block",
                "close_block_hash",
                "weights_vector_hash",
            }
            or observation_doc.get("schema_version")
            != "leadpoet.chain_realized_weight_observation_ref.v1"
        ):
            raise ChampionSettlementV2Error(
                "chain realized credit evidence documents are invalid"
            )
        for evidence, fields in (
            (
                attribution_doc,
                (
                    "source_bundle_hash",
                    "source_allocation_hash",
                    "source_allocation_receipt_hash",
                ),
            ),
            (
                observation_doc,
                ("observation_hash", "weights_vector_hash"),
            ),
        ):
            if any(
                not re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(evidence.get(field) or ""),
                )
                for field in fields
            ):
                raise ChampionSettlementV2Error(
                    "chain realized credit evidence documents are invalid"
                )
        credit_hash = sha256_json(dict(document))
        settlement_hash = str(row.get("settlement_hash") or "")
        settlement_key = (netuid, epoch_id, settlement_hash)
        if settlement_key not in settlements:
            raise ChampionSettlementV2Error(
                "chain realized credit has no complete epoch settlement"
            )
        expected = {
            "netuid": netuid,
            "epoch_id": epoch_id,
            "schema_version": credit_schema_version,
            "obligation_kind": kind,
            "obligation_source_id": source_id,
            "miner_hotkey": hotkey,
            "miner_uid": miner_uid,
            "credit_hash": credit_hash,
            "credit_doc": dict(document),
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ChampionSettlementV2Error(
                    "chain realized credit row differs at %s" % field
                )
        row_credit_policy = row.get("champion_credit_policy")
        if (
            credit_schema_version
            == CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2
            and row_credit_policy != champion_credit_policy
        ) or (
            credit_schema_version
            == CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1
            and row_credit_policy
            not in (
                None,
                CHAIN_REALIZED_CHAMPION_CREDIT_POLICY_LEGACY_V1,
            )
        ):
            raise ChampionSettlementV2Error(
                "chain realized credit row champion policy differs"
            )
        for field, value in (
            ("observed_chain_alpha_percent", observed),
            ("lab_attributed_alpha_percent", attributed),
            ("scheduled_alpha_percent", scheduled),
            ("credited_alpha_percent", credited),
        ):
            if _non_negative_decimal_v1(row.get(field), field) != value:
                raise ChampionSettlementV2Error(
                    "chain realized credit row differs at %s" % field
                )
        if credit_hash in seen_credit_hashes:
            raise ChampionSettlementV2Error(
                "chain realized credit hash is duplicated"
            )
        seen_credit_hashes.add(credit_hash)
        obligation_identity_kind = (
            "champion"
            if kind in {"champion", "queued_champion"}
            else kind
        )
        obligation_key = (
            netuid,
            epoch_id,
            obligation_identity_kind,
            source_id,
        )
        if obligation_key in seen_obligations:
            raise ChampionSettlementV2Error(
                "chain realized obligation credit is duplicated"
            )
        seen_obligations.add(obligation_key)
        uid_key = (settlement_key, miner_uid)
        existing_observation = observed_by_uid.get(uid_key)
        if existing_observation is not None and existing_observation != (
            hotkey,
            observed,
        ):
            raise ChampionSettlementV2Error(
                "chain realized UID observation is inconsistent"
            )
        observed_by_uid[uid_key] = (hotkey, observed)
        attributed_by_uid[uid_key] += attributed
        if attributed_by_uid[uid_key] > observed:
            raise ChampionSettlementV2Error(
                "chain realized UID attribution exceeds observed weight"
            )
        receipt_hash = str(row.get("credit_receipt_hash") or "")
        _chain_realized_receipt_root_v1(
            receipt_hash=receipt_hash,
            receipt_graphs=receipt_graphs,
            purpose=CHAIN_REALIZED_SETTLEMENT_RECEIPT_PURPOSE_V1,
            output_root=settlement_hash,
        )
        credits_by_settlement[settlement_key].append(
            {
                "epoch": epoch_id,
                "netuid": netuid,
                "schema_version": credit_schema_version,
                "settlement_hash": settlement_hash,
                "credit_hash": credit_hash,
                "credit_receipt_hash": receipt_hash,
                "obligation_kind": kind,
                "obligation_source_id": source_id,
                "miner_hotkey": hotkey,
                "miner_uid": miner_uid,
                "observed_chain_alpha_percent": observed,
                "lab_attributed_alpha_percent": attributed,
                "scheduled_alpha_percent": scheduled,
                "credited_alpha_percent": credited,
                "champion_credit_policy": champion_credit_policy,
                "attribution_doc": dict(document["attribution_doc"]),
                "observation_doc": dict(document["observation_doc"]),
            }
        )

    normalized: list[dict[str, Any]] = []
    for key, settlement in sorted(settlements.items()):
        expected_hashes = set(str(item) for item in settlement["credit_hashes"])
        credits = credits_by_settlement.get(key, [])
        observed_hashes = {str(item["credit_hash"]) for item in credits}
        if observed_hashes != expected_hashes:
            raise ChampionSettlementV2Error(
                "chain realized settlement credit set is incomplete"
            )
        settlement_schema_version = str(
            settlement["settlement_doc"]["schema_version"]
        )
        authority_types = list(settlement["authority_types"])
        if authority_types not in (
            [CHAIN_REALIZED_AUTHORITY_TYPE_V1],
            [CHAIN_REALIZED_UNATTRIBUTED_AUTHORITY_TYPE_V1],
        ):
            raise ChampionSettlementV2Error(
                "chain realized settlement authority type is invalid"
            )
        authority_type = authority_types[0]
        expected_credit_schema = (
            CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V2
            if settlement_schema_version
            == CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3
            else CHAIN_REALIZED_OBLIGATION_CREDIT_SCHEMA_VERSION_V1
        )
        expected_credit_policy = (
            CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
            if settlement_schema_version
            == CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3
            else CHAIN_REALIZED_CHAMPION_CREDIT_POLICY_LEGACY_V1
        )
        if any(
            credit["schema_version"] != expected_credit_schema
            or credit["champion_credit_policy"]
            != expected_credit_policy
            for credit in credits
        ):
            raise ChampionSettlementV2Error(
                "chain realized settlement mixes champion credit policies"
            )
        allocation_doc: dict[str, Any] = {
            "schema_version": settlement_schema_version,
            "epoch": int(settlement["epoch"]),
            "netuid": int(settlement["netuid"]),
            "settlement_hash": str(settlement["settlement_hash"]),
            "authority_type": authority_type,
            "source": "chain_realized_obligation_credits",
            "source_add_allocations": [],
            "reimbursement_allocations": [],
            "champion_allocations": [],
            "queued_champion_allocations": [],
        }
        if (
            settlement_schema_version
            == CHAIN_REALIZED_EPOCH_SETTLEMENT_SCHEMA_VERSION_V3
        ):
            allocation_doc["champion_credit_policy"] = (
                CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
            )
        for credit in sorted(
            credits,
            key=lambda item: (
                str(item["obligation_kind"]),
                str(item["obligation_source_id"]),
            ),
        ):
            section = _CHAIN_CREDIT_SECTION_BY_KIND_V1[
                str(credit["obligation_kind"])
            ]
            item = {
                "uid": int(credit["miner_uid"]),
                "miner_uid": int(credit["miner_uid"]),
                "miner_hotkey": str(credit["miner_hotkey"]),
                "source_id": str(credit["obligation_source_id"]),
                "paid_alpha_percent": float(credit["credited_alpha_percent"]),
                "base_desired_alpha_percent": float(
                    credit["scheduled_alpha_percent"]
                ),
                "observed_chain_alpha_percent": float(
                    credit["observed_chain_alpha_percent"]
                ),
                "lab_attributed_alpha_percent": float(
                    credit["lab_attributed_alpha_percent"]
                ),
                "credit_hash": str(credit["credit_hash"]),
                "credit_receipt_hash": str(credit["credit_receipt_hash"]),
                "settlement_hash": str(credit["settlement_hash"]),
                "reason": CHAIN_REALIZED_AUTHORITY_TYPE_V1,
            }
            if (
                credit["champion_credit_policy"]
                == CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
            ):
                item["champion_credit_policy"] = (
                    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
                )
            if section in {"champion_allocations", "queued_champion_allocations"}:
                item["champion_reward_id"] = str(credit["obligation_source_id"])
            if section == "source_add_allocations":
                item["source_add_reward_id"] = str(credit["obligation_source_id"])
            if section == "reimbursement_allocations":
                item["schedule_id"] = str(credit["obligation_source_id"])
            allocation_doc[section].append(item)
        normalized.append(
            {
                "epoch": int(settlement["epoch"]),
                "netuid": int(settlement["netuid"]),
                "allocation_hash": str(settlement["settlement_hash"]),
                "allocation_doc": allocation_doc,
                "authority_types": authority_types,
                "chain_realized_settlement_hash": str(
                    settlement["settlement_hash"]
                ),
                "chain_realized_settlement_receipt_hash": str(
                    settlement["settlement_receipt_hash"]
                ),
                "chain_realized_credit_hashes": sorted(observed_hashes),
                "chain_realized_credit_receipt_hashes": sorted(
                    str(item["credit_receipt_hash"]) for item in credits
                ),
            }
        )
    return normalized


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


def merge_settled_allocation_histories_v2(
    finalized_rows: Sequence[Mapping[str, Any]],
    chain_realized_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer complete chain-realized epoch credits over weight intent.

    A finalized weight bundle proves what this validator submitted.  A complete
    chain-realized settlement proves what the hotkeys actually received for
    that epoch window.  When both exist, the realized settlement wins for that
    netuid/epoch and prevents double-counting.
    """

    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for raw_row in finalized_rows:
        row = dict(raw_row)
        merged[(int(row["netuid"]), int(row["epoch"]))] = row
    for raw_row in chain_realized_rows:
        row = dict(raw_row)
        key = (int(row["netuid"]), int(row["epoch"]))
        existing = merged.get(key)
        if existing is not None:
            row["replaced_authority_types"] = sorted(
                set(existing.get("authority_types") or ())
            )
            row["replaced_allocation_hash"] = str(
                existing.get("allocation_hash") or ""
            )
        merged[key] = row
    return [merged[key] for key in sorted(merged)]


def _export_validated_receipt_graph_records_v2(
    *,
    rows: Sequence[Mapping[str, Any]],
    receipt_field: str,
    loaded_graphs: Mapping[str, Mapping[str, Any]],
    output: dict[str, dict[str, Any]],
) -> None:
    """Retain validated graphs with the source epoch used by the enclave."""

    for row in rows:
        root = str(row.get(receipt_field) or "")
        if not root:
            continue
        graph = loaded_graphs.get(root)
        try:
            epoch_id = int(row["epoch_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ChampionSettlementV2Error(
                "validated settlement authority epoch is invalid"
            ) from exc
        if not isinstance(graph, Mapping):
            raise ChampionSettlementV2Error(
                "validated settlement authority graph is unavailable"
            )
        record = {
            "epoch_id": epoch_id,
            "graph": dict(graph),
        }
        existing = output.get(root)
        if existing is not None and existing != record:
            raise ChampionSettlementV2Error(
                "validated settlement authority graph conflicts"
            )
        output[root] = record


async def load_finalized_allocation_history_v2(
    *,
    netuid: int,
    start_epoch: int,
    end_epoch: int,
    _receipt_graph_records_out: dict[str, dict[str, Any]] | None = None,
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
    native = await asyncio.to_thread(
        validate_finalized_allocation_authorities_v2,
        native_rows,
        finalization_graphs=graphs,
    )
    migrated = await asyncio.to_thread(
        validate_legacy_settlement_migrations_v2,
        legacy_rows,
        receipt_graphs=migration_graphs,
    )
    if _receipt_graph_records_out is not None:
        _export_validated_receipt_graph_records_v2(
            rows=native_rows,
            receipt_field="finalization_receipt_hash",
            loaded_graphs=loaded_graphs,
            output=_receipt_graph_records_out,
        )
        _export_validated_receipt_graph_records_v2(
            rows=legacy_rows,
            receipt_field="settlement_receipt_hash",
            loaded_graphs=loaded_graphs,
            output=_receipt_graph_records_out,
        )
    return merge_finalized_allocation_histories_v2(native, migrated)


async def validate_chain_realized_settlement_bootstrap_v1(
    *,
    netuid: int,
    target_epoch: int,
    maximum_backlog: int = 100,
) -> dict[str, Any]:
    """Validate a safe pre-launch settlement state for enclave repair.

    Migration 126 creates an immutable activation row before the candidate
    enclave is launched. The old enclave cannot produce the new measured
    settlement receipts, so a pre-shutdown read may legitimately see either
    no settlement rows or a validated contiguous prefix left by an earlier
    repair attempt. Gaps, unverified rows, and rows beyond the target remain
    hard errors.
    """

    from gateway.research_lab.store import select_all, select_many

    normalized_netuid = int(netuid)
    normalized_target = int(target_epoch)
    if normalized_netuid <= 0 or int(maximum_backlog) <= 0:
        raise ChampionSettlementV2Error(
            "chain realized settlement bootstrap policy is invalid"
        )
    activation_rows = await select_many(
        CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1,
        columns=(
            "netuid,schema_version,first_epoch_id,source_bundle_hash,"
            "source_bundle_epoch_id,source_finalized_block"
        ),
        filters=(("netuid", normalized_netuid),),
        order_by=(("first_epoch_id", False),),
        limit=2,
    )
    if len(activation_rows) != 1:
        raise ChampionSettlementV2Error(
            "chain realized settlement activation is unavailable"
        )
    activation = activation_rows[0]
    try:
        activation_netuid = int(activation["netuid"])
        activation_epoch = int(activation["first_epoch_id"])
        source_epoch = int(activation["source_bundle_epoch_id"])
        source_finalized_block = int(activation["source_finalized_block"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChampionSettlementV2Error(
            "chain realized settlement activation is invalid"
        ) from exc
    source_bundle_hash = str(activation.get("source_bundle_hash") or "")
    if (
        activation.get("schema_version")
        != "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        or activation_netuid != normalized_netuid
        or activation_epoch < 0
        or source_epoch != activation_epoch
        or source_finalized_block < 0
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_bundle_hash)
    ):
        raise ChampionSettlementV2Error(
            "chain realized settlement activation is invalid"
        )
    if normalized_target < activation_epoch:
        raise ChampionSettlementV2Error(
            "chain realized settlement bootstrap target predates activation"
        )
    existing_rows = await select_all(
        CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1,
        columns="netuid,epoch_id,settlement_hash",
        filters=(("netuid", normalized_netuid),),
        order_by=(("epoch_id", False),),
        max_rows=10000,
        allow_partial=False,
    )
    try:
        existing_epochs = [int(row["epoch_id"]) for row in existing_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ChampionSettlementV2Error(
            "chain realized settlement history is invalid"
        ) from exc
    if len(existing_epochs) != len(set(existing_epochs)):
        raise ChampionSettlementV2Error(
            "chain realized settlement history is incomplete"
        )
    settled_through = (
        max(existing_epochs) if existing_epochs else activation_epoch - 1
    )
    if settled_through > normalized_target:
        raise ChampionSettlementV2Error(
            "chain realized settlement history is ahead of target"
        )
    expected_existing_epochs = list(
        range(activation_epoch, settled_through + 1)
    )
    if sorted(existing_epochs) != expected_existing_epochs:
        raise ChampionSettlementV2Error(
            "chain realized settlement history is incomplete"
        )
    pending_epoch_count = normalized_target - settled_through
    if pending_epoch_count > int(maximum_backlog):
        raise ChampionSettlementV2Error(
            "chain-realized settlement backlog exceeds policy"
        )
    if existing_epochs:
        await load_chain_realized_allocation_history_v1(
            netuid=normalized_netuid,
            start_epoch=activation_epoch,
            end_epoch=settled_through,
        )

    source_rows = await select_many(
        FINALIZED_ALLOCATION_VIEW_V2,
        columns=(
            "bundle_hash,netuid,epoch_id,finalized_block,"
            "finalization_receipt_hash"
        ),
        filters=(
            ("netuid", normalized_netuid),
            ("epoch_id", source_epoch),
            ("bundle_hash", source_bundle_hash),
            ("finalized_block", source_finalized_block),
        ),
        order_by=(("bundle_hash", False),),
        limit=2,
    )
    if len(source_rows) != 1:
        raise ChampionSettlementV2Error(
            "chain realized settlement activation source is unavailable"
        )

    finalized = await load_finalized_allocation_history_v2(
        netuid=normalized_netuid,
        start_epoch=activation_epoch,
        end_epoch=normalized_target,
    )
    source_authorities = [
        row
        for row in finalized
        if int(row.get("epoch") or -1) == source_epoch
        and source_bundle_hash
        in set(row.get("finalized_bundle_hashes") or ())
    ]
    if len(source_authorities) != 1:
        raise ChampionSettlementV2Error(
            "chain realized settlement activation source is not authoritative"
        )
    return {
        "schema_version": (
            "leadpoet.chain_realized_settlement_bootstrap_readiness.v1"
        ),
        "status": (
            "resumable_bootstrap_pending"
            if existing_epochs
            else "pristine_bootstrap_pending"
        ),
        "netuid": normalized_netuid,
        "activation_epoch": activation_epoch,
        "target_epoch": normalized_target,
        "settled_through_epoch": (
            settled_through if existing_epochs else None
        ),
        "backlog_epoch_count": pending_epoch_count,
        "source_bundle_hash": source_bundle_hash,
        "source_finalized_block": source_finalized_block,
        "validated_chain_realized_epochs": sorted(existing_epochs),
        "validated_finalized_candidate_epochs": sorted(
            int(row["epoch"]) for row in finalized
        ),
    }


async def load_chain_realized_allocation_history_v1(
    *,
    netuid: int,
    start_epoch: int,
    end_epoch: int,
    _receipt_graph_records_out: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load complete realized-chain Lab settlement epochs."""

    if int(end_epoch) < int(start_epoch):
        return []
    from gateway.research_lab.attested_v2_store import load_receipt_graphs_v2
    from gateway.research_lab.store import select_all, select_many

    activation_rows = await select_many(
        CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1,
        columns=(
            "netuid,schema_version,first_epoch_id,source_bundle_hash,"
            "source_bundle_epoch_id,source_finalized_block"
        ),
        filters=(("netuid", int(netuid)),),
        order_by=(("first_epoch_id", False),),
        limit=1,
    )
    if len(activation_rows) != 1:
        raise ChampionSettlementV2Error(
            "chain realized settlement activation is unavailable"
        )

    settlement_rows = await select_all(
        CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1,
        filters=(
            ("netuid", int(netuid)),
            ("epoch_id", "gte", int(start_epoch)),
            ("epoch_id", "lte", int(end_epoch)),
        ),
        order_by=(("epoch_id", False),),
        max_rows=max(1000, int(end_epoch) - int(start_epoch) + 1),
        allow_partial=False,
    )
    activation = activation_rows[0]
    try:
        activation_netuid = int(activation["netuid"])
        activation_epoch = int(activation["first_epoch_id"])
        source_epoch = int(activation["source_bundle_epoch_id"])
        source_finalized_block = int(activation["source_finalized_block"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChampionSettlementV2Error(
            "chain realized settlement activation is invalid"
        ) from exc
    if (
        activation.get("schema_version")
        != "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        or activation_netuid != int(netuid)
        or activation_epoch < 0
        or source_epoch != activation_epoch
        or source_finalized_block < 0
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(activation.get("source_bundle_hash") or ""),
        )
    ):
        raise ChampionSettlementV2Error(
            "chain realized settlement activation is invalid"
        )
    expected_epochs = set(
        range(
            max(int(start_epoch), activation_epoch),
            int(end_epoch) + 1,
        )
    )
    observed_epochs = {int(row["epoch_id"]) for row in settlement_rows}
    if observed_epochs != expected_epochs:
        raise ChampionSettlementV2Error(
            "chain realized settlement history is incomplete"
        )
    if not settlement_rows:
        return []
    settlement_roots = {
        str(row.get("settlement_receipt_hash") or "")
        for row in settlement_rows
        if row.get("settlement_receipt_hash")
    }
    settlement_graphs_loaded = await load_receipt_graphs_v2(settlement_roots)
    settlement_graphs = {
        root: settlement_graphs_loaded[root]
        for root in settlement_roots
        if root in settlement_graphs_loaded
    }
    settlements = await asyncio.to_thread(
        validate_chain_realized_epoch_settlements_v1,
        settlement_rows,
        receipt_graphs=settlement_graphs,
    )
    credit_rows = await select_all(
        CHAIN_REALIZED_OBLIGATION_CREDIT_TABLE_V1,
        filters=(
            ("netuid", int(netuid)),
            ("epoch_id", "gte", int(start_epoch)),
            ("epoch_id", "lte", int(end_epoch)),
        ),
        order_by=(("epoch_id", False), ("obligation_kind", False), ("obligation_source_id", False)),
        max_rows=max(1000, (int(end_epoch) - int(start_epoch) + 1) * 100),
        allow_partial=False,
    )
    credit_roots = {
        str(row.get("credit_receipt_hash") or "")
        for row in credit_rows
        if row.get("credit_receipt_hash")
    }
    credit_graphs_loaded = await load_receipt_graphs_v2(credit_roots)
    credit_graphs = {
        root: credit_graphs_loaded[root]
        for root in credit_roots
        if root in credit_graphs_loaded
    }
    credits = await asyncio.to_thread(
        validate_chain_realized_obligation_credits_v1,
        credit_rows,
        settlement_rows=settlements,
        receipt_graphs=credit_graphs,
    )
    if _receipt_graph_records_out is not None:
        _export_validated_receipt_graph_records_v2(
            rows=settlement_rows,
            receipt_field="settlement_receipt_hash",
            loaded_graphs=settlement_graphs,
            output=_receipt_graph_records_out,
        )
        _export_validated_receipt_graph_records_v2(
            rows=credit_rows,
            receipt_field="credit_receipt_hash",
            loaded_graphs=credit_graphs,
            output=_receipt_graph_records_out,
        )
    return credits


async def load_settled_allocation_history_v2(
    *,
    netuid: int,
    start_epoch: int,
    end_epoch: int,
    _receipt_graph_records_out: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load settlement history, preferring actual chain-realized credits."""

    if int(end_epoch) < int(start_epoch):
        return []
    # Validate the small contiguous chain history first. If it is incomplete,
    # do not launch the substantially larger finalized-allocation graph scan.
    if _receipt_graph_records_out is not None:
        _receipt_graph_records_out.clear()
    history_kwargs = {
        "netuid": int(netuid),
        "start_epoch": int(start_epoch),
        "end_epoch": int(end_epoch),
    }
    chain_kwargs = dict(history_kwargs)
    finalized_kwargs = dict(history_kwargs)
    if _receipt_graph_records_out is not None:
        chain_kwargs["_receipt_graph_records_out"] = (
            _receipt_graph_records_out
        )
        finalized_kwargs["_receipt_graph_records_out"] = (
            _receipt_graph_records_out
        )
    chain_realized = await load_chain_realized_allocation_history_v1(
        **chain_kwargs
    )
    finalized = await load_finalized_allocation_history_v2(
        **finalized_kwargs
    )
    return merge_settled_allocation_histories_v2(finalized, chain_realized)


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
    return await asyncio.to_thread(
        validate_legacy_allocation_nonfinalizations_v2,
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
    _authority_graph_records_out: dict[str, dict[str, Any]] | None = None,
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
    authority_graph_records: dict[str, dict[str, Any]] = {}
    history_kwargs = {
        "netuid": int(netuid),
        "start_epoch": min(starts) if starts else 0,
        "end_epoch": int(epoch) - 1,
    }
    if _authority_graph_records_out is not None:
        history_kwargs["_receipt_graph_records_out"] = (
            authority_graph_records
        )
    finalized = (
        await load_settled_allocation_history_v2(**history_kwargs)
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
    if _authority_graph_records_out is not None:
        _authority_graph_records_out.clear()
        _authority_graph_records_out.update(authority_graph_records)
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
    chain_realized_authority_types = {
        CHAIN_REALIZED_AUTHORITY_TYPE_V1,
        CHAIN_REALIZED_UNATTRIBUTED_AUTHORITY_TYPE_V1,
    }
    chain_realized_epochs = {
        epoch_id
        for epoch_id, item in finalized_by_epoch.items()
        if chain_realized_authority_types
        & set(item.get("authority_types") or ())
    }
    nonfinalized_by_epoch = {
        int(item["epoch"]): item for item in nonfinalized
    }
    conflicting_classification_epochs = sorted(
        (
            set(finalized_by_epoch)
            & set(nonfinalized_by_epoch)
        )
        - chain_realized_epochs
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
        if settlement_epoch in chain_realized_epochs:
            # The normalized chain-realized row is already validated against
            # its complete coordinator receipt and credit set. It deliberately
            # uses the settlement hash rather than the legacy allocation hash,
            # so it must not be reclassified by the pre-activation snapshot
            # path below.
            continue
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
        (
            set(current_payment_allocations)
            | set(anchors_by_epoch)
        )
        - chain_realized_epochs
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
