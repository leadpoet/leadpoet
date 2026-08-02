"""Measured conversion of one attested allocation state into a bounded frontier."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

from gateway.tee.coordinator_allocation_source_v2 import (
    _receipt_graphs_by_declared_root,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.reward_executor_v2 import (
    champion_reward_row_projection_v2,
    source_add_reward_row_projection_v2,
)
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION,
    build_allocation_settlement_frontier_bootstrap_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    ALPHA_QUANT,
    build_allocation_settlement_frontier_v2,
    build_reward_settlement_checkpoint_v2,
)
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    merkle_root,
    sha256_json,
    validate_signed_execution_receipt,
)


_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_SOURCE_ROWS = 100


class CoordinatorAllocationFrontierBootstrapV2Error(RuntimeError):
    """The first settlement frontier cannot be derived fail closed."""


def _same(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise CoordinatorAllocationFrontierBootstrapV2Error(
            "%s must be an integer" % field
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CoordinatorAllocationFrontierBootstrapV2Error(
            "%s must be an integer" % field
        ) from exc
    if result < minimum:
        raise CoordinatorAllocationFrontierBootstrapV2Error(
            "%s is outside the allowed range" % field
        )
    return result


def _alpha(value: Any, field: str) -> Decimal:
    try:
        amount = Decimal(str(value)).quantize(
            ALPHA_QUANT,
            rounding=ROUND_HALF_UP,
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise CoordinatorAllocationFrontierBootstrapV2Error(
            "%s is invalid" % field
        ) from exc
    if not amount.is_finite() or amount < 0:
        raise CoordinatorAllocationFrontierBootstrapV2Error(
            "%s is invalid" % field
        )
    return amount


def select_latest_allocation_source_row_v2(
    rows: Sequence[Mapping[str, Any]],
    *,
    through_epoch: int,
) -> Dict[str, Any]:
    """Select one deterministic latest source, rejecting ambiguity/truncation."""

    normalized_through = _integer(
        through_epoch,
        "allocation source through_epoch",
        minimum=1,
    )
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or not rows
        or len(rows) > _MAX_SOURCE_ROWS
    ):
        raise CoordinatorAllocationFrontierBootstrapV2Error(
            "latest allocation execution source is unavailable or truncated"
        )
    normalized: list[Dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "latest allocation execution source is invalid"
            )
        row = dict(raw)
        epoch = _integer(row.get("epoch_id"), "allocation source epoch")
        receipt_hash = str(row.get("receipt_hash") or "").lower()
        result = row.get("result_doc")
        artifacts = row.get("artifact_hashes")
        if (
            epoch > normalized_through
            or not _HASH_RE.fullmatch(receipt_hash)
            or row.get("receipt_hash") != receipt_hash
            or row.get("schema_version")
            != "leadpoet.attested_execution_result.v2"
            or row.get("role") != "gateway_coordinator"
            or row.get("operation") != "research_lab_allocation"
            or row.get("purpose") != "research_lab.allocation.v2"
            or not isinstance(result, Mapping)
            or not isinstance(artifacts, list)
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "latest allocation execution source scope is invalid"
            )
        source_state = result.get("source_state")
        source_state_hash = str(result.get("source_state_hash") or "").lower()
        allocation = result.get("allocation")
        normalized_artifacts = [str(item or "").lower() for item in artifacts]
        if (
            not isinstance(source_state, Mapping)
            or not isinstance(allocation, Mapping)
            or not _HASH_RE.fullmatch(source_state_hash)
            or result.get("source_state_hash") != source_state_hash
            or sha256_json(dict(source_state)) != source_state_hash
            or int(source_state.get("epoch", -1)) != epoch
            or int(source_state.get("netuid", -1)) <= 0
            or row.get("result_hash") != sha256_json(dict(result))
            or normalized_artifacts != sorted(set(normalized_artifacts))
            or any(not _HASH_RE.fullmatch(item) for item in normalized_artifacts)
            or source_state_hash not in normalized_artifacts
            or row.get("artifact_root")
            != merkle_root(normalized_artifacts, domain="leadpoet-artifact-v2")
            or row.get("output_root")
            != sha256_json({"allocation": dict(allocation)})
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "latest allocation execution source differs from its artifacts"
            )
        normalized.append(row)
    expected_order = sorted(
        normalized,
        key=lambda row: (-int(row["epoch_id"]), str(row["receipt_hash"])),
    )
    if not _same(normalized, expected_order):
        raise CoordinatorAllocationFrontierBootstrapV2Error(
            "latest allocation execution sources are not canonically ordered"
        )
    latest_epoch = int(normalized[0]["epoch_id"])
    candidates = [
        row for row in normalized if int(row["epoch_id"]) == latest_epoch
    ]
    if len(normalized) == _MAX_SOURCE_ROWS and len(candidates) == len(normalized):
        raise CoordinatorAllocationFrontierBootstrapV2Error(
            "latest allocation execution source is truncated"
        )
    canonical_state = candidates[0]["result_doc"]
    if any(
        not _same(candidate["result_doc"], canonical_state)
        for candidate in candidates[1:]
    ):
        raise CoordinatorAllocationFrontierBootstrapV2Error(
            "latest allocation execution source is ambiguous"
        )
    return dict(candidates[0])


class CoordinatorAllocationFrontierBootstrapV2:
    """Derive a first frontier from the latest measured allocation state."""

    def __init__(self, reader: SupabaseSourceReaderV2) -> None:
        self._reader = reader

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "netuid",
            "through_epoch",
            "allocation_source_receipt_hash",
        }:
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation frontier bootstrap request fields are invalid"
            )
        if (
            payload.get("schema_version")
            != ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation frontier bootstrap request schema is unsupported"
            )
        netuid = _integer(payload.get("netuid"), "bootstrap netuid", minimum=1)
        through_epoch = _integer(
            payload.get("through_epoch"),
            "bootstrap through_epoch",
            minimum=1,
        )
        if through_epoch != int(context.epoch_id):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation frontier bootstrap epoch differs from execution scope"
            )
        requested_receipt = str(
            payload.get("allocation_source_receipt_hash") or ""
        ).lower()
        if (
            not _HASH_RE.fullmatch(requested_receipt)
            or payload.get("allocation_source_receipt_hash") != requested_receipt
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation frontier bootstrap source receipt is invalid"
            )
        if self._read(
            "allocation_settlement_frontier_activation",
            {"netuid": netuid},
            context,
        ) or self._read(
            "allocation_settlement_frontiers",
            {"netuid": netuid, "before_epoch": through_epoch + 1},
            context,
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation settlement frontier is already initialized"
            )

        source_rows = self._read(
            "latest_attested_allocation_execution_results",
            {"through_epoch": through_epoch},
            context,
        )
        source_row = select_latest_allocation_source_row_v2(
            source_rows,
            through_epoch=through_epoch,
        )
        if source_row["receipt_hash"] != requested_receipt:
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "requested allocation source is not the latest authority"
            )
        result = dict(source_row["result_doc"])
        source_state = dict(result["source_state"])
        source_epoch = int(source_row["epoch_id"])
        if (
            int(source_state.get("netuid", -1)) != netuid
            or source_state.get("settlement_frontier") is not None
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation source state is not eligible for bootstrap"
            )
        self._validate_source_receipt(
            source_row=source_row,
            context=context,
        )

        policy = source_state.get("policy")
        if not isinstance(policy, Mapping):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation source policy is unavailable"
            )
        enable_champ_cap = policy.get("enable_champ_cap", True)
        if not isinstance(enable_champ_cap, bool):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation source champion policy is invalid"
            )
        champion_state = self._obligation_index(
            source_state.get("champion_obligations"),
            count=source_state.get("champion_obligation_count"),
            id_fields=("source_id", "champion_reward_id"),
            label="champion",
        )
        source_add_state = self._obligation_index(
            source_state.get("source_add_obligations", []),
            count=source_state.get("source_add_obligation_count", 0),
            id_fields=("source_id", "source_add_reward_id"),
            label="source add",
        )
        champion_rows = [
            self._read_exact_reward(
                policy_id="champion_reward_by_id",
                parameters={"champion_reward_id": source_id},
                source_id=source_id,
                identity_field="champion_reward_id",
                label="champion",
                context=context,
            )
            for source_id in sorted(champion_state)
        ]
        source_add_rows = [
            self._read_exact_reward(
                policy_id="source_add_reward_by_ref",
                parameters={"reward_ref": source_id},
                source_id=source_id,
                identity_field="reward_ref",
                label="source add",
                context=context,
            )
            for source_id in sorted(source_add_state)
        ]
        required_parents: Set[str] = {requested_receipt}
        checkpoints = self._reward_checkpoints(
            source_state=source_state,
            source_epoch=source_epoch,
            champion_rows=champion_rows,
            source_add_rows=source_add_rows,
            enable_champ_cap=enable_champ_cap,
            champion_state=champion_state,
            source_add_state=source_add_state,
            context=context,
            required_parents=required_parents,
        )
        if required_parents != set(context.parent_receipt_hashes):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation frontier bootstrap parent set differs"
            )
        frontier = build_allocation_settlement_frontier_v2(
            mode="legacy_full_history_bootstrap",
            netuid=netuid,
            allocation_epoch=source_epoch,
            predecessor_frontier_hash=None,
            reward_checkpoints=checkpoints,
        )
        return build_allocation_settlement_frontier_bootstrap_v2(
            netuid=netuid,
            bootstrap_epoch=through_epoch,
            allocation_source_receipt_hash=requested_receipt,
            source_state_hash=str(result["source_state_hash"]),
            frontier=frontier,
        )

    def _validate_source_receipt(
        self,
        *,
        source_row: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> None:
        receipt_hash = str(source_row["receipt_hash"])
        receipt_rows = self._read(
            "attested_receipt_by_hash",
            {"receipt_hash": receipt_hash},
            context,
        )
        if len(receipt_rows) != 1 or not isinstance(
            receipt_rows[0].get("receipt_doc"), Mapping
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation source receipt is unavailable"
            )
        receipt = dict(receipt_rows[0]["receipt_doc"])
        validate_signed_execution_receipt(receipt)
        if (
            receipt.get("receipt_hash") != receipt_hash
            or receipt.get("role") != source_row.get("role")
            or receipt.get("purpose") != source_row.get("purpose")
            or receipt.get("job_id") != source_row.get("job_id")
            or receipt.get("epoch_id") != source_row.get("epoch_id")
            or receipt.get("sequence") != source_row.get("sequence")
            or receipt.get("input_root") != source_row.get("input_root")
            or receipt.get("output_root") != source_row.get("output_root")
            or receipt.get("artifact_root") != source_row.get("artifact_root")
            or receipt.get("status") != "succeeded"
            or receipt_hash not in context.parent_receipt_hashes
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation source receipt differs"
            )
        graphs = _receipt_graphs_by_declared_root(
            context.external_receipt_graphs,
            context.parent_receipt_hashes,
        )
        if receipt_hash not in graphs:
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation source receipt graph is unavailable"
            )

    def _reward_checkpoints(
        self,
        *,
        source_state: Mapping[str, Any],
        source_epoch: int,
        champion_rows: Sequence[Mapping[str, Any]],
        source_add_rows: Sequence[Mapping[str, Any]],
        enable_champ_cap: bool,
        champion_state: Mapping[str, Mapping[str, Any]],
        source_add_state: Mapping[str, Mapping[str, Any]],
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> list[Dict[str, Any]]:
        skipped = source_state.get("skipped")
        if not isinstance(skipped, Mapping):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "allocation source skipped state is unavailable"
            )
        for field in ("champions", "source_add"):
            values = skipped.get(field, [])
            if not isinstance(values, list) or values:
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "allocation source contains an unrepresentable skipped reward"
                )
        checkpoints: list[Dict[str, Any]] = []
        seen_champions: set[str] = set()
        for raw_row in champion_rows:
            row = dict(raw_row)
            source_id = str(row.get("champion_reward_id") or "")
            if not source_id or source_id in seen_champions:
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "champion bootstrap source is duplicated"
                )
            seen_champions.add(source_id)
            projection = champion_reward_row_projection_v2(row)
            self._require_reward_receipt(
                artifact_kind="champion_reward_decision",
                artifact_ref=source_id,
                expected_output_root=sha256_json(projection),
                context=context,
                required_parents=required_parents,
            )
            checkpoints.append(
                self._checkpoint_from_source_state(
                    reward_kind="champion",
                    source_id=source_id,
                    row=row,
                    obligation=champion_state.get(source_id),
                    obligation_hash=sha256_json(projection),
                    source_epoch=source_epoch,
                    champ_cap_enabled=enable_champ_cap,
                )
            )
        if set(champion_state).difference(seen_champions):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "champion source state is absent from measured rewards"
            )

        seen_source_add: set[str] = set()
        for raw_row in source_add_rows:
            row = dict(raw_row)
            source_id = str(row.get("reward_ref") or "")
            if not source_id or source_id in seen_source_add:
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "source-add bootstrap source is duplicated"
                )
            seen_source_add.add(source_id)
            projection = source_add_reward_row_projection_v2(
                "source_add_leg%d" % int(row.get("leg") or 0),
                {**row, "initial_reward_status": "active"},
            )
            self._require_reward_receipt(
                artifact_kind="source_add_reward_decision",
                artifact_ref=source_id,
                expected_output_root=sha256_json(projection),
                context=context,
                required_parents=required_parents,
            )
            checkpoints.append(
                self._checkpoint_from_source_state(
                    reward_kind="source_add",
                    source_id=source_id,
                    row={
                        **row,
                        "start_epoch": row.get("start_epoch"),
                        "epoch_count": (
                            row.get("epoch_count") or row.get("reward_epochs")
                        ),
                        "desired_alpha_percent": (
                            row.get("desired_alpha_percent")
                            or row.get("alpha_percent")
                        ),
                    },
                    obligation=source_add_state.get(source_id),
                    obligation_hash=sha256_json(projection),
                    source_epoch=source_epoch,
                    champ_cap_enabled=True,
                )
            )
        if set(source_add_state).difference(seen_source_add):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "source-add source state is absent from measured rewards"
            )
        return checkpoints

    def _read_exact_reward(
        self,
        *,
        policy_id: str,
        parameters: Mapping[str, Any],
        source_id: str,
        identity_field: str,
        label: str,
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        rows = self._read(policy_id, parameters, context)
        if (
            len(rows) != 1
            or not isinstance(rows[0], Mapping)
            or str(rows[0].get(identity_field) or "") != source_id
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "%s immutable reward source is missing or ambiguous" % label
            )
        return dict(rows[0])

    @staticmethod
    def _obligation_index(
        raw: Any,
        *,
        count: Any,
        id_fields: Tuple[str, str],
        label: str,
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw, list) or _integer(
            count, "%s obligation count" % label
        ) != len(raw):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "%s obligation state is invalid" % label
            )
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "%s obligation state is invalid" % label
                )
            identities = {str(item.get(field) or "") for field in id_fields}
            if len(identities) != 1 or "" in identities:
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "%s obligation identity differs" % label
                )
            source_id = next(iter(identities))
            if source_id in indexed:
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "%s obligation is duplicated" % label
                )
            indexed[source_id] = dict(item)
        return indexed

    @staticmethod
    def _checkpoint_from_source_state(
        *,
        reward_kind: str,
        source_id: str,
        row: Mapping[str, Any],
        obligation: Any,
        obligation_hash: str,
        source_epoch: int,
        champ_cap_enabled: bool,
    ) -> Dict[str, Any]:
        start_epoch = _integer(row.get("start_epoch"), "reward start_epoch")
        epoch_count = _integer(
            row.get("epoch_count"),
            "reward epoch_count",
            minimum=1,
        )
        desired = _alpha(
            row.get("desired_alpha_percent"),
            "reward desired alpha",
        )
        total_due = (desired * Decimal(epoch_count)).quantize(
            ALPHA_QUANT,
            rounding=ROUND_HALF_UP,
        )
        if desired <= 0 or int(source_epoch) < start_epoch:
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "reward is outside the attested allocation source"
            )
        if obligation is None:
            nominal_active = int(source_epoch) < start_epoch + epoch_count
            if not champ_cap_enabled and nominal_active:
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "uncapped nominal reward is missing from source state"
                )
            paid = total_due
        else:
            if not isinstance(obligation, Mapping):
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "reward obligation state is invalid"
                )
            state_cap = obligation.get("champ_cap_enabled")
            if state_cap is not champ_cap_enabled:
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "reward obligation cap policy differs"
                )
            paid = _alpha(
                obligation.get("paid_alpha_percent_to_date"),
                "reward paid alpha",
            )
            remaining = _alpha(
                obligation.get("remaining_alpha_percent"),
                "reward remaining alpha",
            )
            current_desired = _alpha(
                obligation.get("current_epoch_desired_alpha_percent"),
                "reward current desired alpha",
            )
            expected_current = min(desired, remaining) if champ_cap_enabled else desired
            expected_replay = (
                "extended_replay"
                if int(source_epoch) >= start_epoch + epoch_count
                else "nominal_window"
            )
            if (
                _integer(obligation.get("start_epoch"), "state start epoch")
                != start_epoch
                or _integer(obligation.get("epoch_count"), "state epoch count")
                != epoch_count
                or _integer(
                    obligation.get("nominal_end_epoch"),
                    "state nominal end epoch",
                )
                != start_epoch + epoch_count
                or _alpha(
                    obligation.get("desired_alpha_percent"),
                    "state desired alpha",
                )
                != desired
                or _alpha(
                    obligation.get("total_due_alpha_percent"),
                    "state total due alpha",
                )
                != total_due
                or paid > total_due
                or remaining != total_due - paid
                or current_desired != expected_current
                or obligation.get("replay_status") != expected_replay
                or str(obligation.get("miner_hotkey") or "")
                != str(row.get("miner_hotkey") or "")
            ):
                raise CoordinatorAllocationFrontierBootstrapV2Error(
                    "reward obligation balance differs from immutable source"
                )
        return build_reward_settlement_checkpoint_v2(
            reward_kind=reward_kind,
            source_id=source_id,
            obligation_hash=obligation_hash,
            start_epoch=start_epoch,
            epoch_count=epoch_count,
            desired_alpha_percent=desired,
            applied_alpha_percent=paid,
            realized_alpha_percent=paid,
            excess_alpha_percent=Decimal("0"),
        )

    def _require_reward_receipt(
        self,
        *,
        artifact_kind: str,
        artifact_ref: str,
        expected_output_root: str,
        context: ExecutionContextV2,
        required_parents: Set[str],
    ) -> None:
        links = self._read(
            "attested_business_artifact_by_ref",
            {
                "artifact_kind": artifact_kind,
                "artifact_ref": artifact_ref,
                "artifact_hash": expected_output_root,
            },
            context,
        )
        if len(links) != 1:
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "%s receipt link is missing or ambiguous" % artifact_kind
            )
        receipt_hash = str(links[0].get("receipt_hash") or "")
        rows = self._read(
            "attested_receipt_by_hash",
            {"receipt_hash": receipt_hash},
            context,
        )
        if len(rows) != 1 or not isinstance(rows[0].get("receipt_doc"), Mapping):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "%s receipt is unavailable" % artifact_kind
            )
        receipt = dict(rows[0]["receipt_doc"])
        validate_signed_execution_receipt(receipt)
        if (
            receipt.get("receipt_hash") != receipt_hash
            or receipt.get("role") != "gateway_coordinator"
            or receipt.get("purpose") != "research_lab.reward_decision.v2"
            or receipt.get("output_root") != expected_output_root
            or links[0].get("artifact_hash") != expected_output_root
            or receipt_hash not in context.parent_receipt_hashes
        ):
            raise CoordinatorAllocationFrontierBootstrapV2Error(
                "%s receipt does not bind its decision" % artifact_kind
            )
        required_parents.add(receipt_hash)

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
