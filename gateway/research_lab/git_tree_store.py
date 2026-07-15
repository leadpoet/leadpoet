"""Append-only persistence adapter for authoritative autoresearch trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from gateway.research_lab import store
from gateway.research_lab.git_tree_models import (
    TreeChildSlot,
    TreeNode,
    generation_operation_id,
)
from leadpoet_canonical.attested_v2 import sha256_json


ZERO_HASH = "sha256:" + "0" * 64


class GitTreeStoreError(RuntimeError):
    """The database rejected a tree state transition."""


@dataclass
class GitTreeStore:
    """Small RPC client; all concurrency decisions remain in PostgreSQL."""

    async def get_tree_current(self, *, tree_id: str) -> Mapping[str, Any] | None:
        return await store.select_one(
            "research_lab_autoresearch_tree_current",
            columns=(
                "tree_id,run_id,root_artifact_hash,root_manifest_hash,"
                "root_source_tree_hash,root_git_commit,root_image_digest,"
                "policy_hash,evaluator_commitment_hash,current_event_type,"
                "current_event_hash,current_round_index,current_frontier_hash,"
                "current_frontier_doc"
            ),
            filters=(("tree_id", tree_id),),
        )

    async def list_operations(self, *, tree_id: str) -> list[Mapping[str, Any]]:
        rows = await store.select_all(
            "research_lab_autoresearch_operation_current",
            columns=(
                "logical_operation_id,tree_id,node_id,operation_kind,"
                "operation_status,request_hash,result_hash,"
                "settled_cost_microusd,provider_call_count,settlement_doc"
            ),
            filters=(("tree_id", tree_id),),
            order_by=(("logical_operation_id", False),),
            batch_size=100,
            max_rows=1000,
        )
        return [dict(row) for row in rows]

    async def get_latest_recovery_event(
        self, *, tree_id: str
    ) -> Mapping[str, Any] | None:
        rows = await store.select_many(
            "research_lab_autoresearch_tree_events",
            columns="tree_id,seq,event_type,event_doc,event_hash,created_at",
            filters=(
                ("tree_id", tree_id),
                (
                    "event_type",
                    "in",
                    ["node_generated", "checkpoint_committed"],
                ),
            ),
            order_by=(("seq", True),),
            limit=1,
        )
        return dict(rows[0]) if rows else None

    async def get_node_event(
        self, *, tree_id: str, node_id: str, event_type: str
    ) -> Mapping[str, Any] | None:
        return await store.select_one(
            "research_lab_autoresearch_tree_events",
            columns="tree_id,seq,event_type,node_id,event_doc,event_hash,created_at",
            filters=(
                ("tree_id", tree_id),
                ("node_id", node_id),
                ("event_type", event_type),
            ),
        )

    async def create_tree(
        self,
        *,
        tree_id: str,
        run_id: str,
        root_artifact_hash: str,
        root_manifest_hash: str,
        root_source_tree_hash: str,
        root_git_commit: str,
        root_image_digest: str,
        policy_hash: str,
        evaluator_commitment_hash: str,
        tree_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        identity = {
            "schema_version": "research_lab.git_tree.v1",
            "tree_id": tree_id,
            "run_id": str(run_id),
            "root_artifact_hash": root_artifact_hash,
            "root_manifest_hash": root_manifest_hash,
            "root_source_tree_hash": root_source_tree_hash,
            "root_git_commit": root_git_commit,
            "root_image_digest": root_image_digest,
            "policy_hash": policy_hash,
            "evaluator_commitment_hash": evaluator_commitment_hash,
            "tree_doc": dict(tree_doc),
        }
        data = await store.call_rpc(
            "create_research_lab_autoresearch_tree",
            {
                "requested_tree_id": tree_id,
                "requested_run_id": str(run_id),
                "requested_root_artifact_hash": root_artifact_hash,
                "requested_root_manifest_hash": root_manifest_hash,
                "requested_root_source_tree_hash": root_source_tree_hash,
                "requested_root_git_commit": root_git_commit,
                "requested_root_image_digest": root_image_digest,
                "requested_policy_hash": policy_hash,
                "requested_evaluator_commitment_hash": evaluator_commitment_hash,
                "requested_tree_doc": dict(tree_doc),
                "requested_identity_hash": sha256_json(identity),
            },
        )
        return _result_envelope(data, "tree create", "tree")

    async def plan_node(
        self,
        *,
        slot: TreeChildSlot,
        request_hash: str,
        node_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        operation_id = generation_operation_id(slot)
        identity = {
            "schema_version": "research_lab.git_tree_node_identity.v1",
            **slot.to_dict(),
            "generation_operation_id": operation_id,
            "generation_request_hash": request_hash,
            "node_doc": dict(node_doc),
        }
        reservation = _operation_transition_doc(
            logical_operation_id=operation_id,
            tree_id=slot.tree_id,
            node_id=slot.node_id,
            operation_kind="generation",
            operation_status="reserved",
            request_hash=request_hash,
        )
        data = await store.call_rpc(
            "plan_research_lab_autoresearch_tree_node",
            {
                "requested_tree_id": slot.tree_id,
                "requested_node_id": slot.node_id,
                "requested_parent_node_id": slot.parent_node_id,
                "requested_root_branch_id": slot.root_branch_id,
                "requested_depth": slot.depth,
                "requested_child_ordinal": slot.slot_index,
                "requested_generation_operation_id": operation_id,
                "requested_generation_request_hash": request_hash,
                "requested_generation_transition_hash": sha256_json(reservation),
                "requested_node_doc": dict(node_doc),
                "requested_identity_hash": sha256_json(identity),
            },
        )
        return _plan_envelope(data)

    async def transition_operation(
        self,
        *,
        logical_operation_id: str,
        tree_id: str,
        node_id: str,
        operation_kind: str,
        operation_status: str,
        request_hash: str,
        result_hash: str = "",
        settled_cost_microusd: int = 0,
        provider_call_count: int = 0,
        settlement_doc: Mapping[str, Any] | None = None,
        expected_current_status: str | None = None,
    ) -> Mapping[str, Any]:
        transition = _operation_transition_doc(
            logical_operation_id=logical_operation_id,
            tree_id=tree_id,
            node_id=node_id,
            operation_kind=operation_kind,
            operation_status=operation_status,
            request_hash=request_hash,
            result_hash=result_hash,
            settled_cost_microusd=settled_cost_microusd,
            provider_call_count=provider_call_count,
            settlement_doc=settlement_doc,
        )
        data = await store.call_rpc(
            "transition_research_lab_autoresearch_operation",
            {
                "requested_logical_operation_id": logical_operation_id,
                "requested_tree_id": tree_id,
                "requested_node_id": node_id,
                "requested_operation_kind": operation_kind,
                "requested_operation_status": operation_status,
                "requested_request_hash": request_hash,
                "requested_result_hash": result_hash,
                "requested_settled_cost_microusd": transition[
                    "settled_cost_microusd"
                ],
                "requested_provider_call_count": transition[
                    "provider_call_count"
                ],
                "requested_settlement_doc": transition["settlement_doc"],
                "requested_transition_hash": sha256_json(transition),
                "expected_current_status": expected_current_status,
            },
        )
        return _result_envelope(data, "operation transition", "operation")

    async def get_operation(
        self, *, logical_operation_id: str
    ) -> Mapping[str, Any]:
        row = await store.select_one(
            "research_lab_autoresearch_operation_current",
            columns=(
                "logical_operation_id,tree_id,node_id,operation_kind,"
                "operation_status,request_hash,result_hash,"
                "settled_cost_microusd,provider_call_count,settlement_doc"
            ),
            filters=(("logical_operation_id", logical_operation_id),),
        )
        if row is None:
            return {
                "exists": False,
                "logical_operation_id": str(logical_operation_id),
            }
        return {
            "exists": True,
            "logical_operation_id": str(logical_operation_id),
            "operation": dict(row),
        }

    async def append_event(
        self,
        *,
        tree_id: str,
        event_type: str,
        event_doc: Mapping[str, Any],
        previous_event_hash: str,
        node_id: str = "",
    ) -> Mapping[str, Any]:
        body = {
            "schema_version": "research_lab.git_tree_event.v1",
            "tree_id": tree_id,
            "event_type": event_type,
            "node_id": node_id,
            "previous_event_hash": previous_event_hash,
            "event_doc": dict(event_doc),
        }
        data = await store.call_rpc(
            "append_research_lab_autoresearch_tree_event",
            {
                "requested_tree_id": tree_id,
                "requested_event_type": event_type,
                "requested_node_id": node_id,
                "requested_previous_event_hash": previous_event_hash,
                "requested_event_doc": dict(event_doc),
                "requested_event_hash": sha256_json(body),
            },
        )
        return _one(data, "tree event")

    async def append_event_next(
        self,
        *,
        tree_id: str,
        event_type: str,
        event_doc: Mapping[str, Any],
        node_id: str = "",
        attempts: int = 5,
    ) -> Mapping[str, Any]:
        last_error: BaseException | None = None
        for _attempt in range(max(1, int(attempts))):
            current = await store.select_one(
                "research_lab_autoresearch_tree_current",
                columns="current_event_hash",
                filters=(("tree_id", tree_id),),
            )
            previous = str((current or {}).get("current_event_hash") or ZERO_HASH)
            try:
                return await self.append_event(
                    tree_id=tree_id,
                    event_type=event_type,
                    event_doc=event_doc,
                    previous_event_hash=previous,
                    node_id=node_id,
                )
            except Exception as exc:  # PostgreSQL CAS arbitrates concurrent writers.
                last_error = exc
                message = str(exc).lower()
                if "40001" not in message and "event_conflict" not in message:
                    raise
        assert last_error is not None
        raise last_error

    async def commit_frontier(
        self,
        *,
        tree_id: str,
        round_index: int,
        expected_previous_hash: str,
        frontier_hash: str,
        frontier_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        body = {
            "schema_version": "research_lab.git_tree_frontier.v1",
            "tree_id": tree_id,
            "round_index": int(round_index),
            "expected_previous_hash": expected_previous_hash,
            "frontier_hash": frontier_hash,
            "frontier_doc": dict(frontier_doc),
        }
        data = await store.call_rpc(
            "commit_research_lab_autoresearch_frontier",
            {
                "requested_tree_id": tree_id,
                "requested_round_index": int(round_index),
                "requested_expected_previous_hash": expected_previous_hash,
                "requested_frontier_hash": frontier_hash,
                "requested_frontier_doc": dict(frontier_doc),
                "requested_commitment_hash": sha256_json(body),
            },
        )
        return _one(data, "frontier commit")

    async def commit_frontier_next(
        self,
        *,
        tree_id: str,
        frontier_hash: str,
        frontier_doc: Mapping[str, Any],
        attempts: int = 5,
    ) -> Mapping[str, Any]:
        last_error: BaseException | None = None
        for _attempt in range(max(1, int(attempts))):
            current = await store.select_one(
                "research_lab_autoresearch_tree_current",
                columns="current_round_index,current_frontier_hash,current_frontier_doc",
                filters=(("tree_id", tree_id),),
            )
            if str((current or {}).get("current_frontier_hash") or "") == str(
                frontier_hash
            ):
                existing = await store.select_one(
                    "research_lab_autoresearch_frontier_commitments",
                    columns=(
                        "tree_id,round_index,schema_version,expected_previous_hash,"
                        "frontier_hash,frontier_doc,commitment_hash,created_at"
                    ),
                    filters=(
                        ("tree_id", tree_id),
                        ("frontier_hash", frontier_hash),
                    ),
                )
                if not isinstance(existing, Mapping):
                    raise GitTreeStoreError(
                        "frontier commitment is missing after idempotent retry"
                    )
                if dict(existing.get("frontier_doc") or {}) != dict(frontier_doc):
                    raise GitTreeStoreError(
                        "frontier commitment changed after it was recorded"
                    )
                return dict(existing)
            raw_round = (current or {}).get("current_round_index")
            round_index = (int(raw_round) if raw_round is not None else -1) + 1
            previous = str(
                (current or {}).get("current_frontier_hash") or ZERO_HASH
            )
            try:
                return await self.commit_frontier(
                    tree_id=tree_id,
                    round_index=round_index,
                    expected_previous_hash=previous,
                    frontier_hash=frontier_hash,
                    frontier_doc=frontier_doc,
                )
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "40001" not in message and "frontier_conflict" not in message:
                    raise
        assert last_error is not None
        raise last_error

    async def select_final(
        self,
        *,
        tree_id: str,
        node_id: str,
        selection_hash: str,
        selection_doc: Mapping[str, Any],
        attempts: int = 5,
    ) -> Mapping[str, Any]:
        last_error: BaseException | None = None
        for _attempt in range(max(1, int(attempts))):
            current = await store.select_one(
                "research_lab_autoresearch_tree_current",
                columns="current_event_hash",
                filters=(("tree_id", tree_id),),
            )
            previous = str((current or {}).get("current_event_hash") or ZERO_HASH)
            event_doc = {
                "selection_hash": selection_hash,
                "selection": dict(selection_doc),
            }
            event_hash = sha256_json(
                {
                    "schema_version": "research_lab.git_tree_event.v1",
                    "tree_id": tree_id,
                    "event_type": "final_selected",
                    "node_id": node_id,
                    "previous_event_hash": previous,
                    "event_doc": event_doc,
                }
            )
            try:
                data = await store.call_rpc(
                    "select_research_lab_autoresearch_tree_final",
                    {
                        "requested_tree_id": tree_id,
                        "requested_node_id": node_id,
                        "requested_selection_hash": selection_hash,
                        "requested_selection_doc": dict(selection_doc),
                        "requested_previous_event_hash": previous,
                        "requested_event_hash": event_hash,
                    },
                )
                return _result_envelope(data, "final selection", "event")
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "40001" not in message and "event_conflict" not in message:
                    raise
        assert last_error is not None
        raise last_error


def tree_node_event_doc(node: TreeNode) -> dict[str, Any]:
    """Bounded node projection; private feedback/source remain in artifacts."""
    evaluation = node.evaluation.to_dict() if node.evaluation is not None else None
    if isinstance(evaluation, dict):
        evaluation.pop("feedback", None)
    return {**node.to_dict(), "evaluation": evaluation}


def _one(value: Any, operation: str) -> Mapping[str, Any]:
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, Mapping):
        raise GitTreeStoreError(f"{operation} returned no row")
    return dict(value)


def _operation_transition_doc(
    *,
    logical_operation_id: str,
    tree_id: str,
    node_id: str,
    operation_kind: str,
    operation_status: str,
    request_hash: str,
    result_hash: str = "",
    settled_cost_microusd: int = 0,
    provider_call_count: int = 0,
    settlement_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "research_lab.git_tree_operation.v1",
        "logical_operation_id": logical_operation_id,
        "tree_id": tree_id,
        "node_id": node_id,
        "operation_kind": operation_kind,
        "operation_status": operation_status,
        "request_hash": request_hash,
        "result_hash": result_hash,
        "settled_cost_microusd": max(0, int(settled_cost_microusd)),
        "provider_call_count": max(0, int(provider_call_count)),
        "settlement_doc": dict(settlement_doc or {}),
    }


def _plan_envelope(value: Any) -> Mapping[str, Any]:
    result = dict(_one(value, "node plan"))
    if set(result) != {"created", "node", "operation"}:
        raise GitTreeStoreError("node plan returned an invalid envelope")
    if not isinstance(result.get("node"), Mapping) or not isinstance(
        result.get("operation"), Mapping
    ):
        raise GitTreeStoreError("node plan returned invalid rows")
    return {
        "created": bool(result["created"]),
        "node": dict(result["node"]),
        "operation": dict(result["operation"]),
    }


def _result_envelope(value: Any, operation: str, row_key: str) -> Mapping[str, Any]:
    result = dict(_one(value, operation))
    # Unit-test and pre-migration adapters may return the raw RPC parameters;
    # production returns the explicit created + row envelope.
    if "created" not in result and row_key not in result:
        return result
    if set(result) != {"created", row_key} or not isinstance(
        result.get(row_key), Mapping
    ):
        raise GitTreeStoreError(f"{operation} returned an invalid envelope")
    return {"created": bool(result["created"]), row_key: dict(result[row_key])}
