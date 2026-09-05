"""Bound active legacy receipt ancestry before stopping the old gateway.

This operator path is intentionally read-only until the measured coordinator
has validated and signed an exact legacy graph.  It never builds an allocation
or changes scoring inputs: it selects the same already-durable allocation and
sourcing authorities consumed by the weight path, asks the enclave to issue
one bounded recursive checkpoint per selected active root, and atomically
persists those checkpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

from gateway.research_lab.attested_coordinator_v2 import execute_coordinator_v2
from gateway.research_lab.attested_scoring_v2 import (
    DEFAULT_RELEASE_MANIFEST_PATH,
)
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.tee.coordinator_executor_v2 import (
    OP_ANCESTRY_CHECKPOINT_BOOTSTRAP_V2,
)
from gateway.tee.release_lineage_v2 import (
    build_release_lineage_boot_verifier_v2,
    load_approved_release_lineage_v2,
)
from gateway.tee.release_manifest_v2 import (
    prior_role_expectation,
    validate_prior_release_manifest,
)
from gateway.utils.tee_client import coordinator_tee_client
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION,
    build_checkpointed_receipt_graph_from_full_graph_v2,
    derive_ancestry_lineage_id_v2,
    select_ancestry_checkpoint_resume_frontier_v2,
    validate_ancestry_checkpoint_bootstrap_result_v2,
)
from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS,
    RECEIPT_GRAPH_SCHEMA_VERSION,
    canonical_json,
)
from Leadpoet.utils.subnet_epoch import load_subnet_epoch_cutover


logger = logging.getLogger(__name__)

BOOTSTRAP_PURPOSE_V2 = "research_lab.ancestry_checkpoint_bootstrap.v2"
DEFAULT_MAX_STABILITY_ROUNDS = 3
_COORDINATOR_ROLE = "gateway_coordinator"
_ALLOWED_ISSUER_ROLES = (
    "gateway_autoresearch",
    "gateway_coordinator",
    "gateway_scoring",
    "validator_weights",
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ActiveAncestryCheckpointBootstrapV2Error(RuntimeError):
    """Active ancestry could not be checkpointed without changing authority."""


class ActiveAncestryCheckpointBootstrapV2Unsupported(
    ActiveAncestryCheckpointBootstrapV2Error
):
    """The otherwise valid running coordinator predates the measured operation."""


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _lineage_id() -> str:
    cutover = load_subnet_epoch_cutover()
    return derive_ancestry_lineage_id_v2(
        cutover_mapping_hash=str(cutover.mapping_hash),
        network_genesis_hash=str(cutover.network_genesis_hash),
        netuid=int(cutover.netuid),
    )


def _load_release_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "gateway release manifest is unavailable"
        ) from exc
    return validate_prior_release_manifest(value)


class _LazyApprovedReleaseBootVerifier:
    """Verify each observed boot against its exact approved release on demand."""

    def __init__(
        self,
        *,
        current_release: Mapping[str, Any],
        release_channel_loader: Any = None,
        lineage_loader: Callable[..., Mapping[str, Mapping[str, Any]]] = (
            load_approved_release_lineage_v2
        ),
        verifier_builder: Callable[[Mapping[str, Mapping[str, Any]]], Any] = (
            build_release_lineage_boot_verifier_v2
        ),
    ) -> None:
        self._current_release = validate_prior_release_manifest(current_release)
        self._release_channel_loader = release_channel_loader
        self._lineage_loader = lineage_loader
        self._verifier_builder = verifier_builder
        self._releases: dict[str, Mapping[str, Any]] = {
            str(self._current_release["commit_sha"]): self._current_release
        }

    def __call__(self, identity: Mapping[str, Any]) -> Mapping[str, Any]:
        commit = str(identity.get("commit_sha") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ActiveAncestryCheckpointBootstrapV2Error(
                "receipt ancestry boot commit is invalid"
            )
        existing = self._releases.get(commit)
        needs_validator_release = identity.get(
            "physical_role"
        ) == "validator_weights" and (
            not isinstance(existing, Mapping)
            or "validator_release_manifest" not in existing
        )
        if commit not in self._releases or needs_validator_release:
            graph = {
                "boot_identities": [dict(identity)],
            }
            kwargs = {
                "current_release": self._current_release,
                "parent_graphs": (graph,),
            }
            if self._release_channel_loader is not None:
                kwargs["release_channel_loader"] = self._release_channel_loader
            loaded = self._lineage_loader(**kwargs)
            if not isinstance(loaded, Mapping) or commit not in loaded:
                raise ActiveAncestryCheckpointBootstrapV2Error(
                    "receipt ancestry release lineage is incomplete"
                )
            self._releases.update(
                {str(key).lower(): value for key, value in loaded.items()}
            )
        verifier = self._verifier_builder(self._releases)
        verified = verifier(identity)
        if not isinstance(verified, Mapping):
            raise ActiveAncestryCheckpointBootstrapV2Error(
                "receipt ancestry boot verifier returned no identity"
            )
        return verified


async def _verify_coordinator_capability(
    *,
    client: Any,
    release: Mapping[str, Any],
    boot_verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ancestry_lineage_id: str,
) -> dict[str, Any]:
    health = await client.coordinator_v2_health()
    boot_identity = await client.v2_get_boot_identity()
    if not isinstance(health, Mapping) or not isinstance(boot_identity, Mapping):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "coordinator health or boot identity is unavailable"
        )
    worker_count = health.get("worker_count")
    supported = health.get("supported_operations")
    if (
        health.get("authority") != "v2_only"
        or health.get("role") != _COORDINATOR_ROLE
        or health.get("physical_role") != _COORDINATOR_ROLE
        or type(worker_count) is not int
        or not 1 <= worker_count <= 10
        or health.get("configured_worker_count") != 0
        or not health.get("workers_alive")
        or health.get("ancestry_checkpoints") is not True
        or health.get("ancestry_lineage_id") != ancestry_lineage_id
        or health.get("boot_identity_hash") != boot_identity.get("boot_identity_hash")
        or not isinstance(supported, list)
        or any(not isinstance(item, str) for item in supported)
        or supported != sorted(set(supported))
    ):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "coordinator V2 health is invalid"
        )
    expectation = prior_role_expectation(release, _COORDINATOR_ROLE)
    boot_verifier(boot_identity)
    for field in (
        "commit_sha",
        "pcr0",
        "build_manifest_hash",
        "dependency_lock_hash",
    ):
        if boot_identity.get(field) != expectation[field]:
            raise ActiveAncestryCheckpointBootstrapV2Error(
                "coordinator boot %s differs from the running release" % field
            )
    if OP_ANCESTRY_CHECKPOINT_BOOTSTRAP_V2 not in supported:
        raise ActiveAncestryCheckpointBootstrapV2Unsupported(
            "running coordinator does not advertise ancestry checkpoint bootstrap"
        )
    return dict(health)


def _graph_root(graph: Mapping[str, Any]) -> str:
    root = str(graph.get("root_receipt_hash") or "").lower()
    if not _HASH_RE.fullmatch(root):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "active receipt graph root is invalid"
        )
    if graph.get("schema_version") not in {
        RECEIPT_GRAPH_SCHEMA_VERSION,
        *CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS,
    }:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "active receipt graph schema is unsupported"
        )
    if graph.get("root_receipt_hash") != root:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "active receipt graph root is not canonical"
        )
    return root


async def _load_frontier_bounded_allocation_graphs(
    *,
    epoch_id: int,
    netuid: int,
    policy: Mapping[str, Any],
    load_frontier_context: Any = None,
    load_parent_graphs: Any = None,
    load_graphs: Any = None,
) -> list[dict[str, Any]]:
    """Select the exact allocation ancestry without replaying settled history."""

    if load_frontier_context is None or load_graphs is None:
        from gateway.research_lab.attested_v2_store import (
            load_allocation_settlement_frontier_context_v2,
            load_receipt_graphs_v2,
        )

        load_frontier_context = (
            load_frontier_context or load_allocation_settlement_frontier_context_v2
        )
        load_graphs = load_graphs or load_receipt_graphs_v2
    if load_parent_graphs is None:
        from gateway.research_lab.v2_authority import (
            _load_allocation_parent_graphs_v2,
        )

        load_parent_graphs = _load_allocation_parent_graphs_v2

    context = await _maybe_await(
        load_frontier_context(
            netuid=int(netuid),
            before_epoch=int(epoch_id) + 1,
        )
    )
    if context is None:
        return list(
            await _maybe_await(
                load_parent_graphs(
                    epoch_id=int(epoch_id),
                    netuid=int(netuid),
                    policy=dict(policy),
                )
            )
        )
    if not isinstance(context, Mapping):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "allocation settlement frontier context is invalid"
        )
    frontier = context.get("frontier")
    try:
        frontier_epoch = int(frontier["allocation_epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "allocation settlement frontier epoch is invalid"
        ) from exc
    if frontier_epoch < 0 or frontier_epoch > int(epoch_id):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "allocation settlement frontier is outside the active epoch"
        )
    if frontier_epoch < int(epoch_id):
        return list(
            await _maybe_await(
                load_parent_graphs(
                    epoch_id=int(epoch_id),
                    netuid=int(netuid),
                    policy=dict(policy),
                    settlement_frontier_context=context,
                )
            )
        )

    source = context.get("source")
    source_receipt = source.get("receipt") if isinstance(source, Mapping) else None
    parent_roots = (
        source_receipt.get("parent_receipt_hashes")
        if isinstance(source_receipt, Mapping)
        else None
    )
    if not isinstance(parent_roots, list):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "current allocation frontier parents are invalid"
        )
    if any(
        not isinstance(root, str)
        or root != root.lower()
        or not _HASH_RE.fullmatch(root)
        for root in parent_roots
    ) or len(set(parent_roots)) != len(parent_roots):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "current allocation frontier parent set is invalid"
        )
    normalized_roots = sorted(parent_roots)
    loaded = await _maybe_await(load_graphs(normalized_roots))
    if not isinstance(loaded, Mapping) or set(loaded) != set(normalized_roots):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "current allocation frontier parent graphs are incomplete"
        )
    selected = []
    for root in normalized_roots:
        graph = loaded[root]
        if not isinstance(graph, Mapping) or _graph_root(graph) != root:
            raise ActiveAncestryCheckpointBootstrapV2Error(
                "current allocation frontier parent graph differs"
            )
        selected.append(dict(graph))
    return selected


async def _select_active_graphs(
    *,
    epoch_id: int,
    netuid: int,
    policy: Mapping[str, Any],
    load_allocation_graphs: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]],
    load_sourcing_graphs: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]],
    load_source_add_graphs: (
        Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] | None
    ) = None,
) -> dict[str, dict[str, Any]]:
    loaders = [
        load_allocation_graphs(
            epoch_id=int(epoch_id),
            netuid=int(netuid),
            policy=dict(policy),
        ),
        load_sourcing_graphs(current_epoch=int(epoch_id), window=30),
    ]
    if load_source_add_graphs is not None:
        loaders.append(load_source_add_graphs(current_epoch=int(epoch_id)))
    graph_sets = await asyncio.gather(*loaders)
    selected: dict[str, dict[str, Any]] = {}
    for raw_graph in [graph for graphs in graph_sets for graph in graphs]:
        if not isinstance(raw_graph, Mapping):
            raise ActiveAncestryCheckpointBootstrapV2Error(
                "active receipt graph is not an object"
            )
        graph = dict(raw_graph)
        root = _graph_root(graph)
        previous = selected.get(root)
        if previous is not None and previous != graph:
            raise ActiveAncestryCheckpointBootstrapV2Error(
                "active receipt graph conflicts for one immutable root"
            )
        selected[root] = graph
    return {root: selected[root] for root in sorted(selected)}


def _canonical_size(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def _marker(
    *,
    started_at: float,
    stage: str,
    round_number: int,
    **values: Any,
) -> None:
    fields = " ".join("%s=%s" % (key, values[key]) for key in sorted(values))
    logger.info(
        "active_ancestry_checkpoint_bootstrap_v2 "
        "stage=%s round=%s elapsed_seconds=%.3f%s",
        stage,
        round_number,
        time.monotonic() - started_at,
        (" " + fields) if fields else "",
    )


async def _bootstrap_one_graph(
    *,
    graph: Mapping[str, Any],
    epoch_id: int,
    release: Mapping[str, Any],
    ancestry_lineage_id: str,
    boot_verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    client: Any,
    execute: Callable[..., Awaitable[Mapping[str, Any]]],
    load_proofs: Callable[..., Awaitable[Mapping[str, Mapping[str, Any]]]],
    load_checkpointed_graphs: Callable[..., Awaitable[Mapping[str, Mapping[str, Any]]]],
    persist_checkpoint: Callable[..., Awaitable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Issue and durably verify exactly one selected-root checkpoint."""

    proof_load_started = time.monotonic()
    root = _graph_root(graph)
    if graph.get("schema_version") != RECEIPT_GRAPH_SCHEMA_VERSION:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "checkpoint bootstrap requires a legacy full graph"
        )
    root_receipts = [
        receipt
        for receipt in graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
        and str(receipt.get("receipt_hash") or "").lower() == root
    ]
    if len(root_receipts) != 1:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "legacy full graph selected root receipt is unavailable"
        )
    direct_parent_roots = sorted(
        {
            str(parent_root).lower()
            for parent_root in root_receipts[0].get("parent_receipt_hashes") or ()
        }
    )
    if any(not _HASH_RE.fullmatch(item) for item in direct_parent_roots):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "legacy full graph selected root parent set is invalid"
        )
    # A recursive checkpoint needs only an already-durable selected root
    # (race/idempotency short-circuit) and already-durable direct parents.
    # Querying every historical receipt would make a partially checkpointed
    # graph quadratic because each proof is checked against the full graph.
    proof_query_roots = sorted({root, *direct_parent_roots})
    durable = await _maybe_await(
        load_proofs(
            proof_query_roots,
            expected_lineage_id=ancestry_lineage_id,
            boot_attestation_verifier=boot_verifier,
            allowed_issuer_roles=_ALLOWED_ISSUER_ROLES,
        )
    )
    if not isinstance(durable, Mapping) or any(
        key not in proof_query_roots or not isinstance(value, Mapping)
        for key, value in durable.items()
    ):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "durable checkpoint proof selection is invalid"
        )
    if root in durable:
        bounded = await _maybe_await(load_checkpointed_graphs((root,)))
        if (
            not isinstance(bounded, Mapping)
            or set(bounded) != {root}
            or bounded[root].get("ancestry_proof") != durable[root]
        ):
            raise ActiveAncestryCheckpointBootstrapV2Error(
                "durable active root checkpoint graph is unavailable"
            )
        return {
            "new_proof_count": 0,
            "resume_proof_count": 0,
            "proof_load_seconds": round(time.monotonic() - proof_load_started, 3),
            "execution_seconds": 0.0,
            "persistence_seconds": 0.0,
            "readback_seconds": 0.0,
        }

    durable_values = [
        dict(durable[item]) for item in direct_parent_roots if item in durable
    ]
    frontier = select_ancestry_checkpoint_resume_frontier_v2(
        full_graphs=(graph,),
        selected_root_receipt_hashes=(root,),
        durable_compact_proofs=durable_values,
        allowed_failed_receipt_hashes_by_graph=((),),
        expected_lineage_id=ancestry_lineage_id,
        boot_attestation_verifier=boot_verifier,
        allowed_issuer_roles=_ALLOWED_ISSUER_ROLES,
    )
    request = {
        "schema_version": ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION,
        "selected_root_receipt_hashes": [root],
    }
    proof_load_seconds = time.monotonic() - proof_load_started
    execution_started = time.monotonic()
    outcome = await execute(
        operation=OP_ANCESTRY_CHECKPOINT_BOOTSTRAP_V2,
        purpose=BOOTSTRAP_PURPOSE_V2,
        epoch_id=int(epoch_id),
        sequence=0,
        payload=request,
        parent_graphs=(graph,),
        parent_ancestry_proofs=tuple(frontier),
        require_egress_proxy=False,
        release_manifest=release,
        client=client,
        boot_verifier=boot_verifier,
    )
    execution_seconds = time.monotonic() - execution_started
    if not isinstance(outcome, Mapping) or not isinstance(
        outcome.get("result"), Mapping
    ):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "measured checkpoint bootstrap returned no result"
        )
    result = validate_ancestry_checkpoint_bootstrap_result_v2(
        outcome["result"],
        expected_selected_root_receipt_hashes=(root,),
        existing_compact_proofs=frontier,
        expected_lineage_id=ancestry_lineage_id,
        boot_attestation_verifier=boot_verifier,
        allowed_issuer_roles=_ALLOWED_ISSUER_ROLES,
    )
    if (
        len(result["checkpoint_proofs"]) != 1
        or result["checkpoint_proofs"][0]["certificate"]["claim"][
            "output_root_receipt_hash"
        ]
        != root
    ):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "measured checkpoint bootstrap did not return one selected-root proof"
        )
    expected_proofs = {
        str(proof["certificate"]["claim"]["output_root_receipt_hash"]): dict(proof)
        for proof in frontier
    }
    persistence_started = time.monotonic()
    for proof in result["checkpoint_proofs"]:
        proof_root = str(proof["certificate"]["claim"]["output_root_receipt_hash"])
        checkpointed_graph = build_checkpointed_receipt_graph_from_full_graph_v2(
            graph,
            proof,
            expected_lineage_id=ancestry_lineage_id,
            boot_attestation_verifier=boot_verifier,
            allowed_issuer_roles=_ALLOWED_ISSUER_ROLES,
        )
        persistence = await _maybe_await(
            persist_checkpoint(
                proof,
                checkpointed_graph=checkpointed_graph,
                expected_lineage_id=ancestry_lineage_id,
                boot_attestation_verifier=boot_verifier,
                allowed_issuer_roles=_ALLOWED_ISSUER_ROLES,
            )
        )
        if (
            not isinstance(persistence, Mapping)
            or persistence.get("root_receipt_hash") != proof_root
            or persistence.get("proof_hash") != proof.get("proof_hash")
            or persistence.get("root_activated") is not True
        ):
            raise ActiveAncestryCheckpointBootstrapV2Error(
                "ancestry checkpoint persistence acknowledgment differs"
            )
        expected_proofs[proof_root] = dict(proof)
    persistence_seconds = time.monotonic() - persistence_started

    expected_roots = set(result["checkpoint_root_receipt_hashes"])
    if set(expected_proofs) != expected_roots:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "checkpoint bootstrap result proof set differs"
        )
    readback_started = time.monotonic()
    durable_readback = await _maybe_await(
        load_proofs(
            expected_roots,
            expected_lineage_id=ancestry_lineage_id,
            boot_attestation_verifier=boot_verifier,
            allowed_issuer_roles=_ALLOWED_ISSUER_ROLES,
        )
    )
    if (
        not isinstance(durable_readback, Mapping)
        or set(durable_readback) != (expected_roots)
        or any(
            durable_readback[item] != expected_proofs[item] for item in expected_roots
        )
    ):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "checkpoint proof durable readback differs"
        )
    bounded_readback = await _maybe_await(load_checkpointed_graphs((root,)))
    if (
        not isinstance(bounded_readback, Mapping)
        or set(bounded_readback) != {root}
        or bounded_readback[root].get("root_receipt_hash") != root
        or bounded_readback[root].get("ancestry_proof") != durable_readback[root]
    ):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "checkpoint root graph durable readback differs"
        )
    return {
        "new_proof_count": len(result["checkpoint_proofs"]),
        "resume_proof_count": len(frontier),
        "proof_load_seconds": round(proof_load_seconds, 3),
        "execution_seconds": round(execution_seconds, 3),
        "persistence_seconds": round(persistence_seconds, 3),
        "readback_seconds": round(time.monotonic() - readback_started, 3),
    }


async def bootstrap_active_ancestry_checkpoints_v2(
    *,
    epoch_id: int | None = None,
    netuid: int | None = None,
    release_manifest: Mapping[str, Any] | None = None,
    release_manifest_path: Path = DEFAULT_RELEASE_MANIFEST_PATH,
    max_stability_rounds: int = DEFAULT_MAX_STABILITY_ROUNDS,
    client: Any = coordinator_tee_client,
    execute: Any = execute_coordinator_v2,
    load_allocation_graphs: Any = None,
    load_sourcing_graphs: Any = None,
    load_proofs: Any = None,
    load_checkpointed_graphs: Any = None,
    persist_checkpoint: Any = None,
    resolve_epoch: Any = None,
    boot_verifier: Any = None,
    release_channel_loader: Any = None,
    ensure_allocation_frontier: Any = None,
) -> dict[str, Any]:
    """Checkpoint all active full roots and prove the selection stayed stable."""

    if (
        not isinstance(max_stability_rounds, int)
        or isinstance(max_stability_rounds, bool)
        or not 1 <= max_stability_rounds <= 10
    ):
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "checkpoint stability round limit is invalid"
        )
    if resolve_epoch is None:
        from gateway.research_lab.maintenance import _resolve_maintenance_epoch

        resolve_epoch = _resolve_maintenance_epoch
    if load_allocation_graphs is None:
        load_allocation_graphs = _load_frontier_bounded_allocation_graphs
    if load_sourcing_graphs is None:
        from gateway.research_lab.attested_v2_store import (
            load_sourcing_epoch_graphs_v2,
        )

        load_sourcing_graphs = load_sourcing_epoch_graphs_v2
    if (
        load_proofs is None
        or load_checkpointed_graphs is None
        or (persist_checkpoint is None)
    ):
        from gateway.research_lab.attested_v2_store import (
            load_ancestry_checkpoint_proofs_v2,
            load_checkpointed_receipt_graphs_v2,
            persist_ancestry_checkpoint_v2,
        )

        load_proofs = load_proofs or load_ancestry_checkpoint_proofs_v2
        load_checkpointed_graphs = (
            load_checkpointed_graphs or load_checkpointed_receipt_graphs_v2
        )
        persist_checkpoint = persist_checkpoint or persist_ancestry_checkpoint_v2

    effective_epoch = int(
        epoch_id if epoch_id is not None else await resolve_epoch(None)
    )
    if effective_epoch < 0:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "active checkpoint epoch is invalid"
        )
    if netuid is None:
        from gateway.config import BITTENSOR_NETUID

        effective_netuid = int(BITTENSOR_NETUID)
    else:
        effective_netuid = int(netuid)
    if effective_netuid <= 0:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "active checkpoint netuid is invalid"
        )
    release = (
        validate_prior_release_manifest(release_manifest)
        if (release_manifest is not None)
        else _load_release_manifest(release_manifest_path)
    )
    ancestry_lineage_id = _lineage_id()
    verifier = boot_verifier or _LazyApprovedReleaseBootVerifier(
        current_release=release,
        release_channel_loader=release_channel_loader,
    )
    started_at = time.monotonic()
    health = await _verify_coordinator_capability(
        client=client,
        release=release,
        boot_verifier=verifier,
        ancestry_lineage_id=ancestry_lineage_id,
    )
    _marker(
        started_at=started_at,
        stage="coordinator_verified",
        round_number=0,
        commit_sha=release["commit_sha"],
        operation=OP_ANCESTRY_CHECKPOINT_BOOTSTRAP_V2,
    )
    if ensure_allocation_frontier is None:
        from gateway.tee.bootstrap_allocation_settlement_frontier_v2 import (
            ensure_allocation_settlement_frontier_v2,
        )

        ensure_allocation_frontier = ensure_allocation_settlement_frontier_v2
    try:
        frontier_result = await _maybe_await(
            ensure_allocation_frontier(
                netuid=effective_netuid,
                through_epoch=effective_epoch,
                release_manifest=release,
                supported_operations=health["supported_operations"],
                client=client,
                boot_verifier=verifier,
                execute=execute,
            )
        )
    except Exception as exc:
        from gateway.tee.bootstrap_allocation_settlement_frontier_v2 import (
            AllocationSettlementFrontierBootstrapV2Unsupported,
        )

        if isinstance(
            exc,
            AllocationSettlementFrontierBootstrapV2Unsupported,
        ):
            raise ActiveAncestryCheckpointBootstrapV2Unsupported(
                str(exc)
            ) from exc
        raise
    if not isinstance(frontier_result, Mapping) or frontier_result.get(
        "status"
    ) not in {
        "initialized",
        "already_initialized",
        "awaiting_first_allocation",
        "recovered_signed_frontier",
    }:
        raise ActiveAncestryCheckpointBootstrapV2Error(
            "allocation settlement frontier bootstrap returned no authority"
        )
    _marker(
        started_at=started_at,
        stage="allocation_frontier_ready",
        round_number=0,
        status=frontier_result["status"],
        frontier_hash=frontier_result.get("frontier_hash", "existing"),
    )
    policy = ResearchLabGatewayConfig.from_env().reimbursement_policy_doc(enabled=True)
    total_new_proofs = 0
    total_resume_proofs = 0
    total_legacy_graphs = 0
    selected_root_count = 0

    for round_number in range(1, max_stability_rounds + 1):
        selected = await _select_active_graphs(
            epoch_id=effective_epoch,
            netuid=effective_netuid,
            policy=policy,
            load_allocation_graphs=load_allocation_graphs,
            load_sourcing_graphs=load_sourcing_graphs,
        )
        selected_roots = tuple(selected)
        selected_root_count = len(selected_roots)
        legacy = {
            root: graph
            for root, graph in selected.items()
            if graph.get("schema_version") == RECEIPT_GRAPH_SCHEMA_VERSION
        }
        selected_sizes = {
            root: _canonical_size(graph) for root, graph in selected.items()
        }
        selected_bytes = sum(selected_sizes.values())
        _marker(
            started_at=started_at,
            stage="selected",
            round_number=round_number,
            selected_root_count=len(selected),
            legacy_root_count=len(legacy),
            selected_graph_bytes=selected_bytes,
        )
        for root in sorted(legacy):
            graph_started = time.monotonic()
            counts = await _bootstrap_one_graph(
                graph=legacy[root],
                epoch_id=effective_epoch,
                release=release,
                ancestry_lineage_id=ancestry_lineage_id,
                boot_verifier=verifier,
                client=client,
                execute=execute,
                load_proofs=load_proofs,
                load_checkpointed_graphs=load_checkpointed_graphs,
                persist_checkpoint=persist_checkpoint,
            )
            total_legacy_graphs += 1
            total_new_proofs += counts["new_proof_count"]
            total_resume_proofs += counts["resume_proof_count"]
            _marker(
                started_at=started_at,
                stage="root_persisted",
                round_number=round_number,
                root_receipt_hash=root,
                graph_bytes=selected_sizes[root],
                root_seconds="%.3f" % (time.monotonic() - graph_started),
                **counts,
            )

        # Freeze the authority snapshot selected after the old gateway stops.
        # A long first-time conversion can cross an epoch boundary while the
        # gateway is offline; advancing the selection here would require a
        # chain-realized settlement that only the restarted gateway can attest.
        # Re-selecting the frozen epoch still proves that its exact active root
        # set did not mutate while checkpoints were persisted.
        observed_epoch = effective_epoch
        observed = await _select_active_graphs(
            epoch_id=observed_epoch,
            netuid=effective_netuid,
            policy=policy,
            load_allocation_graphs=load_allocation_graphs,
            load_sourcing_graphs=load_sourcing_graphs,
        )
        observed_roots = tuple(observed)
        observed_legacy = [
            root
            for root, graph in observed.items()
            if graph.get("schema_version") == RECEIPT_GRAPH_SCHEMA_VERSION
        ]
        stable = (
            observed_epoch == effective_epoch
            and observed_roots == selected_roots
            and not observed_legacy
        )
        _marker(
            started_at=started_at,
            stage="reselected",
            round_number=round_number,
            observed_epoch=observed_epoch,
            observed_root_count=len(observed),
            observed_legacy_root_count=len(observed_legacy),
            roots_stable=str(stable).lower(),
        )
        if stable:
            result = {
                "schema_version": ("leadpoet.active_ancestry_checkpoint_bootstrap.v2"),
                "status": "complete",
                "epoch_id": effective_epoch,
                "netuid": effective_netuid,
                "stability_rounds": round_number,
                "active_root_count": selected_root_count,
                "legacy_graphs_processed": total_legacy_graphs,
                "new_proof_count": total_new_proofs,
                "resume_proof_count": total_resume_proofs,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }
            _marker(
                started_at=started_at,
                stage="complete",
                round_number=round_number,
                active_root_count=selected_root_count,
                new_proof_count=total_new_proofs,
            )
            return result
    raise ActiveAncestryCheckpointBootstrapV2Error(
        "active ancestry roots did not stabilize within the bounded rounds"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Checkpoint active legacy V2 receipt ancestry"
    )
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--netuid", type=int)
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=DEFAULT_RELEASE_MANIFEST_PATH,
    )
    parser.add_argument(
        "--max-stability-rounds",
        type=int,
        default=DEFAULT_MAX_STABILITY_ROUNDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        result = asyncio.run(
            bootstrap_active_ancestry_checkpoints_v2(
                epoch_id=args.epoch,
                netuid=args.netuid,
                release_manifest_path=args.release_manifest,
                max_stability_rounds=args.max_stability_rounds,
            )
        )
    except ActiveAncestryCheckpointBootstrapV2Unsupported as exc:
        print(
            json.dumps(
                {
                    "schema_version": (
                        "leadpoet.active_ancestry_checkpoint_bootstrap.v2"
                    ),
                    "status": "unsupported",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 3
    except Exception as exc:
        logger.exception("active ancestry checkpoint bootstrap failed")
        print("active ancestry checkpoint bootstrap failed: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
