"""Strict host bridge to authoritative V2 Research Lab scoring enclaves.

This module contains transport and verification only. Scoring formulas remain
in their existing modules and execute through ``gateway.tee.scoring_executor``
inside the shared measured scoring role. A result is returned only after
its complete graph is independently verified and durably persisted.
"""

from __future__ import annotations

import asyncio
import base64
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from gateway.tee.execution_job_manager_v2 import (
    JOB_SCHEMA_VERSION,
    PARENT_RECEIPT_GRAPHS_FIELD,
)
from gateway.tee.release_manifest_v2 import (
    role_expectation,
    validate_release_manifest,
)
from gateway.tee.release_lineage_v2 import (
    build_release_lineage_boot_verifier_v2,
    load_approved_release_lineage_v2,
)
from gateway.tee.scoring_executor_v2 import SCORING_OPERATIONS_V2
from gateway.utils.tee_artifact_store_v2 import (
    ATTESTED_V2_ARTIFACT_KEY_PREFIX,
)
from gateway.utils.tee_client import coordinator_tee_client, scoring_tee_client
from leadpoet_canonical.attested_v2 import (
    build_receipt_graph,
    canonical_json,
    EMPTY_ARTIFACT_ROOT,
    merkle_root,
    sha256_bytes,
    sha256_json,
    validate_receipt_graph,
    validate_signed_execution_receipt,
    verify_boot_identity_nitro,
)


DEFAULT_RELEASE_MANIFEST_PATH = Path(
    "/home/ec2-user/tee/gateway-v2-release-manifest.json"
)
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_POLL_SECONDS = 0.25
UPLOAD_CHUNK_BYTES = 512 * 1024
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _scoring_proxy_required() -> bool:
    return any(
        _env_flag(name)
        for name in (
            "RESEARCH_LAB_REQUIRE_QUALIFICATION_PROXY",
            "RESEARCH_LAB_SCORING_WORKER_REQUIRE_PROXY",
            "RESEARCH_LAB_V2_REQUIRE_TLS_PROXY",
        )
    )


class AttestedScoringV2Error(RuntimeError):
    """An authoritative V2 scoring result is unavailable or unverifiable."""

    def __init__(self, message: str, *, authority: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.authority = dict(authority) if isinstance(authority, Mapping) else None


def scoring_enclave_shard_for_worker(worker_index: int) -> int:
    try:
        normalized = int(worker_index)
    except (TypeError, ValueError) as exc:
        raise AttestedScoringV2Error("scoring worker index is invalid") from exc
    if normalized < 0 or normalized >= 500:
        raise AttestedScoringV2Error("scoring worker index is outside 0-499")
    return 0


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _compact_parent_graphs_for_transport(
    parent_graphs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Omit graph roots already authenticated inside another supplied graph."""

    normalized = [dict(graph) for graph in parent_graphs]
    roots = [str(graph.get("root_receipt_hash") or "") for graph in normalized]
    if any(not _HASH_RE.fullmatch(root) for root in roots):
        raise AttestedScoringV2Error("parent graph root is invalid")
    if len(set(roots)) != len(roots):
        raise AttestedScoringV2Error("parent receipt graph is duplicated")

    receipt_sets = []
    for graph in normalized:
        receipts = graph.get("receipts")
        if not isinstance(receipts, list):
            raise AttestedScoringV2Error("parent graph receipts are invalid")
        receipt_hashes = {
            str(receipt.get("receipt_hash") or "")
            for receipt in receipts
            if isinstance(receipt, Mapping)
        }
        if len(receipt_hashes) != len(receipts) or any(
            not _HASH_RE.fullmatch(receipt_hash)
            for receipt_hash in receipt_hashes
        ):
            raise AttestedScoringV2Error("parent graph receipt hashes are invalid")
        receipt_sets.append(receipt_hashes)

    retained = [
        graph
        for index, graph in enumerate(normalized)
        if not any(
            roots[index] in receipt_hashes
            for other, receipt_hashes in enumerate(receipt_sets)
            if other != index
        )
    ]
    retained_receipts = {
        receipt_hash
        for graph in retained
        for receipt_hash in {
            str(receipt.get("receipt_hash") or "")
            for receipt in graph["receipts"]
            if isinstance(receipt, Mapping)
        }
    }
    if not set(roots).issubset(retained_receipts):
        raise AttestedScoringV2Error(
            "compacted parent graphs do not cover declared ancestry"
        )
    return retained


def _load_release(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttestedScoringV2Error("V2 release manifest is unavailable") from exc
    return validate_release_manifest(value)


def _merge_graphs(
    *,
    root_receipt: Mapping[str, Any],
    boot_identity: Mapping[str, Any],
    local_receipts: Sequence[Mapping[str, Any]],
    transport_attempts: Sequence[Mapping[str, Any]],
    parent_graphs: Sequence[Mapping[str, Any]],
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> Dict[str, Any]:
    boots = {str(boot_identity["boot_identity_hash"]): dict(boot_identity)}
    receipts = {}
    for receipt in local_receipts:
        validate_signed_execution_receipt(receipt)
        key = str(receipt["receipt_hash"])
        if key in receipts:
            raise AttestedScoringV2Error("local receipt is duplicated")
        receipts[key] = dict(receipt)
    root_hash = str(root_receipt["receipt_hash"])
    if root_hash not in receipts or receipts[root_hash] != dict(root_receipt):
        raise AttestedScoringV2Error(
            "root receipt is missing from the local receipt chain"
        )
    attempts = {
        str(item["attempt_hash"]): dict(item) for item in transport_attempts
    }
    host_operations = {}
    local_hashes = set(receipts)
    expected_parent_roots = {
        str(parent_hash)
        for receipt in receipts.values()
        for parent_hash in receipt["parent_receipt_hashes"]
        if str(parent_hash) not in local_hashes
    }
    allowed_failed = {str(item) for item in allowed_failed_receipt_hashes}
    observed_parent_roots = set()
    for graph in parent_graphs:
        graph_receipt_hashes = {
            str(item.get("receipt_hash") or "")
            for item in graph.get("receipts") or ()
            if isinstance(item, Mapping)
        }
        validate_receipt_graph(
            graph,
            allowed_failed_receipt_hashes=(allowed_failed & graph_receipt_hashes),
        )
        parent_root = str(graph["root_receipt_hash"])
        if parent_root in observed_parent_roots:
            raise AttestedScoringV2Error("parent receipt graph is duplicated")
        observed_parent_roots.add(parent_root)
        for identity in graph["boot_identities"]:
            key = str(identity["boot_identity_hash"])
            if key in boots and boots[key] != dict(identity):
                raise AttestedScoringV2Error("boot identity hash conflicts across graphs")
            boots[key] = dict(identity)
        for receipt in graph["receipts"]:
            key = str(receipt["receipt_hash"])
            if key in receipts and receipts[key] != dict(receipt):
                raise AttestedScoringV2Error("receipt hash conflicts across graphs")
            receipts[key] = dict(receipt)
        for attempt in graph["transport_attempts"]:
            key = str(attempt["attempt_hash"])
            if key in attempts and attempts[key] != dict(attempt):
                raise AttestedScoringV2Error("transport hash conflicts across graphs")
            attempts[key] = dict(attempt)
        for record in graph["host_operations"]:
            key = str(record["request"]["request_hash"])
            if key in host_operations and host_operations[key] != dict(record):
                raise AttestedScoringV2Error(
                    "host operation request conflicts across graphs"
                )
            host_operations[key] = dict(record)
    if observed_parent_roots != expected_parent_roots:
        raise AttestedScoringV2Error(
            "parent graphs differ from direct receipt ancestry"
        )
    return build_receipt_graph(
        root_receipt_hash=root_hash,
        boot_identities=[boots[key] for key in sorted(boots)],
        receipts=[receipts[key] for key in sorted(receipts)],
        transport_attempts=[attempts[key] for key in sorted(attempts)],
        host_operations=[host_operations[key] for key in sorted(host_operations)],
        allowed_failed_receipt_hashes=allowed_failed,
    )


def _release_boot_verifier(release: Mapping[str, Any]):
    def verify(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        physical_role = str(identity.get("physical_role") or "")
        expectation = role_expectation(release, physical_role)
        if identity.get("commit_sha") != expectation["commit_sha"]:
            raise AttestedScoringV2Error("boot commit differs from V2 release")
        if identity.get("build_manifest_hash") != expectation["build_manifest_hash"]:
            raise AttestedScoringV2Error("boot manifest differs from V2 release")
        if identity.get("dependency_lock_hash") != expectation["dependency_lock_hash"]:
            raise AttestedScoringV2Error("boot dependency lock differs from V2 release")
        return verify_boot_identity_nitro(
            identity,
            expected_pcr0=expectation["pcr0"],
            certificate_validity_at_attestation_time=True,
        )

    return verify


def derive_execution_job_id_v2(
    *,
    operation: str,
    purpose: str,
    epoch_id: int,
    sequence: int,
    payload_sha256: str,
    parent_receipt_hashes: Sequence[str],
    input_artifact_hashes: Sequence[str],
    release_hash: str,
    physical_role: str,
) -> str:
    identity_hash = sha256_bytes(
        _canonical_bytes(
            {
                "operation": operation,
                "purpose": purpose,
                "epoch_id": epoch_id,
                "sequence": sequence,
                "payload_sha256": payload_sha256,
                "parent_receipt_hashes": sorted(parent_receipt_hashes),
                "input_artifact_hashes": sorted(input_artifact_hashes),
                "release_hash": release_hash,
                "physical_role": physical_role,
            }
        )
    )
    return "scoring-v2:%s:%s" % (
        operation.replace("_", "-")[:80],
        identity_hash.split(":", 1)[1][:32],
    )


async def execute_scoring_v2(
    *,
    operation: str,
    purpose: str,
    epoch_id: int,
    sequence: int,
    payload: Mapping[str, Any],
    worker_index: int,
    parent_graphs: Sequence[Mapping[str, Any]] = (),
    allowed_failed_parent_receipt_hashes: Iterable[str] = (),
    input_artifact_hashes: Iterable[str] = (),
    provider_credential_profile: str = "default",
    provider_credential_ref_hashes: Optional[Mapping[str, str]] = None,
    internally_provisioned_credential_slots: Iterable[str] = (),
    require_egress_proxy: Optional[bool] = None,
    provider_profile_loader: Any = None,
    provider_profile_provisioner: Any = None,
    additional_job_credential_envelopes: Sequence[Mapping[str, Any]] = (),
    additional_job_credential_envelope_builder: Any = None,
    job_credential_provisioner: Any = None,
    credential_coordinator_client: Any = coordinator_tee_client,
    release_manifest: Optional[Mapping[str, Any]] = None,
    release_manifest_path: Path = DEFAULT_RELEASE_MANIFEST_PATH,
    release_channel_loader: Any = None,
    client: Any = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    persist_graph: Any = None,
    boot_verifier: Any = None,
    operation_registry: Optional[Mapping[str, Iterable[str]]] = None,
    physical_role_override: Optional[str] = None,
    expected_service_role: str = "gateway_scoring",
    rpc_namespace: str = "scoring_v2",
    artifact_coordinator_client: Any = coordinator_tee_client,
    persist_artifact: Any = None,
    artifact_bucket: Optional[str] = None,
    artifact_key_prefix: str = ATTESTED_V2_ARTIFACT_KEY_PREFIX,
    artifact_lineage_attestor: Any = None,
    persist_sidecars: Any = None,
    receipt_output_projector: Any = None,
    allow_persistence_bound_artifact_descriptors: bool = False,
    load_replayable_result: Any = None,
    persist_replayable_result: Any = None,
) -> Dict[str, Any]:
    """Execute one scoring operation as V2 authority and persist its graph."""

    registry = dict(operation_registry or SCORING_OPERATIONS_V2)
    if operation not in registry:
        raise AttestedScoringV2Error("unsupported V2 scoring operation")
    if purpose not in registry[operation]:
        raise AttestedScoringV2Error("purpose is not authorized for scoring operation")
    if not isinstance(epoch_id, int) or epoch_id < 0:
        raise AttestedScoringV2Error("epoch_id is invalid")
    if not isinstance(sequence, int) or sequence < 0:
        raise AttestedScoringV2Error("sequence is invalid")
    release = validate_release_manifest(release_manifest) if release_manifest else _load_release(release_manifest_path)
    if physical_role_override is None:
        scoring_enclave_shard_for_worker(worker_index)
        normalized_worker_index = int(worker_index)
        scoring_client = client or scoring_tee_client
        physical_role = "gateway_scoring"
    else:
        physical_role = str(physical_role_override)
        normalized_worker_index = int(worker_index)
        scoring_client = client
        if scoring_client is None:
            raise AttestedScoringV2Error("V2 role client is required")
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", rpc_namespace):
        raise AttestedScoringV2Error("V2 RPC namespace is invalid")

    def rpc_method(action: str):
        method = getattr(scoring_client, "%s_%s" % (rpc_namespace, action), None)
        if method is None:
            raise AttestedScoringV2Error("V2 role client method is unavailable")
        return method

    health = await rpc_method("health")()
    execution_worker_count = health.get("worker_count")
    configured_worker_count = health.get("configured_worker_count")
    coordinator_capacity_valid = (
        physical_role == "gateway_coordinator"
        and configured_worker_count == 0
        and normalized_worker_index == 0
    )
    configured_capacity_valid = (
        physical_role != "gateway_coordinator"
        and type(configured_worker_count) is int
        and 1 <= configured_worker_count <= 500
        and normalized_worker_index < configured_worker_count
    )
    if (
        health.get("authority") != "v2_only"
        or health.get("role") != expected_service_role
        or health.get("physical_role") != physical_role
        or type(execution_worker_count) is not int
        or not 1 <= execution_worker_count <= 10
        or not (coordinator_capacity_valid or configured_capacity_valid)
        or not health.get("workers_alive")
    ):
        raise AttestedScoringV2Error("V2 scoring enclave health is invalid")
    boot_identity = await scoring_client.v2_get_boot_identity()
    expectation = role_expectation(release, physical_role)
    if health.get("boot_identity_hash") != boot_identity.get("boot_identity_hash"):
        raise AttestedScoringV2Error("V2 scoring health boot identity mismatch")
    current_release_verifier = boot_verifier or _release_boot_verifier(release)
    current_release_verifier(boot_identity)
    if boot_identity.get("commit_sha") != expectation["commit_sha"]:
        raise AttestedScoringV2Error("V2 scoring commit differs from release")
    if boot_verifier is not None:
        verifier = boot_verifier
    else:
        try:
            approved_lineage = await asyncio.to_thread(
                load_approved_release_lineage_v2,
                current_release=release,
                parent_graphs=parent_graphs,
                release_channel_loader=release_channel_loader,
            )
            verifier = build_release_lineage_boot_verifier_v2(approved_lineage)
        except Exception as exc:
            raise AttestedScoringV2Error(
                "V2 receipt release lineage is unavailable"
            ) from exc

    normalized_profile = str(provider_credential_profile or "default")
    if normalized_profile not in {
        "default",
        "benchmark_model",
        "benchmark_scorer",
        "source_add_judge",
    }:
        raise AttestedScoringV2Error("provider credential profile is invalid")
    if provider_profile_loader is None:
        from gateway.research_lab.provider_profiles_v2 import (
            load_provider_profile_v2,
        )

        provider_profile_loader = load_provider_profile_v2
    profile_document = provider_profile_loader(
        normalized_profile,
        execution_role="gateway_scoring",
        worker_index=normalized_worker_index,
        require_egress_proxy=(
            _scoring_proxy_required()
            if require_egress_proxy is None
            else bool(require_egress_proxy)
        ),
    )
    if not isinstance(profile_document, Mapping):
        raise AttestedScoringV2Error(
            "provider credential profile document is invalid"
        )
    profile_refs = dict(profile_document.get("credential_ref_hashes") or {})
    supplied_refs = dict(provider_credential_ref_hashes or {})
    if any(
        slot in supplied_refs and supplied_refs[slot] != digest
        for slot, digest in profile_refs.items()
    ):
        raise AttestedScoringV2Error(
            "provider credential references differ from profile"
        )
    provider_credential_ref_hashes = {**profile_refs, **supplied_refs}
    credential_refs = {
        str(provider_id): str(digest or "").lower()
        for provider_id, digest in dict(
            provider_credential_ref_hashes or {}
        ).items()
    }
    if any(
        not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", provider_id)
        or not _HASH_RE.fullmatch(digest)
        for provider_id, digest in credential_refs.items()
    ):
        raise AttestedScoringV2Error(
            "provider credential profile commitment is invalid"
        )
    expected_additional_refs = {
        slot: digest
        for slot, digest in credential_refs.items()
        if slot not in profile_refs
    }
    internal_slots = {
        str(slot) for slot in internally_provisioned_credential_slots
    }
    if (
        any(
            not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", slot)
            for slot in internal_slots
        )
        or not internal_slots.issubset(expected_additional_refs)
    ):
        raise AttestedScoringV2Error(
            "internally provisioned credential slots are invalid"
        )
    envelope_expected_refs = {
        slot: digest
        for slot, digest in expected_additional_refs.items()
        if slot not in internal_slots
    }
    if (
        "_v2_provider_credential_ref_hashes" in payload
        or "_v2_provider_credential_profile" in payload
        or PARENT_RECEIPT_GRAPHS_FIELD in payload
    ):
        raise AttestedScoringV2Error("payload uses a reserved V2 authority field")
    artifact_hashes = sorted(
        {str(item).lower() for item in input_artifact_hashes}
        | set(credential_refs.values())
    )
    if any(not _HASH_RE.fullmatch(item) for item in artifact_hashes):
        raise AttestedScoringV2Error("input artifact hash is invalid")
    parent_roots = sorted(str(graph.get("root_receipt_hash") or "") for graph in parent_graphs)
    if any(not _HASH_RE.fullmatch(item) for item in parent_roots):
        raise AttestedScoringV2Error("parent graph root is invalid")
    allowed_failed = {
        str(item).lower() for item in allowed_failed_parent_receipt_hashes
    }
    if any(not _HASH_RE.fullmatch(item) for item in allowed_failed):
        raise AttestedScoringV2Error("allowed failed parent receipt is invalid")
    parent_graph_receipt_hashes = {
        str(item.get("receipt_hash") or "")
        for graph in parent_graphs
        for item in graph.get("receipts") or ()
        if isinstance(item, Mapping)
    }
    if not allowed_failed.issubset(parent_graph_receipt_hashes):
        raise AttestedScoringV2Error(
            "allowed failed receipts must be present in parent graphs"
        )
    transport_parent_graphs = _compact_parent_graphs_for_transport(parent_graphs)
    payload_document = dict(payload)
    if transport_parent_graphs:
        payload_document[PARENT_RECEIPT_GRAPHS_FIELD] = transport_parent_graphs
    if normalized_profile != "default":
        payload_document["_v2_provider_credential_profile"] = normalized_profile
    if credential_refs:
        payload_document["_v2_provider_credential_ref_hashes"] = dict(
            sorted(credential_refs.items())
        )
    payload_bytes = _canonical_bytes(payload_document)
    payload_hash = sha256_bytes(payload_bytes)
    job_id = derive_execution_job_id_v2(
        operation=operation,
        purpose=purpose,
        epoch_id=epoch_id,
        sequence=sequence,
        payload_sha256=payload_hash,
        parent_receipt_hashes=parent_roots,
        input_artifact_hashes=artifact_hashes,
        release_hash=release["release_hash"],
        physical_role=physical_role,
    )
    from gateway.research_lab.attested_v2_store import (
        replayable_execution_result_v2,
    )

    replayable_result = (
        physical_role == "gateway_coordinator"
        and replayable_execution_result_v2(
            operation=operation,
            purpose=purpose,
        )
    )
    if replayable_result:
        if load_replayable_result is None:
            from gateway.research_lab.attested_v2_store import (
                load_execution_result_v2,
            )

            load_replayable_result = load_execution_result_v2
        replay = await load_replayable_result(
            role=expected_service_role,
            operation=operation,
            purpose=purpose,
            job_id=job_id,
        )
        if replay is not None:
            replay_result = replay.get("result")
            replay_receipt = replay.get("receipt")
            replay_graph = replay.get("receipt_graph")
            replay_artifacts = replay.get("artifact_hashes")
            replay_row = replay.get("row")
            if (
                not isinstance(replay_result, Mapping)
                or not isinstance(replay_receipt, Mapping)
                or not isinstance(replay_graph, Mapping)
                or not isinstance(replay_artifacts, list)
                or not isinstance(replay_row, Mapping)
            ):
                raise AttestedScoringV2Error(
                    "V2 durable execution replay is incomplete"
                )
            validate_receipt_graph(
                replay_graph,
                required_purposes=(purpose,),
                boot_attestation_verifier=verifier,
                require_boot_attestation_verification=True,
            )
            replay_boots = {
                str(item.get("boot_identity_hash") or ""): item
                for item in replay_graph.get("boot_identities") or ()
                if isinstance(item, Mapping)
            }
            replay_boot = replay_boots.get(
                str(replay_receipt.get("boot_identity_hash") or "")
            )
            if not isinstance(replay_boot, Mapping):
                raise AttestedScoringV2Error(
                    "V2 durable execution replay boot identity is missing"
                )
            current_release_verifier(replay_boot)
            projected_replay = dict(replay_result)
            if receipt_output_projector is not None:
                projected_replay = receipt_output_projector(
                    operation,
                    replay_result,
                )
            if (
                replay_graph.get("root_receipt_hash")
                != replay_receipt.get("receipt_hash")
                or replay_receipt.get("role") != expected_service_role
                or replay_receipt.get("purpose") != purpose
                or replay_receipt.get("job_id") != job_id
                or replay_receipt.get("epoch_id") != epoch_id
                or replay_receipt.get("sequence") != sequence
                or replay_receipt.get("input_root") != payload_hash
                or replay_receipt.get("parent_receipt_hashes") != parent_roots
                or replay_receipt.get("output_root")
                != sha256_bytes(_canonical_bytes(dict(projected_replay)))
                or replay_receipt.get("artifact_root")
                != merkle_root(
                    replay_artifacts,
                    domain="leadpoet-artifact-v2",
                )
                or replay_row.get("release_hash") != release["release_hash"]
            ):
                raise AttestedScoringV2Error(
                    "V2 durable execution replay differs from current authority"
                )
            replay_attempts = [
                dict(item)
                for item in replay_graph.get("transport_attempts") or ()
                if item.get("job_id") == job_id
                and item.get("purpose") == purpose
            ]
            persistence = {
                "graph_hash": sha256_json(dict(replay_graph)),
                "root_receipt_hash": str(replay_graph["root_receipt_hash"]),
                "boot_count": len(replay_graph["boot_identities"]),
                "receipt_count": len(replay_graph["receipts"]),
                "transport_attempt_count": len(
                    replay_graph["transport_attempts"]
                ),
                "host_operation_count": len(
                    replay_graph["host_operations"]
                ),
            }
            return {
                "status": "succeeded",
                "result": dict(replay_result),
                "receipt": dict(replay_receipt),
                "execution_receipt": dict(replay_receipt),
                "receipt_graph": dict(replay_graph),
                "transitions": [],
                "transport_attempts": replay_attempts,
                "artifact_persistence": [],
                "artifact_hashes": list(replay_artifacts),
                "persistence": persistence,
                "sidecar_persistence": {},
                "release_hash": release["release_hash"],
                "physical_role": physical_role,
                "replay_status": "durable_exact",
            }
    raw_additional_envelopes = additional_job_credential_envelopes
    if additional_job_credential_envelope_builder is not None:
        if additional_job_credential_envelopes:
            raise AttestedScoringV2Error(
                "additional job credential sources are ambiguous"
            )
        raw_additional_envelopes = additional_job_credential_envelope_builder(
            job_id
        )
        if inspect.isawaitable(raw_additional_envelopes):
            raw_additional_envelopes = await raw_additional_envelopes
    if raw_additional_envelopes is None:
        raw_additional_envelopes = ()
    if not isinstance(raw_additional_envelopes, (list, tuple)):
        raise AttestedScoringV2Error(
            "additional job credential envelopes are invalid"
        )
    from gateway.utils.tee_kms_provision_v2 import (
        validate_job_credential_envelope_v2,
    )

    additional_envelopes = [
        validate_job_credential_envelope_v2(item)
        for item in raw_additional_envelopes
    ]
    if any(item["job_id"] != job_id for item in additional_envelopes):
        raise AttestedScoringV2Error(
            "additional job credential job binding differs"
        )
    additional_slots = [
        str(item["credential_slot"]) for item in additional_envelopes
    ]
    if len(set(additional_slots)) != len(additional_envelopes):
        raise AttestedScoringV2Error(
            "additional job credential slot is duplicated"
        )
    if sorted(
        str(item["credential_value_hash"]) for item in additional_envelopes
    ) != sorted(envelope_expected_refs.values()):
        raise AttestedScoringV2Error(
            "additional job credential commitments differ"
        )
    manifest = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "operation": operation,
        "purpose": purpose,
        "epoch_id": epoch_id,
        "sequence": sequence,
        "payload_sha256": payload_hash,
        "payload_size_bytes": len(payload_bytes),
        "parent_receipt_hashes": parent_roots,
        "input_artifact_hashes": artifact_hashes,
        "provider_credential_profile": normalized_profile,
        "provider_credential_ref_hashes": dict(sorted(credential_refs.items())),
    }
    async def run_remote_job() -> tuple[dict[str, Any], str, dict[str, Any]]:
        submitted = await rpc_method("submit_job")(manifest)
        if submitted.get("state") == "uploading":
            offset = int(submitted.get("uploaded_bytes") or 0)
            while offset < len(payload_bytes):
                chunk = payload_bytes[offset : offset + UPLOAD_CHUNK_BYTES]
                await rpc_method("put_chunk")(
                    job_id=job_id,
                    offset=offset,
                    data=chunk,
                )
                offset += len(chunk)
            await rpc_method("seal_job")(job_id)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(1.0, float(timeout_seconds))
        while True:
            status = await rpc_method("get_status")(job_id)
            state = str(status.get("state") or "")
            if state in {"succeeded", "failed", "cancelled"}:
                break
            if loop.time() >= deadline:
                await rpc_method("cancel_job")(job_id)
                raise AttestedScoringV2Error("V2 scoring job timed out")
            await asyncio.sleep(max(0.01, float(poll_seconds)))
        receipt = await rpc_method("get_receipt")(job_id)
        return dict(status), state, dict(receipt)

    leased_credentials = False
    leased_slot_count = 0
    try:
        if profile_document is not None and profile_refs:
            if provider_profile_provisioner is None:
                from gateway.research_lab.provider_profiles_v2 import (
                    provision_provider_profile_v2,
                )

                provider_profile_provisioner = provision_provider_profile_v2
            lease = await provider_profile_provisioner(
                profile_document,
                job_id=job_id,
                client=credential_coordinator_client,
            )
            if (
                lease.get("job_id") != job_id
                or dict(lease.get("credential_ref_hashes") or {})
                != profile_refs
                or int(lease.get("leased_credential_count") or 0)
                != len(profile_refs)
            ):
                raise AttestedScoringV2Error(
                    "provider credential profile lease differs"
                )
            leased_credentials = True
            leased_slot_count += len(profile_refs)
        if additional_envelopes:
            if job_credential_provisioner is None:
                from gateway.utils.tee_kms_provision_v2 import (
                    provision_job_credential_envelope_v2,
                )

                job_credential_provisioner = provision_job_credential_envelope_v2
            for envelope in additional_envelopes:
                wire_envelope = {
                    key: item
                    for key, item in envelope.items()
                    if key not in {"ciphertext_blob", "envelope_kind"}
                }
                leased = await job_credential_provisioner(
                    wire_envelope,
                    client=credential_coordinator_client,
                )
                if (
                    leased.get("job_id") != job_id
                    or leased.get("credential_slot")
                    != envelope["credential_slot"]
                    or leased.get("credential_ref_hash")
                    != envelope["credential_value_hash"]
                ):
                    raise AttestedScoringV2Error(
                        "additional job credential lease differs"
                    )
                leased_credentials = True
                leased_slot_count += 1
        status, state, receipt = await run_remote_job()
    finally:
        if leased_credentials:
            released = await credential_coordinator_client.v2_release_job_credentials(
                job_id
            )
            if (
                released.get("status") != "released"
                or released.get("job_id") != job_id
                or int(released.get("released_slot_count") or 0)
                != leased_slot_count
            ):
                raise AttestedScoringV2Error(
                    "provider credential profile release failed"
                )
    validate_signed_execution_receipt(receipt)
    if receipt.get("input_root") != payload_hash:
        raise AttestedScoringV2Error("V2 scoring receipt input mismatch")
    if receipt.get("boot_identity_hash") != boot_identity.get("boot_identity_hash"):
        raise AttestedScoringV2Error("V2 scoring receipt boot mismatch")
    succeeded = state == "succeeded" and receipt.get("status") == "succeeded"
    if succeeded:
        result_bytes = bytearray()
        offset = 0
        while True:
            part = await rpc_method("get_result")(job_id, offset=offset)
            chunk = base64.b64decode(str(part.get("data_b64") or ""), validate=True)
            if sha256_bytes(chunk) != part.get("chunk_sha256"):
                raise AttestedScoringV2Error("V2 scoring result chunk hash mismatch")
            result_bytes.extend(chunk)
            offset += len(chunk)
            if part.get("eof"):
                if part.get("result_sha256") != status.get("result_sha256"):
                    raise AttestedScoringV2Error("V2 scoring result hash mismatch")
                break
        try:
            result = json.loads(bytes(result_bytes).decode("utf-8"))
        except Exception as exc:
            raise AttestedScoringV2Error("V2 scoring result is invalid JSON") from exc
        if not isinstance(result, dict) or _canonical_bytes(result) != bytes(result_bytes):
            raise AttestedScoringV2Error("V2 scoring result is not canonical")
        receipt_output = result
        if receipt_output_projector is not None:
            receipt_output = receipt_output_projector(operation, result)
            if not isinstance(receipt_output, Mapping):
                raise AttestedScoringV2Error(
                    "V2 scoring receipt projection is invalid"
                )
        if receipt.get("output_root") != sha256_bytes(
            _canonical_bytes(dict(receipt_output))
        ):
            raise AttestedScoringV2Error("V2 scoring receipt output mismatch")
    else:
        failure_code = str(status.get("error_code") or state)
        result = {"status": "failed", "failure_code": failure_code}
        if (
            receipt.get("status") != "failed"
            or receipt.get("failure_code") != failure_code
            or receipt.get("output_root") != sha256_bytes(_canonical_bytes(result))
        ):
            raise AttestedScoringV2Error(
                "V2 scoring failure receipt differs from terminal status"
            )

    local_receipts = await rpc_method("get_receipts")(job_id)
    local_receipt_hashes = {
        str(item.get("receipt_hash") or "")
        for item in local_receipts
        if isinstance(item, Mapping)
    }
    external_parent_hashes = {
        str(parent_hash)
        for item in local_receipts
        if isinstance(item, Mapping)
        for parent_hash in item.get("parent_receipt_hashes") or ()
        if str(parent_hash) not in local_receipt_hashes
    }
    if external_parent_hashes != set(parent_roots):
        raise AttestedScoringV2Error("V2 scoring receipt ancestry mismatch")
    transport_attempts = await rpc_method("get_transport_attempts")(job_id)
    job_artifact_hashes = await rpc_method("get_artifact_hashes")(job_id)
    if not isinstance(job_artifact_hashes, list) or any(
        not _HASH_RE.fullmatch(str(item or "")) for item in job_artifact_hashes
    ):
        raise AttestedScoringV2Error("V2 scoring artifact hash set is invalid")
    expected_artifact_root = (
        merkle_root(job_artifact_hashes, domain="leadpoet-artifact-v2")
        if job_artifact_hashes
        else EMPTY_ARTIFACT_ROOT
    )
    if receipt.get("artifact_root") != expected_artifact_root:
        raise AttestedScoringV2Error("V2 scoring artifact root differs")
    transitions = await rpc_method("get_transitions")(job_id) if succeeded else []
    graph_allowed_failed = set(allowed_failed)
    if not succeeded:
        graph_allowed_failed.add(str(receipt["receipt_hash"]))
    request_artifact_hashes = sorted(
        str(attempt.get("request_artifact_hash") or "")
        for attempt in transport_attempts
        if attempt.get("provider_id") != "aws_s3_object_lock"
    )
    response_artifact_hashes = sorted(
        str(attempt.get("response_artifact_hash") or "")
        for attempt in transport_attempts
        if attempt.get("terminal_status")
        in {"authenticated_response", "attested_local_response"}
        and attempt.get("provider_id") != "aws_s3_object_lock"
    )
    transport_artifact_hashes = sorted(
        request_artifact_hashes + response_artifact_hashes
    )
    if not succeeded:
        graph = _merge_graphs(
            root_receipt=receipt,
            boot_identity=boot_identity,
            local_receipts=local_receipts,
            transport_attempts=transport_attempts,
            parent_graphs=parent_graphs,
            allowed_failed_receipt_hashes=graph_allowed_failed,
        )
        validate_receipt_graph(
            graph,
            required_purposes=(purpose,),
            allowed_failed_receipt_hashes=graph_allowed_failed,
            boot_attestation_verifier=verifier,
            require_boot_attestation_verification=True,
        )
        outcome = {
            "status": "failed",
            "result": result,
            "receipt": dict(receipt),
            "execution_receipt": dict(receipt),
            "receipt_graph": graph,
            "transitions": [],
            "transport_attempts": list(transport_attempts),
            "artifact_hashes": list(job_artifact_hashes),
            "release_hash": release["release_hash"],
            "physical_role": physical_role,
        }
        if job_artifact_hashes:
            from gateway.research_lab.attested_artifacts_v2 import (
                persist_execution_transport_artifacts_v2,
            )

            artifact_outcome = await persist_execution_transport_artifacts_v2(
                job_id=job_id,
                purpose=purpose,
                epoch_id=epoch_id,
                sequence=sequence,
                source_receipt=receipt,
                source_graph=graph,
                transport_attempts=transport_attempts,
                execution_artifact_hashes=job_artifact_hashes,
                release_manifest=release,
                client=artifact_coordinator_client,
                bucket=artifact_bucket,
                key_prefix=artifact_key_prefix,
            )
            artifact_graph = artifact_outcome.get("receipt_graph")
            artifact_receipt = artifact_outcome.get("receipt")
            if not isinstance(artifact_graph, Mapping) or not isinstance(
                artifact_receipt, Mapping
            ):
                raise AttestedScoringV2Error(
                    "V2 scoring failure artifact lineage is unavailable"
                )
            validate_receipt_graph(
                artifact_graph,
                required_purposes=(purpose, "leadpoet.artifact_persistence.v2"),
                allowed_failed_receipt_hashes=graph_allowed_failed,
                boot_attestation_verifier=verifier,
                require_boot_attestation_verification=True,
            )
            if (
                artifact_graph.get("root_receipt_hash")
                != artifact_receipt.get("receipt_hash")
                or artifact_receipt.get("purpose")
                != "leadpoet.artifact_persistence.v2"
                or receipt["receipt_hash"]
                not in artifact_receipt.get("parent_receipt_hashes", [])
            ):
                raise AttestedScoringV2Error(
                    "V2 scoring failure artifact ancestry is invalid"
                )
            outcome.update(
                {
                    "receipt": dict(artifact_receipt),
                    "receipt_graph": dict(artifact_graph),
                    "artifact_persistence": dict(artifact_outcome),
                }
            )
        else:
            if persist_graph is None:
                from gateway.research_lab.attested_v2_store import (
                    persist_receipt_graph_v2,
                )

                persist_graph = persist_receipt_graph_v2
            persistence = await persist_graph(
                graph,
                allowed_failed_receipt_hashes=graph_allowed_failed,
            )
            if persistence.get("root_receipt_hash") != receipt["receipt_hash"]:
                raise AttestedScoringV2Error(
                    "V2 scoring failure durable readback root mismatch"
                )
            outcome["persistence"] = dict(persistence)
        raise AttestedScoringV2Error(
            "V2 scoring failed closed: %s" % result["failure_code"],
            authority=outcome,
        )
    sealed_model_artifacts = result.get("sealed_artifacts") or []
    if not isinstance(sealed_model_artifacts, list) or any(
        not isinstance(item, Mapping) for item in sealed_model_artifacts
    ):
        raise AttestedScoringV2Error("V2 sealed model artifacts are invalid")
    expected_sealed_hashes = []
    expected_sealed_ids = set()
    for descriptor in sealed_model_artifacts:
        if (
            descriptor.get("status") != "sealed"
            or descriptor.get("job_id") != job_id
            or descriptor.get("purpose") != purpose
            or descriptor.get("artifact_kind")
            not in {"model_output", "model_trace", "provider_evidence_tape"}
            or not _HASH_RE.fullmatch(str(descriptor.get("artifact_id") or ""))
            or not _HASH_RE.fullmatch(
                str(descriptor.get("plaintext_hash") or "")
            )
        ):
            raise AttestedScoringV2Error(
                "V2 sealed model artifact commitment is invalid"
            )
        expected_sealed_ids.add(str(descriptor["artifact_id"]))
        expected_sealed_hashes.append(str(descriptor["plaintext_hash"]))
    expected_artifact_hashes = sorted(
        transport_artifact_hashes + expected_sealed_hashes
    )
    graph = _merge_graphs(
        root_receipt=receipt,
        boot_identity=boot_identity,
        local_receipts=local_receipts,
        transport_attempts=transport_attempts,
        parent_graphs=parent_graphs,
        allowed_failed_receipt_hashes=allowed_failed,
    )
    validate_receipt_graph(
        graph,
        required_purposes=(purpose,),
        allowed_failed_receipt_hashes=allowed_failed,
        boot_attestation_verifier=verifier,
        require_boot_attestation_verification=True,
    )
    artifact_persistence = []
    reuse_persisted_artifacts = False
    if expected_artifact_hashes:
        listed = await artifact_coordinator_client.v2_list_encrypted_artifacts(
            job_id=job_id,
            purpose=purpose,
        )
        artifacts = listed.get("artifacts")
        if not isinstance(artifacts, list):
            raise AttestedScoringV2Error(
                "V2 coordinator encrypted artifact list is invalid"
            )
        observed_hashes = sorted(
            str(item.get("plaintext_hash") or "")
            for item in artifacts
            if isinstance(item, Mapping)
        )
        observed_ids = {
            str(item.get("artifact_id") or "")
            for item in artifacts
            if isinstance(item, Mapping)
        }
        committed_hashes = set(job_artifact_hashes)
        observed_descriptor_hashes = {
            str(item.get(field) or "")
            for item in artifacts
            if isinstance(item, Mapping)
            for field in (
                "artifact_id",
                "plaintext_hash",
                "ciphertext_hash",
                "encryption_context_hash",
            )
            if item.get(field)
        }
        expected_counts = Counter(expected_artifact_hashes)
        observed_counts = Counter(observed_hashes)
        observed_commitments = (
            set(observed_hashes)
            if allow_persistence_bound_artifact_descriptors
            else observed_descriptor_hashes
        )
        if (
            any(observed_counts[key] < count for key, count in expected_counts.items())
            or not expected_sealed_ids.issubset(observed_ids)
            or not observed_commitments.issubset(committed_hashes)
        ):
            raise AttestedScoringV2Error(
                "V2 encrypted artifacts differ from execution commitments"
            )
        reuse_persisted_artifacts = bool(artifacts) and all(
            item.get("persisted") is True for item in artifacts
        )
        from gateway.tee.coordinator_executor_v2 import (
            OP_ATTEST_ARTIFACT_PERSISTENCE,
        )

        lineage_payload = {
            "source_receipt_hash": receipt["receipt_hash"],
            "artifact_ids": [item["artifact_id"] for item in artifacts],
            "artifact_plaintext_hashes": expected_artifact_hashes,
        }
        lineage_payload_document = {
            **lineage_payload,
            PARENT_RECEIPT_GRAPHS_FIELD: [dict(graph)],
        }
        lineage_job_id = derive_execution_job_id_v2(
            operation=OP_ATTEST_ARTIFACT_PERSISTENCE,
            purpose="leadpoet.artifact_persistence.v2",
            epoch_id=epoch_id,
            sequence=sequence,
            payload_sha256=sha256_bytes(
                _canonical_bytes(lineage_payload_document)
            ),
            parent_receipt_hashes=(str(graph["root_receipt_hash"]),),
            input_artifact_hashes=(),
            release_hash=release["release_hash"],
            physical_role="gateway_coordinator",
        )
        if not reuse_persisted_artifacts:
            if persist_artifact is None:
                from gateway.utils.tee_artifact_store_v2 import (
                    persist_enclave_artifact_v2,
                )

                persist_artifact = persist_enclave_artifact_v2
            bucket = str(
                artifact_bucket
                or os.getenv("RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET", "")
                or ""
            ).strip()
            if not bucket:
                raise AttestedScoringV2Error(
                    "V2 encrypted artifact bucket is not configured"
                )
            for artifact in artifacts:
                persistence_result = await persist_artifact(
                    str(artifact["artifact_id"]),
                    bucket=bucket,
                    key_prefix=artifact_key_prefix,
                    client=artifact_coordinator_client,
                    attestation_job_id=lineage_job_id,
                )
                if persistence_result.get("status") != "persisted":
                    raise AttestedScoringV2Error(
                        "V2 encrypted artifact persistence failed closed"
                    )
                artifact_persistence.append(dict(persistence_result))
    if persist_graph is None:
        from gateway.research_lab.attested_v2_store import persist_receipt_graph_v2

        persist_graph = persist_receipt_graph_v2
    if artifact_persistence or reuse_persisted_artifacts:
        if artifact_lineage_attestor is None:
            from gateway.research_lab.attested_coordinator_v2 import (
                execute_coordinator_v2,
            )
            async def artifact_lineage_attestor(**kwargs):
                return await execute_coordinator_v2(
                    operation=OP_ATTEST_ARTIFACT_PERSISTENCE,
                    purpose="leadpoet.artifact_persistence.v2",
                    epoch_id=kwargs["epoch_id"],
                    sequence=kwargs["sequence"],
                    payload=kwargs["payload"],
                    parent_graphs=(kwargs["source_graph"],),
                    input_artifact_hashes=kwargs["input_artifact_hashes"],
                    release_manifest=kwargs["release_manifest"],
                    client=kwargs["client"],
                    timeout_seconds=kwargs["timeout_seconds"],
                    poll_seconds=kwargs["poll_seconds"],
                    persist_graph=kwargs["persist_graph"],
                    boot_verifier=kwargs["boot_verifier"],
                    allowed_failed_parent_receipt_hashes=kwargs[
                        "allowed_failed_parent_receipt_hashes"
                    ],
                )

        lineage = await artifact_lineage_attestor(
            epoch_id=epoch_id,
            sequence=sequence,
            payload=lineage_payload,
            source_graph=graph,
            input_artifact_hashes=(),
            release_manifest=release,
            client=artifact_coordinator_client,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            persist_graph=persist_graph,
            boot_verifier=verifier,
            allowed_failed_parent_receipt_hashes=allowed_failed,
        )
        lineage_graph = lineage.get("receipt_graph")
        lineage_receipt = lineage.get("receipt")
        if not isinstance(lineage_graph, Mapping) or not isinstance(
            lineage_receipt, Mapping
        ):
            raise AttestedScoringV2Error(
                "V2 encrypted artifact lineage receipt is unavailable"
            )
        validate_receipt_graph(
            lineage_graph,
            required_purposes=(purpose, "leadpoet.artifact_persistence.v2"),
            allowed_failed_receipt_hashes=allowed_failed,
            boot_attestation_verifier=verifier,
            require_boot_attestation_verification=True,
        )
        if (
            lineage_graph.get("root_receipt_hash")
            != lineage_receipt.get("receipt_hash")
            or lineage_receipt.get("purpose")
            != "leadpoet.artifact_persistence.v2"
            or receipt["receipt_hash"]
            not in lineage_receipt.get("parent_receipt_hashes", [])
        ):
            raise AttestedScoringV2Error(
                "V2 encrypted artifact lineage ancestry is invalid"
            )
        if reuse_persisted_artifacts:
            lineage_result = lineage.get("result")
            persisted = (
                lineage_result.get("artifacts")
                if isinstance(lineage_result, Mapping)
                else None
            )
            descriptor_by_id = {
                str(item["artifact_id"]): item for item in artifacts
            }
            if (
                not isinstance(persisted, list)
                or {str(item.get("artifact_id") or "") for item in persisted}
                != set(descriptor_by_id)
                or any(not isinstance(item, Mapping) for item in persisted)
            ):
                raise AttestedScoringV2Error(
                    "V2 persisted artifact lineage output is invalid"
                )
            artifact_persistence = [
                {
                    "status": "persisted",
                    "artifact_id": str(item["artifact_id"]),
                    "artifact_ref": str(item["artifact_ref"]),
                    "artifact_kind": str(
                        descriptor_by_id[str(item["artifact_id"])]["artifact_kind"]
                    ),
                    "artifact_hash": str(item["ciphertext_hash"]),
                    "encryption_context_hash": str(
                        item["encryption_context_hash"]
                    ),
                    "object_lock_mode": str(item["object_lock_mode"]),
                    "retain_until": str(item["retain_until"]),
                    "storage_document_hash": str(item["storage_document_hash"]),
                    "transport_root": str(item["transport_root"]),
                }
                for item in persisted
            ]
        if persist_sidecars is None:
            from gateway.research_lab.attested_v2_store import (
                persist_execution_sidecars_v2,
            )

            persist_sidecars = persist_execution_sidecars_v2
        sidecar_persistence = await persist_sidecars(
            artifact_receipt_hash=str(lineage_receipt["receipt_hash"]),
            artifacts=artifact_persistence,
            transitions=transitions,
        )
        if replayable_result:
            if persist_replayable_result is None:
                from gateway.research_lab.attested_v2_store import (
                    persist_execution_result_v2,
                )

                persist_replayable_result = persist_execution_result_v2
            await persist_replayable_result(
                operation=operation,
                result=result,
                receipt=receipt,
                artifact_hashes=job_artifact_hashes,
                release_hash=release["release_hash"],
            )
        return {
            "status": "succeeded",
            "result": result,
            "receipt": dict(lineage_receipt),
            "execution_receipt": dict(receipt),
            "execution_receipt_graph": dict(graph),
            "receipt_graph": dict(lineage_graph),
            "transitions": list(transitions),
            "transport_attempts": list(transport_attempts),
            "artifact_persistence": artifact_persistence,
            "artifact_hashes": list(job_artifact_hashes),
            "persistence": dict(lineage["persistence"]),
            "sidecar_persistence": dict(sidecar_persistence),
            "release_hash": release["release_hash"],
            "physical_role": physical_role,
        }
    if allowed_failed:
        persistence = await persist_graph(
            graph,
            allowed_failed_receipt_hashes=allowed_failed,
        )
    else:
        persistence = await persist_graph(graph)
    if persistence.get("root_receipt_hash") != receipt["receipt_hash"]:
        raise AttestedScoringV2Error("V2 scoring durable readback root mismatch")
    sidecar_persistence = {}
    if transitions:
        if persist_sidecars is None:
            from gateway.research_lab.attested_v2_store import (
                persist_execution_sidecars_v2,
            )

            persist_sidecars = persist_execution_sidecars_v2
        sidecar_persistence = await persist_sidecars(
            artifact_receipt_hash=str(receipt["receipt_hash"]),
            artifacts=(),
            transitions=transitions,
        )
    if replayable_result:
        if persist_replayable_result is None:
            from gateway.research_lab.attested_v2_store import (
                persist_execution_result_v2,
            )

            persist_replayable_result = persist_execution_result_v2
        await persist_replayable_result(
            operation=operation,
            result=result,
            receipt=receipt,
            artifact_hashes=job_artifact_hashes,
            release_hash=release["release_hash"],
        )
    return {
        "status": "succeeded",
        "result": result,
        "receipt": dict(receipt),
        "receipt_graph": graph,
        "transitions": list(transitions),
        "transport_attempts": list(transport_attempts),
        "artifact_persistence": artifact_persistence,
        "artifact_hashes": list(job_artifact_hashes),
        "persistence": dict(persistence),
        "sidecar_persistence": dict(sidecar_persistence),
        "release_hash": release["release_hash"],
        "physical_role": physical_role,
    }
