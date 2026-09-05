"""Bounded V2-only enclave job queue with signed receipts and transitions."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import inspect
import json
import queue
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    DIRECT_EGRESS_REF_HASH,
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    RECEIPT_GRAPH_SCHEMA_VERSION,
    ROLE_PURPOSES,
    build_execution_receipt_body,
    build_transition_command_body,
    canonical_json,
    create_signed_execution_receipt,
    create_signed_transition_command,
    host_operation_root,
    merkle_root,
    sha256_bytes,
    transport_root,
    validate_boot_identity,
    validate_receipt_graph,
    validate_receipt_graphs,
    validate_transport_attempt,
)
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION,
    build_compact_ancestry_proof_from_delta_v2,
    build_full_graph_parent_v2,
    issue_ancestry_certificate_v2,
    issue_legacy_ancestry_checkpoint_bootstrap_v2,
    validate_compact_ancestry_proof_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    MAX_REWARD_CHECKPOINTS,
)


JOB_SCHEMA_VERSION = "leadpoet.enclave_execution_job.v2"
PARENT_RECEIPT_GRAPHS_FIELD = "_v2_parent_receipt_graphs"
PARENT_RECEIPT_GRAPH_SET_FIELD = "_v2_parent_receipt_graph_set"
LEGACY_PARENT_RECEIPT_GRAPH_SET_SCHEMA_VERSION = (
    "leadpoet.parent_receipt_graph_set.v2"
)
PARENT_RECEIPT_GRAPH_SET_SCHEMA_VERSION = (
    "leadpoet.parent_receipt_graph_set.v3"
)
PARENT_ANCESTRY_PROOFS_FIELD = "_v2_parent_ancestry_proofs"
MAX_JOB_COUNT = 256
MAX_QUEUED_JOBS = 64
MIN_TERMINAL_EVICTION_AGE_SECONDS = 300
MAX_INPUT_BYTES = 64 * 1024 * 1024
# Allocation authority and its direct weight-publication consumers carry the
# same complete, independently validated receipt ancestry. Uploads remain
# chunked; allow only these exact coordinator operation/purpose pairs to exceed
# the ordinary V2 input and parent-graph bounds without discarding ancestry.
# The complete measured allocation ancestry is transported in one bounded
# logical job.  Keep the larger ceiling scoped to the exact authority
# operation/purpose allowlist below; ordinary V2 jobs remain at 64 MiB.
MAX_ALLOCATION_ANCESTRY_INPUT_BYTES = 256 * 1024 * 1024
_ALLOCATION_FRONTIER_BOOTSTRAP_SCOPE = (
    "allocation_settlement_frontier_bootstrap_v2",
    "research_lab.allocation_settlement_frontier_bootstrap.v2",
)
_ALLOCATION_ANCESTRY_JOB_SCOPES = frozenset(
    {
        (
            "ancestry_checkpoint_bootstrap_v2",
            "research_lab.ancestry_checkpoint_bootstrap.v2",
        ),
        ("research_lab_allocation", "research_lab.allocation.v2"),
        ("attest_artifact_persistence", "leadpoet.artifact_persistence.v2"),
        ("attest_weight_input", "research_lab.allocation.v2"),
        ("attest_weight_input", "research_lab.champion_input.v2"),
        ("attest_weight_input", "research_lab.reimbursement_input.v2"),
        ("attest_weight_input", "research_lab.source_add_reward_input.v2"),
        ("attest_weight_input", "research_lab.anomaly_adjustment_input.v2"),
        ("attest_weight_publication", "gateway.weights.publication.v2"),
        _ALLOCATION_FRONTIER_BOOTSTRAP_SCOPE,
    }
)
MAX_OUTPUT_BYTES = 128 * 1024 * 1024

# Coordinator-owned persistence is intentionally direct-only even while the
# measured provider request uses the scoring worker's assigned proxy.  Bind
# that exception to the exact internal Supabase sidecar namespaces: an
# ordinary scoring request to Supabase must not acquire the service-role route.
_DIRECT_SUPABASE_SIDECAR_NAMESPACES = frozenset(
    {
        "provider-outcome",
        "provider-evidence-cache",
    }
)
MAX_CHUNK_BYTES = 1024 * 1024
MAX_RESULT_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_RESULT_CHUNK_BYTES = 512 * 1024
# Independent finalized/auditor authorities can remain separate after ancestry
# compaction. Keep their object count bounded while the stricter aggregate
# MAX_INPUT_BYTES limit continues to cap the authenticated request body.
MAX_EXTERNAL_RECEIPT_GRAPHS = 128
# Historical allocation bootstrap can contain more independent direct
# authorities than an ordinary scoring job. Keep the larger object bound tied
# to the same exact operation/purpose allowlist as the 256 MiB input exception.
MAX_ALLOCATION_ANCESTRY_AUTHORITIES = 256
MAX_ALLOCATION_FRONTIER_BOOTSTRAP_AUTHORITIES = MAX_REWARD_CHECKPOINTS + 1
# Checkpoint bootstrap accepts two independently bounded authority sets: up to
# 256 complete legacy graphs and up to 256 already-issued resume proofs.  The
# resume proofs are authenticated job inputs, not additional parents of the
# bootstrap session receipt (the selected full-graph roots are those parents).
MAX_CHECKPOINT_BOOTSTRAP_INPUT_AUTHORITIES = (
    MAX_ALLOCATION_ANCESTRY_AUTHORITIES * 2
)
MAX_EXTERNAL_RECEIPT_GRAPH_BYTES = 64 * 1024 * 1024
MAX_EXTERNAL_ANCESTRY_PROOF_BYTES = 4 * 1024 * 1024
TERMINAL_STATES = frozenset({"cancelled", "failed", "succeeded"})
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_RECEIPT_GRAPH_FIELDS = frozenset(
    {
        "schema_version",
        "root_receipt_hash",
        "boot_identities",
        "receipts",
        "transport_attempts",
        "host_operations",
    }
)
_CHECKPOINTED_RECEIPT_GRAPH_FIELDS = _RECEIPT_GRAPH_FIELDS | frozenset(
    {"ancestry_lineage_id", "ancestry_proof"}
)
_RECEIPT_GRAPH_SET_FIELDS = frozenset(
    {
        "schema_version",
        "graphs",
        "boot_identities",
        "receipts",
        "transport_attempts",
        "host_operations",
    }
)
_LEGACY_RECEIPT_GRAPH_DESCRIPTOR_FIELDS = frozenset(
    {
        "root_receipt_hash",
        "boot_identity_hashes",
        "receipt_hashes",
        "transport_attempt_hashes",
        "host_operation_request_hashes",
    }
)
_RECEIPT_GRAPH_DESCRIPTOR_FIELDS = (
    _LEGACY_RECEIPT_GRAPH_DESCRIPTOR_FIELDS | frozenset({"schema_version"})
)
_CHECKPOINTED_RECEIPT_GRAPH_DESCRIPTOR_FIELDS = (
    _RECEIPT_GRAPH_DESCRIPTOR_FIELDS
    | frozenset({"ancestry_lineage_id", "ancestry_proof"})
)


def _execution_failure_code(exc: Exception) -> str:
    """Project a bounded exception type into the signed failure code."""

    return "execution_%s" % type(exc).__name__.lower()[:80]


class ExecutionJobV2Error(RuntimeError):
    """A V2 job is malformed, unmeasured, duplicated, or unavailable."""


@dataclass(frozen=True)
class TransitionSpecV2:
    operation: str
    target: str
    idempotency_key: str
    expected_state_hash: str
    payload_hash: str
    ttl_seconds: int = 300


@dataclass
class ExecutionResultV2:
    output: Mapping[str, Any]
    receipt_output: Optional[Mapping[str, Any]] = None
    transport_attempts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    artifact_hashes: Sequence[str] = field(default_factory=tuple)
    transitions: Sequence[TransitionSpecV2] = field(default_factory=tuple)
    ancestry_checkpoint_bootstrap: bool = False


@dataclass(frozen=True)
class StageReceiptSpecV2:
    purpose: str
    input_root: str
    output_root: str
    artifact_hashes: Sequence[str] = field(default_factory=tuple)


@dataclass
class ExecutionContextV2:
    job_id: str
    purpose: str
    epoch_id: int
    parent_receipt_hashes: tuple = field(default_factory=tuple)
    provider_credential_profile: str = "default"
    provider_credential_ref_hashes: dict = field(default_factory=dict)
    transport_attempts: list = field(default_factory=list)
    artifact_hashes: list = field(default_factory=list)
    stage_receipts: list = field(default_factory=list)
    external_receipt_graphs: list = field(default_factory=list)
    external_receipt_graph_policies: dict = field(default_factory=dict)
    external_ancestry_proofs: list = field(default_factory=list)
    allowed_failed_receipt_hashes: set = field(default_factory=set)
    host_operation_channel: Any = None
    allowed_purposes: frozenset = frozenset()
    max_external_receipt_graph_bytes: int = MAX_EXTERNAL_RECEIPT_GRAPH_BYTES
    max_external_ancestry_authorities: int = MAX_EXTERNAL_RECEIPT_GRAPHS
    _transport_lock: Any = field(
        default_factory=threading.RLock,
        repr=False,
    )
    _transport_frozen: bool = field(default=False, repr=False)
    _frozen_transport_attempts: tuple = field(
        default_factory=tuple,
        repr=False,
    )
    _artifact_lock: Any = field(
        default_factory=threading.RLock,
        repr=False,
    )
    _artifacts_frozen: bool = field(default=False, repr=False)
    _frozen_artifact_hashes: tuple = field(
        default_factory=tuple,
        repr=False,
    )
    _external_receipt_lock: Any = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def record_transport(self, attempt: Mapping[str, Any]) -> None:
        validate_transport_attempt(attempt)
        if attempt["job_id"] != self.job_id or attempt["purpose"] != self.purpose:
            raise ExecutionJobV2Error("transport attempt differs from execution scope")
        provider_id = str(attempt.get("provider_id") or "")
        expected_credential = self.provider_credential_ref_hashes.get(provider_id)
        if (
            expected_credential is not None
            and attempt.get("credential_ref_hash") != expected_credential
        ):
            raise ExecutionJobV2Error(
                "transport credential differs from the attested job profile "
                "for provider %s (expected=%s observed=%s)"
                % (
                    str(attempt.get("provider_id") or ""),
                    str(expected_credential)[:15],
                    str(attempt.get("credential_ref_hash") or "")[:15],
                )
            )
        expected_proxy = self.provider_credential_ref_hashes.get("egress_proxy")
        logical_operation_id = str(attempt.get("logical_operation_id") or "")
        direct_supabase_sidecar = (
            provider_id == "supabase"
            and any(
                logical_operation_id.startswith(
                    "%s:%s:" % (self.job_id, namespace)
                )
                for namespace in _DIRECT_SUPABASE_SIDECAR_NAMESPACES
            )
        )
        expected_transport_proxy = (
            DIRECT_EGRESS_REF_HASH if direct_supabase_sidecar else expected_proxy
        )
        if (
            expected_transport_proxy is not None
            and attempt.get("egress_proxy_ref_hash") != expected_transport_proxy
        ):
            raise ExecutionJobV2Error(
                "transport proxy differs from the attested job profile "
                "(expected=%s observed=%s)"
                % (
                    str(expected_transport_proxy)[:15],
                    str(attempt.get("egress_proxy_ref_hash") or "")[:15],
                )
            )
        with self._transport_lock:
            if self._transport_frozen:
                raise ExecutionJobV2Error(
                    "transport attempt arrived after execution was finalized"
                )
            if any(
                item["attempt_hash"] == attempt["attempt_hash"]
                for item in self.transport_attempts
            ):
                raise ExecutionJobV2Error("transport attempt is duplicated")
            self.transport_attempts.append(dict(attempt))

    def freeze_transport_attempts(self) -> tuple[dict[str, Any], ...]:
        """Return the one immutable terminal-attempt snapshot for this job."""

        with self._transport_lock:
            if not self._transport_frozen:
                self._frozen_transport_attempts = tuple(
                    json.loads(_canonical_bytes(item).decode("utf-8"))
                    for item in self.transport_attempts
                )
                self._transport_frozen = True
            return tuple(dict(item) for item in self._frozen_transport_attempts)

    def record_artifact(self, artifact_hash: str) -> None:
        digest = str(artifact_hash or "").lower()
        if not _HASH_RE.fullmatch(digest):
            raise ExecutionJobV2Error("execution artifact hash is invalid")
        with self._artifact_lock:
            if self._artifacts_frozen:
                raise ExecutionJobV2Error(
                    "execution artifact arrived after execution was finalized"
                )
            if digest not in self.artifact_hashes:
                self.artifact_hashes.append(digest)

    def freeze_artifact_hashes(self) -> tuple[str, ...]:
        """Return the one immutable artifact commitment snapshot for this job."""

        with self._artifact_lock:
            if not self._artifacts_frozen:
                self._frozen_artifact_hashes = tuple(self.artifact_hashes)
                self._artifacts_frozen = True
            return tuple(self._frozen_artifact_hashes)

    def record_external_receipt_graph(
        self,
        graph: Mapping[str, Any],
        *,
        allowed_failed_receipt_hashes: Iterable[str] = (),
    ) -> str:
        """Bind a validated nested enclave execution into this job's ancestry."""
        allowed_failed = {
            _hash(value, "allowed failed receipt hash")
            for value in allowed_failed_receipt_hashes
        }
        validate_receipt_graph(
            graph,
            allowed_failed_receipt_hashes=allowed_failed,
        )
        encoded = _canonical_bytes(graph)
        if len(encoded) > self.max_external_receipt_graph_bytes:
            raise ExecutionJobV2Error("external receipt graph exceeds size limit")
        normalized = json.loads(encoded.decode("utf-8"))
        root_hash = _hash(
            normalized.get("root_receipt_hash"),
            "external root receipt hash",
        )
        with self._external_receipt_lock:
            proof_roots = {
                str(item["certificate"]["claim"]["output_root_receipt_hash"])
                for item in self.external_ancestry_proofs
            }
            if root_hash in proof_roots:
                raise ExecutionJobV2Error(
                    "external ancestry root is supplied as graph and proof"
                )
            existing = {
                str(item["root_receipt_hash"]): item
                for item in self.external_receipt_graphs
            }
            if root_hash in existing:
                if existing[root_hash] != normalized:
                    raise ExecutionJobV2Error(
                        "external receipt graph conflicts with existing root"
                    )
                if self.external_receipt_graph_policies.get(root_hash) != tuple(
                    sorted(allowed_failed)
                ):
                    raise ExecutionJobV2Error(
                        "external receipt graph policy conflicts with existing root"
                    )
                return root_hash
            if (
                len(self.external_receipt_graphs)
                + len(self.external_ancestry_proofs)
                >= self.max_external_ancestry_authorities
            ):
                raise ExecutionJobV2Error("external receipt graph count exceeds limit")
            self.external_receipt_graphs.append(normalized)
            self.external_receipt_graph_policies[root_hash] = tuple(
                sorted(allowed_failed)
            )
            self.allowed_failed_receipt_hashes.update(allowed_failed)
        return root_hash

    def record_external_receipt_graphs(
        self,
        graphs: Sequence[Mapping[str, Any]],
        *,
        allowed_failed_receipt_hashes_by_graph: Optional[
            Sequence[Iterable[str]]
        ] = None,
        _encoded_sizes: Optional[Sequence[int]] = None,
        _share_objects: bool = False,
        boot_attestation_verifier: Optional[
            Callable[[Mapping[str, Any]], Any]
        ] = None,
        require_boot_attestation_verification: bool = False,
    ) -> tuple:
        """Bind overlapping graphs after one fail-closed batch verification."""

        graph_list = list(graphs)
        if allowed_failed_receipt_hashes_by_graph is None:
            failed_by_graph = [()] * len(graph_list)
        else:
            failed_by_graph = list(allowed_failed_receipt_hashes_by_graph)
            if len(failed_by_graph) != len(graph_list):
                raise ExecutionJobV2Error(
                    "external receipt graph failure policies differ from graph count"
                )
        if _encoded_sizes is not None and len(_encoded_sizes) != len(graph_list):
            raise ExecutionJobV2Error(
                "external receipt graph sizes differ from graph count"
            )
        validate_receipt_graphs(
            graph_list,
            allowed_failed_receipt_hashes_by_graph=failed_by_graph,
            boot_attestation_verifier=boot_attestation_verifier,
            require_boot_attestation_verification=(
                require_boot_attestation_verification
            ),
        )

        normalized_graphs = []
        allowed_failed_sets = []
        roots = []
        for index, (graph, allowed_failed_values) in enumerate(
            zip(graph_list, failed_by_graph)
        ):
            encoded = None if _encoded_sizes is not None else _canonical_bytes(graph)
            encoded_size = (
                int(_encoded_sizes[index])
                if _encoded_sizes is not None
                else len(encoded or b"")
            )
            if encoded_size > self.max_external_receipt_graph_bytes:
                raise ExecutionJobV2Error(
                    "external receipt graph exceeds size limit"
                )
            normalized = (
                dict(graph)
                if _share_objects
                else json.loads((encoded or _canonical_bytes(graph)).decode("utf-8"))
            )
            root_hash = _hash(
                normalized.get("root_receipt_hash"),
                "external root receipt hash",
            )
            if root_hash in roots:
                raise ExecutionJobV2Error(
                    "external receipt graph is duplicated"
                )
            roots.append(root_hash)
            normalized_graphs.append(normalized)
            allowed_failed_sets.append(
                {
                    _hash(value, "allowed failed receipt hash")
                    for value in allowed_failed_values
                }
            )

        with self._external_receipt_lock:
            existing = {
                str(item["root_receipt_hash"]): item
                for item in self.external_receipt_graphs
            }
            proof_roots = {
                str(item["certificate"]["claim"]["output_root_receipt_hash"])
                for item in self.external_ancestry_proofs
            }
            for root_hash, normalized, allowed_failed in zip(
                roots, normalized_graphs, allowed_failed_sets
            ):
                if root_hash in proof_roots:
                    raise ExecutionJobV2Error(
                        "external ancestry root is supplied as graph and proof"
                    )
                if root_hash in existing and existing[root_hash] != normalized:
                    raise ExecutionJobV2Error(
                        "external receipt graph conflicts with existing root"
                    )
                previous_policy = self.external_receipt_graph_policies.get(
                    root_hash
                )
                if (
                    root_hash in existing
                    and previous_policy != tuple(sorted(allowed_failed))
                ):
                    raise ExecutionJobV2Error(
                        "external receipt graph policy conflicts with existing root"
                    )
            new_graph_count = sum(root not in existing for root in roots)
            if (
                len(self.external_receipt_graphs)
                + len(self.external_ancestry_proofs)
                + new_graph_count
                > self.max_external_ancestry_authorities
            ):
                raise ExecutionJobV2Error(
                    "external receipt graph count exceeds limit"
                )
            for root_hash, normalized, allowed_failed in zip(
                roots,
                normalized_graphs,
                allowed_failed_sets,
            ):
                if root_hash not in existing:
                    self.external_receipt_graphs.append(normalized)
                    existing[root_hash] = normalized
                    self.external_receipt_graph_policies[root_hash] = tuple(
                        sorted(allowed_failed)
                    )
                self.allowed_failed_receipt_hashes.update(allowed_failed)
        return tuple(roots)

    def external_receipt_roots(self) -> tuple:
        with self._external_receipt_lock:
            return tuple(
                sorted(
                    str(item["root_receipt_hash"])
                    for item in self.external_receipt_graphs
                )
            )

    def record_external_ancestry_proof(
        self,
        proof: Mapping[str, Any],
        *,
        expected_lineage_id: str,
        boot_attestation_verifier: Callable[[Mapping[str, Any]], Any],
        allowed_issuer_roles: Iterable[str],
        required_receipt_hashes: Iterable[str] = (),
    ) -> str:
        """Bind one bounded, recursively attested parent authority."""

        normalized = validate_compact_ancestry_proof_v2(
            proof,
            expected_lineage_id=expected_lineage_id,
            boot_attestation_verifier=boot_attestation_verifier,
            allowed_issuer_roles=allowed_issuer_roles,
            required_receipt_hashes=required_receipt_hashes,
        )
        encoded = _canonical_bytes(normalized)
        if len(encoded) > MAX_EXTERNAL_ANCESTRY_PROOF_BYTES:
            raise ExecutionJobV2Error("external ancestry proof exceeds size limit")
        normalized = json.loads(encoded.decode("utf-8"))
        root_hash = _hash(
            normalized["certificate"]["claim"]["output_root_receipt_hash"],
            "external ancestry root receipt hash",
        )
        with self._external_receipt_lock:
            existing_graph_roots = {
                str(item["root_receipt_hash"])
                for item in self.external_receipt_graphs
            }
            if root_hash in existing_graph_roots:
                raise ExecutionJobV2Error(
                    "external ancestry root is supplied as graph and proof"
                )
            existing = {
                str(item["certificate"]["claim"]["output_root_receipt_hash"]): item
                for item in self.external_ancestry_proofs
            }
            if root_hash in existing:
                if existing[root_hash] != normalized:
                    raise ExecutionJobV2Error(
                        "external ancestry proof conflicts with existing root"
                    )
                return root_hash
            if (
                len(self.external_receipt_graphs)
                + len(self.external_ancestry_proofs)
                >= self.max_external_ancestry_authorities
            ):
                raise ExecutionJobV2Error(
                    "external ancestry authority count exceeds limit"
                )
            self.external_ancestry_proofs.append(normalized)
        return root_hash

    def external_ancestry_roots(self) -> tuple:
        with self._external_receipt_lock:
            return tuple(
                sorted(
                    str(item["certificate"]["claim"]["output_root_receipt_hash"])
                    for item in self.external_ancestry_proofs
                )
            )

    def external_receipt_authority_graphs(self) -> tuple[dict[str, Any], ...]:
        """Expose full and compact parent authorities through one graph view.

        Compact proofs are validated before they enter this context. Their
        disclosed receipts and boot identities are the bounded business view
        needed by measured executors after graph transport is checkpointed.
        """

        with self._external_receipt_lock:
            graphs = [
                json.loads(_canonical_bytes(item).decode("utf-8"))
                for item in self.external_receipt_graphs
            ]
            roots = {
                _hash(item.get("root_receipt_hash"), "external root receipt hash")
                for item in graphs
            }
            for proof in self.external_ancestry_proofs:
                normalized = json.loads(_canonical_bytes(proof).decode("utf-8"))
                try:
                    claim = normalized["certificate"]["claim"]
                    lineage_id = str(claim["lineage_id"])
                    root_hash = _hash(
                        claim["output_root_receipt_hash"],
                        "external ancestry root receipt hash",
                    )
                    if not isinstance(normalized["disclosed_receipts"], list):
                        raise TypeError("disclosed receipts are not a list")
                    if not isinstance(
                        normalized["disclosed_boot_identities"], list
                    ):
                        raise TypeError("disclosed boot identities are not a list")
                    receipts = [
                        dict(item)
                        for item in normalized["disclosed_receipts"]
                        if isinstance(item, Mapping)
                    ]
                    boot_identities = [
                        dict(item)
                        for item in normalized["disclosed_boot_identities"]
                        if isinstance(item, Mapping)
                    ]
                    if len(receipts) != len(normalized["disclosed_receipts"]):
                        raise TypeError("disclosed receipt is not an object")
                    if len(boot_identities) != len(
                        normalized["disclosed_boot_identities"]
                    ):
                        raise TypeError("disclosed boot identity is not an object")
                except (KeyError, TypeError) as exc:
                    raise ExecutionJobV2Error(
                        "external ancestry proof disclosure is invalid"
                    ) from exc
                if root_hash in roots or sum(
                    str(item.get("receipt_hash") or "") == root_hash
                    for item in receipts
                ) != 1:
                    raise ExecutionJobV2Error(
                        "external ancestry proof root disclosure is invalid"
                    )
                roots.add(root_hash)
                graphs.append(
                    {
                        "schema_version": (
                            COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
                        ),
                        "root_receipt_hash": root_hash,
                        "boot_identities": boot_identities,
                        "receipts": receipts,
                        "transport_attempts": [],
                        "host_operations": [],
                        "ancestry_lineage_id": lineage_id,
                        "ancestry_proof": normalized,
                    }
                )
            return tuple(graphs)

    def execute_host_operation(
        self,
        *,
        operation: str,
        payload: Mapping[str, Any],
        expected_state_hash: str,
        timeout_seconds: int,
        response_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if self.host_operation_channel is None:
            raise ExecutionJobV2Error("host operations are unavailable for this role")
        return self.host_operation_channel.execute(
            operation=operation,
            payload=payload,
            expected_state_hash=expected_state_hash,
            timeout_seconds=timeout_seconds,
            response_validator=response_validator,
        )

    def host_operation_records(self) -> Sequence[Mapping[str, Any]]:
        if self.host_operation_channel is None:
            return ()
        return self.host_operation_channel.complete_ledger()

    def record_stage(
        self,
        *,
        purpose: str,
        input_root: str,
        output_root: str,
        artifact_hashes: Sequence[str] = (),
    ) -> None:
        normalized_purpose = str(purpose or "")
        if normalized_purpose not in self.allowed_purposes:
            raise ExecutionJobV2Error("stage receipt purpose is not authorized")
        normalized_artifacts = tuple(
            _hash(item, "stage artifact hash") for item in artifact_hashes
        )
        self.stage_receipts.append(
            StageReceiptSpecV2(
                purpose=normalized_purpose,
                input_root=_hash(input_root, "stage input root"),
                output_root=_hash(output_root, "stage output root"),
                artifact_hashes=normalized_artifacts,
            )
        )


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise ExecutionJobV2Error("%s is invalid" % field)
    return normalized


def _identifier(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ExecutionJobV2Error("%s is invalid" % field)
    return normalized


def _normalized_transport_object(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionJobV2Error("%s is not an object" % field)
    try:
        normalized = json.loads(_canonical_bytes(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ExecutionJobV2Error("%s is not canonical JSON" % field) from exc
    if not isinstance(normalized, dict):
        raise ExecutionJobV2Error("%s is not an object" % field)
    return normalized


def _host_operation_request_hash(value: Mapping[str, Any]) -> str:
    request = value.get("request")
    if not isinstance(request, Mapping):
        raise ExecutionJobV2Error("host operation request is missing")
    return _hash(request.get("request_hash"), "host operation request hash")


def pack_parent_receipt_graph_set_v2(
    graphs: Sequence[Mapping[str, Any]],
    *,
    max_graph_count: int = MAX_EXTERNAL_RECEIPT_GRAPHS,
) -> Dict[str, Any]:
    """Deduplicate graph objects without changing any graph membership."""

    if not isinstance(graphs, Sequence) or isinstance(
        graphs, (str, bytes, bytearray)
    ):
        raise ExecutionJobV2Error("parent receipt graphs must be an array")
    max_graph_count = _bounded_external_authority_limit(max_graph_count)
    if len(graphs) > max_graph_count:
        raise ExecutionJobV2Error("external receipt graph count exceeds limit")

    collections: Dict[str, list[Dict[str, Any]]] = {
        "boot_identities": [],
        "receipts": [],
        "transport_attempts": [],
        "host_operations": [],
    }
    indexes: Dict[str, Dict[str, Dict[str, Any]]] = {
        field: {} for field in collections
    }
    key_fields = {
        "boot_identities": "boot_identity_hash",
        "receipts": "receipt_hash",
        "transport_attempts": "attempt_hash",
    }

    def add_object(collection: str, value: Any) -> str:
        if not isinstance(value, Mapping):
            raise ExecutionJobV2Error(
                "%s is not an object"
                % collection.replace("_", " ").rstrip("s")
            )
        if collection == "host_operations":
            object_hash = _host_operation_request_hash(value)
        else:
            key_field = key_fields[collection]
            object_hash = _hash(value.get(key_field), key_field)
        previous = indexes[collection].get(object_hash)
        if previous is not None:
            if previous == dict(value):
                return object_hash
            normalized = _normalized_transport_object(
                value,
                collection.replace("_", " ").rstrip("s"),
            )
            if previous != normalized:
                raise ExecutionJobV2Error(
                    "%s conflicts for hash" % collection.replace("_", " ")
                )
            return object_hash
        normalized = _normalized_transport_object(
            value,
            collection.replace("_", " ").rstrip("s"),
        )
        indexes[collection][object_hash] = normalized
        collections[collection].append(normalized)
        return object_hash

    descriptors: list[Dict[str, Any]] = []
    roots: set[str] = set()
    for graph in graphs:
        if not isinstance(graph, Mapping):
            raise ExecutionJobV2Error("parent receipt graph is not an object")
        normalized_graph = dict(graph)
        graph_schema = normalized_graph.get("schema_version")
        expected_fields = (
            _RECEIPT_GRAPH_FIELDS
            if graph_schema == RECEIPT_GRAPH_SCHEMA_VERSION
            else _CHECKPOINTED_RECEIPT_GRAPH_FIELDS
            if graph_schema == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
            else None
        )
        if expected_fields is None:
            raise ExecutionJobV2Error("parent receipt graph schema is invalid")
        if set(normalized_graph) != expected_fields:
            raise ExecutionJobV2Error("parent receipt graph fields are invalid")
        root = _hash(
            normalized_graph.get("root_receipt_hash"),
            "parent receipt graph root",
        )
        if root in roots:
            raise ExecutionJobV2Error("parent receipt graph is duplicated")
        roots.add(root)
        for field in collections:
            if not isinstance(normalized_graph.get(field), list):
                raise ExecutionJobV2Error(
                    "parent receipt graph %s must be an array" % field
                )
        descriptor = {
            "schema_version": graph_schema,
            "root_receipt_hash": root,
            "boot_identity_hashes": [
                add_object("boot_identities", item)
                for item in normalized_graph["boot_identities"]
            ],
            "receipt_hashes": [
                add_object("receipts", item)
                for item in normalized_graph["receipts"]
            ],
            "transport_attempt_hashes": [
                add_object("transport_attempts", item)
                for item in normalized_graph["transport_attempts"]
            ],
            "host_operation_request_hashes": [
                add_object("host_operations", item)
                for item in normalized_graph["host_operations"]
            ],
        }
        if graph_schema == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION:
            descriptor.update(
                {
                    "ancestry_lineage_id": _hash(
                        normalized_graph.get("ancestry_lineage_id"),
                        "parent receipt graph ancestry lineage",
                    ),
                    "ancestry_proof": _normalized_transport_object(
                        normalized_graph.get("ancestry_proof"),
                        "parent receipt graph ancestry proof",
                    ),
                }
            )
        descriptors.append(descriptor)
    return {
        "schema_version": PARENT_RECEIPT_GRAPH_SET_SCHEMA_VERSION,
        "graphs": descriptors,
        **collections,
    }


def _unpack_parent_receipt_graph_set_v2(
    value: Mapping[str, Any],
    *,
    max_graph_count: int = MAX_EXTERNAL_RECEIPT_GRAPHS,
) -> tuple[list[Dict[str, Any]], list[int]]:
    """Reconstruct exact graph memberships from a bounded deduplicated set."""

    if not isinstance(value, Mapping) or set(value) != _RECEIPT_GRAPH_SET_FIELDS:
        raise ExecutionJobV2Error("parent receipt graph set fields are invalid")
    graph_set_schema = value.get("schema_version")
    if graph_set_schema not in {
        LEGACY_PARENT_RECEIPT_GRAPH_SET_SCHEMA_VERSION,
        PARENT_RECEIPT_GRAPH_SET_SCHEMA_VERSION,
    }:
        raise ExecutionJobV2Error("parent receipt graph set schema is invalid")
    max_graph_count = _bounded_external_authority_limit(max_graph_count)
    descriptors = value.get("graphs")
    if (
        not isinstance(descriptors, list)
        or len(descriptors) > max_graph_count
    ):
        raise ExecutionJobV2Error("parent receipt graph set count is invalid")

    collection_specs = {
        "boot_identities": ("boot_identity_hash", None),
        "receipts": ("receipt_hash", None),
        "transport_attempts": ("attempt_hash", None),
        "host_operations": (None, _host_operation_request_hash),
    }
    indexes: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for field, (key_field, key_loader) in collection_specs.items():
        rows = value.get(field)
        if not isinstance(rows, list):
            raise ExecutionJobV2Error(
                "parent receipt graph set %s must be an array" % field
            )
        index: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            normalized = _normalized_transport_object(
                row,
                field.replace("_", " ").rstrip("s"),
            )
            object_hash = (
                key_loader(normalized)
                if key_loader is not None
                else _hash(normalized.get(key_field), str(key_field))
            )
            if object_hash in index:
                raise ExecutionJobV2Error(
                    "parent receipt graph set %s is duplicated" % field
                )
            index[object_hash] = normalized
        indexes[field] = index

    descriptor_fields = {
        "boot_identity_hashes": "boot_identities",
        "receipt_hashes": "receipts",
        "transport_attempt_hashes": "transport_attempts",
        "host_operation_request_hashes": "host_operations",
    }
    used = {field: set() for field in collection_specs}
    roots: set[str] = set()
    graphs: list[Dict[str, Any]] = []
    graph_sizes: list[int] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise ExecutionJobV2Error(
                "parent receipt graph descriptor fields are invalid"
            )
        if graph_set_schema == LEGACY_PARENT_RECEIPT_GRAPH_SET_SCHEMA_VERSION:
            graph_schema = RECEIPT_GRAPH_SCHEMA_VERSION
            expected_descriptor_fields = (
                _LEGACY_RECEIPT_GRAPH_DESCRIPTOR_FIELDS
            )
        else:
            graph_schema = descriptor.get("schema_version")
            expected_descriptor_fields = (
                _RECEIPT_GRAPH_DESCRIPTOR_FIELDS
                if graph_schema == RECEIPT_GRAPH_SCHEMA_VERSION
                else _CHECKPOINTED_RECEIPT_GRAPH_DESCRIPTOR_FIELDS
                if graph_schema == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
                else None
            )
        if (
            expected_descriptor_fields is None
            or set(descriptor) != expected_descriptor_fields
        ):
            raise ExecutionJobV2Error(
                "parent receipt graph descriptor fields are invalid"
            )
        root = _hash(
            descriptor.get("root_receipt_hash"),
            "parent receipt graph descriptor root",
        )
        if root in roots:
            raise ExecutionJobV2Error(
                "parent receipt graph descriptor is duplicated"
            )
        roots.add(root)
        graph: Dict[str, Any] = {
            "schema_version": graph_schema,
            "root_receipt_hash": root,
        }
        for descriptor_field, collection in descriptor_fields.items():
            hashes = descriptor.get(descriptor_field)
            if not isinstance(hashes, list):
                raise ExecutionJobV2Error(
                    "parent receipt graph descriptor %s must be an array"
                    % descriptor_field
                )
            normalized_hashes = [
                _hash(item, descriptor_field) for item in hashes
            ]
            missing = [
                item for item in normalized_hashes if item not in indexes[collection]
            ]
            if missing:
                raise ExecutionJobV2Error(
                    "parent receipt graph descriptor reference is missing"
                )
            graph[collection] = [indexes[collection][item] for item in normalized_hashes]
            used[collection].update(normalized_hashes)
        if graph_schema == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION:
            graph.update(
                {
                    "ancestry_lineage_id": _hash(
                        descriptor.get("ancestry_lineage_id"),
                        "parent receipt graph descriptor ancestry lineage",
                    ),
                    "ancestry_proof": _normalized_transport_object(
                        descriptor.get("ancestry_proof"),
                        "parent receipt graph descriptor ancestry proof",
                    ),
                }
            )
        graphs.append(graph)
        graph_sizes.append(len(_canonical_bytes(graph)))

    if any(set(indexes[field]) != used[field] for field in collection_specs):
        raise ExecutionJobV2Error(
            "parent receipt graph set contains unreferenced evidence"
        )
    return graphs, graph_sizes


def unpack_parent_receipt_graph_set_v2(
    value: Mapping[str, Any],
    *,
    max_graph_count: int = MAX_EXTERNAL_RECEIPT_GRAPHS,
) -> list[Dict[str, Any]]:
    graphs, _ = _unpack_parent_receipt_graph_set_v2(
        value,
        max_graph_count=max_graph_count,
    )
    return graphs


def _bounded_external_authority_limit(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value
        > max(
            MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
            MAX_ALLOCATION_FRONTIER_BOOTSTRAP_AUTHORITIES,
        )
    ):
        raise ExecutionJobV2Error("external ancestry authority limit is invalid")
    return value


def _job_input_limit_bytes(*, operation: str, purpose: str) -> int:
    if (operation, purpose) in _ALLOCATION_ANCESTRY_JOB_SCOPES:
        return MAX_ALLOCATION_ANCESTRY_INPUT_BYTES
    return MAX_INPUT_BYTES


def _job_external_authority_limit(*, operation: str, purpose: str) -> int:
    if (
        operation == "ancestry_checkpoint_bootstrap_v2"
        and purpose == "research_lab.ancestry_checkpoint_bootstrap.v2"
    ):
        return MAX_CHECKPOINT_BOOTSTRAP_INPUT_AUTHORITIES
    if (operation, purpose) == _ALLOCATION_FRONTIER_BOOTSTRAP_SCOPE:
        return MAX_ALLOCATION_FRONTIER_BOOTSTRAP_AUTHORITIES
    if (operation, purpose) in _ALLOCATION_ANCESTRY_JOB_SCOPES:
        return MAX_ALLOCATION_ANCESTRY_AUTHORITIES
    return MAX_EXTERNAL_RECEIPT_GRAPHS


def _manifest(
    value: Mapping[str, Any],
    *,
    role: str,
    operations: Mapping[str, Iterable[str]],
) -> Dict[str, Any]:
    required = {
        "schema_version",
        "job_id",
        "operation",
        "purpose",
        "epoch_id",
        "sequence",
        "payload_sha256",
        "payload_size_bytes",
        "parent_receipt_hashes",
        "input_artifact_hashes",
        "provider_credential_profile",
        "provider_credential_ref_hashes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ExecutionJobV2Error("V2 job manifest fields are invalid")
    if value["schema_version"] != JOB_SCHEMA_VERSION:
        raise ExecutionJobV2Error("V2 job manifest schema is invalid")
    operation = _identifier(value["operation"], "operation")
    purpose = _identifier(value["purpose"], "purpose")
    allowed = set(operations.get(operation, ()))
    if purpose not in allowed or purpose not in ROLE_PURPOSES.get(role, ()):
        raise ExecutionJobV2Error("operation purpose is not authorized for role")
    epoch_id = value["epoch_id"]
    sequence = value["sequence"]
    size = value["payload_size_bytes"]
    if not isinstance(epoch_id, int) or epoch_id < 0:
        raise ExecutionJobV2Error("epoch_id must be non-negative")
    if not isinstance(sequence, int) or sequence < 0:
        raise ExecutionJobV2Error("sequence must be non-negative")
    max_input_bytes = _job_input_limit_bytes(
        operation=operation,
        purpose=purpose,
    )
    if not isinstance(size, int) or size < 2 or size > max_input_bytes:
        raise ExecutionJobV2Error("payload size is outside limit")
    parents = value["parent_receipt_hashes"]
    artifacts = value["input_artifact_hashes"]
    provider_credentials = value["provider_credential_ref_hashes"]
    provider_profile = _identifier(
        value["provider_credential_profile"],
        "provider credential profile",
    )
    if not isinstance(parents, list) or not isinstance(artifacts, list):
        raise ExecutionJobV2Error("job receipt/artifact roots must be arrays")
    if not isinstance(provider_credentials, Mapping):
        raise ExecutionJobV2Error("job provider credential references must be an object")
    normalized_provider_credentials = {}
    for provider_id, digest in provider_credentials.items():
        normalized_provider_credentials[
            _identifier(provider_id, "provider credential provider_id")
        ] = _hash(digest, "provider credential reference")
    normalized_parents = sorted(
        {_hash(item, "parent_receipt_hash") for item in parents}
    )
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": _identifier(value["job_id"], "job_id"),
        "operation": operation,
        "purpose": purpose,
        "epoch_id": epoch_id,
        "sequence": sequence,
        "payload_sha256": _hash(value["payload_sha256"], "payload_sha256"),
        "payload_size_bytes": size,
        "parent_receipt_hashes": normalized_parents,
        "input_artifact_hashes": sorted(
            {_hash(item, "input_artifact_hash") for item in artifacts}
        ),
        "provider_credential_profile": provider_profile,
        "provider_credential_ref_hashes": dict(
            sorted(normalized_provider_credentials.items())
        ),
    }


def _utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class ExecutionJobManagerV2:
    def __init__(
        self,
        *,
        boot_identity_supplier: Callable[[], Mapping[str, Any]],
        sign_digest: Callable[[bytes], Any],
        operations: Mapping[str, Iterable[str]],
        executor: Callable[[str, Mapping[str, Any], ExecutionContextV2], Any],
        worker_count: int,
        configured_worker_count: Optional[int] = None,
        host_operation_channel_factory: Optional[
            Callable[[str, str], Any]
        ] = None,
        failed_parent_graph_policy: Optional[
            Callable[
                [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
                Iterable[str],
            ]
        ] = None,
        ancestry_lineage_id: Optional[str] = None,
        ancestry_boot_attestation_verifier: Optional[
            Callable[[Mapping[str, Any]], Any]
        ] = None,
        ancestry_allowed_issuer_roles: Iterable[str] = (),
        retention_seconds: int = 3600,
        clock: Callable[[], float] = time.time,
    ) -> None:
        boot = dict(boot_identity_supplier())
        validate_boot_identity(boot)
        self.boot_identity = boot
        self.role = str(boot["role"])
        self._boot_identity_supplier = boot_identity_supplier
        self._sign_digest = sign_digest
        self._operations = {
            str(operation): frozenset(str(item) for item in purposes)
            for operation, purposes in operations.items()
        }
        self._executor = executor
        self._host_operation_channel_factory = host_operation_channel_factory
        self._failed_parent_graph_policy = failed_parent_graph_policy
        ancestry_values = (
            ancestry_lineage_id,
            ancestry_boot_attestation_verifier,
            tuple(ancestry_allowed_issuer_roles),
        )
        if any(ancestry_values) and not all(ancestry_values):
            raise ValueError("ancestry checkpoint configuration is incomplete")
        self._ancestry_lineage_id = (
            _hash(ancestry_lineage_id, "ancestry lineage id")
            if ancestry_lineage_id is not None
            else None
        )
        self._ancestry_boot_attestation_verifier = (
            ancestry_boot_attestation_verifier
        )
        self._ancestry_allowed_issuer_roles = frozenset(
            str(role) for role in ancestry_allowed_issuer_roles
        )
        if self._ancestry_allowed_issuer_roles and (
            self.role not in self._ancestry_allowed_issuer_roles
            or any(role not in ROLE_PURPOSES for role in self._ancestry_allowed_issuer_roles)
        ):
            raise ValueError("ancestry checkpoint issuer roles are invalid")
        self._retention_seconds = max(60, int(retention_seconds))
        self._clock = clock
        self._jobs = {}  # type: Dict[str, Dict[str, Any]]
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=MAX_QUEUED_JOBS)
        self._active = set()
        self._workers = []
        self._terminal_eviction_count = 0
        self._configured_worker_count = (
            int(worker_count)
            if configured_worker_count is None
            else int(configured_worker_count)
        )
        if not 0 <= self._configured_worker_count <= 500:
            raise ValueError("configured worker count is invalid")
        for index in range(max(1, int(worker_count))):
            worker = threading.Thread(
                target=self._worker_loop,
                name="enclave-v2-executor-%s" % (index + 1),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def health(self) -> Dict[str, Any]:
        with self._lock:
            self._purge_locked()
            counts = {}
            for job in self._jobs.values():
                counts[job["state"]] = counts.get(job["state"], 0) + 1
        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "authority": "v2_only",
            "role": self.role,
            "physical_role": self.boot_identity["physical_role"],
            "boot_identity_hash": self.boot_identity["boot_identity_hash"],
            "worker_count": len(self._workers),
            "configured_worker_count": self._configured_worker_count,
            "workers_alive": all(worker.is_alive() for worker in self._workers),
            "queue_depth": self._queue.qsize(),
            "active_job_ids": sorted(self._active),
            "job_counts": counts,
            "terminal_eviction_count": self._terminal_eviction_count,
            "supported_operations": sorted(self._operations),
            "ancestry_checkpoints": self._ancestry_lineage_id is not None,
            "ancestry_lineage_id": self._ancestry_lineage_id,
        }

    def submit(self, manifest: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = _manifest(
            manifest,
            role=self.role,
            operations=self._operations,
        )
        manifest_hash = sha256_bytes(_canonical_bytes(normalized))
        now = self._clock()
        with self._lock:
            self._purge_locked()
            existing = self._jobs.get(normalized["job_id"])
            if existing is not None:
                if existing["manifest_hash"] != manifest_hash:
                    raise ExecutionJobV2Error(
                        "job_id already exists with another manifest"
                    )
                return self._summary(existing)
            if len(self._jobs) >= MAX_JOB_COUNT:
                self._evict_oldest_terminal_locked()
            if len(self._jobs) >= MAX_JOB_COUNT:
                raise ExecutionJobV2Error("V2 job capacity is full")
            job = {
                "manifest": normalized,
                "manifest_hash": manifest_hash,
                "state": "uploading",
                "input": bytearray(),
                "result": b"",
                "result_hash": None,
                "receipt": None,
                "receipts": [],
                "transitions": [],
                "transport_attempts": [],
                "artifact_hashes": [],
                "host_operations": [],
                "external_receipt_graphs": [],
                "external_ancestry_proofs": [],
                "ancestry_compact_proof": None,
                "host_operation_channel": None,
                "error_code": None,
                "cancel_requested": False,
                "created_at": now,
                "updated_at": now,
            }
            self._jobs[normalized["job_id"]] = job
            return self._summary(job)

    def put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data_b64: str,
        chunk_sha256: str,
    ) -> Dict[str, Any]:
        job_id = _identifier(job_id, "job_id")
        if not isinstance(offset, int) or offset < 0:
            raise ExecutionJobV2Error("chunk offset is invalid")
        try:
            chunk = base64.b64decode(str(data_b64), validate=True)
        except Exception as exc:
            raise ExecutionJobV2Error("chunk is invalid base64") from exc
        if not chunk or len(chunk) > MAX_CHUNK_BYTES:
            raise ExecutionJobV2Error("chunk size is outside limit")
        if sha256_bytes(chunk) != _hash(chunk_sha256, "chunk_sha256"):
            raise ExecutionJobV2Error("chunk hash mismatch")
        with self._lock:
            job = self._job(job_id)
            if job["state"] != "uploading":
                raise ExecutionJobV2Error("job does not accept chunks")
            uploaded = len(job["input"])
            if offset < uploaded:
                end = offset + len(chunk)
                if end <= uploaded and bytes(job["input"][offset:end]) == chunk:
                    return self._summary(job)
                raise ExecutionJobV2Error("chunk conflicts with uploaded payload")
            if offset != uploaded:
                raise ExecutionJobV2Error("chunk offset differs from uploaded length")
            if offset + len(chunk) > job["manifest"]["payload_size_bytes"]:
                raise ExecutionJobV2Error("chunk exceeds declared payload")
            job["input"].extend(chunk)
            job["updated_at"] = self._clock()
            return self._summary(job)

    def seal(self, job_id: str) -> Dict[str, Any]:
        job_id = _identifier(job_id, "job_id")
        with self._lock:
            job = self._job(job_id)
            if job["state"] in {"queued", "running"} | TERMINAL_STATES:
                return self._summary(job)
            payload = bytes(job["input"])
            if len(payload) != job["manifest"]["payload_size_bytes"]:
                raise ExecutionJobV2Error("payload size differs from manifest")
            if sha256_bytes(payload) != job["manifest"]["payload_sha256"]:
                raise ExecutionJobV2Error("payload hash differs from manifest")
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except Exception as exc:
                raise ExecutionJobV2Error("payload must be canonical UTF-8 JSON") from exc
            if not isinstance(decoded, Mapping) or _canonical_bytes(decoded) != payload:
                raise ExecutionJobV2Error("payload must be a canonical JSON object")
            try:
                self._queue.put_nowait(job_id)
            except queue.Full as exc:
                raise ExecutionJobV2Error("V2 execution queue is full") from exc
            job["state"] = "queued"
            job["updated_at"] = self._clock()
            return self._summary(job)

    def status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._summary(self._job(_identifier(job_id, "job_id")))

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] in TERMINAL_STATES:
                return self._summary(job)
            job["cancel_requested"] = True
            channel = job.get("host_operation_channel")
            if channel is not None:
                channel.close(failure_code="cancelled")
            if job["state"] in {"uploading", "queued"}:
                job["state"] = "cancelled"
                job["input"] = bytearray()
            job["updated_at"] = self._clock()
            return self._summary(job)

    def result_chunk(
        self,
        *,
        job_id: str,
        offset: int = 0,
        max_bytes: int = DEFAULT_RESULT_CHUNK_BYTES,
    ) -> Dict[str, Any]:
        if not isinstance(offset, int) or offset < 0:
            raise ExecutionJobV2Error("result offset is invalid")
        if not isinstance(max_bytes, int) or not 1 <= max_bytes <= MAX_RESULT_CHUNK_BYTES:
            raise ExecutionJobV2Error("result chunk size is outside limit")
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in {"succeeded", "failed", "cancelled"} or not job[
                "result_hash"
            ]:
                raise ExecutionJobV2Error("job result is unavailable")
            result = job["result"]
            if offset > len(result):
                raise ExecutionJobV2Error("result offset exceeds size")
            chunk = result[offset : offset + max_bytes]
            return {
                "job_id": job_id,
                "offset": offset,
                "data_b64": base64.b64encode(chunk).decode("ascii"),
                "chunk_sha256": sha256_bytes(chunk),
                "result_sha256": job["result_hash"],
                "total_size_bytes": len(result),
                "eof": offset + len(chunk) >= len(result),
            }

    def receipt(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            receipt = self._job(_identifier(job_id, "job_id"))["receipt"]
            if receipt is None:
                raise ExecutionJobV2Error("job receipt is unavailable")
            return dict(receipt)

    def receipts(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["receipt"] is None:
                raise ExecutionJobV2Error("job receipts are unavailable")
            return tuple(dict(item) for item in job["receipts"])

    def transitions(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] != "succeeded":
                raise ExecutionJobV2Error("job transitions are unavailable")
            return tuple(dict(item) for item in job["transitions"])

    def transport_attempts(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in TERMINAL_STATES:
                raise ExecutionJobV2Error("job transport attempts are unavailable")
            return tuple(dict(item) for item in job["transport_attempts"])

    def artifact_hashes(self, job_id: str) -> Sequence[str]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in TERMINAL_STATES:
                raise ExecutionJobV2Error("job artifact hashes are unavailable")
            return tuple(str(item) for item in job["artifact_hashes"])

    def next_host_operation(
        self, *, job_id: str, wait_ms: int = 0
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            channel = job.get("host_operation_channel")
        if channel is None:
            return None
        return channel.next_command(wait_ms=wait_ms)

    def complete_host_operation(
        self,
        *,
        job_id: str,
        request_hash: str,
        terminal_status: str,
        response: Optional[Mapping[str, Any]],
        failure_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            channel = job.get("host_operation_channel")
        if channel is None:
            raise ExecutionJobV2Error("job has no host operation channel")
        return channel.complete(
            request_hash=request_hash,
            terminal_status=terminal_status,
            response=response,
            failure_code=failure_code,
        )

    def host_operations(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in TERMINAL_STATES:
                raise ExecutionJobV2Error("job host operations are unavailable")
            return tuple(dict(item) for item in job["host_operations"])

    def external_receipt_graphs(self, job_id: str) -> Sequence[Dict[str, Any]]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in TERMINAL_STATES:
                raise ExecutionJobV2Error(
                    "job external receipt graphs are unavailable"
                )
            return tuple(
                json.loads(_canonical_bytes(item).decode("utf-8"))
                for item in job["external_receipt_graphs"]
            )

    def ancestry_compact_proof(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._job(_identifier(job_id, "job_id"))
            if job["state"] not in TERMINAL_STATES:
                raise ExecutionJobV2Error(
                    "job ancestry compact proof is unavailable"
                )
            proof = job.get("ancestry_compact_proof")
            if not isinstance(proof, Mapping):
                raise ExecutionJobV2Error(
                    "job ancestry compact proof is unavailable"
                )
            return json.loads(_canonical_bytes(proof).decode("utf-8"))

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._execute(job_id)
            except Exception as exc:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is not None and job["state"] not in TERMINAL_STATES:
                        job["state"] = "failed"
                        job["error_code"] = "receipt_unavailable"
                        job["input"] = bytearray()
                        job["updated_at"] = self._clock()
                print(
                    "[TEE] V2 execution worker failed closed job_id=%s type=%s"
                    % (job_id, type(exc).__name__),
                    flush=True,
                )
            finally:
                self._queue.task_done()

    def _build_ancestry_compact_proof(
        self,
        *,
        manifest: Mapping[str, Any],
        context: ExecutionContextV2,
        root_receipt: Mapping[str, Any],
        local_receipts: Sequence[Mapping[str, Any]],
        transport_attempts: Sequence[Mapping[str, Any]],
        host_operations: Sequence[Mapping[str, Any]],
        include_external_ancestry_proofs: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if self._ancestry_lineage_id is None:
            return None
        if self._ancestry_boot_attestation_verifier is None:
            raise ExecutionJobV2Error(
                "ancestry checkpoint boot verifier is unavailable"
            )
        local = [dict(item) for item in local_receipts]
        if not local or str(root_receipt.get("receipt_hash") or "") not in {
            str(item.get("receipt_hash") or "") for item in local
        }:
            raise ExecutionJobV2Error(
                "ancestry checkpoint local receipt chain is incomplete"
            )
        local_delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": str(root_receipt["receipt_hash"]),
            "boot_identities": [dict(self.boot_identity)],
            "receipts": local,
            "transport_attempts": [
                dict(item) for item in transport_attempts
            ],
            "host_operations": [dict(item) for item in host_operations],
        }
        parent_certificates = (
            [
                dict(item["certificate"])
                for item in context.external_ancestry_proofs
            ]
            if include_external_ancestry_proofs
            else []
        )
        parent_full_graphs = []
        for graph in context.external_receipt_graphs:
            root_hash = str(graph["root_receipt_hash"])
            if (
                graph.get("schema_version")
                == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
            ):
                proof = validate_compact_ancestry_proof_v2(
                    graph.get("ancestry_proof"),
                    expected_lineage_id=str(self._ancestry_lineage_id),
                    boot_attestation_verifier=(
                        self._ancestry_boot_attestation_verifier
                    ),
                    allowed_issuer_roles=self._ancestry_allowed_issuer_roles,
                    required_receipt_hashes=(root_hash,),
                )
                parent_certificates.append(dict(proof["certificate"]))
                continue
            parent_full_graphs.append(
                build_full_graph_parent_v2(
                    graph,
                    allowed_failed_receipt_hashes=(
                        context.external_receipt_graph_policies.get(
                            root_hash, ()
                        )
                    ),
                )
            )
        parent_sequences = [
            int(item["claim"]["certificate_sequence"])
            for item in parent_certificates
        ]
        certificate = issue_ancestry_certificate_v2(
            local_delta=local_delta,
            lineage_id=self._ancestry_lineage_id,
            certificate_sequence=(max(parent_sequences) + 1 if parent_sequences else 0),
            issuer_boot_identity=self.boot_identity,
            issued_at=_utc_timestamp(self._clock()),
            sign_digest=self._sign_digest,
            boot_attestation_verifier=self._ancestry_boot_attestation_verifier,
            allowed_issuer_roles=self._ancestry_allowed_issuer_roles,
            parent_certificates=parent_certificates,
            parent_full_graphs=parent_full_graphs,
            allowed_failed_receipt_hashes=(
                str(item["receipt_hash"])
                for item in local
                if item.get("status") != "succeeded"
            ),
            required_purposes=(str(manifest["purpose"]),),
        )
        return build_compact_ancestry_proof_from_delta_v2(
            local_delta,
            certificate,
            expected_lineage_id=self._ancestry_lineage_id,
            boot_attestation_verifier=self._ancestry_boot_attestation_verifier,
            allowed_issuer_roles=self._ancestry_allowed_issuer_roles,
        )

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] == "cancelled":
                return
            if job["cancel_requested"]:
                job["state"] = "cancelled"
                return
            job["state"] = "running"
            job["updated_at"] = self._clock()
            self._active.add(job_id)
            manifest = dict(job["manifest"])
            payload_bytes = bytes(job["input"])
        context = ExecutionContextV2(
            job_id=job_id,
            purpose=manifest["purpose"],
            epoch_id=manifest["epoch_id"],
            parent_receipt_hashes=tuple(manifest["parent_receipt_hashes"]),
            provider_credential_profile=str(
                manifest["provider_credential_profile"]
            ),
            provider_credential_ref_hashes=dict(
                manifest["provider_credential_ref_hashes"]
            ),
            artifact_hashes=list(manifest["input_artifact_hashes"]),
            allowed_purposes=frozenset(ROLE_PURPOSES[self.role]),
            max_external_receipt_graph_bytes=_job_input_limit_bytes(
                operation=manifest["operation"],
                purpose=manifest["purpose"],
            ),
            max_external_ancestry_authorities=_job_external_authority_limit(
                operation=manifest["operation"],
                purpose=manifest["purpose"],
            ),
        )
        checkpoint_bootstrap_scope = (
            manifest["operation"] == "ancestry_checkpoint_bootstrap_v2"
            and manifest["purpose"]
            == "research_lab.ancestry_checkpoint_bootstrap.v2"
        )
        if self._host_operation_channel_factory is not None:
            context.host_operation_channel = self._host_operation_channel_factory(
                job_id,
                manifest["purpose"],
            )
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    current["host_operation_channel"] = context.host_operation_channel
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
            parent_graphs = payload.pop(PARENT_RECEIPT_GRAPHS_FIELD, None)
            parent_graph_set = payload.pop(PARENT_RECEIPT_GRAPH_SET_FIELD, None)
            payload_parent_hashes = payload.pop("parent_receipt_hashes", None)
            if payload_parent_hashes is not None and (
                payload_parent_hashes != manifest["parent_receipt_hashes"]
            ):
                raise ExecutionJobV2Error(
                    "job payload parent receipt hashes differ from manifest ancestry"
                )
            if parent_graphs is not None and parent_graph_set is not None:
                raise ExecutionJobV2Error(
                    "job supplies multiple parent receipt graph encodings"
                )
            shared_parent_graph_objects = parent_graph_set is not None
            parent_graph_sizes = None
            if parent_graph_set is not None:
                parent_graphs, parent_graph_sizes = _unpack_parent_receipt_graph_set_v2(
                    parent_graph_set,
                    max_graph_count=min(
                        context.max_external_ancestry_authorities,
                        max(
                            MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
                            MAX_ALLOCATION_FRONTIER_BOOTSTRAP_AUTHORITIES,
                        ),
                    ),
                )
            elif parent_graphs is None:
                parent_graphs = []
            parent_proofs = payload.pop(PARENT_ANCESTRY_PROOFS_FIELD, None)
            if parent_proofs is None:
                parent_proofs = []
            if not isinstance(parent_graphs, list) or any(
                not isinstance(graph, Mapping) for graph in parent_graphs
            ):
                raise ExecutionJobV2Error("job parent receipt graphs are invalid")
            allowed_failed_by_graph = []
            if not isinstance(parent_proofs, list) or any(
                not isinstance(proof, Mapping) for proof in parent_proofs
            ):
                raise ExecutionJobV2Error("job parent ancestry proofs are invalid")
            if checkpoint_bootstrap_scope and (
                len(parent_graphs) > MAX_ALLOCATION_ANCESTRY_AUTHORITIES
                or len(parent_proofs) > MAX_ALLOCATION_ANCESTRY_AUTHORITIES
            ):
                raise ExecutionJobV2Error(
                    "checkpoint bootstrap ancestry input count exceeds limit"
                )
            if parent_proofs and self._ancestry_lineage_id is None:
                raise ExecutionJobV2Error(
                    "job parent ancestry proofs are unsupported"
                )
            parent_roots = []
            parent_receipt_hashes = set()
            for graph in parent_graphs:
                if (
                    graph.get("schema_version")
                    == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
                    and (
                        self._ancestry_lineage_id is None
                        or graph.get("ancestry_lineage_id")
                        != self._ancestry_lineage_id
                    )
                ):
                    raise ExecutionJobV2Error(
                        "job checkpointed parent lineage differs"
                    )
                allowed_failed = ()
                if self._failed_parent_graph_policy is not None:
                    allowed_failed = tuple(
                        self._failed_parent_graph_policy(manifest, payload, graph)
                    )
                allowed_failed_by_graph.append(allowed_failed)
            parent_roots = list(
                context.record_external_receipt_graphs(
                    parent_graphs,
                    allowed_failed_receipt_hashes_by_graph=(
                        allowed_failed_by_graph
                    ),
                    _encoded_sizes=parent_graph_sizes,
                    _share_objects=shared_parent_graph_objects,
                    boot_attestation_verifier=(
                        self._ancestry_boot_attestation_verifier
                    ),
                    require_boot_attestation_verification=(
                        self._ancestry_lineage_id is not None
                    ),
                )
            )
            parent_receipt_hashes = set()
            for graph in parent_graphs:
                parent_receipt_hashes.update(
                    str(receipt.get("receipt_hash") or "")
                    for receipt in graph.get("receipts") or ()
                    if isinstance(receipt, Mapping)
                )
            for proof in parent_proofs:
                certificate = proof.get("certificate")
                claim = (
                    certificate.get("claim")
                    if isinstance(certificate, Mapping)
                    else None
                )
                proof_root = str(
                    claim.get("output_root_receipt_hash")
                    if isinstance(claim, Mapping)
                    else ""
                )
                disclosed_graph = {
                    "root_receipt_hash": proof_root,
                    "receipts": proof.get("disclosed_receipts") or [],
                }
                expected_allowed_failed = ()
                if self._failed_parent_graph_policy is not None:
                    expected_allowed_failed = tuple(
                        self._failed_parent_graph_policy(
                            manifest, payload, disclosed_graph
                        )
                    )
                observed_policy = (
                    claim.get("policy")
                    if isinstance(claim, Mapping)
                    else None
                )
                observed_allowed_failed = (
                    tuple(observed_policy.get("allowed_failed_receipt_hashes") or ())
                    if isinstance(observed_policy, Mapping)
                    else ()
                )
                if tuple(sorted(expected_allowed_failed)) != tuple(
                    sorted(observed_allowed_failed)
                ):
                    raise ExecutionJobV2Error(
                        "job parent ancestry failure policy is unauthorized"
                    )
                parent_root = context.record_external_ancestry_proof(
                    proof,
                    expected_lineage_id=str(self._ancestry_lineage_id),
                    boot_attestation_verifier=(
                        self._ancestry_boot_attestation_verifier
                    ),
                    allowed_issuer_roles=self._ancestry_allowed_issuer_roles,
                    required_receipt_hashes=(proof_root,),
                )
                if parent_root in parent_roots:
                    raise ExecutionJobV2Error(
                        "job parent ancestry authority is duplicated"
                    )
                if not checkpoint_bootstrap_scope:
                    parent_roots.append(parent_root)
                parent_receipt_hashes.add(parent_root)
            declared_parent_hashes = set(manifest["parent_receipt_hashes"])
            exact_checkpoint_roots = bool(parent_proofs) or (
                self._ancestry_lineage_id is not None and bool(parent_graphs)
            )
            parent_authority_matches = (
                set(parent_roots) == declared_parent_hashes
                if exact_checkpoint_roots
                else (
                    set(parent_roots).issubset(declared_parent_hashes)
                    and declared_parent_hashes.issubset(parent_receipt_hashes)
                )
            )
            if not parent_authority_matches:
                raise ExecutionJobV2Error(
                    "job parent receipt graphs differ from manifest ancestry"
                )
            value = self._executor(manifest["operation"], payload, context)
            if inspect.isawaitable(value):
                value = asyncio.run(value)
            result = value if isinstance(value, ExecutionResultV2) else ExecutionResultV2(value)
            if not isinstance(result.output, Mapping):
                raise ExecutionJobV2Error("executor output must be an object")
            if result.ancestry_checkpoint_bootstrap:
                if (
                    self._ancestry_lineage_id is None
                    or self._ancestry_boot_attestation_verifier is None
                ):
                    raise ExecutionJobV2Error(
                        "ancestry checkpoint bootstrap is unavailable"
                    )
                if (
                    set(result.output)
                    != {"schema_version", "selected_root_receipt_hashes"}
                    or result.output.get("schema_version")
                    != ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION
                    or result.receipt_output is not None
                    or result.transport_attempts
                    or result.artifact_hashes
                    or result.transitions
                ):
                    raise ExecutionJobV2Error(
                        "ancestry checkpoint bootstrap executor result is invalid"
                    )
                graph_policies = [
                    context.external_receipt_graph_policies.get(
                        str(graph["root_receipt_hash"]), ()
                    )
                    for graph in context.external_receipt_graphs
                ]
                result = ExecutionResultV2(
                    output=issue_legacy_ancestry_checkpoint_bootstrap_v2(
                        full_graphs=context.external_receipt_graphs,
                        selected_root_receipt_hashes=result.output[
                            "selected_root_receipt_hashes"
                        ],
                        existing_compact_proofs=(
                            context.external_ancestry_proofs
                        ),
                        allowed_failed_receipt_hashes_by_graph=(
                            graph_policies
                        ),
                        lineage_id=self._ancestry_lineage_id,
                        issuer_boot_identity=self.boot_identity,
                        issued_at=_utc_timestamp(self._clock()),
                        sign_digest=self._sign_digest,
                        boot_attestation_verifier=(
                            self._ancestry_boot_attestation_verifier
                        ),
                        allowed_issuer_roles=(
                            self._ancestry_allowed_issuer_roles
                        ),
                    )
                )
            for attempt in result.transport_attempts:
                context.record_transport(attempt)
            for artifact_hash in result.artifact_hashes:
                context.record_artifact(artifact_hash)
            result_bytes = _canonical_bytes(dict(result.output))
            if len(result_bytes) > MAX_OUTPUT_BYTES:
                raise ExecutionJobV2Error("executor output exceeds size limit")
            receipt_output = result.receipt_output or result.output
            if not isinstance(receipt_output, Mapping):
                raise ExecutionJobV2Error("receipt output must be an object")
            receipt_output_bytes = _canonical_bytes(dict(receipt_output))
            if len(receipt_output_bytes) > MAX_OUTPUT_BYTES:
                raise ExecutionJobV2Error("receipt output exceeds size limit")
            stage_receipts = self._stage_receipts(
                manifest=manifest,
                context=context,
            )
            root_manifest = dict(manifest)
            root_parents = list(manifest["parent_receipt_hashes"])
            if stage_receipts:
                root_parents = [stage_receipts[-1]["receipt_hash"]]
            root_parents.extend(context.external_receipt_roots())
            if not checkpoint_bootstrap_scope:
                root_parents.extend(context.external_ancestry_roots())
            root_manifest["parent_receipt_hashes"] = sorted(set(root_parents))
            transport_attempts = context.freeze_transport_attempts()
            artifact_hashes = context.freeze_artifact_hashes()
            host_operation_records = tuple(context.host_operation_records())
            receipt = self._receipt(
                manifest=root_manifest,
                context=context,
                output_root=sha256_bytes(receipt_output_bytes),
                status="succeeded",
                failure_code=None,
                transport_attempts=transport_attempts,
                host_operations=host_operation_records,
                artifact_hashes=artifact_hashes,
            )
            local_receipts = list(stage_receipts) + [receipt]
            ancestry_compact_proof = self._build_ancestry_compact_proof(
                manifest=root_manifest,
                context=context,
                root_receipt=receipt,
                local_receipts=local_receipts,
                transport_attempts=transport_attempts,
                host_operations=host_operation_records,
                include_external_ancestry_proofs=(
                    not checkpoint_bootstrap_scope
                ),
            )
            transitions = self._transitions(receipt, result.transitions)
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job["cancel_requested"]:
                    cancelled_bytes = _canonical_bytes(
                        {"status": "failed", "failure_code": "cancelled"}
                    )
                    cancelled_receipt = self._receipt(
                        manifest=root_manifest,
                        context=context,
                        output_root=sha256_bytes(cancelled_bytes),
                        status="failed",
                        failure_code="cancelled",
                        transport_attempts=transport_attempts,
                        host_operations=host_operation_records,
                        artifact_hashes=artifact_hashes,
                    )
                    cancelled_receipts = list(stage_receipts) + [
                        cancelled_receipt
                    ]
                    cancelled_proof = self._build_ancestry_compact_proof(
                        manifest=root_manifest,
                        context=context,
                        root_receipt=cancelled_receipt,
                        local_receipts=cancelled_receipts,
                        transport_attempts=transport_attempts,
                        host_operations=host_operation_records,
                        include_external_ancestry_proofs=(
                            not checkpoint_bootstrap_scope
                        ),
                    )
                    job["state"] = "cancelled"
                    job["result"] = cancelled_bytes
                    job["result_hash"] = sha256_bytes(cancelled_bytes)
                    job["receipt"] = cancelled_receipt
                    job["receipts"] = cancelled_receipts
                    job["transitions"] = []
                    job["transport_attempts"] = list(transport_attempts)
                    job["artifact_hashes"] = list(artifact_hashes)
                    job["host_operations"] = list(
                        host_operation_records
                    )
                    job["external_receipt_graphs"] = list(
                        context.external_receipt_graphs
                    )
                    job["external_ancestry_proofs"] = list(
                        context.external_ancestry_proofs
                    )
                    job["ancestry_compact_proof"] = cancelled_proof
                else:
                    job["state"] = "succeeded"
                    job["result"] = result_bytes
                    job["result_hash"] = sha256_bytes(result_bytes)
                    job["receipt"] = receipt
                    job["receipts"] = local_receipts
                    job["transitions"] = transitions
                    job["transport_attempts"] = list(transport_attempts)
                    job["artifact_hashes"] = list(artifact_hashes)
                    job["host_operations"] = list(host_operation_records)
                    job["external_receipt_graphs"] = list(
                        context.external_receipt_graphs
                    )
                    job["external_ancestry_proofs"] = list(
                        context.external_ancestry_proofs
                    )
                    job["ancestry_compact_proof"] = ancestry_compact_proof
                job["input"] = bytearray()
                job["updated_at"] = self._clock()
        except Exception as exc:
            failure_code = _execution_failure_code(exc)
            failure_bytes = _canonical_bytes(
                {"status": "failed", "failure_code": failure_code}
            )
            failure_transport_attempts = context.freeze_transport_attempts()
            failure_artifact_hashes = context.freeze_artifact_hashes()
            try:
                failure_manifest = dict(manifest)
                failure_parent_roots = (
                    set(manifest["parent_receipt_hashes"])
                    | set(context.external_receipt_roots())
                )
                if not checkpoint_bootstrap_scope:
                    failure_parent_roots.update(
                        context.external_ancestry_roots()
                    )
                failure_manifest["parent_receipt_hashes"] = sorted(
                    failure_parent_roots
                )
                try:
                    failure_host_operations = tuple(
                        context.host_operation_records()
                    )
                except Exception:
                    failure_host_operations = ()
                receipt = self._receipt(
                    manifest=failure_manifest,
                    context=context,
                    output_root=sha256_bytes(failure_bytes),
                    status="failed",
                    failure_code=failure_code,
                    transport_attempts=failure_transport_attempts,
                    host_operations=failure_host_operations,
                    artifact_hashes=failure_artifact_hashes,
                )
                failure_proof = self._build_ancestry_compact_proof(
                    manifest=failure_manifest,
                    context=context,
                    root_receipt=receipt,
                    local_receipts=(receipt,),
                    transport_attempts=failure_transport_attempts,
                    host_operations=failure_host_operations,
                    include_external_ancestry_proofs=(
                        not checkpoint_bootstrap_scope
                    ),
                )
            except Exception:
                receipt = None
                failure_proof = None
                failure_host_operations = ()
                failure_code = "receipt_unavailable"
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["state"] = "failed"
                    job["error_code"] = failure_code
                    job["result"] = failure_bytes
                    job["result_hash"] = sha256_bytes(failure_bytes)
                    job["receipt"] = receipt
                    job["receipts"] = [receipt] if receipt is not None else []
                    job["transport_attempts"] = list(
                        failure_transport_attempts
                    )
                    job["artifact_hashes"] = list(failure_artifact_hashes)
                    job["host_operations"] = list(failure_host_operations)
                    job["external_receipt_graphs"] = list(
                        context.external_receipt_graphs
                    )
                    job["external_ancestry_proofs"] = list(
                        context.external_ancestry_proofs
                    )
                    job["ancestry_compact_proof"] = failure_proof
                    job["input"] = bytearray()
                    job["updated_at"] = self._clock()
        finally:
            with self._lock:
                self._active.discard(job_id)

    def _receipt(
        self,
        *,
        manifest: Mapping[str, Any],
        context: ExecutionContextV2,
        output_root: str,
        status: str,
        failure_code: Optional[str],
        transport_attempts: Sequence[Mapping[str, Any]],
        host_operations: Sequence[Mapping[str, Any]],
        artifact_hashes: Sequence[str],
    ) -> Dict[str, Any]:
        current_boot = dict(self._boot_identity_supplier())
        if current_boot != self.boot_identity:
            raise ExecutionJobV2Error("enclave boot identity changed during execution")
        body = build_execution_receipt_body(
            role=self.role,
            purpose=manifest["purpose"],
            job_id=manifest["job_id"],
            epoch_id=manifest["epoch_id"],
            sequence=manifest["sequence"],
            commit_sha=self.boot_identity["commit_sha"],
            pcr0=self.boot_identity["pcr0"],
            build_manifest_hash=self.boot_identity["build_manifest_hash"],
            dependency_lock_hash=self.boot_identity["dependency_lock_hash"],
            config_hash=self.boot_identity["config_hash"],
            boot_identity_hash=self.boot_identity["boot_identity_hash"],
            input_root=manifest["payload_sha256"],
            output_root=output_root,
            transport_root_hash=(
                transport_root(transport_attempts)
                if transport_attempts
                else EMPTY_TRANSPORT_ROOT
            ),
            host_operation_root_hash=(
                host_operation_root(host_operations)
                if host_operations
                else EMPTY_HOST_OPERATION_ROOT
            ),
            artifact_root=(
                merkle_root(artifact_hashes, domain="leadpoet-artifact-v2")
                if artifact_hashes
                else EMPTY_ARTIFACT_ROOT
            ),
            parent_receipt_hashes=manifest["parent_receipt_hashes"],
            status=status,
            failure_code=failure_code,
            issued_at=_utc_timestamp(self._clock()),
        )
        return create_signed_execution_receipt(
            body=body,
            enclave_pubkey=self.boot_identity["signing_pubkey"],
            sign_digest=self._sign_digest,
        )

    def _stage_receipts(
        self,
        *,
        manifest: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Sequence[Dict[str, Any]]:
        if not context.stage_receipts:
            return ()
        issued_at = _utc_timestamp(self._clock())
        parent_hashes = list(manifest["parent_receipt_hashes"])
        output = []
        root_fragment = manifest["payload_sha256"].split(":", 1)[1][:24]
        for index, spec in enumerate(context.stage_receipts):
            body = build_execution_receipt_body(
                role=self.role,
                purpose=spec.purpose,
                job_id="stage:%s:%s" % (root_fragment, index),
                epoch_id=manifest["epoch_id"],
                sequence=index,
                commit_sha=self.boot_identity["commit_sha"],
                pcr0=self.boot_identity["pcr0"],
                build_manifest_hash=self.boot_identity["build_manifest_hash"],
                dependency_lock_hash=self.boot_identity["dependency_lock_hash"],
                config_hash=self.boot_identity["config_hash"],
                boot_identity_hash=self.boot_identity["boot_identity_hash"],
                input_root=spec.input_root,
                output_root=spec.output_root,
                transport_root_hash=EMPTY_TRANSPORT_ROOT,
                host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
                artifact_root=(
                    merkle_root(
                        spec.artifact_hashes,
                        domain="leadpoet-artifact-v2",
                    )
                    if spec.artifact_hashes
                    else EMPTY_ARTIFACT_ROOT
                ),
                parent_receipt_hashes=parent_hashes,
                status="succeeded",
                failure_code=None,
                issued_at=issued_at,
            )
            receipt = create_signed_execution_receipt(
                body=body,
                enclave_pubkey=self.boot_identity["signing_pubkey"],
                sign_digest=self._sign_digest,
            )
            output.append(receipt)
            parent_hashes = [receipt["receipt_hash"]]
        return tuple(output)

    def _transitions(
        self,
        receipt: Mapping[str, Any],
        specs: Sequence[TransitionSpecV2],
    ) -> Sequence[Dict[str, Any]]:
        issued = datetime.fromtimestamp(self._clock(), tz=timezone.utc)
        output = []
        for spec in specs:
            expires = issued + timedelta(seconds=max(1, int(spec.ttl_seconds)))
            body = build_transition_command_body(
                operation=spec.operation,
                target=spec.target,
                idempotency_key=spec.idempotency_key,
                expected_state_hash=spec.expected_state_hash,
                payload_hash=spec.payload_hash,
                receipt_hash=receipt["receipt_hash"],
                issued_at=issued.strftime("%Y-%m-%dT%H:%M:%SZ"),
                expires_at=expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            output.append(
                create_signed_transition_command(
                    body=body,
                    enclave_pubkey=self.boot_identity["signing_pubkey"],
                    sign_digest=self._sign_digest,
                )
            )
        return tuple(output)

    def _summary(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "job_id": job["manifest"]["job_id"],
            "operation": job["manifest"]["operation"],
            "purpose": job["manifest"]["purpose"],
            "state": job["state"],
            "manifest_hash": job["manifest_hash"],
            "uploaded_bytes": len(job["input"]),
            "expected_bytes": job["manifest"]["payload_size_bytes"],
            "result_sha256": job["result_hash"],
            "result_size_bytes": len(job["result"]),
            "receipt_hash": (job["receipt"] or {}).get("receipt_hash"),
            "receipt_count": len(job["receipts"]),
            "transition_count": len(job["transitions"]),
            "transport_attempt_count": len(job["transport_attempts"]),
            "artifact_hash_count": len(job["artifact_hashes"]),
            "host_operation_count": len(job["host_operations"]),
            "external_receipt_graph_count": len(job["external_receipt_graphs"]),
            "external_ancestry_proof_count": len(
                job["external_ancestry_proofs"]
            ),
            "ancestry_compact_proof_hash": (
                job["ancestry_compact_proof"].get("proof_hash")
                if isinstance(job.get("ancestry_compact_proof"), Mapping)
                else None
            ),
            "error_code": job["error_code"],
            "cancel_requested": bool(job["cancel_requested"]),
        }

    def _job(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise ExecutionJobV2Error("V2 job was not found")
        return job

    def _purge_locked(self) -> None:
        cutoff = self._clock() - self._retention_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job["state"] in TERMINAL_STATES and job["updated_at"] < cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _evict_oldest_terminal_locked(self) -> Optional[str]:
        cutoff = self._clock() - MIN_TERMINAL_EVICTION_AGE_SECONDS
        candidates = [
            (
                float(job["updated_at"]),
                float(job["created_at"]),
                job_id,
            )
            for job_id, job in self._jobs.items()
            if (
                job["state"] in TERMINAL_STATES
                and job_id not in self._active
                and float(job["updated_at"]) <= cutoff
            )
        ]
        if not candidates:
            return None
        _, _, job_id = min(candidates)
        del self._jobs[job_id]
        self._terminal_eviction_count += 1
        return job_id
